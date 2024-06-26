module AutomaticMALA

using Random: Random
using AbstractMCMC: AbstractMCMC
using LogDensityProblems: LogDensityProblems

using DocStringExtensions: TYPEDFIELDS

using LinearAlgebra
using Distributions

export AutoMALA

"""
    AutoMALA

An automatic version of the Metropolis-adjusted Langevin algorithm (MALA).

See [Biron-Lattes, Surjanovic & Syed et al. (2023)](https://arxiv.org/abs/2310.16782) for details.

# Fields
$(TYPEDFIELDS)
"""
struct AutoMALA{T} <: AbstractMCMC.AbstractSampler
    "initial step size to use"
    ϵ_init::T
    "number of unadjusted steps to take"
    num_unadjusted::Int
end

AutoMALA(; ϵ_init=1.0, num_unadjusted = 1) = AutoMALA(float(ϵ_init), num_unadjusted)

struct AutoMALAState{T1,T2,T3}
    "current position"
    x::T1
    "current momentum"
    p::T1
    "current log probability (of joint)"
    lp::T2
    "current `a`, i.e. lower-bound of step size"
    a::T3
    "current `b`, i.e. upper-bound of step size"
    b::T3
    "current (iteration-average) step size"
    ϵ::T3
    "number of halving/doubling steps taken"
    j::Int
    "whether the proposal was accepted"
    isaccept::Bool
    "iteration number"
    iteration::Int
end

isunadjusted(sampler::AutoMALA, state::AutoMALAState) = state.iteration < sampler.num_unadjusted

function compute_logprob(model, x, p)
    ℓ_x = LogDensityProblems.logdensity(model, x)
    return ℓ_x - norm(p)^2 / 2
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::AutoMALA;
    initial_params=nothing,
    kwargs...
)
    model = model_wrapper.logdensity
    a, b = sample_a_and_b(rng)
    p = sample_momentum(rng, model)
    x = if initial_params === nothing
        randn(rng, LogDensityProblems.dimension(model))
    else
        initial_params
    end
    lp = compute_logprob(model, x, p)
    state = AutoMALAState(x, p, lp, a, b, sampler.ϵ_init, 0, true, 1)
    return state, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::AutoMALA,
    state::AutoMALAState;
    kwargs...
)
    model = model_wrapper.logdensity
    ϵ_init = sampler.ϵ_init

    # Proposal.
    x_prev = state.x

    # Sample new momentum.
    p_prev = sample_momentum(rng, model)
    lp_prev = compute_logprob(model, x_prev, p_prev)

    # Sample new `a` and `b`.
    a, b = sample_a_and_b(rng)
    # Select the step size.
    ϵ, j = step_size_selector(model, x_prev, p_prev, lp_prev, a, b, ϵ_init)
    # Propose state.
    x, p, lp = leapfrog_proposal(model, x_prev, p_prev, ϵ)
    # Select the step size for proposal.
    ϵ_prop, j_prop = step_size_selector(model, x, p, lp, a, b, ϵ_init)
    # Compute the next step size.
    ϵₜ = (ϵ + ϵ_prop) / 2
    # Compute acceptance probability.
    logα = lp - lp_prev
    # Accept or reject.
    isaccept = isunadjusted(sampler, state) || (j == j_prop && log(rand(rng)) < logα)

    state_new = if isaccept
        AutoMALAState(x, p, lp, a, b, ϵₜ, j, isaccept, state.iteration + 1)
    else
        AutoMALAState(x_prev, p_prev, lp_prev, a, b, ϵₜ, j, isaccept, state.iteration + 1)
    end

    return state_new, state_new
end

# TODO: Implement for general mass matrix `M⁻¹`.
function leapfrog_proposal(model, x, p, ϵ)
    ℓ_x, ∇ℓ_x = LogDensityProblems.logdensity_and_gradient(model, x)
    p_half = p + (ϵ / 2) .* ∇ℓ_x
    x_new = x + ϵ .* p_half
    ℓ_x_new, ∇ℓ_x_new = LogDensityProblems.logdensity_and_gradient(model, x_new)
    p_new = p_half + (ϵ / 2) .* ∇ℓ_x_new
    return x_new, -p_new, ℓ_x_new - norm(p_new)^2 / 2
end

function step_size_selector(model, x, p, lp, a, b, ϵ_init)
    loga = log(a)
    logb = log(b)

    ϵ = ϵ_init
    x_new, p_new, lp_new = leapfrog_proposal(model, x, p, ϵ)
    lp_ratio = lp_new - lp
    δ = (lp_ratio ≥ logb) - (lp_ratio < loga)
    j = 0  # number of doublings/halvings

    δ == 0 && return ϵ, j
    # TODO: Make this terminate in finite time in case something is wrong.
    while true
        ϵ = ϵ * 2.0^δ
        j = j + δ
        x_new, p_new, lp_new = leapfrog_proposal(model, x, p, ϵ)
        lp_ratio = lp_new - lp
        if δ == 1 && lp_ratio < logb
            return ϵ / 2, j - 1
        elseif δ == -1 && lp_ratio ≥ loga
            return ϵ / 2, j - 1
        end
        # Otherwise, we continue.
    end
end

function sample_momentum(rng::Random.AbstractRNG, model)
    d = LogDensityProblems.dimension(model)
    return randn(rng, d)
end

function sample_a_and_b(rng::Random.AbstractRNG)
    u1, u2 = rand(rng), rand(rng)
    return min(u1, u2), max(u1, u2)
end

"""
    round_based_adaptation([rng, ]model, sample::AutoMAL; num_rounds=10, kwargs...)

Run adaptation for a number of rounds, each time doubling the number of iterations.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model`: The model.
- `sample::AutoMALA`: The sampler.

# Keyword Arguments
- `num_rounds::Int=10`: The number of rounds.
- `kwargs...`: Additional keyword arguments passed to `sample`.

# Returns
- `sampler::AutoMALA`: Sampler containing the adapted parameters.
- `initial_params`: Resulting parameters after adaptation.

"""
function round_based_adaptation(
    model,
    sampler::AutoMALA;
    kwargs...
)
    return round_based_adaptation(Random.default_rng(), model, sampler; kwargs...)
end
function round_based_adaptation(
    rng::Random.AbstractRNG,
    model,
    sampler::AutoMALA;
    num_rounds=10,
    initial_params=nothing,
    kwargs...
)
    # Here we will run adaptation for increasingly long periods of time.
    ϵ_init = sampler.ϵ_init
    for i in 1:num_rounds
        num_iters = 2^i
        sampler_round = AutoMALA(ϵ_init, num_iters)
        states = sample(rng, model, sampler_round, num_iters; initial_params, kwargs...)
        ϵ_init = mean([state.ϵ for state in states])
        initial_params = states[end].x
    end

    return AutoMALA(ϵ_init, 0), initial_params
end

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include(
            "../ext/AutomaticMALATuringExt.jl"
        )
    end
end

end
