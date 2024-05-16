module AutomaticMALATuringExt

if isdefined(Base, :get_extension)
    using AutomaticMALA: AutomaticMALA
    using Turing: Turing
else
    using ..AutomaticMALA: AutomaticMALA
    using ..Turing: Turing
end

Turing.Inference.getparams(::Turing.DynamicPPL.Model, state::AutomaticMALA.AutoMALAState) = state.x

end
