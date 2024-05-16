using AutomaticMALA
using Documenter

DocMeta.setdocmeta!(AutomaticMALA, :DocTestSetup, :(using AutomaticMALA); recursive=true)

makedocs(;
    modules=[AutomaticMALA],
    authors="Tor Erlend Fjelde <tor.erlend95@gmail.com> and contributors",
    sitename="AutomaticMALA.jl",
    format=Documenter.HTML(;
        canonical="https://torfjelde.github.io/AutomaticMALA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/torfjelde/AutomaticMALA.jl",
    devbranch="main",
)
