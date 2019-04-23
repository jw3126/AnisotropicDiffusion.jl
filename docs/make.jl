using Documenter, AnisotropicDiffusion

makedocs(;
    modules=[AnisotropicDiffusion],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/jw3126/AnisotropicDiffusion.jl/blob/{commit}{path}#L{line}",
    sitename="AnisotropicDiffusion.jl",
    authors="Jan Weidner",
    assets=[],
)

deploydocs(;
    repo="github.com/jw3126/AnisotropicDiffusion.jl",
)
