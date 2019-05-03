module AnisotropicDiffusion

using QuickTypes: @qstruct
using ArgCheck: @argcheck
using TiledIteration: EdgeIterator
using ImageFiltering: padindices, Pad
using Base.Cartesian
using LinearAlgebra

include("core.jl")
include("hand.jl")

end # module
