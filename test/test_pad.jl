module TestPad
using Test
using AnisotropicDiffusion: LazyPadArray
using ImageFiltering: Pad

@testset "getindex inner $dims" for dims in [
    (1,), (2,3), (10,2,3),]
    arr = randn(dims)
    padsize = tuple(rand(0:4, length(dims))...)
    pad = Pad(:replicate, padsize)
    larr = LazyPadArray(arr, pad)
    for i in CartesianIndices(axes(arr))
        @test larr[i] === arr[i]
    end
end

end#module
