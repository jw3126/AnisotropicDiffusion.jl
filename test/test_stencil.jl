module TestStencil
using AnisotropicDiffusion
using AnisotropicDiffusion: laplace, dot_grad
using Test

@testset "laplace" begin
    x = randn(3,3,3)
    index = CartesianIndex((2,2,2))
    @test laplace(x, index) ≈ x[1,2,2] + x[2,1,2] + x[2,2,1] + x[3,2,2] + x[2,3,2] + x[2,2,3] - 6x[2,2,2]
    
end

@testset "dot_grad" begin
    x = randn(3,3,3)
    y = randn(3,3,3)
    index = CartesianIndex((2,2,2))
    @test dot_grad(x,y,index) ≈ 
        (x[3,2,2] - x[2,2,2])*(y[3,2,2] - y[2,2,2]) +
        (x[2,3,2] - x[2,2,2])*(y[2,3,2] - y[2,2,2]) +
        (x[2,2,3] - x[2,2,2])*(y[2,2,3] - y[2,2,2])
end

end#module
