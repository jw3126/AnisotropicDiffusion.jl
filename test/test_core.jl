module TestCore

using AnisotropicDiffusion
using ImageFiltering
using LinearAlgebra
using Test

@testset "1d edge" begin
    img = zeros(20)
    img[1:10] .= 100
    img_noisy = img + randn(size(img)...)
    img_clean = denoise(img_noisy, PeronaMalik(niter=3, lambda=0.25));
    ker = centered(normalize(ones(3),1))
    img_baseline = imfilter(img_noisy, ker)
    @test sum(abs2, img - img_clean) < sum(abs2, img - img_baseline)
end

end#module
