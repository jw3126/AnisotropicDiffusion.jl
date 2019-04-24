module AnisotropicDiffusion

using QuickTypes
using ArgCheck
using ImageFiltering
using TiledIteration
using Base.Cartesian
using LinearAlgebra

export PeronaMalik, PM1, PM2, denoise

@qstruct PeronaMalik{PM}(;
    lambda::Float64=0.1,
    variant::PM = PM2(10),
    border::Symbol = :replicate,
    niter::Int=100,
    )

function mapstencil!(stencil::F, out, arrs...; pad) where {F}
    outeraxes = axes(out)
    inneraxes = map(outeraxes) do r
        r[2:end-1]
    end
    
    @simd for i in CartesianIndices(inneraxes)
        @inbounds out[i] = stencil(arrs..., i)
    end
    
    padded_arrs = map(arrs) do arr
        padarray(arr, pad)
    end
    for i in EdgeIterator(outeraxes, inneraxes)
        out[i] = stencil(padded_arrs..., i)
    end
    out
end

function denoise(img, alg::PeronaMalik=PeronaMalik())
    x = first(img)
    T = typeof(exp((x - x)^2 / 2))
    out_buf = similar(img, T)
    img_buf = similar(img, T)
    c_buf = similar(img, T)
    copy!(img_buf, img)
    for i in 1:alg.niter
        pm_step!(out_buf, c_buf, img_buf, alg)
        @show c_buf
        out_buf, img_buf = img_buf, out_buf
    end
    img_buf
end

function get_pad(img::AbstractArray{T,N}, alg::PeronaMalik) where {T, N}
    dims = ntuple(_ -> 1, Val(N))
    Pad{N}(alg.border, dims, dims)
end

function pm_step!(out, c, img, opt::PeronaMalik)
    pad = get_pad(img, opt)
    cstencil = CStencil(opt.variant)
    pmstencil = PMStencil(opt.lambda)
    mapstencil!(cstencil , c  , img,    pad=pad)
    mapstencil!(pmstencil, out, img, c, pad=pad)
    out
end

struct PM1 
    K::Float64
end

struct PM2
    K::Float64
end
    
struct CStencil{PM}
    pm::PM
end

@inline function (f::CStencil{PM1})(img, index)
    ∇img2 = dot_grad(img, img, index)
    K = f.pm.K
    exp(-∇img2/(K^2))
end

@inline function (f::CStencil{PM2})(img, index)
    ∇img2 = dot_grad(img, img, index)
    K = f.pm.K
    1 / (1 + ∇img2/K^2)
end

function inlineit(ex) 
    ret = gensym("ret")
    quote
        $(Expr(:meta, :inline)) 
        $(Expr(:inbounds, true))
        local $ret = $ex
        $(Expr(:inbounds, :pop))
        $ret
    end
end

"""
    dot_grad(x,y,index)

Compute (∇x⋅∇y)[index].
"""
@generated function dot_grad(x::AbstractArray{T, ndims}, y, index::CartesianIndex) where {T, ndims}
    expr_dot_grad(ndims=ndims) |> inlineit
end

function expr_dot_grad(;ndims)
    # args: x,y,index
    ret = quote
        ret = zero(Float64)
        x0 = x[index]
        y0 = y[index]
    end
    for i in 1:ndims
        index₊ = expr_shift_index(:index, ndims=ndims, i=i, shift=:(1))
        ex = quote
            index₊ = $(index₊)
            ret += (x[index₊] - x0)*(y[index₊] - y0)
        end
        append!(ret.args, ex.args)
    end
    push!(ret.args, :ret)
    ret
end 

@generated function laplace(img::AbstractArray{T,ndims}, index) where {T,ndims}
    expr_laplace(ndims=ndims) |> inlineit
end

function expr_laplace(;ndims)
    
    ret = quote
        ret = -2*$ndims * img[index]
    end
    for i in 1:ndims
        index₊ = expr_shift_index(:index, ndims=ndims, i=i, shift=:(1))
        index₋ = expr_shift_index(:index, ndims=ndims, i=i, shift=:(-1))
        ex = quote
            index₊ = $(index₊)
            index₋ = $(index₋)
            ret += img[index₊]
            ret += img[index₋]
        end
        append!(ret.args, ex.args)
    end
    push!(ret.args, :ret)
    ret
end

function expr_shift_index(index; ndims, i, shift)
    @argcheck 1 <= i <= ndims
    args = map(1:ndims) do j
        if i == j
            :($index[$i] + $shift)
        else
            :($index[$j])
        end
    end
    :(@inbounds CartesianIndex{$ndims}($(args...)))
end

struct PMStencil
    lambda::Float64
end

@inline function (f::PMStencil)(img, c, index)
    dt = f.lambda * (dot_grad(c, img, index) + laplace(img, index))
    @inbounds img[index] + dt
end

end # module
