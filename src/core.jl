export PeronaMalik, PM1, PM2, denoise

using ImageFiltering

@qstruct PeronaMalik{PM,S}(;
    lambda::Float64=0.1,
    variant::PM = PM2(10),
    border::Symbol = :replicate,
    niter::Int=100,
    step::S=generic_step!,
    )

function generic_step!(out::AbstractArray,
                       bc::BorderArray,
                       bimg::BorderArray,
                       alg::PeronaMalik)
    cstencil = CStencil(alg.variant)
    pmstencil = PMStencil(alg.lambda)
    mapstencil!(cstencil ,  bc.inner  , bimg,   )
    mapstencil!(pmstencil, out, bimg, bc)
    out
end

function mapstencil_inner!(stencil, out, barrs...; axes)
    arrs = map(barr -> barr.inner, barrs)
    @simd for i in CartesianIndices(axes)
        @inbounds out[i] = stencil(arrs..., i)
    end
    out
end

function mapstencil_border!(stencil, out, barrs...)
    outeraxes = axes(out)
    inneraxes = map(outeraxes) do r
        r[2:end-1]
    end
    for i in EdgeIterator(outeraxes, inneraxes)
        out[i] = stencil(barrs..., i)
    end
    out
end

function mapstencil!(stencil::F, out, barrs...) where {F}
    inneraxes = map(axes(out)) do r
        r[2:end-1]
    end
    mapstencil_inner!(stencil, out, barrs..., axes=inneraxes)
    mapstencil_border!(stencil, out, barrs...)
end

function denoise(img, alg::PeronaMalik=PeronaMalik())
    if alg.variant.k isa AbstractArray
        @argcheck axes(img) == axes(alg.variant.k)
    end
    x = first(img)
    T = typeof(exp((x - x)^2 / 2))
    border = get_border(img, alg)
    out_buf = BorderArray(similar(img, T), border)
    img_buf = BorderArray(similar(img, T), border)
    c_buf   = BorderArray(similar(img, T), border)
    copy!(img_buf.inner, img)
    for i in 1:alg.niter
        alg.step(out_buf.inner, c_buf, img_buf, alg)
        out_buf, img_buf = img_buf, out_buf
    end
    img_buf.inner
end

function get_border(img::AbstractArray{T,N}, alg::PeronaMalik) where {T, N}
    dims = ntuple(_ -> 1, Val(N))
    Pad{N}(alg.border, dims, dims)
end

struct PM1{K}
    k::K
end

struct PM2{K}
    k::K
end
    
struct CStencil{PM}
    pm::PM
end

const PM{K} = Union{PM1{K}, PM2{K}}

@inline function get_K(pm::PM{<:Number}, index)
    pm.k
end

@inline function get_K(pm::PM{<:AbstractArray}, index)
    @inbounds pm.k[index]
end

@inline function c_from_grad2(pm::PM1, K, ∇img2)
    exp(-∇img2/(K^2))
end

@inline function c_from_grad2(pm::PM2, K, ∇img2)
    1 / (1 + ∇img2/K^2)
end

@inline function (f::CStencil)(img, index)
    pm = f.pm
    K = get_K(pm, index)
    ∇img2 = dot_grad(img, img, index)
    c_from_grad2(pm, K, ∇img2)
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
    @inbounds dudt = (dot_grad(c, img, index) + c[index] * laplace(img, index))
    @inbounds img[index] + f.lambda * dudt
end
