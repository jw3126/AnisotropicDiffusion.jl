using Base: @propagate_inbounds

struct LazyPadArray{T,N,ARR,P,I} <: AbstractArray{T, N}
    parent::ARR
    pad::P
    padindices::I
    function LazyPadArray(arr::AbstractArray{T,N}, pad::P) where {T,N,P}
        inds = padindices(arr, pad)
        I = typeof(inds)
        ARR = typeof(arr)
        new{T,N,ARR,P,I}(arr, pad, inds)
    end
end

function Base.axes(arr::LazyPadArray)
    map(eachindex,arr.padindices)
end

function Base.size(arr::LazyPadArray)
    map(length, arr.padindices)
end

@inline @propagate_inbounds function Base.getindex(A::LazyPadArray{T,N}, I::CartesianIndex{N}) where {T,N}
    II = map(getindex, A.padindices, I.I)
    A.parent[CartesianIndex(II)]
end

@inline @propagate_inbounds function Base.getindex(A::LazyPadArray{T,N}, I::Vararg{Int,N}) where {T,N}
    A[CartesianIndex(I)]
end
