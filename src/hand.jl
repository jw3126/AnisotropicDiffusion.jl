function hand_tuned_step!(out::AbstractArray,
                          bc::BorderArray,
                          bu::BorderArray,
                          alg)
    c = bc.inner
    u = bu.inner
    i1 = first(eachindex(c))
    iN = last( eachindex(c))
    pm = alg.variant
    innerinds = (i1+1):(iN-1)

    @inbounds @simd for i in innerinds
        ∇u2 = (u[i+1] - u[i])^2
        K = get_K(pm, i)
        c[i] = c_from_grad2(pm, K, ∇u2)
    end
    cstencil = CStencil(pm)
    mapstencil_border!(cstencil, c, bu)

    @inbounds @simd for i in innerinds
        u₀ = u[i]
        u₊ = u[i+1]
        u₋ = u[i-1]
        c₀ = c[i]
        c₊ = c[i+1]

        ∇u = u₊ - u₀
        ∇c = c₊ - c₀
        Δu = u₊ - 2u₀ + u₋
        dudt = c₀ * Δu + ∇c * ∇u
        out[i] = u[i] + alg.lambda * dudt
    end
    pmstencil = PMStencil(alg.lambda)
    mapstencil_border!(pmstencil, out, bu, bc)
    out
end
