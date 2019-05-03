function hand_tuned_step!(out::AbstractVector,
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

function hand_tuned_step!(out::AbstractMatrix,
                          bc::BorderArray,
                          bu::BorderArray,
                          alg)
    c = bc.inner
    u = bu.inner
    iax, jax = axes(c)
    i1 = first(iax)
    iN = last(iax)
    j1 = first(jax)
    jN = last(jax)
    pm = alg.variant
    iinnerinds = (i1+1):(iN-1)
    jinnerinds = (j1+1):(jN-1)

    @inbounds @simd for j in jinnerinds
        for i in iinnerinds
            ∇u2 = (u[i+1,j] - u[i,j])^2 + (u[i,j+1] - u[i,j])^2
            K = get_K(pm, i)
            c[i,j] = c_from_grad2(pm, K, ∇u2)
        end
    end
    cstencil = CStencil(pm)
    mapstencil_border!(cstencil, c, bu)

    @inbounds @simd for j in jinnerinds
        for i in iinnerinds
            u00 = u[i,  j  ]
            up0 = u[i+1,j  ]
            um0 = u[i-1,j  ]
            u0p = u[i  ,j+1]
            u0m = u[i  ,j-1]
            c00 = c[i  ,j  ]
            cp0 = c[i+1,j  ]
            c0p = c[i  ,j+1]

            ∇c∇u = (cp0 - c00)*(up0 - u00) + (c0p - c00)*(u0p - u00)
            Δu = u0p + up0 -4u00 + u0m + um0
            dudt = c00 * Δu + ∇c∇u
            out[i,j] = u00 + alg.lambda * dudt
        end
    end
    pmstencil = PMStencil(alg.lambda)
    mapstencil_border!(pmstencil, out, bu, bc)
    out
end
