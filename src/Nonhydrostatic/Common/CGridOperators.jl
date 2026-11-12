# Operators and diagnostics shared by the C-grid dycores (ICON, MPAS): state, divergence, vorticity,
# vector Laplacian, mass and ρθ fluxes and their divergence, the metric terms over orography,
# lowest-layer pressure, total mass.

"State `(vn, w, ρ, ρθ)` of the model, zero."
function zero_state(m::CGridNonhydrostatic)
    nc, ne, nlev = num_cells(m.grid), num_edges(m.grid), num_layers(m.vgrid)
    z(n, k) = KernelAbstractions.zeros(m.backend, Float64, n, k)
    return (vn=z(ne, nlev), w=z(nc, nlev + 1), ρ=z(nc, nlev), ρθ=z(nc, nlev))
end

"Horizontal divergence `δ` at the cells and relative vorticity `ζ` at the vertices of every layer."
@kernel function divergence_only_kernel!(δ, @Const(vn), @Const(cell_edges), @Const(divergence))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += divergence[j, i] * vn[cell_edges[j, i], k]
    end
    @inbounds δ[i, k] = s
end

@kernel function vorticity_kernel!(ζ, @Const(vn), @Const(vertex_edges), @Const(curl))
    v, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += curl[j, v] * vn[vertex_edges[j, v], k]
    end
    @inbounds ζ[v, k] = s
end

"Vector Laplacian `(δ_r - δ_l)/d_e - (ζ_{v2} - ζ_{v1})/l_e`, times `scale`."
@kernel function vector_laplacian_kernel!(L, @Const(δ), @Const(ζ), @Const(edge_cells), @Const(edge_vertices),
                                          @Const(inverse_dual), @Const(inverse_primal), scale)
    e, k = @index(Global, NTuple)
    @inbounds L[e, k] = scale * ((δ[edge_cells[2, e], k] - δ[edge_cells[1, e], k]) * inverse_dual[e] -
                                 (ζ[edge_vertices[2, e], k] - ζ[edge_vertices[1, e], k]) * inverse_primal[e])
end

"Horizontal mass and `ρθ` fluxes with the new `vn`, times the layer thickness `dze` at the edges over orography (or 1)."
@kernel function flux_kernel!(F, Fθ, @Const(vn), @Const(ρ), @Const(θ), @Const(edge_cells), @Const(edge_weight), @Const(dze))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        f = (edge_weight[1, e] * ρ[l, k] + edge_weight[2, e] * ρ[r, k]) * vn[e, k] * at_ik(dze, e, k)
        F[e, k] = f
        Fθ[e, k] = f * (edge_weight[1, e] * θ[l, k] + edge_weight[2, e] * θ[r, k])
    end
end

"`x[i, k]` of an array, or the number `x`."
@inline at_ik(x::Number, i, k) = x
@inline at_ik(x::AbstractArray, i, k) = @inbounds x[i, k]

"""
Explicit parts of `ρ` and `ρθ`: base state minus `τ` times the horizontal flux divergence (plus the
heating), the divergence divided by the layer thickness `rdz = 1/Δz_i` over orography (or 1).
"""
@kernel function divergence_kernel!(ρexpl, ρθexpl, @Const(ρ0), @Const(ρθ0), @Const(F), @Const(Fθ), @Const(hρθ),
                                    @Const(cell_edges), @Const(divergence), τ, rdz)
    i, k = @index(Global, NTuple)
    s = sθ = 0.0
    @inbounds for j in axes(cell_edges, 1)
        e = cell_edges[j, i]
        s += divergence[j, i] * F[e, k]
        sθ += divergence[j, i] * Fθ[e, k]
    end
    r = at_ik(rdz, i, k)
    @inbounds ρexpl[i, k] = ρ0[i, k] - τ * r * s
    @inbounds ρθexpl[i, k] = ρθ0[i, k] - τ * (r * sθ - hρθ[i, k])
end

"""
The deviation `π' = π - π_ref` of the extrapolated Exner pressure from the reference state at the
heights of the layers and its vertical derivative at the layer centres (centred, one-sided at
the ground and the lid; ICON `z_exner_ex_pr`, `z_dexner_dz_c`), for the pressure gradient over orography.
"""
@kernel function exner_deviation_kernel!(πp, dπdz, @Const(π), @Const(πref), @Const(zf), nlev)
    i, k = @index(Global, NTuple)
    @inbounds begin
        kl, ku = max(k - 1, 1), min(k + 1, nlev)
        πp[i, k] = π[i, k] - πref[i, k]
        dπdz[i, k] = ((π[i, ku] - πref[i, ku]) - (π[i, kl] - πref[i, kl])) / (zf[i, ku] - zf[i, kl])
    end
end

"""
    height_interpolation(edge_cells, zc, ze) -> (lev, wt)

For the height `ze[e, k]` of edge `e` at level `k` and either cell `c = edge_cells[s, e]`: the
layer `lev[s, e, k]` and the weight `wt[s, e, k]` of `z = zc[c, j] + wt (zc[c, j+1] - zc[c, j])`
between the layer centres `zc` of the column, linearly extrapolated below the lowest and above
the highest. For the horizontal pressure gradient at constant height (`height_difference`).
"""
function height_interpolation(edge_cells, zc, ze)
    ne, nlev = size(ze)
    lev, wt = ones(Int32, 2, ne, nlev), zeros(2, ne, nlev)
    nlev >= 2 || return lev, wt
    for e in 1:ne, s in 1:2
        c = edge_cells[s, e]
        col = view(zc, c, :)
        for k in 1:nlev
            j = clamp(searchsortedlast(col, ze[e, k]), 1, nlev - 1)
            lev[s, e, k], wt[s, e, k] = j, (ze[e, k] - col[j]) / (col[j + 1] - col[j])
        end
    end
    return lev, wt
end

"`π'` of the cells of edge `e` at its height at level `k`, right minus left (`height_interpolation`)."
@inline function height_difference(πp, edge_cells, lev, wt, e, k)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        jl, tl, jr, tr = lev[1, e, k], wt[1, e, k], lev[2, e, k], wt[2, e, k]
        return πp[r, jr] + tr * (πp[r, jr + 1] - πp[r, jr]) - πp[l, jl] - tl * (πp[l, jl + 1] - πp[l, jl])
    end
end

"`w_c` at the interfaces from the full levels `wcf`: linear inside, quadratic to the ground (wgtfacq), zero at the lid."
@kernel function concorr_interface_kernel!(wch, @Const(wcf), @Const(wlow), @Const(q), nlev)
    i, j = @index(Global, NTuple)
    @inbounds wch[i, j] = 2 <= j <= nlev ? wlow[i, j] * wcf[i, j - 1] + (1 - wlow[i, j]) * wcf[i, j] :
                          j == 1 ? (nlev >= 3 ? q[i, 1] * wcf[i, 1] + q[i, 2] * wcf[i, 2] + q[i, 3] * wcf[i, 3] :
                                    wcf[i, 1]) : 0.0
end

"`ŵ = w - w_c`, the contravariant vertical velocity at the interfaces."
@kernel function contravariant_w_kernel!(ŵ, @Const(w), @Const(wch))
    i, j = @index(Global, NTuple)
    @inbounds ŵ[i, j] = w[i, j] - wch[i, j]
end

"Pressure `p₀ π^(c_p/R)` of the lowest layer at every cell (on the host)."
lowest_pressure(m::CGridNonhydrostatic, y) = P0 .* (Rd .* Array(y.ρθ)[:, 1] ./ P0) .^ (Cpd / Cvd)

"Total mass `Σ a² A_i dz_ik ρ_ik` (`dz` per layer, or per cell and layer)."
function total_mass(m::CGridNonhydrostatic, y)
    ρ, dz = Array(y.ρ), Array(m.dz)
    return sum(Array(m.cell_areas) .* (dz isa AbstractVector ? ρ * dz : vec(sum(ρ .* dz; dims=2))))
end
