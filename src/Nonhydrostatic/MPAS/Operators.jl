# TRiSK operators on the extruded Voronoi C grid shared by the MPAS dycores: kinetic energy (cell and
# vertex), horizontal vorticity η, edge density and mass flux, PV at vertices and edges, w tendencies,
# the contravariant correction over orography.

"Potential temperature, Exner pressure and horizontal kinetic energy at the cells."
@kernel function diagnose_kernel!(θ, πe, K, @Const(vn), @Const(ρ), @Const(ρθ), @Const(cell_edges), @Const(kinetic))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += kinetic[j, i] * vn[cell_edges[j, i], k]^2
    end
    @inbounds K[i, k] = s
    @inbounds θ[i, k] = ρθ[i, k] / ρ[i, k]
    @inbounds πe[i, k] = (Rd * ρθ[i, k] / P0)^(Rd / Cvd)
end

"Vertex kinetic energy `Σ_e (A_{e,v}/A_v) vn_e²` of every layer."
@kernel function vertex_kinetic_kernel!(Kv, @Const(vn), @Const(vertex_edges), @Const(weights))
    v, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += weights[j, v] * vn[vertex_edges[j, v], k]^2
    end
    @inbounds Kv[v, k] = s
end

"Add the vertex part `Σ_v (1 - α) K_{i,v}/A_i K_v` to the cell kinetic energy."
@kernel function cell_from_vertex_kernel!(K, @Const(Kv), @Const(cell_vertices), @Const(kites))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_vertices, 1)
        s += kites[j, i] * Kv[cell_vertices[j, i], k]
    end
    @inbounds K[i, k] += s
end

"Add the vertical kinetic energy `(dzh_k w_k² + dzh_{k+1} w_{k+1}²) / 4dz_k` of the layer to `K`."
@kernel function w_kinetic_kernel!(K, @Const(w), @Const(dz), @Const(dzh))
    i, k = @index(Global, NTuple)
    @inbounds K[i, k] += (dzh[i, k] * w[i, k]^2 + dzh[i, k + 1] * w[i, k + 1]^2) / (4 * dz[i, k])
end

"Horizontal vorticity `η = ∂vn/∂z - ∂w/∂n` at the edges of the interior interfaces (zero at ground and top), `dzh` at the edges."
@kernel function eta_kernel!(η, @Const(vn), @Const(w), @Const(edge_cells), @Const(inverse_dual), @Const(dzh))
    e, j = @index(Global, NTuple)
    nlev = size(vn, 2)
    @inbounds if 2 <= j <= nlev
        η[e, j] = (vn[e, j] - vn[e, j - 1]) / dzh[e, j] - (w[edge_cells[2, e], j] - w[edge_cells[1, e], j]) * inverse_dual[e]
    else
        η[e, j] = 0.0
    end
end

"Density `ρ_e` at the edges of the stage state and the mass flux `ρ_e vn`, once per stage for the gathers."
@kernel function edge_density_kernel!(ρe, Fs, @Const(vn), @Const(ρ), @Const(edge_cells), @Const(edge_weight))
    e, k = @index(Global, NTuple)
    @inbounds begin
        r = edge_weight[1, e] * ρ[edge_cells[1, e], k] + edge_weight[2, e] * ρ[edge_cells[2, e], k]
        ρe[e, k] = r
        Fs[e, k] = r * vn[e, k]
    end
end

"""
Explicit tendency of `w` in vector-invariant form, `η ṽn - ∂K/∂z` at the interior
interfaces, `ṽn = (ρ_{e,j-1} vn_{j-1} + ρ_{e,j} vn_j) / 2ρ_e` (`ρ = 1` without
`massweighted`) with the cell inner product.
"""
@kernel function w_tendency_vi_kernel!(tw, @Const(η), @Const(vn), @Const(ρe), @Const(K), @Const(cell_edges),
                                       @Const(inner), @Const(dzh), @Const(wlow), massweighted)
    i, j = @index(Global, NTuple)
    nlev = size(vn, 2)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            ρl, ρu = massweighted ? (ρe[e, j - 1], ρe[e, j]) : (1.0, 1.0)
            s += inner[n, i] * η[e, j] * (ρl * vn[e, j - 1] + ρu * vn[e, j]) /
                 (2 * (wlow[i, j] * ρl + (1 - wlow[i, j]) * ρu))
        end
        tw[i, j] = s - (K[i, j] - K[i, j - 1]) / dzh[i, j]
    else
        tw[i, j] = 0.0
    end
end

"Vorticity `ζ`, density and potential vorticity `(ζ + f)/ρ` at the vertices of every layer."
@kernel function vertex_pv_kernel!(qv, ζv, ρv, @Const(vn), @Const(ρ), @Const(vertex_edges), @Const(curl),
                                   @Const(vertex_cells), @Const(vertex_weight), @Const(coriolis))
    v, k = @index(Global, NTuple)
    ζ = η = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        ζ += curl[j, v] * vn[vertex_edges[j, v], k]
    end
    @inbounds for j in axes(vertex_cells, 1)
        η += vertex_weight[j, v] * ρ[vertex_cells[j, v], k]
    end
    @inbounds ζv[v, k] = ζ
    @inbounds ρv[v, k] = η
    @inbounds qv[v, k] = (ζ + coriolis[v]) / η
end

"PV at the edges: the rhombus PV `(ζ_e + f_e)/ρ_e` (`iap`) or the mean of the vertex PV."
@kernel function edge_pv_kernel!(qe, @Const(qv), @Const(ζv), @Const(ρv), @Const(edge_vertices), @Const(rhombus),
                                 @Const(edge_coriolis), iap)
    e, k = @index(Global, NTuple)
    @inbounds v1, v2 = edge_vertices[1, e], edge_vertices[2, e]
    @inbounds if iap
        ζ = rhombus[1, e] * ζv[v1, k] + rhombus[2, e] * ζv[v2, k]
        η = rhombus[1, e] * ρv[v1, k] + rhombus[2, e] * ρv[v2, k]
        qe[e, k] = (ζ + edge_coriolis[e]) / η
    else
        qe[e, k] = (qv[v1, k] + qv[v2, k]) / 2
    end
end

"Explicit advection of `w` at the interior interfaces: `-u ⋅ ∇w - w ∂w/∂z`."
@kernel function w_tendency_kernel!(tw, @Const(w), @Const(vn), @Const(cell_edges), @Const(kinetic),
                                    @Const(edge_cells), @Const(inverse_dual), @Const(dz))
    i, j = @index(Global, NTuple)
    nlev = size(vn, 2)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            v = (vn[e, j - 1] + vn[e, j]) / 2
            s += 2 * kinetic[n, i] * v * (w[edge_cells[2, e], j] - w[edge_cells[1, e], j]) * inverse_dual[e]
        end
        tw[i, j] = -s - w[i, j] * (w[i, j + 1] - w[i, j - 1]) / (dz[i, j - 1] + dz[i, j])
    else
        tw[i, j] = 0.0
    end
end

"Contravariant correction `w_c = u ⋅ ∇z` at the full levels of the cells, `Σ_e (l_e d_{e,i}/A_i) vn_e ∂z/∂n` (cell inner product)."
@kernel function concorr_cell_kernel!(wcf, @Const(vn), @Const(cell_edges), @Const(inner), @Const(ddxn))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for n in axes(cell_edges, 1)
        e = cell_edges[n, i]
        s += inner[n, i] * vn[e, k] * ddxn[e, k]
    end
    @inbounds wcf[i, k] = s
end

"`tw += w_c ∂w/∂z` at the interior interfaces: the advection of `w` by `w - w_c` in place of `w` over orography."
@kernel function w_concorr_tendency_kernel!(tw, @Const(w), @Const(wch), @Const(dzh), nlev)
    i, j = @index(Global, NTuple)
    @inbounds if 2 <= j <= nlev
        tw[i, j] += wch[i, j] * (w[i, j + 1] - w[i, j - 1]) / (dzh[i, j] + dzh[i, j + 1])
    end
end
