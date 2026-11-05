# Horizontal diffusion of MPAS-Gassmann: Smagorinsky with dissipative heating (ICON-IAP `diff_diss_hori`) and the optional hyperdiffusion.

"Divergence and velocity vector (Perot) at the cells."
@kernel function smagorinsky_cell_kernel!(δ, ux, uy, uz, @Const(vn), @Const(cell_edges), @Const(divergence),
                                          @Const(px), @Const(py), @Const(pz))
    i, k = @index(Global, NTuple)
    s = x = y = z = 0.0
    @inbounds for j in axes(cell_edges, 1)
        v = vn[cell_edges[j, i], k]
        s += divergence[j, i] * v
        x += px[j, i] * v
        y += py[j, i] * v
        z += pz[j, i] * v
    end
    @inbounds δ[i, k], ux[i, k], uy[i, k], uz[i, k] = s, x, y, z
end

"Vorticity, density (kite average) and velocity vector (barycentric interpolation of the cells) at the vertices."
@kernel function smagorinsky_vertex_kernel!(ζ, ρv, vx, vy, vz, @Const(vn), @Const(ρ), @Const(ux), @Const(uy),
                                            @Const(uz), @Const(vertex_edges), @Const(curl), @Const(vertex_cells),
                                            @Const(vertex_weight), @Const(barycentric))
    v, k = @index(Global, NTuple)
    s = r = x = y = z = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += curl[j, v] * vn[vertex_edges[j, v], k]
    end
    @inbounds for j in axes(vertex_cells, 1)
        c, a, b = vertex_cells[j, v], vertex_weight[j, v], barycentric[j, v]
        r += a * ρ[c, k]
        x += b * ux[c, k]
        y += b * uy[c, k]
        z += b * uz[c, k]
    end
    @inbounds ζ[v, k], ρv[v, k], vx[v, k], vy[v, k], vz[v, k] = s, r, x, y, z
end

"Smagorinsky coefficient `K = c A_e |D|` at the edges from the velocity gradient across and along the edge."
@kernel function smagorinsky_coefficient_kernel!(Ke, @Const(ux), @Const(uy), @Const(uz), @Const(vx), @Const(vy),
                                                 @Const(vz), @Const(edge_cells), @Const(edge_vertices), @Const(frame),
                                                 @Const(inverse_dual), @Const(inverse_primal), @Const(coefficient))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        v1, v2 = edge_vertices[1, e], edge_vertices[2, e]
        an = SVector(ux[r, k] - ux[l, k], uy[r, k] - uy[l, k], uz[r, k] - uz[l, k]) * inverse_dual[e]
        at = SVector(vx[v2, k] - vx[v1, k], vy[v2, k] - vy[v1, k], vz[v2, k] - vz[v1, k]) * inverse_primal[e]
        n = SVector(frame[1, e], frame[2, e], frame[3, e])
        t = SVector(frame[4, e], frame[5, e], frame[6, e])
        tension = dot(n, an) - dot(t, at)
        shear = dot(n, at) + dot(t, an)
        Ke[e, k] = coefficient[k] * sqrt(tension^2 + shear^2)
    end
end

"`ρK` at the cells (area mean of the edges) and vertices (mean of the edges)."
@kernel function smagorinsky_cell_coefficient_kernel!(ρKc, @Const(Ke), @Const(ρ), @Const(cell_edges),
                                                      @Const(cell_edge_area))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += cell_edge_area[j, i] * Ke[cell_edges[j, i], k]
    end
    @inbounds ρKc[i, k] = ρ[i, k] * s
end

@kernel function smagorinsky_vertex_coefficient_kernel!(ρKv, @Const(Ke), @Const(ρv), @Const(vertex_edges))
    v, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += Ke[vertex_edges[j, v], k]
    end
    @inbounds ρKv[v, k] = ρv[v, k] * s / size(vertex_edges, 1)
end

"Add `(1/ρ_e)[∇(ρKδ) - k × ∇(ρKζ)]⋅n` to `hd`."
@kernel function smagorinsky_tendency_kernel!(hd, @Const(ρKc), @Const(ρKv), @Const(δ), @Const(ζ), @Const(ρ),
                                              @Const(edge_cells), @Const(edge_vertices), @Const(edge_weight),
                                              @Const(inverse_dual), @Const(inverse_primal))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        v1, v2 = edge_vertices[1, e], edge_vertices[2, e]
        ρe = edge_weight[1, e] * ρ[l, k] + edge_weight[2, e] * ρ[r, k]
        hd[e, k] += ((ρKc[r, k] * δ[r, k] - ρKc[l, k] * δ[l, k]) * inverse_dual[e] -
                     (ρKv[v2, k] * ζ[v2, k] - ρKv[v1, k] * ζ[v1, k]) * inverse_primal[e]) / ρe
    end
end

"Dissipative heating `∂ρθ/∂t = (ρK_i δ_i² + Σ_v (K_{i,v}/A_i) ρK_v ζ_v²) / (c_p π_i)`."
@kernel function smagorinsky_heating_kernel!(hρθ, @Const(ρKc), @Const(ρKv), @Const(δ), @Const(ζ), @Const(ρθ),
                                             @Const(cell_vertices), @Const(cell_kites))
    i, k = @index(Global, NTuple)
    Q = 0.0
    @inbounds for j in axes(cell_vertices, 1)
        v = cell_vertices[j, i]
        Q += cell_kites[j, i] * ρKv[v, k] * ζ[v, k]^2
    end
    @inbounds Q += ρKc[i, k] * δ[i, k]^2
    @inbounds hρθ[i, k] = Q / (Cpd * (Rd * ρθ[i, k] / P0)^(Rd / Cvd))
end

"Vector Laplacian of `vn` times `scale` into `L`."
function vector_laplacian!(L, m::MPASGassmannNonhydrostatic, vn, scale=1.0)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    launch!(m, divergence_only_kernel!, (nc, nlev), w.δ, vn, m.cell_edges, m.divergence)
    launch!(m, vorticity_kernel!, (nv, nlev), w.ζ, vn, m.vertex_edges, m.curl)
    launch!(m, vector_laplacian_kernel!, (ne, nlev), L, w.δ, w.ζ, m.edge_cells, m.edge_vertices, m.inverse_dual,
            m.inverse_primal, scale)
    return L
end

"Hyperdiffusion `-ν₄ L(L(vn))` into `hd` (zero without hyperdiffusion)."
function hyperdiffusion!(hd, m::MPASGassmannNonhydrostatic, vn)
    if m.hyperdiffusion == 0
        fill!(hd, 0.0)
    else
        vector_laplacian!(m.work.Lvn, m, vn)
        vector_laplacian!(hd, m, m.work.Lvn, -m.hyperdiffusion)
    end
    return hd
end

"""
    smagorinsky!(hd, hρθ, m, y)

Add the Smagorinsky diffusion of `vn` in the state `y` to `hd` and put its
heating into `hρθ` (zero without diffusion).
"""
function smagorinsky!(hd, hρθ, m::MPASGassmannNonhydrostatic, y)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    fill!(hρθ, 0.0)
    all(iszero, m.smagorinsky) && return hd
    launch!(m, smagorinsky_cell_kernel!, (nc, nlev), w.δ, w.uc..., y.vn, m.cell_edges, m.divergence, m.perot...)
    launch!(m, smagorinsky_vertex_kernel!, (nv, nlev), w.ζ, w.ρv, w.uv..., y.vn, y.ρ, w.uc..., m.vertex_edges, m.curl,
            m.vertex_cells, m.vertex_weight, m.barycentric)
    launch!(m, smagorinsky_coefficient_kernel!, (ne, nlev), w.Ke, w.uc..., w.uv..., m.edge_cells, m.edge_vertices,
            m.edge_frame, m.inverse_dual, m.inverse_primal, m.smagorinsky)
    launch!(m, smagorinsky_cell_coefficient_kernel!, (nc, nlev), w.ρKc, w.Ke, y.ρ, m.cell_edges, m.cell_edge_area)
    launch!(m, smagorinsky_vertex_coefficient_kernel!, (nv, nlev), w.ρKv, w.Ke, w.ρv, m.vertex_edges)
    launch!(m, smagorinsky_tendency_kernel!, (ne, nlev), hd, w.ρKc, w.ρKv, w.δ, w.ζ, y.ρ, m.edge_cells,
            m.edge_vertices, m.edge_weight, m.inverse_dual, m.inverse_primal)
    launch!(m, smagorinsky_heating_kernel!, (nc, nlev), hρθ, w.ρKc, w.ρKv, w.δ, w.ζ, y.ρθ, m.cell_vertices,
            m.cell_kites)
    return hd
end
