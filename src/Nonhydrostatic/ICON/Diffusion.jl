# Horizontal diffusion of ICON (`mo_nh_diffusion`, `hdiff_order = 5`): Smagorinsky plus fourth-order
# background diffusion of vn, fourth-order diffusion of w, Smagorinsky diffusion of θ; and the optional
# hyperdiffusion `-ν₄ L(L(vn))` in the dynamics step.

"Velocity vector at the vertices by the vector RBF of the edge field `vn` (ICON `rbf_vec_interpol_vertex`)."
@kernel function icon_vertex_velocity_kernel!(ux, uy, uz, @Const(vn), @Const(edges), @Const(cx), @Const(cy), @Const(cz))
    v, k = @index(Global, NTuple)
    a = b = c = 0.0
    @inbounds for j in axes(edges, 1)
        x = vn[edges[j, v], k]
        a += cx[j, v] * x; b += cy[j, v] * x; c += cz[j, v] * x
    end
    @inbounds ux[v, k], uy[v, k], uz[v, k] = a, b, c
end

"""
Component of the vertex velocity `(ux, uy, uz)` at quadrilateral vertex `j` of
edge `e` along `n_e` (`t = false`) or `t_e` (`t = true`), both projected onto
the tangent plane of the vertex (ICON `primal_normal_vert`, `dual_normal_vert`).
"""
@inline function vertex_component(ux, uy, uz, quad, quad_vectors, j, k, e, t)
    @inbounds begin
        v, o = quad[j, e], 6 * (j - 1) + 3 * t
        return ux[v, k] * quad_vectors[o + 1, e] + uy[v, k] * quad_vectors[o + 2, e] + uz[v, k] * quad_vectors[o + 3, e]
    end
end

"""
Smagorinsky coefficient `kh` and `∇²vn` at the edges from the vertex velocities
on the quadrilateral of the edge: vertices 1, 2 at its ends (along `t_e`), 3, 4
opposite in the left and right cell (along `n_e`). `kh = smag(z) Δt |D|` with
the tension `∂u_n/∂n - ∂u_t/∂t` and the shear `∂u_n/∂t + ∂u_t/∂n` of the
deformation `D`, less `offset` and limited by `limit(z)`.
"""
@kernel function icon_smagorinsky_kernel!(kh, n2, @Const(vn), @Const(ux), @Const(uy), @Const(uz), @Const(quad),
                                          @Const(quad_vectors), @Const(inverse_vv), @Const(inverse_primal),
                                          @Const(smag), dt, offset, @Const(limit))
    e, k = @index(Global, NTuple)
    @inbounds begin
        c(j, t) = vertex_component(ux, uy, uz, quad, quad_vectors, j, k, e, t)
        un = ntuple(j -> c(j, false), 4)
        dut_t = c(2, true) - c(1, true)
        dut_n = c(4, true) - c(3, true)
        ivv, ip = inverse_vv[e], inverse_primal[e]
        tension = (un[4] - un[3]) * ivv - dut_t * ip
        shear = (un[2] - un[1]) * ip + dut_n * ivv
        kh[e, k] = min(max(0.0, smag[k] * dt * sqrt(tension^2 + shear^2) - offset), limit[k])
        # twice the distance from the edge to the vertices
        n2[e, k] = 4 * ((un[4] + un[3] - 2 * vn[e, k]) * ivv^2 + (un[2] + un[1] - 2 * vn[e, k]) * ip^2)
    end
end

"`vn += A_e (kh ∇²vn - A_e k₄ ∇⁴vn)` with `∇⁴vn` from the vertex values of `∇²vn` on the quadrilateral."
@kernel function icon_vn_diffusion_kernel!(vn, @Const(kh), @Const(n2), @Const(ux), @Const(uy), @Const(uz), @Const(quad),
                                           @Const(quad_vectors), @Const(inverse_vv), @Const(inverse_primal),
                                           @Const(edge_area), k4)
    e, k = @index(Global, NTuple)
    @inbounds begin
        c(j) = vertex_component(ux, uy, uz, quad, quad_vectors, j, k, e, false)
        n4 = 4 * ((c(3) + c(4) - 2 * n2[e, k]) * inverse_vv[e]^2 + (c(1) + c(2) - 2 * n2[e, k]) * inverse_primal[e]^2)
        A = edge_area[e]
        vn[e, k] += A * (kh[e, k] * n2[e, k] - A * k4 * n4)
    end
end

"Cell Laplacian `∇⋅∇x` (ICON `geofac_n2s`) of `x` at the cells, for the levels `j0:j1`."
@kernel function icon_cell_laplacian_kernel!(L, @Const(x), @Const(cell_edges), @Const(edge_cells), @Const(divergence),
                                             @Const(inverse_dual), j0, j1)
    i, j = @index(Global, NTuple)
    s = 0.0
    @inbounds if j0 <= j <= j1
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            s += divergence[n, i] * (x[edge_cells[2, e], j] - x[edge_cells[1, e], j]) * inverse_dual[e]
        end
    end
    @inbounds L[i, j] = s
end

"`w -= k₄w A² ∇²(∇²w) - n2w A ∇²w` at the interior interfaces (`∇²w` in `L`)."
@kernel function icon_w_diffusion_kernel!(w, @Const(L), @Const(cell_edges), @Const(edge_cells), @Const(divergence),
                                          @Const(inverse_dual), @Const(cell_areas), k4w, @Const(n2w), nlev)
    i, j = @index(Global, NTuple)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            s += divergence[n, i] * (L[edge_cells[2, e], j] - L[edge_cells[1, e], j]) * inverse_dual[e]
        end
        A = cell_areas[i]
        w[i, j] += -k4w * A^2 * s + n2w[j] * A * L[i, j]
    end
end

"""
ICON's enhanced diffusion of grid-point cold pools in the lowest two layers:
`(-5 K - Δθ) 5e-4` where `θ` is more than 7.5 K below the mean of the three
neighbours (`Δθ < -7.5 K`; flat ground, no reference state), else zero.
"""
@inline function cold_pool(θ, i, k, cell_edges, edge_cells)
    @inbounds begin
        s = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            l, r = edge_cells[1, e], edge_cells[2, e]
            s += θ[l == i ? r : l, k]
        end
        dθ = θ[i, k] - s / 3
        return dθ < -7.5 ? (-5 - dθ) * 5e-4 : 0.0
    end
end

"Flux `kh ∇θ` at the edges (`kh` raised by `cold_pool` in the lowest two layers)."
@kernel function icon_theta_flux_kernel!(F, @Const(kh), @Const(θ), @Const(cell_edges), @Const(edge_cells),
                                         @Const(inverse_dual))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        K = kh[e, k]
        k <= 2 && (K = max(K, cold_pool(θ, l, k, cell_edges, edge_cells), cold_pool(θ, r, k, cell_edges, edge_cells)))
        F[e, k] = K * (θ[r, k] - θ[l, k]) * inverse_dual[e]
    end
end

"`θ += A ∇⋅F` at constant `ρ` (ICON updates θ_v and the Exner pressure with it)."
@kernel function icon_theta_diffusion_kernel!(ρθ, @Const(ρ), @Const(θ), @Const(F), @Const(cell_edges),
                                              @Const(divergence), @Const(cell_areas))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for n in axes(cell_edges, 1)
        s += divergence[n, i] * F[cell_edges[n, i], k]
    end
    @inbounds ρθ[i, k] = ρ[i, k] * (θ[i, k] + cell_areas[i] * s)
end

"""
    icon_diffusion!(m::ICONNonhydrostatic, y, dt)

ICON's horizontal diffusion (`mo_nh_diffusion`, `hdiff_order = 5`,
`itype_vn_diffu = 1`, `itype_t_diffu = 2`) of the state `y` over the advective
time step `dt` (`ndyn_substeps` dynamics steps), in place.
"""
function icon_diffusion!(m::ICONNonhydrostatic, y, dt)
    w, d, nlev = m.work, m.diffusion, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    u, u2 = w.uvert, w.uvert2
    launch!(m, icon_vertex_velocity_kernel!, (nv, nlev), u..., y.vn, d.rbf_edges, d.rbf...)
    launch!(m, icon_smagorinsky_kernel!, (ne, nlev), w.kh, w.n2e, y.vn, u..., d.quad, d.quad_vectors, d.inverse_vv,
            m.inverse_primal, d.smag, dt, d.offset, d.limit)
    if d.lhdiff_vn
        launch!(m, icon_vertex_velocity_kernel!, (nv, nlev), u2..., w.n2e, d.rbf_edges, d.rbf...)
        launch!(m, icon_vn_diffusion_kernel!, (ne, nlev), y.vn, w.kh, w.n2e, u2..., d.quad, d.quad_vectors,
                d.inverse_vv, m.inverse_primal, d.edge_area, d.k4)
    end
    if d.lhdiff_w
        launch!(m, icon_cell_laplacian_kernel!, (nc, nlev + 1), w.Lw, y.w, m.cell_edges, m.edge_cells, m.divergence,
                m.inverse_dual, 2, nlev)
        launch!(m, icon_w_diffusion_kernel!, (nc, nlev + 1), y.w, w.Lw, m.cell_edges, m.edge_cells, m.divergence,
                m.inverse_dual, m.cell_areas, d.k4w, d.n2w, nlev)
    end
    if d.lhdiff_temp
        @. w.θ = y.ρθ / y.ρ
        launch!(m, icon_theta_flux_kernel!, (ne, nlev), w.F, w.kh, w.θ, m.cell_edges, m.edge_cells, m.inverse_dual)
        launch!(m, icon_theta_diffusion_kernel!, (nc, nlev), y.ρθ, y.ρ, w.θ, w.F, m.cell_edges, m.divergence,
                m.cell_areas)
    end
    return y
end

"Hyperdiffusion `-ν₄ L(L(vn))` with the C-grid vector Laplacian `∇δ - k × ∇ζ`."
function icon_hyperdiffusion!(hd, m::ICONNonhydrostatic, vn)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    for (out, x, s) in ((w.Lvn, vn, 1.0), (hd, w.Lvn, -m.hyperdiffusion))
        launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ, x, m.cell_edges, m.divergence)
        launch!(m, vorticity_kernel!, (nv, nlev), w.ζ, x, m.vertex_edges, m.curl)
        launch!(m, vector_laplacian_kernel!, (ne, nlev), out, w.δ, w.ζ, m.edge_cells, m.edge_vertices,
                m.inverse_dual, m.inverse_primal, s)
    end
    return hd
end
