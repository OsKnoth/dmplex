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
`(-5 K - Δθ + Δθ_ref) 5e-4` where `Δθ - Δθ_ref`, the deviation `Δθ` of `θ` from the
mean of the three neighbours less that of the reference state `θref`, is below
-5 K in a valley (`Δθ_ref < 0`) or below -7.5 K elsewhere, else zero.
"""
@inline function cold_pool(θ, θref, i, k, cell_edges, edge_cells)
    @inbounds begin
        s = sr = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            l, r = edge_cells[1, e], edge_cells[2, e]
            s += θ[l == i ? r : l, k]
            sr += θref[l == i ? r : l, k]
        end
        dθ, dθr = θ[i, k] - s / 3, θref[i, k] - sr / 3
        return dθ - dθr < -5 && dθr < 0 || dθ - dθr < -7.5 ? (-5 - dθ + dθr) * 5e-4 : 0.0
    end
end

"Flux `kh ∇θ` at the edges (`kh` raised by `cold_pool` in the lowest two layers)."
@kernel function icon_theta_flux_kernel!(F, @Const(kh), @Const(θ), @Const(θref), @Const(cell_edges),
                                         @Const(edge_cells), @Const(inverse_dual))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        K = kh[e, k]
        k <= 2 && (K = max(K, cold_pool(θ, θref, l, k, cell_edges, edge_cells),
                           cold_pool(θ, θref, r, k, cell_edges, edge_cells)))
        F[e, k] = K * (θ[r, k] - θ[l, k]) * inverse_dual[e]
    end
end

"""
    truly_horizontal_points(grid, z, divergence, inverse_dual; cntrwgt=0.5, thslp=0.025, thhgtd=200, active=true)
        -> (cell, lev, coef, nbr, nlev, wt, geo)

The points of ICON's truly horizontal θ diffusion over steep slopes (`l_zdiffu_t`). The slope `s`
and the height difference `h` of cell `i` at layer `k`: the largest `|Δz|/d` and `|Δz|` of the
layer heights `z` across its edges, averaged over the cell and its neighbours (`cell_average`).
A column takes part where its lowest layer has `s ≥ thslp` or `h ≥ thhgtd`, from the lowest layer
above the lowest layers of all neighbours up to the highest layer with that property. At each
point `p` (cell `cell[p]`, layer `lev[p]`) the neighbour `nbr[n, p]` has its `θ` interpolated
linearly to the height of the point between its layers `nlev[n, p]` and `nlev[n, p] + 1` (weight
`wt[n, p]`), `geo[n, p]` its weight in `∇²θ`, and `coef[p] = min(1/500, max(√(s - thslp)/250,
2e-4 √(h - thhgtd)))`. Empty for `active = false`.
"""
function truly_horizontal_points(grid::TRiSKGrid, z::AbstractMatrix, divergence::AbstractMatrix,
                                 inverse_dual::AbstractVector; cntrwgt::Real=0.5, thslp::Real=0.025,
                                 thhgtd::Real=200.0, active::Bool=true)
    nc, nlev = size(z)
    ce, ec = grid.cell_edges, grid.edge_cells
    cell, lev, coef = Int32[], Int32[], Float64[]
    nbr, nlevs, wt, geo = Int32[], Int32[], Float64[], Float64[]
    if active && nlev >= 2
        nb, C = cell_average(grid; cntrwgt)
        slope, height = cell_slopes(ce, ec, z, inverse_dual)
        avg(x, i, k) = C[1, i] * x[i, k] + sum(C[1 + n, i] * x[nb[n, i], k] for n in 1:3)
        steep(i, k) = avg(slope, i, k) >= thslp || avg(height, i, k) >= thhgtd
        for i in 1:nc
            steep(i, 1) || continue
            zmin = maximum(z[nb[n, i], 1] for n in 1:3)
            k0, k1 = findfirst(k -> z[i, k] >= zmin, 1:nlev), findlast(k -> steep(i, k), 1:nlev)
            (k0 === nothing || k0 > k1) && continue
            for k in k0:k1
                s, h = avg(slope, i, k), avg(height, i, k)
                push!(cell, i); push!(lev, k)
                push!(coef, min(1 / 500, max(sqrt(max(0.0, s - thslp)) / 250, 2e-4 * sqrt(max(0.0, h - thhgtd)))))
                for n in 1:3
                    e, j = ce[n, i], nb[n, i]
                    col = view(z, j, :)
                    jl = clamp(searchsortedlast(col, z[i, k]), 1, nlev - 1)
                    push!(nbr, j); push!(nlevs, jl); push!(wt, (z[i, k] - col[jl]) / (col[jl + 1] - col[jl]))
                    push!(geo, divergence[n, i] * (ec[1, e] == i ? 1 : -1) * inverse_dual[e])
                end
            end
        end
    end
    r(x) = reshape(x, 3, :)
    return (cell=cell, lev=lev, coef=coef, nbr=r(nbr), nlev=r(nlevs), wt=r(wt), geo=r(geo))
end

"`T = c ∇²θ` at the points of `truly_horizontal_points`, with `θ` of the neighbours at the height of the point."
@kernel function icon_zdiffu_kernel!(T, @Const(θ), @Const(cell), @Const(lev), @Const(coef), @Const(nbr), @Const(nlev),
                                     @Const(wt), @Const(geo))
    p = @index(Global)
    @inbounds begin
        i, k = cell[p], lev[p]
        s = 0.0
        for n in 1:3
            j, l, t = nbr[n, p], nlev[n, p], wt[n, p]
            s += geo[n, p] * (θ[j, l] + t * (θ[j, l + 1] - θ[j, l]) - θ[i, k])
        end
        T[i, k] = coef[p] * s
    end
end

"""
`θ += A (∇⋅F + T)` at constant `ρ` (`T` of `icon_zdiffu_kernel!`, zero elsewhere), the Exner
pressure linearized with it, `π (1 + R/c_v (θ_new/θ - 1))`.
"""
@kernel function icon_theta_diffusion_kernel!(ρθ, exner, @Const(ρ), @Const(θ), @Const(F), @Const(T),
                                              @Const(cell_edges), @Const(divergence), @Const(cell_areas))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for n in axes(cell_edges, 1)
        s += divergence[n, i] * F[cell_edges[n, i], k]
    end
    @inbounds begin
        θn = θ[i, k] + cell_areas[i] * (s + T[i, k])
        ρθ[i, k] = ρ[i, k] * θn
        exner[i, k] *= 1 + ICON_RD / ICON_CVD * (θn / θ[i, k] - 1)
    end
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
        launch!(m, icon_theta_flux_kernel!, (ne, nlev), w.F, w.kh, w.θ, d.θref, m.cell_edges, m.edge_cells,
                m.inverse_dual)
        p = d.zdiffu
        isempty(p.cell) || launch!(m, icon_zdiffu_kernel!, (length(p.cell),), w.Tθ, w.θ, p.cell, p.lev, p.coef, p.nbr,
                                   p.nlev, p.wt, p.geo)
        launch!(m, icon_theta_diffusion_kernel!, (nc, nlev), y.ρθ, y.exner, y.ρ, w.θ, w.F, w.Tθ, m.cell_edges,
                m.divergence, m.cell_areas)
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
