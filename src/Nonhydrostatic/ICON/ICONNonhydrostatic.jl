"""
    ICONNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius=6.37122e6, omega=7.292e-5,
                       beta=0.65, damping_height=Inf, damping_rate=0, divdamp_fac=0.0025,
                       veladv_offctr=0.25, exner_expol=1/3, rbf_scale=0.5, hyperdiffusion=0,
                       backend=CPU())

Dry compressible Euler equations on the extruded triangular C grid of ICON
(Zängl et al. 2015, `mo_solve_nonhydro`, `mo_velocity_advection`), shallow
atmosphere, flat ground. `grid` is the `TRiSKGrid` of a triangle mesh with
circumcentres. Lorenz staggering: `vn` at the edges × layers, `w` at the cells
× interfaces, `ρ`, `ρθ` at the cells × layers (state `(vn, w, ρ, ρθ)` as for
`TRiSKNonhydrostatic`).

**Velocity tendencies** (`mo_velocity_advection`), horizontal operators of
`ICONShallowWater` per layer:

    vt_e    = Σ_k c_{e,k} vn_k                     tangential velocity, vector RBF (4 edges)
    K_e     = (vn_e² + vt_e²)/2,  K_i = Σ_e b_{i,e} K_e      bilinear weights (e_bln_c_s)
    ζ_e     = (ζ_{v1} + ζ_{v2})/2,  ζ_v = (1/A_v) Σ t_{e,v} d_e vn_e
    ∂vn/∂t  = (f_e + ζ_e) vt_e - [K_e (a₁ - a₂) + a₂ K_r - a₁ K_l] - w_e ∂vn/∂z
    ∂w/∂t   = -Σ_e b_{i,e} (vn_ie ∂w/∂n + vt_ie ∂w/∂t)_e - w ∂w/∂z

with the kinetic-energy gradient of ICON (`coeff_gradekin`), `vn_ie` the
interpolation of `vn` to the interfaces (`wgtfac_e`), `w_e` the full-level `w`
at the edges (`c_lin_e`), and `∂w/∂t` along the edge from `w` at the vertices
(area weights, `cells_aw_verts`).

**Fast waves** (`solve_nh`): horizontal pressure gradient `-c_p θ_e ∂π/∂n`,
mass flux `ρ_e vn` and `ρθ` flux `ρ_e θ_e vn` with `ρ_e`, `θ_e` from the
second-order upwind reconstruction of Miura (2007): the value of the upwind
cell plus its Green–Gauss gradient times the distance from the cell centre to
the backtrajectory point `x_e - (Δt/2)(vn n_e + vt t_e)` (ICON
`upwind_hflux_miura` with `iadv_rhotheta = 2`). The vertical terms are
implicit per column with the off-centring `beta` (`0.5 + vwind_offctr`),
centred θ and ρ at the interfaces, linearized around the Exner pressure of
the old state: the column solver of `TRiSKNonhydrostatic` with
`vertical_theta = :centred` and the lumped vertical mass.

**Time stepping** (`step!`): the two-time-level predictor–corrector of ICON.
Predictor: velocity tendencies at `n`, the pressure gradient of the Exner
pressure extrapolated in time, `π^n + exner_expol (π^n - π^{n-1})`, and
`ρ_e`, `θ_e` of `n`; `vn* = vn^n + Δt (T_vn^n - c_p θ_e ∂π/∂n)`, the fluxes of
`vn*` and the column solve give `w*`, `ρ*`, `ρθ*`. Corrector: velocity
tendencies at the predictor state, weighted `(0.5 - veladv_offctr, 0.5 +
veladv_offctr)`; the same pressure gradient and `ρ_e`, `θ_e`;
`vn^{n+1} = vn^n + Δt (Σ w T_vn - c_p θ_e ∂π/∂n)` plus the fourth-order
divergence damping of `vn*`, `-divdamp_fac Ā² ∇(∇⋅∇(∇⋅vn*))` (`divdamp_order =
4`, the 2D divergence, `Ā` the mean cell area); then the fluxes of
`vn^{n+1}` and a second column solve with the weighted `w` tendency.

Differences to ICON: `ρθ` is prognostic and `π` follows from the equation of
state in every step (ICON carries `exner` and `θ_v` and updates `exner` with
the linearized equation); no reference state (with flat ground its horizontal
gradient vanishes); no dynamics substeps, tracer or physics coupling; the
horizontal diffusion is an optional `hyperdiffusion` `-ν₄ L(L(vn))` in place of
ICON's `mo_nh_diffusion`; the Rayleigh damping of `w` as in
`TRiSKNonhydrostatic`. Serial.
"""
struct ICONNonhydrostatic{M, V, I, BE} <: CGridNonhydrostatic
    grid::TRiSKGrid
    vgrid::VerticalGrid
    cell_edges::I
    divergence::M        # n_{e,i} l_e / (a A_i)
    bilinear::M          # e_bln_c_s
    edge_cells::I
    edge_vertices::I
    edge_height::M       # c_lin_e
    gradient::M          # coeff_gradekin / a
    inverse_dual::V      # 1 / (a d_e)
    inverse_primal::V    # 1 / (a l_e)
    tangent_sign::V      # ±1: t_e = x_e × n_e points from vertex 1 to vertex 2 (+1) or back
    rbf_edges::I
    rbf::M
    vertex_edges::I
    curl::M              # t_{e,v} d_e / (a A_v)
    vertex_cells::I
    vertex_weight::M     # K_{i,v} / A_v (cells_aw_verts)
    edge_coriolis::V
    green_gauss::NTuple{3, M}   # n_{e,i} l_e n_e / A_i on the unit sphere: Green–Gauss gradient at the cells
    edge_vectors::M      # 12 × ne: x_e - x_l, x_e - x_r, n_e, t_e (unit sphere)
    cell_areas::V        # a² A_i
    dz::V
    dzh::V
    wlow::V
    rayleigh::V
    work::NamedTuple
    radius::Float64
    beta::Float64
    divdamp::Float64     # divdamp_fac Ā²
    weights::NTuple{2, Float64}   # velocity tendencies of n and of the predictor in the corrector
    exner_expol::Float64
    hyperdiffusion::Float64
    vertical_theta::Int  # 0: centred, as ICON
    backend::BE
end

function ICONNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius::Real=6.37122e6, omega::Real=7.292e-5,
                            beta::Real=0.65, damping_height::Real=Inf, damping_rate::Real=0.0,
                            divdamp_fac::Real=0.0025, veladv_offctr::Real=0.25, exner_expol::Real=1 / 3,
                            rbf_scale::Real=0.5, hyperdiffusion::Real=0.0, backend=KernelAbstractions.CPU())
    size(grid.cell_edges, 1) == 3 || throw(ArgumentError("ICONNonhydrostatic needs a triangle mesh"))
    a = Float64(radius)
    nc, ne, nv, nlev = num_cells(grid), num_edges(grid), num_vertices(grid), num_layers(vgrid)
    l, d, dc = grid.primal_lengths, grid.dual_lengths, grid.cell_distances
    divergence = zeros(3, nc)
    green_gauss = ntuple(_ -> zeros(3, nc), 3)
    for i in 1:nc, k in 1:3
        e = grid.cell_edges[k, i]
        divergence[k, i] = grid.cell_signs[k, i] * l[e] / (a * grid.cell_areas[i])
        g = grid.cell_signs[k, i] * l[e] / grid.cell_areas[i] .* grid.edge_normals[e]
        foreach(c -> green_gauss[c][k, i] = g[c], 1:3)
    end
    edge_height = [dc[2, :]'; dc[1, :]'] ./ d'
    gradient = [(dc[2, :] ./ (dc[1, :] .* d))'; (dc[1, :] ./ (dc[2, :] .* d))'] ./ a
    curl = similar(grid.vertex_signs)
    for v in 1:nv, k in axes(curl, 1)
        curl[k, v] = grid.vertex_signs[k, v] * d[grid.vertex_edges[k, v]] / (a * grid.vertex_areas[v])
    end
    rbf_edges, rbf = rbf_coefficients(grid; scale=rbf_scale)
    tangent_sign = zeros(ne)
    edge_vectors = zeros(12, ne)
    for e in 1:ne
        x, n = grid.edge_points[e], grid.edge_normals[e]
        t = cross(x, n)
        v1, v2 = grid.vertices[grid.edge_vertices[1, e]], grid.vertices[grid.edge_vertices[2, e]]
        tangent_sign[e] = sign(dot(t, v2 - v1))
        edge_vectors[1:3, e] .= x - grid.centers[grid.edge_cells[1, e]]
        edge_vectors[4:6, e] .= x - grid.centers[grid.edge_cells[2, e]]
        edge_vectors[7:9, e] .= n
        edge_vectors[10:12, e] .= t
    end
    z_top = vgrid.z_half[end]
    rayleigh = map(vgrid.z_half) do z
        z > damping_height ? damping_rate * sin(π / 2 * (z - damping_height) / (z_top - damping_height))^2 : 0.0
    end
    dev(x) = to_backend(backend, x)
    z(n, k) = KernelAbstractions.zeros(backend, Float64, n, k)
    work = (θ=z(nc, nlev), π=z(nc, nlev), πprev=z(nc, nlev), first=Ref(true), K=z(nc, nlev), Ke=z(ne, nlev),
            vt=z(ne, nlev), ζv=z(nv, nlev), wv=z(nv, nlev + 1), gw=z(ne, nlev + 1),
            tvn_now=z(ne, nlev), tvn_new=z(ne, nlev), tw_now=z(nc, nlev + 1), tw_new=z(nc, nlev + 1),
            tw=z(nc, nlev + 1), grad=ntuple(_ -> z(nc, nlev), 6), ρe=z(ne, nlev), θe=z(ne, nlev),
            pg=z(ne, nlev), F=z(ne, nlev), Fθ=z(ne, nlev), hρθ=z(nc, nlev), ρexpl=z(nc, nlev), ρθexpl=z(nc, nlev),
            θi=z(nc, nlev + 1), θm=z(nc, nlev), band=KernelAbstractions.zeros(backend, Float64, nc, nlev, 2),
            cprime=z(nc, nlev + 1), dprime=z(nc, nlev + 1), wmass=dev(vertical_mass(vgrid, :lumped)),
            π0=z(nc, nlev), γ0=z(nc, nlev), δ=z(nc, nlev), G=z(ne, nlev), ζ=z(nv, nlev), Lvn=z(ne, nlev),
            hd=z(ne, nlev), y0=(vn=z(ne, nlev), w=z(nc, nlev + 1), ρ=z(nc, nlev), ρθ=z(nc, nlev)))
    mean_area = 4π * a^2 / nc
    return ICONNonhydrostatic(grid, vgrid, dev(grid.cell_edges), dev(divergence), dev(bilinear_weights(grid)),
                              dev(grid.edge_cells), dev(grid.edge_vertices), dev(edge_height), dev(gradient),
                              dev(1 ./ (a .* d)), dev(1 ./ (a .* l)), dev(tangent_sign), dev(rbf_edges), dev(rbf),
                              dev(grid.vertex_edges), dev(curl), dev(grid.vertex_cells),
                              dev(grid.vertex_kites ./ grid.vertex_areas'), dev([2omega * x[3] for x in grid.edge_points]),
                              map(dev, green_gauss), dev(edge_vectors), dev(a^2 .* grid.cell_areas),
                              dev(vgrid.dz), dev(vgrid.dzh), dev(vgrid.wlow), dev(rayleigh), work, a, Float64(beta),
                              Float64(divdamp_fac) * mean_area^2, (0.5 - veladv_offctr, 0.5 + veladv_offctr),
                              Float64(exner_expol), Float64(hyperdiffusion), 0, backend)
end

ICONNonhydrostatic(dm::Plex, vgrid::VerticalGrid; kwargs...) = ICONNonhydrostatic(TRiSKGrid(dm), vgrid; kwargs...)

# --- kernels -------------------------------------------------------------------

"θ and the Exner pressure at the cells."
@kernel function icon_thermo_kernel!(θ, πe, @Const(ρ), @Const(ρθ))
    i, k = @index(Global, NTuple)
    @inbounds θ[i, k] = ρθ[i, k] / ρ[i, k]
    @inbounds πe[i, k] = (Rd * ρθ[i, k] / P0)^(Rd / Cvd)
end

"Tangential velocity (RBF) and kinetic energy at the edges."
@kernel function icon_edge_kinetic_kernel!(vt, Ke, @Const(vn), @Const(rbf_edges), @Const(rbf))
    e, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(rbf_edges, 1)
        s += rbf[j, e] * vn[rbf_edges[j, e], k]
    end
    @inbounds vt[e, k] = s
    @inbounds Ke[e, k] = (vn[e, k]^2 + s^2) / 2
end

"Cell sum `Σ_j c_{j,i} x_{e_j}` of an edge field (bilinear cell average, divergence)."
@kernel function icon_cell_sum_kernel!(y, @Const(x), @Const(cell_edges), @Const(c))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += c[j, i] * x[cell_edges[j, i], k]
    end
    @inbounds y[i, k] = s
end

"Velocity tendency of `vn`: vorticity flux, kinetic-energy gradient (`coeff_gradekin`), vertical advection."
@kernel function icon_vn_tendency_kernel!(tvn, @Const(vn), @Const(vt), @Const(Ke), @Const(K), @Const(ζv), @Const(w),
                                          @Const(edge_cells), @Const(edge_vertices), @Const(edge_height),
                                          @Const(gradient), @Const(edge_coriolis), @Const(dz), @Const(wlow), nlev)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        ζ = (ζv[edge_vertices[1, e], k] + ζv[edge_vertices[2, e], k]) / 2
        g1, g2 = gradient[1, e], gradient[2, e]
        grad_K = Ke[e, k] * (g1 - g2) + g2 * K[r, k] - g1 * K[l, k]
        we = (edge_height[1, e] * (w[l, k] + w[l, k + 1]) + edge_height[2, e] * (w[r, k] + w[r, k + 1])) / 2
        vlow = k > 1 ? wlow[k] * vn[e, k - 1] + (1 - wlow[k]) * vn[e, k] : vn[e, k]
        vup = k < nlev ? wlow[k + 1] * vn[e, k] + (1 - wlow[k + 1]) * vn[e, k + 1] : vn[e, k]
        tvn[e, k] = (edge_coriolis[e] + ζ) * vt[e, k] - grad_K - we * (vup - vlow) / dz[k]
    end
end

"`w` at the vertices (area weights of the cells)."
@kernel function icon_vertex_w_kernel!(wv, @Const(w), @Const(vertex_cells), @Const(weight))
    v, j = @index(Global, NTuple)
    s = 0.0
    @inbounds for n in axes(vertex_cells, 1)
        c = vertex_cells[n, v]
        c > 0 && (s += weight[n, v] * w[c, j])
    end
    @inbounds wv[v, j] = s
end

"`vn_ie ∂w/∂n + vt_ie ∂w/∂t` at the edges of the interior interfaces."
@kernel function icon_edge_w_gradient_kernel!(gw, @Const(vn), @Const(vt), @Const(w), @Const(wv), @Const(edge_cells),
                                              @Const(edge_vertices), @Const(inverse_dual), @Const(inverse_primal),
                                              @Const(tangent_sign), @Const(wlow), nlev)
    e, j = @index(Global, NTuple)
    @inbounds if 2 <= j <= nlev
        vni = wlow[j] * vn[e, j - 1] + (1 - wlow[j]) * vn[e, j]
        vti = wlow[j] * vt[e, j - 1] + (1 - wlow[j]) * vt[e, j]
        dn = (w[edge_cells[2, e], j] - w[edge_cells[1, e], j]) * inverse_dual[e]
        dt = tangent_sign[e] * (wv[edge_vertices[2, e], j] - wv[edge_vertices[1, e], j]) * inverse_primal[e]
        gw[e, j] = vni * dn + vti * dt
    else
        gw[e, j] = 0.0
    end
end

"Advection of `w` at the interior interfaces: `-Σ_e b_{i,e} gw_e - w ∂w/∂z`."
@kernel function icon_w_tendency_kernel!(tw, @Const(gw), @Const(w), @Const(cell_edges), @Const(bilinear), @Const(dz),
                                         nlev)
    i, j = @index(Global, NTuple)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            s += bilinear[n, i] * gw[cell_edges[n, i], j]
        end
        tw[i, j] = -s - w[i, j] * (w[i, j + 1] - w[i, j - 1]) / (dz[j - 1] + dz[j])
    else
        tw[i, j] = 0.0
    end
end

"Green–Gauss gradients of `ρ` and `θ` at the cells (edge values by `c_lin_e`), unit sphere."
@kernel function icon_cell_gradient_kernel!(g1, g2, g3, h1, h2, h3, @Const(ρ), @Const(θ), @Const(cell_edges),
                                            @Const(edge_cells), @Const(edge_height), @Const(gx), @Const(gy),
                                            @Const(gz))
    i, k = @index(Global, NTuple)
    a1 = a2 = a3 = b1 = b2 = b3 = 0.0
    @inbounds for n in axes(cell_edges, 1)
        e = cell_edges[n, i]
        l, r = edge_cells[1, e], edge_cells[2, e]
        ρe = edge_height[1, e] * ρ[l, k] + edge_height[2, e] * ρ[r, k]
        θe = edge_height[1, e] * θ[l, k] + edge_height[2, e] * θ[r, k]
        a1 += gx[n, i] * ρe; a2 += gy[n, i] * ρe; a3 += gz[n, i] * ρe
        b1 += gx[n, i] * θe; b2 += gy[n, i] * θe; b3 += gz[n, i] * θe
    end
    @inbounds g1[i, k], g2[i, k], g3[i, k] = a1, a2, a3
    @inbounds h1[i, k], h2[i, k], h3[i, k] = b1, b2, b3
end

"""
`ρ_e`, `θ_e` of Miura's second-order upwind reconstruction at the
backtrajectory point of the edge over `Δt/2`, and the horizontal pressure
gradient `c_p θ_e ∂π/∂n` of the extrapolated Exner pressure.
"""
@kernel function icon_miura_kernel!(ρe, θe, pg, @Const(vn), @Const(vt), @Const(ρ), @Const(θ), @Const(πex),
                                    @Const(g1), @Const(g2), @Const(g3), @Const(h1), @Const(h2), @Const(h3),
                                    @Const(edge_cells), @Const(edge_vectors), @Const(inverse_dual), half_dt_a)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        u, v = vn[e, k], vt[e, k]
        up, o = u >= 0 ? (l, 0) : (r, 3)
        d1 = edge_vectors[o + 1, e] - half_dt_a * (u * edge_vectors[7, e] + v * edge_vectors[10, e])
        d2 = edge_vectors[o + 2, e] - half_dt_a * (u * edge_vectors[8, e] + v * edge_vectors[11, e])
        d3 = edge_vectors[o + 3, e] - half_dt_a * (u * edge_vectors[9, e] + v * edge_vectors[12, e])
        ρe[e, k] = ρ[up, k] + g1[up, k] * d1 + g2[up, k] * d2 + g3[up, k] * d3
        t = θ[up, k] + h1[up, k] * d1 + h2[up, k] * d2 + h3[up, k] * d3
        θe[e, k] = t
        pg[e, k] = Cpd * t * (πex[r, k] - πex[l, k]) * inverse_dual[e]
    end
end

"`vn = vn0 + Δt (a T_now + b T_new - pg + hd)`."
@kernel function icon_vn_update_kernel!(vn, @Const(vn0), @Const(tnow), @Const(tnew), @Const(pg), @Const(hd), dt, a, b)
    e, k = @index(Global, NTuple)
    @inbounds vn[e, k] = vn0[e, k] + dt * (a * tnow[e, k] + b * tnew[e, k] - pg[e, k] + hd[e, k])
end

"Mass and `ρθ` fluxes `ρ_e vn`, `ρ_e θ_e vn`."
@kernel function icon_flux_kernel!(F, Fθ, @Const(vn), @Const(ρe), @Const(θe))
    e, k = @index(Global, NTuple)
    @inbounds f = ρe[e, k] * vn[e, k]
    @inbounds F[e, k] = f
    @inbounds Fθ[e, k] = f * θe[e, k]
end

"`G = scale (δ_r - δ_l)/d_e`, the gradient of a cell field at the edges."
@kernel function icon_edge_gradient_kernel!(G, @Const(δ), @Const(edge_cells), @Const(inverse_dual), scale)
    e, k = @index(Global, NTuple)
    @inbounds G[e, k] = scale * (δ[edge_cells[2, e], k] - δ[edge_cells[1, e], k]) * inverse_dual[e]
end

# --- time stepping ---------------------------------------------------------------

"Velocity tendencies of the state `y` into `tvn`, `tw`; θ, π, `vt` of `y` stay in the work arrays."
function velocity_tendencies!(tvn, tw, m::ICONNonhydrostatic, y)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    launch!(m, icon_thermo_kernel!, (nc, nlev), w.θ, w.π, y.ρ, y.ρθ)
    launch!(m, icon_edge_kinetic_kernel!, (ne, nlev), w.vt, w.Ke, y.vn, m.rbf_edges, m.rbf)
    launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.K, w.Ke, m.cell_edges, m.bilinear)
    launch!(m, vorticity_kernel!, (nv, nlev), w.ζv, y.vn, m.vertex_edges, m.curl)
    launch!(m, icon_vn_tendency_kernel!, (ne, nlev), tvn, y.vn, w.vt, w.Ke, w.K, w.ζv, y.w, m.edge_cells,
            m.edge_vertices, m.edge_height, m.gradient, m.edge_coriolis, m.dz, m.wlow, nlev)
    launch!(m, icon_vertex_w_kernel!, (nv, nlev + 1), w.wv, y.w, m.vertex_cells, m.vertex_weight)
    launch!(m, icon_edge_w_gradient_kernel!, (ne, nlev + 1), w.gw, y.vn, w.vt, y.w, w.wv, m.edge_cells,
            m.edge_vertices, m.inverse_dual, m.inverse_primal, m.tangent_sign, m.wlow, nlev)
    launch!(m, icon_w_tendency_kernel!, (nc, nlev + 1), tw, w.gw, y.w, m.cell_edges, m.bilinear, m.dz, nlev)
    return nothing
end

"Horizontal flux divergence of `vn` with `ρ_e`, `θ_e`, then the column solve from the base state `y0` into `y`."
function fast_waves!(m::ICONNonhydrostatic, y, y0, ys, tw, dt)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    launch!(m, icon_flux_kernel!, (ne, nlev), w.F, w.Fθ, y.vn, w.ρe, w.θe)
    launch!(m, divergence_kernel!, (nc, nlev), w.ρexpl, w.ρθexpl, y0.ρ, y0.ρθ, w.F, w.Fθ, w.hρθ, m.cell_edges,
            m.divergence, dt)
    launch!(m, theta_interface_kernel!, nc, w.θi, w.band, w.θm, w.θ, y0.ρ, y0.ρθ, ys.ρ, ys.w, m.dz, m.dzh, m.wlow,
            dt, nlev, false)
    launch!(m, vertical_solve_kernel!, nc, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρθ, w.π0, w.γ0, w.θ, w.θi,
            ys.ρ, tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.wlow, m.rayleigh, w.wmass, dt, m.beta, nlev, false)
    return y
end

"""
    step!(m::ICONNonhydrostatic, y, dt, ys=zero_state(m))

One predictor–corrector step of ICON (see `ICONNonhydrostatic`) in place;
`ys` holds the predictor state.
"""
function step!(m::ICONNonhydrostatic, y, dt, ys=zero_state(m))
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    y0 = w.y0
    foreach(copyto!, y0, y)
    launch!(m, base_exner_kernel!, (nc, nlev), w.π0, w.γ0, y0.ρθ)
    w.first[] && (copyto!(w.πprev, w.π0); w.first[] = false)
    m.hyperdiffusion == 0 ? fill!(w.hd, 0.0) : icon_hyperdiffusion!(w.hd, m, y0.vn)
    # predictor: velocity tendencies of n, ρ_e, θ_e and the pressure gradient of the extrapolated π
    velocity_tendencies!(w.tvn_now, w.tw_now, m, y0)
    @. w.π = (1 + m.exner_expol) * w.π0 - m.exner_expol * w.πprev
    g = w.grad
    launch!(m, icon_cell_gradient_kernel!, (nc, nlev), g..., y0.ρ, w.θ, m.cell_edges, m.edge_cells, m.edge_height,
            m.green_gauss...)
    launch!(m, icon_miura_kernel!, (ne, nlev), w.ρe, w.θe, w.pg, y0.vn, w.vt, y0.ρ, w.θ, w.π, g..., m.edge_cells,
            m.edge_vectors, m.inverse_dual, dt / (2 * m.radius))
    copyto!(w.πprev, w.π0)
    launch!(m, icon_vn_update_kernel!, (ne, nlev), ys.vn, y0.vn, w.tvn_now, w.tvn_now, w.pg, w.hd, dt, 1.0, 0.0)
    fast_waves!(m, ys, y0, y0, w.tw_now, dt)
    # corrector: velocity tendencies of the predictor state, weighted, and the divergence damping of vn*
    velocity_tendencies!(w.tvn_new, w.tw_new, m, ys)
    a, b = m.weights
    launch!(m, icon_vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn_now, w.tvn_new, w.pg, w.hd, dt, a, b)
    if m.divdamp > 0
        launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ, ys.vn, m.cell_edges, m.divergence)
        launch!(m, icon_edge_gradient_kernel!, (ne, nlev), w.G, w.δ, m.edge_cells, m.inverse_dual, 1.0)
        launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ, w.G, m.cell_edges, m.divergence)
        launch!(m, icon_edge_gradient_kernel!, (ne, nlev), w.G, w.δ, m.edge_cells, m.inverse_dual, -m.divdamp)
        y.vn .+= w.G
    end
    @. w.tw = a * w.tw_now + b * w.tw_new
    fast_waves!(m, y, y0, ys, w.tw, dt)
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
