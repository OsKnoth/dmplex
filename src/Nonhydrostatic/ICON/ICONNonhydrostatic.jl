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
`MPASGassmannNonhydrostatic`).

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
the old state: the column solver of `MPASGassmannNonhydrostatic` with
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
`MPASGassmannNonhydrostatic`. Serial.
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
