"""
    ICONShallowWater(dm; gravity=1, coriolis=x -> 2x[3], rbf_scale=0.5, divergence_damping=0, centers=nothing,
                     comm=nothing, part=nothing, backend=CPU())
    ICONShallowWater(grid::TRiSKGrid; kwargs...)

Nonlinear rotating shallow-water equations in vector-invariant form on the
triangular C grid of ICON, with the horizontal operators of its
nonhydrostatic dynamical core (`velocity_tendencies`): normal velocities
`u_e` at the edge midpoints, heights at the circumcenters of the triangles,
vorticity at the vertices (the hexagonal and pentagonal dual cells).

    v_e     = Σ_k c_{e,k} u_k                       tangential velocity by vector RBF (4 edges, `rbf_coefficients`)
    K_e     = (u_e² + v_e²) / 2,   K_i = Σ_e b_{i,e} K_e     bilinear edge → cell weights (`e_bln_c_s`)
    ζ_v     = (1/A_v) Σ_e t_{e,v} d_e u_e,   ζ_e = (ζ_{v1} + ζ_{v2}) / 2
    ∂u_e/∂t = (f_e + ζ_e) v_e - [K_e (a_1 - a_2) + a_2 K_r - a_1 K_l] - g (h_r - h_l) / d_e
    ∂h_i/∂t = -(1/A_i) Σ_e n_{e,i} l_e h_e u_e,   h_e = (d_{e,r} h_l + d_{e,l} h_r) / d_e

with the kinetic-energy gradient of ICON (`coeff_gradekin`:
`a_1 = d_{e,r}/(d_{e,l} d_e)`, `a_2 = d_{e,l}/(d_{e,r} d_e)`) and the linear
interpolation of the height to the edge midpoint (`c_lin_e`; ICON's
nonhydrostatic core uses an upwind reconstruction of the density there).
Mass is conserved, energy is not exactly (the RBF tangential velocity is not
part of an antisymmetric Coriolis operator).

`divergence_damping = ν₄` adds ICON's fourth-order divergence damping
`-ν₄ ∇(∇⋅(∇(∇⋅u)))` (`divdamp_order = 4`) to the momentum tendency, with the
discrete divergence of the normal velocities at the cells and the gradient
across the edges. ICON adds `-divdamp_fac Ā² ∇∇⋅∇∇⋅u` once per dynamics time
step (`divdamp_fac = 0.0025`), which corresponds to `ν₄ = divdamp_fac Ā² / Δt`
with the mean cell area `Ā`. Without it, or another dissipation, the
triangular C grid is unstable in flows with grid-scale structure (its
divergence and checkerboard modes are not controlled): the Galewsky jet blows
up within two days.

The grid is a `TRiSKGrid` of the triangle mesh with circumcenters (the
default `cell_centers` of a triangle mesh); distribution, backends, the
state layout and the diagnostics are those of `TRiSKShallowWater` (see
`CGridShallowWater`): two vertex rings of ghost cells, one halo exchange per
tendency, KernelAbstractions kernels without atomics.
"""
struct ICONShallowWater{I, M, V, D, BE} <: CGridShallowWater
    grid::TRiSKGrid
    cell_edges::I
    divergence::M        # n_{e,i} l_e / A_i
    bilinear::M          # e_bln_c_s
    edge_cells::I
    edge_vertices::I
    edge_height::M       # c_lin_e
    gradient::M          # coeff_gradekin
    inverse_dual::V      # 1 / d_e
    rbf_edges::I
    rbf::M
    edge_coriolis::V     # f at the edges
    vertex_edges::I
    curl::M              # t_{e,v} d_e / A_v
    vertex_cells::I
    vertex_height::M     # K_{i,v} / A_v (diagnostics)
    coriolis::V          # f at the vertices (diagnostics)
    cell_areas::V
    vertex_areas::V
    work::NamedTuple{(:v, :Ke, :K, :F, :ζv, :hv, :qv, :D, :G), NTuple{9, V}}
    distribution::D
    gravity::Float64
    divergence_damping::Float64
    backend::BE
end

"""
Bilinear edge-to-cell weights of ICON (`e_bln_c_s`): `Σ_k w_k = 1` and
`Σ_k w_k x_k = 0` for the edge midpoints `x_k` in the tangent plane of the
cell center, so linear fields are not aliased into a checkerboard.
"""
function bilinear_weights(g::TRiSKGrid)
    weights = zeros(size(g.cell_edges))
    for i in 1:num_cells(g)
        c = g.centers[i]
        a = normalize(cross(c, abs(c[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)))
        b = cross(c, a)
        edges = g.cell_edges[:, i]
        length(edges) == 3 || throw(ArgumentError("bilinear weights need a triangle mesh"))
        A = [ones(3)'; [dot(g.edge_points[e], a) for e in edges]'; [dot(g.edge_points[e], b) for e in edges]']
        weights[:, i] .= A \ [1.0, 0.0, 0.0]
    end
    return weights
end

function ICONShallowWater(grid::TRiSKGrid; gravity::Real=1.0, coriolis=x -> 2x[3], rbf_scale::Real=0.5,
                          divergence_damping::Real=0.0, comm=nothing, part=nothing,
                          backend=KernelAbstractions.CPU())
    nc, ne, nv = num_cells(grid), num_edges(grid), num_vertices(grid)
    l, d, dc = grid.primal_lengths, grid.dual_lengths, grid.cell_distances
    divergence = zeros(size(grid.cell_edges))
    for i in 1:nc, k in axes(grid.cell_edges, 1)
        divergence[k, i] = grid.cell_signs[k, i] * l[grid.cell_edges[k, i]] / grid.cell_areas[i]
    end
    edge_height = [dc[2, :]'; dc[1, :]'] ./ d'
    gradient = [(dc[2, :] ./ (dc[1, :] .* d))'; (dc[1, :] ./ (dc[2, :] .* d))']
    curl = similar(grid.vertex_signs)
    for v in 1:nv, k in axes(curl, 1)
        curl[k, v] = grid.vertex_signs[k, v] * d[grid.vertex_edges[k, v]] / grid.vertex_areas[v]
    end
    rbf_edges, rbf = rbf_coefficients(grid; scale=rbf_scale)

    dist = TRiSKDistribution(grid, comm; part)
    cells, edges, vertices = dist.cells.global_ids, dist.edges.global_ids, dist.vertex_ids
    local_cell, local_edge = local_numbers(cells, nc), local_numbers(edges, ne)
    local_vertex = local_numbers(vertices, nv)
    cell_edges, divergence = local_table(grid.cell_edges, divergence, cells, local_edge)
    _, bilinear = local_table(grid.cell_edges, bilinear_weights(grid), cells, local_edge)
    edge_cells, edge_height = local_table(grid.edge_cells, edge_height, edges, local_cell)
    _, gradient = local_table(grid.edge_cells, gradient, edges, local_cell)
    edge_vertices, _ = local_table(grid.edge_vertices, nothing, edges, local_vertex)
    rbf_edges, rbf = local_table(rbf_edges, rbf, edges, local_edge)
    vertex_edges, curl = local_table(grid.vertex_edges, curl, vertices, local_edge)
    vertex_cells, vertex_height = local_table(grid.vertex_cells, grid.vertex_kites ./ grid.vertex_areas', vertices,
                                              local_cell)

    dev(x) = to_backend(backend, x)
    zero_vector(n) = KernelAbstractions.zeros(backend, Float64, n)
    nlc, nle, nlv = length(cells), length(edges), length(vertices)
    work = (v=zero_vector(nle), Ke=zero_vector(nle), K=zero_vector(nlc), F=zero_vector(nle), ζv=zero_vector(nlv),
            hv=zero_vector(nlv), qv=zero_vector(nlv), D=zero_vector(nlc), G=zero_vector(nle))
    return ICONShallowWater(grid, dev(cell_edges), dev(divergence), dev(bilinear), dev(edge_cells),
                            dev(edge_vertices), dev(edge_height), dev(gradient), dev(1 ./ d[edges]), dev(rbf_edges),
                            dev(rbf), dev(map(coriolis, grid.edge_points[edges])), dev(vertex_edges), dev(curl),
                            dev(vertex_cells), dev(vertex_height), dev(map(coriolis, grid.vertices[vertices])),
                            dev(grid.cell_areas[cells]), dev(grid.vertex_areas[vertices]), work,
                            to_backend(backend, dist), Float64(gravity), Float64(divergence_damping), backend)
end

ICONShallowWater(dm::Plex; centers=nothing, kwargs...) = ICONShallowWater(TRiSKGrid(dm; centers); kwargs...)

# --- kernels -------------------------------------------------------------------

@kernel function icon_edge_kernel!(v, Ke, F, @Const(u), @Const(h), @Const(rbf_edges), @Const(rbf),
                                   @Const(edge_cells), @Const(edge_height))
    e = @index(Global)
    s = 0.0
    @inbounds for k in axes(rbf_edges, 1)
        s += rbf[k, e] * u[rbf_edges[k, e]]
    end
    @inbounds v[e] = s
    @inbounds Ke[e] = (u[e]^2 + s^2) / 2
    @inbounds F[e] = (edge_height[1, e] * h[edge_cells[1, e]] + edge_height[2, e] * h[edge_cells[2, e]]) * u[e]
end

@kernel function icon_momentum_kernel!(du, @Const(v), @Const(Ke), @Const(K), @Const(h), @Const(ζv),
                                       @Const(edge_cells), @Const(edge_vertices), @Const(gradient),
                                       @Const(inverse_dual), @Const(edge_coriolis), gravity)
    e = @index(Global)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        ζ = (ζv[edge_vertices[1, e]] + ζv[edge_vertices[2, e]]) / 2
        grad_K = Ke[e] * (gradient[1, e] - gradient[2, e]) + gradient[2, e] * K[r] - gradient[1, e] * K[l]
        du[e] = (edge_coriolis[e] + ζ) * v[e] - grad_K - gravity * (h[r] - h[l]) * inverse_dual[e]
    end
end

@kernel function edge_gradient_kernel!(G, @Const(D), @Const(edge_cells), @Const(inverse_dual), scale)
    e = @index(Global)
    @inbounds G[e] = scale * (D[edge_cells[2, e]] - D[edge_cells[1, e]]) * inverse_dual[e]
end

# --- tendency ------------------------------------------------------------------------

"""
Tangential velocity, edge and cell kinetic energy and mass flux at the local
edges and cells, in the work vectors.
"""
function kinetic_energy!(m::ICONShallowWater, u, h)
    w = m.work
    run!(m, icon_edge_kernel!, length(w.v), w.v, w.Ke, w.F, u, h, m.rbf_edges, m.rbf, m.edge_cells, m.edge_height)
    run!(m, edge_sum_kernel!, length(w.K), w.K, w.Ke, m.cell_edges, m.bilinear)
    return w
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::ICONShallowWater)
    w, d = m.work, m.distribution
    exchange_state!(m, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    du, dh = view(dy, velocity_dofs(m)), view(dy, height_dofs(m))
    kinetic_energy!(m, u, h)
    run!(m, edge_sum_kernel!, length(w.ζv), w.ζv, u, m.vertex_edges, m.curl)
    run!(m, icon_momentum_kernel!, d.edges.n_owned, du, w.v, w.Ke, w.K, h, w.ζv, m.edge_cells, m.edge_vertices,
         m.gradient, m.inverse_dual, m.edge_coriolis, m.gravity)
    run!(m, continuity_kernel!, d.cells.n_owned, dh, w.F, m.cell_edges, m.divergence)
    m.divergence_damping > 0 && divergence_damping!(du, m, u)
    if is_distributed(d)
        fill!(view(du, (d.edges.n_owned + 1):length(du)), 0.0)
        fill!(view(dh, (d.cells.n_owned + 1):length(dh)), 0.0)
    end
    return dy
end

"Add `-ν₄ ∇∇⋅∇∇⋅u` to the owned momentum tendencies (`continuity_kernel!` computes `-∇⋅`)."
function divergence_damping!(du, m::ICONShallowWater, u)
    w, n = m.work, m.distribution.edges.n_owned
    run!(m, continuity_kernel!, length(w.D), w.D, u, m.cell_edges, m.divergence)                   # -∇⋅u
    run!(m, edge_gradient_kernel!, length(w.G), w.G, w.D, m.edge_cells, m.inverse_dual, 1.0)        # -∇∇⋅u
    run!(m, continuity_kernel!, length(w.D), w.D, w.G, m.cell_edges, m.divergence)                 # ∇⋅∇∇⋅u
    run!(m, edge_gradient_kernel!, n, w.G, w.D, m.edge_cells, m.inverse_dual, -m.divergence_damping)
    view(du, 1:n) .+= view(w.G, 1:n)
    return du
end

"RBF tangential velocity at the local edges, approximating `u ⋅ (r̂ × n_e)`."
function tangential_velocity(m::ICONShallowWater, u)
    v = KernelAbstractions.zeros(m.backend, Float64, length(m.inverse_dual))
    run!(m, edge_sum_kernel!, length(v), v, u, m.rbf_edges, m.rbf)
    return v
end
