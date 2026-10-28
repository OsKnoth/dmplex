"""
    TRiSKShallowWater(dm; gravity=1, coriolis=x -> 2x[3], centers=nothing, comm=nothing, part=nothing,
                      backend=CPU())
    TRiSKShallowWater(grid::TRiSKGrid; kwargs...)

Nonlinear rotating shallow-water equations in vector-invariant form,

    ∂u/∂t + q r̂ × (h u) + ∇(K + g h) = 0,    ∂h/∂t + ∇⋅(h u) = 0,

on the C grid of a Voronoi mesh with the TRiSK operators of Thuburn et al.
(2009) and Ringler et al. (2010), as in MPAS and the hexagonal ICON
(`i_cori_method = 4` of ICON-IAP): normal velocities `u_e` on the edges,
heights `h_i` at the generators, vorticity at the vertices (dual triangles).

    (D F)_i = (1/A_i) Σ_e n_{e,i} l_e F_e            divergence
    (G φ)_e = (φ_right - φ_left) / d_e              gradient
    ζ_v     = (1/A_v) Σ_e t_{e,v} d_e u_e           curl
    K_i     = (1/A_i) Σ_e l_e d_{e,i} u_e² / 2       kinetic energy (ICON `e_inn_c`)
    F_e     = h_e u_e,  h_e = (d_{e,l} h_l + d_{e,r} h_r) / d_e
    h_v     = (1/A_v) Σ_i K_{i,v} h_i,   q_v = (ζ_v + f_v) / h_v,   q_e = (q_{v1} + q_{v2}) / 2
    ∂u_e/∂t = (1/d_e) Σ_{e'} w_{e,e'} l_{e'} F_{e'} (q_e + q_{e'})/2 - G(K + g h)_e
    ∂h_i/∂t = -(D F)_i

With the antisymmetric weights the PV flux term does no work, and `K` and `h_e`
are consistent with the energy `Σ_i A_i (h_i K_i + g h_i²/2)`: mass and energy
are conserved by the semi-discretization (up to the time-stepping error).

Every term is a KernelAbstractions kernel gathering over the neighbour tables
(no atomics); with `backend` (`CUDABackend()` after `using CUDA`) the state,
the tables, the work vectors and the halo buffers live on the device.

With an MPI communicator `comm` the model is distributed (`TRiSKDistribution`:
equal-area partition of the cells unless `part` is given, two vertex rings of
ghost cells). The state vector is the local `[u; h]` (local edges, then local
cells, owned ones first); `tendency!` exchanges the ghosts of the state once
and computes the owned tendencies, the ghost entries of `dy` are zero.
`gather_state` assembles the global state (edges by `edge_index`, cells by
`cell_index`), which serially is the state itself. `grid` is the global grid.
"""
struct TRiSKShallowWater{I, M, V, D, BE}
    grid::TRiSKGrid
    cell_edges::I
    divergence::M        # n_{e,i} l_e / A_i
    kinetic::M           # l_e d_{e,i} / (2 A_i)
    edge_cells::I
    edge_vertices::I
    edge_height::M       # d_{e,i} / d_e
    inverse_dual::V      # 1 / d_e
    vertex_edges::I
    curl::M              # t_{e,v} d_e / A_v
    vertex_cells::I
    vertex_height::M     # K_{i,v} / A_v
    coriolis::V          # f at the vertices
    trisk_edges::I
    trisk::M             # w_{e,e'} l_{e'} / d_e
    cell_areas::V        # local
    vertex_areas::V
    work::NamedTuple{(:K, :B, :F, :hv, :qv, :qe), NTuple{6, V}}
    distribution::D      # on the backend
    gravity::Float64
    backend::BE
end

function TRiSKShallowWater(grid::TRiSKGrid; gravity::Real=1.0, coriolis=x -> 2x[3], comm=nothing, part=nothing,
                           backend=KernelAbstractions.CPU())
    nc, ne, nv = num_cells(grid), num_edges(grid), num_vertices(grid)
    l, d = grid.primal_lengths, grid.dual_lengths
    # coefficients of the global grid
    divergence = zeros(size(grid.cell_edges))
    kinetic = zeros(size(grid.cell_edges))
    for i in 1:nc, k in axes(grid.cell_edges, 1)
        e, s = grid.cell_edges[k, i], grid.cell_signs[k, i]
        s == 0 && continue
        side = grid.edge_cells[1, e] == i ? 1 : 2
        divergence[k, i] = s * l[e] / grid.cell_areas[i]
        kinetic[k, i] = l[e] * grid.cell_distances[side, e] / (2 * grid.cell_areas[i])
    end
    curl = similar(grid.vertex_signs)
    for v in 1:nv, k in axes(curl, 1)
        curl[k, v] = grid.vertex_signs[k, v] * d[grid.vertex_edges[k, v]] / grid.vertex_areas[v]
    end
    vertex_height = grid.vertex_kites ./ grid.vertex_areas'
    trisk = similar(grid.trisk_weights)
    for e in 1:ne, k in axes(trisk, 1)
        trisk[k, e] = grid.trisk_weights[k, e] * l[grid.trisk_edges[k, e]] / d[e]
    end

    # restricted to the local entities, in local numbering
    dist = TRiSKDistribution(grid, comm; part)
    cells, edges, vertices = dist.cells.global_ids, dist.edges.global_ids, dist.vertex_ids
    local_cell, local_edge = local_numbers(cells, nc), local_numbers(edges, ne)
    local_vertex = local_numbers(vertices, nv)
    cell_edges, divergence = local_table(grid.cell_edges, divergence, cells, local_edge)
    _, kinetic = local_table(grid.cell_edges, kinetic, cells, local_edge)
    edge_cells, edge_height = local_table(grid.edge_cells, grid.cell_distances ./ d', edges, local_cell)
    edge_vertices, _ = local_table(grid.edge_vertices, nothing, edges, local_vertex)
    vertex_edges, curl = local_table(grid.vertex_edges, curl, vertices, local_edge)
    vertex_cells, vertex_height = local_table(grid.vertex_cells, vertex_height, vertices, local_cell)
    trisk_edges, trisk = local_table(grid.trisk_edges, trisk, edges, local_edge)

    dev(x) = to_backend(backend, x)
    zero_vector(n) = KernelAbstractions.zeros(backend, Float64, n)
    nlc, nle, nlv = length(cells), length(edges), length(vertices)
    work = (K=zero_vector(nlc), B=zero_vector(nlc), F=zero_vector(nle), hv=zero_vector(nlv), qv=zero_vector(nlv),
            qe=zero_vector(nle))
    return TRiSKShallowWater(grid, dev(cell_edges), dev(divergence), dev(kinetic), dev(edge_cells),
                             dev(edge_vertices), dev(edge_height), dev(1 ./ d[edges]), dev(vertex_edges), dev(curl),
                             dev(vertex_cells), dev(vertex_height), dev(map(coriolis, grid.vertices[vertices])),
                             dev(trisk_edges), dev(trisk), dev(grid.cell_areas[cells]),
                             dev(grid.vertex_areas[vertices]), work, to_backend(backend, dist), Float64(gravity),
                             backend)
end

TRiSKShallowWater(dm::Plex; centers=nothing, kwargs...) = TRiSKShallowWater(TRiSKGrid(dm; centers); kwargs...)

"Local state ranges of velocity (edges) and height (cells)."
velocity_dofs(m::TRiSKShallowWater) = 1:num_local(m.distribution.edges)
height_dofs(m::TRiSKShallowWater) = num_local(m.distribution.edges) .+ (1:num_local(m.distribution.cells))
"Global state ranges (of `gather_state`)."
global_velocity_dofs(m::TRiSKShallowWater) = 1:num_edges(m.grid)
global_height_dofs(m::TRiSKShallowWater) = num_edges(m.grid) .+ (1:num_cells(m.grid))

"""
Local state `[u; h]` on the model's backend: normal components of
`velocity(x)` at the edge points and point values of `height(x)` at the
generators.
"""
function initial_state(m::TRiSKShallowWater, velocity, height)
    g, d = m.grid, m.distribution
    u = [dot(velocity(g.edge_points[e]), g.edge_normals[e]) for e in d.edges.global_ids]
    return to_backend(m.backend, [u; map(height, g.centers[d.cells.global_ids])])
end

"Global state `[u; h]` with the owned values of all ranks (identical on every rank, on the host)."
function gather_state(m::TRiSKShallowWater, y)
    d = m.distribution
    return [gather(d.edges, Array(view(y, velocity_dofs(m))), num_edges(m.grid), d.comm);
            gather(d.cells, Array(view(y, height_dofs(m))), num_cells(m.grid), d.comm)]
end

"Update the ghost entries of the local state `y` (one message per neighbour)."
exchange_state!(m::TRiSKShallowWater, y) = exchange!(y, m.distribution.state, m.distribution.comm)

# --- kernels -------------------------------------------------------------------

@kernel function kinetic_energy_kernel!(K, B, @Const(u), @Const(h), @Const(cell_edges), @Const(kinetic), gravity)
    i = @index(Global)
    s = 0.0
    @inbounds for k in axes(cell_edges, 1)
        s += kinetic[k, i] * u[cell_edges[k, i]]^2
    end
    @inbounds K[i] = s
    @inbounds B[i] = s + gravity * h[i]
end

@kernel function mass_flux_kernel!(F, @Const(u), @Const(h), @Const(edge_cells), @Const(edge_height))
    e = @index(Global)
    @inbounds F[e] = (edge_height[1, e] * h[edge_cells[1, e]] + edge_height[2, e] * h[edge_cells[2, e]]) * u[e]
end

@kernel function vertex_pv_kernel!(qv, hv, @Const(u), @Const(h), @Const(vertex_edges), @Const(curl),
                                   @Const(vertex_cells), @Const(vertex_height), @Const(coriolis))
    v = @index(Global)
    ζ = η = 0.0
    @inbounds for k in axes(vertex_edges, 1)
        ζ += curl[k, v] * u[vertex_edges[k, v]]
    end
    @inbounds for k in axes(vertex_cells, 1)
        η += vertex_height[k, v] * h[vertex_cells[k, v]]
    end
    @inbounds hv[v] = η
    @inbounds qv[v] = (ζ + coriolis[v]) / η
end

@kernel function edge_pv_kernel!(qe, @Const(qv), @Const(edge_vertices))
    e = @index(Global)
    @inbounds qe[e] = (qv[edge_vertices[1, e]] + qv[edge_vertices[2, e]]) / 2
end

@kernel function momentum_kernel!(du, @Const(F), @Const(qe), @Const(B), @Const(trisk_edges), @Const(trisk),
                                  @Const(edge_cells), @Const(inverse_dual))
    e = @index(Global)
    s = 0.0
    @inbounds for k in axes(trisk_edges, 1)
        e′ = trisk_edges[k, e]
        s += trisk[k, e] * F[e′] * (qe[e] + qe[e′])
    end
    @inbounds du[e] = s / 2 - (B[edge_cells[2, e]] - B[edge_cells[1, e]]) * inverse_dual[e]
end

@kernel function continuity_kernel!(dh, @Const(F), @Const(cell_edges), @Const(divergence))
    i = @index(Global)
    s = 0.0
    @inbounds for k in axes(cell_edges, 1)
        s += divergence[k, i] * F[cell_edges[k, i]]
    end
    @inbounds dh[i] = -s
end

@kernel function edge_sum_kernel!(y, @Const(x), @Const(edges), @Const(weights))
    j = @index(Global)
    s = 0.0
    @inbounds for k in axes(edges, 1)
        s += weights[k, j] * x[edges[k, j]]
    end
    @inbounds y[j] = s
end

"""
Run `kernel` with `n` work items on the model's backend. On GPUs the kernels
of a tendency run in stream order without waiting (the halo exchange
synchronizes before MPI); on the CPU backend every kernel is waited for,
since host code reads the results.
"""
function run!(m::TRiSKShallowWater, kernel, n, args...)
    n == 0 && return nothing
    kernel(m.backend)(args...; ndrange=n)
    m.backend isa KernelAbstractions.CPU && KernelAbstractions.synchronize(m.backend)
    return nothing
end

# --- tendency and diagnostics ------------------------------------------------------

"Kinetic energy and Bernoulli function `K + g h` at the local cells, in the work vectors."
function kinetic_energy!(m::TRiSKShallowWater, u, h)
    w = m.work
    run!(m, kinetic_energy_kernel!, length(w.K), w.K, w.B, u, h, m.cell_edges, m.kinetic, m.gravity)
    return w
end

"Potential vorticity and height at the local vertices, in the work vectors."
function vertex_pv!(m::TRiSKShallowWater, u, h)
    w = m.work
    run!(m, vertex_pv_kernel!, length(w.qv), w.qv, w.hv, u, h, m.vertex_edges, m.curl, m.vertex_cells,
         m.vertex_height, m.coriolis)
    return w
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::TRiSKShallowWater)
    w, d = m.work, m.distribution
    exchange_state!(m, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    du, dh = view(dy, velocity_dofs(m)), view(dy, height_dofs(m))
    kinetic_energy!(m, u, h)
    run!(m, mass_flux_kernel!, length(w.F), w.F, u, h, m.edge_cells, m.edge_height)
    vertex_pv!(m, u, h)
    run!(m, edge_pv_kernel!, length(w.qe), w.qe, w.qv, m.edge_vertices)
    run!(m, momentum_kernel!, d.edges.n_owned, du, w.F, w.qe, w.B, m.trisk_edges, m.trisk, m.edge_cells,
         m.inverse_dual)
    run!(m, continuity_kernel!, d.cells.n_owned, dh, w.F, m.cell_edges, m.divergence)
    if is_distributed(d)
        fill!(view(du, (d.edges.n_owned + 1):length(du)), 0.0)
        fill!(view(dh, (d.cells.n_owned + 1):length(dh)), 0.0)
    end
    return dy
end

"Total mass `Σ_i A_i h_i` over the owned cells of all ranks."
total_mass(m::TRiSKShallowWater, y) =
    global_sum(m.distribution, dot(view(m.cell_areas, owned(m.distribution.cells)),
                                   view(view(y, height_dofs(m)), owned(m.distribution.cells))))

"""
Energy `Σ_i A_i (h_i K_i + g h_i²/2)` over the owned cells of all ranks,
conserved by the semi-discretization. Updates the ghost entries of `y` first.
"""
function total_energy(m::TRiSKShallowWater, y)
    exchange_state!(m, y)
    h = view(y, height_dofs(m))
    w, o = kinetic_energy!(m, view(y, velocity_dofs(m)), h), owned(m.distribution.cells)
    return global_sum(m.distribution, sum(view(m.cell_areas, o) .* view(h, o) .* (view(w.K, o) .+ m.gravity .* view(h, o) ./ 2)))
end

"Potential enstrophy `Σ_v A_v h_v q_v² / 2` over the owned vertices of all ranks."
function potential_enstrophy(m::TRiSKShallowWater, y)
    exchange_state!(m, y)
    w = vertex_pv!(m, view(y, velocity_dofs(m)), view(y, height_dofs(m)))
    o = 1:m.distribution.n_owned_vertices
    return global_sum(m.distribution, sum(view(m.vertex_areas, o) .* view(w.hv, o) .* view(w.qv, o) .^ 2) / 2)
end

"Relative vorticity `ζ_v` at the local vertices (on the model's backend; valid away from the halo edge)."
function relative_vorticity(m::TRiSKShallowWater, y)
    ζ = KernelAbstractions.zeros(m.backend, Float64, length(m.vertex_areas))
    run!(m, edge_sum_kernel!, length(ζ), ζ, view(y, velocity_dofs(m)), m.vertex_edges, m.curl)
    return ζ
end

"TRiSK tangential velocity `(1/d_e) Σ w_{e,e'} l_{e'} u_{e'}` at the local edges, approximating `u ⋅ (r̂ × n_e)`."
function tangential_velocity(m::TRiSKShallowWater, u)
    v = KernelAbstractions.zeros(m.backend, Float64, length(m.inverse_dual))
    run!(m, edge_sum_kernel!, length(v), v, u, m.trisk_edges, m.trisk)
    return v
end
