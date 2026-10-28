"""
Rows of the owned dofs of `rows` of the global matrix `A`, with the columns of the local dofs of `cols`
(a `DofDistribution` or the global column of every local column).
"""
owned_rows(A::SparseMatrixCSC, rows::DofDistribution, cols::DofDistribution) = owned_rows(A, rows, cols.global_ids)
owned_rows(A::SparseMatrixCSC, rows::DofDistribution, cols::AbstractVector{<:Integer}) =
    A[rows.global_ids[owned(rows)], cols]

"Owned rows of `A` (see `owned_rows`) as a `RowMatrix` on `backend`."
row_matrix(A::SparseMatrixCSC, rows::DofDistribution, cols, backend) =
    RowMatrix(sparse(owned_rows(A, rows, cols)'), backend)

"Vector of `values` at the owned dofs of `d` (global vector restricted, in local numbering)."
owned_values(values::AbstractVector, d::DofDistribution) = values[d.global_ids[owned(d)]]

"""
    LinearShallowWater(dm, k; gravity=1, depth=1, coriolis=x -> 2x[3], comm=nothing, part=nothing,
                       solver=:gathered, backend=CPU())
    LinearShallowWater(U, H; kwargs...)

Linearized rotating shallow-water equations on the sphere,

    ∂u/∂t + f r̂ × u + g ∇h = 0,    ∂h/∂t + H ∇⋅u = 0,

discretized with velocity in `RaviartThomas(k)` and height in
`Discontinuous(k)` (or the given spaces `U`, `H`):

    M_u u̇ = -C u + g Dᵀ h,    M_h ḣ = -H D u,

with `D[i, j] = ∫ ψ_i ∇⋅φ_j` and the antisymmetric Coriolis matrix `C`. The
semi-discrete system conserves mass `∫ h` and energy
`½ H uᵀ M_u u + ½ g hᵀ M_h h` exactly. The state vector is `[u; h]`. The mass
solvers are the cached ones of the spaces.

With an MPI communicator `comm` the model is distributed (`CellDistribution`:
equal-area partition of the cells unless `part` is given, one vertex ring of
ghost cells). The state is the local `[u; h]` (owned dofs first, then ghosts);
`tendency!` updates the ghosts by halo exchange, applies the owned rows of the
operators and solves with the distributed mass solvers (`solver=:gathered` on
the CPU, `:cudss` on GPUs; see `distributed_mass_solver`). `gather_state`
assembles the global state; `Mu`, `Mh` are the global mass matrices.

`backend` is the KernelAbstractions backend of the run-time data (state,
operator rows, work vectors, halo buffers, block-diagonal mass solvers): a
time step runs on the device. Coupled mass matrices are solved on the device
with `solver=:cudss`, otherwise through the host (`HostSolver`,
`GatheredMassSolver`).
"""
struct LinearShallowWater{SU, SH, RM, V, C, D, BE}
    U::FESpace
    H::FESpace
    Mu::SparseMatrixCSC{Float64, Int}     # global mass matrices
    Mh::SparseMatrixCSC{Float64, Int}
    Mu_solver::SU
    Mh_solver::SH
    D::SparseMatrixCSC{Float64, Int}      # global operators
    C::SparseMatrixCSC{Float64, Int}
    rows_u::RM      # owned rows of [-C  g Dᵀ], local state columns
    rows_h::RM      # owned rows of -H D, local velocity columns
    Mu_rows::RM     # owned rows of the mass matrices (energy)
    Mh_rows::RM
    mass_weights::V                       # ∫ ψ_i of the owned height dofs
    rhs_u::V
    rhs_h::V
    cells::C
    distribution::D                       # (u = DofDistribution, h = DofDistribution), on the backend
    gravity::Float64
    depth::Float64
    backend::BE
end

function LinearShallowWater(U::FESpace, H::FESpace; gravity::Real=1.0, depth::Real=1.0, coriolis=x -> 2x[3],
                            comm=nothing, part=nothing, solver::Symbol=:gathered, backend=KernelAbstractions.CPU())
    cells = CellDistribution(U.dm, comm; part)
    (du, owner_u), (dh, owner_h) = DofDistribution(U, cells), DofDistribution(H, cells)
    Mu, Mh = mass_solver(U).matrix, mass_solver(H).matrix
    D, C = assemble_divergence(H, U), assemble_coriolis(U, coriolis)
    rows_u = row_matrix([-C Float64(gravity) * sparse(D')], du, [du.global_ids; num_dofs(U) .+ dh.global_ids], backend)
    dev(x) = to_backend(backend, x)
    return LinearShallowWater(U, H, Mu, Mh, distributed_mass_solver(U, du, owner_u, comm; solver, backend),
                              distributed_mass_solver(H, dh, owner_h, comm; solver, backend), D, C, rows_u,
                              row_matrix(-Float64(depth) * D, dh, du, backend), row_matrix(Mu, du, du, backend),
                              row_matrix(Mh, dh, dh, backend), dev(owned_values(vec(sum(Mh; dims=1)), dh)),
                              dev(zeros(num_local(du))), dev(zeros(num_local(dh))), cells,
                              (u=to_backend(backend, du), h=to_backend(backend, dh)), Float64(gravity),
                              Float64(depth), backend)
end

LinearShallowWater(dm::Plex, k::Integer; kwargs...) =
    LinearShallowWater(FESpace(dm, RaviartThomas(k); name="u"), FESpace(dm, Discontinuous(k); name="h"); kwargs...)

"Local state ranges of velocity and height."
velocity_dofs(m::LinearShallowWater) = 1:num_local(m.distribution.u)
height_dofs(m::LinearShallowWater) = num_local(m.distribution.u) .+ (1:num_local(m.distribution.h))
"Global state ranges (of `gather_state`)."
global_velocity_dofs(m::LinearShallowWater) = 1:num_dofs(m.U)
global_height_dofs(m::LinearShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))

"Local state `[u; h]` from the L2 projections of velocity and height fields, on the model's backend."
function initial_state(m::LinearShallowWater, velocity, height)
    d = m.distribution
    return to_backend(m.backend, [project(m.U, velocity)[d.u.global_ids]; project(m.H, height)[d.h.global_ids]])
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::LinearShallowWater)
    d = m.distribution
    exchange_state!(m, y)
    row_mul!(m.rhs_u, m.rows_u, y)       # owned dofs come first
    row_mul!(m.rhs_h, m.rows_h, view(y, velocity_dofs(m)))
    ldiv!(view(dy, velocity_dofs(m)), m.Mu_solver, m.rhs_u)
    ldiv!(view(dy, height_dofs(m)), m.Mh_solver, m.rhs_h)
    return dy
end

"Energy `½ H uᵀ M_u u + ½ g hᵀ M_h h` over all ranks; updates the ghost entries of `y` first."
function total_energy(m::LinearShallowWater, y)
    exchange_state!(m, y)
    d = m.distribution
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    Mu, Mh = view(m.rhs_u, owned(d.u)), view(m.rhs_h, owned(d.h))
    row_mul!(Mu, m.Mu_rows, u)
    row_mul!(Mh, m.Mh_rows, h)
    energy = m.depth * dot(view(u, owned(d.u)), Mu) + m.gravity * dot(view(h, owned(d.h)), Mh)
    return global_sum(m.cells, energy / 2)
end
