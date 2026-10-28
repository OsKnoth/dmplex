"""
    NonlinearShallowWater(dm, k; gravity=1, coriolis=x -> 2x[3], degree, comm=nothing, part=nothing,
                          solver=:gathered, backend=CPU())
    NonlinearShallowWater(U, H, Q; kwargs...)

Nonlinear rotating shallow-water equations in vector-invariant form,

    ∂u/∂t + q r̂ × (h u) + ∇(K + g h) = 0,    ∂h/∂t + ∇⋅(h u) = 0,

with kinetic energy `K = |u|²/2` and potential vorticity `q = ω/h`,
`ω = ζ + f`, discretized in the compatible finite-element framework of
Cotter & Shipton (2012) and McRae & Cotter (2014), as in Gusto/Firedrake, with
`u, F ∈ RaviartThomas(k)`, `h, K ∈ Discontinuous(k)` and
`ω ∈ Lagrange(k + 1)` (or the given spaces `U`, `H`, `Q`):

    ∫ γ ω   = -∫ u ⋅ (r̂ × ∇γ) + ∫ γ f        (absolute vorticity)
    ∫ w ⋅ F = ∫ w ⋅ h u                       (mass flux)
    ∫ φ K   = ∫ φ |u|²/2
    ∫ w ⋅ u̇ = -∫ w ⋅ (ω/h r̂ × F) + ∫ (∇⋅w) (g h + K)
    ∫ φ ḣ   = -∫ φ ∇⋅F

The vorticity is diagnosed with the constant mass matrix of `Q`, so all mass
matrices are factorized once (and shared through `mass_solver`); the PV
`q = ω/h` is evaluated pointwise at the quadrature points. The PV flux term
does no work on `F` for any `q`, so mass and energy `∫ h|u|²/2 + g h²/2` are
conserved by the semi-discretization. The state vector is `[u; h]`. Nonlinear
terms are integrated with quadrature `degree`.

With an MPI communicator `comm` the model is distributed like
`LinearShallowWater` (owned cells plus one vertex ring of ghost cells, local
numbering, `solver=:gathered` or `:cudss` for the coupled mass matrices of
`U` and `Q`): the nonlinear terms are integrated over the local cells, which
gives complete right-hand sides for the owned dofs, and the diagnosed fields
`ω`, `F`, `K` are valid on all local dofs after the distributed solves.

The nonlinear terms are KernelAbstractions kernels over the local cells (one
thread per cell, atomic scatter) on the physical bases tabulated once
(`CellTables`); with `backend` (`CPU()` by default, `CUDABackend()` after
`using CUDA`) all run-time data lives on the device, as for
`LinearShallowWater`. All cells must have the same shape.
"""
struct NonlinearShallowWater{TU, TH, TQ, SU, SH, SQ, RM, V, W, C, D, BE}
    U::FESpace
    H::FESpace
    Q::FESpace
    basis_u::TU     # CellTables of the local cells (owned first), local dof indices
    basis_h::TH
    basis_q::TQ
    Mu::SparseMatrixCSC{Float64, Int}     # global mass matrices
    Mh::SparseMatrixCSC{Float64, Int}
    Mu_solver::SU
    Mh_solver::SH
    Mq_solver::SQ
    D::SparseMatrixCSC{Float64, Int}   # ∫ φ_i ∇⋅w_j (global)
    R::SparseMatrixCSC{Float64, Int}   # ∫ w_j ⋅ (r̂ × ∇γ_i) (global)
    Dt_rows::RM     # owned rows of Dᵀ, local height columns
    D_rows::RM      # owned rows of D, local velocity columns
    R_rows::RM      # owned rows of R, local velocity columns
    coriolis_load::V                   # ∫ γ_i f at the owned vorticity dofs
    mass_weights::V                    # ∫ ψ_i of the owned height dofs
    work::W
    cells::C
    distribution::D                    # (u, h, q) DofDistributions, on the backend
    gravity::Float64
    backend::BE
end

"""
    CellTables

Physical basis of one space on the local cells at the points of one
quadrature rule, as dense arrays for kernels: `φ[dof, point, cell]` (numbers,
or Cartesian vectors for H(div)) with the orientation signs applied, the
local dof `indices[dof, cell]`, the area weights `dA[point, cell]` and the
unit surface `normals[point, cell]`.
"""
struct CellTables{F, I, M, N}
    φ::F
    indices::I
    dA::M
    normals::N
end

"Tables of the local cells of `cells` with the local dof indices of `d`, on `backend`."
function CellTables(V::FESpace, degree::Integer, cells::CellDistribution, d::DofDistribution, backend)
    to_local = zeros(Int, num_dofs(V))
    to_local[d.global_ids] .= 1:num_local(d)
    T = Tabulation(V, degree)
    first_cell = first(cell_range(V.dm)) - 1
    bases = [cell_basis(T, first_cell + c) for c in cells.cell_ids]
    allequal(size(b.φ) for b in bases) || throw(ArgumentError("all cells must have the same shape"))
    (na, nq), nc = size(first(bases).φ), length(bases)
    φ = similar(first(bases).φ, na, nq, nc)
    indices, dA, normals = zeros(Int, na, nc), zeros(nq, nc), Matrix{SVector{3, Float64}}(undef, nq, nc)
    for (c, b) in enumerate(bases)
        φ[:, :, c] .= b.φ
        indices[:, c] .= to_local[b.indices]
        dA[:, c] .= b.dA
        normals[:, c] .= b.normals
    end
    return CellTables((to_backend(backend, x) for x in (φ, indices, dA, normals))...)
end

function NonlinearShallowWater(U::FESpace, H::FESpace, Q::FESpace; gravity::Real=1.0, coriolis=x -> 2x[3],
                               degree::Integer=3 * U.family.order + 6, comm=nothing, part=nothing,
                               solver::Symbol=:gathered, backend=KernelAbstractions.CPU())
    cells = CellDistribution(U.dm, comm; part)
    (du, owner_u), (dh, owner_h), (dq, owner_q) = DofDistribution(U, cells), DofDistribution(H, cells),
                                                  DofDistribution(Q, cells)
    D = assemble_divergence(H, U; degree)
    R = assemble_bilinear(Q, U; degree) do t, u, a, b, q
        dot(u.φ[b, q], cross(t.normals[q], t.grad[a, q])) * t.dA[q]
    end
    Mu, Mh = mass_solver(U).matrix, mass_solver(H).matrix
    nu, nh, nq = num_local(du), num_local(dh), num_local(dq)
    work = NamedTuple{(:ω, :F, :K, :p, :rhs_ω, :rhs_F, :rhs_K, :rhs_u, :rhs_h, :cell_values)}(Tuple(
        KernelAbstractions.zeros(backend, Float64, n) for n in (nq, nu, nh, nh, nq, nu, nh, nu, nh, cells.n_owned_cells)))
    mass_solver_of(V, d, owner) = distributed_mass_solver(V, d, owner, comm; solver, backend)
    dev(x) = to_backend(backend, x)
    return NonlinearShallowWater(U, H, Q, CellTables(U, degree, cells, du, backend),
                                 CellTables(H, degree, cells, dh, backend), CellTables(Q, degree, cells, dq, backend),
                                 Mu, Mh, mass_solver_of(U, du, owner_u), mass_solver_of(H, dh, owner_h),
                                 mass_solver_of(Q, dq, owner_q), D, R, row_matrix(sparse(D'), du, dh, backend),
                                 row_matrix(D, dh, du, backend), row_matrix(R, dq, du, backend),
                                 dev(owned_values(assemble_load(Q, coriolis; degree), dq)),
                                 dev(owned_values(vec(sum(Mh; dims=1)), dh)), work, cells,
                                 (u=dev(du), h=dev(dh), q=dev(dq)), Float64(gravity), backend)
end

NonlinearShallowWater(dm::Plex, k::Integer; kwargs...) =
    NonlinearShallowWater(FESpace(dm, RaviartThomas(k); name="u"), FESpace(dm, Discontinuous(k); name="h"),
                          FESpace(dm, Lagrange(k + 1); name="q"); kwargs...)

velocity_dofs(m::NonlinearShallowWater) = 1:num_local(m.distribution.u)
height_dofs(m::NonlinearShallowWater) = num_local(m.distribution.u) .+ (1:num_local(m.distribution.h))
global_velocity_dofs(m::NonlinearShallowWater) = 1:num_dofs(m.U)
global_height_dofs(m::NonlinearShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))

function initial_state(m::NonlinearShallowWater, velocity, height)
    d = m.distribution
    return to_backend(m.backend, [project(m.U, velocity)[d.u.global_ids]; project(m.H, height)[d.h.global_ids]])
end

# --- kernels -------------------------------------------------------------------

"Value of a discrete field with coefficients `x` at point `q` of cell `c`."
@inline function cell_value(x, φ, indices, q, c)
    s = zero(eltype(φ))
    @inbounds for a in axes(indices, 1)
        s += x[indices[a, c]] * φ[a, q, c]
    end
    return s
end

@kernel function flux_and_kinetic_kernel!(rhs_F, rhs_K, @Const(u), @Const(h), @Const(φu), @Const(iu), @Const(φh),
                                          @Const(ih), @Const(dA))
    c = @index(Global)
    @inbounds for q in axes(dA, 1)
        uq, w = cell_value(u, φu, iu, q, c), dA[q, c]
        hw = cell_value(h, φh, ih, q, c) * w
        for a in axes(iu, 1)
            @atomic rhs_F[iu[a, c]] += dot(φu[a, q, c], uq) * hw
        end
        kinetic = dot(uq, uq) / 2 * w
        for a in axes(ih, 1)
            @atomic rhs_K[ih[a, c]] += φh[a, q, c] * kinetic
        end
    end
end

@kernel function pv_flux_kernel!(rhs_u, @Const(ω), @Const(h), @Const(F), @Const(φu), @Const(iu), @Const(φh),
                                 @Const(ih), @Const(φq), @Const(iq), @Const(dA), @Const(normals))
    c = @index(Global)
    @inbounds for q in axes(dA, 1)
        pv = cell_value(ω, φq, iq, q, c) / cell_value(h, φh, ih, q, c)
        flux = (pv * dA[q, c]) * cross(normals[q, c], cell_value(F, φu, iu, q, c))
        for a in axes(iu, 1)
            @atomic rhs_u[iu[a, c]] -= dot(φu[a, q, c], flux)
        end
    end
end

@kernel function cell_energy_kernel!(energy, @Const(u), @Const(h), @Const(φu), @Const(iu), @Const(φh), @Const(ih),
                                     @Const(dA), gravity)
    c = @index(Global)
    e = 0.0
    @inbounds for q in axes(dA, 1)
        uq, hq = cell_value(u, φu, iu, q, c), cell_value(h, φh, ih, q, c)
        e += (hq * dot(uq, uq) + gravity * hq^2) / 2 * dA[q, c]
    end
    @inbounds energy[c] = e
end

@kernel function cell_enstrophy_kernel!(enstrophy, @Const(ω), @Const(h), @Const(φq), @Const(iq), @Const(φh),
                                        @Const(ih), @Const(dA))
    c = @index(Global)
    e = 0.0
    @inbounds for q in axes(dA, 1)
        e += cell_value(ω, φq, iq, q, c)^2 / (2 * cell_value(h, φh, ih, q, c)) * dA[q, c]
    end
    @inbounds enstrophy[c] = e
end

# --- diagnostics and tendency ----------------------------------------------------

"""
    diagnostics(m, u, h) -> (ω, F, K)

Absolute vorticity, mass flux and kinetic energy of the state `(u, h)` (local
vectors; their ghost entries are updated first), on all local dofs. Returns copies.
"""
function diagnostics(m::NonlinearShallowWater, u::AbstractVector, h::AbstractVector)
    d, comm = m.distribution, m.cells.comm
    exchange!(u, d.u, comm)
    exchange!(h, d.h, comm)
    w = compute_diagnostics!(m, u, h)
    return copy(w.ω), copy(w.F), copy(w.K)
end

"Diagnostics of a local state with valid ghost entries, in the work vectors."
function compute_diagnostics!(m::NonlinearShallowWater, u::AbstractVector, h::AbstractVector)
    w, d, bu, bh = m.work, m.distribution, m.basis_u, m.basis_h
    row_mul!(w.rhs_ω, m.R_rows, u, -1.0)     # owned dofs come first
    view(w.rhs_ω, owned(d.q)) .+= m.coriolis_load
    ldiv!(w.ω, m.Mq_solver, w.rhs_ω)
    fill!(w.rhs_F, 0.0)
    fill!(w.rhs_K, 0.0)
    launch!(flux_and_kinetic_kernel!, w.rhs_F, num_local_cells(m.cells), w.rhs_F, w.rhs_K, u, h, bu.φ, bu.indices,
            bh.φ, bh.indices, bu.dA)
    ldiv!(w.F, m.Mu_solver, w.rhs_F)
    ldiv!(w.K, m.Mh_solver, w.rhs_K)
    return w
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::NonlinearShallowWater)
    w, bu, bh, bq = m.work, m.basis_u, m.basis_h, m.basis_q
    exchange_state!(m, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    compute_diagnostics!(m, u, h)
    @. w.p = m.gravity * h + w.K
    fill!(w.rhs_u, 0.0)
    row_mul!(w.rhs_u, m.Dt_rows, w.p)
    launch!(pv_flux_kernel!, w.rhs_u, num_local_cells(m.cells), w.rhs_u, w.ω, h, w.F, bu.φ, bu.indices, bh.φ,
            bh.indices, bq.φ, bq.indices, bu.dA, bu.normals)
    row_mul!(w.rhs_h, m.D_rows, w.F, -1.0)
    ldiv!(view(dy, velocity_dofs(m)), m.Mu_solver, w.rhs_u)
    ldiv!(view(dy, height_dofs(m)), m.Mh_solver, w.rhs_h)
    return dy
end

"Energy `∫ h|u|²/2 + g h²/2` over the owned cells of all ranks; updates the ghost entries of `y` first."
function total_energy(m::NonlinearShallowWater, y)
    exchange_state!(m, y)
    w, bu, bh = m.work, m.basis_u, m.basis_h
    launch!(cell_energy_kernel!, w.cell_values, m.cells.n_owned_cells, w.cell_values, view(y, velocity_dofs(m)),
            view(y, height_dofs(m)), bu.φ, bu.indices, bh.φ, bh.indices, bu.dA, m.gravity)
    return global_sum(m.cells, sum(w.cell_values))
end

"Potential enstrophy `∫ q² h / 2 = ∫ ω² / (2h)`, conserved by the continuous equations."
function potential_enstrophy(m::NonlinearShallowWater, y)
    exchange_state!(m, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    w, bq, bh = compute_diagnostics!(m, u, h), m.basis_q, m.basis_h
    launch!(cell_enstrophy_kernel!, w.cell_values, m.cells.n_owned_cells, w.cell_values, w.ω, h, bq.φ, bq.indices,
            bh.φ, bh.indices, bq.dA)
    return global_sum(m.cells, sum(w.cell_values))
end

# --- shared by the classical models ------------------------------------------

"Global state vector `[u; h]` with the owned values of all ranks (identical on every rank, on the host)."
function gather_state(m::Union{LinearShallowWater, NonlinearShallowWater}, y)
    d, comm = m.distribution, m.cells.comm
    return [gather(d.u, Array(view(y, velocity_dofs(m))), num_dofs(m.U), comm);
            gather(d.h, Array(view(y, height_dofs(m))), num_dofs(m.H), comm)]
end

"Update the ghost entries of the local state `y`."
function exchange_state!(m::Union{LinearShallowWater, NonlinearShallowWater}, y)
    d, comm = m.distribution, m.cells.comm
    exchange!(view(y, velocity_dofs(m)), d.u, comm)
    exchange!(view(y, height_dofs(m)), d.h, comm)
    return y
end

"Total mass `∫ h`, summed over the owned dofs of all ranks."
total_mass(m::Union{LinearShallowWater, NonlinearShallowWater}, y) =
    global_sum(m.cells, dot(m.mass_weights, view(view(y, height_dofs(m)), owned(m.distribution.h))))
