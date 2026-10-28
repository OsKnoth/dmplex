"""
    KiteQuadrature(dm, degree, coriolis; projected=true, cells=1:num_cells(dm))

Metric terms of the kites `cells` (cell indices; column `j` belongs to
`cells[j]`) at the points of one reference quadrature rule,
computed once: the contravariant Piola factor `P = J / detJ`, the area weight
`dA = w detJ`, the unit surface normal and the Coriolis parameter. With the
reference basis tabulated once, a physical H(div) basis function is
`φ = P φ̂`, so a kernel gathers `û = Σ sᵢ uᵢ φ̂ᵢ`, applies the 3×2 factor and
scatters `φ̂ᵢ ⋅ (Pᵀ g)`; no per-cell basis is stored.
"""
struct KiteQuadrature{V, P, M, N}
    points::Vector{SVector{2, Float64}}     # host only, for tabulation
    weights::V
    piola::P      # SMatrix{3, 2} per (point, cell)
    dA::M
    normal::N     # SVector{3} per (point, cell)
    coriolis::M
end

FEM.to_backend(backend, Q::KiteQuadrature) =
    KiteQuadrature(Q.points, (to_backend(backend, x) for x in (Q.weights, Q.piola, Q.dA, Q.normal, Q.coriolis))...)

function KiteQuadrature(dm::Plex, degree::Integer, coriolis; projected::Bool=true,
                        cells::AbstractVector{<:Integer}=1:num_cells(dm))
    rule = quadrature_rule(Quadrilateral(), degree)
    nq, nc = length(rule.points), length(cells)
    piola = Matrix{SMatrix{3, 2, Float64, 6}}(undef, nq, nc)
    dA, f = zeros(nq, nc), zeros(nq, nc)
    normal = Matrix{SVector{3, Float64}}(undef, nq, nc)
    for (j, k) in enumerate(cells)
        cmap = CellMap(dm, first(cell_range(dm)) + k - 1; projected)
        for (q, (ξ, w)) in enumerate(zip(rule.points, rule.weights))
            x, J, detJ = map_point(cmap, ξ)
            piola[q, j] = SMatrix{3, 2}(J) / detJ
            dA[q, j] = w * detJ
            normal[q, j] = surface_normal(J)
            f[q, j] = coriolis(surface_point(dm, x))
        end
    end
    return KiteQuadrature(rule.points, rule.weights, piola, dA, normal, f)
end

"Reference basis values at the quadrature points and the cell dofs of a space, as dense arrays."
function dense_tables(V::FESpace, points)
    values, gradients = tabulate(reference_element(V, first(cell_range(V.dm))), points)
    first_dofs = first(cell_dofs(V, first(cell_range(V.dm))))
    indices = zeros(Int, length(first_dofs), num_cells(V.dm))
    signs = zeros(length(first_dofs), num_cells(V.dm))
    for c in cell_range(V.dm)
        idx, sgn = cell_dofs(V, c)
        indices[:, cell_index(V.dm, c)] .= idx
        signs[:, cell_index(V.dm, c)] .= sgn
    end
    return values, gradients, indices, signs
end

"Preallocated work vectors of one tendency evaluation."
struct KiteWorkspace{V<:AbstractVector{Float64}}
    ζ::V
    rhs_ζ::V
    K::V
    rhs_K::V
    F::V
    rhs_F::V
    p::V
    rhs_u::V
    rhs_h::V
    energy::V    # kinetic energy per owned kite
end

KiteWorkspace(backend, nu, nh, nz, nk) =
    KiteWorkspace((KernelAbstractions.zeros(backend, Float64, n) for n in (nz, nz, nh, nh, nu, nu, nh, nu, nh, nk))...)


"""
    NonlinearKiteShallowWater(primal, k; height_order=k + 1, gravity=1, coriolis=x -> 2x[3],
                              flux=:projected, vorticity=:potential, degree, geometry=:sphere,
                              centers=nothing, comm=nothing, part=nothing)

Nonlinear rotating shallow-water equations in vector-invariant form with the
staggered kite method (Knoth, EGU 2026): `h ∈ KiteRadau(height_order)` (diagonal
mass), `u ∈ KiteBrokenRT(k)` (dual-cell mass blocks), relative vorticity
`ζ ∈ Lagrange(k + 1)` on the kites (GLL-lumped, diagonal mass) and kinetic
energy `K` in the height space:

    ⟨ζ, φ⟩ = -∫ u ⋅ rot φ,           ⟨K, η⟩ = ∫ |u|²/2 η,
    ∫ u̇ ⋅ v = -∫ (ζ + f) (r̂ × u) ⋅ v + G(g h + K; v),
    ⟨ḣ, η⟩ = Σ_K ∫_K h u ⋅ ∇η - Σ_K Σ_{F ⊂ ∂K ∩ primal edges} ∫_F η h_K (u ⋅ n_K),

with the kite gradient `G = -Bᵀ`. This is the poster form, selected with
`flux=:pointwise, vorticity=:absolute`. The defaults are the conservative
variants:

- `flux=:projected`: the mass flux is `F = Π_U(h u)` (one solve with the block
  diagonal velocity mass), and `ḣ = M_h⁻¹ B F`; `F ⋅ n` is single valued on
  primal edges, so mass is conserved exactly.
- `vorticity=:potential`: the vorticity term is `(ζ + f)/h (r̂ × F)` with the
  projected flux; together with `flux=:projected` the semi-discretization
  conserves the energy `∫ h|u|²/2 + g hᵀ M_h h / 2` exactly.

All nonlinear terms use one quadrature rule of `degree` (default `2(k + 1)`,
3 × 3 points for RT1) with precomputed metric terms (`KiteQuadrature`) and are
evaluated by KernelAbstractions kernels over the kites with atomic scatter.
`geometry` is `:sphere` (curved elements, default) or `:chordal`.

`velocity` selects the velocity space: `:rt` (`KiteBrokenRT(k)`, default),
`:bdm` (`KiteBrokenBDM(k)`) or `:dual_radau` (`KiteDualRadau(k)`, the pair of
Wess et al. with lumped velocity mass).

With an MPI communicator `comm` the model is distributed over its ranks
(`KiteDistribution`, equal-area partition of the primal cells unless `part`
is given). All run-time data is in local numbering (owned dofs first, then
ghosts); `tendency!` updates the ghosts of the state, the vorticity and the
mass flux by halo exchange and computes the owned dofs. The state vector is
the local `[u; h]`; `gather_state` assembles the global one. `Mu` and `Mh` are
the global mass matrices (for diagnostics on gathered states).

`backend` is the KernelAbstractions backend of all run-time arrays
(`CPU()` by default, `CUDABackend()` after `using CUDA`): the state, the work
vectors, the metric terms, the dof tables, the block-diagonal mass solvers
(`BlockSolver`) and the halo buffers live there, so a time step runs entirely
on the device. Halo exchange passes device buffers to MPI if it is CUDA-aware.
"""
struct NonlinearKiteShallowWater{Q, A3, A4, IM, FM, V, S, RM, T, W, D, BE}
    primal::Plex
    kites::Plex
    U::FESpace
    H::FESpace
    Z::FESpace
    quadrature::Q           # metric terms of the local kites
    φu::A3                  # (dof, component, point)
    φh::A3
    ∇φh::A4                 # reference gradients (dof, 1, direction, point)
    φz::A3
    iu::IM                  # (dof, local kite), local dof numbers
    su::FM
    ih::IM
    iz::IM
    Mu::SparseMatrixCSC{Float64, Int}     # global mass matrices
    Mh::SparseMatrixCSC{Float64, Int}
    mass_h::V                             # local diagonal of the height mass
    Mu_solver::S                          # owned velocity blocks (local numbering)
    Mh_solver::S                          # owned height dofs
    Mh_local_solver::S                    # all local height dofs (kinetic energy)
    Mz_solver::S                          # owned vorticity dofs
    B::SparseMatrixCSC{Float64, Int}      # global operators
    R::SparseMatrixCSC{Float64, Int}      # ∫ φ_j ⋅ (r̂ × ∇γ_i): ⟨ζ, γ⟩ = -R u
    R_owned::RM                           # owned rows, transposed, local columns
    Bt_owned::RM
    B_owned::RM
    faces::T                              # primal-face tables for the pointwise flux
    work::W
    row_buffers::NTuple{3, V}
    distribution::D                       # halo buffers on the backend
    velocity_range::UnitRange{Int}
    height_range::UnitRange{Int}
    gravity::Float64
    flux::Symbol
    vorticity::Symbol
    backend::BE
end

function NonlinearKiteShallowWater(primal::Plex, k::Integer; height_order::Integer=k + 1, gravity::Real=1.0,
                                   coriolis=x -> 2x[3], flux::Symbol=:projected, vorticity::Symbol=:potential,
                                   degree::Integer=2 * (k + 1), geometry::Symbol=:sphere, centers=nothing,
                                   comm=nothing, part=nothing, velocity::Symbol=:rt,
                                   backend=KernelAbstractions.CPU())
    flux in (:pointwise, :projected) || throw(ArgumentError("flux must be :pointwise or :projected"))
    vorticity in (:absolute, :potential) || throw(ArgumentError("vorticity must be :absolute or :potential"))
    vorticity == :potential && flux != :projected &&
        throw(ArgumentError("vorticity=:potential needs flux=:projected"))
    kites = create_kite_mesh(primal; centers)
    H = FESpace(kites, KiteRadau(height_order); name="h", geometry)
    U = FESpace(kites, kite_velocity_family(velocity, k); name="u", geometry)
    Z = FESpace(kites, Lagrange(k + 1); name="zeta", geometry)
    Sh, Sz = mass_solver(H; lumped=height_order > 0), mass_solver(Z; lumped=true)
    Su = mass_solver(U; lumped=velocity == :dual_radau)
    operator_degree = 2 * max(k, height_order) + 2
    B = assemble_kite_divergence(H, U; degree=operator_degree)
    R = assemble_bilinear(Z, U; degree=operator_degree) do t, u, a, b, q
        dot(u.φ[b, q], cross(t.normals[q], t.grad[a, q])) * t.dA[q]
    end
    d = comm === nothing ? KiteDistribution(primal, kites, U, H, Z) :
        part === nothing ? KiteDistribution(primal, kites, U, H, Z, comm) :
                           KiteDistribution(primal, kites, U, H, Z, comm; part)

    # local numbering of kites and dofs
    Q = KiteQuadrature(kites, degree, coriolis; projected=geometry == :sphere, cells=d.kite_ids)
    local_index(dd) = (map = zeros(Int, maximum(dd.global_ids; init=0)); map[dd.global_ids] .= 1:num_local(dd); map)
    tables(V, dd) = begin
        φ, ∇φ, idx, sgn = dense_tables(V, Q.points)
        map = local_index(dd)
        φ, ∇φ, [map[i] for i in idx[:, d.kite_ids]], sgn[:, d.kite_ids]
    end
    φu, _, iu, su = tables(U, d.u)
    φh, ∇φh, ih, _ = tables(H, d.h)
    φz, _, iz, _ = tables(Z, d.z)
    # owned rows, transposed
    rows(A, dr, dc) = RowMatrix(sparse(A[dr.global_ids[owned(dr)], dc.global_ids]'), backend)
    solver(S, dd; ghosts=false) = block_solver(local_solver(S, dd; ghosts), backend)
    dev(x) = to_backend(backend, x)
    faces = map(primal_face_tables(H, U, degree)) do f
        (weights=dev(f.weights), η=dev(f.η), φ=dev(f.φ), ν=f.ν)
    end
    nu, nh, nz = num_local(d.u), num_local(d.h), num_local(d.z)
    return NonlinearKiteShallowWater(primal, kites, U, H, Z, dev(Q), dev(φu), dev(φh), dev(∇φh), dev(φz),
                                     dev(iu), dev(su), dev(ih), dev(iz),
                                     Su.matrix, Sh.matrix, dev([Sh.matrix[i, i] for i in d.h.global_ids]),
                                     solver(Su, d.u), solver(Sh, d.h), solver(Sh, d.h; ghosts=true), solver(Sz, d.z),
                                     B, R, rows(R, d.z, d.u), rows(sparse(B'), d.u, d.h), rows(B, d.h, d.u),
                                     faces, KiteWorkspace(backend, nu, nh, nz, d.n_owned_kites),
                                     Tuple(KernelAbstractions.zeros(backend, Float64, n)
                                           for n in (d.z.n_owned, d.u.n_owned, d.h.n_owned)),
                                     to_backend(backend, d), 1:nu, nu .+ (1:nh), Float64(gravity), flux, vorticity,
                                     backend)
end

"Reference tables of the height and velocity elements on the primal faces of a kite."
function primal_face_tables(H::FESpace, U::FESpace, degree::Integer)
    c = first(cell_range(H.dm))
    elh, elu = reference_element(H, c), reference_element(U, c)
    t, wt = gauss_legendre(cld(degree + 1, 2))
    return map(FEM.KITE_PRIMAL_FACES) do e
        points = [edge_point(elh.shape, e, ti) for ti in t]
        (weights=wt, η=first(tabulate(elh, points)), φ=first(tabulate(elu, points)), ν=edge_normal(elh.shape, e))
    end
end

"Local state ranges of velocity and height."
velocity_dofs(m::NonlinearKiteShallowWater) = m.velocity_range
height_dofs(m::NonlinearKiteShallowWater) = m.height_range
"Global state ranges (of `gather_state`)."
global_velocity_dofs(m::NonlinearKiteShallowWater) = 1:num_dofs(m.U)
global_height_dofs(m::NonlinearKiteShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))

"Local state `[u; h]` from the L2 projections of velocity and height fields, on the model's backend."
function initial_state(m::NonlinearKiteShallowWater, velocity, height)
    d = m.distribution
    return to_backend(m.backend, [project(m.U, velocity)[d.u.global_ids]; project(m.H, height)[d.h.global_ids]])
end

"Global state vector `[u; h]` with the owned values of all ranks (identical on every rank, on the host)."
function gather_state(m::NonlinearKiteShallowWater, y)
    d = m.distribution
    return [gather(d.u, Array(view(y, velocity_dofs(m))), num_dofs(m.U), d.comm);
            gather(d.h, Array(view(y, height_dofs(m))), num_dofs(m.H), d.comm)]
end

# --- kernels -------------------------------------------------------------------

@inline function gather_scalar(x, φ, idx, c, q)
    s = 0.0
    @inbounds for a in axes(idx, 1)
        s += x[idx[a, c]] * φ[a, 1, q]
    end
    return s
end

"Reference vector `Σ sᵢ xᵢ φ̂ᵢ` at point `q` of cell `c`."
@inline function gather_vector(x, φ, idx, sgn, c, q)
    s1 = s2 = 0.0
    @inbounds for b in axes(idx, 1)
        coefficient = sgn[b, c] * x[idx[b, c]]
        s1 += coefficient * φ[b, 1, q]
        s2 += coefficient * φ[b, 2, q]
    end
    return SVector(s1, s2)
end

"Atomically add `sᵢ φ̂ᵢ ⋅ g` to the entries of cell `c`."
@inline function scatter_vector!(rhs, g, φ, idx, sgn, c, q)
    @inbounds for b in axes(idx, 1)
        @atomic rhs[idx[b, c]] += sgn[b, c] * (φ[b, 1, q] * g[1] + φ[b, 2, q] * g[2])
    end
    return rhs
end

@kernel function kinetic_and_flux_kernel!(rhs_K, rhs_F, @Const(u), @Const(h), @Const(piola), @Const(dA),
                                          @Const(φu), @Const(iu), @Const(su), @Const(φh), @Const(ih), flux)
    c = @index(Global)
    @inbounds for q in axes(dA, 1)
        P, w = piola[q, c], dA[q, c]
        uq = P * gather_vector(u, φu, iu, su, c, q)
        kinetic = dot(uq, uq) / 2 * w
        for a in axes(ih, 1)
            @atomic rhs_K[ih[a, c]] += φh[a, 1, q] * kinetic
        end
        if flux
            hq = gather_scalar(h, φh, ih, c, q)
            scatter_vector!(rhs_F, P' * (hq * w * uq), φu, iu, su, c, q)
        end
    end
end

@kernel function vorticity_kernel!(rhs_u, @Const(transported), @Const(h), @Const(ζ), @Const(piola), @Const(dA),
                                   @Const(normal), @Const(coriolis), @Const(φu), @Const(iu), @Const(su),
                                   @Const(φh), @Const(ih), @Const(φz), @Const(iz), potential)
    c = @index(Global)
    @inbounds for q in axes(dA, 1)
        P = piola[q, c]
        v = P * gather_vector(transported, φu, iu, su, c, q)
        absolute = gather_scalar(ζ, φz, iz, c, q) + coriolis[q, c]
        potential && (absolute /= gather_scalar(h, φh, ih, c, q))
        force = (absolute * dA[q, c]) * cross(normal[q, c], v)
        scatter_vector!(rhs_u, -(P' * force), φu, iu, su, c, q)
    end
end

@kernel function pointwise_flux_kernel!(rhs, @Const(u), @Const(h), @Const(weights), @Const(φu), @Const(iu),
                                        @Const(su), @Const(φh), @Const(∇φh), @Const(ih), faces)
    c = @index(Global)
    @inbounds for (q, w) in enumerate(weights)
        û = gather_vector(u, φu, iu, su, c, q)
        hq = gather_scalar(h, φh, ih, c, q)
        for a in axes(ih, 1)  # metric free: h û ⋅ ∇̂η̂ dξ
            @atomic rhs[ih[a, c]] += w * hq * (û[1] * ∇φh[a, 1, 1, q] + û[2] * ∇φh[a, 1, 2, q])
        end
    end
    @inbounds for face in faces, (q, w) in enumerate(face.weights)
        ûn = dot(gather_vector(u, face.φ, iu, su, c, q), face.ν)
        hq = gather_scalar(h, face.η, ih, c, q)
        for a in axes(ih, 1)
            @atomic rhs[ih[a, c]] -= w * face.η[a, 1, q] * hq * ûn
        end
    end
end

"Run a kernel over the `n` local kites on the backend of `array`."
function launch!(kernel, array, n, args...)
    n == 0 && return nothing
    backend = get_backend(array)
    kernel(backend)(args...; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function kinetic_energy_kernel!(energy, @Const(u), @Const(h), @Const(piola), @Const(dA), @Const(φu),
                                        @Const(iu), @Const(su), @Const(φh), @Const(ih))
    c = @index(Global)
    e = 0.0
    @inbounds for q in axes(dA, 1)
        uq = piola[q, c] * gather_vector(u, φu, iu, su, c, q)
        e += gather_scalar(h, φh, ih, c, q) * dot(uq, uq) / 2 * dA[q, c]
    end
    @inbounds energy[c] = e
end

@kernel function center_value_kernel!(values, @Const(x), @Const(φ), @Const(idx))
    c = @index(Global)
    @inbounds values[c] = gather_scalar(x, φ, idx, c, 1)
end

# --- diagnostics and tendency ----------------------------------------------------

"""
    diagnostics(m, u, h) -> (ζ, K, F)

Relative vorticity, kinetic energy and, for `flux=:projected`, the projected
mass flux `Π_U(h u)` (otherwise `nothing`), in local numbering. Returns copies.
"""
function diagnostics(m::NonlinearKiteShallowWater, u::AbstractVector, h::AbstractVector)
    compute_diagnostics!(m, u, h)
    w = m.work
    return copy(w.ζ), copy(w.K), m.flux == :projected ? copy(w.F) : nothing
end

function compute_diagnostics!(m::NonlinearKiteShallowWater, u, h)
    w, d, Q = m.work, m.distribution, m.quadrature
    nk = num_local_kites(d)
    tz = m.row_buffers[1]
    row_mul!(tz, m.R_owned, u, -1.0)
    copyto!(w.rhs_ζ, 1, tz, 1, length(tz))      # owned dofs come first
    ldiv!(w.ζ, m.Mz_solver, w.rhs_ζ)
    exchange!(w.ζ, d.z, d.comm)

    projected = m.flux == :projected
    fill!(w.rhs_K, 0.0)
    projected && fill!(w.rhs_F, 0.0)
    launch!(kinetic_and_flux_kernel!, w.rhs_K, nk, w.rhs_K, w.rhs_F, u, h, Q.piola, Q.dA, m.φu, m.iu, m.su,
            m.φh, m.ih, projected)
    # every kite of a ghost primal cell is local: K is complete on all local dofs, no exchange
    ldiv!(w.K, m.Mh_local_solver, w.rhs_K)
    if projected
        ldiv!(w.F, m.Mu_solver, w.rhs_F)
        exchange!(w.F, d.u, d.comm)
    end
    return w
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::NonlinearKiteShallowWater)
    w, d, Q = m.work, m.distribution, m.quadrature
    nk = num_local_kites(d)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    exchange!(u, d.u, d.comm)
    exchange!(h, d.h, d.comm)
    compute_diagnostics!(m, u, h)
    @. w.p = m.gravity * h + w.K

    tu, th = m.row_buffers[2], m.row_buffers[3]
    fill!(w.rhs_u, 0.0)
    row_mul!(tu, m.Bt_owned, w.p, -1.0)
    copyto!(w.rhs_u, 1, tu, 1, length(tu))
    launch!(vorticity_kernel!, w.rhs_u, nk, w.rhs_u, m.vorticity == :absolute ? u : w.F, h, w.ζ, Q.piola, Q.dA,
            Q.normal, Q.coriolis, m.φu, m.iu, m.su, m.φh, m.ih, m.φz, m.iz, m.vorticity == :potential)
    if m.flux == :projected
        row_mul!(th, m.B_owned, w.F)
        copyto!(w.rhs_h, 1, th, 1, length(th))
    else
        fill!(w.rhs_h, 0.0)
        launch!(pointwise_flux_kernel!, w.rhs_h, nk, w.rhs_h, u, h, Q.weights, m.φu, m.iu, m.su, m.φh, m.∇φh,
                m.ih, m.faces)
    end
    is_distributed(d) && fill!(dy, 0.0)
    ldiv!(view(dy, velocity_dofs(m)), m.Mu_solver, w.rhs_u)
    ldiv!(view(dy, height_dofs(m)), m.Mh_solver, w.rhs_h)
    return dy
end

"Total mass `Σ M_h h` over the owned height dofs of all ranks."
function total_mass(m::NonlinearKiteShallowWater, y)
    o = owned(m.distribution.h)
    return global_sum(m.distribution, dot(view(m.mass_h, o), view(view(y, height_dofs(m)), o)))
end

"""
Energy `∫ h|u|²/2 dA + g hᵀ M_h h / 2` (the potential part with the model's
height mass), summed over the ranks. Updates the ghost entries of `y` first.
"""
function total_energy(m::NonlinearKiteShallowWater, y)
    d = m.distribution
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    exchange!(u, d.u, d.comm)
    exchange!(h, d.h, d.comm)
    Q, w, o = m.quadrature, m.work, owned(d.h)
    launch!(kinetic_energy_kernel!, w.energy, d.n_owned_kites, w.energy, u, h, Q.piola, Q.dA, m.φu, m.iu, m.su,
            m.φh, m.ih)
    ho = view(h, o)
    potential = dot(view(m.mass_h, o) .* ho, ho)
    return global_sum(d, sum(w.energy) + m.gravity * potential / 2)
end

"""
    center_values(m, y; field=:vorticity) -> values per owned kite

Values of the relative vorticity (`field=:vorticity`, diagnosed from `y`) or
the height (`field=:height`) at the centers of the owned local kites, as a
vector on the model's backend (for output with `write_vtx!` without a copy
to the host). Updates the ghost entries of `y`.
"""
function center_values(m::NonlinearKiteShallowWater, y; field::Symbol=:vorticity)
    d = m.distribution
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    exchange!(u, d.u, d.comm)
    exchange!(h, d.h, d.comm)
    V, x, idx = field == :vorticity ? (m.Z, (compute_diagnostics!(m, u, h); m.work.ζ), m.iz) :
                field == :height ? (m.H, h, m.ih) : throw(ArgumentError("field must be :vorticity or :height"))
    φ = to_backend(m.backend, first(tabulate(reference_element(V, first(cell_range(V.dm))), [SVector(0.0, 0.0)])))
    values = KernelAbstractions.zeros(m.backend, Float64, d.n_owned_kites)
    launch!(center_value_kernel!, values, d.n_owned_kites, values, x, φ, idx)
    return values
end
