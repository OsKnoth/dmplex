"""
    KiteQuadrature(dm, degree, coriolis; projected=true)

Metric terms of all kites at the points of one reference quadrature rule,
computed once: the contravariant Piola factor `P = J / detJ`, the area weight
`dA = w detJ`, the unit surface normal and the Coriolis parameter. With the
reference basis tabulated once, a physical H(div) basis function is
`φ = P φ̂`, so a kernel gathers `û = Σ sᵢ uᵢ φ̂ᵢ`, applies the 3×2 factor and
scatters `φ̂ᵢ ⋅ (Pᵀ g)`; no per-cell basis is stored.
"""
struct KiteQuadrature
    points::Vector{SVector{2, Float64}}
    weights::Vector{Float64}
    piola::Matrix{SMatrix{3, 2, Float64, 6}}  # (point, cell)
    dA::Matrix{Float64}
    normal::Matrix{SVector{3, Float64}}
    coriolis::Matrix{Float64}
end

function KiteQuadrature(dm::Plex, degree::Integer, coriolis; projected::Bool=true)
    rule = quadrature_rule(Quadrilateral(), degree)
    nq, nc = length(rule.points), num_cells(dm)
    piola = Matrix{SMatrix{3, 2, Float64, 6}}(undef, nq, nc)
    dA, f = zeros(nq, nc), zeros(nq, nc)
    normal = Matrix{SVector{3, Float64}}(undef, nq, nc)
    for c in cell_range(dm)
        cmap = CellMap(dm, c; projected)
        j = cell_index(dm, c)
        for (q, (ξ, w)) in enumerate(zip(rule.points, rule.weights))
            x, J, detJ = map_point(cmap, ξ)
            piola[q, j] = SMatrix{3, 2}(J) / detJ
            dA[q, j] = w * detJ
            normal[q, j] = surface_normal(J)
            f[q, j] = coriolis(normalize(x))
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
struct KiteWorkspace
    ζ::Vector{Float64}
    rhs_ζ::Vector{Float64}
    K::Vector{Float64}
    rhs_K::Vector{Float64}
    F::Vector{Float64}
    rhs_F::Vector{Float64}
    p::Vector{Float64}
    rhs_u::Vector{Float64}
    rhs_h::Vector{Float64}
end

KiteWorkspace(nu, nh, nz) = KiteWorkspace(zeros(nz), zeros(nz), zeros(nh), zeros(nh), zeros(nu), zeros(nu),
                                          zeros(nh), zeros(nu), zeros(nh))

"""
    NonlinearKiteShallowWater(primal, k; height_order=k + 1, gravity=1, coriolis=x -> 2x[3],
                              flux=:projected, vorticity=:potential, degree, geometry=:sphere,
                              centers=nothing)

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

With `k = 1` (RT1 velocity, CG2 vorticity) the default `height_order = 2`
converges at second order for Williamson test case 2 and conserves mass and
energy exactly; with the pointwise flux, `height_order = 2` was observed to
grow on refined hexagonal kites.

All nonlinear terms use one quadrature rule of `degree` with precomputed
metric terms (`KiteQuadrature`); a tendency evaluation does not allocate. The
default `degree = 2(k + 1)` (3 × 3 Gauss points for RT1) is as accurate as
5 × 5 points for Williamson test case 2. `geometry` is `:sphere` (curved
elements, default) or `:chordal` (flat bilinear elements, about 2.5 times less
accurate at the same cost). The state vector is `[u; h]`.
"""
struct NonlinearKiteShallowWater{SU, SH, SZ, T}
    primal::Plex
    kites::Plex
    U::FESpace
    H::FESpace
    Z::FESpace
    quadrature::KiteQuadrature
    φu::Array{Float64, 3}   # (dof, component, point)
    φh::Array{Float64, 3}
    ∇φh::Array{Float64, 4}  # reference gradients (dof, 1, direction, point)
    φz::Array{Float64, 3}
    iu::Matrix{Int}         # (dof, cell)
    su::Matrix{Float64}
    ih::Matrix{Int}
    iz::Matrix{Int}
    Mu::SparseMatrixCSC{Float64, Int}
    Mh::SparseMatrixCSC{Float64, Int}
    Mu_solver::SU
    Mh_solver::SH
    Mz_solver::SZ
    B::SparseMatrixCSC{Float64, Int}
    R::SparseMatrixCSC{Float64, Int}      # ∫ φ_j ⋅ (r̂ × ∇γ_i): ⟨ζ, γ⟩ = -R u
    faces::T                              # primal-face tables for the pointwise flux
    work::KiteWorkspace
    velocity_range::UnitRange{Int}
    height_range::UnitRange{Int}
    gravity::Float64
    flux::Symbol
    vorticity::Symbol
end

function NonlinearKiteShallowWater(primal::Plex, k::Integer; height_order::Integer=k + 1, gravity::Real=1.0,
                                   coriolis=x -> 2x[3], flux::Symbol=:projected, vorticity::Symbol=:potential,
                                   degree::Integer=2 * (k + 1), geometry::Symbol=:sphere,
                                   centers=nothing)
    flux in (:pointwise, :projected) || throw(ArgumentError("flux must be :pointwise or :projected"))
    vorticity in (:absolute, :potential) || throw(ArgumentError("vorticity must be :absolute or :potential"))
    vorticity == :potential && flux != :projected &&
        throw(ArgumentError("vorticity=:potential needs flux=:projected"))
    kites = create_kite_mesh(primal; centers)
    H = FESpace(kites, KiteRadau(height_order); name="h", geometry)
    U = FESpace(kites, KiteBrokenRT(k); name="u", geometry)
    Z = FESpace(kites, Lagrange(k + 1); name="zeta", geometry)
    Sh, Su, Sz = mass_solver(H; lumped=height_order > 0), mass_solver(U), mass_solver(Z; lumped=true)
    operator_degree = 2 * max(k, height_order) + 2
    B = assemble_kite_divergence(H, U; degree=operator_degree)
    R = assemble_bilinear(Z, U; degree=operator_degree) do t, u, a, b, q
        dot(u.φ[b, q], cross(t.normals[q], t.grad[a, q])) * t.dA[q]
    end
    Q = KiteQuadrature(kites, degree, coriolis; projected=geometry == :sphere)
    φu, _, iu, su = dense_tables(U, Q.points)
    φh, ∇φh, ih, _ = dense_tables(H, Q.points)
    φz, _, iz, _ = dense_tables(Z, Q.points)
    return NonlinearKiteShallowWater(primal, kites, U, H, Z, Q, φu, φh, ∇φh, φz, iu, su, ih, iz,
                                     Su.matrix, Sh.matrix, Su, Sh, Sz, B, R, primal_face_tables(H, U, degree),
                                     KiteWorkspace(num_dofs(U), num_dofs(H), num_dofs(Z)),
                                     1:num_dofs(U), num_dofs(U) .+ (1:num_dofs(H)), Float64(gravity), flux, vorticity)
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

velocity_dofs(m::NonlinearKiteShallowWater) = m.velocity_range
height_dofs(m::NonlinearKiteShallowWater) = m.height_range
initial_state(m::NonlinearKiteShallowWater, velocity, height) = [project(m.U, velocity); project(m.H, height)]

# --- allocation-free kernels ------------------------------------------------------

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

"Add `sᵢ φ̂ᵢ ⋅ g` to the entries of cell `c`."
@inline function scatter_vector!(rhs, g, φ, idx, sgn, c, q)
    @inbounds for b in axes(idx, 1)
        rhs[idx[b, c]] += sgn[b, c] * (φ[b, 1, q] * g[1] + φ[b, 2, q] * g[2])
    end
    return rhs
end

"Right-hand sides of `⟨K, η⟩ = ∫ |u|²/2 η` and, if `flux`, of `⟨F, w⟩ = ∫ h u ⋅ w`."
function kinetic_and_flux_rhs!(rhs_K, rhs_F, u, h, m::NonlinearKiteShallowWater, flux::Bool)
    Q = m.quadrature
    fill!(rhs_K, 0)
    flux && fill!(rhs_F, 0)
    @inbounds for c in axes(Q.dA, 2), q in axes(Q.dA, 1)
        P, dA = Q.piola[q, c], Q.dA[q, c]
        uq = P * gather_vector(u, m.φu, m.iu, m.su, c, q)
        kinetic = dot(uq, uq) / 2 * dA
        for a in axes(m.ih, 1)
            rhs_K[m.ih[a, c]] += m.φh[a, 1, q] * kinetic
        end
        if flux
            hq = gather_scalar(h, m.φh, m.ih, c, q)
            scatter_vector!(rhs_F, P' * (hq * dA * uq), m.φu, m.iu, m.su, c, q)
        end
    end
    return rhs_K
end

"Subtract the vorticity term `∫ q (r̂ × v) ⋅ w` from `rhs_u` (`v = u` or `F`)."
function vorticity_term!(rhs_u, transported, h, ζ, m::NonlinearKiteShallowWater, potential::Bool)
    Q = m.quadrature
    @inbounds for c in axes(Q.dA, 2), q in axes(Q.dA, 1)
        P = Q.piola[q, c]
        v = P * gather_vector(transported, m.φu, m.iu, m.su, c, q)
        absolute = gather_scalar(ζ, m.φz, m.iz, c, q) + Q.coriolis[q, c]
        potential && (absolute /= gather_scalar(h, m.φh, m.ih, c, q))
        force = (absolute * Q.dA[q, c]) * cross(Q.normal[q, c], v)
        scatter_vector!(rhs_u, -(P' * force), m.φu, m.iu, m.su, c, q)
    end
    return rhs_u
end

"Right-hand side of the poster's height equation with the pointwise flux `h_K u`."
function pointwise_flux_rhs!(rhs, u, h, m::NonlinearKiteShallowWater)
    fill!(rhs, 0)
    Q = m.quadrature
    @inbounds for c in axes(Q.dA, 2)
        for (q, w) in enumerate(Q.weights)
            û = gather_vector(u, m.φu, m.iu, m.su, c, q)
            hq = gather_scalar(h, m.φh, m.ih, c, q)
            for a in axes(m.ih, 1)
                # metric free: h û ⋅ ∇̂η̂ dξ
                rhs[m.ih[a, c]] += w * hq * (û[1] * m.∇φh[a, 1, 1, q] + û[2] * m.∇φh[a, 1, 2, q])
            end
        end
        for face in m.faces, (q, w) in enumerate(face.weights)
            ûn = dot(gather_vector(u, face.φ, m.iu, m.su, c, q), face.ν)
            hq = gather_scalar(h, face.η, m.ih, c, q)
            for a in axes(m.ih, 1)
                rhs[m.ih[a, c]] -= w * face.η[a, 1, q] * hq * ûn
            end
        end
    end
    return rhs
end

"""
    diagnostics(m, u, h) -> (ζ, K, F)

Relative vorticity, kinetic energy and, for `flux=:projected`, the projected
mass flux `Π_U(h u)` (otherwise `nothing`). Returns copies of the workspace.
"""
function diagnostics(m::NonlinearKiteShallowWater, u::AbstractVector, h::AbstractVector)
    compute_diagnostics!(m, u, h)
    w = m.work
    return copy(w.ζ), copy(w.K), m.flux == :projected ? copy(w.F) : nothing
end

function compute_diagnostics!(m::NonlinearKiteShallowWater, u, h)
    w = m.work
    mul!(w.rhs_ζ, m.R, u, -1.0, 0.0)
    ldiv!(w.ζ, m.Mz_solver, w.rhs_ζ)
    kinetic_and_flux_rhs!(w.rhs_K, w.rhs_F, u, h, m, m.flux == :projected)
    ldiv!(w.K, m.Mh_solver, w.rhs_K)
    m.flux == :projected && ldiv!(w.F, m.Mu_solver, w.rhs_F)
    return w
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::NonlinearKiteShallowWater)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    w = compute_diagnostics!(m, u, h)
    @. w.p = m.gravity * h + w.K
    mul!(w.rhs_u, transpose(m.B), w.p, -1.0, 0.0)
    vorticity_term!(w.rhs_u, m.vorticity == :absolute ? u : w.F, h, w.ζ, m, m.vorticity == :potential)
    if m.flux == :projected
        mul!(w.rhs_h, m.B, w.F)
    else
        pointwise_flux_rhs!(w.rhs_h, u, h, m)
    end
    ldiv!(view(dy, velocity_dofs(m)), m.Mu_solver, w.rhs_u)
    ldiv!(view(dy, height_dofs(m)), m.Mh_solver, w.rhs_h)
    return dy
end

total_mass(m::NonlinearKiteShallowWater, y) = sum(m.Mh * view(y, height_dofs(m)))

"Energy `∫ h|u|²/2 dA + g hᵀ M_h h / 2` (the potential part with the model's height mass)."
function total_energy(m::NonlinearKiteShallowWater, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    Q = m.quadrature
    kinetic = 0.0
    for c in axes(Q.dA, 2), q in axes(Q.dA, 1)
        uq = Q.piola[q, c] * gather_vector(u, m.φu, m.iu, m.su, c, q)
        kinetic += gather_scalar(h, m.φh, m.ih, c, q) * dot(uq, uq) / 2 * Q.dA[q, c]
    end
    return kinetic + m.gravity * dot(h, m.Mh * h) / 2
end
