"""
    LinearKiteShallowWater(primal, k; height_order=k, gravity=1, depth=1, coriolis=x -> 2x[3], centers=nothing)

Linearized rotating shallow-water equations with the staggered kite method
(Knoth, EGU 2026) on the kite grid of any polygonal `primal` mesh:

- height `h ∈ KiteRadau(height_order)`: broken H(grad) on primal cells, with
  a diagonal mass matrix (Gauss–Radau lumping for order ≥ 1; for order 0 the
  exact mass, the primal cell area, is diagonal already),
- velocity `u ∈ KiteBrokenRT(k)`: broken H(div) on dual cells, with one
  mass block per dual cell.

    M_h ḣ = H B u,    M_u u̇ = -C u - g Bᵀ h,

with the kite divergence `B` (`assemble_kite_divergence`), whose adjoint is
the kite gradient, and the Coriolis matrix `C`. Mass and energy
`½ H uᵀ M_u u + ½ g hᵀ M_h h` are conserved exactly by the semi-discretization.
The state vector is `[u; h]`. `centers` are passed to `create_kite_mesh`.

Use `k ≥ 1`. At lowest order the pairing is inconsistent: with
`KiteRadau(0) × KiteBrokenRT(0)` the discrete gradient does not converge (a
height constant per primal cell does not control the cell-local velocity
dofs), and with `KiteRadau(1) × KiteBrokenRT(0)` the discrete divergence does
not. `KiteRadau(1) × KiteBrokenRT(1)` (the configuration of the EGU poster)
converges at second order.
"""
struct LinearKiteShallowWater{SU, SH}
    primal::Plex
    kites::Plex
    U::FESpace
    H::FESpace
    Mu::SparseMatrixCSC{Float64, Int}
    Mh::SparseMatrixCSC{Float64, Int}
    Mu_solver::SU
    Mh_solver::SH
    B::SparseMatrixCSC{Float64, Int}
    C::SparseMatrixCSC{Float64, Int}
    gravity::Float64
    depth::Float64
end

function LinearKiteShallowWater(primal::Plex, k::Integer; height_order::Integer=k, gravity::Real=1.0,
                                depth::Real=1.0, coriolis=x -> 2x[3], centers=nothing)
    kites = create_kite_mesh(primal; centers)
    H = FESpace(kites, KiteRadau(height_order); name="h")
    U = FESpace(kites, KiteBrokenRT(k); name="u")
    Sh, Su = mass_solver(H; lumped=height_order > 0), mass_solver(U)
    B = assemble_kite_divergence(H, U; degree=2 * max(k, height_order) + 2)
    return LinearKiteShallowWater(primal, kites, U, H, Su.matrix, Sh.matrix, Su, Sh, B,
                                  assemble_coriolis(U, coriolis), Float64(gravity), Float64(depth))
end

velocity_dofs(m::LinearKiteShallowWater) = 1:num_dofs(m.U)
height_dofs(m::LinearKiteShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))
initial_state(m::LinearKiteShallowWater, velocity, height) = [project(m.U, velocity); project(m.H, height)]

function tendency!(dy::AbstractVector, y::AbstractVector, m::LinearKiteShallowWater)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    dy[velocity_dofs(m)] = m.Mu_solver \ (-(m.C * u) - m.gravity * (m.B' * h))
    dy[height_dofs(m)] = m.Mh_solver \ (m.depth * (m.B * u))
    return dy
end

total_mass(m::LinearKiteShallowWater, y) = sum(m.Mh * view(y, height_dofs(m)))

function total_energy(m::LinearKiteShallowWater, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    return (m.depth * dot(u, m.Mu * u) + m.gravity * dot(h, m.Mh * h)) / 2
end

"""
    NonlinearKiteShallowWater(primal, k; height_order=k + 1, gravity=1, coriolis=x -> 2x[3],
                              flux=:projected, vorticity=:potential, centers=nothing)

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
grow on refined hexagonal kites. The state vector is `[u; h]`.
"""
struct NonlinearKiteShallowWater{BU, BH, BZ, SU, SH, SZ, T}
    primal::Plex
    kites::Plex
    U::FESpace
    H::FESpace
    Z::FESpace
    basis_u::Vector{BU}
    basis_h::Vector{BH}
    basis_z::Vector{BZ}
    coriolis_at::Vector{Vector{Float64}}  # f at the quadrature points of every kite
    Mu::SparseMatrixCSC{Float64, Int}
    Mh::SparseMatrixCSC{Float64, Int}
    Mu_solver::SU
    Mh_solver::SH
    Mz_solver::SZ
    B::SparseMatrixCSC{Float64, Int}
    R::SparseMatrixCSC{Float64, Int}      # ∫ φ_j ⋅ (r̂ × ∇γ_i): ⟨ζ, γ⟩ = -R u
    tables::T                             # reference tables for the pointwise flux
    gravity::Float64
    flux::Symbol
    vorticity::Symbol
end

function NonlinearKiteShallowWater(primal::Plex, k::Integer; height_order::Integer=k + 1, gravity::Real=1.0,
                                   coriolis=x -> 2x[3], flux::Symbol=:projected, vorticity::Symbol=:potential,
                                   centers=nothing, degree::Integer=3 * max(k, height_order) + 3)
    flux in (:pointwise, :projected) || throw(ArgumentError("flux must be :pointwise or :projected"))
    vorticity in (:absolute, :potential) || throw(ArgumentError("vorticity must be :absolute or :potential"))
    vorticity == :potential && flux != :projected &&
        throw(ArgumentError("vorticity=:potential needs flux=:projected"))
    kites = create_kite_mesh(primal; centers)
    H = FESpace(kites, KiteRadau(height_order); name="h")
    U = FESpace(kites, KiteBrokenRT(k); name="u")
    Z = FESpace(kites, Lagrange(k + 1); name="zeta")
    Sh, Su, Sz = mass_solver(H; lumped=height_order > 0), mass_solver(U), mass_solver(Z; lumped=true)
    B = assemble_kite_divergence(H, U; degree)
    R = assemble_bilinear(Z, U; degree) do t, u, a, b, q
        dot(u.φ[b, q], cross(normalize(t.points[q]), t.grad[a, q])) * t.dA[q]
    end
    basis_u, basis_h = precompute_basis(U, degree), precompute_basis(H, degree)
    return NonlinearKiteShallowWater(primal, kites, U, H, Z, basis_u, basis_h, precompute_basis(Z, degree),
                                     [coriolis.(b.points) for b in basis_u], Su.matrix, Sh.matrix,
                                     Su, Sh, Sz, B, R, kite_flux_tables(H, U, degree), Float64(gravity),
                                     flux, vorticity)
end

"Reference tables of the height and velocity elements on the kite and on its primal faces."
function kite_flux_tables(H::FESpace, U::FESpace, degree::Integer)
    c = first(cell_range(H.dm))
    elh, elu = reference_element(H, c), reference_element(U, c)
    rule = quadrature_rule(elh.shape, degree)
    η, ∇η = tabulate(elh, rule.points)
    φ, _ = tabulate(elu, rule.points)
    t, wt = gauss_legendre(cld(degree + 1, 2))
    faces = map(FEM.KITE_PRIMAL_FACES) do e
        points = [edge_point(elh.shape, e, ti) for ti in t]
        (weights=wt, η=first(tabulate(elh, points)), φ=first(tabulate(elu, points)), ν=edge_normal(elh.shape, e))
    end
    dofs = [(cell_dofs(H, c)[1], cell_dofs(U, c)...) for c in cell_range(H.dm)]
    return (weights=rule.weights, η=η, ∇η=∇η, φ=φ, faces=faces, dofs=dofs)
end

velocity_dofs(m::NonlinearKiteShallowWater) = 1:num_dofs(m.U)
height_dofs(m::NonlinearKiteShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))
initial_state(m::NonlinearKiteShallowWater, velocity, height) = [project(m.U, velocity); project(m.H, height)]

"Reference velocity `û = Σ sᵢ uᵢ φ̂ᵢ` at column `q` of a reference table."
reference_velocity(φ, u, iu, su, q) =
    SVector(sum(su[b] * u[iu[b]] * φ[b, 1, q] for b in eachindex(iu)), sum(su[b] * u[iu[b]] * φ[b, 2, q] for b in eachindex(iu)))

"Right-hand side of the poster's height equation with the pointwise flux `h_K u`."
function pointwise_flux_divergence(m::NonlinearKiteShallowWater, u, h)
    T = m.tables
    rhs = zeros(num_dofs(m.H))
    for (ih, iu, su) in T.dofs
        for (q, w) in enumerate(T.weights)
            û = reference_velocity(T.φ, u, iu, su, q)
            hq = sum(h[ih[a]] * T.η[a, 1, q] for a in eachindex(ih))
            for a in eachindex(ih)
                rhs[ih[a]] += w * hq * (û[1] * T.∇η[a, 1, 1, q] + û[2] * T.∇η[a, 1, 2, q])
            end
        end
        for face in T.faces, (q, w) in enumerate(face.weights)
            ûn = dot(reference_velocity(face.φ, u, iu, su, q), face.ν)
            hq = sum(h[ih[a]] * face.η[a, 1, q] for a in eachindex(ih))
            for a in eachindex(ih)
                rhs[ih[a]] -= w * face.η[a, 1, q] * hq * ûn
            end
        end
    end
    return rhs
end

"""
    diagnostics(m, u, h) -> (ζ, K, F)

Relative vorticity, kinetic energy and, for `flux=:projected`, the projected
mass flux `Π_U(h u)` (otherwise `nothing`).
"""
function diagnostics(m::NonlinearKiteShallowWater, u::AbstractVector, h::AbstractVector)
    rhs_K = zeros(num_dofs(m.H))
    rhs_F = m.flux == :projected ? zeros(num_dofs(m.U)) : nothing
    for (bu, bh) in zip(m.basis_u, m.basis_h), q in eachindex(bu.dA)
        uq = field_at(bu, u, q)
        for a in eachindex(bh.indices)
            rhs_K[bh.indices[a]] += bh.φ[a, q] * dot(uq, uq) / 2 * bu.dA[q]
        end
        if rhs_F !== nothing
            hq = field_at(bh, h, q)
            for b in eachindex(bu.indices)
                rhs_F[bu.indices[b]] += dot(bu.φ[b, q], hq * uq) * bu.dA[q]
            end
        end
    end
    ζ = m.Mz_solver \ (-(m.R * u))
    return ζ, m.Mh_solver \ rhs_K, rhs_F === nothing ? nothing : m.Mu_solver \ rhs_F
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::NonlinearKiteShallowWater)
    u, h = y[velocity_dofs(m)], y[height_dofs(m)]
    ζ, K, F = diagnostics(m, u, h)
    rhs_u = -(m.B' * (m.gravity * h + K))
    for (bu, bh, bz, f) in zip(m.basis_u, m.basis_h, m.basis_z, m.coriolis_at)
        for q in eachindex(bu.dA)
            absolute = field_at(bz, ζ, q) + f[q]
            transported = m.vorticity == :absolute ? field_at(bu, u, q) : field_at(bu, F, q)
            m.vorticity == :potential && (absolute /= field_at(bh, h, q))
            force = absolute * cross(normalize(bu.points[q]), transported) * bu.dA[q]
            for b in eachindex(bu.indices)
                rhs_u[bu.indices[b]] -= dot(bu.φ[b, q], force)
            end
        end
    end
    rhs_h = m.flux == :projected ? m.B * F : pointwise_flux_divergence(m, u, h)
    dy[velocity_dofs(m)] = m.Mu_solver \ rhs_u
    dy[height_dofs(m)] = m.Mh_solver \ rhs_h
    return dy
end

total_mass(m::NonlinearKiteShallowWater, y) = sum(m.Mh * view(y, height_dofs(m)))

"Energy `∫ h|u|²/2 dA + g hᵀ M_h h / 2` (the potential part with the model's height mass)."
function total_energy(m::NonlinearKiteShallowWater, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    kinetic = sum(zip(m.basis_u, m.basis_h)) do (bu, bh)
        sum(field_at(bh, h, q) * dot(field_at(bu, u, q), field_at(bu, u, q)) / 2 * bu.dA[q] for q in eachindex(bu.dA))
    end
    return kinetic + m.gravity * dot(h, m.Mh * h) / 2
end
