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
`velocity` selects the velocity space: `:rt` (`KiteBrokenRT(k)`, default),
`:bdm` (`KiteBrokenBDM(k)`) or `:dual_radau` (`KiteDualRadau(k)`, the pair of
Wess et al. with lumped velocity mass).

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
                                depth::Real=1.0, coriolis=x -> 2x[3], centers=nothing, velocity::Symbol=:rt)
    kites = create_kite_mesh(primal; centers)
    H = FESpace(kites, KiteRadau(height_order); name="h")
    U = FESpace(kites, kite_velocity_family(velocity, k); name="u")
    Sh, Su = mass_solver(H; lumped=height_order > 0), mass_solver(U; lumped=velocity == :dual_radau)
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

"Velocity family of the kite models: broken RT, broken BDM or the dual-Radau space of Wess et al."
function kite_velocity_family(velocity::Symbol, k::Integer)
    velocity == :rt && return KiteBrokenRT(k)
    velocity == :bdm && return KiteBrokenBDM(k)
    velocity == :dual_radau && return KiteDualRadau(k)
    throw(ArgumentError("velocity must be :rt, :bdm or :dual_radau"))
end
