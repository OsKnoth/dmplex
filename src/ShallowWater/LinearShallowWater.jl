"""
    LinearShallowWater(dm, k; gravity=1, depth=1, coriolis=x -> 2x[3])
    LinearShallowWater(U, H; gravity=1, depth=1, coriolis=x -> 2x[3])

Linearized rotating shallow-water equations on the sphere,

    ∂u/∂t + f r̂ × u + g ∇h = 0,    ∂h/∂t + H ∇⋅u = 0,

discretized with velocity in `RaviartThomas(k)` and height in
`Discontinuous(k)` (or the given spaces `U`, `H`):

    M_u u̇ = -C u + g Dᵀ h,    M_h ḣ = -H D u,

with `D[i, j] = ∫ ψ_i ∇⋅φ_j` and the antisymmetric Coriolis matrix `C`. The
semi-discrete system conserves mass `∫ h` and energy
`½ H uᵀ M_u u + ½ g hᵀ M_h h` exactly. The state vector is `[u; h]`. The mass
solvers are the cached ones of the spaces.
"""
struct LinearShallowWater{SU, SH}
    U::FESpace
    H::FESpace
    Mu::SparseMatrixCSC{Float64, Int}
    Mh::SparseMatrixCSC{Float64, Int}
    Mu_solver::SU
    Mh_solver::SH
    D::SparseMatrixCSC{Float64, Int}
    C::SparseMatrixCSC{Float64, Int}
    gravity::Float64
    depth::Float64
end

function LinearShallowWater(U::FESpace, H::FESpace; gravity::Real=1.0, depth::Real=1.0, coriolis=x -> 2x[3])
    Su, Sh = mass_solver(U), mass_solver(H)
    return LinearShallowWater(U, H, Su.matrix, Sh.matrix, Su, Sh, assemble_divergence(H, U),
                              assemble_coriolis(U, coriolis), Float64(gravity), Float64(depth))
end

LinearShallowWater(dm::Plex, k::Integer; kwargs...) =
    LinearShallowWater(FESpace(dm, RaviartThomas(k); name="u"), FESpace(dm, Discontinuous(k); name="h"); kwargs...)

velocity_dofs(m::LinearShallowWater) = 1:num_dofs(m.U)
height_dofs(m::LinearShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))

"State vector `[u; h]` from the L2 projections of velocity and height fields."
initial_state(m::LinearShallowWater, velocity, height) = [project(m.U, velocity); project(m.H, height)]

function tendency!(dy::AbstractVector, y::AbstractVector, m::LinearShallowWater)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    dy[velocity_dofs(m)] = m.Mu_solver \ (m.gravity * (m.D' * h) - m.C * u)
    dy[height_dofs(m)] = m.Mh_solver \ (-m.depth * (m.D * u))
    return dy
end

total_mass(m::LinearShallowWater, y) = sum(m.Mh * view(y, height_dofs(m)))

function total_energy(m::LinearShallowWater, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    return (m.depth * dot(u, m.Mu * u) + m.gravity * dot(h, m.Mh * h)) / 2
end
