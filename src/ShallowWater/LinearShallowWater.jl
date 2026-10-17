"""
    LinearShallowWater(dm, k; gravity=1, depth=1, coriolis=x -> 2x[3])

Linearized rotating shallow-water equations on the sphere,

    ∂u/∂t + f r̂ × u + g ∇h = 0,    ∂h/∂t + H ∇⋅u = 0,

discretized with velocity in `RaviartThomas(k)` and height in
`Discontinuous(k)`:

    M_u u̇ = -C u + g Dᵀ h,    M_h ḣ = -H D u,

with `D[i, j] = ∫ ψ_i ∇⋅φ_j` and the antisymmetric Coriolis matrix `C`. The
semi-discrete system conserves mass `∫ h` and energy
`½ H uᵀ M_u u + ½ g hᵀ M_h h` exactly. The state vector is `[u; h]`.
"""
struct LinearShallowWater{L1, L2}
    U::FESpace
    H::FESpace
    Mu::SparseMatrixCSC{Float64, Int}
    Mh::SparseMatrixCSC{Float64, Int}
    D::SparseMatrixCSC{Float64, Int}
    C::SparseMatrixCSC{Float64, Int}
    Mu_factor::L1
    Mh_factor::L2
    gravity::Float64
    depth::Float64
end

function LinearShallowWater(dm::Plex, k::Integer; gravity::Real=1.0, depth::Real=1.0, coriolis=x -> 2x[3])
    U = FESpace(dm, RaviartThomas(k); name="u")
    H = FESpace(dm, Discontinuous(k); name="h")
    Mu, Mh = assemble_mass(U), assemble_mass(H)
    D = assemble_divergence(H, U)
    C = assemble_coriolis(U, coriolis)
    return LinearShallowWater(U, H, Mu, Mh, D, C, cholesky(Symmetric(Mu)), cholesky(Symmetric(Mh)),
                              Float64(gravity), Float64(depth))
end

velocity_dofs(m::LinearShallowWater) = 1:num_dofs(m.U)
height_dofs(m::LinearShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))

"State vector `[u; h]` from the L2 projections of velocity and height fields."
initial_state(m::LinearShallowWater, velocity, height) = [project(m.U, velocity); project(m.H, height)]

function tendency!(dy::AbstractVector, y::AbstractVector, m::LinearShallowWater)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    dy[velocity_dofs(m)] = m.Mu_factor \ (m.gravity * (m.D' * h) - m.C * u)
    dy[height_dofs(m)] = m.Mh_factor \ (-m.depth * (m.D * u))
    return dy
end

total_mass(m::LinearShallowWater, y) = sum(m.Mh * view(y, height_dofs(m)))

function total_energy(m::LinearShallowWater, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    return (m.depth * dot(u, m.Mu * u) + m.gravity * dot(h, m.Mh * h)) / 2
end
