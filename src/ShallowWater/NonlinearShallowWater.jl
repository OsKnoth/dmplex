"""
    NonlinearShallowWater(dm, k; gravity=1, coriolis=x -> 2x[3], degree)

Nonlinear rotating shallow-water equations in vector-invariant form,

    ∂u/∂t + q r̂ × (h u) + ∇(K + g h) = 0,    ∂h/∂t + ∇⋅(h u) = 0,

with kinetic energy `K = |u|²/2` and potential vorticity `q = (ζ + f)/h`,
discretized after McRae & Cotter (2014) with `u, F ∈ RaviartThomas(k)`,
`h, K ∈ Discontinuous(k)` and `q ∈ Lagrange(k + 1)`:

    ∫ γ q h = -∫ u ⋅ (r̂ × ∇γ) + ∫ γ f        (diagnostic PV)
    ∫ w ⋅ F = ∫ w ⋅ h u                       (mass flux)
    ∫ φ K   = ∫ φ |u|²/2
    ∫ w ⋅ u̇ = -∫ w ⋅ (q r̂ × F) + ∫ (∇⋅w) (g h + K)
    ∫ φ ḣ   = -∫ φ ∇⋅F

The PV flux term does no work on `F`, so mass and energy
`∫ h|u|²/2 + g h²/2` are conserved by the semi-discretization. The state
vector is `[u; h]`. Nonlinear terms are integrated with quadrature `degree`.
"""
struct NonlinearShallowWater{BU, BH, BQ, LU, LH}
    U::FESpace
    H::FESpace
    Q::FESpace
    basis_u::Vector{BU}
    basis_h::Vector{BH}
    basis_q::Vector{BQ}
    Mu::SparseMatrixCSC{Float64, Int}
    Mh::SparseMatrixCSC{Float64, Int}
    Mu_factor::LU
    Mh_factor::LH
    D::SparseMatrixCSC{Float64, Int}   # ∫ φ_i ∇⋅w_j
    R::SparseMatrixCSC{Float64, Int}   # ∫ w_j ⋅ (r̂ × ∇γ_i)
    coriolis_load::Vector{Float64}     # ∫ γ_i f
    gravity::Float64
end

function NonlinearShallowWater(dm::Plex, k::Integer; gravity::Real=1.0, coriolis=x -> 2x[3],
                               degree::Integer=3k + 6)
    U = FESpace(dm, RaviartThomas(k); name="u")
    H = FESpace(dm, Discontinuous(k); name="h")
    Q = FESpace(dm, Lagrange(k + 1); name="q")
    Mu, Mh = assemble_mass(U), assemble_mass(H)
    D = assemble_divergence(H, U; degree)
    R = assemble_bilinear(Q, U; degree) do t, u, a, b, q
        dot(u.φ[b, q], cross(normalize(t.points[q]), t.grad[a, q])) * t.dA[q]
    end
    return NonlinearShallowWater(U, H, Q, precompute_basis(U, degree), precompute_basis(H, degree),
                                 precompute_basis(Q, degree), Mu, Mh,
                                 cholesky(Symmetric(Mu)), cholesky(Symmetric(Mh)), D, R,
                                 assemble_load(Q, coriolis; degree), Float64(gravity))
end

velocity_dofs(m::NonlinearShallowWater) = 1:num_dofs(m.U)
height_dofs(m::NonlinearShallowWater) = num_dofs(m.U) .+ (1:num_dofs(m.H))
initial_state(m::NonlinearShallowWater, velocity, height) = [project(m.U, velocity); project(m.H, height)]

"Value of a discrete field with coefficients `x` at quadrature point `q` of a cell basis."
field_at(basis, x, q) = sum(x[basis.indices[a]] * basis.φ[a, q] for a in eachindex(basis.indices))

"""
    diagnostics(m, u, h) -> (q, F, K)

Potential vorticity, mass flux and kinetic energy of the state `(u, h)`.
"""
function diagnostics(m::NonlinearShallowWater, u::AbstractVector, h::AbstractVector)
    rhs_F = zeros(num_dofs(m.U))
    rhs_K = zeros(num_dofs(m.H))
    rows, cols, vals = Int[], Int[], Float64[]
    for (bu, bh, bq) in zip(m.basis_u, m.basis_h, m.basis_q)
        nq = length(bq.indices)
        local_mass = zeros(nq, nq)
        for q in eachindex(bu.dA)
            uq, hq, dA = field_at(bu, u, q), field_at(bh, h, q), bu.dA[q]
            for a in eachindex(bu.indices)
                rhs_F[bu.indices[a]] += dot(bu.φ[a, q], hq * uq) * dA
            end
            for a in eachindex(bh.indices)
                rhs_K[bh.indices[a]] += bh.φ[a, q] * dot(uq, uq) / 2 * dA
            end
            for b in 1:nq, a in 1:nq
                local_mass[a, b] += hq * bq.φ[a, q] * bq.φ[b, q] * dA
            end
        end
        for b in 1:nq, a in 1:nq
            push!(rows, bq.indices[a]); push!(cols, bq.indices[b]); push!(vals, local_mass[a, b])
        end
    end
    Mq = sparse(rows, cols, vals, num_dofs(m.Q), num_dofs(m.Q))
    q = cholesky(Symmetric(Mq)) \ (m.coriolis_load - m.R * u)
    return q, m.Mu_factor \ rhs_F, m.Mh_factor \ rhs_K
end

function tendency!(dy::AbstractVector, y::AbstractVector, m::NonlinearShallowWater)
    u, h = y[velocity_dofs(m)], y[height_dofs(m)]
    q, F, K = diagnostics(m, u, h)
    rhs_u = m.D' * (m.gravity * h + K)
    for (bu, bq) in zip(m.basis_u, m.basis_q)
        for p in eachindex(bu.dA)
            flux = field_at(bq, q, p) * cross(normalize(bu.points[p]), field_at(bu, F, p)) * bu.dA[p]
            for a in eachindex(bu.indices)
                rhs_u[bu.indices[a]] -= dot(bu.φ[a, p], flux)
            end
        end
    end
    dy[velocity_dofs(m)] = m.Mu_factor \ rhs_u
    dy[height_dofs(m)] = m.Mh_factor \ (-(m.D * F))
    return dy
end

total_mass(m::NonlinearShallowWater, y) = sum(m.Mh * view(y, height_dofs(m)))

function total_energy(m::NonlinearShallowWater, y)
    u, h = view(y, velocity_dofs(m)), view(y, height_dofs(m))
    return sum(zip(m.basis_u, m.basis_h)) do (bu, bh)
        sum(eachindex(bu.dA)) do q
            uq, hq = field_at(bu, u, q), field_at(bh, h, q)
            (hq * dot(uq, uq) + m.gravity * hq^2) / 2 * bu.dA[q]
        end
    end
end

"Potential enstrophy `∫ q² h / 2`, conserved by the continuous equations."
function potential_enstrophy(m::NonlinearShallowWater, y)
    u, h = y[velocity_dofs(m)], y[height_dofs(m)]
    q, _, _ = diagnostics(m, u, h)
    return sum(zip(m.basis_q, m.basis_h)) do (bq, bh)
        sum(field_at(bq, q, p)^2 * field_at(bh, h, p) / 2 * bq.dA[p] for p in eachindex(bq.dA))
    end
end
