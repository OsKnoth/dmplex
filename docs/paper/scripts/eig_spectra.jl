# spectra of the one-layer kite tendency linearized around the uniform flow (1, 0.3), c = 0.5,
# periodic unit square 6 × 6, λ in units of |u|/Δx
#
#   julia --project eig_spectra.jl
using dmplex, LinearAlgebra, StaticArrays, Serialization, Printf
function spectrum(cells, ht, adv; c=0.5, vy=0.3, nx=6)
    primal = create_periodic_plane(nx, nx; cells)
    m2 = NonlinearKiteShallowWater(primal, 1; gravity=1.0, coriolis=x -> 0.0)
    m = KiteNonhydrostatic(primal, VerticalGrid(1, 1.0), 1; radius=1.0, coriolis=x -> 0.0,
                           equation_of_state=:shallow_water, gravity=1.0, horizontal_theta=ht, horizontal_advection=adv)
    y2 = initial_state(m2, x -> SVector(1.0, vy, 0.0), x -> c^2)
    y = zero_state(m); y.u[1, :] .= y2[velocity_dofs(m2)]; y.ρ[1, :] .= y2[height_dofs(m2)]; y.ρθ .= y.ρ
    nu = size(y.u, 1)
    function tend(x)
        s = map(copy, y); s.u[1, :] .= x[1:nu]; s.ρ[1, :] .= x[nu+1:end]; s.ρθ .= s.ρ
        d = horizontal_tendency!(m.work.dy, m, s)
        return vcat(vec(d.u), vec(d.ρ))
    end
    x0 = vcat(vec(y.u), vec(y.ρ)); f0 = tend(x0); ε = 1e-7
    J = hcat([(tend(x0 .+ ε .* (1:length(x0) .== i)) .- f0) ./ ε for i in 1:length(x0)]...)
    return eigvals(J) ./ (nx * hypot(1.0, vy))
end
res = Dict()
for cells in (:quadrilateral, :triangle), (name, ht, adv) in (("vector invariant", :dg_upwind, :vector_invariant),
                                                             ("DG, central", :dg, :dg), ("DG, upwind", :dg_upwind, :dg))
    λ = spectrum(cells, ht, adv)
    res[(cells, name)] = λ
    @printf("%-14s %-17s max Re λ %.4f  max |Im λ| %.2f\n", cells, name, maximum(real, λ), maximum(abs ∘ imag, λ))
end
serialize(joinpath(@__DIR__, "data", "spectra.jls"), res)
