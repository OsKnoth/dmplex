const PHI = (1.0 + sqrt(5.0)) / 2.0

"""
    create_spherical_mesh(subdivisions; spring_beta=nothing)

Icosahedral triangle mesh on the unit sphere; every subdivision splits each
triangle into four and projects the new edge midpoints onto the sphere.

With a number `spring_beta` the vertices are optimized after every subdivision by the spring
dynamics of Tomita et al. (2002) (see [`spring_dynamics!`](@ref)), as the
grids of NICAM and, in a modified form, ICON: plain bisection leaves the cell
shapes changing abruptly across the edges of the coarser levels, and the
grid defects do not shrink with refinement. Tomita et al. recommend
`spring_beta = 1.2`; larger values fold the mesh over.
"""
function create_spherical_mesh(subdivisions::Int; spring_beta=nothing)
    subdivisions >= 0 || throw(ArgumentError("subdivisions must be non-negative"))
    raw_coords = [
        SVector(-1.0,  PHI,  0.0), SVector( 1.0,  PHI,  0.0), SVector(-1.0, -PHI,  0.0), SVector( 1.0, -PHI,  0.0),
        SVector( 0.0, -1.0,  PHI), SVector( 0.0,  1.0,  PHI), SVector( 0.0, -1.0, -PHI), SVector( 0.0,  1.0, -PHI),
        SVector( PHI,  0.0, -1.0), SVector( PHI,  0.0,  1.0), SVector(-PHI,  0.0, -1.0), SVector(-PHI,  0.0,  1.0)
    ]
    vertices = [normalize(v) for v in raw_coords]
    triangles = [
        (1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
        (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
        (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
        (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2)
    ]

    for level in 1:subdivisions
        midpoints = Dict{Tuple{Int, Int}, Int}()
        midpoint(a, b) = get!(midpoints, minmax(a, b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        refined = NTuple{3, Int}[]
        for (a, b, c) in triangles
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            push!(refined, (a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
        end
        triangles = refined
        if spring_beta !== nothing   # relax every level: a good start for the next one
            edges = unique!([minmax(t[k], t[mod1(k + 1, 3)]) for t in triangles for k in 1:3])
            # mean grid interval: the side of the equilateral triangle of the mean cell area
            spring_dynamics!(vertices, edges; beta=spring_beta, mean_length=sqrt(4π / (5sqrt(3))) / 2^level,
                             fixed=1:12)
            all(((a, b, c),) -> dot(cross(vertices[b] - vertices[a], vertices[c] - vertices[a]), vertices[a]) > 0,
                triangles) || throw(ArgumentError("spring_beta = $spring_beta folds the mesh over; take a smaller one"))
        end
    end

    return plex_from_polygons([collect(t) for t in triangles], vertices)
end

"""
    spring_dynamics!(x, edges; beta=1.2, mean_length, fixed=(), tolerance=1e-10, max_iterations=100_000)

Optimize the points `x` on the unit sphere by the spring dynamics of Tomita
et al. (2002, J. Comput. Phys. 183): every edge `(i, j)` is a spring of
natural length `β d̄` (`d̄ = mean_length`, by default the mean chord length of
the initial `x`), the force on a point is
`Σⱼ (|xⱼ - xᵢ| - β d̄) eᵢⱼ` projected onto the tangent plane, and the damped
motion is integrated to the equilibrium (the largest tangential force below
`tolerance d̄`). The `fixed` points (the icosahedron vertices) do not
move. Returns the number of iterations.
"""
function spring_dynamics!(x::Vector{SVector{3, Float64}}, edges; beta::Real=1.2,
                          mean_length::Real=sum(((i, j),) -> norm(x[j] - x[i]), edges) / length(edges),
                          fixed=(), tolerance::Real=1e-10, max_iterations::Integer=100_000)
    free = trues(length(x))
    free[collect(fixed)] .= false
    d0 = beta * mean_length
    force = similar(x)
    velocity = zero.(x)
    # time step and damping in units of the spring stiffness: about six springs per point
    dt, damping = 0.3, 0.05
    for iteration in 1:max_iterations
        fill!(force, zero(eltype(x)))
        for (i, j) in edges
            r = x[j] - x[i]
            f = (norm(r) - d0) / norm(r) * r
            force[i] += f
            force[j] -= f
        end
        residual = 0.0
        for i in eachindex(x)
            free[i] || continue
            f = force[i] - dot(force[i], x[i]) * x[i]
            residual = max(residual, norm(f))
            velocity[i] = (1 - damping) * velocity[i] + dt * f
            velocity[i] -= dot(velocity[i], x[i]) * x[i]
            x[i] = normalize(x[i] + dt * velocity[i])
        end
        residual < tolerance * d0 && return iteration
    end
    @warn "spring dynamics did not converge in $max_iterations iterations"
    return max_iterations
end
