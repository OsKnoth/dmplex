const PHI = (1.0 + sqrt(5.0)) / 2.0

"""
    create_spherical_mesh(subdivisions; spring_beta=nothing, centroidal=false)

Icosahedral triangle mesh on the unit sphere; every subdivision splits each
triangle into four and projects the new edge midpoints onto the sphere.

With a number `spring_beta` the vertices are optimized after every subdivision by the spring
dynamics of Tomita et al. (2002) (see [`spring_dynamics!`](@ref)), as the
grids of NICAM and, in a modified form, ICON: plain bisection leaves the cell
shapes changing abruptly across the edges of the coarser levels, and the
grid defects do not shrink with refinement. Tomita et al. recommend
`spring_beta = 1.2`; larger values fold the mesh over.

With `centroidal = true` the vertices are then moved by Lloyd's algorithm (see
[`lloyd!`](@ref)) until they are the centroids of their Voronoi cells, after
every subdivision too: the hexagonal dual is then a spherical centroidal
Voronoi tessellation (SCVT) as the MPAS grids, on which the TRiSK operators
are consistent.
"""
function create_spherical_mesh(subdivisions::Int; spring_beta=nothing, centroidal::Bool=false)
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
        centroidal && lloyd!(vertices, triangles; fixed=1:12)
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

"""
    lloyd!(x, triangles; fixed=(), tolerance=1e-8, history=16, max_iterations=10_000)

Move the vertices `x` of the counter-clockwise triangulation `triangles` of the
unit sphere to the centroids of their Voronoi cells (Lloyd's algorithm for
spherical centroidal Voronoi tessellations, Du, Gunzburger & Ju 2003). The
Voronoi cell of a vertex is the polygon of the circumcenters `cₖ` of its
triangles; its centroid is the normalized `∫ x dA = ½ Σₖ θₖ (cₖ × cₖ₊₁)/|cₖ × cₖ₊₁|`
(`θₖ` the arc from `cₖ` to `cₖ₊₁`). The connectivity is kept, so the
triangulation must stay Delaunay (checked at the end). The fixed-point
iteration `x ← centroid(x)` is accelerated by Anderson mixing over the last
`history` steps (`history = 0`: plain Lloyd); it stops when the largest
distance of a free vertex from its centroid is below `tolerance` times the mean
edge length. Returns the number of iterations.
"""
function lloyd!(x::Vector{SVector{3, Float64}}, triangles; fixed=(), tolerance::Real=1e-8,
                history::Integer=16, max_iterations::Integer=10_000)
    free = trues(length(x))
    free[collect(fixed)] .= false
    # triangles around every vertex, counter-clockwise: the triangle after (v, b, c) contains (v, c)
    star = [Int[] for _ in eachindex(x)]
    after = Dict{Tuple{Int, Int}, Int}()   # (v, b) -> triangle with the corner v followed by b
    for (t, (a, b, c)) in enumerate(triangles)
        after[(a, b)], after[(b, c)], after[(c, a)] = t, t, t
    end
    for (t, tri) in enumerate(triangles), k in 1:3
        v, w = tri[k], tri[mod1(k + 2, 3)]   # corner v, previous corner w: the next triangle has (v, w)
        isempty(star[v]) || continue
        push!(star[v], t)
        while true
            next = after[(v, w)]
            next == t && break
            push!(star[v], next)
            tn = triangles[next]
            w = tn[mod1(findfirst(==(v), tn) + 2, 3)]
        end
    end
    h = sum(((a, b, c),) -> norm(x[b] - x[a]), triangles) / length(triangles)
    centers = similar(x, length(triangles))
    # Anderson mixing (type II) over the last m steps: with the differences ΔX, ΔF of the iterates and of the
    # steps f = g(x) - x, x ← x + f - (ΔX + ΔF) γ, γ = argmin |f - ΔF γ| (normal equations, regularized)
    f, fold, xold = (zeros(SVector{3, Float64}, length(x)) for _ in 1:3)
    ΔX, ΔF = zeros(3length(x), history), zeros(3length(x), history)
    flat(v) = reinterpret(Float64, v)
    m = 0
    for iteration in 1:max_iterations
        residual = lloyd_step!(f, centers, x, triangles, star, free)
        if residual < tolerance * h
            all(t -> is_delaunay(x, triangles, after, t), triangles) ||
                throw(ArgumentError("Lloyd's algorithm left the triangulation non-Delaunay"))
            return iteration
        end
        if history > 0 && iteration > 1
            k = mod1(iteration - 1, history)    # ring buffer of the last `history` differences
            ΔX[:, k] .= flat(x) .- flat(xold)
            ΔF[:, k] .= flat(f) .- flat(fold)
            m = min(m + 1, history)
        end
        xold .= x
        fold .= f
        if m > 0
            A, B = view(ΔF, :, 1:m), view(ΔX, :, 1:m)
            N = A' * A
            γ = (N + 1e-12 * maximum(diag(N)) * I) \ (A' * flat(f))
            flat(x) .+= flat(f) .- (B * γ .+ A * γ)
        else
            x .+= f
        end
        x .= normalize.(x)
    end
    @warn "Lloyd's algorithm did not converge in $max_iterations iterations"
    return max_iterations
end

"Lloyd step `centroid - x` of the free vertices into `f`; returns its largest length."
function lloyd_step!(f, centers, x, triangles, star, free)
    for (t, (a, b, c)) in enumerate(triangles)
        centers[t] = normalize(cross(x[b] - x[a], x[c] - x[a]))
    end
    residual = 0.0
    for v in eachindex(x)
        if !free[v]
            f[v] = zero(eltype(f))
            continue
        end
        ring = star[v]
        m = zero(eltype(f))
        for k in eachindex(ring)
            p, q = centers[ring[k]], centers[ring[mod1(k + 1, length(ring))]]
            n = cross(p, q)
            m += atan(norm(n), dot(p, q)) / norm(n) * n
        end
        f[v] = normalize(m) - x[v]
        residual = max(residual, norm(f[v]))
    end
    return residual
end

"Delaunay test of triangle `(a, b, c)`: the vertex opposite to every edge lies outside its circumcircle."
function is_delaunay(x, triangles, after, (a, b, c))
    center = normalize(cross(x[b] - x[a], x[c] - x[a]))
    return all(((a, b), (b, c), (c, a))) do (p, q)
        o = first(setdiff(triangles[after[(q, p)]], (p, q)))
        dot(x[o], center) <= dot(x[a], center) + 1e-14
    end
end
