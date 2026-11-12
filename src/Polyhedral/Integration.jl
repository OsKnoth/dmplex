# Exact integration of monomials over segments, polygons and polyhedra (homogeneous numerical
# integration, Chin, Lasserre and Sukumar 2015)

"""
    MonomialBasis(p)

The monomials `y^α = y₁^α₁ y₂^α₂ y₃^α₃` of total degree `|α| ≤ p` in the order of the degree,
`exponents[j] = α`; `lower[i, j]` is the index of `α - e_i` (0 for `α_i = 0`).
"""
struct MonomialBasis
    degree::Int
    exponents::Vector{NTuple{3, Int}}
    lower::Matrix{Int}
end

function MonomialBasis(p::Integer)
    p >= 0 || throw(ArgumentError("the degree must be nonnegative"))
    exponents = [(a, b, q - a - b) for q in 0:p for a in q:-1:0 for b in (q - a):-1:0]
    index = Dict(α => j for (j, α) in enumerate(exponents))
    lower = [α[i] > 0 ? index[ntuple(l -> α[l] - (l == i), 3)] : 0 for i in 1:3, α in exponents]
    return MonomialBasis(Int(p), exponents, lower)
end

num_monomials(B::MonomialBasis) = length(B.exponents)

monomial(y, α) = y[1]^α[1] * y[2]^α[2] * y[3]^α[3]

"""
The recursion of the homogeneous integration on a `k`-dimensional polytope `P` with the moments
`boundary[j] = Σ_G d_G ∫_G y^α` of its facets `G` (`d_G` the distance of the hull of `G` from the
point `x0` of the hull of `P` along the outward normal of `G` in the hull of `P`):
`(k + |α|) ∫_P y^α = Σ_G d_G ∫_G y^α + Σ_i x0_i α_i ∫_P y^{α - e_i}` (Euler's identity and the
divergence theorem in the hull of `P`), in the order of the degree.
"""
function homogeneous_recursion!(I, boundary, x0, k, B::MonomialBasis)
    for (j, α) in enumerate(B.exponents)
        s = boundary[j]
        for i in 1:3
            l = B.lower[i, j]
            l > 0 && (s += x0[i] * α[i] * I[l])
        end
        I[j] = s / (k + sum(α))
    end
    return I
end

"""
    segment_moments(a, b, B) -> Vector

`∫_S y^α ds` over the segment from `a` to `b` for the monomials of `B`.
"""
function segment_moments(a::SVector{3}, b::SVector{3}, B::MonomialBasis)
    t = normalize(b - a)
    x0 = a - dot(a, t) * t                 # the foot of the origin on the line
    boundary = [dot(t, b) * monomial(b, α) - dot(t, a) * monomial(a, α) for α in B.exponents]
    return homogeneous_recursion!(similar(boundary), boundary, x0, 1, B)
end

"Area vector `½ Σ yᵢ × yᵢ₊₁` of the closed loop `ys` (Newell; normal by the right-hand rule)."
area_vector(ys) = sum(i -> cross(ys[i], ys[mod1(i + 1, length(ys))]), eachindex(ys)) / 2

"""
    polygon_moments(ys, B) -> Vector

`∫_P y^α dA` over the planar polygon with the vertices `ys` in cyclic order for the monomials of `B`.
"""
function polygon_moments(ys::AbstractVector{<:SVector{3}}, B::MonomialBasis)
    n = normalize(area_vector(ys))
    x0 = dot(n, ys[1]) * n                 # the foot of the origin on the plane
    boundary = zeros(num_monomials(B))
    for i in eachindex(ys)
        a, b = ys[i], ys[mod1(i + 1, length(ys))]
        ν = cross(normalize(b - a), n)     # outward in the plane for the loop counter-clockwise about n
        boundary .+= dot(ν, a) .* segment_moments(a, b, B)
    end
    return homogeneous_recursion!(similar(boundary), boundary, x0, 2, B)
end

"""
    planar_pieces(ys; tol=1e-10) -> Vector

The loop `ys` itself if its vertices lie in one plane (to `tol` times its size), else the triangles of
the fan from its vertex mean, oriented as the loop (a non-planar face, as the spherical polygons of
an extruded shell).
"""
function planar_pieces(ys::AbstractVector{<:SVector{3}}; tol::Real=1e-10)
    length(ys) == 3 && return [collect(ys)]
    m = sum(ys) / length(ys)
    n = normalize(area_vector(ys .- Ref(m)))
    size = maximum(y -> norm(y - m), ys)
    all(y -> abs(dot(n, y - m)) <= tol * size, ys) && return [collect(ys)]
    return [[m, ys[i], ys[mod1(i + 1, length(ys))]] for i in eachindex(ys)]
end

"Vertex mean and largest distance of a vertex from it, the default centre and scale of the moments."
function default_frame(xs)
    m = sum(xs) / length(xs)
    return m, maximum(x -> norm(x - m), xs)
end

face_points(cx::PolyComplex, f::Integer) = cx.coordinates[face_loop(cx, f)]

"""
    face_moments(cx, f, B; center, scale, rotation=I) -> Vector

`∫_F (Q (x - center)/scale)^α dA` over face `f` of `cx` (non-planar faces as `planar_pieces`), by
default about the vertex mean of the face and scaled by its largest vertex distance; `Q = rotation`
an orthogonal matrix (the local axes as rows).
"""
function face_moments(cx::PolyComplex, f::Integer, B::MonomialBasis; center=nothing, scale=nothing, rotation=I)
    xs = face_points(cx, f)
    m, h = default_frame(xs)
    center = something(center, m)
    scale = something(scale, h)
    ys = [SVector{3}(rotation * (x - center)) / scale for x in xs]
    return scale^2 .* sum(P -> polygon_moments(P, B), planar_pieces(ys))
end

"""
    cell_moments(cx, c, B; center, scale, rotation=I) -> Vector

`∫_C (Q (x - center)/scale)^α dV` over cell `c` of `cx`, by default about the vertex mean of the cell
and scaled by its largest vertex distance, `Q = rotation` orthogonal (the local axes as rows): `(3 + |α|) ∫_C y^α = Σ_F d_F ∫_F y^α` over its boundary
faces (outward, non-planar faces as `planar_pieces`), `d_F` the distance of the plane of `F` from
the origin of `y`.
"""
function cell_moments(cx::PolyComplex, c::Integer, B::MonomialBasis; center=nothing, scale=nothing, rotation=I)
    faces, signs = cell_cone(cx, c)
    loops = [signs[i] > 0 ? face_loop(cx, f) : reverse(face_loop(cx, f)) for (i, f) in enumerate(faces)]
    m, h = default_frame(cx.coordinates[unique(reduce(vcat, loops))])
    center = something(center, m)
    scale = something(scale, h)
    boundary = zeros(num_monomials(B))
    for loop in loops, P in planar_pieces([SVector{3}(rotation * (x - center)) / scale for x in cx.coordinates[loop]])
        boundary .+= dot(normalize(area_vector(P)), P[1]) .* polygon_moments(P, B)
    end
    return scale^3 .* homogeneous_recursion!(similar(boundary), boundary, zero(SVector{3, Float64}), 3, B)
end

# seven-point rule of degree 5 on a triangle (Dunavant 1985): barycentric points, weights summing to 1
const TRIANGLE_RULE = let a1 = 0.059715871789770, b1 = 0.470142064105115, a2 = 0.797426985353087, b2 = 0.101286507323456,
                          w1 = 0.132394152788506, w2 = 0.125939180544827
    ([(1 / 3, 1 / 3, 1 / 3), (a1, b1, b1), (b1, a1, b1), (b1, b1, a1), (a2, b2, b2), (b2, a2, b2), (b2, b2, a2)],
     [0.225, w1, w1, w1, w2, w2, w2])
end

"""
    face_quadrature(cx, f) -> (points, weights)

Quadrature of degree 5 on face `f`: the seven-point rule on each triangle of the fan from the
vertex mean; `weights[g]` is the area vector of point `g` (area weight times the unit normal of
its triangle, oriented as the face), so that `Σ_g φ(x_g) weights[g] ≈ ∫_F φ n dA`.
"""
function face_quadrature(cx::PolyComplex, f::Integer)
    xs = face_points(cx, f)
    m = sum(xs) / length(xs)
    bary, w = TRIANGLE_RULE
    points, weights = SVector{3, Float64}[], SVector{3, Float64}[]
    for i in eachindex(xs)
        a, b = xs[i], xs[mod1(i + 1, length(xs))]
        A = cross(a - m, b - m) / 2
        for (λ, wk) in zip(bary, w)
            push!(points, λ[1] * m + λ[2] * a + λ[3] * b)
            push!(weights, wk * A)
        end
    end
    return points, weights
end
