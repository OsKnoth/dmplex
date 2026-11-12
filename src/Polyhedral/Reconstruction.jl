# Polynomial reconstruction from cell averages that preserves the average of the cell

"""
    PolyReconstruction(cx, geometry, degree; basis=:full, rings=nothing, up=normalize, rtol=1e-8)

For every cell `C` the polynomial `p_C = Σ_k c_k y^{α_k}` of degree `≤ degree` in the local
coordinates `y = Q (x - x_C)/h_C` (centroid `x_C`, largest distance `h_C` of a vertex) that has the
average of `C` and fits the averages of the other cells of its stencil in the least-squares sense:
the constraint eliminates the constant, the remaining problem is solved by the pseudo-inverse with
the relative rank tolerance `rtol` (POLYEDER `TolRank`). The coefficients are linear in the averages
of the stencil, `c = R ū_S` (`coefficients[C]`, monomials × stencil).

`stencil[C]` is `C` followed by the cells sharing a vertex with it, `rings` times; by default the
rings grow (up to 4) until the least-squares problem has full rank and at least 1.5 times as many
cells as monomials (more rings at boundaries). `basis = :full`
uses all monomials of the degree (`Q = I`); `basis = :tangential` the monomials without the third
local coordinate along `up(x_C)` (the radius on a spherical shell): a reconstruction in the tangent
plane for one layer of cells, whose radial structure the stencil cannot resolve.
"""
struct PolyReconstruction
    basis::MonomialBasis
    terms::Vector{Int}                         # the monomials used, the constant first
    stencil::Vector{Vector{Int}}
    coefficients::Vector{Matrix{Float64}}
    center::Vector{SVector{3, Float64}}
    scale::Vector{Float64}
    rotation::Vector{SMatrix{3, 3, Float64, 9}}
end

"Rows of an orthogonal matrix with `n` as the third axis."
function frame_with_normal(n)
    a = abs(n[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    t1 = normalize(a - dot(a, n) * n)
    t2 = cross(n, t1)
    return SMatrix{3, 3}(vcat(t1', t2', n'))
end

function PolyReconstruction(cx::PolyComplex, geometry::PolyGeometry, degree::Integer; basis::Symbol=:full,
                            rings::Union{Nothing, Integer}=nothing, up=normalize, rtol::Real=1e-8)
    basis in (:full, :tangential) || throw(ArgumentError("basis must be :full or :tangential"))
    B = MonomialBasis(degree)
    terms = basis == :full ? collect(1:num_monomials(B)) : findall(α -> α[3] == 0, B.exponents)
    nc = num_poly_cells(cx)
    nb = vertex_neighbours(cx)
    center = geometry.cell_centroid
    scale = [maximum(v -> norm(cx.coordinates[v] - center[c]), cell_vertices(cx, c)) for c in 1:nc]
    rotation = [basis == :full ? SMatrix{3, 3}(1.0I) : frame_with_normal(up(center[c])) for c in 1:nc]
    nt = length(terms)
    # the averages of the monomials of cell c over the cells S, and the reduced matrix of the fit
    averages(c, S) = reduce(vcat, map(S) do s
        I = cell_moments(cx, s, B; center=center[c], scale=scale[c], rotation=rotation[c])
        (I[terms] ./ I[1])'
    end)
    reduced(A) = A[2:end, 2:end] .- A[1:1, 2:end]     # Σ_k (a_jk - a_Ck) c_k = ū_j - ū_C
    stencil, coefficients = Vector{Vector{Int}}(undef, nc), Vector{Matrix{Float64}}(undef, nc)
    for c in 1:nc
        S, A = [c], averages(c, [c])
        for ring in 1:something(rings, 4)
            S = unique(vcat(S, reduce(vcat, (nb[s] for s in S))))
            A = averages(c, S)
            rings === nothing || continue
            nt == 1 || (length(S) >= 1.5nt && rank(reduced(A); rtol) == nt - 1) && break
        end
        ns = length(S)
        R = zeros(nt, ns)
        if nt > 1
            P = pinv(reduced(A); rtol)
            R[2:end, 2:end] .= P
            R[2:end, 1] .= -vec(sum(P; dims=2))
        end
        R[1, :] .= -(A[1, 2:end]' * R[2:end, :])'
        R[1, 1] += 1
        stencil[c], coefficients[c] = S, R
    end
    return PolyReconstruction(B, terms, stencil, coefficients, center, scale, rotation)
end

"Value at `x` of the polynomial of cell `c` from the cell averages `u`."
function reconstruct(rec::PolyReconstruction, c::Integer, u::AbstractVector, x::SVector{3})
    y = rec.rotation[c] * (x - rec.center[c]) / rec.scale[c]
    coef = rec.coefficients[c] * u[rec.stencil[c]]
    return sum(coef[k] * monomial(y, rec.basis.exponents[rec.terms[k]]) for k in eachindex(coef))
end

"""
    face_value_matrices(cx, rec) -> (Q₁, Q₂)

Sparse matrices (faces × cells) of the averages over every face of the polynomials of its two cells
(`face_cells`: the cell the normal points out of, then the one it points into; empty rows on the
boundary), so that `Q_s ū` are the face values of side `s`.
"""
function face_value_matrices(cx::PolyComplex, rec::PolyReconstruction)
    fc = face_cells(cx)
    nf, nc = num_poly_faces(cx), num_poly_cells(cx)
    matrices = map(1:2) do side
        I, J, V = Int[], Int[], Float64[]
        for f in 1:nf
            c = fc[side, f]
            c == 0 && continue
            m = face_moments(cx, f, rec.basis; center=rec.center[c], scale=rec.scale[c], rotation=rec.rotation[c])
            w = (m[rec.terms] ./ m[1])' * rec.coefficients[c]
            append!(I, fill(f, length(w))); append!(J, rec.stencil[c]); append!(V, vec(w))
        end
        sparse(I, J, V, nf, nc)
    end
    return matrices[1], matrices[2]
end

"""
    point_value_matrices(cx, rec, points) -> (P₁, P₂)

Sparse matrices of the values of the polynomials of the two cells of every face (`face_cells`) at
the points `points[f]` of the face, rows in the order of the faces and their points.
"""
function point_value_matrices(cx::PolyComplex, rec::PolyReconstruction, points::AbstractVector)
    fc = face_cells(cx)
    nrow = sum(length, points)
    matrices = map(1:2) do side
        I, J, V = Int[], Int[], Float64[]
        row = 0
        for f in eachindex(points), x in points[f]
            row += 1
            c = fc[side, f]
            c == 0 && continue
            y = rec.rotation[c] * (x - rec.center[c]) / rec.scale[c]
            w = [monomial(y, rec.basis.exponents[k]) for k in rec.terms]' * rec.coefficients[c]
            append!(I, fill(row, length(w))); append!(J, rec.stencil[c]); append!(V, vec(w))
        end
        sparse(I, J, V, nrow, num_poly_cells(cx))
    end
    return matrices[1], matrices[2]
end
