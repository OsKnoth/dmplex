# Face gradients and face values from the interaction regions around the vertices (multipoint flux
# approximation, POLYEDER `TGrad`)

"Faces of every vertex (the faces whose boundary loop contains it)."
function vertex_faces(cx::PolyComplex)
    at = [Int[] for _ in 1:num_poly_vertices(cx)]
    for f in 1:num_poly_faces(cx), v in face_loop(cx, f)
        push!(at[v], f)
    end
    return at
end

"""
    subface_weights(cx, f) -> Vector

The shares of the vertices of face `f` (in the order of `face_loop`) in its area: the area of the
quadrilateral of the vertex, the midpoints of its two edges and the vertex mean of the face,
divided by the area of the face.
"""
function subface_weights(cx::PolyComplex, f::Integer)
    xs = face_points(cx, f)
    n, m = length(xs), sum(xs) / length(xs)
    a = map(1:n) do i
        v, next, prev = xs[i], (xs[i] + xs[mod1(i + 1, n)]) / 2, (xs[i] + xs[mod1(i - 1, n)]) / 2
        norm(area_vector([v, next, m, prev]))
    end
    return a ./ sum(a)
end

"""
    interaction_operators(cx, geometry) -> (G, T)

Sparse matrices (faces × cells) of the normal gradient `∇u⋅n_F` and of the value of `u` at the
faces from the cell averages, by the interaction regions around the vertices. In the region of a
vertex every cell `C` at it (a corner) has the linear gradient `g_C = G_C (u_F - u_C)_F` from the
values at the centroids of its faces at the vertex (`G_C` the pseudo-inverse of the vectors from
its centroid to those of the faces); the values at the faces are fixed by the continuity of
`g⋅n_F` across every interior face of the region, `g⋅n_F = 0` on the boundary. The normal gradient
and the value at a face are the means over its vertices weighted by `subface_weights`. Exact for
linear `u` at faces whose vertices are inside the domain.
"""
function interaction_operators(cx::PolyComplex, geometry::PolyGeometry)
    nf, nc = num_poly_faces(cx), num_poly_cells(cx)
    fc = face_cells(cx)
    loops = [face_loop(cx, f) for f in 1:nf]
    weights = [subface_weights(cx, f) for f in 1:nf]
    xf, xc, nrm = geometry.face_centroid, geometry.cell_centroid, geometry.normal
    IG, JG, VG = Int[], Int[], Float64[]
    IT, JT, VT = Int[], Int[], Float64[]
    for (v, faces) in enumerate(vertex_faces(cx))
        cells = unique(filter(>(0), vec(fc[:, faces])))
        nF, nC = length(faces), length(cells)
        fpos = Dict(f => i for (i, f) in enumerate(faces))
        cpos = Dict(c => i for (i, c) in enumerate(cells))
        # g_C = Gu u_faces + Gc u_cells for every corner
        corner = Dict{Int, Tuple{Matrix{Float64}, Matrix{Float64}}}()
        for c in cells
            own = [f for f in faces if fc[1, f] == c || fc[2, f] == c]
            D = reduce(vcat, ((xf[f] - xc[c])' for f in own))
            P = pinv(D)                                   # 3 × faces of the corner
            Gu, Gc = zeros(3, nF), zeros(3, nC)
            for (k, f) in enumerate(own)
                Gu[:, fpos[f]] .+= P[:, k]
                Gc[:, cpos[c]] .-= P[:, k]
            end
            corner[c] = (Gu, Gc)
        end
        # continuity of the normal gradient (zero on the boundary): A u_faces = B u_cells
        A, B = zeros(nF, nF), zeros(nF, nC)
        for (i, f) in enumerate(faces)
            n = nrm[f]'
            c1, c2 = fc[1, f], fc[2, f]
            for (c, s) in ((c1, 1.0), (c2, -1.0))
                c == 0 && continue
                Gu, Gc = corner[c]
                A[i, :] .+= s .* vec(n * Gu)
                B[i, :] .-= s .* vec(n * Gc)
            end
        end
        U = A \ B                                         # faces × cells: the values at the faces
        for (i, f) in enumerate(faces)
            w = weights[f][findfirst(==(v), loops[f])]
            c = fc[1, f] > 0 ? fc[1, f] : fc[2, f]
            Gu, Gc = corner[c]
            grad = nrm[f]' * (Gu * U .+ Gc)               # 1 × cells
            for (j, cj) in enumerate(cells)
                push!(IG, f); push!(JG, cj); push!(VG, w * grad[j])
                push!(IT, f); push!(JT, cj); push!(VT, w * U[i, j])
            end
        end
    end
    return sparse(IG, JG, VG, nf, nc), sparse(IT, JT, VT, nf, nc)
end

"""
    cell_vectors(cx, geometry, un) -> Vector{SVector{3}}

The vector at every cell from the normal components `un` at its faces (oriented as the faces),
`v_C = |C|⁻¹ Σ_F s_CF |F| un_F (x_F - x_C)` (Perot 2000; POLYEDER `SumNormal`), exact for constant
vectors when all its faces are given.
"""
function cell_vectors(cx::PolyComplex, geometry::PolyGeometry, un::AbstractVector)
    return map(1:num_poly_cells(cx)) do c
        faces, signs = cell_cone(cx, c)
        sum(s * geometry.area[f] * un[f] * (geometry.face_centroid[f] - geometry.cell_centroid[c])
            for (f, s) in zip(faces, signs)) / geometry.volume[c]
    end
end

"""
    face_normal_components(cx, geometry, vc) -> Vector

The normal components at the faces of the cell vectors `vc`, interpolated linearly along the
line between the two cell centroids to the face (the vector of the one cell on the boundary).
"""
function face_normal_components(cx::PolyComplex, geometry::PolyGeometry, vc::AbstractVector)
    fc = face_cells(cx)
    return map(1:num_poly_faces(cx)) do f
        c1, c2 = fc[1, f], fc[2, f]
        n = geometry.normal[f]
        (c1 == 0 || c2 == 0) && return dot(n, vc[max(c1, c2)])
        d1 = abs(dot(n, geometry.face_centroid[f] - geometry.cell_centroid[c1]))
        d2 = abs(dot(n, geometry.cell_centroid[c2] - geometry.face_centroid[f]))
        dot(n, (d2 * vc[c1] + d1 * vc[c2]) / (d1 + d2))
    end
end

"""
    linear_vector_reconstruction(cx, geometry; basis=:tangential, up=normalize, own_weight=1) -> (R₁, R₂, R₃)

Sparse matrices (cells × faces) of the Cartesian components of the vector at every cell centroid from
the normal components at the interior faces: the least-squares fit of a linear vector field
`v(x) = v₀ + B y`, `y` the local coordinates of the cell (`PolyReconstruction`), to the normal
components on the faces of the cell and of the cells sharing a vertex with it; `v_C = v₀`. Exact for
linear fields on any mesh (Perot's `cell_vectors` only for constant ones); the faces of the cell itself
weighted by `own_weight` (needed for the stability of the shallow-water scheme). `basis = :tangential`: `v`
and its variation in the tangent plane of the cell (`up` its normal), for one layer over the sphere.
"""
function linear_vector_reconstruction(cx::PolyComplex, geometry::PolyGeometry; basis::Symbol=:tangential,
                                      up=normalize, own_weight::Real=1.0)
    nc, nf = num_poly_cells(cx), num_poly_faces(cx)
    fc = face_cells(cx)
    interior(f) = fc[1, f] > 0 && fc[2, f] > 0
    nb = vertex_neighbours(cx)
    I, J, V = [Int[] for _ in 1:3], [Int[] for _ in 1:3], [Float64[] for _ in 1:3]
    for c in 1:nc
        xc = geometry.cell_centroid[c]
        Q = basis == :full ? SMatrix{3, 3}(1.0I) : frame_with_normal(up(xc))
        dims = basis == :full ? 3 : 2
        axes = [Q[k, :] for k in 1:dims]                 # the directions of v and of its variation
        h = maximum(v -> norm(cx.coordinates[v] - xc), cell_vertices(cx, c))
        faces = unique(f for s in [c; nb[c]] for f in cell_cone(cx, s)[1] if interior(f))
        own = Set(cell_cone(cx, c)[1])
        w = [f in own ? own_weight : 1.0 for f in faces]
        rows = map(faces) do f
            n, y = geometry.normal[f], Q * (geometry.face_centroid[f] - xc) / h
            # n ⋅ (Σ_i a_i e_i + Σ_i Σ_j b_ij y_j e_i)
            [[dot(n, e) for e in axes]; [dot(n, e) * y[j] for e in axes for j in 1:dims]]'
        end
        P = (pinv(w .* reduce(vcat, rows)) .* w')[1:dims, :]   # v₀ in the axes, weighted fit
        for k in 1:3, (i, f) in enumerate(faces)
            w = sum(P[d, i] * axes[d][k] for d in 1:dims)
            push!(I[k], c); push!(J[k], f); push!(V[k], w)
        end
    end
    return ntuple(k -> sparse(I[k], J[k], V[k], nc, nf), 3)
end
