"""
    cell_centers(dm; method=:auto)

Cell centers on the unit sphere, chosen so that they coincide with the
vertices of the orthogonal dual mesh whenever the mesh has one:

- `:circumcenter` (triangles only): the circumcenter, i.e. the vertex of the
  Voronoi dual (ICON triangles, MPAS Delaunay triangles).
- `:voronoi`: the generators of the Voronoi diagram that fits the mesh best.
  In a Voronoi diagram every edge lies on the perpendicular bisector of the
  two generators it separates: for an edge with great-circle normal `n`
  between cells `i` and `j`, `gᵢ - gⱼ ∥ n` and `n ⋅ (gᵢ + gⱼ) = 0`. These
  conditions are linear and homogeneous in all generators; their
  least-squares null vector is found by inverse iteration. For a Voronoi
  mesh (MPAS) this recovers the generators, i.e. the vertices of the
  Delaunay dual.
- `:centroid`: the normalized mean of the cell's vertices.
- `:auto` (default): `:circumcenter` for triangle meshes, `:voronoi` if the
  mesh is a Voronoi diagram (see [`orthogonality_defect`](@ref)), otherwise
  `:centroid` (cubed sphere, HEALPix).

`:voronoi` is a global problem; compute the centers on the whole mesh before
partitioning it. On a plane the centroids are used.
"""
function cell_centers(dm::Plex; method::Symbol=:auto, tolerance::Real=1e-8)
    if dm.surface == :plane
        method in (:auto, :centroid) || throw(ArgumentError("only centroids are implemented on planes"))
        return [surface_point(dm, sum(cell_coordinates(dm, c)) / get_cone_size(dm, c)) for c in cell_range(dm)]
    end
    if method == :auto
        all(c -> get_cone_size(dm, c) == 3, cell_range(dm)) && return cell_centers(dm; method=:circumcenter)
        generators = cell_centers(dm; method=:voronoi)
        orthogonality_defect(dm, generators) < tolerance && return generators
        return cell_centers(dm; method=:centroid)
    elseif method == :centroid
        return [normalize(sum(get_coordinates(dm, v) for v in get_cell_vertices(dm, c))) for c in cell_range(dm)]
    elseif method == :circumcenter
        return map(cell_range(dm)) do c
            get_cone_size(dm, c) == 3 || throw(ArgumentError("circumcenters need a triangle mesh"))
            a, b, d = (get_coordinates(dm, v) for v in get_cell_vertices(dm, c))
            normalize(cross(b - a, d - a))  # counter-clockwise cells: points outward
        end
    elseif method == :voronoi
        return voronoi_generators(dm)
    end
    throw(ArgumentError("unknown center method $method"))
end

function voronoi_generators(dm::Plex; iterations::Integer=4)
    rows, cols, vals = Int[], Int[], Float64[]
    nrow = 0
    for e in edge_range(dm)
        left, right = get_edge_cells(dm, e)
        (left == 0 || right == 0) && continue
        i, j = cell_index(dm, left), cell_index(dm, right)
        a, b = (get_coordinates(dm, v) for v in get_cone(dm, e))
        n = normalize(cross(a, b))
        # (I - n nᵀ)(gᵢ - gⱼ) = 0 as two rows with a tangent basis t₁, t₂ of the plane ⟂ n,
        # and n ⋅ (gᵢ + gⱼ) = 0
        t1 = normalize(cross(n, abs(n[1]) < 0.9 ? SVector(1.0, 0, 0) : SVector(0, 1.0, 0)))
        t2 = cross(n, t1)
        for (w, sj) in ((t1, -1), (t2, -1), (n, 1))
            nrow += 1
            for k in 1:3
                push!(rows, nrow, nrow); push!(cols, 3(i - 1) + k, 3(j - 1) + k); push!(vals, w[k], sj * w[k])
            end
        end
    end
    A = sparse(rows, cols, vals, nrow, 3 * num_cells(dm))
    B = A' * A
    F = cholesky(Symmetric(B + 1e-10 * maximum(abs, diag(B)) * I))
    centroids = cell_centers(dm; method=:centroid)
    x = reduce(vcat, centroids)
    for _ in 1:iterations
        x = F \ x
        x ./= norm(x)
    end
    generators = [normalize(SVector(x[3c - 2], x[3c - 1], x[3c])) for c in 1:num_cells(dm)]
    sum(dot.(generators, centroids)) < 0 && (generators .= .-generators)  # null vector sign
    return generators
end

"""
    orthogonality_defect(dm, centers)

Largest `|cos|` of the angle between a primal edge and the great circle
through the centers of its two cells; zero for a mesh with an orthogonal dual
(Voronoi–Delaunay pair).
"""
function orthogonality_defect(dm::Plex, centers)
    dm.surface == :sphere || return Inf
    defect = 0.0
    for e in edge_range(dm)
        left, right = get_edge_cells(dm, e)
        (left == 0 || right == 0) && continue
        a, b = (get_coordinates(dm, v) for v in get_cone(dm, e))
        dual = cross(centers[cell_index(dm, left)], centers[cell_index(dm, right)])
        defect = max(defect, abs(dot(normalize(dual), normalize(cross(a, b)))))
    end
    return defect
end
