"""
    create_periodic_plane(nx, ny; Lx=1, Ly=1, cells=:quadrilateral)

Doubly periodic plane `[0, Lx) × [0, Ly)` with `nx × ny` square cells
(`cells=:quadrilateral`) or each square split into two triangles
(`cells=:triangle`). The surface is a torus: closed, Euler characteristic 0.
"""
function create_periodic_plane(nx::Int, ny::Int; Lx::Real=1.0, Ly::Real=1.0, cells::Symbol=:quadrilateral)
    (nx >= 3 && ny >= 3) || throw(ArgumentError("a periodic plane needs at least 3 cells per direction"))
    cells in (:quadrilateral, :triangle) || throw(ArgumentError("cells must be :quadrilateral or :triangle"))
    vertices = [SVector(i * Lx / nx, j * Ly / ny, 0.0) for j in 0:(ny - 1) for i in 0:(nx - 1)]
    id(i, j) = mod(i, nx) + nx * mod(j, ny) + 1
    polygons = Vector{Int}[]
    for j in 0:(ny - 1), i in 0:(nx - 1)
        a, b, c, d = id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)
        if cells == :quadrilateral
            push!(polygons, [a, b, c, d])
        else
            push!(polygons, [a, b, c], [a, c, d])
        end
    end
    return plex_from_polygons(polygons, vertices; period=(Lx, Ly))
end
