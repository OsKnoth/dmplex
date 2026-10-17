# Readers for ICON and MPAS grid files (NetCDF). Cells and vertices keep the
# numbering of the file; edges are renumbered by `plex_from_polygons`.

"""
Read a whole NetCDF variable as a Julia array, raw (without fill-value or
scaling conversion); dimensions are reversed with respect to `ncdump`.
"""
function read_variable(ds, name::AbstractString)
    raw = ds[name].var
    return raw[ntuple(_ -> Colon(), ndims(raw))...]
end

lonlat_to_xyz(λ::Real, φ::Real) = SVector(cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ))

"Arrange a connectivity table as `n × k` (one row per entity) given `n`."
function rows_per_entity(table::AbstractMatrix, n::Integer)
    size(table, 1) == n && return table
    size(table, 2) == n && return permutedims(table)
    throw(DimensionMismatch("connectivity table of size $(size(table)) does not match $n entities"))
end

"""
    sort_ring(ids, coordinates, axis)

Sort the points `coordinates[ids]` counter-clockwise around `axis` (seen from
outside the sphere).
"""
function sort_ring(ids::AbstractVector{<:Integer}, coordinates, axis::SVector{3})
    e1 = coordinates[ids[1]] - axis
    angle(id) = (v = coordinates[id] - axis; atan(dot(cross(e1, v), axis), dot(e1, v)))
    return ids[sortperm(angle.(ids))]
end

"""
    dual_polygons(rings, coordinates, centers)

Polygons of a dual mesh: `rings[v]` lists the primal cells around primal
vertex `v`, which become the corners of dual cell `v`.
"""
function dual_polygons(rings, coordinates, centers)
    return [sort_ring(ring, coordinates, normalize(centers[v])) for (v, ring) in enumerate(rings)]
end

"""
    read_icon_grid(filename; dual=false)

ICON triangular grid from a grid file (`vlon`, `vlat`, `vertex_of_cell`).
With `dual=true` the hexagonal/pentagonal dual is returned instead: its
vertices are the triangle centers (`clon`, `clat`) and its cells follow
`cells_of_vertex`. The dual is only available for global grids.
"""
function read_icon_grid(filename::AbstractString; dual::Bool=false)
    NCDataset(filename) do ds
        vertices = lonlat_to_xyz.(read_variable(ds, "vlon"), read_variable(ds, "vlat"))
        centers = lonlat_to_xyz.(read_variable(ds, "clon"), read_variable(ds, "clat"))
        if !dual
            table = rows_per_entity(read_variable(ds, "vertex_of_cell"), length(centers))
            return plex_from_polygons([Int.(table[c, :]) for c in axes(table, 1)], vertices)
        end
        table = rows_per_entity(read_variable(ds, "cells_of_vertex"), length(vertices))
        rings = [Int[c for c in table[v, :] if 1 <= c <= length(centers)] for v in axes(table, 1)]
        all(ring -> length(ring) >= 5, rings) ||
            throw(ArgumentError("the dual of a regional ICON grid is not supported"))
        return plex_from_polygons(dual_polygons(rings, centers, vertices), centers)
    end
end

"""
    read_mpas_grid(filename; dual=false)

MPAS Voronoi grid from a grid file (`x/y/zVertex`, `verticesOnCell`,
`nEdgesOnCell`); coordinates are projected onto the unit sphere. With
`dual=true` the Delaunay triangulation is returned instead: vertices at the
cell generators (`x/y/zCell`) and triangles from `cellsOnVertex`; triangles
touching a missing cell (regional grids) are dropped.
"""
function read_mpas_grid(filename::AbstractString; dual::Bool=false)
    NCDataset(filename) do ds
        if haskey(ds.attrib, "on_a_sphere") && strip(string(ds.attrib["on_a_sphere"])) != "YES"
            throw(ArgumentError("$filename is a planar MPAS mesh"))
        end
        xyz(prefix) = normalize.(SVector.(read_variable(ds, "x$prefix"), read_variable(ds, "y$prefix"),
                                          read_variable(ds, "z$prefix")))
        vertices, centers = xyz("Vertex"), xyz("Cell")
        if !dual
            table = rows_per_entity(read_variable(ds, "verticesOnCell"), length(centers))
            nedges = read_variable(ds, "nEdgesOnCell")
            return plex_from_polygons([Int.(table[c, 1:nedges[c]]) for c in axes(table, 1)], vertices)
        end
        table = rows_per_entity(read_variable(ds, "cellsOnVertex"), length(vertices))
        triangles = [Int.(table[v, :]) for v in axes(table, 1) if all(c -> 1 <= c <= length(centers), table[v, :])]
        return plex_from_polygons(triangles, centers)
    end
end

"""
    read_grid(filename; dual=false)

Read an ICON or MPAS grid file, detected from its variables.
"""
function read_grid(filename::AbstractString; dual::Bool=false)
    kind = NCDataset(ds -> haskey(ds, "vertex_of_cell") ? :icon : haskey(ds, "verticesOnCell") ? :mpas : :unknown,
                     filename)
    kind == :icon && return read_icon_grid(filename; dual)
    kind == :mpas && return read_mpas_grid(filename; dual)
    throw(ArgumentError("$filename is neither an ICON nor an MPAS grid file"))
end
