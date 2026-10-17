"""
    create_hexagon_mesh(subdivisions)

Hexagonal-pentagonal mesh obtained as the dual of the icosahedral mesh with
the same number of subdivisions: triangle centers become vertices and every
triangle vertex becomes a polygon.
"""
function create_hexagon_mesh(subdivisions::Int)
    primal = create_spherical_mesh(subdivisions)

    centers = [normalize(sum(get_coordinates(primal, v) for v in get_cell_vertices(primal, c)))
               for c in cell_range(primal)]

    polygons = Vector{Int}[]
    for v in vertex_range(primal)
        ring = [cell_index(primal, c) for c in get_star(primal, v) if c in cell_range(primal)]
        axis = get_coordinates(primal, v)
        e1 = centers[ring[1]] - axis
        angle(k) = atan(dot(cross(e1, centers[k] - axis), axis), dot(e1, centers[k] - axis))
        push!(polygons, ring[sortperm(angle.(ring))])
    end

    return plex_from_polygons(polygons, centers)
end
