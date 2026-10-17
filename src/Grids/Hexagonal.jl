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
    rings = [[cell_index(primal, c) for c in get_star(primal, v) if c in cell_range(primal)]
             for v in vertex_range(primal)]
    return plex_from_polygons(dual_polygons(rings, centers, primal.coordinates), centers)
end
