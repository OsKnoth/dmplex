"""
    create_hexagon_mesh(subdivisions; spring_beta=nothing)

Hexagonal-pentagonal mesh obtained as the Voronoi dual of the icosahedral
mesh with the same number of subdivisions: triangle circumcenters become
vertices and every triangle vertex becomes a polygon. The result is an exact
spherical Voronoi diagram whose generators are the icosahedral vertices
(optimized by spring dynamics with `spring_beta`, see
[`create_spherical_mesh`](@ref)).
"""
function create_hexagon_mesh(subdivisions::Int; spring_beta=nothing)
    primal = create_spherical_mesh(subdivisions; spring_beta)
    centers = cell_centers(primal; method=:circumcenter)
    rings = [[cell_index(primal, c) for c in get_star(primal, v) if c in cell_range(primal)]
             for v in vertex_range(primal)]
    return plex_from_polygons(dual_polygons(rings, centers, primal.coordinates), centers)
end
