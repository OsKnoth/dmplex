module SphereMesh

using LinearAlgebra
using StaticArrays
using ..DMPlexModule

export create_spherical_mesh

const PHI = (1.0 + sqrt(5.0)) / 2.0

"""
    create_spherical_mesh(subdivisions)

Icosahedral triangle mesh on the unit sphere; every subdivision splits each
triangle into four and projects the new edge midpoints onto the sphere.
"""
function create_spherical_mesh(subdivisions::Int)
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

    for _ in 1:subdivisions
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
    end

    return plex_from_polygons([collect(t) for t in triangles], vertices)
end

end # module
