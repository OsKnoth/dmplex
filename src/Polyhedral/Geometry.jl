# Geometry of the cells and faces from their moments

"""
    PolyGeometry(cx::PolyComplex)

Volumes and centroids of the cells, areas, unit normals (oriented as the face, by its boundary)
and centroids of the faces of `cx`, all from the moments of degree ≤ 1 (`cell_moments`,
`face_moments`); the area vector of a non-planar face is that of its `planar_pieces`.
"""
struct PolyGeometry
    volume::Vector{Float64}
    cell_centroid::Vector{SVector{3, Float64}}
    area::Vector{Float64}
    normal::Vector{SVector{3, Float64}}
    face_centroid::Vector{SVector{3, Float64}}
end

function PolyGeometry(cx::PolyComplex)
    B = MonomialBasis(1)          # 1, y₁, y₂, y₃
    first_moments(I, center, scale) = (I[1], center + scale * SVector(I[2], I[3], I[4]) / I[1])
    nc, nf = num_poly_cells(cx), num_poly_faces(cx)
    volume, cc = zeros(nc), Vector{SVector{3, Float64}}(undef, nc)
    for c in 1:nc
        faces, _ = cell_cone(cx, c)
        center, scale = default_frame(cx.coordinates[unique(reduce(vcat, (face_loop(cx, f) for f in faces)))])
        volume[c], cc[c] = first_moments(cell_moments(cx, c, B; center, scale) ./ scale^3, center, scale)
        volume[c] *= scale^3
    end
    area, normal, fc = zeros(nf), Vector{SVector{3, Float64}}(undef, nf), Vector{SVector{3, Float64}}(undef, nf)
    for f in 1:nf
        xs = face_points(cx, f)
        center, scale = default_frame(xs)
        A = sum(P -> area_vector(P), planar_pieces([(x - center) / scale for x in xs])) * scale^2
        area[f], normal[f] = norm(A), normalize(A)
        _, fc[f] = first_moments(face_moments(cx, f, B; center, scale) ./ scale^2, center, scale)
    end
    return PolyGeometry(volume, cc, area, normal, fc)
end
