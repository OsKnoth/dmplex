const HEALPIX_JRLL = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
const HEALPIX_JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)

"""
    healpix_point(x, y, face)

Point on the unit sphere at fractional coordinates `x, y ∈ [0, 1]` of HEALPix
base face `face ∈ 0:11` (continuous version of `xyf2loc` in healpix_base).
"""
function healpix_point(x::Real, y::Real, face::Integer)
    jr = HEALPIX_JRLL[face + 1] - x - y
    if jr < 1      # north polar cap
        nr = jr
        z = 1 - nr^2 / 3
    elseif jr > 3  # south polar cap
        nr = 4 - jr
        z = nr^2 / 3 - 1
    else           # equatorial belt
        nr = 1.0
        z = (2 - jr) * 2 / 3
    end
    t = mod(HEALPIX_JPLL[face + 1] * nr + x - y, 8)
    φ = nr < 1e-15 ? 0.0 : π / 4 * t / nr
    s = sqrt(max(0.0, 1 - z^2))
    return SVector(s * cos(φ), s * sin(φ), z)
end

"""
    create_healpix_mesh(nside)

HEALPix mesh with `12 nside²` quadrilateral cells: the corners of the HEALPix
pixels connected by great-circle arcs (as in CGDycore). Cells are ordered by
base face and then by the face coordinates `(i, j)`; with great-circle edges
the cell areas are close to, but not exactly, equal.
"""
function create_healpix_mesh(nside::Int)
    nside > 0 || throw(ArgumentError("nside must be positive"))
    vertices = SVector{3, Float64}[]
    vertex_ids = Dict{NTuple{3, Float64}, Int}()
    vertex_id(point) = get!(vertex_ids, Tuple(round.(point, digits=12) .+ 0.0)) do
        push!(vertices, point)
        length(vertices)
    end

    quads = Vector{Int}[]
    for face in 0:11
        corner = [vertex_id(healpix_point(i / nside, j / nside, face)) for i in 0:nside, j in 0:nside]
        for j in 1:nside, i in 1:nside
            push!(quads, [corner[i, j], corner[i + 1, j], corner[i + 1, j + 1], corner[i, j + 1]])
        end
    end
    return plex_from_polygons(quads, vertices)
end
