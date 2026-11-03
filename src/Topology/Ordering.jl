# Renumbering of a Plex along a space-filling curve for memory locality.

"""
    hilbert_key(x, bits)

Index of the integer point `x` (three coordinates in `0:2^bits-1`) along the
three-dimensional Hilbert curve (Skilling 2004: axes to transpose, then the
bits interleaved).
"""
function hilbert_key(x::NTuple{3, UInt64}, bits::Integer)
    X = collect(x)
    M = UInt64(1) << (bits - 1)
    Q = M
    while Q > 1   # inverse undo of the excess work
        P = Q - 1
        for i in 1:3
            if X[i] & Q != 0
                X[1] ⊻= P
            else
                t = (X[1] ⊻ X[i]) & P
                X[1] ⊻= t
                X[i] ⊻= t
            end
        end
        Q >>= 1
    end
    for i in 2:3   # Gray encode
        X[i] ⊻= X[i - 1]
    end
    t, Q = UInt64(0), M
    while Q > 1
        X[3] & Q != 0 && (t ⊻= Q - 1)
        Q >>= 1
    end
    X .⊻= t
    key = UInt64(0)
    for b in bits - 1:-1:0, i in 1:3
        key = (key << 1) | ((X[i] >> b) & 1)
    end
    return key
end

"""
    hilbert_order(dm; bits=16)

The same mesh with cells numbered along the Hilbert curve of their centroids
(in the bounding box of the vertices), vertices in order of first appearance in
the new cell order and edges, as in `plex_from_polygons`, in order of first
appearance too, so that neighbouring cells, edges and vertices are close in
memory (for the gathers of the finite-volume and nonhydrostatic kernels on
GPUs). Cell `c` of the result is cell `perm[c]` of `dm` with
`perm = hilbert_permutation(dm)`; labels are carried over.
"""
function hilbert_order(dm::Plex; bits::Integer=16)
    perm = hilbert_permutation(dm; bits)
    polys = [vertex_index.(Ref(dm), get_cell_vertices(dm, first(cell_range(dm)) + c - 1)) for c in perm]
    vnew = zeros(Int, num_vertices(dm))
    n = 0
    for poly in polys, v in poly
        vnew[v] == 0 && (vnew[v] = n += 1)
    end
    n == num_vertices(dm) || throw(ArgumentError("vertices without cells"))
    coordinates = similar(dm.coordinates)
    coordinates[vnew] .= dm.coordinates
    period = dm.surface == :plane && any(!iszero, dm.period) ? dm.period : nothing
    new = plex_from_polygons([vnew[poly] for poly in polys], coordinates; period)
    isempty(dm.labels) && return new
    # labels: cells by the permutation, vertices by first appearance, edges by their vertices
    cnew = invperm(perm)
    edge_of = Dict(minmax((vertex_index(new, v) for v in get_cone(new, e))...) => e for e in edge_range(new))
    for (name, label) in dm.labels, (p, value) in label
        q = if p in cell_range(dm)
            first(cell_range(new)) + cnew[p - first(cell_range(dm)) + 1] - 1
        elseif p in vertex_range(dm)
            first(vertex_range(new)) + vnew[vertex_index(dm, p)] - 1
        else
            edge_of[minmax((vnew[vertex_index(dm, v)] for v in get_cone(dm, p))...)]
        end
        set_label!(new, name, q, value)
    end
    return new
end

"Cells of `dm` in the order of the Hilbert curve of their centroids."
function hilbert_permutation(dm::Plex; bits::Integer=16)
    centroids = [sum(cell_coordinates(dm, c)) / length(get_cone(dm, c)) for c in cell_range(dm)]
    lo = reduce((a, b) -> min.(a, b), centroids)
    hi = reduce((a, b) -> max.(a, b), centroids)
    scale = (2.0^bits - 1) ./ max.(hi .- lo, eps())
    keys = [hilbert_key(Tuple(UInt64.(round.(Int, (x .- lo) .* scale))), bits) for x in centroids]
    return sortperm(keys)
end
