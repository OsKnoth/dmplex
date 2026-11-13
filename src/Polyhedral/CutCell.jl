# Cut cells: the part of a complex above a surface (POLYEDER `CutGrid`), small cut cells merged

"""
    cut_complex(cx, level; tol=1e-12) -> (complex, parent, terrain)

The part of `cx` where `level(x) > 0`, the fluid above the surface `level(x) = 0` (the orography):
the vertices with `level < 0` are removed, every edge from such a vertex to one in the fluid is
cut at the root of `level` on it (to `tol` times its length), every face with cut edges is closed
by the edge between its two cut points, and every cut cell by the face through the new edges of
its faces (the terrain face, in general not planar, oriented outward). `parent` is the cell of
`cx` of every new cell, `terrain` the terrain faces (boundary faces of the new complex). A face
crossed by the surface more than once, or a vertex with `level` zero, is an error.
"""
function cut_complex(cx::PolyComplex, level; tol::Real=1e-12)
    lv = [Float64(level(x)) for x in cx.coordinates]
    any(iszero, lv) && throw(ArgumentError("level vanishes at a vertex; shift the surface slightly"))
    fluid = lv .> 0
    # vertices: the fluid ones, then the cut points
    vnew = zeros(Int, num_poly_vertices(cx))
    coordinates = cx.coordinates[fluid]
    vnew[fluid] .= eachindex(coordinates)
    # edges: the fluid ones, the cut ones from their fluid vertex to the cut point (same orientation)
    enew, cut_point = zeros(Int, num_poly_edges(cx)), zeros(Int, num_poly_edges(cx))
    ev = Int[]
    for e in 1:num_poly_edges(cx)
        a, b = cx.edge_vertices[1, e], cx.edge_vertices[2, e]
        fluid[a] || fluid[b] || continue
        if fluid[a] && fluid[b]
            append!(ev, (vnew[a], vnew[b]))
        else
            xa, xb = cx.coordinates[a], cx.coordinates[b]
            t = edge_root(level, xa, xb, lv[a], lv[b], tol)
            push!(coordinates, xa + t * (xb - xa))
            cut_point[e] = length(coordinates)
            append!(ev, fluid[a] ? (vnew[a], cut_point[e]) : (cut_point[e], vnew[b]))
        end
        enew[e] = length(ev) ÷ 2
    end
    oriented_head(e, s) = s > 0 ? ev[2e] : ev[2e - 1]
    # faces: the kept edges, a cut face closed by the edge from the cut point where its boundary
    # leaves the fluid to the one where it enters again
    fnew, closing = zeros(Int, num_poly_faces(cx)), zeros(Int, num_poly_faces(cx))
    face_offsets, face_edges, face_signs = [1], Int[], Int[]
    for f in 1:num_poly_faces(cx)
        edges, signs = face_cone(cx, f)
        any(e -> enew[e] > 0, edges) || continue
        exits, entries = Int[], Int[]
        for (e, s) in zip(edges, signs)
            enew[e] > 0 || continue
            push!(face_edges, enew[e])
            push!(face_signs, s)
            cut_point[e] > 0 || continue
            push!(oriented_head(enew[e], s) == cut_point[e] ? exits : entries, cut_point[e])
        end
        if !isempty(exits) || !isempty(entries)
            length(exits) == length(entries) == 1 || throw(ArgumentError("face $f is crossed by the surface more than once"))
            append!(ev, (exits[1], entries[1]))
            closing[f] = length(ev) ÷ 2
            push!(face_edges, closing[f])
            push!(face_signs, 1)
        end
        push!(face_offsets, length(face_edges) + 1)
        fnew[f] = length(face_offsets) - 1
    end
    # cells: the kept faces, a cut cell closed by the terrain face, whose boundary cancels the
    # closing edges of its faces
    cell_offsets, cell_faces, cell_signs = [1], Int[], Int[]
    parent, terrain = Int[], Int[]
    for c in 1:num_poly_cells(cx)
        faces, signs = cell_cone(cx, c)
        any(f -> fnew[f] > 0, faces) || continue
        bottom_edges, bottom_signs = Int[], Int[]
        for (f, s) in zip(faces, signs)
            fnew[f] > 0 || continue
            push!(cell_faces, fnew[f])
            push!(cell_signs, s)
            closing[f] > 0 && (push!(bottom_edges, closing[f]); push!(bottom_signs, -s))
        end
        if !isempty(bottom_edges)
            append!(face_edges, bottom_edges)
            append!(face_signs, bottom_signs)
            push!(face_offsets, length(face_edges) + 1)
            push!(terrain, length(face_offsets) - 1)
            push!(cell_faces, terrain[end])
            push!(cell_signs, 1)
        end
        push!(cell_offsets, length(cell_faces) + 1)
        push!(parent, c)
    end
    new = PolyComplex(cell_offsets, cell_faces, cell_signs, face_offsets, face_edges, face_signs,
                      reshape(ev, 2, :), coordinates)
    foreach(f -> face_loop(new, f), terrain)       # closed loops, else the surface cuts a cell twice
    return new, parent, terrain
end

"Root of `level` on the segment from `xa` to `xb` (`fa`, `fb` of opposite signs) as the fraction `t` of the segment, Illinois method."
function edge_root(level, xa, xb, fa, fb, tol)
    t0, t1, f0, f1 = 0.0, 1.0, fa, fb
    t, side = 0.5, 0
    for _ in 1:200
        t = (t0 * f1 - t1 * f0) / (f1 - f0)
        ft = level(xa + t * (xb - xa))
        ft == 0 && break
        if (ft > 0) == (f1 > 0)
            t1, f1 = t, ft
            side == 1 && (f0 /= 2)
            side = 1
        else
            t0, f0 = t, ft
            side == -1 && (f1 /= 2)
            side = -1
        end
        abs(t1 - t0) <= tol && break
    end
    return t
end

"""
    merge_cells(cx, group) -> (complex, faces)

The complex of the unions of the cells of `cx` with the same `group[c]` (numbered `1:maximum(group)`):
the faces between cells of one group are removed, unused edges and vertices dropped. `faces` is the
new number of every face of `cx` (0 if removed).
"""
function merge_cells(cx::PolyComplex, group::AbstractVector{<:Integer})
    ng = maximum(group)
    fc = face_cells(cx)
    inner(f) = fc[1, f] > 0 && fc[2, f] > 0 && group[fc[1, f]] == group[fc[2, f]]
    fnew = zeros(Int, num_poly_faces(cx))
    kept = filter(!inner, 1:num_poly_faces(cx))
    fnew[kept] .= eachindex(kept)
    members = [Int[] for _ in 1:ng]
    foreach(c -> push!(members[group[c]], c), 1:num_poly_cells(cx))
    any(isempty, members) && throw(ArgumentError("every group needs a cell"))
    cell_offsets, cell_faces, cell_signs = [1], Int[], Int[]
    for cs in members
        for c in cs, (f, s) in zip(cell_cone(cx, c)...)
            fnew[f] > 0 || continue
            push!(cell_faces, fnew[f])
            push!(cell_signs, s)
        end
        push!(cell_offsets, length(cell_faces) + 1)
    end
    # edges and vertices still in use
    eused = falses(num_poly_edges(cx))
    for f in kept
        eused[face_cone(cx, f)[1]] .= true
    end
    enew = cumsum(eused) .* eused
    vused = falses(num_poly_vertices(cx))
    vused[cx.edge_vertices[:, eused]] .= true
    vnew = cumsum(vused) .* vused
    face_offsets, face_edges, face_signs = [1], Int[], Int[]
    for f in kept
        edges, signs = face_cone(cx, f)
        append!(face_edges, enew[edges])
        append!(face_signs, signs)
        push!(face_offsets, length(face_edges) + 1)
    end
    ev = vnew[cx.edge_vertices[:, eused]]
    return PolyComplex(cell_offsets, cell_faces, cell_signs, face_offsets, face_edges, face_signs, ev,
                       cx.coordinates[vused]), fnew
end

"""
    merge_small_cells(cx, reference; fraction=0.5, geometry=PolyGeometry(cx), passes=5) -> (complex, group)

Cells smaller than `fraction` times their `reference` volume (the uncut cell, `volume(parent)`)
merged with the neighbour across their largest interior face, repeated for unions still too small;
`group` is the new cell of every cell of `cx`. The merged cell has the reference volume of its
largest member.
"""
function merge_small_cells(cx::PolyComplex, reference::AbstractVector{<:Real}; fraction::Real=0.5,
                           geometry::PolyGeometry=PolyGeometry(cx), passes::Integer=5)
    nc = num_poly_cells(cx)
    group = collect(1:nc)
    current, ref, geo = cx, Float64.(reference), geometry
    for _ in 1:passes
        small = findall(c -> geo.volume[c] < fraction * ref[c], 1:num_poly_cells(current))
        isempty(small) && break
        fc = face_cells(current)
        up = collect(1:num_poly_cells(current))          # union-find of the merge pairs
        root(c) = (while up[c] != c; c = up[c]; end; c)
        for c in small
            best, area = 0, 0.0
            for f in cell_cone(current, c)[1]
                other = fc[1, f] == c ? fc[2, f] : fc[1, f]
                other > 0 && geo.area[f] > area && ((best, area) = (other, geo.area[f]))
            end
            best > 0 && (up[root(c)] = root(best))
        end
        roots = [root(c) for c in eachindex(up)]
        number = zeros(Int, length(roots))
        number[unique(roots)] .= 1:length(unique(roots))
        g = number[roots]
        newref = zeros(maximum(g))
        for c in eachindex(g)
            newref[g[c]] = max(newref[g[c]], ref[c])
        end
        current, _ = merge_cells(current, g)
        group, ref, geo = g[group], newref, PolyGeometry(current)
    end
    return current, group
end
