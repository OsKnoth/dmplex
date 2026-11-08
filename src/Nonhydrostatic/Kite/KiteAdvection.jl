# Momentum advection of the kite model in DG advective form (`momentum=:dg`): the broken
# velocity is advected like a DG field over all kite faces, so that the operator linearized
# around a uniform flow is skew in the kinetic energy (docs/KITE3D_PLAN.md, supercritical flow).
# All terms are kernels over layers × (kites or paired face points) on the model's backend; the
# fields are (layers, dofs) arrays (layers first), a vector is one layer.

"""
    kite_face_pairs(sw, faces) -> (; cK, qK, fK, cE, qE, fE)

The points of the faces `faces` (local edges of the reference kite, each with the
points `t` of `face_points(f)`) of the local kites in pairs: point `qK` of face
`fK` of kite `cK` coincides with point `qE` of face `fE` of the neighbour kite
`cE`. Points without a local neighbour (the outer edge of the ghost cells of a
partition) are left out: they only touch ghost rows.
"""
function kite_face_pairs(sw, edges, t)
    kites = sw.kites
    shape = reference_element(sw.H, first(cell_range(kites))).shape
    lattice = face_lattice(kites)
    open = Dict{NTuple{3, Int}, NTuple{3, Int}}()
    pairs = NTuple{6, Int}[]
    for (c, kite) in enumerate(sw.distribution.kite_ids)
        cmap = CellMap(kites, first(cell_range(kites)) + kite - 1)
        for (f, e) in enumerate(edges), q in eachindex(t)
            key = lattice(first(map_point(cmap, edge_point(shape, e, t[q]))))
            other = pop!(open, key, nothing)
            other === nothing ? (open[key] = (c, q, f)) : push!(pairs, (other..., c, q, f))
        end
    end
    sw.distribution.comm === nothing && !isempty(open) && error("unpaired kite-face points: $(length(open))")
    return NamedTuple{(:cK, :qK, :fK, :cE, :qE, :fE)}(Tuple([p[i] for p in pairs] for i in 1:6))
end

"""
    face_lattice(kites) -> key

Key of a point on a 1e-9 L lattice (L the largest coordinate), so that coinciding
quadrature points of neighbouring kites get the same key; wrapped on periodic
planes, where the kites are unwrapped across the seams.
"""
function face_lattice(kites)
    L = maximum(norm, kites.coordinates)
    p = kites.period
    box = [p[i] > 0 ? round(Int, 1e9 * p[i] / L) : 0 for i in 1:2]
    wrap(x, i) = i <= 2 && p[i] > 0 ? mod(round(Int, 1e9 * mod(x, p[i]) / L), box[i]) : round(Int, 1e9 * x / L)
    return x -> ntuple(i -> wrap(x[i], i), 3)
end

"""
    stacked_faces(faces, pairs, backend; P=nothing) -> (; cK, qK, fK, cE, qE, fE, weights, η, φ, ν, P)

The face tables `faces` (each with `weights`, `η`, `φ`, `ν`) stacked by face for the
kernels, `weights[q, f]`, `η[a, q, f]`, `φ[b, i, q, f]`, `ν[f]`, with the point
`pairs` and the Piola factors `P[q, c, f]` at the face points, on `backend`.
"""
function stacked_faces(faces, pairs, backend; P=nothing)
    dev(x) = to_backend(backend, x)
    weights = reduce(hcat, [Array(f.weights) for f in faces])
    η = cat([Array(f.η)[:, 1, :] for f in faces]...; dims=3)
    φ = cat([Array(f.φ) for f in faces]...; dims=4)
    ν = [SVector{2, Float64}(f.ν[1], f.ν[2]) for f in faces]
    return (; map(dev, pairs)..., weights=dev(weights), η=dev(η), φ=dev(φ), ν=dev(ν),
            P=P === nothing ? nothing : dev(P))
end

"Number of point pairs of stacked face tables."
num_pairs(F) = length(F.cK)

"""
    momentum_tables(sw, backend) -> (; ∇φu, dP, faces)

Tables of the DG momentum advection (`momentum_advection!`) for the kite spaces
`sw`: reference gradients of the velocity basis at the volume points, the
derivatives `∂P/∂ξ_k` of the contravariant Piola factor there (central
differences of the cell map), and for all four kite edges the stacked face
tables with the Piola factor at the edge points of every local kite and the
pairs of coinciding edge points.
"""
function momentum_tables(sw, backend)
    kites, Q = sw.kites, sw.quadrature
    c1 = first(cell_range(kites))
    elh, elu = reference_element(sw.H, c1), reference_element(sw.U, c1)
    shape = elh.shape
    points = Array(Q.points)
    _, ∇φu = tabulate(elu, points)
    t, wt = gauss_legendre(isqrt(length(points)))
    ids = sw.distribution.kite_ids
    piola(cmap, ξ) = (r = map_point(cmap, ξ); SMatrix{3, 2}(r[2]) / r[3])
    h = 1e-3    # Richardson extrapolation of central differences: O(h⁴)
    dP = Array{SMatrix{3, 2, Float64, 6}}(undef, length(points), length(ids), 2)
    central(cmap, ξ, e) = (piola(cmap, ξ + e) - piola(cmap, ξ - e)) / (2 * norm(e))
    faces = map(1:4) do e
        p = [edge_point(shape, e, ti) for ti in t]
        (weights=wt, η=first(tabulate(elh, p)), φ=first(tabulate(elu, p)), ν=edge_normal(shape, e))
    end
    P = Array{SMatrix{3, 2, Float64, 6}}(undef, length(t), length(ids), 4)
    for (c, kite) in enumerate(ids)
        cmap = CellMap(kites, c1 + kite - 1)
        for (q, ξ) in enumerate(points), k in 1:2
            e = SVector(k == 1 ? h : 0.0, k == 2 ? h : 0.0)
            dP[q, c, k] = (4 * central(cmap, ξ, e) - central(cmap, ξ, 2e)) / 3
        end
        for e in 1:4, q in eachindex(t)
            P[q, c, e] = piola(cmap, edge_point(shape, e, t[q]))
        end
    end
    dev(x) = to_backend(backend, x)
    return (∇φu=dev(∇φu), dP=dev(dP), faces=stacked_faces(faces, kite_face_pairs(sw, 1:4, t), backend; P))
end

# --- gathers and scatters of layered fields ------------------------------------------

"`Σ x[k, idx[a, c]] φ[a, 1, q]`: a scalar field of layer `k` at point `q` of kite `c`."
@inline function layer_scalar(x, φ, idx, c, q, k)
    s = 0.0
    @inbounds for a in axes(idx, 1)
        s += x[k, idx[a, c]] * φ[a, 1, q]
    end
    return s
end

"Reference vector `Σ sᵢ xᵢ φ̂ᵢ` of layer `k` at point `q` of kite `c` (`φ[b, i, q]`, or `φ[b, i, d, q]` with `d`)."
@inline function layer_vector(x, φ, idx, sgn, c, q, k)
    s1 = s2 = 0.0
    @inbounds for b in axes(idx, 1)
        v = sgn[b, c] * x[k, idx[b, c]]
        s1 += v * φ[b, 1, q]
        s2 += v * φ[b, 2, q]
    end
    return SVector(s1, s2)
end

@inline function layer_vector(x, φ, idx, sgn, c, d, q, k)
    s1 = s2 = 0.0
    @inbounds for b in axes(idx, 1)
        v = sgn[b, c] * x[k, idx[b, c]]
        s1 += v * φ[b, 1, d, q]
        s2 += v * φ[b, 2, d, q]
    end
    return SVector(s1, s2)
end

"Reference gradient of a height field of layer `k` at point `q` of kite `c`."
@inline function layer_gradient(x, ∇φ, idx, c, q, k)
    g1 = g2 = 0.0
    @inbounds for a in axes(idx, 1)
        v = x[k, idx[a, c]]
        g1 += v * ∇φ[a, 1, 1, q]
        g2 += v * ∇φ[a, 1, 2, q]
    end
    return SVector(g1, g2)
end

"Scalar of layer `k` at point `q` of face `f` of kite `c` (stacked `η[a, q, f]`)."
@inline function face_scalar(x, η, idx, c, q, f, k)
    s = 0.0
    @inbounds for a in axes(idx, 1)
        s += x[k, idx[a, c]] * η[a, q, f]
    end
    return s
end

"Reference vector of layer `k` at point `q` of face `f` of kite `c` (stacked `φ[b, i, q, f]`)."
@inline function face_vector(x, φ, idx, sgn, c, q, f, k)
    s1 = s2 = 0.0
    @inbounds for b in axes(idx, 1)
        v = sgn[b, c] * x[k, idx[b, c]]
        s1 += v * φ[b, 1, q, f]
        s2 += v * φ[b, 2, q, f]
    end
    return SVector(s1, s2)
end

"Atomically add `sᵢ φ̂ᵢ ⋅ g` to the rows of kite `c` in layer `k`."
@inline function scatter_layer!(rhs, g, φ, idx, sgn, c, q, k)
    @inbounds for b in axes(idx, 1)
        @atomic rhs[k, idx[b, c]] += sgn[b, c] * (φ[b, 1, q] * g[1] + φ[b, 2, q] * g[2])
    end
    return nothing
end

"Atomically add `sᵢ φ̂ᵢ ⋅ g` at point `q` of face `f` to the rows of kite `c` in layer `k`."
@inline function scatter_face!(rhs, g, φ, idx, sgn, c, q, f, k)
    @inbounds for b in axes(idx, 1)
        @atomic rhs[k, idx[b, c]] += sgn[b, c] * (φ[b, 1, q, f] * g[1] + φ[b, 2, q, f] * g[2])
    end
    return nothing
end

"A field of one layer as a (1, dofs) array (a layer of a layered field as a view of that layer), a layered field unchanged."
layered(x::AbstractVector) = reshape(x, 1, :)
layered(x::SubArray{<:Any, 1, <:AbstractMatrix, <:Tuple{Integer, Base.Slice}}) =
    view(parent(x), x.indices[1]:x.indices[1], :)
layered(x::AbstractMatrix) = x

# --- momentum advection ----------------------------------------------------------------

@kernel function momentum_volume_kernel!(tu, @Const(u), @Const(a), @Const(ρ), @Const(weights), @Const(piola),
                                         @Const(dP), @Const(φu), @Const(∇φu), @Const(iu), @Const(su), @Const(φh),
                                         @Const(ih), scale)
    k, c = @index(Global, NTuple)
    @inbounds for q in eachindex(weights)
        P = piola[q, c]
        â = layer_vector(a, φu, iu, su, c, q, k)
        û = layer_vector(u, φu, iu, su, c, q, k)
        ∂1, ∂2 = layer_vector(u, ∇φu, iu, su, c, 1, q, k), layer_vector(u, ∇φu, iu, su, c, 2, q, k)
        adv = â[1] * (dP[q, c, 1] * û + P * ∂1) + â[2] * (dP[q, c, 2] * û + P * ∂2)
        scatter_layer!(tu, P' * (scale * weights[q] * layer_scalar(ρ, φh, ih, c, q, k) * adv), φu, iu, su, c, q, k)
    end
end

@kernel function momentum_face_kernel!(tu, @Const(u), @Const(a), @Const(ρ), F, @Const(iu), @Const(su), @Const(ih),
                                       α, scale)
    k, i = @index(Global, NTuple)
    @inbounds begin
        cK, qK, fK, cE, qE, fE = F.cK[i], F.qK[i], F.fK[i], F.cE[i], F.qE[i], F.fE[i]
        PK, PE = F.P[qK, cK, fK], F.P[qE, cE, fE]
        an = (dot(face_vector(a, F.φ, iu, su, cK, qK, fK, k), F.ν[fK]) -
              dot(face_vector(a, F.φ, iu, su, cE, qE, fE, k), F.ν[fE])) / 2
        jump = PK * face_vector(u, F.φ, iu, su, cK, qK, fK, k) - PE * face_vector(u, F.φ, iu, su, cE, qE, fE, k)
        ρ̂ = (face_scalar(ρ, F.η, ih, cK, qK, fK, k) + face_scalar(ρ, F.η, ih, cE, qE, fE, k)) / 2
        s = scale * F.weights[qK, fK] * ρ̂
        scatter_face!(tu, PK' * (s * (-an / 2 + α * abs(an) / 2) * jump), F.φ, iu, su, cK, qK, fK, k)
        scatter_face!(tu, PE' * (s * (-an / 2 - α * abs(an) / 2) * jump), F.φ, iu, su, cE, qE, fE, k)
    end
end

"""
    momentum_advection!(tu, m, u, ρ, α; a=u, scale=1)

Adds the momentum advection `(ρ a ⋅ ∇) u` of every layer in DG advective form,
tested with every velocity basis function `v` and times `scale`, to `tu`:

    tu_v += scale (Σ_K ∫_K ρ v ⋅ (a ⋅ ∇) u - Σ_F ∫_F ρ̂ {a ⋅ n_K} [u] ⋅ {v} + α Σ_F ∫_F ρ̂ |{a ⋅ n}|/2 [u] ⋅ [v]),

over all kite faces `F` (`[u] = u^K - u^E`; across primal faces only the
tangential part jumps, across dual faces the whole velocity), `ρ̂` the mean of `ρ`.
For a uniform `ρ` and a divergence-free `a` the central part is skew in `u`,
`tu(u; v) = -tu(v; u)`; the penalty (`α = 1`, upwind) only removes energy. The
volume term is metric free except for the derivative of the Piola factor:
`(a ⋅ ∇) u dA = Σ_k â_k ∂u/∂ξ_k dξ`.
"""
function momentum_advection!(tu, m::KiteNonhydrostatic, u, ρ, α; a=u, scale=1.0)
    sw, T = m.horizontal, m.advection.tables
    Q = sw.quadrature
    out, tu, u, a, ρ = tu, layered(tu), layered(u), layered(a), layered(ρ)
    L = size(u, 1)
    launch!(m, momentum_volume_kernel!, (L, num_local_kites(sw.distribution)), tu, u, a, ρ, Q.weights, Q.piola, T.dP,
            sw.φu, T.∇φu, sw.iu, sw.su, sw.φh, sw.ih, scale)
    num_pairs(T.faces) > 0 &&
        launch!(m, momentum_face_kernel!, (L, num_pairs(T.faces)), tu, u, a, ρ, T.faces, sw.iu, sw.su, sw.ih, α, scale)
    return out
end

# --- Coriolis and vertical advection of u ----------------------------------------------

@kernel function momentum_source_kernel!(tu, @Const(u), @Const(ρ), @Const(w), @Const(η), @Const(dz), @Const(dzh),
                                         @Const(piola), @Const(dA), @Const(normal), @Const(coriolis), @Const(φu),
                                         @Const(iu), @Const(su), @Const(φh), @Const(ih), nlev, vertical)
    k, c = @index(Global, NTuple)
    @inbounds for q in axes(dA, 1)
        P = piola[q, c]
        uq = P * layer_vector(u, φu, iu, su, c, q, k)
        f = -dA[q, c] * coriolis[q, c] * layer_scalar(ρ, φh, ih, c, q, k) * cross(normal[q, c], uq)
        if vertical
            for j in (k, k + 1)
                2 <= j <= nlev || continue
                ρw = 0.0                # the nodal product ρ w_j, as the explicit column terms
                for a in axes(ih, 1)
                    ρw += ρ[k, ih[a, c]] * w[j, ih[a, c]] * φh[a, 1, q]
                end
                f += (-dzh[j] / (2 * dz[k]) * ρw * dA[q, c]) * (P * layer_vector(η, φu, iu, su, c, q, j))
            end
        end
        scatter_layer!(tu, P' * f, φu, iu, su, c, q, k)
    end
end

"""
    momentum_sources!(tu, m, y)

Adds the Coriolis term `-∫ ρ f (r̂ × u) ⋅ v` of every layer (pointwise at the
quadrature points, it does no work) and, with `vertical_advection`, the vertical
advection `-Σ_j dzh_j/(2 dz_k) ∫ ρ w_j η_j ⋅ v` with the shear `η_j` in
`m.work.η`, to the velocity rows `tu`.
"""
function momentum_sources!(tu, m::KiteNonhydrostatic, y)
    sw = m.horizontal
    Q = sw.quadrature
    launch!(m, momentum_source_kernel!, (num_layers(m), num_local_kites(sw.distribution)), tu, y.u, y.ρ, y.w,
            m.work.η, m.dz, m.dzh, Q.piola, Q.dA, Q.normal, Q.coriolis, sw.φu, sw.iu, sw.su, sw.φh, sw.ih,
            num_layers(m), m.vertical_advection)
    return tu
end

# --- density-weighted velocity mass --------------------------------------------------

"""
    weighted_mass_blocks(sw, nlev, backend) -> (; indices, size, matrix, rhs, first, kites, position, column,
                                                  rows, row, cols, M, r)

The dual-cell blocks of the velocity mass (those of `sw.Mu_solver`) for the
density-weighted mass `∫ ρ φ_a ⋅ φ_b` of every layer: the dofs of every block
(`indices[rhs[b] + p]`, `size[b]` of them), the offsets of its matrix in the work
array `M` and of its right-hand side in `r` (layers first, so that the threads of the
layers of one block read adjacent memory), the local kites of every block
(`kites[first[b]:first[b + 1] - 1]`; the kites of ghost dual cells belong to no
block), the position of the dofs of every kite in its block and the pattern of the
Cholesky factor `L` of every block (the union of the kite cliques and its fill): for
the row `t = rhs[b] + j` of block `b` the rows `i > j` of column `j`
(`rows[column[t]:column[t + 1] - 1]`) and the columns `l < j` of row `j`
(`cols[row[t]:row[t + 1] - 1]`).
"""
function weighted_mass_blocks(sw, nlev, backend)
    S = sw.Mu_solver
    count, rows = Array(S.count), Array(S.indices)
    starts = Int[]
    t = 1
    while t <= length(rows)
        push!(starts, t)
        t += count[t]
    end
    n = [count[s] for s in starts]
    matrix = cumsum([0; n .^ 2])[1:end - 1]
    rhs = starts .- 1
    block, position = zeros(Int, maximum(rows; init=0)), zeros(Int, maximum(rows; init=0))
    for (k, s) in enumerate(starts), p in 1:n[k]
        block[rows[s + p - 1]], position[rows[s + p - 1]] = k, p
    end
    iu = Array(sw.iu)
    get0(v, i) = i <= length(v) ? v[i] : 0
    kite_block = [get0(block, iu[1, c]) for c in axes(iu, 2)]
    all(get0(block, iu[b, c]) == kite_block[c] for b in axes(iu, 1), c in axes(iu, 2)) ||
        error("a kite spans two velocity blocks")
    members = [Int[] for _ in n]
    for (c, b) in enumerate(kite_block)
        b > 0 && push!(members[b], c)
    end
    pos = [get0(position, iu[b, c]) for b in axes(iu, 1), c in axes(iu, 2)]
    column, lrows, row, lcols = [1], Int[], [1], Int[]
    for (k, nk) in enumerate(n)
        P = falses(nk, nk)                       # pattern of the lower factor: the kite cliques and their fill
        for c in members[k], a in axes(pos, 1), b in axes(pos, 1)
            pos[a, c] > 0 && pos[b, c] > 0 && (P[pos[a, c], pos[b, c]] = true)
        end
        for j in 1:nk, i in j + 1:nk
            P[i, j] || continue
            for l in i + 1:nk
                P[l, j] && (P[l, i] = true)
            end
        end
        for j in 1:nk
            append!(lrows, (i for i in j + 1:nk if P[i, j])); push!(column, length(lrows) + 1)
            append!(lcols, (l for l in 1:j - 1 if P[j, l])); push!(row, length(lcols) + 1)
        end
    end
    dev(x) = to_backend(backend, x)
    P = Array(sw.quadrature.piola)
    return (indices=dev(rows), size=dev(n), matrix=dev(matrix), rhs=dev(rhs),
            first=dev(cumsum([1; length.(members)])), kites=dev(reduce(vcat, members; init=Int[])),
            position=dev(pos), column=dev(column), rows=dev(lrows), row=dev(row), cols=dev(lcols),
            G=dev(map(x -> SMatrix{2, 2}(x' * x), P)),
            ω=KernelAbstractions.zeros(backend, Float64, nlev, size(P)...),
            M=KernelAbstractions.zeros(backend, Float64, nlev, sum(n .^ 2; init=0)),
            r=KernelAbstractions.zeros(backend, Float64, nlev, length(rows)))
end

"`ω[k, q, c] = dA ρ`: the density times the area element at the quadrature points of every layer and kite."
@kernel function quadrature_density_kernel!(ω, @Const(ρ), @Const(dA), @Const(φh), @Const(ih))
    k, c = @index(Global, NTuple)
    @inbounds for q in axes(dA, 1)
        ω[k, q, c] = dA[q, c] * layer_scalar(ρ, φh, ih, c, q, k)
    end
end

@kernel function weighted_mass_solve_kernel!(du, M, r, @Const(rhs), W, @Const(φu), @Const(iu), @Const(su))
    k, blk = @index(Global, NTuple)
    @inbounds begin
        o, s, n = W.matrix[blk], W.rhs[blk], W.size[blk]
        A(i, j) = o + (i - 1) * n + j
        for j in 1:n                     # zero the pattern of L (the rest of the block stays zero)
            M[k, A(j, j)] = 0.0
            for e in W.column[s + j]:W.column[s + j + 1] - 1
                M[k, A(W.rows[e], j)] = 0.0
            end
        end
        for t in W.first[blk]:W.first[blk + 1] - 1      # assembly of the lower triangle over the kites of the block
            c = W.kites[t]
            for b in axes(iu, 1), a in axes(iu, 1)
                pa, pb = W.position[a, c], W.position[b, c]
                pa >= pb || continue
                x = 0.0
                for q in axes(W.G, 1)
                    v = W.G[q, c] * SVector(φu[b, 1, q], φu[b, 2, q])
                    x += W.ω[k, q, c] * (φu[a, 1, q] * v[1] + φu[a, 2, q] * v[2])
                end
                M[k, A(pa, pb)] += su[a, c] * su[b, c] * x
            end
        end
        for j in 1:n                     # Cholesky on the pattern of L, the lower factor in place
            R = W.row[s + j]:W.row[s + j + 1] - 1
            d = M[k, A(j, j)]
            for e in R
                d -= M[k, A(j, W.cols[e])]^2
            end
            d = sqrt(d)
            M[k, A(j, j)] = d
            for f in W.column[s + j]:W.column[s + j + 1] - 1
                i = W.rows[f]
                x = M[k, A(i, j)]
                for e in R
                    l = W.cols[e]
                    x -= M[k, A(i, l)] * M[k, A(j, l)]
                end
                M[k, A(i, j)] = x / d
            end
        end
        for i in 1:n                     # L y = b
            x = rhs[k, W.indices[s + i]]
            for e in W.row[s + i]:W.row[s + i + 1] - 1
                l = W.cols[e]
                x -= M[k, A(i, l)] * r[k, s + l]
            end
            r[k, s + i] = x / M[k, A(i, i)]
        end
        for i in n:-1:1                  # Lᵀ x = y
            x = r[k, s + i]
            for e in W.column[s + i]:W.column[s + i + 1] - 1
                l = W.rows[e]
                x -= M[k, A(l, i)] * r[k, s + l]
            end
            r[k, s + i] = x / M[k, A(i, i)]
            du[k, W.indices[s + i]] = r[k, s + i]
        end
    end
end

"""
    weighted_mass_solve!(du, m, ρ, rhs)

`du` from `∫ ρ du ⋅ v = rhs(v)` for all velocity basis functions `v` in every
layer: the density-weighted velocity mass (the kinetic energy `∫ ρ |u|²/2` of
the layer), assembled with the rule of the nonlinear terms and factorized per
dual cell (Cholesky on the sparse pattern of the block), one thread per layer and block.
"""
function weighted_mass_solve!(du, m::KiteNonhydrostatic, ρ, rhs)
    sw, W = m.horizontal, m.advection.mass
    Q = sw.quadrature
    out, du, ρ, rhs = du, layered(du), layered(ρ), layered(rhs)
    length(W.size) == 0 && return out
    launch!(m, quadrature_density_kernel!, (size(ρ, 1), num_local_kites(sw.distribution)), W.ω, ρ, Q.dA, sw.φh, sw.ih)
    launch!(m, weighted_mass_solve_kernel!, (size(ρ, 1), length(W.size)), du, W.M, W.r, rhs, W, sw.φu, sw.iu, sw.su)
    return out
end
