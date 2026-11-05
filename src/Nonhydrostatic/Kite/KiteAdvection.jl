# Momentum advection of the kite model in DG advective form (`momentum=:dg`): the broken
# velocity is advected like a DG field over all kite faces, so that the operator linearized
# around a uniform flow is skew in the kinetic energy (docs/KITE3D_PLAN.md, supercritical flow).

"""
    kite_face_pairs(sw, edges) -> (; cK, qK, fK, cE, qE, fE)

The quadrature points of the faces `edges` (local edges of the reference kite) of
the local kites in pairs: point `qK` of edge `fK` of kite `cK` coincides with
point `qE` of edge `fE` of the neighbour kite `cE`. Serial only.
"""
function kite_face_pairs(sw, edges)
    kites = sw.kites
    shape = reference_element(sw.H, first(cell_range(kites))).shape
    t, _ = gauss_legendre(isqrt(length(sw.quadrature.weights)))
    lattice = face_lattice(kites)
    open = Dict{NTuple{3, Int}, NTuple{3, Int}}()
    pairs = NTuple{6, Int}[]
    for (c, kite) in enumerate(sw.distribution.kite_ids)
        cmap = CellMap(kites, first(cell_range(kites)) + kite - 1)
        for f in edges, q in eachindex(t)
            key = lattice(first(map_point(cmap, edge_point(shape, f, t[q]))))
            other = pop!(open, key, nothing)
            other === nothing ? (open[key] = (c, q, f)) : push!(pairs, (other..., c, q, f))
        end
    end
    isempty(open) || error("unpaired kite-face points: $(length(open))")
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
    momentum_tables(sw) -> (; ∇φu, dP, faces, pairs)

Tables of the DG momentum advection (`momentum_advection!`) for the kite spaces
`sw`: reference gradients of the velocity basis at the volume points, the
derivatives `∂P/∂ξ_k` of the contravariant Piola factor there (central
differences of the cell map), and for all four kite edges the reference
tables, the Piola factor at the edge points of every local kite and the pairs
of coinciding edge points.
"""
function momentum_tables(sw)
    kites, Q = sw.kites, sw.quadrature
    c1 = first(cell_range(kites))
    elh, elu = reference_element(sw.H, c1), reference_element(sw.U, c1)
    shape = elh.shape
    _, ∇φu = tabulate(elu, Q.points)
    t, wt = gauss_legendre(isqrt(length(Q.weights)))
    ids = sw.distribution.kite_ids
    piola(cmap, ξ) = (r = map_point(cmap, ξ); SMatrix{3, 2}(r[2]) / r[3])
    h = 1e-3    # Richardson extrapolation of central differences: O(h⁴)
    dP = Array{SMatrix{3, 2, Float64, 6}}(undef, length(Q.points), length(ids), 2)
    central(cmap, ξ, e) = (piola(cmap, ξ + e) - piola(cmap, ξ - e)) / (2 * norm(e))
    faces = map(1:4) do e
        points = [edge_point(shape, e, ti) for ti in t]
        (weights=wt, η=first(tabulate(elh, points)), φ=first(tabulate(elu, points)), ν=edge_normal(shape, e),
         P=Matrix{SMatrix{3, 2, Float64, 6}}(undef, length(t), length(ids)))
    end
    for (c, kite) in enumerate(ids)
        cmap = CellMap(kites, c1 + kite - 1)
        for (q, ξ) in enumerate(Q.points), k in 1:2
            e = SVector(k == 1 ? h : 0.0, k == 2 ? h : 0.0)
            dP[q, c, k] = (4 * central(cmap, ξ, e) - central(cmap, ξ, 2e)) / 3
        end
        for (e, face) in enumerate(faces), q in eachindex(t)
            face.P[q, c] = piola(cmap, edge_point(shape, e, t[q]))
        end
    end
    return (∇φu=∇φu, dP=dP, faces=faces, pairs=kite_face_pairs(sw, 1:4))
end

"""
    momentum_advection!(tu, sw, T, u, ρ, α; a=u)

The momentum advection `(ρ a ⋅ ∇) u` of one layer in DG advective form, tested
with every velocity basis function `v` (owned rows of `tu`):

    tu_v = Σ_K ∫_K ρ v ⋅ (a ⋅ ∇) u - Σ_F ∫_F ρ̂ {a ⋅ n_K} [u] ⋅ {v} + α Σ_F ∫_F ρ̂ |{a ⋅ n}|/2 [u] ⋅ [v],

over all kite faces `F` (`[u] = u^K - u^E`; across primal faces only the
tangential part jumps, across dual faces the whole velocity), `ρ̂` the mean of `ρ`.
For a uniform `ρ` and a divergence-free `a` the central part is skew in `u`,
`tu(u; v) = -tu(v; u)`; the penalty (`α = 1`, upwind) only removes energy. The
volume term is metric free except for the derivative of the Piola factor:
`(a ⋅ ∇) u dA = Σ_k â_k ∂u/∂ξ_k dξ`. `T` are the `momentum_tables`. Serial CPU.
"""
function momentum_advection!(tu, sw, T, u, ρ, α; a=u)
    Q, ih, iu, su = sw.quadrature, sw.ih, sw.iu, sw.su
    φu, φh, ∇φu = sw.φu, sw.φh, T.∇φu
    fill!(tu, 0.0)
    @inbounds for c in axes(iu, 2), (q, wq) in enumerate(Q.weights)
        P = Q.piola[q, c]
        â = ShallowWater.gather_vector(a, φu, iu, su, c, q)
        û = ShallowWater.gather_vector(u, φu, iu, su, c, q)
        ∂û = (ShallowWater.gather_vector(u, view(∇φu, :, :, 1, :), iu, su, c, q),
              ShallowWater.gather_vector(u, view(∇φu, :, :, 2, :), iu, su, c, q))
        adv = â[1] * (T.dP[q, c, 1] * û + P * ∂û[1]) + â[2] * (T.dP[q, c, 2] * û + P * ∂û[2])
        g = P' * (wq * ShallowWater.gather_scalar(ρ, φh, ih, c, q) * adv)
        for b in axes(iu, 1)
            tu[iu[b, c]] += su[b, c] * (φu[b, 1, q] * g[1] + φu[b, 2, q] * g[2])
        end
    end
    p = T.pairs
    @inbounds for i in eachindex(p.cK)
        cK, qK, cE, qE = p.cK[i], p.qK[i], p.cE[i], p.qE[i]
        fK, fE = T.faces[p.fK[i]], T.faces[p.fE[i]]
        an = (dot(ShallowWater.gather_vector(a, fK.φ, iu, su, cK, qK), fK.ν) -
              dot(ShallowWater.gather_vector(a, fE.φ, iu, su, cE, qE), fE.ν)) / 2
        jump = fK.P[qK, cK] * ShallowWater.gather_vector(u, fK.φ, iu, su, cK, qK) -
               fE.P[qE, cE] * ShallowWater.gather_vector(u, fE.φ, iu, su, cE, qE)
        ρ̂ = (ShallowWater.gather_scalar(ρ, fK.η, ih, cK, qK) + ShallowWater.gather_scalar(ρ, fE.η, ih, cE, qE)) / 2
        s = fK.weights[qK] * ρ̂
        gK = fK.P[qK, cK]' * (s * (-an / 2 + α * abs(an) / 2) * jump)
        gE = fE.P[qE, cE]' * (s * (-an / 2 - α * abs(an) / 2) * jump)
        for b in axes(iu, 1)
            tu[iu[b, cK]] += su[b, cK] * (fK.φ[b, 1, qK] * gK[1] + fK.φ[b, 2, qK] * gK[2])
            tu[iu[b, cE]] += su[b, cE] * (fE.φ[b, 1, qE] * gE[1] + fE.φ[b, 2, qE] * gE[2])
        end
    end
    return tu
end

"""
    weighted_mass_blocks(sw) -> (; indices, blocks, block, position)

The dual-cell blocks of the velocity mass (those of `sw.Mu_solver`) with dense
work matrices for the density-weighted mass `∫ ρ φ_a ⋅ φ_b`, and for every dof
of every local kite its block and position in the block (a kite lies in one
dual cell). Serial only.
"""
function weighted_mass_blocks(sw)
    S = sw.Mu_solver
    indices = Vector{Int}[]
    t = 1
    while t <= length(S.indices)
        push!(indices, S.indices[t:t + S.count[t] - 1])
        t += S.count[t]
    end
    block, position = zeros(Int, length(S.indices)), zeros(Int, length(S.indices))
    for (k, idx) in enumerate(indices), (p, i) in enumerate(idx)
        block[i], position[i] = k, p
    end
    iu = sw.iu
    kite_block = [block[iu[1, c]] for c in axes(iu, 2)]
    all(block[iu[b, c]] == kite_block[c] for b in axes(iu, 1), c in axes(iu, 2)) ||
        error("a kite spans two velocity blocks")
    return (indices=indices, blocks=[zeros(length(i), length(i)) for i in indices], block=kite_block,
            position=[position[iu[b, c]] for b in axes(iu, 1), c in axes(iu, 2)],
            rhs=[zeros(length(i)) for i in indices])
end

"""
    weighted_mass_solve!(du, sw, W, ρ, rhs)

`du` from `∫ ρ du ⋅ v = rhs(v)` for all velocity basis functions `v`: the
density-weighted velocity mass (the kinetic energy `∫ ρ |u|²/2` of the layer),
assembled with the rule of the nonlinear terms and factorized per dual cell.
`W` are the `weighted_mass_blocks`. Serial CPU.
"""
function weighted_mass_solve!(du, sw, W, ρ, rhs)
    Q, iu, su, φu = sw.quadrature, sw.iu, sw.su, sw.φu
    foreach(M -> fill!(M, 0.0), W.blocks)
    nb = size(iu, 1)
    v = Vector{SVector{3, Float64}}(undef, nb)
    @inbounds for c in axes(iu, 2)
        M, pos = W.blocks[W.block[c]], view(W.position, :, c)
        for q in eachindex(Q.weights)
            P, w = Q.piola[q, c], Q.dA[q, c] * ShallowWater.gather_scalar(ρ, sw.φh, sw.ih, c, q)
            for b in 1:nb
                v[b] = su[b, c] * (P * SVector(φu[b, 1, q], φu[b, 2, q]))
            end
            for b in 1:nb, a in 1:nb
                M[pos[a], pos[b]] += w * dot(v[a], v[b])
            end
        end
    end
    @inbounds for (k, idx) in enumerate(W.indices)
        r = W.rhs[k]
        for (p, i) in enumerate(idx)
            r[p] = rhs[i]
        end
        ldiv!(cholesky!(Symmetric(W.blocks[k])), r)
        for (p, i) in enumerate(idx)
            du[i] = r[p]
        end
    end
    return du
end

"""
    coriolis_momentum!(tu, sw, u, ρ)

Adds the Coriolis term `-∫ ρ f (r̂ × u) ⋅ v` of one layer, pointwise at the
quadrature points (it does no work), to the owned velocity rows `tu`.
"""
function coriolis_momentum!(tu, sw, u, ρ)
    Q, iu, su, φu = sw.quadrature, sw.iu, sw.su, sw.φu
    @inbounds for c in axes(iu, 2), q in eachindex(Q.weights)
        P = Q.piola[q, c]
        uq = P * ShallowWater.gather_vector(u, φu, iu, su, c, q)
        g = P' * (-Q.dA[q, c] * Q.coriolis[q, c] * ShallowWater.gather_scalar(ρ, sw.φh, sw.ih, c, q) *
                  cross(Q.normal[q, c], uq))
        for b in axes(iu, 1)
            tu[iu[b, c]] += su[b, c] * (φu[b, 1, q] * g[1] + φu[b, 2, q] * g[2])
        end
    end
    return tu
end

"""
    weighted_vector!(tu, sw, v, h)

Adds `∫ h v ⋅ φ` for the velocity-space field `v` and the height-space field `h`
(pointwise product at the quadrature points) to the owned velocity rows `tu`.
"""
function weighted_vector!(tu, sw, v, h)
    Q, iu, su, φu = sw.quadrature, sw.iu, sw.su, sw.φu
    @inbounds for c in axes(iu, 2), q in eachindex(Q.weights)
        P = Q.piola[q, c]
        vq = P * ShallowWater.gather_vector(v, φu, iu, su, c, q)
        g = P' * (Q.dA[q, c] * ShallowWater.gather_scalar(h, sw.φh, sw.ih, c, q) * vq)
        for b in axes(iu, 1)
            tu[iu[b, c]] += su[b, c] * (φu[b, 1, q] * g[1] + φu[b, 2, q] * g[2])
        end
    end
    return tu
end
