# Prototype: shallow water with corner (subface) unknowns on one layer of prisms
#   state [h; U], U the normal momentum on the two halves of every interior lateral face
#   (one per corner = vertical edge of the column), mass matrix M = Cᵀ W C from the corner vectors,
#   pressure the adjoint of the divergence, Coriolis at the corners, advection at the cells.
using dmplex, LinearAlgebra, SparseArrays, StaticArrays, Printf
const PH = dmplex.Polyhedral

struct CornerSW
    cx; geo; nc; ns
    sub_face::Vector{Int}          # face of every subface
    sub_vertex::Vector{Int}        # bottom vertex of every subface
    D::SparseMatrixCSC{Float64, Int}   # cells × subfaces, divergence (per volume)
    B::SparseMatrixCSC{Float64, Int}   # |C| D
    C::SparseMatrixCSC{Float64, Int}   # 2 corners × subfaces, corner vectors in (e1, e2)
    E::Vector{SMatrix{3, 2, Float64, 6}}  # frame of every corner
    ω::Vector{Float64}             # weight of every corner
    fk::Vector{Float64}            # Coriolis parameter at every corner
    corner_cell::Vector{Int}
    Mf                             # factorization of M
    A::SparseMatrixCSC{Float64, Int}   # faces(interior) × subfaces: face mean
    Rc::NTuple{3, SparseMatrixCSC{Float64, Int}}  # cells × subfaces, cell momentum
    T; Q1; Q2; fc; interior; cfaces
    g::Float64
end

function CornerSW(cx; g=9.80616, Ω=7.292e-5, degree=2)
    geo = PolyGeometry(cx)
    nc, nf = num_poly_cells(cx), num_poly_faces(cx)
    fc = face_cells(cx)
    interior = [fc[1, f] > 0 && fc[2, f] > 0 for f in 1:nf]
    r(v) = norm(cx.coordinates[v])
    bottom(f) = (l = PH.face_loop(cx, f); rmin = minimum(r, l); filter(v -> r(v) < rmin * (1 + 1e-9), l))
    sub_face, sub_vertex = Int[], Int[]
    sub = Dict{Tuple{Int, Int}, Int}()
    for f in 1:nf
        interior[f] || continue
        for v in bottom(f)
            push!(sub_face, f); push!(sub_vertex, v); sub[(f, v)] = length(sub_face)
        end
    end
    ns = length(sub_face)
    # divergence
    I, J, V = Int[], Int[], Float64[]
    for c in 1:nc, (f, s) in zip(PH.cell_cone(cx, c)...)
        interior[f] || continue
        for v in bottom(f)
            push!(I, c); push!(J, sub[(f, v)]); push!(V, s * geo.area[f] / 2)
        end
    end
    B = sparse(I, J, V, nc, ns)
    D = spdiagm(1 ./ geo.volume) * B
    # corners
    IC, JC, VC = Int[], Int[], Float64[]
    E, ω, fk, ccell = SMatrix{3, 2, Float64, 6}[], Float64[], Float64[], Int[]
    for c in 1:nc
        faces, signs = PH.cell_cone(cx, c)
        lat = [f for f in faces if interior[f]]
        bot = [f for f in faces if !interior[f] && abs(dot(geo.normal[f], normalize(geo.face_centroid[f]))) > 0.5]
        fb = bot[argmin([r(PH.face_loop(cx, f)[1]) for f in bot])]
        loop, wts = PH.face_loop(cx, fb), PH.subface_weights(cx, fb)
        for (v, wv) in zip(loop, wts)
            two = [f for f in lat if v in bottom(f)]
            length(two) == 2 || error("corner ($c, $v) has $(length(two)) faces")
            up = normalize(cx.coordinates[v])
            Q = PH.frame_with_normal(up)              # rows: e1, e2, up
            e1, e2 = SVector{3}(Q[1, :]), SVector{3}(Q[2, :])
            cross(e1, e2) ⋅ up < 0 && ((e1, e2) = (e2, e1))
            N = [dot(geo.normal[f], e) for f in two, e in (e1, e2)]   # U_f = n_f ⋅ u
            Ninv = inv(N)
            k = length(ω) + 1
            for i in 1:2, j in 1:2
                push!(IC, 2k - 2 + i); push!(JC, sub[(two[j], v)]); push!(VC, Ninv[i, j])
            end
            push!(E, hcat(e1, e2)); push!(ω, wv * geo.volume[c]); push!(fk, 2Ω * up[3]); push!(ccell, c)
        end
    end
    nk = length(ω)
    C = sparse(IC, JC, VC, 2nk, ns)
    W = spdiagm(repeat(ω, inner=2))
    M = Symmetric(C' * W * C)
    Mf = cholesky(M)
    # face mean of the subfaces, cell momentum (mean of the corner vectors weighted by ω)
    A = sparse(1:ns, sub_face, fill(0.5, ns), ns, nf)'
    Rc = ntuple(3) do d
        P = sparse(ccell, collect(1:nk), ω ./ geo.volume[ccell], nc, nk)
        Ed = sparse(repeat(1:nk, inner=2), 1:2nk, vec(reduce(hcat, [[e[d, 1], e[d, 2]] for e in E])), nk, 2nk)
        P * Ed * C
    end
    _, T = interaction_operators(cx, geo)
    Q = face_value_matrices(cx, PolyReconstruction(cx, geo, degree; basis=:tangential))
    cfaces = [collect(zip(PH.cell_cone(cx, c)...)) for c in 1:nc]
    return CornerSW(cx, geo, nc, ns, sub_face, sub_vertex, D, B, C, E, ω, fk, ccell, Mf, A, Rc, T, Q[1], Q[2],
                    fc, interior, cfaces, g)
end

"normal momentum h u⋅n at the subfaces (vertex column, mid-height of the face)"
function initial_state(m::CornerSW, u, h)
    cx, geo = m.cx, m.geo
    hc = [h(x) for x in geo.cell_centroid]
    U = map(1:m.ns) do s
        f, v = m.sub_face[s], m.sub_vertex[s]
        x = cx.coordinates[v] * (norm(geo.face_centroid[f]) / norm(cx.coordinates[v]))
        h(x) * dot(u(x), geo.normal[f])
    end
    return [hc; U]
end

function tendency!(dy, y, m::CornerSW; advection=true)
    nc = m.nc
    h, U = view(y, 1:nc), view(y, nc + 1:length(y))
    dh, dU = view(dy, 1:nc), view(dy, nc + 1:length(dy))
    dh .= -(m.D * U)
    rhs = m.B' * (m.g / 2 .* h .^ 2)
    cu = m.C * U
    # Coriolis at the corners: -f k × u, k × (u1 e1 + u2 e2) = u1 e2 - u2 e1
    cor = similar(cu)
    for k in eachindex(m.ω)
        a = m.ω[k] * m.fk[k]
        cor[2k - 1] = a * cu[2k]
        cor[2k] = -a * cu[2k - 1]
    end
    if advection
        mc = (m.Rc[1] * U, m.Rc[2] * U, m.Rc[3] * U)
        Uf = m.A * U
        hF = m.T * h
        t = zeros(3, nc)
        for d in 1:3
            q1, q2 = m.Q1 * mc[d], m.Q2 * mc[d]
            for c in 1:nc, (f, s) in m.cfaces[c]
                m.interior[f] || continue
                mass = m.geo.area[f] * Uf[f]
                t[d, c] -= s * mass / hF[f] * (mass >= 0 ? q1[f] : q2[f]) / m.geo.volume[c]
            end
        end
        for k in eachindex(m.ω)
            c, e = m.corner_cell[k], m.E[k]
            tc = SVector(t[1, c], t[2, c], t[3, c])
            cor[2k - 1] += m.ω[k] * dot(e[:, 1], tc)
            cor[2k] += m.ω[k] * dot(e[:, 2], tc)
        end
    end
    rhs .+= m.C' * cor
    dU .= m.Mf \ rhs
    return dy
end
