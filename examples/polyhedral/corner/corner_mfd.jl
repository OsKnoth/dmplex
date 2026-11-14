# subface unknowns with the cell-wise mimetic inner product M_C = R Rᵀ/|C| + γ (I - N (NᵀN)⁻¹ Nᵀ)
# (consistent on any polygon, SPD, not local); pressure Bᵀ, Coriolis and advection as cell forces
# projected by R (adjoint of the cell vector Rᵀ U / |C|, energy-neutral Coriolis)
include("corner_sw.jl")

struct MFDSW
    base::CornerSW
    Mf
    R::Vector{Tuple{Vector{Int}, Matrix{Float64}}}   # per cell: subfaces, 3 × subfaces (columns σ|F_s|(x_s - x_C), tangential)
    up::Vector{SVector{3, Float64}}
end

function MFDSW(cx; stab=1.0, kw...)
    m = CornerSW(cx; kw...)
    geo = m.geo
    II, JJ, VV = Int[], Int[], Float64[]
    Rs = map(1:m.nc) do c
        js, σ = findnz(m.B[c, :])
        xc = geo.cell_centroid[c]
        up = normalize(xc); P = I3 - up * up'
        R = zeros(3, length(js)); N = zeros(3, length(js))
        for (i, j) in enumerate(js)
            f, v = m.sub_face[j], m.sub_vertex[j]
            xv = cx.coordinates[v] * (norm(geo.face_centroid[f]) / norm(cx.coordinates[v]))
            R[:, i] .= σ[i] .* (P * ((geo.face_centroid[f] + xv) / 2 - xc))     # σ = s |F|/2
            N[:, i] .= P * geo.normal[f]
        end
        M0 = R' * R / geo.volume[c]
        Nn = N' ; Π = I - Nn * pinv(Nn)
        M1 = stab * tr(M0) / length(js) * Π
        Mc = M0 + M1
        for a in eachindex(js), b in eachindex(js)
            push!(II, js[a]); push!(JJ, js[b]); push!(VV, Mc[a, b])
        end
        (js, R)
    end
    M = sparse(II, JJ, VV, m.ns, m.ns)
    return MFDSW(m, cholesky(Symmetric((M + M') / 2)), Rs, [normalize(x) for x in geo.cell_centroid])
end
const I3 = SMatrix{3, 3}(1.0I)

function tendency!(dy, y, w::MFDSW; advection=true, Ω=7.292e-5)
    m = w.base; nc = m.nc; geo = m.geo
    h, U = view(y, 1:nc), view(y, nc + 1:length(y))
    dh, dU = view(dy, 1:nc), view(dy, nc + 1:length(dy))
    dh .= -(m.D * U)
    rhs = m.B' * (m.g / 2 .* h .^ 2)
    mc = [w.R[c][2] * U[w.R[c][1]] / geo.volume[c] for c in 1:nc]       # cell momentum (Perot over subfaces)
    t = [-(2Ω * w.up[c][3]) * cross(w.up[c], mc[c]) for c in 1:nc]
    if advection
        Uf = m.A * U; hF = m.T * h
        for d in 1:3
            q = [mc[c][d] for c in 1:nc]
            q1, q2 = m.Q1 * q, m.Q2 * q
            for c in 1:nc, (f, s) in m.cfaces[c]
                m.interior[f] || continue
                mass = geo.area[f] * Uf[f]
                t[c][d] -= s * mass / hF[f] * (mass >= 0 ? q1[f] : q2[f]) / geo.volume[c]
            end
        end
    end
    for c in 1:nc
        js, R = w.R[c]
        rhs[js] .+= R' * t[c]
    end
    dU .= w.Mf \ rhs
    return dy
end
initial_state(w::MFDSW, u, h) = initial_state(w.base, u, h)
