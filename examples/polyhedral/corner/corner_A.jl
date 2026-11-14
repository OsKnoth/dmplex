# corner tensors A_v = R_vᵀ N_v of the corner consistency condition: anisotropy and asymmetry
include("corner_sw.jl")
a = 6.37122e6
for (name, mesh) in [("hexagonal", create_hexagon_mesh), ("cubed", create_cubed_sphere_mesh), ("icosahedral", create_spherical_mesh)], l in (3,)
    cx = extruded_complex(ExtrudedPlex(mesh(l), 1), [0.0, 1000.0]; radius=a)
    m = CornerSW(cx)
    nk = length(m.ω); asym = 0.0; aniso = Float64[]; ratio = Float64[]; mineig = Inf
    for k in 1:nk
        c = m.corner_cell[k]
        js = findnz(m.C[2k-1, :])[1]
        Ninv = Matrix(m.C[2k-1:2k, js]); N = inv(Ninv)
        e = m.E[k]
        R = zeros(2, 2)                      # rows: subfaces, cols: e1, e2
        for (i, j) in enumerate(js)
            f, v = m.sub_face[j], m.sub_vertex[j]
            xv = cx.coordinates[v] * (norm(m.geo.face_centroid[f]) / norm(cx.coordinates[v]))
            d = (m.geo.face_centroid[f] + xv) / 2 - m.geo.cell_centroid[c]
            R[i, :] .= m.B[c, j] .* (e' * d)
        end
        A = R' * N
        asym = max(asym, norm(A - A') / norm(A))
        λ = eigvals(Symmetric((A + A') / 2))
        push!(aniso, λ[2] / λ[1]); push!(ratio, sum(λ) / 2 / m.ω[k]); mineig = min(mineig, λ[1] / m.ω[k])
    end
    @printf("%-11s %d: asym max %.2e, eig ratio %.3f..%.3f, mean eig / ω %.3f..%.3f, min eig/ω %.3f\n", name, l, asym,
            extrema(aniso)..., extrema(ratio)..., mineig)
end
