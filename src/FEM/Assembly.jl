"""
    assemble_mass(V; degree=2 * order + 4)

Global mass matrix `∫ φ_i ⋅ φ_j dA` of `V` on the sphere, with physical basis
functions obtained from the reference basis by the identity (H1, L2), the
contravariant Piola map (H(div)) or the covariant Piola map (H(curl)).
`degree` is the reference quadrature degree; the sphere map is not polynomial,
so the default adds some slack. Rows and columns cover the whole section.
"""
function assemble_mass(V::FESpace; degree::Integer=2 * V.family.order + 4)
    dm = V.dm
    space = function_space(V)
    nc = V.ncomp
    rows, cols, vals = Int[], Int[], Float64[]
    tables = Dict{Int, Tuple{QuadratureRule, Array{Float64, 3}}}()
    for cell in cell_range(dm)
        el = reference_element(V, cell)
        rule, φ = get!(tables, get_cone_size(dm, cell)) do
            rule = quadrature_rule(el.shape, degree)
            rule, first(tabulate(el, rule.points))
        end
        cmap = CellMap(dm, cell)
        n = num_dofs(el)
        local_mass = zeros(n, n)
        for (q, (ξ, w)) in enumerate(zip(rule.points, rule.weights))
            _, J, detJ = map_point(cmap, ξ)
            if space in (:H1, :L2)
                for b in 1:n, a in 1:n
                    local_mass[a, b] += w * detJ * φ[a, 1, q] * φ[b, 1, q]
                end
            else
                # Piola: H(div) gives JᵀJ / detJ, H(curl) gives (JᵀJ)⁻¹ detJ
                G = J' * J
                metric = space == :Hdiv ? G / detJ : inv(G) * detJ
                for b in 1:n, a in 1:n
                    φa = SVector(φ[a, 1, q], φ[a, 2, q])
                    φb = SVector(φ[b, 1, q], φ[b, 2, q])
                    local_mass[a, b] += w * dot(φa, metric * φb)
                end
            end
        end
        indices, signs = cell_dofs(V, cell)
        for b in 1:n, a in 1:n, c in 1:nc
            i, j = (a - 1) * nc + c, (b - 1) * nc + c
            push!(rows, indices[i])
            push!(cols, indices[j])
            push!(vals, signs[i] * signs[j] * local_mass[a, b])
        end
    end
    return sparse(rows, cols, vals, num_dofs(V), num_dofs(V))
end
