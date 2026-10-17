# Operators of the staggered kite discretization. With the contravariant Piola
# map, `φ ⋅ ∇η dA = φ̂ ⋅ ∇̂η̂ dξ`, `(∇⋅φ) η dA = (∇̂⋅φ̂) η̂ dξ` and
# `φ ⋅ n ds = φ̂ ⋅ ν̂ dt` with the scaled reference edge normal `ν̂`, so both
# operators are independent of the metric.

const KITE_PRIMAL_FACES = (1, 4)    # halves of primal edges
const KITE_INTERIOR_FACES = (2, 3)  # center–edge edges, boundaries of dual cells

"""
    assemble_kite_divergence(H, U; degree)

Weak divergence of the kite method, from the broken H(div) space `U` into the
broken H(grad) space `H`:

    B[i, j] = Σ_K ∫_K φ_j ⋅ ∇η_i - Σ_K Σ_{F ⊂ ∂K ∩ primal edges} ∫_F η_i (φ_j ⋅ n_K),

where `η_i` takes its trace from inside `K`. Since `φ ⋅ n` is continuous across
primal edges, constants are in the left null space: `onesᵀ B = 0`.
"""
assemble_kite_divergence(H::FESpace, U::FESpace; degree::Integer=2 * U.family.order + 2) =
    kite_pairing(H, U, KITE_PRIMAL_FACES, :gradient_test; degree)

"""
    assemble_kite_gradient(U, H; degree)

Weak gradient of the kite method, from `H` into `U`:

    G[j, i] = Σ_K ∫_K η_i ∇⋅φ_j - Σ_K Σ_{F ⊂ ∂K ∩ dual edges} ∫_F η_i (φ_j ⋅ n_K),

with `η_i` single-valued on the dual edges. Integration by parts shows
`G = -Bᵀ`, the discrete adjoint that makes the kite scheme energy conserving.
"""
assemble_kite_gradient(U::FESpace, H::FESpace; degree::Integer=2 * U.family.order + 2) =
    sparse(kite_pairing(H, U, KITE_INTERIOR_FACES, :divergence_test; degree)')

function kite_pairing(H::FESpace, U::FESpace, faces, volume::Symbol; degree::Integer)
    H.dm === U.dm || throw(ArgumentError("spaces live on different meshes"))
    rows, cols, vals = Int[], Int[], Float64[]
    tables = Dict{Int, Any}()
    for cell in cell_range(H.dm)
        elh, elu = reference_element(H, cell), reference_element(U, cell)
        shape = elh.shape
        rule, φh, ∇φh, φu, ∇φu, edge_tables = get!(tables, get_cone_size(H.dm, cell)) do
            rule = quadrature_rule(shape, degree)
            t, wt = gauss_legendre(cld(degree + 1, 2))
            edges = map(faces) do e
                points = [edge_point(shape, e, ti) for ti in t]
                (wt, first(tabulate(elh, points)), first(tabulate(elu, points)), edge_normal(shape, e))
            end
            (rule, tabulate(elh, rule.points)..., tabulate(elu, rule.points)..., edges)
        end
        ih, _ = cell_dofs(H, cell)
        iu, su = cell_dofs(U, cell)
        nh, nu = length(ih), length(iu)
        local_matrix = zeros(nh, nu)
        for (q, w) in enumerate(rule.weights), b in 1:nu, a in 1:nh
            local_matrix[a, b] += w * (volume == :gradient_test ?
                φu[b, 1, q] * ∇φh[a, 1, 1, q] + φu[b, 2, q] * ∇φh[a, 1, 2, q] :
                φh[a, 1, q] * (∇φu[b, 1, 1, q] + ∇φu[b, 2, 2, q]))
        end
        for (wt, ηe, φe, ν) in edge_tables, (q, w) in enumerate(wt), b in 1:nu, a in 1:nh
            local_matrix[a, b] -= w * ηe[a, 1, q] * (φe[b, 1, q] * ν[1] + φe[b, 2, q] * ν[2])
        end
        for b in 1:nu, a in 1:nh
            push!(rows, ih[a]); push!(cols, iu[b]); push!(vals, su[b] * local_matrix[a, b])
        end
    end
    return sparse(rows, cols, vals, num_dofs(H), num_dofs(U))
end
