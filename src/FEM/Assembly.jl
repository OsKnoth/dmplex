"""
    assemble_mass(V; degree, lumped=false)

Global mass matrix `∫ φ_i ⋅ φ_j dA` of `V` on the sphere, with physical basis
functions obtained from the reference basis by the identity (H1, L2), the
contravariant Piola map (H(div)) or the covariant Piola map (H(curl)).
`degree` is the reference quadrature degree; the sphere map is not polynomial,
so the default adds some slack. With `lumped=true` the element nodes and their
`nodal_weights` serve as quadrature, which makes the matrix diagonal (nodal
elements on quadrilaterals). Rows and columns cover the whole section.
"""
function assemble_mass(V::FESpace; degree::Integer=default_degree(V), lumped::Bool=false)
    lumped && return assemble_lumped_mass(V)
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
        cmap = CellMap(V, cell)
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

function assemble_lumped_mass(V::FESpace)
    diagonal = zeros(num_dofs(V))
    nc = V.ncomp
    for cell in cell_range(V.dm)
        el = reference_element(V, cell)
        isempty(el.nodal_weights) && throw(ArgumentError("$(el.family) on $(el.shape) has no lumped mass"))
        cmap = CellMap(V, cell)
        indices, _ = cell_dofs(V, cell)
        for (a, (ξ, w)) in enumerate(zip(el.nodes, el.nodal_weights)), c in 1:nc
            diagonal[indices[(a - 1) * nc + c]] += w * map_point(cmap, ξ)[3]
        end
    end
    n = num_dofs(V)
    used = findall(!iszero, diagonal)
    return sparse(used, used, diagonal[used], n, n)
end

"""
    cell_basis(V, cell, rule, values, gradients)

Physical basis of `V` on `cell` at the points of `rule`, with the orientation
signs of `cell_dofs` already applied. Returns a named tuple with the global
`indices`, physical `points`, area weights `dA = w detJ`, the basis values
`φ[dof, point]` (numbers for H1/L2, Cartesian vectors for H(div)/H(curl)), the
unit surface `normals`,
for H(div) the surface divergence `div[dof, point]` and for H1 the surface
gradient `grad[dof, point]` (`J (JᵀJ)⁻¹ ∇̂φ̂`).
"""
function cell_basis(V::FESpace, cell::Integer, rule::QuadratureRule, values, gradients)
    V.ncomp == 1 || throw(ArgumentError("cell_basis supports scalar or intrinsically vector spaces only"))
    cmap = CellMap(V, cell)
    indices, signs = cell_dofs(V, cell)
    n, nq = length(indices), length(rule.points)
    space = function_space(V)
    points = Vector{SVector{3, Float64}}(undef, nq)
    normals = Vector{SVector{3, Float64}}(undef, nq)
    dA = zeros(nq)
    φ = space in (:H1, :L2) ? zeros(n, nq) : Matrix{SVector{3, Float64}}(undef, n, nq)
    div = zeros(n, space == :Hdiv ? nq : 0)
    grad = Matrix{SVector{3, Float64}}(undef, n, space == :H1 ? nq : 0)
    for (q, (ξ, w)) in enumerate(zip(rule.points, rule.weights))
        x, J, detJ = map_point(cmap, ξ)
        points[q] = x
        normals[q] = surface_normal(J)
        dA[q] = w * detJ
        for a in 1:n
            if space in (:H1, :L2)
                φ[a, q] = values[a, 1, q]
                space == :H1 && (grad[a, q] = piola_hcurl(J, SVector(gradients[a, 1, 1, q], gradients[a, 1, 2, q])))
                continue
            end
            v̂ = SVector(values[a, 1, q], values[a, 2, q])
            if space == :Hdiv
                φ[a, q] = signs[a] * piola_hdiv(J, detJ, v̂)
                div[a, q] = signs[a] * (gradients[a, 1, 1, q] + gradients[a, 2, 2, q]) / detJ
            else
                φ[a, q] = signs[a] * piola_hcurl(J, v̂)
            end
        end
    end
    return (; indices, points, normals, dA, φ, div, grad)
end

"Tabulated reference basis of `V` for the quadrature rule of each cell shape."
struct Tabulation
    space::FESpace
    degree::Int
    tables::Dict{Int, Tuple{QuadratureRule, Array{Float64, 3}, Array{Float64, 4}}}
end

Tabulation(V::FESpace, degree::Integer) = Tabulation(V, Int(degree), Dict())

function cell_basis(T::Tabulation, cell::Integer)
    V = T.space
    rule, values, gradients = get!(T.tables, get_cone_size(V.dm, cell)) do
        el = reference_element(V, cell)
        rule = quadrature_rule(el.shape, T.degree)
        (rule, tabulate(el, rule.points)...)
    end
    return cell_basis(V, cell, rule, values, gradients)
end

"Physical bases of all cells of `V` (indexed by `cell_index`), for repeated nonlinear evaluations."
precompute_basis(V::FESpace, degree::Integer) =
    (T = Tabulation(V, degree); [cell_basis(T, c) for c in cell_range(V.dm)])

default_degree(spaces::FESpace...) = 2 * maximum(V.family.order for V in spaces) + 4

"""
    assemble_bilinear(kernel, test, trial; degree)

Sparse matrix `A[i, j] = Σ_cells Σ_q kernel(t, u, a, b, q)` where `t` and `u`
are the `cell_basis` tuples of the test and trial space, `a`, `b` local dofs
and `q` the quadrature point. Both spaces must live on the same mesh.
"""
function assemble_bilinear(kernel, test::FESpace, trial::FESpace; degree::Integer=default_degree(test, trial))
    test.dm === trial.dm || throw(ArgumentError("test and trial spaces live on different meshes"))
    test.geometry == trial.geometry || throw(ArgumentError("test and trial spaces use different geometries"))
    Tt, Tu = Tabulation(test, degree), Tabulation(trial, degree)
    rows, cols, vals = Int[], Int[], Float64[]
    for cell in cell_range(test.dm)
        t, u = cell_basis(Tt, cell), cell_basis(Tu, cell)
        for b in eachindex(u.indices), a in eachindex(t.indices)
            push!(rows, t.indices[a])
            push!(cols, u.indices[b])
            push!(vals, sum(kernel(t, u, a, b, q) for q in eachindex(t.dA)))
        end
    end
    return sparse(rows, cols, vals, num_dofs(test), num_dofs(trial))
end

"Weak divergence `D[i, j] = ∫ ψ_i ∇⋅φ_j dA` from H(div) space `U` into scalar space `H`."
function assemble_divergence(H::FESpace, U::FESpace; kwargs...)
    function_space(U) == :Hdiv || throw(ArgumentError("trial space must be H(div)"))
    return assemble_bilinear((t, u, a, b, q) -> t.φ[a, q] * u.div[b, q] * t.dA[q], H, U; kwargs...)
end

"""
    assemble_coriolis(U, f)

Coriolis matrix `C[i, j] = ∫ f(x) φ_i ⋅ (r̂ × φ_j) dA` for a vector space `U`,
with the element normal `r̂` and `f` evaluated at the surface point;
antisymmetric, so it does no work.
"""
assemble_coriolis(U::FESpace, f; kwargs...) =
    assemble_bilinear(U, U; kwargs...) do t, u, a, b, q
        f(surface_point(U.dm, t.points[q])) * dot(t.φ[a, q], cross(t.normals[q], u.φ[b, q])) * t.dA[q]
    end

"Load vector `b[i] = ∫ f(x) ⋅ φ_i dA` (`f` scalar or Cartesian vector valued)."
function assemble_load(V::FESpace, f; degree::Integer=default_degree(V))
    T = Tabulation(V, degree)
    b = zeros(num_dofs(V))
    for cell in cell_range(V.dm)
        t = cell_basis(T, cell)
        for a in eachindex(t.indices), q in eachindex(t.dA)
            b[t.indices[a]] += dot(f(t.points[q]), t.φ[a, q]) * t.dA[q]
        end
    end
    return b
end

"L2 projection of `f` onto `V`, using the cached mass solver."
project(V::FESpace, f; degree::Integer=default_degree(V)) = mass_solver(V; degree) \ assemble_load(V, f; degree)
