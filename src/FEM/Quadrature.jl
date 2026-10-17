"Gauss–Legendre nodes and weights on [-1, 1] (Golub–Welsch), exact for degree 2n - 1."
function gauss_legendre(n::Integer)
    n >= 1 || throw(ArgumentError("need at least one point"))
    n == 1 && return [0.0], [2.0]
    β = [m / sqrt(4m^2 - 1) for m in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

function legendre_columns(x::Vector{Float64}, N::Int)
    P = zeros(length(x), N + 1)
    P[:, 1] .= 1.0
    N >= 1 && (P[:, 2] .= x)
    for k in 2:N
        P[:, k + 1] .= ((2k - 1) .* x .* P[:, k] .- (k - 1) .* P[:, k - 1]) ./ k
    end
    return P
end

"Gauss–Lobatto–Legendre nodes (ascending, including ±1) and weights, `n ≥ 2`."
function gauss_lobatto(n::Integer)
    n >= 2 || throw(ArgumentError("Gauss–Lobatto needs at least two points"))
    N = n - 1
    x = -cos.(π .* (0:N) ./ N)
    for _ in 1:100
        P = legendre_columns(x, N)
        dx = (x .* P[:, N + 1] .- P[:, N]) ./ (n .* P[:, N + 1])
        x .-= dx
        maximum(abs, dx) < 1e-15 && break
    end
    P = legendre_columns(x, N)
    return x, 2 ./ (N * n .* P[:, N + 1] .^ 2)
end

"""
    gauss_radau(n)

Gauss–Radau nodes (ascending, including the right end point `+1`) and weights
on [-1, 1], exact for degree `2n - 2`. The left rule (with `-1`) has nodes at
the roots of `P_{n-1} + P_n`; the right rule is its mirror image.
"""
function gauss_radau(n::Integer)
    n >= 1 || throw(ArgumentError("need at least one point"))
    n == 1 && return [1.0], [2.0]
    x = -cos.(2π .* (0:(n - 1)) ./ (2n - 1))  # Chebyshev–Gauss–Radau guess, x[1] = -1
    for _ in 1:100
        dx = zeros(n)
        for i in 2:n
            P, dP = legendre(n, x[i])
            dx[i] = (P[n] + P[n + 1]) / (dP[n] + dP[n + 1])
        end
        x .-= dx
        maximum(abs, dx) < 1e-15 && break
    end
    w = [i == 1 ? 2 / n^2 : (1 - x[i]) / (n^2 * legendre(n - 1, x[i])[1][n]^2) for i in 1:n]
    return -reverse(x), reverse(w)
end

"Quadrature rule on a reference cell: `sum(w * f(ξ) for (ξ, w) in zip(points, weights))`."
struct QuadratureRule
    points::Vector{SVector{2, Float64}}
    weights::Vector{Float64}
end

"Tensor Gauss rule on the square, exact for polynomials of degree `degree`."
function quadrature_rule(::Quadrilateral, degree::Integer)
    x, w = gauss_legendre(max(1, cld(degree + 1, 2)))
    points = [SVector(x[i], x[j]) for j in eachindex(x) for i in eachindex(x)]
    weights = [w[i] * w[j] for j in eachindex(w) for i in eachindex(w)]
    return QuadratureRule(points, weights)
end

"Collapsed (Duffy) Gauss rule on the triangle, exact for degree `degree`."
function quadrature_rule(::Triangle, degree::Integer)
    a, w = gauss_legendre(max(1, cld(degree + 2, 2)))
    points = SVector{2, Float64}[]
    weights = Float64[]
    for j in eachindex(a), i in eachindex(a)
        push!(points, SVector((1 + a[i]) * (1 - a[j]) / 2 - 1, a[j]))
        push!(weights, w[i] * w[j] * (1 - a[j]) / 2)
    end
    return QuadratureRule(points, weights)
end
