# Polynomials on the reference cells are stored as coefficient matrices
# c[i+1, j+1] in the tensor Legendre basis P_i(x) P_j(y). Products P_i P_j
# with i + j <= k span P_k and with i, j <= k span Q_k, the basis is well
# conditioned on [-1, 1]², and multiplication by x or y stays in the basis.

"Values and first derivatives of the Legendre polynomials P_0, …, P_n at `x`."
function legendre(n::Integer, x::Real)
    P = zeros(n + 1)
    dP = zeros(n + 1)
    P[1] = 1.0
    if n >= 1
        P[2] = x
        dP[2] = 1.0
    end
    for m in 1:(n - 1)
        P[m + 2] = ((2m + 1) * x * P[m + 1] - m * P[m]) / (m + 1)
        dP[m + 2] = dP[m] + (2m + 1) * P[m + 1]
    end
    return P, dP
end

"Coefficient matrix of `P_i(x) P_j(y)` with maximal degree `D` per variable."
function legendre_product(i::Integer, j::Integer, D::Integer)
    (i <= D && j <= D) || throw(ArgumentError("degree ($i, $j) exceeds $D"))
    c = zeros(D + 1, D + 1)
    c[i + 1, j + 1] = 1.0
    return c
end

"Multiply by `x`, using x P_i = ((i + 1) P_{i+1} + i P_{i-1}) / (2i + 1)."
function mul_x(c::AbstractMatrix)
    D = size(c, 1) - 1
    all(iszero, view(c, D + 1, :)) || throw(ArgumentError("degree overflow in mul_x"))
    out = zeros(size(c))
    for j in axes(c, 2), i in 0:(D - 1)
        a = c[i + 1, j]
        a == 0 && continue
        out[i + 2, j] += (i + 1) / (2i + 1) * a
        i >= 1 && (out[i, j] += i / (2i + 1) * a)
    end
    return out
end

mul_y(c::AbstractMatrix) = permutedims(mul_x(permutedims(c)))

"""
    evaluate_coefficients(coeffs, points) -> (values, gradients)

Evaluate functions given as Legendre coefficients `coeffs[i, j, component,
function]` at reference `points`. Returns `values[function, component, point]`
and `gradients[function, component, direction, point]`.
"""
function evaluate_coefficients(coeffs::AbstractArray{<:Real, 4}, points::AbstractVector{<:SVector{2}})
    D = size(coeffs, 1) - 1
    ncomp, nfun = size(coeffs, 3), size(coeffs, 4)
    values = zeros(nfun, ncomp, length(points))
    gradients = zeros(nfun, ncomp, 2, length(points))
    for (q, ξ) in enumerate(points)
        Px, dPx = legendre(D, ξ[1])
        Py, dPy = legendre(D, ξ[2])
        for b in 1:nfun, c in 1:ncomp
            v = gx = gy = 0.0
            for j in 1:(D + 1), i in 1:(D + 1)
                a = coeffs[i, j, c, b]
                a == 0 && continue
                v += a * Px[i] * Py[j]
                gx += a * dPx[i] * Py[j]
                gy += a * Px[i] * dPy[j]
            end
            values[b, c, q] = v
            gradients[b, c, 1, q] = gx
            gradients[b, c, 2, q] = gy
        end
    end
    return values, gradients
end
