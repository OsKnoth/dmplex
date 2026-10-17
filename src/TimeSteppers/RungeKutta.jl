"Work arrays for `rk4_step!`."
struct RK4Cache{T}
    k1::Vector{T}
    k2::Vector{T}
    k3::Vector{T}
    k4::Vector{T}
    stage::Vector{T}
end

RK4Cache(y::AbstractVector{T}) where {T} = RK4Cache((similar(y) for _ in 1:5)...)

"""
    rk4_step!(y, dt, tendency!, cache, args...)

Classical fourth-order Runge–Kutta step in place; `tendency!(dy, y, args...)`
writes the right-hand side into `dy`.
"""
function rk4_step!(y::AbstractVector, dt::Real, tendency!, cache::RK4Cache, args...)
    (; k1, k2, k3, k4, stage) = cache
    tendency!(k1, y, args...)
    @. stage = y + dt / 2 * k1
    tendency!(k2, stage, args...)
    @. stage = y + dt / 2 * k2
    tendency!(k3, stage, args...)
    @. stage = y + dt * k3
    tendency!(k4, stage, args...)
    @. y += dt / 6 * (k1 + 2k2 + 2k3 + k4)
    return y
end
