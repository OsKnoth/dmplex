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

"Work arrays for `ssprk3_step!`."
struct SSPRK3Cache{T}
    k::Vector{T}
    stage::Vector{T}
end

SSPRK3Cache(y::AbstractVector{T}) where {T} = SSPRK3Cache(similar(y), similar(y))

"""
    ssprk3_step!(y, dt, tendency!, cache, args...)

Three-stage, third-order strong-stability-preserving Runge–Kutta step
(Shu–Osher) in place.
"""
function ssprk3_step!(y::AbstractVector, dt::Real, tendency!, cache::SSPRK3Cache, args...)
    (; k, stage) = cache
    tendency!(k, y, args...)
    @. stage = y + dt * k
    tendency!(k, stage, args...)
    @. stage = 3 / 4 * y + 1 / 4 * (stage + dt * k)
    tendency!(k, stage, args...)
    @. y = 1 / 3 * y + 2 / 3 * (stage + dt * k)
    return y
end
