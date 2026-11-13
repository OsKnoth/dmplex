# The compressible Euler equations on extruded polyhedral cells, the vertical acoustics implicit

const RD, CPD = 287.0, 1004.0
const CVD = CPD - RD
const P00, GRAVITY = 1.0e5, 9.80616
euler_pressure(Θ) = P00 * (RD * Θ / P00)^(CPD / CVD)

"""
    PolyEuler(cx, levels; radius=nothing, omega=7.292e-5, gravity=9.80616, degree=1, projection=:adjoint,
              reference_temperature=250, damping_height=Inf, damping_rate=0, oversampling=1.5,
              boundary_oversampling=2.5, geometry=PolyGeometry(cx))

The compressible Euler equations on the `extruded_complex` of the heights `levels` (cells numbered
column by column, `c = (i - 1) layers + k`), over the sphere of `radius` (`up` the radial
direction) or, with `radius = nothing`, over the plane (`up = ẑ`, no rotation). Heights are those
of the extrusion, not of the points: a cell is at the middle of its layer, a face between layers at
its level (the faces are planar, so the bottom of a cell sags below the sphere by `d²/2a` for cells
of size `d`, 1 km at 240 km). Gravity is constant and acts across the faces between layers, as in
the shallow atmosphere of the other dycores. The state vector is `[ρ; ρθ; M]`: the
cell averages of `ρ` and `ρθ` and the normal momentum `M = (ρu)⋅n` at the faces (zero on the
boundary faces), as the cut-cell mountain flow of the documentation:

- mass and `ρθ`: fluxes `|F| M_F` through the faces, `θ` upwind from the reconstruction of degree
  `degree` (`PolyReconstruction`);
- momentum at the faces: the advection of the cell momentum (Perot's `cell_vector_matrices` `R`,
  its velocity upwind from the reconstruction) and the Coriolis force `-f r̂ × m_C` with
  `f = 2Ω⋅r̂` (traditional approximation, as the shallow-atmosphere test cases), both forces `A_C`
  of the cells; the gradient of `p'` between the two cells along the normal and, at the faces
  between layers, the buoyancy `∓g ρ'`, with `p'`, `ρ'` the deviations from an isothermal state at
  rest at `reference_temperature`; Rayleigh damping of the vertical momentum above
  `damping_height`;
- `projection` of the cell forces to the faces: `:adjoint`, the adjoint of Perot's reconstruction,
  `[(x_F - x₁)⋅A₁ + (x₂ - x_F)⋅A₂] / n⋅(x₂ - x₁)`, or `:interpolated` along the normal. The face
  momenta have a null space that `R` cannot see (three lateral faces per cell against two
  horizontal components); its divergence-free part feels no pressure, and a force interpolated to
  the faces drives it linearly (the steady baroclinic wave drifts by 3.5 m/s per day at hexagon 4
  and blows up by day 5). The adjoint keeps the forces orthogonal to it, and the Coriolis force
  is energy neutral (`Σ_C |C| m_C⋅(-f r̂ × m_C) = 0`; interpolated it grows at the poles within
  days); first order where the line between the centroids is not normal to the face.

`step!` is the IMEX scheme ARS(2,3,2) of Ascher, Ruuth and Spiteri (1997): the vertical acoustics
and buoyancy (the fluxes through the faces between layers, their pressure gradient and buoyancy and
the damping) implicit, linearized about the state at the beginning of every step (`linearize!`;
with a linearization about the initial state the explicit remainder of the vertical sound waves
grows with the change of `ρθ`, `c √(γ Δθ/θ)`, and limits the step to 20 s at 1 km layers once `θ`
has changed by 1 %), solved column by column (a tridiagonal system for the vertical momentum),
everything else explicit; the time step is limited by the horizontal sound speed only.
"""
struct PolyEuler{W <: NamedTuple}
    complex::PolyComplex
    geometry::PolyGeometry
    layers::Int
    radius::Float64                            # NaN on a plane
    gravity::Float64
    levels::Vector{Float64}
    cell_height::Vector{Float64}
    face_height::Vector{Float64}
    fc::Matrix{Int}
    cell_offsets::Vector{Int}                  # the faces of the cells, compressed
    cell_faces::Vector{Int}
    cell_signs::Vector{Float64}
    interior::Vector{Bool}
    vertical::Vector{Bool}                     # the faces between layers
    dist::Vector{Float64}                      # n⋅(x₂ - x₁)
    w1::Vector{Float64}                        # weight of the first cell at the face
    buoyancy::Vector{Float64}                  # ±g at the faces between layers (sign of n⋅up), else 0
    damping::Vector{Float64}
    projection::Symbol
    geometric_dist::Vector{Float64}            # n⋅(x₂ - x₁) of the centroids
    up::Vector{SVector{3, Float64}}            # at the cells
    omega::SVector{3, Float64}
    R::NTuple{3, SparseMatrixCSC{Float64, Int}}   # cell momentum, transposed (rows as columns)
    Q::NTuple{2, SparseMatrixCSC{Float64, Int}}   # face values of the two sides, transposed
    reference::Vector{Float64}                 # [ρ; ρθ; 0] at rest
    pref::Vector{Float64}
    column_faces::Matrix{Int}                  # the faces between the layers of every column, bottom up
    column_sign::Matrix{Float64}               # +1 where the normal of such a face points up
    work::W
end

is_sphere(m::PolyEuler) = !isnan(m.radius)
up_direction(m::PolyEuler, x) = is_sphere(m) ? normalize(x) : SVector(0.0, 0.0, 1.0)
num_columns(m::PolyEuler) = length(m.geometry.volume) ÷ m.layers
num_states(m::PolyEuler) = 2 * length(m.geometry.volume) + length(m.geometry.area)

"Threaded `y = A x` with the transpose `At = sparse(A')`."
function tmul!(y, At::SparseMatrixCSC, x)
    rv, nz = rowvals(At), nonzeros(At)
    Threads.@threads for i in eachindex(y)
        acc = 0.0
        @inbounds for k in nzrange(At, i)
            acc += nz[k] * x[rv[k]]
        end
        y[i] = acc
    end
    return y
end

function PolyEuler(cx::PolyComplex, levels::AbstractVector{<:Real}; radius::Union{Nothing, Real}=nothing, omega::Real=7.292e-5,
                   gravity::Real=GRAVITY, degree::Integer=1, projection::Symbol=:adjoint, reference_temperature::Real=250.0,
                   damping_height::Real=Inf, damping_rate::Real=0.0, oversampling::Real=1.5,
                   boundary_oversampling::Real=2.5, geometry::PolyGeometry=PolyGeometry(cx))
    nc, nf = num_poly_cells(cx), num_poly_faces(cx)
    layers = length(levels) - 1
    nc % layers == 0 || throw(ArgumentError("the number of cells is not a multiple of the layers"))
    geo = geometry
    a = radius === nothing ? NaN : Float64(radius)
    sphere = radius !== nothing
    up(x) = sphere ? normalize(x) : SVector(0.0, 0.0, 1.0)
    fc = face_cells(cx)
    cells = [cell_cone(cx, c) for c in 1:nc]
    cell_offsets = cumsum([1; [length(cells[c][1]) for c in 1:nc]])
    cell_faces = reduce(vcat, (collect(f) for (f, _) in cells))
    cell_signs = Float64.(reduce(vcat, (collect(s) for (_, s) in cells)))
    interior = [fc[1, f] > 0 && fc[2, f] > 0 for f in 1:nf]
    vertical = [abs(dot(geo.normal[f], up(geo.face_centroid[f]))) > 0.5 for f in 1:nf]
    # the layers of a column are stacked: cell c + 1 above c
    for c in 1:nc
        above = any(f -> vertical[f] && fc[1, f] + fc[2, f] == 2c + 1 && min(fc[1, f], fc[2, f]) == c, cells[c][1])
        above == (c % layers != 0) ||
            throw(ArgumentError("the cells are not $layers layers of an extruded complex numbered by columns"))
    end
    buoyancy = [vertical[f] ? gravity * sign(dot(geo.normal[f], up(geo.face_centroid[f]))) : 0.0 for f in 1:nf]
    # heights of the extrusion: the cells at the middle of their layer, the faces between layers at the level
    layer(c) = (c - 1) % layers + 1
    cell_height = [(levels[layer(c)] + levels[layer(c) + 1]) / 2 for c in 1:nc]
    face_height = map(1:nf) do f
        c = max(fc[1, f], fc[2, f])
        vertical[f] || return cell_height[c]
        interior[f] && return levels[layer(min(fc[1, f], fc[2, f])) + 1]
        s = fc[1, f] == c ? 1 : -1                  # the boundary face points out of c: up at the top
        s * buoyancy[f] > 0 ? levels[layer(c) + 1] : levels[layer(c)]
    end
    # distances along the normal between the centroids, between layers those of the levels (all
    # columns alike, so that a horizontally uniform state at rest stays at rest)
    dist, w1 = zeros(nf), zeros(nf)
    for f in 1:nf
        interior[f] || continue
        c1, c2, n = fc[1, f], fc[2, f], geo.normal[f]
        d1, d2 = if vertical[f]
            abs(face_height[f] - cell_height[c1]), abs(cell_height[c2] - face_height[f])
        else
            dot(n, geo.face_centroid[f] - geo.cell_centroid[c1]), dot(n, geo.cell_centroid[c2] - geo.face_centroid[f])
        end
        dist[f], w1[f] = d1 + d2, d2 / (d1 + d2)
    end
    ztop = levels[end]
    damping = map(1:nf) do f
        zf = face_height[f]
        vertical[f] && zf > damping_height ? damping_rate * sin(π / 2 * (zf - damping_height) / (ztop - damping_height))^2 : 0.0
    end
    R = map(A -> sparse(A'), cell_vector_matrices(cx, geo))
    rec = PolyReconstruction(cx, geo, degree; oversampling, boundary_oversampling)
    Q = map(A -> sparse(A'), face_value_matrices(cx, rec))
    # isothermal reference state at rest, hydrostatic with g(r)
    Tr = Float64(reference_temperature)
    function rest(z)
        p = P00 * exp(-gravity * z / (RD * Tr))
        return p / (RD * Tr), P00 / RD * (p / P00)^(CVD / CPD)
    end
    ρr, Θr = first.(rest.(cell_height)), last.(rest.(cell_height))
    reference = [ρr; Θr; zeros(nf)]
    pref = euler_pressure.(Θr)
    n = 2nc + nf
    projection in (:adjoint, :interpolated) || throw(ArgumentError("projection must be :adjoint or :interpolated"))
    # the faces between the layers of every column, bottom up, and the sign of their normal
    ncol = nc ÷ layers
    column_faces, column_sign = zeros(Int, max(layers - 1, 0), ncol), zeros(max(layers - 1, 0), ncol)
    for i in 1:ncol, k in 1:(layers - 1)
        c = (i - 1) * layers + k
        f = only(f for f in cells[c][1] if vertical[f] && min(fc[1, f], fc[2, f]) == c && max(fc[1, f], fc[2, f]) == c + 1)
        column_faces[k, i], column_sign[k, i] = f, fc[1, f] == c ? 1.0 : -1.0
    end
    work = (θ=zeros(nc), pp=zeros(nc), ρp=zeros(nc), F=zeros(nf), G=zeros(nf), q1=zeros(nf), q2=zeros(nf),
            m=[zeros(nc) for _ in 1:3], v=[zeros(nc) for _ in 1:3], A=[zeros(nc) for _ in 1:3],
            cs=zeros(nc), θF=zeros(nf), tri=[zeros(max(layers - 1, 1), 4) for _ in 1:Threads.maxthreadid()])
    Ω = sphere ? SVector(0.0, 0.0, Float64(omega)) : SVector(0.0, 0.0, 0.0)
    m = PolyEuler(cx, geo, Int(layers), a, Float64(gravity), Float64.(levels), cell_height, face_height, fc, cell_offsets,
                  cell_faces, cell_signs, interior, vertical, dist, w1, buoyancy, damping, projection,
                  [interior[f] ? dot(geo.normal[f], geo.cell_centroid[fc[2, f]] - geo.cell_centroid[fc[1, f]]) : 0.0 for f in 1:nf],
                  up.(geo.cell_centroid), Ω, R, Q,
                  reference, pref, column_faces, column_sign, work)
    return linearize!(m, reference)
end

"""
    linearize!(m::PolyEuler, y) -> m

The coefficients of the implicit part `L` (linear in `y - reference`) from the state `y`: the
sound speed `γ p/ρθ` of the cells and `θ` at the faces between layers (interpolated). `step!`
calls it at the beginning of every step.
"""
function linearize!(m::PolyEuler, y)
    nc = length(m.geometry.volume)
    w = m.work
    Threads.@threads for c in 1:nc
        @inbounds w.cs[c] = CPD / CVD * euler_pressure(y[nc + c]) / y[nc + c]
    end
    Threads.@threads for i in axes(m.column_faces, 2)
        @inbounds for k in axes(m.column_faces, 1)
            f = m.column_faces[k, i]
            c1, c2 = m.fc[1, f], m.fc[2, f]
            w.θF[f] = m.w1[f] * y[nc + c1] / y[c1] + (1 - m.w1[f]) * y[nc + c2] / y[c2]
        end
    end
    return m
end

"""
    implicit_tendency!(I, r, m) -> I

`I = L r` (`r = y - reference`): at the faces between layers the upward momentum `W = σ M` gives
the fluxes `-|C|⁻¹ (A W)` of `ρ` and `-|C|⁻¹ (A θ_F W)` of `ρθ`, and
`∂ₜW = -(c_u ρθ'_u - c_l ρθ'_l)/d - g (w_l ρ'_l + w_u ρ'_u) - τ W` (`l`, `u` the cells below and above).
"""
function implicit_tendency!(I, r, m::PolyEuler)
    nc, L = length(m.geometry.volume), m.layers
    V, A, w = m.geometry.volume, m.geometry.area, m.work
    fill!(I, 0.0)
    Threads.@threads for i in axes(m.column_faces, 2)
        @inbounds for k in axes(m.column_faces, 1)
            f, σ = m.column_faces[k, i], m.column_sign[k, i]
            l, u = (i - 1) * L + k, (i - 1) * L + k + 1
            W = σ * r[2nc + f]
            I[l] -= A[f] * W / V[l]; I[u] += A[f] * W / V[u]
            I[nc + l] -= A[f] * w.θF[f] * W / V[l]; I[nc + u] += A[f] * w.θF[f] * W / V[u]
            wl = σ > 0 ? m.w1[f] : 1 - m.w1[f]
            dW = -(w.cs[u] * r[nc + u] - w.cs[l] * r[nc + l]) / m.dist[f] - m.gravity * (wl * r[l] + (1 - wl) * r[u]) -
                 m.damping[f] * W
            I[2nc + f] = σ * dW
        end
    end
    return I
end

"""
    implicit_solve!(x, r, m, s) -> x

`(1 - s L) x = r` column by column: `ρ'` and `ρθ'` eliminated, a tridiagonal system for the upward
momentum `W` of the faces between layers (Thomas algorithm), then `ρ'`, `ρθ'`; the other rows are
`x = r`.
"""
function implicit_solve!(x, r, m::PolyEuler, s)
    nc, L, g = length(m.geometry.volume), m.layers, m.gravity
    V, A, w = m.geometry.volume, m.geometry.area, m.work
    x .= r
    L == 1 && return x
    Threads.@threads :static for i in axes(m.column_faces, 2)
        T = w.tri[Threads.threadid()]          # lower, diagonal, upper, right-hand side
        @inbounds begin
            for k in 1:(L - 1)
                f, σ = m.column_faces[k, i], m.column_sign[k, i]
                l, u = (i - 1) * L + k, (i - 1) * L + k + 1
                wl = σ > 0 ? m.w1[f] : 1 - m.w1[f]
                wu, d, cl, cu = 1 - wl, m.dist[f], w.cs[l], w.cs[u]
                a, θa = A[f], A[f] * w.θF[f]
                T[k, 2] = 1 + s * m.damping[f] + s^2 * (θa * (cu / V[u] + cl / V[l]) / d + g * a * (wu / V[u] - wl / V[l]))
                if k < L - 1
                    fu = m.column_faces[k + 1, i]
                    T[k, 3] = -s^2 * (cu * A[fu] * w.θF[fu] / (d * V[u]) + g * wu * A[fu] / V[u])
                end
                if k > 1
                    fl = m.column_faces[k - 1, i]
                    T[k, 1] = -s^2 * cl * A[fl] * w.θF[fl] / (d * V[l]) + s^2 * g * wl * A[fl] / V[l]
                end
                T[k, 4] = σ * r[2nc + f] - s * (cu * r[nc + u] - cl * r[nc + l]) / d - s * g * (wu * r[u] + wl * r[l])
            end
            for k in 2:(L - 1)                 # forward elimination
                q = T[k, 1] / T[k - 1, 2]
                T[k, 2] -= q * T[k - 1, 3]
                T[k, 4] -= q * T[k - 1, 4]
            end
            T[L - 1, 4] /= T[L - 1, 2]
            for k in (L - 2):-1:1
                T[k, 4] = (T[k, 4] - T[k, 3] * T[k + 1, 4]) / T[k, 2]
            end
            for k in 1:(L - 1)                 # W, then ρ' and ρθ' of the cells below and above
                f = m.column_faces[k, i]
                W = T[k, 4]
                x[2nc + f] = m.column_sign[k, i] * W
                l, u = (i - 1) * L + k, (i - 1) * L + k + 1
                x[l] -= s * A[f] * W / V[l]; x[u] += s * A[f] * W / V[u]
                x[nc + l] -= s * A[f] * w.θF[f] * W / V[l]; x[nc + u] += s * A[f] * w.θF[f] * W / V[u]
            end
        end
    end
    return x
end

"`dq_C = -|C|⁻¹ Σ_F s_CF F_F`, gathered over the faces of every cell."
function divergence!(dq, F, m::PolyEuler)
    V = m.geometry.volume
    Threads.@threads for c in eachindex(dq)
        s = 0.0
        @inbounds for k in m.cell_offsets[c]:(m.cell_offsets[c + 1] - 1)
            s += m.cell_signs[k] * F[m.cell_faces[k]]
        end
        @inbounds dq[c] = -s / V[c]
    end
    return dq
end

"Upwind transport `G_F = F_F q_F` of the cell values `q` with the mass fluxes `F`."
function upwind_flux!(G, q, m::PolyEuler)
    w = m.work
    tmul!(w.q1, m.Q[1], q)
    tmul!(w.q2, m.Q[2], q)
    F = w.F
    Threads.@threads for f in eachindex(G)
        @inbounds G[f] = F[f] >= 0 ? F[f] * w.q1[f] : F[f] * w.q2[f]
    end
    return G
end

"The full tendency of `[ρ; ρθ; M]`."
function tendency!(dy, y, m::PolyEuler)
    nc, nf = length(m.geometry.volume), length(m.geometry.area)
    ρ, Θ, M = view(y, 1:nc), view(y, (nc + 1):(2nc)), view(y, (2nc + 1):(2nc + nf))
    dρ, dΘ, dM = view(dy, 1:nc), view(dy, (nc + 1):(2nc)), view(dy, (2nc + 1):(2nc + nf))
    w, geo = m.work, m.geometry
    ρr = view(m.reference, 1:nc)
    Threads.@threads for c in 1:nc
        @inbounds begin
            w.θ[c] = Θ[c] / ρ[c]
            w.pp[c] = euler_pressure(Θ[c]) - m.pref[c]
            w.ρp[c] = ρ[c] - ρr[c]
        end
    end
    Threads.@threads for f in 1:nf
        @inbounds w.F[f] = m.interior[f] ? geo.area[f] * M[f] : 0.0
    end
    divergence!(dρ, w.F, m)
    divergence!(dΘ, upwind_flux!(w.G, w.θ, m), m)
    # the momentum of the cells: advection of its velocity, Coriolis
    for k in 1:3
        tmul!(w.m[k], m.R[k], M)
        w.v[k] .= w.m[k] ./ ρ
    end
    for k in 1:3
        divergence!(w.A[k], upwind_flux!(w.G, w.v[k], m), m)
    end
    Ω, adjoint = m.omega, m.projection == :adjoint
    if !iszero(Ω)
        Threads.@threads for c in 1:nc
            @inbounds begin
                r = m.up[c]
                cor = -2 * dot(Ω, r) * cross(r, SVector(w.m[1][c], w.m[2][c], w.m[3][c]))   # traditional
                w.A[1][c] += cor[1]; w.A[2][c] += cor[2]; w.A[3][c] += cor[3]
            end
        end
    end
    Threads.@threads for f in 1:nf
        @inbounds if m.interior[f]
            c1, c2 = m.fc[1, f], m.fc[2, f]
            n = geo.normal[f]
            a1, a2 = m.w1[f], 1 - m.w1[f]
            A1 = SVector(w.A[1][c1], w.A[2][c1], w.A[3][c1])
            A2 = SVector(w.A[1][c2], w.A[2][c2], w.A[3][c2])
            force = if adjoint
                xF = geo.face_centroid[f]
                (dot(xF - geo.cell_centroid[c1], A1) + dot(geo.cell_centroid[c2] - xF, A2)) / m.geometric_dist[f]
            else
                dot(n, a1 * A1 + a2 * A2)
            end
            dM[f] = force - (w.pp[c2] - w.pp[c1]) / m.dist[f] - m.buoyancy[f] * (a1 * w.ρp[c1] + a2 * w.ρp[c2]) -
                    m.damping[f] * M[f]
        else
            dM[f] = 0.0
        end
    end
    return dy
end

"Work arrays of `step!`."
function zero_state(m::PolyEuler)
    n = num_states(m)
    return (y0=zeros(n), E1=zeros(n), E2=zeros(n), E3=zeros(n), I2=zeros(n), I3=zeros(n), r=zeros(n))
end

const ARS_γ = 1 - 1 / sqrt(2)
const ARS_δ = -2 * sqrt(2) / 3

"Explicit part `E = tendency - L (y - reference)` and implicit part `I = L (y - reference)`."
function split_tendency!(E, I, y, m::PolyEuler, r)
    tendency!(E, y, m)
    r .= y .- m.reference
    implicit_tendency!(I, r, m)
    E .-= I
    return E, I
end

"""
    step!(m::PolyEuler, y, dt, ys=zero_state(m))

One step of ARS(2,3,2): `Y₂ = y + dt γ (E₁ + I₂)`, `Y₃ = y + dt (δ E₁ + (1-δ) E₂ + (1-γ) I₂ + γ I₃)`,
`y ← y + dt ((1-γ)(E₂ + I₂) + γ (E₃ + I₃))`, every stage solving `(1 - γ dt L)(Y - reference)`
column by column, `L` linearized about `y`.
"""
function step!(m::PolyEuler, y, dt, ys=zero_state(m))
    γ, δ = ARS_γ, ARS_δ
    y0, r = ys.y0, ys.r
    y0 .= y
    linearize!(m, y0)
    split_tendency!(ys.E1, ys.I2, y0, m, r)                  # I₂ as scratch
    r .= y0 .- m.reference .+ dt * γ .* ys.E1
    implicit_solve!(y, r, m, γ * dt) .+= m.reference          # Y₂
    split_tendency!(ys.E2, ys.I2, y, m, r)
    r .= y0 .- m.reference .+ dt .* (δ .* ys.E1 .+ (1 - δ) .* ys.E2 .+ (1 - γ) .* ys.I2)
    implicit_solve!(y, r, m, γ * dt) .+= m.reference          # Y₃
    split_tendency!(ys.E3, ys.I3, y, m, r)
    y .= y0 .+ dt .* ((1 - γ) .* (ys.E2 .+ ys.I2) .+ γ .* (ys.E3 .+ ys.I3))
    return y
end

"""
    initial_state(m::PolyEuler, profile) -> y

The state from `profile(λ, φ, z) -> (T, p, u, v)` (longitude, latitude and height on the sphere,
`x`, `y`, `z` on the plane; `u`, `v` east and north, or along `x` and `y`): `θ` from `T` and `p` at
the cell centroids, `ρ` of the lowest cell from `p` and `T`, above in the discrete hydrostatic
balance of the model at rest (the vertical momentum tendency zero at every face between layers),
`M` from `u`, `v` at the lateral faces with `ρ` interpolated, zero between layers. The implicit part
of `m` is linearized about this state (`linearize!`).
"""
function initial_state(m::PolyEuler, profile)
    geo, nc, nf, L = m.geometry, length(m.geometry.volume), length(m.geometry.area), m.layers
    coords(x, z) = is_sphere(m) ? (atan(x[2], x[1]), asin(clamp(x[3] / norm(x), -1, 1)), z) : (x[1], x[2], z)
    ρ, Θ = zeros(nc), zeros(nc)
    θ = map(1:nc) do c
        T, p = profile(coords(geo.cell_centroid[c], m.cell_height[c])...)
        T * (P00 / p)^(RD / CPD)
    end
    below = zeros(Int, nc)                     # the face between cell c and the cell below
    for c in 1:nc
        (c - 1) % L == 0 && continue
        below[c] = only(f for f in cell_cone(m.complex, c)[1] if min(m.fc[1, f], m.fc[2, f]) == c - 1 && max(m.fc[1, f], m.fc[2, f]) == c)
    end
    ρr = view(m.reference, 1:nc)
    for i in 1:num_columns(m)
        c = (i - 1) * L + 1
        T, p = profile(coords(geo.cell_centroid[c], m.cell_height[c])...)
        ρ[c] = p / (RD * T)
        Θ[c] = ρ[c] * θ[c]
        for k in 2:L
            c, f = (i - 1) * L + k, below[(i - 1) * L + k]
            c1, c2 = m.fc[1, f], m.fc[2, f]
            other = c1 == c ? c2 : c1
            residual(r) = (pp(cc, rc) = euler_pressure(rc * θ[cc]) - m.pref[cc];
                           rr(cc, rc) = rc - ρr[cc];
                           v1 = c1 == c ? r : ρ[c1]; v2 = c2 == c ? r : ρ[c2];
                           (pp(c2, v2) - pp(c1, v1)) / m.dist[f] + m.buoyancy[f] * (m.w1[f] * rr(c1, v1) + (1 - m.w1[f]) * rr(c2, v2)))
            r = ρ[other] * θ[other] / θ[c]     # first guess: the same pressure
            for _ in 1:30
                h = 1e-7 * r
                d = (residual(r + h) - residual(r - h)) / 2h
                Δ = residual(r) / d
                r -= Δ
                abs(Δ) < 1e-14 * r && break
            end
            ρ[c], Θ[c] = r, r * θ[c]
        end
    end
    M = map(1:nf) do f
        (m.interior[f] && !m.vertical[f]) || return 0.0
        x, n = geo.face_centroid[f], geo.normal[f]
        _, _, u, v = profile(coords(x, m.face_height[f])...)
        e, nn = if is_sphere(m)
            east = normalize(SVector(-x[2], x[1], 0.0))
            east, cross(normalize(x), east)
        else
            SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)
        end
        c1, c2 = m.fc[1, f], m.fc[2, f]
        (m.w1[f] * ρ[c1] + (1 - m.w1[f]) * ρ[c2]) * dot(u * e + v * nn, n)
    end
    y = [ρ; Θ; M]
    linearize!(m, y)
    return y
end

"Total mass `Σ_C |C| ρ_C`."
total_mass(m::PolyEuler, y) = dot(m.geometry.volume, view(y, 1:length(m.geometry.volume)))

"Surface pressure of every column: `p` of the lowest cell extrapolated hydrostatically to the ground."
function surface_pressure(m::PolyEuler, y)
    nc = length(m.geometry.volume)
    return map(1:num_columns(m)) do i
        c = (i - 1) * m.layers + 1
        p = euler_pressure(y[nc + c])
        T = p / (RD * y[c])
        p * exp(m.gravity * (m.cell_height[c] - m.levels[1]) / (RD * T))
    end
end

"Temperature of the lowest cell of every column."
function lowest_temperature(m::PolyEuler, y)
    nc = length(m.geometry.volume)
    return [(c = (i - 1) * m.layers + 1; euler_pressure(y[nc + c]) / (RD * y[c])) for i in 1:num_columns(m)]
end

"Vertical velocity `±M_F/ρ_F` at the faces between layers (upward positive, `ρ_F` interpolated)."
function vertical_velocity(m::PolyEuler, y)
    nc = length(m.geometry.volume)
    faces = findall(f -> m.vertical[f] && m.interior[f], eachindex(m.vertical))
    return map(faces) do f
        c1, c2 = m.fc[1, f], m.fc[2, f]
        sign(m.buoyancy[f]) * y[2nc + f] / (m.w1[f] * y[c1] + (1 - m.w1[f]) * y[c2])
    end
end
