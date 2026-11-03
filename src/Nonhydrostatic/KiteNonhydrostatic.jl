# Nonhydrostatic model on the extruded kite mesh: the spaces of the 2D kite model times the
# layers of a vertical grid (docs/KITE3D_PLAN.md). Step 1: spaces, state, projections and the
# mass divergence; step 2: the vertically implicit column solve; step 3: the horizontal terms
# per layer and the HEVI step.

"""
    KiteNonhydrostatic(primal, vgrid, k=1; height_order=k + 1, radius=6.37122e6, omega=7.292e-5,
                       beta=0.65, damping_height=Inf, damping_rate=0, vertical_theta=:upwind,
                       vertical_mass=:consistent, backend=CPU())

Dry compressible Euler equations on the extruded kite mesh of `primal` with
the spaces of `NonlinearKiteShallowWater(primal, k)` (its `horizontal` part, on
the unit sphere) times the layers of `vgrid`, lowest vertical order and
Lorenz staggering:

| field | space | array |
|---|---|---|
| ρ, ρθ | `KiteRadau(height_order)` ⊗ DG0 | (height dofs, nlev) |
| w     | `KiteRadau(height_order)` ⊗ CG1 | (height dofs, nlev + 1) |
| u_h   | `KiteBrokenRT(k)` ⊗ DG0         | (velocity dofs, nlev) |

The Radau nodes of the height space are its dofs and quadrature points, so its
mass matrix is diagonal and every height dof carries an independent column for
the vertically implicit terms (`num_columns`) at `column_points`. They are
solved by the kernels of `TRiSKNonhydrostatic` (`theta_interface_kernel!`,
`vertical_solve_kernel!`, off-centring `beta`, Rayleigh damping of `w` above
`damping_height`, `vertical_theta` as there) with the consistent CG1 mass of `w`
in the vertical (`vertical_mass=:consistent`, tridiagonal) or the lumped one
(`:lumped`, pointwise at the interfaces as in `TRiSKNonhydrostatic`). Horizontal operators act layer by
layer with the 2D kite operators (scaled from the unit sphere by `radius`).
"""
struct KiteNonhydrostatic{SW, V, W, BE}
    horizontal::SW
    vgrid::VerticalGrid
    dz::V
    dzh::V
    wlow::V
    rayleigh::V          # damping rate at the interfaces
    work::W
    radius::Float64
    omega::Float64
    beta::Float64
    vertical_theta::Int  # 0 centred, 1 upwind in the flux, 2 upwind also in ∂π/∂z
    backend::BE
end

function KiteNonhydrostatic(primal::Plex, vgrid::VerticalGrid, k::Integer=1; height_order::Integer=k + 1,
                            radius::Real=6.37122e6, omega::Real=7.292e-5, beta::Real=0.65,
                            damping_height::Real=Inf, damping_rate::Real=0.0, vertical_theta::Symbol=:upwind,
                            vertical_mass::Symbol=:consistent, backend=KernelAbstractions.CPU())
    vtheta = get(Dict(:centred => 0, :upwind => 1, :iap => 2), vertical_theta) do
        throw(ArgumentError("vertical_theta must be :upwind, :iap or :centred"))
    end
    sw = NonlinearKiteShallowWater(primal, k; height_order, coriolis=x -> 2omega * x[3], backend)
    nh, nu, nlev = length(height_dofs(sw)), length(velocity_dofs(sw)), num_layers(vgrid)
    z_top = vgrid.z_half[end]
    rayleigh = map(vgrid.z_half) do z
        z > damping_height ? damping_rate * sin(π / 2 * (z - damping_height) / (z_top - damping_height))^2 : 0.0
    end
    z(n, l) = KernelAbstractions.zeros(backend, Float64, n, l)
    work = (θ=z(nh, nlev), θi=z(nh, nlev + 1), θm=z(nh, nlev), band=KernelAbstractions.zeros(backend, Float64, nh, nlev, 2),
            cprime=z(nh, nlev + 1), dprime=z(nh, nlev + 1),
            wmass=to_backend(backend, Nonhydrostatic.vertical_mass(vgrid, vertical_mass)), π0=z(nh, nlev), γ0=z(nh, nlev),
            tw=z(nh, nlev + 1), ρexpl=z(nh, nlev), ρθexpl=z(nh, nlev), π=z(nh, nlev),
            F=KernelAbstractions.zeros(backend, Float64, nu), Fθ=KernelAbstractions.zeros(backend, Float64, nu),
            g=KernelAbstractions.zeros(backend, Float64, nu), gθ=KernelAbstractions.zeros(backend, Float64, nu),
            dy=(u=z(nu, nlev), ρ=z(nh, nlev), ρθ=z(nh, nlev)),
            y0=(u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev)))
    dev(x) = to_backend(backend, x)
    return KiteNonhydrostatic(sw, vgrid, dev(vgrid.dz), dev(vgrid.dzh), dev(vgrid.wlow), dev(rayleigh), work,
                              Float64(radius), Float64(omega), Float64(beta), vtheta, backend)
end

num_layers(m::KiteNonhydrostatic) = num_layers(m.vgrid)
"Number of vertical columns: the height dofs (Radau nodes) of the horizontal space."
num_columns(m::KiteNonhydrostatic) = length(height_dofs(m.horizontal))
num_velocity_dofs(m::KiteNonhydrostatic) = length(velocity_dofs(m.horizontal))

"State `(u, w, ρ, ρθ)` of the model, zero."
function zero_state(m::KiteNonhydrostatic)
    nh, nu, nlev = num_columns(m), num_velocity_dofs(m), num_layers(m)
    z(n, l) = KernelAbstractions.zeros(m.backend, Float64, n, l)
    return (u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev))
end

"""
    project_layers(m, f; levels=:full) -> (height dofs, levels)

L2 projection of `f(x, z)` (`x` on the unit sphere, `z` the height in m) into
the height space at every layer centre (`levels=:full`) or interface
(`levels=:half`), on the model's backend.
"""
function project_layers(m::KiteNonhydrostatic, f; levels::Symbol=:full)
    z = levels == :full ? m.vgrid.z_full : levels == :half ? m.vgrid.z_half :
        throw(ArgumentError("levels must be :full or :half"))
    H = m.horizontal.H
    return to_backend(m.backend, reduce(hcat, [project(H, x -> f(x, zk)) for zk in z]))
end

"""
    project_velocity(m, v) -> (velocity dofs, nlev)

L2 projection of the horizontal velocity `v(x, z)` (a tangent 3-vector in m/s)
into the velocity space at every layer centre.
"""
function project_velocity(m::KiteNonhydrostatic, v)
    U = m.horizontal.U
    return to_backend(m.backend, reduce(hcat, [project(U, x -> v(x, zk)) for zk in m.vgrid.z_full]))
end

"""
    mass_divergence!(dρ, m, F, W)

`dρ = -∇ ⋅ (F, W)`: the horizontal mass flux `F` (velocity space, layers) by the
kite divergence of every layer, `M_h⁻¹ B F / a`, and the vertical mass flux `W`
(height space, interfaces) by `-(W_{k+1} - W_k)/dz_k` in every column. With
`W = 0` at the ground and the top, `Σ_k dz_k M_h dρ` sums to zero (mass is
conserved to round-off).
"""
function mass_divergence!(dρ, m::KiteNonhydrostatic, F, W)
    sw, a = m.horizontal, m.radius
    th = sw.row_buffers[3]
    for k in 1:num_layers(m)
        row_mul!(th, sw.B_owned, view(F, :, k), 1 / a)
        ldiv!(view(dρ, :, k), sw.Mh_solver, th)
    end
    dρ .-= (view(W, :, 2:size(W, 2)) .- view(W, :, 1:size(W, 2) - 1)) ./ m.dz'
    return dρ
end

"Diagonal of the height mass matrix (on the unit sphere): the weights of the column integrals."
column_weights(m::KiteNonhydrostatic) = m.horizontal.mass_h

"""
    column_points(m) -> Vector{SVector{3}}

Positions of the columns on the unit sphere: the Radau nodes of the height
dofs, in the local numbering of the state.
"""
function column_points(m::KiteNonhydrostatic)
    H, d = m.horizontal.H, m.horizontal.distribution
    dm = H.dm
    local_id = zeros(Int, num_dofs(H))
    local_id[d.h.global_ids] .= 1:length(d.h.global_ids)
    x = Vector{SVector{3, Float64}}(undef, length(d.h.global_ids))
    for c in cell_range(dm)
        nodes, cmap = reference_element(H, c).nodes, CellMap(dm, c)
        for (a, i) in enumerate(first(cell_dofs(H, c)))
            local_id[i] == 0 || (x[local_id[i]] = surface_point(dm, first(map_point(cmap, nodes[a]))))
        end
    end
    return x
end

"""
    initial_state(m::KiteNonhydrostatic, profile)

State from `profile(λ, φ, z) = (T, p, u, v)` as for `TRiSKNonhydrostatic`:
every column in the discrete hydrostatic balance of the column solver at its
Radau node (`balanced_column`), the velocity `u east + v north` projected
into the velocity space per layer, `w = 0`.
"""
function initial_state(m::KiteNonhydrostatic, profile)
    v, nlev = m.vgrid, num_layers(m)
    x = column_points(m)
    ρ, ρθ = zeros(length(x), nlev), zeros(length(x), nlev)
    for (i, xi) in enumerate(x)
        λ, φ = lonlat(xi)
        ρ[i, :], ρθ[i, :] = balanced_column(z -> profile(λ, φ, z), v, m.vertical_theta == 2)
    end
    u = project_velocity(m, (x, z) -> begin
        λ, φ = lonlat(x)
        east, north = east_north(x)
        _, _, ue, vn = profile(λ, φ, z)
        ue * east + vn * north
    end)
    dev(a) = to_backend(m.backend, a)
    return (u=u, w=KernelAbstractions.zeros(m.backend, Float64, length(x), nlev + 1), ρ=dev(ρ), ρθ=dev(ρθ))
end

"""
    vertical_stage!(m, y, y0, ys, τ)

The vertically implicit part of a stage in every column: `w`, `ρ`, `ρθ` of
`y` at `t + τ` from the base state `y0`, the stage state `ys` (θ, ρ and the
mass flux of the upwind θ) and the explicit parts in `m.work` (`tw`, `ρexpl`,
`ρθexpl`), linearized around the Exner pressure of `y0` (`base_exner!`).
"""
function vertical_stage!(m::KiteNonhydrostatic, y, y0, ys, τ)
    w, nh, nlev = m.work, num_columns(m), num_layers(m)
    w.θ .= ys.ρθ ./ ys.ρ
    launch!(m, theta_interface_kernel!, nh, w.θi, w.band, w.θm, w.θ, y0.ρ, y0.ρθ, ys.ρ, ys.w, m.dz, m.dzh, m.wlow,
            τ, nlev, m.vertical_theta > 0)
    launch!(m, vertical_solve_kernel!, nh, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρθ, w.π0, w.γ0, w.θ, w.θi,
            ys.ρ, w.tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.wlow, m.rayleigh, w.wmass, τ, m.beta, nlev,
            m.vertical_theta == 2)
    return y
end

"Exner pressure of the base state `y0` and its derivative by `ρθ` into the work arrays."
function base_exner!(m::KiteNonhydrostatic, y0)
    launch!(m, base_exner_kernel!, size(y0.ρθ), m.work.π0, m.work.γ0, y0.ρθ)
    return nothing
end

"""
    vertical_step!(m, y, dt, ys=zero_state(m))

One Wicker–Skamarock RK3 step of the vertical terms alone (acoustic and
buoyancy terms and gravity, implicit in every column; no horizontal or
advective terms), in place: the column solver on its own.
"""
function vertical_step!(m::KiteNonhydrostatic, y, dt, ys=zero_state(m))
    y0 = map(copy, (w=y.w, ρ=y.ρ, ρθ=y.ρθ))
    base_exner!(m, y0)
    fill!(m.work.tw, 0.0)
    copyto!(m.work.ρexpl, y0.ρ)
    copyto!(m.work.ρθexpl, y0.ρθ)
    s = (w=ys.w, ρ=ys.ρ, ρθ=ys.ρθ)
    vertical_stage!(m, s, y0, y0, dt / 3)
    vertical_stage!(m, y, y0, s, dt / 2)
    foreach(copyto!, s, (w=y.w, ρ=y.ρ, ρθ=y.ρθ))
    vertical_stage!(m, y, y0, s, dt)
    return y
end

# --- horizontal terms (step 3) -----------------------------------------------------

"""
    weighted_projection!(out, m, v, h; kinetic=nothing)

`Π_U(h v)`: the velocity-space field `v` times the height-space field `h` at
the quadrature points, projected into the velocity space (one solve with the
block-diagonal velocity mass). With `kinetic`, `∫ |v|²/2 η` is added to it.
"""
function weighted_projection!(out, m::KiteNonhydrostatic, v, h; kinetic=nothing)
    sw = m.horizontal
    Q, w = sw.quadrature, sw.work
    rhs_K = kinetic === nothing ? w.rhs_h : kinetic   # rhs_h: scratch for the unused kinetic energy
    kinetic === nothing && fill!(rhs_K, 0.0)
    fill!(w.rhs_F, 0.0)
    ShallowWater.launch!(ShallowWater.kinetic_and_flux_kernel!, w.rhs_F, num_local_kites(sw.distribution), rhs_K,
                         w.rhs_F, v, h, Q.piola, Q.dA, sw.φu, sw.iu, sw.su, sw.φh, sw.ih, true)
    return ldiv!(out, sw.Mu_solver, w.rhs_F)
end

"""
    horizontal_tendency!(dy, m, y)

Horizontal terms of `(u, ρ, ρθ)` in every layer, with the 2D kite operators
and `ρ` in place of the height (`a` the radius, `B` the kite divergence):

    F = Π_U(ρ u),  Fθ = Π_U(θ F),  θ = ρθ/ρ and π = (R ρθ/p0)^(R/cv) at the nodes,
    u̇ = M_u⁻¹ (-Bᵀ K / a - ∫ (ζ/a + f)/ρ (r̂ × F) ⋅ v) - c_p Π_U(θ ∇π),  ∇π = M_u⁻¹ Bᵀ π / a,
    ρ̇ = M_h⁻¹ B F / a,  ρθ̇ = M_h⁻¹ B Fθ / a.

The kinetic energy, PV and pressure terms exchange energy without loss:
`Σ_k dz_k (Fᵀ M_u u̇ + (K + g z)ᵀ M_h ρ̇ + c_p πᵀ M_h ρθ̇) = 0`. Vertical
advection is not included.
"""
function horizontal_tendency!(dy, m::KiteNonhydrostatic, y)
    sw, w, a = m.horizontal, m.work, m.radius
    Q, s = sw.quadrature, sw.work
    tz, tu, th = sw.row_buffers
    nk = num_local_kites(sw.distribution)
    w.θ .= y.ρθ ./ y.ρ
    @. w.π = (Rd * y.ρθ / P0)^(Rd / Cvd)
    for k in 1:num_layers(m)
        u, ρ, θ, π = view(y.u, :, k), view(y.ρ, :, k), view(w.θ, :, k), view(w.π, :, k)
        du = view(dy.u, :, k)
        # relative vorticity, mass flux and kinetic energy of the layer
        row_mul!(tz, sw.R_owned, u, -1 / a)
        copyto!(s.rhs_ζ, 1, tz, 1, length(tz))
        ldiv!(s.ζ, sw.Mz_solver, s.rhs_ζ)
        fill!(s.rhs_K, 0.0)
        weighted_projection!(w.F, m, u, ρ; kinetic=s.rhs_K)
        ldiv!(s.K, sw.Mh_local_solver, s.rhs_K)
        # momentum: kinetic energy gradient and PV term, then the pressure gradient c_p θ ∇π
        row_mul!(tu, sw.Bt_owned, s.K, -1 / a)
        copyto!(s.rhs_u, 1, tu, 1, length(tu))
        ShallowWater.launch!(ShallowWater.vorticity_kernel!, s.rhs_u, nk, s.rhs_u, w.F, ρ, s.ζ, Q.piola, Q.dA,
                             Q.normal, Q.coriolis, sw.φu, sw.iu, sw.su, sw.φh, sw.ih, sw.φz, sw.iz, true)
        ldiv!(du, sw.Mu_solver, s.rhs_u)
        row_mul!(tu, sw.Bt_owned, π, 1 / a)
        ldiv!(w.g, sw.Mu_solver, tu)
        weighted_projection!(w.gθ, m, w.g, θ)
        du .-= Cpd .* w.gθ
        # mass and ρθ
        horizontal_divergence!(view(dy.ρ, :, k), view(dy.ρθ, :, k), m, θ)
    end
    return dy
end

"`dρ = M_h⁻¹ B F / a` and `dρθ = M_h⁻¹ B Π_U(θ F) / a` for the flux `F` in `m.work.F`."
function horizontal_divergence!(dρ, dρθ, m::KiteNonhydrostatic, θ)
    sw, w = m.horizontal, m.work
    th = sw.row_buffers[3]
    row_mul!(th, sw.B_owned, w.F, 1 / m.radius)
    ldiv!(dρ, sw.Mh_solver, th)
    weighted_projection!(w.Fθ, m, w.F, θ)
    row_mul!(th, sw.B_owned, w.Fθ, 1 / m.radius)
    ldiv!(dρθ, sw.Mh_solver, th)
    return nothing
end

"""
    stage!(m::KiteNonhydrostatic, y, y0, ys, τ)

One HEVI stage `y = y0 + τ T(ys)`: the horizontal momentum terms of the stage
state, then (forward–backward) the mass and ρθ fluxes of the new velocity with
`ρ` and `θ` of the stage state, then the implicit vertical terms in every
column (`vertical_stage!`). Not yet included: vertical advection of `u` and the
horizontal terms of `w` (`tw = 0`).
"""
function stage!(m::KiteNonhydrostatic, y, y0, ys, τ)
    w = m.work
    horizontal_tendency!(w.dy, m, ys)
    @. y.u = y0.u + τ * w.dy.u
    for k in 1:num_layers(m)
        weighted_projection!(w.F, m, view(y.u, :, k), view(ys.ρ, :, k))
        horizontal_divergence!(view(w.ρexpl, :, k), view(w.ρθexpl, :, k), m, view(w.θ, :, k))
    end
    @. w.ρexpl = y0.ρ + τ * w.ρexpl
    @. w.ρθexpl = y0.ρθ + τ * w.ρθexpl
    fill!(w.tw, 0.0)
    return vertical_stage!(m, y, y0, ys, τ)
end

"""
    step!(m::KiteNonhydrostatic, y, dt, ys=zero_state(m))

One Wicker–Skamarock RK3 step (stages `dt/3`, `dt/2`, `dt`) in place; the base
state is kept in the work arrays.
"""
function step!(m::KiteNonhydrostatic, y, dt, ys=zero_state(m))
    y0 = m.work.y0
    foreach(copyto!, y0, y)
    base_exner!(m, y0)
    stage!(m, ys, y0, y0, dt / 3)
    stage!(m, y, y0, ys, dt / 2)
    foreach(copyto!, ys, y)
    stage!(m, y, y0, ys, dt)
    return y
end
