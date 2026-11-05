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
(`:lumped`, pointwise at the interfaces as in `TRiSKNonhydrostatic`).
`velocity` is the velocity space as in `NonlinearKiteShallowWater`: `:rt`
(`KiteBrokenRT(k)`, default) or `:dual_radau` (`KiteDualRadau(k)`, the equal
order pair of Wess et al. with `height_order = k`), always with the exact mass in
the rule of the projections. `vertical_advection=:vector_invariant` adds the vertical coupling of the
vector-invariant form, `-η w` in the `u` equation and `η ⋅ ṽ - ∂K/∂z` in the
`w` equation (`horizontal_tendency!`); `:none` leaves it out. `horizontal_theta=:upwind` adds the
upwind trace of θ on the primal faces to the horizontal ρθ flux (`upwind_theta!`, serial CPU only);
`:central` (default) keeps the energy-conserving flux `B Π_U(θ F)`. `:dg` and `:dg_upwind` transport
ρθ by DG with the kite mass flux, `Σ_K ∫ θ F ⋅ ∇η - Σ_F ∫ θ̂ F ⋅ n [η]` with the central or upwind
trace θ̂ (`dg_divergence!`), and use its adjoint in the pressure gradient (`dg_gradient!`), so the
energy exchange stays exact; ρ keeps `B F`. With these options the horizontal advection
of `w` is also DG (`w_advection!`, its adjoint `w_advection_flux!` in the momentum equation)
in place of the projected gradient `M_u⁻¹ Bᵀ w` in `η` (serial CPU only).
`tracer=true` adds a passive tracer `ρq` to the state (`tracer_divergence!`,
`tracer_column_kernel!`), transported as ρθ with the same flux and upwind option;
for `q = 1` its equations are those of `ρ`.
`equation_of_state=:shallow_water` replaces the dry Exner pressure in the
horizontal pressure gradient by `g ρθ / c_p` (`exner!`): with one layer and
θ = 1 the model is the kite shallow-water model with `ρ` as the height (a test;
the column solver keeps the dry equation of state). Horizontal operators act layer by
layer with the 2D kite operators (scaled from the unit sphere by `radius`).
On a periodic plane (`create_periodic_plane` with lengths in m) use `radius=1`
and `coriolis`, the Coriolis parameter `f(x)` (default `2Ω x₃` on the sphere).
"""
struct KiteNonhydrostatic{SW, V, W, UP, AD, BE}
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
    vertical_advection::Bool
    upwind::UP           # paired primal-face points for the upwind traces, or nothing
    upwind_density::Bool # upwind jumps of ρ and ρθ (true) or of θ alone
    dg::Bool             # DG transport of ρθ and ρq with the kite mass flux
    dg_upwind::Float64   # 0: central trace, 1: upwind trace of the DG transport
    advection::AD        # DG momentum advection (`horizontal_advection=:dg`): tables and mass blocks, or nothing
    shallow_water::Float64   # g of the shallow-water equation of state π = g ρθ / c_p, 0 for the dry one
    tracer::Bool         # passive tracer ρq in the state
    backend::BE
end

function KiteNonhydrostatic(primal::Plex, vgrid::VerticalGrid, k::Integer=1; height_order::Integer=k + 1,
                            radius::Real=6.37122e6, omega::Real=7.292e-5, beta::Real=0.65,
                            damping_height::Real=Inf, damping_rate::Real=0.0, vertical_theta::Symbol=:upwind,
                            vertical_mass::Symbol=:consistent, vertical_advection::Symbol=:vector_invariant,
                            velocity::Symbol=:rt, horizontal_theta::Symbol=:central,
                            horizontal_advection::Symbol=:vector_invariant,
                            equation_of_state::Symbol=:dry, gravity::Real=GRAV, tracer::Bool=false,
                            coriolis=x -> 2omega * x[3], backend=KernelAbstractions.CPU())
    vtheta = get(Dict(:centred => 0, :upwind => 1, :iap => 2), vertical_theta) do
        throw(ArgumentError("vertical_theta must be :upwind, :iap or :centred"))
    end
    vertical_advection in (:vector_invariant, :none) ||
        throw(ArgumentError("vertical_advection must be :vector_invariant or :none"))
    horizontal_theta in (:central, :upwind, :upwind_density, :dg, :dg_upwind) ||
        throw(ArgumentError("horizontal_theta must be :central, :upwind, :upwind_density, :dg or :dg_upwind"))
    horizontal_advection in (:vector_invariant, :dg) ||
        throw(ArgumentError("horizontal_advection must be :vector_invariant or :dg"))
    horizontal_advection == :dg && !(horizontal_theta in (:dg, :dg_upwind)) &&
        throw(ArgumentError("horizontal_advection=:dg needs horizontal_theta=:dg or :dg_upwind"))
    equation_of_state in (:dry, :shallow_water) ||
        throw(ArgumentError("equation_of_state must be :dry or :shallow_water"))
    # the velocity mass uses the rule of the projections (the default): Π_U(θ̄ F) = θ̄ F for a constant θ̄,
    # so the horizontal fluxes of ρ and ρθ do not create θ′
    # (also for the dual Radau velocity of Wess et al.: its exact mass, not the nodal one)
    sw = NonlinearKiteShallowWater(primal, k; height_order, coriolis, velocity,
                                   velocity_lumped=false, backend)
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
            dy=tracer ? (u=z(nu, nlev), ρ=z(nh, nlev), ρθ=z(nh, nlev), ρq=z(nh, nlev)) :
                        (u=z(nu, nlev), ρ=z(nh, nlev), ρθ=z(nh, nlev)),
            q=z(nh, nlev), ρqexpl=z(nh, nlev), Fq=KernelAbstractions.zeros(backend, Float64, nu),
            K=z(nh, nlev), Fs=z(nu, nlev), η=z(nu, nlev + 1), a=KernelAbstractions.zeros(backend, Float64, nu),
            b=KernelAbstractions.zeros(backend, Float64, nu), hw=KernelAbstractions.zeros(backend, Float64, nh),
            y0=tracer ? (u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev), ρq=z(nh, nlev)) :
                        (u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev)))
    dev(x) = to_backend(backend, x)
    return KiteNonhydrostatic(sw, vgrid, dev(vgrid.dz), dev(vgrid.dzh), dev(vgrid.wlow), dev(rayleigh), work,
                              Float64(radius), Float64(omega), Float64(beta), vtheta, vertical_advection == :vector_invariant,
                              horizontal_theta == :central ? nothing : upwind_faces(sw), horizontal_theta == :upwind_density,
                              horizontal_theta in (:dg, :dg_upwind), horizontal_theta == :dg_upwind ? 1.0 : 0.0,
                              horizontal_advection == :dg ? (tables=momentum_tables(sw), mass=weighted_mass_blocks(sw)) :
                                                            nothing,
                              equation_of_state == :shallow_water ? Float64(gravity) : 0.0, tracer, backend)
end

num_layers(m::KiteNonhydrostatic) = num_layers(m.vgrid)
"Number of vertical columns: the height dofs (Radau nodes) of the horizontal space."
num_columns(m::KiteNonhydrostatic) = length(height_dofs(m.horizontal))
num_velocity_dofs(m::KiteNonhydrostatic) = length(velocity_dofs(m.horizontal))

"State `(u, w, ρ, ρθ)` of the model, with `tracer` also `ρq`, zero."
function zero_state(m::KiteNonhydrostatic)
    nh, nu, nlev = num_columns(m), num_velocity_dofs(m), num_layers(m)
    z(n, l) = KernelAbstractions.zeros(m.backend, Float64, n, l)
    y = (u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev))
    return m.tracer ? (; y..., ρq=z(nh, nlev)) : y
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
    initial_state(m::KiteNonhydrostatic, profile; tracer=(λ, φ, z) -> 1.0)

State from `profile(λ, φ, z) = (T, p, u, v)` as for `TRiSKNonhydrostatic`
(on a plane `profile(x, y, z)` and `tracer(x, y, z)` with `u`, `v` along x and y):
every column in the discrete hydrostatic balance of the column solver at its
Radau node (`balanced_column`), the velocity `u east + v north` projected
into the velocity space per layer, `w = 0`; with a tracer `ρq = ρ q` for the
mixing ratio `tracer(λ, φ, z)` at the nodes and layer centres (default 1).
"""
function initial_state(m::KiteNonhydrostatic, profile; tracer=(λ, φ, z) -> 1.0)
    v, nlev = m.vgrid, num_layers(m)
    x = column_points(m)
    plane = m.horizontal.primal.surface != :sphere
    coordinates(x) = plane ? (x[1], x[2]) : lonlat(x)
    ρ, ρθ = zeros(length(x), nlev), zeros(length(x), nlev)
    for (i, xi) in enumerate(x)
        λ, φ = coordinates(xi)
        ρ[i, :], ρθ[i, :] = balanced_column(z -> profile(λ, φ, z), v, m.vertical_theta == 2)
    end
    u = project_velocity(m, (x, z) -> begin
        λ, φ = coordinates(x)
        east, north = plane ? (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)) : east_north(x)
        _, _, ue, vn = profile(λ, φ, z)
        ue * east + vn * north
    end)
    dev(a) = to_backend(m.backend, a)
    y = (u=u, w=KernelAbstractions.zeros(m.backend, Float64, length(x), nlev + 1), ρ=dev(ρ), ρθ=dev(ρθ))
    m.tracer || return y
    q = [tracer(coordinates(xi)..., v.z_full[k]) for xi in x, k in 1:nlev]
    return (; y..., ρq=dev(ρ .* q))
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

Explicit tendencies in every layer, with the 2D kite operators and `ρ` in place
of the height (`a` the radius, `B` the kite divergence):

    F = Π_U(ρ u),  Fθ = Π_U(θ F),  θ = ρθ/ρ and π = (R ρθ/p0)^(R/cv) at the nodes,
    u̇ = M_u⁻¹ (-Bᵀ K / a - ∫ (ζ/a + f)/ρ (r̂ × F) ⋅ v) - c_p Π_U(θ ∇π) - (η w)_k,  ∇π = M_u⁻¹ Bᵀ π / a,
    ρ̇ = M_h⁻¹ B F / a,  ρθ̇ = M_h⁻¹ B Fθ / a,

and with `vertical_advection` the explicit `w` tendency `m.work.tw`:

    η_j = (u_j - u_{j-1}) / dzh_j - M_u⁻¹ Bᵀ w_j / a   (horizontal vorticity at interface j),
    (η w)_k = (dzh_k Π_U(w_k η_k) + dzh_{k+1} Π_U(w_{k+1} η_{k+1})) / 2dz_k,
    tw_j = M_h⁻¹ ∫ η_j ⋅ (u_{j-1} + u_j)/2 ψ - (K_j - K_{j-1}) / dzh_j,

where `K` includes `(dzh_k w_k² + dzh_{k+1} w_{k+1}²)/4dz_k`. The kinetic energy,
PV and pressure terms exchange energy without loss:
`Σ_k dz_k (Fᵀ M_u u̇ + (K + g z)ᵀ M_h ρ̇ + c_p πᵀ M_h ρθ̇) = 0`, and the two
η terms are adjoint, `Σ_k dz_k u_kᵀ M_u (η w)_k = Σ_j dzh_j w_jᵀ M_h (η ⋅ ṽ)_j`.
"""
function horizontal_tendency!(dy, m::KiteNonhydrostatic, y)
    sw, w, a, v = m.horizontal, m.work, m.radius, m.vgrid
    Q, s = sw.quadrature, sw.work
    tz, tu, th = sw.row_buffers
    nk, nlev = num_local_kites(sw.distribution), num_layers(m)
    w.θ .= y.ρθ ./ y.ρ
    m.tracer && (w.q .= y.ρq ./ y.ρ)
    exner!(w.π, m, y.ρθ)
    # mass flux and kinetic energy of every layer (with the w² part)
    for k in 1:nlev
        fill!(s.rhs_K, 0.0)
        weighted_projection!(view(w.Fs, :, k), m, view(y.u, :, k), view(y.ρ, :, k); kinetic=s.rhs_K)
        ldiv!(view(w.K, :, k), sw.Mh_local_solver, s.rhs_K)
    end
    if m.vertical_advection
        @views @. w.K += (m.dzh[1:nlev]' * y.w[:, 1:nlev]^2 + m.dzh[2:nlev + 1]' * y.w[:, 2:nlev + 1]^2) / (4 * m.dz')
        # horizontal vorticity η = ∂u/∂z - ∇w at the interior interfaces
        fill!(w.η, 0.0)
        for j in 2:nlev
            @views @. w.η[:, j] = (y.u[:, j] - y.u[:, j - 1]) / v.dzh[j]
            m.dg && continue       # the -∇w part is the DG advection of w (w_advection!, w_advection_flux!)
            row_mul!(tu, sw.Bt_owned, view(y.w, :, j), 1 / a)
            ldiv!(w.a, sw.Mu_solver, tu)
            @views @. w.η[:, j] -= w.a
        end
    end
    for k in 1:nlev
        u, ρ, θ, π = view(y.u, :, k), view(y.ρ, :, k), view(w.θ, :, k), view(w.π, :, k)
        F, du = view(w.Fs, :, k), view(dy.u, :, k)
        if m.advection !== nothing
            dg_momentum!(du, m, y, k)
            horizontal_divergence!(view(dy.ρ, :, k), view(dy.ρθ, :, k), m, θ, u, ρ, view(y.ρθ, :, k))
            m.tracer && tracer_divergence!(view(dy.ρq, :, k), m, view(w.q, :, k), u, view(y.ρq, :, k))
            continue
        end
        # relative vorticity of the layer
        row_mul!(tz, sw.R_owned, u, -1 / a)
        copyto!(s.rhs_ζ, 1, tz, 1, length(tz))
        ldiv!(s.ζ, sw.Mz_solver, s.rhs_ζ)
        # momentum: kinetic energy gradient and PV term, then the pressure gradient c_p θ ∇π
        row_mul!(tu, sw.Bt_owned, view(w.K, :, k), -1 / a)
        copyto!(s.rhs_u, 1, tu, 1, length(tu))
        ShallowWater.launch!(ShallowWater.vorticity_kernel!, s.rhs_u, nk, s.rhs_u, F, ρ, s.ζ, Q.piola, Q.dA,
                             Q.normal, Q.coriolis, sw.φu, sw.iu, sw.su, sw.φh, sw.ih, sw.φz, sw.iz, true)
        ldiv!(du, sw.Mu_solver, s.rhs_u)
        if m.dg
            dg_gradient!(tu, m, F, θ, π, 1 / a)
            ldiv!(w.gθ, sw.Mu_solver, tu)
        else
            row_mul!(tu, sw.Bt_owned, π, 1 / a)
            ldiv!(w.g, sw.Mu_solver, tu)
            weighted_projection!(w.gθ, m, w.g, θ)
        end
        du .-= Cpd .* w.gθ
        # vertical coupling -η w at the two interfaces of the layer
        if m.vertical_advection
            for (j, c) in ((k, v.dzh[k]), (k + 1, v.dzh[k + 1]))
                2 <= j <= nlev || continue
                weighted_projection!(w.gθ, m, view(w.η, :, j), view(y.w, :, j))
                du .-= (c / (2 * v.dz[k])) .* w.gθ
                if m.dg
                    w_advection_flux!(tu, m, view(y.w, :, j), 1 / a)
                    ldiv!(w.gθ, sw.Mu_solver, tu)
                    du .+= (c / (2 * v.dz[k])) .* w.gθ
                end
            end
        end
        # mass and ρθ
        copyto!(w.F, F)
        horizontal_divergence!(view(dy.ρ, :, k), view(dy.ρθ, :, k), m, θ, u, ρ, view(y.ρθ, :, k))
        m.tracer && tracer_divergence!(view(dy.ρq, :, k), m, view(w.q, :, k), u, view(y.ρq, :, k))
    end
    fill!(w.tw, 0.0)
    if m.vertical_advection
        for j in 2:nlev
            @views @. w.a = w.η[:, j] + (y.u[:, j - 1] + y.u[:, j]) / 2
            @views @. w.b = w.η[:, j] - (y.u[:, j - 1] + y.u[:, j]) / 2
            dot_product!(view(w.tw, :, j), m, w.a, w.b)
            if m.dg
                @. w.g = (w.a - w.b) / 2
                ldiv!(w.hw, sw.Mh_solver, w_advection!(th, m, w.g, view(y.w, :, j), 1 / a))
                @views w.tw[:, j] .-= w.hw
            end
            @views @. w.tw[:, j] -= (w.K[:, j] - w.K[:, j - 1]) / v.dzh[j]
        end
    end
    return dy
end

"""
    dg_momentum!(du, m, y, k)

Velocity tendency of layer `k` with `horizontal_advection=:dg`, from the
density-weighted momentum equation (`a` the radius)

    ∫ ρ u̇ ⋅ v = -N(u; v)/a - ∫ ρ f (r̂ × u) ⋅ v - c_p (Σ_K ∫_K ρθ v ⋅ ∇π - Σ_F ∫_F (ρθ)^ (v ⋅ n_K) [π])/a
                - Σ_j dzh_j/(2 dz_k) ∫ ρ w_j η_j ⋅ v,

`N` the DG advection `momentum_advection!`, the pressure term the adjoint of the
DG ρθ transport (`dg_gradient!` with the weight ρθ, so the exchange with the
internal energy is exact) and `η_j = (u_j - u_{j-1})/dzh_j` the vertical
shear at the interfaces of the layer (vertical advection of `u`), solved with
the density-weighted mass (`weighted_mass_solve!`). Uses `m.work.η`, `m.work.π`.
"""
function dg_momentum!(du, m::KiteNonhydrostatic, y, k)
    sw, w, a, v, A = m.horizontal, m.work, m.radius, m.vgrid, m.advection
    tu = sw.row_buffers[2]
    u, ρ, ρθ = view(y.u, :, k), view(y.ρ, :, k), view(y.ρθ, :, k)
    rhs = w.g
    momentum_advection!(tu, sw, A.tables, u, ρ, m.dg_upwind)
    @. rhs = -tu / a
    coriolis_momentum!(rhs, sw, u, ρ)
    dg_gradient!(tu, m, u, ρθ, view(w.π, :, k), 1 / a)
    @. rhs -= Cpd * tu
    if m.vertical_advection
        for (j, c) in ((k, v.dzh[k]), (k + 1, v.dzh[k + 1]))
            2 <= j <= num_layers(m) || continue
            @views @. w.hw = -c / (2 * v.dz[k]) * ρ * y.w[:, j]
            weighted_vector!(rhs, sw, view(w.η, :, j), w.hw)
        end
    end
    return weighted_mass_solve!(du, sw, A.mass, ρ, rhs)
end

"""
    exner!(π, m, ρθ)

The Exner pressure of the horizontal pressure gradient: `(R ρθ / p0)^(R/c_v)`, or
with `equation_of_state=:shallow_water` `g ρθ / c_p`, so that for θ = 1 the term
`-c_p θ ∇π` is `-g ∇ρ` and the pressure `p = g ρ²/2` of shallow water (the
energy stays consistent, `∂E/∂ρθ = c_p π = g ρθ`).
"""
function exner!(π, m::KiteNonhydrostatic, ρθ)
    g = m.shallow_water
    g == 0 ? (@. π = (Rd * ρθ / P0)^(Rd / Cvd)) : (@. π = g * ρθ / Cpd)
    return π
end

"""
    dot_product!(out, m, p, q)

`M_h⁻¹ ∫ a ⋅ b ψ` into the height space for `p = a + b`, `q = a - b` (velocity
space): polarization of the kinetic energy, `a ⋅ b = (|p|² - |q|²)/4`.
"""
function dot_product!(out, m::KiteNonhydrostatic, p, q)
    sw = m.horizontal
    Q, s = sw.quadrature, sw.work
    nk = num_local_kites(sw.distribution)
    for (x, rhs) in ((p, s.rhs_K), (q, s.rhs_h))
        fill!(rhs, 0.0)
        ShallowWater.launch!(ShallowWater.kinetic_and_flux_kernel!, rhs, nk, rhs, s.rhs_F, x, s.K, Q.piola, Q.dA,
                             sw.φu, sw.iu, sw.su, sw.φh, sw.ih, false)
    end
    s.rhs_K .= (s.rhs_K .- s.rhs_h) ./ 2
    ldiv!(out, sw.Mh_local_solver, s.rhs_K)
    return out
end

"""
    horizontal_divergence!(dρ, dρθ, m, θ, u, ρ, ρθ)

`dρ = M_h⁻¹ B F / a` and `dρθ = M_h⁻¹ B Π_U(θ F) / a` for the flux `F` in
`m.work.F`, with `horizontal_theta=:upwind` the upwind θ trace in `dρθ`
(`upwind_theta!`), with `:upwind_density` the upwind traces of `ρ` and `ρθ`
with the velocity `u` (`upwind_jump!`), which are identical for θ = 1.
"""
function horizontal_divergence!(dρ, dρθ, m::KiteNonhydrostatic, θ, u, ρ, ρθ)
    sw, w = m.horizontal, m.work
    th = sw.row_buffers[3]
    if m.advection !== nothing      # DG transport with the pointwise velocity
        ldiv!(dρ, sw.Mh_solver, dg_divergence!(th, m, u, ρ, 1 / m.radius))
        ldiv!(dρθ, sw.Mh_solver, dg_divergence!(th, m, u, ρθ, 1 / m.radius))
        return nothing
    end
    density = m.upwind !== nothing && m.upwind_density
    row_mul!(th, sw.B_owned, w.F, 1 / m.radius)
    density && upwind_jump!(th, m, u, ρ, 1 / m.radius)
    ldiv!(dρ, sw.Mh_solver, th)
    if m.dg
        dg_divergence!(th, m, w.F, θ, 1 / m.radius)
        ldiv!(dρθ, sw.Mh_solver, th)
        return nothing
    end
    weighted_projection!(w.Fθ, m, w.F, θ)
    row_mul!(th, sw.B_owned, w.Fθ, 1 / m.radius)
    if density
        upwind_jump!(th, m, u, ρθ, 1 / m.radius)
    elseif m.upwind !== nothing
        upwind_theta!(th, m, w.F, θ, 1 / m.radius)
    end
    ldiv!(dρθ, sw.Mh_solver, th)
    return nothing
end

"""
    upwind_faces(sw) -> (; cK, qK, fK, cE, qE, fE)

The quadrature points of the primal faces (halves of primal edges) of the local
kites in pairs: point `qK` of face `fK` of kite `cK` coincides with point `qE`
of face `fE` of the neighbour kite `cE` across the primal edge. Serial only.
"""
function upwind_faces(sw)
    kites, H = sw.kites, sw.H
    shape = reference_element(H, first(cell_range(kites))).shape
    lattice = face_lattice(kites)
    open = Dict{NTuple{3, Int}, NTuple{3, Int}}()
    pairs = NTuple{6, Int}[]
    for (c, kite) in enumerate(sw.distribution.kite_ids)
        cmap = CellMap(kites, first(cell_range(kites)) + kite - 1)
        for (f, face) in enumerate(sw.faces)
            t, _ = gauss_legendre(length(face.weights))
            for q in eachindex(t)
                key = lattice(first(map_point(cmap, edge_point(shape, KITE_PRIMAL_FACES[f], t[q]))))
                other = pop!(open, key, nothing)
                other === nothing ? (open[key] = (c, q, f)) : push!(pairs, (other..., c, q, f))
            end
        end
    end
    isempty(open) || error("unpaired primal-face points: $(length(open))")
    return NamedTuple{(:cK, :qK, :fK, :cE, :qE, :fE)}(Tuple([p[i] for p in pairs] for i in 1:6))
end

"""
    upwind_jump!(th, m, u, q, scale)

Adds the upwind correction of the density `q` (ρ or ρθ) on the primal faces,
`-scale ∫_F η_i^K |u ⋅ n|/2 (q^K - q^E)` for both kites, to `th`: the
Lax–Friedrichs/upwind penalty of the jumps of a broken density, conservative
and dissipative in `q²`.
"""
function upwind_jump!(th, m::KiteNonhydrostatic, u, q, scale)
    sw, up = m.horizontal, m.upwind
    ih = sw.ih
    @inbounds for p in eachindex(up.cK)
        cK, qK, cE, qE = up.cK[p], up.qK[p], up.cE[p], up.qE[p]
        fK, fE = sw.faces[up.fK[p]], sw.faces[up.fE[p]]
        ûn = dot(ShallowWater.gather_vector(u, fK.φ, sw.iu, sw.su, cK, qK), fK.ν)
        jump = ShallowWater.gather_scalar(q, fK.η, ih, cK, qK) - ShallowWater.gather_scalar(q, fE.η, ih, cE, qE)
        g = scale * fK.weights[qK] * abs(ûn) * jump / 2
        for a in axes(ih, 1)
            th[ih[a, cK]] -= g * fK.η[a, 1, qK]
            th[ih[a, cE]] += g * fE.η[a, 1, qE]
        end
    end
    return th
end

"""
    upwind_theta!(th, m, F, θ, scale)

Adds the upwind trace of θ on the primal faces to the ρθ divergence `th = B Π_U(θ F)`:

    -scale ∫_F η_i^K |F ⋅ n|/2 (θ^K - θ^E)

for the test functions of both kites `K`, `E` at every face. It vanishes for θ
continuous across the primal edges and conserves ρθ; it damps the jumps of θ.
"""
function upwind_theta!(th, m::KiteNonhydrostatic, F, θ, scale)
    sw, up = m.horizontal, m.upwind
    ih = sw.ih
    @inbounds for p in eachindex(up.cK)
        cK, qK, cE, qE = up.cK[p], up.qK[p], up.cE[p], up.qE[p]
        fK, fE = sw.faces[up.fK[p]], sw.faces[up.fE[p]]
        ûn = dot(ShallowWater.gather_vector(F, fK.φ, sw.iu, sw.su, cK, qK), fK.ν)
        jump = ShallowWater.gather_scalar(θ, fK.η, ih, cK, qK) - ShallowWater.gather_scalar(θ, fE.η, ih, cE, qE)
        g = scale * fK.weights[qK] * abs(ûn) * jump / 2
        for a in axes(ih, 1)
            th[ih[a, cK]] -= g * fK.η[a, 1, qK]
            th[ih[a, cE]] += g * fE.η[a, 1, qE]
        end
    end
    return th
end

"Trace of `q` on a primal face: the mean, with `dg_upwind` the upwind value for the normal flux `ûn`."
@inline face_trace(qK, qE, ûn, α) = (qK + qE) / 2 + α * sign(ûn) * (qK - qE) / 2

"""
    dg_divergence!(th, m, F, q, scale)

DG transport of a density `ρq = ρ q` with the kite mass flux `F` of one layer
(`horizontal_theta=:dg`, `:dg_upwind`), into the owned height rows `th`:

    th_i = scale (Σ_K ∫_K q F ⋅ ∇η_i - Σ_F ∫_F q̂ (F ⋅ n_K) (η_i^K - η_i^E)),

`q̂` the central or upwind trace (`face_trace`). For `q = 1` it is `B F`, so
`ρq = ρ` is kept; with the central trace it is skew in `q` for a divergence-free
`F`, with the upwind trace it damps the jumps of `q`. Serial CPU.
"""
function dg_divergence!(th, m::KiteNonhydrostatic, F, q, scale)
    sw, up = m.horizontal, m.upwind
    ih, α = sw.ih, m.dg_upwind
    fill!(th, 0.0)
    # volume term and the interior trace q_K on every face, then the trace corrected to q̂
    ShallowWater.launch!(ShallowWater.pointwise_flux_kernel!, th, num_local_kites(sw.distribution), th, F, q,
                         sw.quadrature.weights, sw.φu, sw.iu, sw.su, sw.φh, sw.∇φh, ih, sw.faces)
    @inbounds for p in eachindex(up.cK)
        cK, qK, cE, qE = up.cK[p], up.qK[p], up.cE[p], up.qE[p]
        fK, fE = sw.faces[up.fK[p]], sw.faces[up.fE[p]]
        ûn = dot(ShallowWater.gather_vector(F, fK.φ, sw.iu, sw.su, cK, qK), fK.ν)
        vK = ShallowWater.gather_scalar(q, fK.η, ih, cK, qK)
        vE = ShallowWater.gather_scalar(q, fE.η, ih, cE, qE)
        q̂ = face_trace(vK, vE, ûn, α)
        gK, gE = fK.weights[qK] * ûn * (q̂ - vK), fK.weights[qK] * ûn * (q̂ - vE)
        for a in axes(ih, 1)
            th[ih[a, cK]] -= gK * fK.η[a, 1, qK]
            th[ih[a, cE]] += gE * fE.η[a, 1, qE]
        end
    end
    th .*= scale
    return th
end

"""
    dg_gradient!(tu, m, F, θ, π, scale)

The pressure gradient adjoint to `dg_divergence!`, into the owned velocity rows `tu`:

    tu_v = scale (Σ_K ∫_K θ φ_v ⋅ ∇π - Σ_F ∫_F θ̂ (φ_v ⋅ n_K) (π^K - π^E)),

with the trace `θ̂` of the flux `F`, so that `πᵀ dg_divergence(F, θ) = Fᵀ tu`
and the exchange of kinetic and internal energy is exact. Serial CPU.
"""
function dg_gradient!(tu, m::KiteNonhydrostatic, F, θ, π, scale)
    sw, up = m.horizontal, m.upwind
    ih, iu, su, α = sw.ih, sw.iu, sw.su, m.dg_upwind
    fill!(tu, 0.0)
    @inbounds for c in axes(ih, 2), (q, wq) in enumerate(sw.quadrature.weights)  # metric free: θ φ̂ ⋅ ∇̂π̂ dξ
        θq = ShallowWater.gather_scalar(θ, sw.φh, ih, c, q)
        g1 = g2 = 0.0
        for a in axes(ih, 1)
            g1 += π[ih[a, c]] * sw.∇φh[a, 1, 1, q]
            g2 += π[ih[a, c]] * sw.∇φh[a, 1, 2, q]
        end
        g = wq * θq
        for b in axes(iu, 1)
            tu[iu[b, c]] += su[b, c] * g * (sw.φu[b, 1, q] * g1 + sw.φu[b, 2, q] * g2)
        end
    end
    @inbounds for p in eachindex(up.cK)
        cK, qK, cE, qE = up.cK[p], up.qK[p], up.cE[p], up.qE[p]
        fK, fE = sw.faces[up.fK[p]], sw.faces[up.fE[p]]
        ûn = dot(ShallowWater.gather_vector(F, fK.φ, iu, su, cK, qK), fK.ν)
        θ̂ = face_trace(ShallowWater.gather_scalar(θ, fK.η, ih, cK, qK),
                       ShallowWater.gather_scalar(θ, fE.η, ih, cE, qE), ûn, α)
        jump = ShallowWater.gather_scalar(π, fK.η, ih, cK, qK) - ShallowWater.gather_scalar(π, fE.η, ih, cE, qE)
        g = fK.weights[qK] * θ̂ * jump
        for b in axes(iu, 1)   # the normal trace on the face is single valued: kite K's basis suffices
            tu[iu[b, cK]] -= su[b, cK] * g * (fK.φ[b, 1, qK] * fK.ν[1] + fK.φ[b, 2, qK] * fK.ν[2])
        end
    end
    tu .*= scale
    return tu
end

"""
    w_advection!(th, m, u, w, scale)

Horizontal advection of `w` (broken across the primal edges) by the velocity `u`
in DG advective form, into the owned height rows `th` (`horizontal_theta=:dg`, `:dg_upwind`):

    th_i = scale (Σ_K ∫_K (u ⋅ ∇w) ψ_i - Σ_F ∫_F (u ⋅ n_K) [w] {ψ_i} + α Σ_F ∫_F |u ⋅ n|/2 [w] [ψ_i]),

`[w] = w^K - w^E`, `α = 1` for `:dg_upwind`. The central part is skew in `w` for a
divergence-free `u`; the projected gradient `M_u⁻¹ Bᵀ w` it replaces is not (its
eigenvalues are those of `B Π_U(ρ u)`). Serial CPU.
"""
function w_advection!(th, m::KiteNonhydrostatic, u, w, scale)
    sw, up = m.horizontal, m.upwind
    ih, iu, su, α = sw.ih, sw.iu, sw.su, m.dg_upwind
    fill!(th, 0.0)
    @inbounds for c in axes(ih, 2), (q, wq) in enumerate(sw.quadrature.weights)  # metric free: (û ⋅ ∇̂w) ψ dξ
        û = ShallowWater.gather_vector(u, sw.φu, iu, su, c, q)
        g = 0.0
        for a in axes(ih, 1)
            g += w[ih[a, c]] * (û[1] * sw.∇φh[a, 1, 1, q] + û[2] * sw.∇φh[a, 1, 2, q])
        end
        for a in axes(ih, 1)
            th[ih[a, c]] += wq * g * sw.φh[a, 1, q]
        end
    end
    @inbounds for p in eachindex(up.cK)
        cK, qK, cE, qE = up.cK[p], up.qK[p], up.cE[p], up.qE[p]
        fK, fE = sw.faces[up.fK[p]], sw.faces[up.fE[p]]
        ûn = dot(ShallowWater.gather_vector(u, fK.φ, iu, su, cK, qK), fK.ν)
        jump = ShallowWater.gather_scalar(w, fK.η, ih, cK, qK) - ShallowWater.gather_scalar(w, fE.η, ih, cE, qE)
        gc, gu = fK.weights[qK] * ûn * jump / 2, fK.weights[qK] * α * abs(ûn) * jump / 2
        for a in axes(ih, 1)
            th[ih[a, cK]] -= (gc - gu) * fK.η[a, 1, qK]
            th[ih[a, cE]] -= (gc + gu) * fE.η[a, 1, qE]
        end
    end
    th .*= scale
    return th
end

"""
    w_advection_flux!(tu, m, w, scale)

The momentum term adjoint to the central part of `w_advection!` (the `w ∇w` of
`-η w`), into the owned velocity rows `tu`: `tu_v = ∂/∂u_v` of `wᵀ w_advection(u, w)`,

    tu_v = scale (Σ_K ∫_K (φ_v ⋅ ∇w) w - Σ_F ∫_F (φ_v ⋅ n_K) [w] {w}),

so that `uᵀ tu = wᵀ w_advection(u, w)` (central) and the energy exchange of the
vertical coupling stays exact. Serial CPU.
"""
function w_advection_flux!(tu, m::KiteNonhydrostatic, w, scale)
    sw, up = m.horizontal, m.upwind
    ih, iu, su = sw.ih, sw.iu, sw.su
    fill!(tu, 0.0)
    @inbounds for c in axes(ih, 2), (q, wq) in enumerate(sw.quadrature.weights)
        g1 = g2 = 0.0
        for a in axes(ih, 1)
            g1 += w[ih[a, c]] * sw.∇φh[a, 1, 1, q]
            g2 += w[ih[a, c]] * sw.∇φh[a, 1, 2, q]
        end
        g = wq * ShallowWater.gather_scalar(w, sw.φh, ih, c, q)
        for b in axes(iu, 1)
            tu[iu[b, c]] += su[b, c] * g * (sw.φu[b, 1, q] * g1 + sw.φu[b, 2, q] * g2)
        end
    end
    @inbounds for p in eachindex(up.cK)
        cK, qK, cE, qE = up.cK[p], up.qK[p], up.cE[p], up.qE[p]
        fK, fE = sw.faces[up.fK[p]], sw.faces[up.fE[p]]
        wK, wE = ShallowWater.gather_scalar(w, fK.η, ih, cK, qK), ShallowWater.gather_scalar(w, fE.η, ih, cE, qE)
        g = fK.weights[qK] * (wK - wE) * (wK + wE) / 2
        for b in axes(iu, 1)
            tu[iu[b, cK]] -= su[b, cK] * g * (fK.φ[b, 1, qK] * fK.ν[1] + fK.φ[b, 2, qK] * fK.ν[2])
        end
    end
    tu .*= scale
    return tu
end

"""
    stage!(m::KiteNonhydrostatic, y, y0, ys, τ)

One HEVI stage `y = y0 + τ T(ys)`: the horizontal momentum terms of the stage
state, then (forward–backward) the mass and ρθ fluxes of the new velocity with
`ρ` and `θ` of the stage state, then the implicit vertical terms in every
column (`vertical_stage!`) with the explicit `w` tendency of the stage state.
"""
function stage!(m::KiteNonhydrostatic, y, y0, ys, τ)
    w = m.work
    horizontal_tendency!(w.dy, m, ys)
    @. y.u = y0.u + τ * w.dy.u
    for k in 1:num_layers(m)
        weighted_projection!(w.F, m, view(y.u, :, k), view(ys.ρ, :, k))
        horizontal_divergence!(view(w.ρexpl, :, k), view(w.ρθexpl, :, k), m, view(w.θ, :, k), view(y.u, :, k),
                               view(ys.ρ, :, k), view(ys.ρθ, :, k))
        m.tracer && tracer_divergence!(view(w.ρqexpl, :, k), m, view(w.q, :, k), view(y.u, :, k), view(ys.ρq, :, k))
    end
    @. w.ρexpl = y0.ρ + τ * w.ρexpl
    @. w.ρθexpl = y0.ρθ + τ * w.ρθexpl
    vertical_stage!(m, y, y0, ys, τ)
    if m.tracer
        @. w.ρqexpl = y0.ρq + τ * w.ρqexpl
        launch!(m, tracer_column_kernel!, num_columns(m), y.ρq, w.ρqexpl, w.q, ys.ρ, y.w, y0.w, m.dz, m.wlow, τ,
                m.beta, num_layers(m))
    end
    return y
end

"""
    tracer_divergence!(dρq, m, q, u, ρq)

Horizontal tracer transport of one layer, `dρq = M_h⁻¹ B Π_U(q F) / a` with the
mass flux `F` in `m.work.F`, as `ρθ` (with the same upwind option): for `q = 1`
it is the mass divergence.
"""
function tracer_divergence!(dρq, m::KiteNonhydrostatic, q, u, ρq)
    sw, w = m.horizontal, m.work
    th = sw.row_buffers[3]
    m.advection !== nothing && return ldiv!(dρq, sw.Mh_solver, dg_divergence!(th, m, u, ρq, 1 / m.radius))
    m.dg && return ldiv!(dρq, sw.Mh_solver, dg_divergence!(th, m, w.F, q, 1 / m.radius))
    weighted_projection!(w.Fq, m, w.F, q)
    row_mul!(th, sw.B_owned, w.Fq, 1 / m.radius)
    if m.upwind !== nothing
        m.upwind_density ? upwind_jump!(th, m, u, ρq, 1 / m.radius) : upwind_theta!(th, m, w.F, q, 1 / m.radius)
    end
    ldiv!(dρq, sw.Mh_solver, th)
    return dρq
end

"""
Vertical tracer transport of one column: `ρq = ρq_expl - τ δ_z(q_j M_j)` with
the mass flux of the column solve, `M_j = ρ_j (β w_j + (1 - β) w0_j)`, and `q`,
`ρ` of the stage state linearly interpolated to the interfaces; for `q = 1` the
mass update of `vertical_solve_kernel!`.
"""
@kernel function tracer_column_kernel!(ρq, @Const(ρqexpl), @Const(q), @Const(ρs), @Const(w), @Const(w0), @Const(dz),
                                       @Const(wlow), τ, β, nlev)
    i = @index(Global)
    @inbounds begin
        Flow = 0.0
        for k in 1:nlev
            F = 0.0
            if k < nlev
                j = k + 1
                ρj = wlow[j] * ρs[i, j - 1] + (1 - wlow[j]) * ρs[i, j]
                qj = wlow[j] * q[i, j - 1] + (1 - wlow[j]) * q[i, j]
                F = qj * ρj * (β * w[i, j] + (1 - β) * w0[i, j])
            end
            ρq[i, k] = ρqexpl[i, k] - τ / dz[k] * (F - Flow)
            Flow = F
        end
    end
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
