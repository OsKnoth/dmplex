# Nonhydrostatic model on the extruded kite mesh: the spaces of the 2D kite model times the
# layers of a vertical grid (docs/KITE3D_PLAN.md). Step 1: spaces, state, projections and the
# mass divergence; step 2: the vertically implicit column solve; step 3: the horizontal terms
# per layer and the HEVI step.

"""
    KiteNonhydrostatic(primal, vgrid, k=1; height_order=k + 1, radius=6.37122e6, omega=7.292e-5,
                       beta=0.65, damping_height=Inf, damping_rate=0, vertical_theta=:central,
                       backend=CPU())

Dry compressible Euler equations on the extruded kite mesh of `primal` with
the spaces of `NonlinearKiteShallowWater(primal, k)` (its `horizontal` part, on
the unit sphere) times the layers of `vgrid`, lowest vertical order and
Lorenz staggering:

| field | space | array |
|---|---|---|
| ρ, ρθ | `KiteRadau(height_order)` ⊗ DG0 | (nlev, height dofs) |
| w     | `KiteRadau(height_order)` ⊗ CG1 | (nlev + 1, height dofs) |
| u_h   | `KiteBrokenRT(k)` ⊗ DG0         | (nlev, velocity dofs) |

The arrays are layers first, so that the threads of the layers of a point read adjacent memory.

The Radau nodes of the height space are its dofs and quadrature points, so its
mass matrix is diagonal and every height dof carries an independent column for
the vertically implicit terms (`num_columns`) at `column_points`. The vertical
terms are the Galerkin terms of DG0 × CG1 with the trapezoidal rule of every
layer (KiteColumn.jl): the mass of `w` is diagonal, the mass flux at interface
`j` is `ρ̄_j w_j` with the thickness-weighted mean `ρ̄`, the ρθ flux `θ̄_j F_j`,
and the pressure gradient `c_p θ̄_j (π_j - π_{j-1})/dzh_j` is θ̄ times the adjoint
of the divergence, so the vertical energy exchange is exact
(`vertical_tendency!`). Every column is solved implicitly (`kite_column_kernel!`,
off-centring `beta`, Rayleigh damping of `w` above `damping_height`).
`vertical_theta=:upwind` adds the upwind penalty `|F_j| (θ_{j-1} - θ_j)/2` of
the stage state to the ρθ flux (the upwind flux of DG0, dissipative);
`:central` (default) keeps `θ̄_j F_j`, second order in the vertical (docs/dycores/Kite-vs-ICON.md).
`velocity` is the velocity space as in `NonlinearKiteShallowWater`: `:rt`
(`KiteBrokenRT(k)`, default) or `:dual_radau` (`KiteDualRadau(k)`, the equal
order pair of Wess et al. with `height_order = k`), always with the exact mass in
the rule of the projections. `vertical_advection=:vector_invariant` adds the vertical coupling of the
vector-invariant form, `-η w` in the `u` equation and `η ⋅ ṽ - ∂K/∂z` in the
`w` equation (`horizontal_tendency!`); `:none` leaves it out. `horizontal_theta=:upwind` adds the
upwind trace of θ on the primal faces to the horizontal ρθ flux (`upwind_theta!`);
`:central` (default) keeps the energy-conserving flux `B Π_U(θ F)`. `:dg` and `:dg_upwind` transport
ρθ by DG with the kite mass flux, `Σ_K ∫ θ F ⋅ ∇η - Σ_F ∫ θ̂ F ⋅ n [η]` with the central or upwind
trace θ̂ (`dg_divergence!`), and use its adjoint in the pressure gradient (`dg_gradient!`), so the
energy exchange stays exact; ρ keeps `B F`. With these options the horizontal advection
of `w` is also DG (`w_advection!`, its adjoint `w_advection_flux!` in the momentum equation)
in place of the projected gradient `M_u⁻¹ Bᵀ w` in `η`. All terms are kernels over layers × (kites or
face points) and run on CPUs and GPUs (`backend`) and with MPI (`comm`, `part`: the distribution
of the 2D kite model, ghost rows by `exchange!` of the layered fields).
`tracer=true` adds a passive tracer `ρq` to the state (`tracer_divergence!`,
`kite_tracer_column_kernel!`), transported as ρθ with the same flux and upwind option;
for `q = 1` its equations are those of `ρ`.
`equation_of_state=:shallow_water` replaces the dry Exner pressure in the
horizontal pressure gradient by `g ρθ / c_p` (`exner!`): with one layer and
θ = 1 the model is the kite shallow-water model with `ρ` as the height (a test;
the column solver keeps the dry equation of state). Horizontal operators act layer by
layer with the 2D kite operators (scaled from the unit sphere by `radius`).
On a periodic plane (`create_periodic_plane` with lengths in m) use `radius=1`
and `coriolis`, the Coriolis parameter `f(x)` (default `2Ω x₃` on the sphere).
"""
struct KiteNonhydrostatic{SW, V, W, UP, AD, MU, BE}
    horizontal::SW
    vgrid::VerticalGrid
    dz::V
    dzh::V
    below::V             # weight of the layer below in the interface mean (`interface_weights`)
    rayleigh::V          # damping rate at the interfaces
    work::W
    radius::Float64
    omega::Float64
    beta::Float64
    theta_penalty::Float64   # 1: upwind penalty of the vertical ρθ flux, 0: central
    vertical_advection::Bool
    upwind::UP           # paired primal-face points for the upwind traces, or nothing
    upwind_density::Bool # upwind jumps of ρ and ρθ (true) or of θ alone
    dg::Bool             # DG transport of ρθ and ρq with the kite mass flux
    dg_upwind::Float64   # 0: central trace, 1: upwind trace of the DG transport
    advection::AD        # DG momentum advection (`horizontal_advection=:dg`): tables and mass blocks, or nothing
    velocity_solver::MU  # the velocity mass of the layered solves: sparse Cholesky factors of the dual-cell blocks
    shallow_water::Float64   # g of the shallow-water equation of state π = g ρθ / c_p, 0 for the dry one
    tracer::Bool         # passive tracer ρq in the state
    backend::BE
end

function KiteNonhydrostatic(primal::Plex, vgrid::VerticalGrid, k::Integer=1; height_order::Integer=k + 1,
                            radius::Real=6.37122e6, omega::Real=7.292e-5, beta::Real=0.65,
                            damping_height::Real=Inf, damping_rate::Real=0.0, vertical_theta::Symbol=:central,
                            vertical_advection::Symbol=:vector_invariant,
                            velocity::Symbol=:rt, horizontal_theta::Symbol=:central,
                            horizontal_advection::Symbol=:vector_invariant,
                            equation_of_state::Symbol=:dry, gravity::Real=GRAV, tracer::Bool=false,
                            coriolis=x -> 2omega * x[3], comm=nothing, part=nothing,
                            backend=KernelAbstractions.CPU())
    vertical_theta in (:upwind, :central) || throw(ArgumentError("vertical_theta must be :upwind or :central"))
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
                                   velocity_lumped=false, comm, part, backend)
    nh, nu, nlev = length(height_dofs(sw)), length(velocity_dofs(sw)), num_layers(vgrid)
    nz = length(sw.work.ζ)
    z_top = vgrid.z_half[end]
    rayleigh = map(vgrid.z_half) do z
        z > damping_height ? damping_rate * sin(π / 2 * (z - damping_height) / (z_top - damping_height))^2 : 0.0
    end
    z(n, l) = KernelAbstractions.zeros(backend, Float64, l, n)   # n dofs × l layers, layers first
    work = (θ=z(nh, nlev), P=z(nh, nlev + 1), cprime=z(nh, nlev + 1), dprime=z(nh, nlev + 1), π0=z(nh, nlev),
            γ0=z(nh, nlev),
            tw=z(nh, nlev + 1), ρexpl=z(nh, nlev), ρθexpl=z(nh, nlev), π=z(nh, nlev),
            dy=tracer ? (u=z(nu, nlev), ρ=z(nh, nlev), ρθ=z(nh, nlev), ρq=z(nh, nlev)) :
                        (u=z(nu, nlev), ρ=z(nh, nlev), ρθ=z(nh, nlev)),
            q=z(nh, nlev), ρqexpl=z(nh, nlev),
            K=z(nh, nlev), Fs=z(nu, nlev), η=z(nu, nlev + 1),
            th=z(nh, nlev + 1), tu=z(nu, nlev + 1), ub=z(nu, nlev + 1), hw2=z(nh, nlev + 1),
            thl=z(nh, nlev), tul=z(nu, nlev),   # the atomic sums go to whole arrays (or vectors), not to views
            ζ=z(nz, nlev), rz=z(nz, nlev), g=z(nu, nlev), gθ=z(nu, nlev), Fθ=z(nu, nlev),
            ηw=z(nu, nlev + 1), uw=z(nu, nlev + 1),
            hw=KernelAbstractions.zeros(backend, Float64, nh),
            y0=tracer ? (u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev), ρq=z(nh, nlev)) :
                        (u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev)))
    dev(x) = to_backend(backend, x)
    return KiteNonhydrostatic(sw, vgrid, dev(vgrid.dz), dev(vgrid.dzh), dev(interface_weights(vgrid)), dev(rayleigh),
                              work, Float64(radius), Float64(omega), Float64(beta), vertical_theta == :upwind ? 1.0 : 0.0,
                              vertical_advection == :vector_invariant,
                              horizontal_theta == :central ? nothing : upwind_faces(sw, backend),
                              horizontal_theta == :upwind_density,
                              horizontal_theta in (:dg, :dg_upwind), horizontal_theta == :dg_upwind ? 1.0 : 0.0,
                              horizontal_advection == :dg ? (tables=momentum_tables(sw, backend),
                                                             mass=weighted_mass_blocks(sw, nlev, backend)) : nothing,
                              velocity_factors(sw, backend),
                              equation_of_state == :shallow_water ? Float64(gravity) : 0.0, tracer, backend)
end

num_layers(m::KiteNonhydrostatic) = num_layers(m.vgrid)

"""
    velocity_factors(sw, backend) -> SparseBlockSolver

The dual-cell blocks of the velocity mass of `sw.Mu_solver` (owned, local numbering) as
sparse Cholesky factors: the blocks are sparse (the dofs of a kite couple only with
the dofs of the same kite) and their factors have little fill, so a solve reads a few
times fewer values than the product with the dense inverses.
"""
function velocity_factors(sw, backend)
    S = sw.Mu_solver
    count, rows = Array(S.count), Array(S.indices)
    ids = sw.distribution.u.global_ids
    blocks = Vector{Int}[]
    t = 1
    while t <= length(rows)
        push!(blocks, rows[t:t + count[t] - 1])
        t += count[t]
    end
    return sparse_block_solver(blocks, [sw.Mu[ids[b], ids[b]] for b in blocks], backend)
end

# kernels over (layers, kites or points): the layers of a point in adjacent threads (`layer_workgroup`)
function launch!(m::KiteNonhydrostatic, kernel, ndrange::Tuple{Integer, Integer}, args...)
    kernel(m.backend, layer_workgroup(ndrange[1]))(args...; ndrange)
    KernelAbstractions.synchronize(m.backend)
    return nothing
end
"Number of vertical columns: the height dofs (Radau nodes) of the horizontal space."
num_columns(m::KiteNonhydrostatic) = length(height_dofs(m.horizontal))
num_velocity_dofs(m::KiteNonhydrostatic) = length(velocity_dofs(m.horizontal))

"State `(u, w, ρ, ρθ)` of the model, with `tracer` also `ρq`, zero."
function zero_state(m::KiteNonhydrostatic)
    nh, nu, nlev = num_columns(m), num_velocity_dofs(m), num_layers(m)
    z(n, l) = KernelAbstractions.zeros(m.backend, Float64, l, n)
    y = (u=z(nu, nlev), w=z(nh, nlev + 1), ρ=z(nh, nlev), ρθ=z(nh, nlev))
    return m.tracer ? (; y..., ρq=z(nh, nlev)) : y
end

"""
    project_layers(m, f; levels=:full) -> (levels, height dofs)

L2 projection of `f(x, z)` (`x` on the unit sphere, `z` the height in m) into
the height space at every layer centre (`levels=:full`) or interface
(`levels=:half`), on the model's backend.
"""
function project_layers(m::KiteNonhydrostatic, f; levels::Symbol=:full)
    z = levels == :full ? m.vgrid.z_full : levels == :half ? m.vgrid.z_half :
        throw(ArgumentError("levels must be :full or :half"))
    H, ids = m.horizontal.H, m.horizontal.distribution.h.global_ids
    return to_backend(m.backend, reduce(vcat, [permutedims(project(H, x -> f(x, zk))[ids]) for zk in z]))
end

"""
    project_velocity(m, v) -> (nlev, velocity dofs)

L2 projection of the horizontal velocity `v(x, z)` (a tangent 3-vector in m/s)
into the velocity space at every layer centre.
"""
function project_velocity(m::KiteNonhydrostatic, v)
    U, ids = m.horizontal.U, m.horizontal.distribution.u.global_ids
    return to_backend(m.backend, reduce(vcat, [permutedims(project(U, x -> v(x, zk))[ids]) for zk in m.vgrid.z_full]))
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
    sw = m.horizontal
    ldiv!(dρ, sw.Mh_solver, row_mul!(m.work.thl, sw.B_owned, F, 1 / m.radius))
    dρ .-= (view(W, 2:size(W, 1), :) .- view(W, 1:size(W, 1) - 1, :)) ./ m.dz
    return dρ
end

"Diagonal of the height mass matrix (on the unit sphere): the weights of the column integrals."
column_weights(m::KiteNonhydrostatic) = m.horizontal.mass_h

"The local columns owned by this rank (the first ones): sums over them, summed over the ranks, are global."
owned_columns(m::KiteNonhydrostatic) = 1:m.horizontal.distribution.h.n_owned

"""
    gather_layers(m, x) -> Matrix

A layered height field (layers, `num_columns`) or velocity field
(layers, `num_velocity_dofs`) with the owned values of all ranks in the global
numbering, identical on every rank, on the host.
"""
function gather_layers(m::KiteNonhydrostatic, x)
    sw = m.horizontal
    d = size(x, 2) == num_columns(m) ? sw.distribution.h : sw.distribution.u
    n = d === sw.distribution.h ? num_dofs(sw.H) : num_dofs(sw.U)
    X = Array(x)
    return reduce(vcat, [permutedims(gather(d, X[k, :], n, sw.distribution.comm)) for k in axes(X, 1)])
end

"Update the ghost entries of every field of the state `y`."
function exchange_state!(m::KiteNonhydrostatic, y)
    d = m.horizontal.distribution
    d.comm === nothing && return y
    exchange!(y.u, d.u, d.comm)
    for x in (y.w, y.ρ, y.ρθ)
        exchange!(x, d.h, d.comm)
    end
    m.tracer && exchange!(y.ρq, d.h, d.comm)
    return y
end

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

State from `profile(λ, φ, z) = (T, p, u, v)` as for `MPASGassmannNonhydrostatic`
(on a plane `profile(x, y, z)` and `tracer(x, y, z)` with `u`, `v` along x and y):
every column in the discrete hydrostatic balance of the column step at its
Radau node (`kite_balanced_column`), the velocity `u east + v north` projected
into the velocity space per layer, `w = 0`; with a tracer `ρq = ρ q` for the
mixing ratio `tracer(λ, φ, z)` at the nodes and layer centres (default 1).
"""
function initial_state(m::KiteNonhydrostatic, profile; tracer=(λ, φ, z) -> 1.0)
    v, nlev = m.vgrid, num_layers(m)
    x = column_points(m)
    plane = m.horizontal.primal.surface != :sphere
    coordinates(x) = plane ? (x[1], x[2]) : lonlat(x)
    ρ, ρθ = zeros(nlev, length(x)), zeros(nlev, length(x))
    for (i, xi) in enumerate(x)
        λ, φ = coordinates(xi)
        ρ[:, i], ρθ[:, i] = kite_balanced_column(z -> profile(λ, φ, z), v)
    end
    u = project_velocity(m, (x, z) -> begin
        λ, φ = coordinates(x)
        east, north = plane ? (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)) : east_north(x)
        _, _, ue, vn = profile(λ, φ, z)
        ue * east + vn * north
    end)
    dev(a) = to_backend(m.backend, a)
    y = (u=u, w=KernelAbstractions.zeros(m.backend, Float64, nlev + 1, length(x)), ρ=dev(ρ), ρθ=dev(ρθ))
    m.tracer || return y
    q = [tracer(coordinates(xi)..., v.z_full[k]) for k in 1:nlev, xi in x]
    return (; y..., ρq=dev(ρ .* q))
end

"""
    vertical_stage!(m, y, y0, ys, τ)

The vertically implicit part of a stage in every column: `w`, `ρ`, `ρθ` of
`y` at `t + τ` from the base state `y0`, the stage state `ys` (θ, ρ and the
mass flux of the upwind penalty) and the explicit parts in `m.work` (`tw`, `ρexpl`,
`ρθexpl`), linearized around the Exner pressure of `y0` (`base_exner!`).
"""
function vertical_stage!(m::KiteNonhydrostatic, y, y0, ys, τ)
    w, nh, nlev = m.work, num_columns(m), num_layers(m)
    w.θ .= ys.ρθ ./ ys.ρ
    launch!(m, kite_theta_penalty_kernel!, nh, w.P, w.θ, ys.ρ, ys.w, m.below, m.theta_penalty, nlev)
    launch!(m, kite_column_kernel!, nh, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρθ, w.π0, w.γ0, w.θ, ys.ρ, w.P,
            w.tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.below, m.rayleigh, τ, m.beta, nlev)
    return y
end

"Exner pressure of the base state `y0` and its derivative by `ρθ` into the work arrays."
function base_exner!(m::KiteNonhydrostatic, y0)
    launch!(m, kite_exner_kernel!, size(y0.ρθ), m.work.π0, m.work.γ0, y0.ρθ)
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
function horizontal_tendency!(dy, m::KiteNonhydrostatic, y; momentum_only::Bool=false)
    sw, w, a = m.horizontal, m.work, m.radius
    d, nlev = sw.distribution, num_layers(m)
    exchange_state!(m, y)
    w.θ .= y.ρθ ./ y.ρ
    m.tracer && (w.q .= y.ρq ./ y.ρ)
    exner!(w.π, m, y.ρθ)
    # mass flux (not with the DG transport) and kinetic energy of every layer (with the w² part)
    m.advection === nothing && mass_flux!(w.Fs, m, y.u, y.ρ)
    layer_dot!(w.thl, m, y.u, y.u, 0.5)
    ldiv!(w.K, sw.Mh_local_solver, w.thl)
    if m.vertical_advection
        @views @. w.K += (m.dzh[1:nlev] * y.w[1:nlev, :]^2 + m.dzh[2:nlev + 1] * y.w[2:nlev + 1, :]^2) / (4 * m.dz)
        # horizontal vorticity η = ∂u/∂z - ∇w at the interior interfaces
        fill!(w.η, 0.0)
        @views @. w.η[2:nlev, :] = (y.u[2:nlev, :] - y.u[1:nlev - 1, :]) / m.dzh[2:nlev]
        if !m.dg       # with DG the -∇w part is the DG advection of w (w_advection!, w_advection_flux!)
            ldiv!(w.ηw, m.velocity_solver, row_mul!(w.uw, sw.Bt_owned, y.w, 1 / a))
            d.comm === nothing || exchange!(w.ηw, d.u, d.comm)
            @views @. w.η[2:nlev, :] -= w.ηw[2:nlev, :]
        end
    end
    if m.advection !== nothing
        dg_momentum!(dy.u, m, y)
        if !momentum_only
            dg_transport!(dy.ρ, m, y.u, y.ρ)
            dg_transport!(dy.ρθ, m, y.u, y.ρθ)
            m.tracer && dg_transport!(dy.ρq, m, y.u, y.ρq)
        end
    else
        vi_momentum!(dy.u, m, y)
        if !momentum_only
            horizontal_divergence!(dy.ρ, dy.ρθ, m, w.Fs, w.θ, y.u, y.ρ, y.ρθ)
            m.tracer && tracer_divergence!(dy.ρq, m, w.Fs, w.q, y.u, y.ρq)
        end
    end
    fill!(w.tw, 0.0)
    if m.vertical_advection
        # η ⋅ ū with the interface mean ū of u, all interfaces at once
        fill!(w.ub, 0.0)
        @views @. w.ub[2:nlev, :] = (y.u[1:nlev - 1, :] + y.u[2:nlev, :]) / 2
        layer_dot!(w.th, m, w.η, w.ub, 1.0)
        ldiv!(w.tw, sw.Mh_local_solver, w.th)
        if m.dg      # DG advection of w by ū
            w_advection!(w.th, m, w.ub, y.w, 1 / a)
            ldiv!(w.hw2, sw.Mh_solver, w.th)
            @views w.tw[2:nlev, :] .-= w.hw2[2:nlev, :]
        end
        @views @. w.tw[2:nlev, :] -= (w.K[2:nlev, :] - w.K[1:nlev - 1, :]) / m.dzh[2:nlev]
    end
    return dy
end

@kernel function layer_flux_kernel!(rhs, @Const(v), @Const(h), @Const(piola), @Const(dA), @Const(φu), @Const(iu),
                                    @Const(su), @Const(φh), @Const(ih))
    k, c = @index(Global, NTuple)
    @inbounds for q in axes(dA, 1)
        P = piola[q, c]
        vq = P * layer_vector(v, φu, iu, su, c, q, k)
        scatter_layer!(rhs, P' * ((layer_scalar(h, φh, ih, c, q, k) * dA[q, c]) * vq), φu, iu, su, c, q, k)
    end
end

"""
    layer_projection!(out, m, v, h, rhs)

`Π_U(h v)` in every layer (or interface) of the velocity field `v` and the height
field `h`, with `rhs` (same size as `v`) for the right-hand side: the layered
`weighted_projection!`, with the ghost entries updated.
"""
function layer_projection!(out, m::KiteNonhydrostatic, v, h, rhs)
    sw = m.horizontal
    Q, d = sw.quadrature, sw.distribution
    fill!(rhs, 0.0)
    launch!(m, layer_flux_kernel!, (size(v, 1), num_local_kites(d)), rhs, v, h, Q.piola, Q.dA, sw.φu, sw.iu, sw.su,
            sw.φh, sw.ih)
    ldiv!(out, m.velocity_solver, rhs)
    d.comm === nothing || exchange!(out, d.u, d.comm)
    return out
end

"`F = Π_U(ρ u)`: the mass flux of every layer."
mass_flux!(F, m::KiteNonhydrostatic, u, ρ) = layer_projection!(F, m, u, ρ, m.work.tul)

@kernel function layer_vorticity_kernel!(rhs, @Const(F), @Const(ρ), @Const(ζ), @Const(piola), @Const(dA),
                                         @Const(normal), @Const(coriolis), @Const(φu), @Const(iu), @Const(su),
                                         @Const(φh), @Const(ih), @Const(φz), @Const(iz))
    k, c = @index(Global, NTuple)
    @inbounds for q in axes(dA, 1)
        P = piola[q, c]
        v = P * layer_vector(F, φu, iu, su, c, q, k)
        pv = (layer_scalar(ζ, φz, iz, c, q, k) + coriolis[q, c]) / layer_scalar(ρ, φh, ih, c, q, k)
        scatter_layer!(rhs, -(P' * ((pv * dA[q, c]) * cross(normal[q, c], v))), φu, iu, su, c, q, k)
    end
end

"""
    vi_momentum!(du, m, y)

Velocity tendency of all layers in vector-invariant form (`horizontal_advection=:vector_invariant`),
with the mass flux `F` in `m.work.Fs`, the kinetic energy `K`, θ, π and η of `horizontal_tendency!`:

    u̇ = M_u⁻¹ (-Bᵀ K / a - ∫ (ζ/a + f)/ρ (r̂ × F) ⋅ v) - c_p Π_U(θ ∇π) - (η w)_k,  ∇π = M_u⁻¹ Bᵀ π / a,

with `horizontal_theta=:dg`, `:dg_upwind` the pressure term `M_u⁻¹ dg_gradient(F, θ, π) / a` and the
DG `w ∇w` part of `(η w)_k` (`w_advection_flux!`). One launch per term for all layers.
"""
function vi_momentum!(du, m::KiteNonhydrostatic, y)
    sw, w, a, nlev = m.horizontal, m.work, m.radius, num_layers(m)
    Q, d = sw.quadrature, sw.distribution
    # relative vorticity of every layer
    ldiv!(w.ζ, sw.Mz_solver, row_mul!(w.rz, sw.R_owned, y.u, -1 / a))
    d.comm === nothing || exchange!(w.ζ, d.z, d.comm)
    # kinetic energy gradient and PV term
    rhs = row_mul!(w.tul, sw.Bt_owned, w.K, -1 / a)
    launch!(m, layer_vorticity_kernel!, (nlev, num_local_kites(d)), rhs, w.Fs, y.ρ, w.ζ, Q.piola, Q.dA, Q.normal,
            Q.coriolis, sw.φu, sw.iu, sw.su, sw.φh, sw.ih, sw.φz, sw.iz)
    ldiv!(du, m.velocity_solver, rhs)
    # pressure gradient c_p θ ∇π
    if m.dg
        ldiv!(w.gθ, m.velocity_solver, dg_gradient!(rhs, m, w.Fs, w.θ, w.π, 1 / a))
    else
        ldiv!(w.g, m.velocity_solver, row_mul!(rhs, sw.Bt_owned, w.π, 1 / a))
        d.comm === nothing || exchange!(w.g, d.u, d.comm)
        layer_projection!(w.gθ, m, w.g, w.θ, rhs)
    end
    du .-= Cpd .* w.gθ
    # vertical coupling -η w: Π_U(η_j w_j) at all interfaces, the mean of the two of every layer
    if m.vertical_advection
        layer_projection!(w.ηw, m, w.η, y.w, w.tu)
        if m.dg
            ldiv!(w.tu, m.velocity_solver, w_advection_flux!(w.uw, m, y.w, 1 / a))
            w.ηw .-= w.tu
        end
        @views @. du -= (m.dzh[1:nlev] * w.ηw[1:nlev, :] + m.dzh[2:nlev + 1] * w.ηw[2:nlev + 1, :]) / (2 * m.dz)
    end
    return du
end

"`out = M_h⁻¹ dg_divergence(u, q) / a` in every layer: the DG transport of the density `q` by `u`."
function dg_transport!(out, m::KiteNonhydrostatic, u, q)
    th = q isa AbstractVector ? m.work.hw : m.work.thl
    dg_divergence!(th, m, u, q, 1 / m.radius)
    return ldiv!(out, m.horizontal.Mh_solver, th)
end

"""
    dg_momentum!(du, m, y)

Velocity tendency of all layers with `horizontal_advection=:dg`, from the
density-weighted momentum equation (`a` the radius)

    ∫ ρ u̇ ⋅ v = -N(u; v)/a - ∫ ρ f (r̂ × u) ⋅ v - c_p (Σ_K ∫_K ρθ v ⋅ ∇π - Σ_F ∫_F (ρθ)^ (v ⋅ n_K) [π])/a
                - Σ_j dzh_j/(2 dz_k) ∫ ρ w_j η_j ⋅ v,

`N` the DG advection `momentum_advection!`, the pressure term the adjoint of the
DG ρθ transport (`dg_gradient!` with the weight ρθ, so the exchange with the
internal energy is exact) and `η_j = (u_j - u_{j-1})/dzh_j` the vertical shear at
the interfaces of the layer (vertical advection of `u`, `momentum_sources!`), solved
with the density-weighted mass (`weighted_mass_solve!`). Uses `m.work.η`, `m.work.π`.
"""
function dg_momentum!(du, m::KiteNonhydrostatic, y)
    w, a = m.work, m.radius
    rhs = w.tul
    fill!(rhs, 0.0)
    momentum_advection!(rhs, m, y.u, y.ρ, m.dg_upwind; scale=-1 / a)
    momentum_sources!(rhs, m, y)
    add_dg_gradient!(rhs, m, y.u, y.ρθ, w.π, -Cpd / a)
    return weighted_mass_solve!(du, m, y.ρ, rhs)
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

@kernel function layer_dot_kernel!(rhs, @Const(x), @Const(y), @Const(piola), @Const(dA), @Const(φu), @Const(iu),
                                   @Const(su), @Const(φh), @Const(ih), factor)
    k, c = @index(Global, NTuple)
    @inbounds for q in axes(dA, 1)
        P = piola[q, c]
        d = factor * dA[q, c] * dot(P * layer_vector(x, φu, iu, su, c, q, k), P * layer_vector(y, φu, iu, su, c, q, k))
        for a in axes(ih, 1)
            @atomic rhs[k, ih[a, c]] += φh[a, 1, q] * d
        end
    end
end

"""
    layer_dot!(rhs, m, x, y, factor)

`rhs = factor ∫ x ⋅ y ψ` for the velocity-space fields `x`, `y` and every height
basis function `ψ`, in every layer (column) of `x` and `y`: with `x = y` and
`factor = 1/2` the right-hand side of the kinetic energy.
"""
function layer_dot!(rhs, m::KiteNonhydrostatic, x, y, factor)
    sw = m.horizontal
    Q = sw.quadrature
    fill!(rhs, 0.0)
    launch!(m, layer_dot_kernel!, (size(x, 1), num_local_kites(sw.distribution)), rhs, x, y, Q.piola, Q.dA, sw.φu,
            sw.iu, sw.su, sw.φh, sw.ih, factor)
    return rhs
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
    horizontal_divergence!(dρ, dρθ, m, F, θ, u, ρ, ρθ)

`dρ = M_h⁻¹ B F / a` and `dρθ = M_h⁻¹ B Π_U(θ F) / a` in every layer for the flux
`F`, with `horizontal_theta=:upwind` the upwind θ trace in `dρθ`
(`upwind_theta!`), with `:upwind_density` the upwind traces of `ρ` and `ρθ`
with the velocity `u` (`upwind_jump!`), which are identical for θ = 1, with
`:dg`, `:dg_upwind` the DG transport of ρθ with the flux `F` (`dg_divergence!`).
"""
function horizontal_divergence!(dρ, dρθ, m::KiteNonhydrostatic, F, θ, u, ρ, ρθ)
    sw, w, a = m.horizontal, m.work, m.radius
    th = w.thl
    density = m.upwind !== nothing && m.upwind_density
    row_mul!(th, sw.B_owned, F, 1 / a)
    density && upwind_jump!(th, m, u, ρ, 1 / a)
    ldiv!(dρ, sw.Mh_solver, th)
    if m.dg
        ldiv!(dρθ, sw.Mh_solver, dg_divergence!(th, m, F, θ, 1 / a))
        return nothing
    end
    layer_projection!(w.Fθ, m, F, θ, w.tul)
    row_mul!(th, sw.B_owned, w.Fθ, 1 / a)
    if density
        upwind_jump!(th, m, u, ρθ, 1 / a)
    elseif m.upwind !== nothing
        upwind_theta!(th, m, F, θ, 1 / a)
    end
    ldiv!(dρθ, sw.Mh_solver, th)
    return nothing
end

"""
    upwind_faces(sw, backend)

The quadrature points of the primal faces (halves of primal edges) of the local
kites in pairs (`kite_face_pairs`: point `qK` of face `fK` of kite `cK` coincides
with point `qE` of face `fE` of the neighbour kite `cE` across the primal edge),
with the primal-face tables `sw.faces` stacked for the kernels (`stacked_faces`).
"""
function upwind_faces(sw, backend)
    t, _ = gauss_legendre(length(first(sw.faces).weights))
    return stacked_faces(sw.faces, kite_face_pairs(sw, KITE_PRIMAL_FACES, t), backend)
end

@kernel function upwind_penalty_kernel!(th, @Const(F), @Const(q), U, @Const(iu), @Const(su), @Const(ih), scale)
    k, i = @index(Global, NTuple)
    @inbounds begin
        cK, qK, fK, cE, qE, fE = U.cK[i], U.qK[i], U.fK[i], U.cE[i], U.qE[i], U.fE[i]
        ûn = dot(face_vector(F, U.φ, iu, su, cK, qK, fK, k), U.ν[fK])
        jump = face_scalar(q, U.η, ih, cK, qK, fK, k) - face_scalar(q, U.η, ih, cE, qE, fE, k)
        g = scale * U.weights[qK, fK] * abs(ûn) * jump / 2
        for a in axes(ih, 1)
            @atomic th[k, ih[a, cK]] -= g * U.η[a, qK, fK]
            @atomic th[k, ih[a, cE]] += g * U.η[a, qE, fE]
        end
    end
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
    out, th, u, q = th, layered(th), layered(u), layered(q)
    num_pairs(up) > 0 && launch!(m, upwind_penalty_kernel!, (size(q, 1), num_pairs(up)), th, u, q, up, sw.iu, sw.su,
                                 sw.ih, scale)
    return out
end

"""
    upwind_theta!(th, m, F, θ, scale)

Adds the upwind trace of θ on the primal faces to the ρθ divergence `th = B Π_U(θ F)`:

    -scale ∫_F η_i^K |F ⋅ n|/2 (θ^K - θ^E)

for the test functions of both kites `K`, `E` at every face. It vanishes for θ
continuous across the primal edges and conserves ρθ; it damps the jumps of θ.
"""
upwind_theta!(th, m::KiteNonhydrostatic, F, θ, scale) = upwind_jump!(th, m, F, θ, scale)

"Trace of `q` on a primal face: the mean, with `dg_upwind` the upwind value for the normal flux `ûn`."
@inline face_trace(qK, qE, ûn, α) = (qK + qE) / 2 + α * sign(ûn) * (qK - qE) / 2

@kernel function dg_volume_kernel!(th, @Const(F), @Const(q), @Const(weights), @Const(φu), @Const(iu), @Const(su),
                                   @Const(φh), @Const(∇φh), @Const(ih), U, scale)
    k, c = @index(Global, NTuple)
    @inbounds for p in eachindex(weights)  # metric free: q F̂ ⋅ ∇̂η̂ dξ
        g = scale * weights[p] * layer_scalar(q, φh, ih, c, p, k) * layer_vector(F, φu, iu, su, c, p, k)
        for a in axes(ih, 1)
            @atomic th[k, ih[a, c]] += g[1] * ∇φh[a, 1, 1, p] + g[2] * ∇φh[a, 1, 2, p]
        end
    end
    @inbounds for f in axes(U.weights, 2), p in axes(U.weights, 1)   # the interior trace on the primal faces
        ûn = dot(face_vector(F, U.φ, iu, su, c, p, f, k), U.ν[f])
        g = scale * U.weights[p, f] * face_scalar(q, U.η, ih, c, p, f, k) * ûn
        for a in axes(ih, 1)
            @atomic th[k, ih[a, c]] -= g * U.η[a, p, f]
        end
    end
end

@kernel function dg_trace_kernel!(th, @Const(F), @Const(q), U, @Const(iu), @Const(su), @Const(ih), α, scale)
    k, i = @index(Global, NTuple)
    @inbounds begin
        cK, qK, fK, cE, qE, fE = U.cK[i], U.qK[i], U.fK[i], U.cE[i], U.qE[i], U.fE[i]
        ûn = dot(face_vector(F, U.φ, iu, su, cK, qK, fK, k), U.ν[fK])
        vK, vE = face_scalar(q, U.η, ih, cK, qK, fK, k), face_scalar(q, U.η, ih, cE, qE, fE, k)
        q̂ = face_trace(vK, vE, ûn, α)
        w = scale * U.weights[qK, fK] * ûn
        for a in axes(ih, 1)
            @atomic th[k, ih[a, cK]] -= w * (q̂ - vK) * U.η[a, qK, fK]
            @atomic th[k, ih[a, cE]] += w * (q̂ - vE) * U.η[a, qE, fE]
        end
    end
end

"""
    dg_divergence!(th, m, F, q, scale)

DG transport of a density `ρq = ρ q` with the kite mass flux `F` of every layer
(`horizontal_theta=:dg`, `:dg_upwind`), into the height rows `th`:

    th_i = scale (Σ_K ∫_K q F ⋅ ∇η_i - Σ_F ∫_F q̂ (F ⋅ n_K) (η_i^K - η_i^E)),

`q̂` the central or upwind trace (`face_trace`). For `q = 1` it is `B F`, so
`ρq = ρ` is kept; with the central trace it is skew in `q` for a divergence-free
`F`, with the upwind trace it damps the jumps of `q`.
"""
function dg_divergence!(th, m::KiteNonhydrostatic, F, q, scale)
    sw, up = m.horizontal, m.upwind
    out, F, q = th, layered(F), layered(q)
    th = layered(th)
    fill!(th, 0.0)
    L = size(q, 1)
    # volume term and the interior trace q_K on every face, then the trace corrected to q̂
    launch!(m, dg_volume_kernel!, (L, num_local_kites(sw.distribution)), th, F, q, sw.quadrature.weights, sw.φu,
            sw.iu, sw.su, sw.φh, sw.∇φh, sw.ih, up, scale)
    num_pairs(up) > 0 &&
        launch!(m, dg_trace_kernel!, (L, num_pairs(up)), th, F, q, up, sw.iu, sw.su, sw.ih, m.dg_upwind, scale)
    return out
end

@kernel function dg_gradient_volume_kernel!(tu, @Const(θ), @Const(π), @Const(weights), @Const(φu), @Const(iu),
                                            @Const(su), @Const(φh), @Const(∇φh), @Const(ih), scale)
    k, c = @index(Global, NTuple)
    @inbounds for q in eachindex(weights)  # metric free: θ φ̂ ⋅ ∇̂π̂ dξ
        g = scale * weights[q] * layer_scalar(θ, φh, ih, c, q, k) * layer_gradient(π, ∇φh, ih, c, q, k)
        scatter_layer!(tu, g, φu, iu, su, c, q, k)
    end
end

@kernel function dg_gradient_face_kernel!(tu, @Const(F), @Const(θ), @Const(π), U, @Const(iu), @Const(su), @Const(ih),
                                          α, scale)
    k, i = @index(Global, NTuple)
    @inbounds begin
        cK, qK, fK, cE, qE, fE = U.cK[i], U.qK[i], U.fK[i], U.cE[i], U.qE[i], U.fE[i]
        ûn = dot(face_vector(F, U.φ, iu, su, cK, qK, fK, k), U.ν[fK])
        θ̂ = face_trace(face_scalar(θ, U.η, ih, cK, qK, fK, k), face_scalar(θ, U.η, ih, cE, qE, fE, k), ûn, α)
        jump = face_scalar(π, U.η, ih, cK, qK, fK, k) - face_scalar(π, U.η, ih, cE, qE, fE, k)
        # the normal trace on the face is single valued: kite K's basis suffices
        scatter_face!(tu, -(scale * U.weights[qK, fK] * θ̂ * jump) * U.ν[fK], U.φ, iu, su, cK, qK, fK, k)
    end
end

"""
    dg_gradient!(tu, m, F, θ, π, scale)

The pressure gradient adjoint to `dg_divergence!`, into the velocity rows `tu`:

    tu_v = scale (Σ_K ∫_K θ φ_v ⋅ ∇π - Σ_F ∫_F θ̂ (φ_v ⋅ n_K) (π^K - π^E)),

with the trace `θ̂` of the flux `F`, so that `πᵀ dg_divergence(F, θ) = Fᵀ tu`
and the exchange of kinetic and internal energy is exact.
"""
function dg_gradient!(tu, m::KiteNonhydrostatic, F, θ, π, scale)
    fill!(tu, 0.0)
    return add_dg_gradient!(tu, m, F, θ, π, scale)
end

"Adds `dg_gradient!` to `tu`."
function add_dg_gradient!(tu, m::KiteNonhydrostatic, F, θ, π, scale)
    sw, up = m.horizontal, m.upwind
    out, F, θ, π = tu, layered(F), layered(θ), layered(π)
    tu = layered(tu)
    L = size(θ, 1)
    launch!(m, dg_gradient_volume_kernel!, (L, num_local_kites(sw.distribution)), tu, θ, π, sw.quadrature.weights,
            sw.φu, sw.iu, sw.su, sw.φh, sw.∇φh, sw.ih, scale)
    num_pairs(up) > 0 && launch!(m, dg_gradient_face_kernel!, (L, num_pairs(up)), tu, F, θ, π, up, sw.iu, sw.su,
                                 sw.ih, m.dg_upwind, scale)
    return out
end

@kernel function w_advection_volume_kernel!(th, @Const(u), @Const(w), @Const(weights), @Const(φu), @Const(iu),
                                            @Const(su), @Const(φh), @Const(∇φh), @Const(ih), scale)
    k, c = @index(Global, NTuple)
    @inbounds for q in eachindex(weights)  # metric free: (û ⋅ ∇̂w) ψ dξ
        g = scale * weights[q] * dot(layer_vector(u, φu, iu, su, c, q, k), layer_gradient(w, ∇φh, ih, c, q, k))
        for a in axes(ih, 1)
            @atomic th[k, ih[a, c]] += g * φh[a, 1, q]
        end
    end
end

@kernel function w_advection_face_kernel!(th, @Const(u), @Const(w), U, @Const(iu), @Const(su), @Const(ih), α, scale)
    k, i = @index(Global, NTuple)
    @inbounds begin
        cK, qK, fK, cE, qE, fE = U.cK[i], U.qK[i], U.fK[i], U.cE[i], U.qE[i], U.fE[i]
        ûn = dot(face_vector(u, U.φ, iu, su, cK, qK, fK, k), U.ν[fK])
        jump = face_scalar(w, U.η, ih, cK, qK, fK, k) - face_scalar(w, U.η, ih, cE, qE, fE, k)
        s = scale * U.weights[qK, fK] * jump / 2
        gc, gu = s * ûn, s * α * abs(ûn)
        for a in axes(ih, 1)
            @atomic th[k, ih[a, cK]] -= (gc - gu) * U.η[a, qK, fK]
            @atomic th[k, ih[a, cE]] -= (gc + gu) * U.η[a, qE, fE]
        end
    end
end

"""
    w_advection!(th, m, u, w, scale)

Horizontal advection of `w` (broken across the primal edges) by the velocity `u`
in DG advective form, into the height rows `th` (`horizontal_theta=:dg`, `:dg_upwind`):

    th_i = scale (Σ_K ∫_K (u ⋅ ∇w) ψ_i - Σ_F ∫_F (u ⋅ n_K) [w] {ψ_i} + α Σ_F ∫_F |u ⋅ n|/2 [w] [ψ_i]),

`[w] = w^K - w^E`, `α = 1` for `:dg_upwind`. The central part is skew in `w` for a
divergence-free `u`; the projected gradient `M_u⁻¹ Bᵀ w` it replaces is not (its
eigenvalues are those of `B Π_U(ρ u)`). `u` and `w` may be layered (interfaces).
"""
function w_advection!(th, m::KiteNonhydrostatic, u, w, scale)
    sw, up = m.horizontal, m.upwind
    out, u, w = th, layered(u), layered(w)
    th = layered(th)
    fill!(th, 0.0)
    L = size(w, 1)
    launch!(m, w_advection_volume_kernel!, (L, num_local_kites(sw.distribution)), th, u, w, sw.quadrature.weights,
            sw.φu, sw.iu, sw.su, sw.φh, sw.∇φh, sw.ih, scale)
    num_pairs(up) > 0 && launch!(m, w_advection_face_kernel!, (L, num_pairs(up)), th, u, w, up, sw.iu, sw.su, sw.ih,
                                 m.dg_upwind, scale)
    return out
end

@kernel function w_flux_volume_kernel!(tu, @Const(w), @Const(weights), @Const(φu), @Const(iu), @Const(su),
                                       @Const(φh), @Const(∇φh), @Const(ih), scale)
    k, c = @index(Global, NTuple)
    @inbounds for q in eachindex(weights)
        g = scale * weights[q] * layer_scalar(w, φh, ih, c, q, k) * layer_gradient(w, ∇φh, ih, c, q, k)
        scatter_layer!(tu, g, φu, iu, su, c, q, k)
    end
end

@kernel function w_flux_face_kernel!(tu, @Const(w), U, @Const(iu), @Const(su), @Const(ih), scale)
    k, i = @index(Global, NTuple)
    @inbounds begin
        cK, qK, fK, cE, qE, fE = U.cK[i], U.qK[i], U.fK[i], U.cE[i], U.qE[i], U.fE[i]
        wK, wE = face_scalar(w, U.η, ih, cK, qK, fK, k), face_scalar(w, U.η, ih, cE, qE, fE, k)
        scatter_face!(tu, -(scale * U.weights[qK, fK] * (wK - wE) * (wK + wE) / 2) * U.ν[fK], U.φ, iu, su,
                      cK, qK, fK, k)
    end
end

"""
    w_advection_flux!(tu, m, w, scale)

The momentum term adjoint to the central part of `w_advection!` (the `w ∇w` of
`-η w`), into the velocity rows `tu`: `tu_v = ∂/∂u_v` of `wᵀ w_advection(u, w)`,

    tu_v = scale (Σ_K ∫_K (φ_v ⋅ ∇w) w - Σ_F ∫_F (φ_v ⋅ n_K) [w] {w}),

so that `uᵀ tu = wᵀ w_advection(u, w)` (central) and the energy exchange of the
vertical coupling stays exact.
"""
function w_advection_flux!(tu, m::KiteNonhydrostatic, w, scale)
    sw, up = m.horizontal, m.upwind
    out, w = tu, layered(w)
    tu = layered(tu)
    fill!(tu, 0.0)
    L = size(w, 1)
    launch!(m, w_flux_volume_kernel!, (L, num_local_kites(sw.distribution)), tu, w, sw.quadrature.weights, sw.φu,
            sw.iu, sw.su, sw.φh, sw.∇φh, sw.ih, scale)
    num_pairs(up) > 0 &&
        launch!(m, w_flux_face_kernel!, (L, num_pairs(up)), tu, w, up, sw.iu, sw.su, sw.ih, scale)
    return out
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
    horizontal_tendency!(w.dy, m, ys; momentum_only=true)
    @. y.u = y0.u + τ * w.dy.u
    d = m.horizontal.distribution
    d.comm === nothing || exchange!(y.u, d.u, d.comm)
    if m.advection !== nothing      # DG transport with the pointwise velocity, all layers at once
        dg_transport!(w.ρexpl, m, y.u, ys.ρ)
        dg_transport!(w.ρθexpl, m, y.u, ys.ρθ)
        m.tracer && dg_transport!(w.ρqexpl, m, y.u, ys.ρq)
    else                            # the mass flux of the new velocity with ρ of the stage state, all layers
        mass_flux!(w.Fs, m, y.u, ys.ρ)
        horizontal_divergence!(w.ρexpl, w.ρθexpl, m, w.Fs, w.θ, y.u, ys.ρ, ys.ρθ)
        m.tracer && tracer_divergence!(w.ρqexpl, m, w.Fs, w.q, y.u, ys.ρq)
    end
    @. w.ρexpl = y0.ρ + τ * w.ρexpl
    @. w.ρθexpl = y0.ρθ + τ * w.ρθexpl
    vertical_stage!(m, y, y0, ys, τ)
    if m.tracer
        @. w.ρqexpl = y0.ρq + τ * w.ρqexpl
        launch!(m, kite_tracer_column_kernel!, num_columns(m), y.ρq, w.ρqexpl, w.q, ys.ρ, ys.w, y.w, y0.w, m.dz,
                m.below, m.theta_penalty, τ, m.beta, num_layers(m))
    end
    return y
end

"""
    tracer_divergence!(dρq, m, F, q, u, ρq)

Horizontal tracer transport of every layer, `dρq = M_h⁻¹ B Π_U(q F) / a` with the
mass flux `F`, as `ρθ` (with the same upwind or DG option): for `q = 1` it is the
mass divergence.
"""
function tracer_divergence!(dρq, m::KiteNonhydrostatic, F, q, u, ρq)
    sw, w, a = m.horizontal, m.work, m.radius
    th = w.thl
    m.dg && return ldiv!(dρq, sw.Mh_solver, dg_divergence!(th, m, F, q, 1 / a))
    layer_projection!(w.Fθ, m, F, q, w.tul)
    row_mul!(th, sw.B_owned, w.Fθ, 1 / a)
    if m.upwind !== nothing
        m.upwind_density ? upwind_jump!(th, m, u, ρq, 1 / a) : upwind_theta!(th, m, F, q, 1 / a)
    end
    return ldiv!(dρq, sw.Mh_solver, th)
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
    return wicker_skamarock!(m, y, y0, ys, dt)
end

# the diagnostics of the dycore interface (ShallowWater): the arrays are layers × points
lowest_layer(m::KiteNonhydrostatic, x) = Array(x)[1, :]

"Temperature of the lowest layer (K) at the points of the kites."
function lowest_temperature(m::KiteNonhydrostatic, y)
    ρ, ρθ = lowest_layer(m, y.ρ), lowest_layer(m, y.ρθ)
    return (Rd .* ρθ ./ P0) .^ (Rd / Cvd) .* ρθ ./ ρ
end

"Surface pressure (Pa): the pressure of the lowest layer extrapolated hydrostatically to `z = 0`."
function surface_pressure(m::KiteNonhydrostatic, y)
    ρθ, T = lowest_layer(m, y.ρθ), lowest_temperature(m, y)
    return P0 .* (Rd .* ρθ ./ P0) .^ (Cpd / Cvd) .* exp.(GRAV * m.vgrid.z_full[1] ./ (Rd .* T))
end

vertical_velocity(m::KiteNonhydrostatic, y) = Array(y.w)

"Total mass `a² Σ_k dz_k Σ_i W_i ρ_ki` (`W = column_weights`, the owned columns of this rank)."
function total_mass(m::KiteNonhydrostatic, y)
    o = owned_columns(m)
    return m.radius^2 * sum(Array(view(y.ρ, :, o)) * Array(view(column_weights(m), o)) .* m.vgrid.dz)
end
