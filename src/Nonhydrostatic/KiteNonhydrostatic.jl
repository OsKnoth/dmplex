# Nonhydrostatic model on the extruded kite mesh: the spaces of the 2D kite model times the
# layers of a vertical grid (docs/KITE3D_PLAN.md). Step 1: spaces, state, projections and the
# mass divergence.

"""
    KiteNonhydrostatic(primal, vgrid, k=1; height_order=k + 1, radius=6.37122e6, omega=7.292e-5,
                       backend=CPU())

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
the vertically implicit terms (`num_columns`). Horizontal operators act layer
by layer with the 2D kite operators (scaled from the unit sphere by `radius`).
"""
struct KiteNonhydrostatic{SW, V, BE}
    horizontal::SW
    vgrid::VerticalGrid
    dz::V
    radius::Float64
    omega::Float64
    backend::BE
end

function KiteNonhydrostatic(primal::Plex, vgrid::VerticalGrid, k::Integer=1; height_order::Integer=k + 1,
                            radius::Real=6.37122e6, omega::Real=7.292e-5, backend=KernelAbstractions.CPU())
    sw = NonlinearKiteShallowWater(primal, k; height_order, coriolis=x -> 2omega * x[3], backend)
    return KiteNonhydrostatic(sw, vgrid, to_backend(backend, vgrid.dz), Float64(radius), Float64(omega), backend)
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
