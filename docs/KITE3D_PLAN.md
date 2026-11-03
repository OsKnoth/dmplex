# Plan: 3D kite model as a direct product

The nonhydrostatic compressible Euler equations on the extruded kite mesh. The spaces are tensor
products of the 2D kite spaces of `NonlinearKiteShallowWater` with 1D spaces on the vertical
grid, in the split-order form of Witt, Bendall & Shipton (arXiv:2603.16571), with Lorenz staggering
as in ICON and `TRiSKNonhydrostatic`. The 3D dual-cell method of Codecasa, Kapidani, Schöberl &
Wess (arXiv:2604.13921, Maxwell on tetrahedra with barycentric duals) is not used: its dual cells
cut through the columns, so there is no column structure for a HEVI solver.

## Spaces

Horizontal factors (2D kite spaces, order k):
- H_h = `KiteRadau(k)`: Q_k on the kites with Gauss–Radau nodes, broken across primal edges.
  Diagonal mass matrix.
- U_h = `KiteBrokenRT(k)`: broken H(div) on the dual cells. Block-diagonal mass matrix, one block
  per dual cell.

Vertical factors (order v on the layers of `VerticalGrid`): DG_v per layer, CG_{v+1} continuous
across the interfaces.

| Field | Space | Lowest order (k = 1, v = 0) |
|---|---|---|
| ρ | H_h ⊗ DG_v | Q2 Radau × constant per layer |
| ρθ | H_h ⊗ DG_v (Lorenz: with ρ) | as ρ |
| w | H_h ⊗ CG_{v+1} | Q2 Radau × linear, values at the interfaces |
| u_h | U_h ⊗ DG_v | RT1 kite × constant per layer |

Together (u_h, w) ∈ V1 = (U_h ⊗ DG_v) ⊕ (H_h ⊗ CG_{v+1}) and ρ, ρθ ∈ V2 = H_h ⊗ DG_v: the
divergence maps V1 into V2, so the discrete continuity equation is exact per kite column.

**Lorenz staggering.** θ lives in the layers with ρ, and the pressure gradient at the w points
interpolates θ to the interfaces. This is the layout of ICON and of `TRiSKNonhydrostatic`, with its
computational mode in the vertical; the upwind θ at the interfaces of `theta_interface_kernel!`
(ICON-IAP `ltheta_up_vert`) carries over. Higher-order reconstruction of θ inside the DG_v layers is
possible later. That is the reason for Lorenz rather than Charney–Phillips (θ with w), which the
split-order paper uses.

## Mass matrices and columns

The Radau nodes of H_h are both the dofs and the quadrature points, so the horizontal mass
matrix of ρ, ρθ, w is diagonal. The vertical terms (∂_z of the mass and ρθ fluxes, the vertical
pressure gradient, buoyancy) involve no horizontal derivative, so they decouple into **independent
columns, one per horizontal Radau node**. For v = 0 a column is the same tridiagonal system in w,
after eliminating ρ and ρθ, as in `vertical_solve_kernel!`. For v > 0 it is a block-banded system
of width v + 1.

The horizontal momentum has the mass matrix (dual-cell block of U_h) ⊗ (layer block of DG_v), so
it can be inverted block-diagonally, which allows explicit horizontal terms.

## Time stepping (HEVI)

As in `TRiSKNonhydrostatic`: Wicker–Skamarock RK3. In each stage:
1. **Explicit horizontal terms**, per layer, with the 2D kite operators of
   `NonlinearKiteShallowWater` and ρ in place of h:
   - mass and ρθ fluxes in U_h;
   - vorticity in the continuous kite space and the PV term;
   - kinetic energy;
   - horizontal pressure gradient c_p θ ∇π, tested with U_h.
2. **Implicit vertical terms**, per Radau column: w, ρ, ρθ at t + τ, linearized around the Exner
   pressure of the base state. π0 and κπ0/ρθ0 are computed once per step.
3. **Vertical advection of u_h and w**, explicit, in the ICON-IAP mass-flux form.

## Steps

Done (`src/Nonhydrostatic/KiteNonhydrostatic.jl`, tests in `test/extruded.jl`): steps 1 and 2 at
lowest vertical order, serial. The state is `(u, w, ρ, ρθ)` as arrays (dofs × levels).

1. **Extruded kite spaces** (done: `zero_state`, `project_layers`, `project_velocity`,
   `mass_divergence!`; mass to round-off, column sums exact): dof numbering of H_h ⊗ DG_v, H_h ⊗ CG_{v+1} and U_h ⊗ DG_v on top of
   `create_kite_mesh` and `VerticalGrid`, with layer-innermost or point-innermost layouts for the
   GPU (blocks of points × levels, as `BLOCK` in `TRiSKNonhydrostatic`). Tests: dimensions, and the
   divergence V1 → V2 exact per column.
2. **Column solver for v = 0** (done: `column_points`, `initial_state` with `balanced_column`,
   `vertical_stage!`, `vertical_step!`; rest state w < 1e-12 with `:upwind` and `:centred`, column
   mass to round-off; `:iap` leaves rest as in `TRiSKNonhydrostatic`): reuse `vertical_solve_kernel!` and `theta_interface_kernel!` with
   one column per Radau node. Tests: hydrostatic rest state at rest, and a column at rest staying
   at rest with upwind θ.
3. **Horizontal terms per layer** (done without vertical advection: `horizontal_tendency!`, `stage!`,
   `step!`; F = Π_U(ρu), Fθ = Π_U(θF), pressure gradient −c_p Π_U(θ∇π) with ∇π = M_u⁻¹Bᵀπ/a, which
   makes the energy exchange exact (residual 1e-16 of the summed products); rest state at rest.
   **Open: a sub-cell instability**, see below) from the kite shallow-water operators. Tests: mass conservation
   to round-off, energy exchange between the horizontal and vertical terms.
4. **Test cases:**
   - inertia-gravity wave (Skamarock–Klemp on the plane with `create_periodic_plane`, the
     Baldauf–Brdar sphere version);
   - the baroclinic wave, compared with `TRiSKNonhydrostatic` on the same hexagon mesh.
5. **GPU** with KernelAbstractions, as the 2D kite model.
6. **Higher vertical order** (v = 1) and the θ reconstruction in the layers.

## Open: sub-cell instability on non-affine kites (2026-11-03)

The full step (horizontal explicit, columns implicit) is unstable on the hexagon and triangle meshes:
small noise on a stratified rest state grows by about ×6 every 2 h (hexagon 3, 10 layers), the
unperturbed baroclinic wave blows up within 16 h. What is known:

- **Semi-discrete:** the same growth at dt = 15, 50, 75, 150 s; the explicit acoustic limit is
  between 250 and 300 s on hexagon 3.
- **Not the vertical solver alone:** `vertical_step!` with the same noise stays at |w| ~ 1e-13;
  `:centred`, `:upwind` θ and β = 0.5, 0.65, 1 all grow.
- **Not the horizontal terms alone:** one layer (no vertical motion) decays, two layers stay bounded.
- **Not rotation:** the same growth with Ω = 0.
- **Not the vertical mass of w:** the consistent CG1 mass (tridiagonal, now the default) and the
  lumped one grow alike.
- **The 2D kite shallow-water model** runs on planes of all three grid types (quadrilaterals,
  triangles, hexagons), and the one-layer 3D model is stable, so the horizontal pair alone is fine.
- **Mesh:** grows on hexagons (plain and spring-optimized) and faster on triangles, largest near
  the pentagons; bounded on the cubed sphere, whose kites are affine (squares). The kites of
  hexagons and triangles are not affine.
- **Order of the height space:** `KiteRadau(0)` (one column per primal cell, as TRiSK) is stable,
  `KiteRadau(1)` (13 columns per hexagon) grows slowly (×2 per 2 h), `KiteRadau(2)` (37 columns) fast.
- **Structure:** 99 % of the mode is inside the primal cells (zero cell mean), largest in the
  lowest layers.
- **Energy:** the total energy stays constant (it decreases slightly by the off-centring) while
  the kinetic energy grows by 10³: the mode releases potential and internal energy, so the
  conserved energy is not a sufficient stability bound for these sub-cell modes. `TRiSKNonhydrostatic`
  on the same vertical grid and noise stays bounded without any diffusion.

Next: the linearized operator on a small mesh (hexagon 1, a few layers) to find the unstable
eigenmodes and the term that makes the pseudo-energy indefinite.
