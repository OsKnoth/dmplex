# Kite finite elements: shallow water and the 3D compressible Euler equations

A short summary. The full 3D discretization as implemented is in
[KITE3D_DISCRETIZATION.md](KITE3D_DISCRETIZATION.md), its status and tests in
[KITE3D_PLAN.md](KITE3D_PLAN.md).

## Mesh

Every primal cell (hexagon, pentagon, triangle or square) is split into kites: primal vertex,
edge midpoint, cell centre, edge midpoint (`create_kite_mesh`). The kites around a primal vertex
form its dual cell. A kite has two kinds of edges:

- **primal faces** (halves of the primal edges), between two primal cells;
- **dual faces** (centre–edge-midpoint edges), between two dual cells, inside a primal cell.

## Spaces (degree k)

| field | space | continuity | mass matrix |
|---|---|---|---|
| height h (3D: ρ, ρθ) | `KiteRadau(k+1)`: Q_{k+1} with tensor Gauss–Radau nodes | continuous inside a primal cell, broken across primal faces | diagonal (Radau nodes as quadrature) |
| velocity u | `KiteBrokenRT(k)` (RTCF_k, contravariant Piola), or `KiteDualRadau(k)` | normal component continuous across primal faces, broken across dual faces | block diagonal, one block per dual cell |
| vorticity ζ | `Lagrange(k+1)` | continuous | diagonal (GLL lumped) |

The Radau nodes never lie on primal faces, so every height dof belongs to exactly one primal cell.

## Shallow water (`NonlinearKiteShallowWater`)

The divergence pairs the broken height space with the velocity space:

$$
B_{ij} = \sum_K\int_K \phi_j\cdot\nabla\eta_i - \sum_K\sum_{F\subset\partial K\cap\text{primal}} \int_F \eta_i^K\,\phi_j\cdot n_K,
\qquad \mathbf 1^TB = 0,
$$

and the gradient is its adjoint, $G = -B^T$ (integration by parts moves the face terms onto the
dual faces, where η is single valued). With $\Pi_U(hv) = M_U^{-1}(\int h\,v\cdot\phi_j)_j$,
$\langle\zeta,\gamma\rangle = -\int u\cdot\mathrm{rot}\,\gamma$ and $\langle K,\eta\rangle = \int |u|^2/2\,\eta$:

$$
F = \Pi_U(hu),\qquad
M_U\dot u = -\int \frac{\zeta+f}{h}(\hat r\times F)\cdot\phi_j - B^T(gh + K),\qquad
M_H\dot h = BF .
$$

- Mass is conserved exactly ($F\cdot n$ is single valued on the primal faces).
- Energy $\int h|u|^2/2 + g\,h^TM_Hh/2$ is conserved by the semi-discretization: the PV term
  vanishes pointwise, and $F^TM_U\dot u + (gh+K)^TM_H\dot h = 0$ by $G=-B^T$.
- $B$ and the curl are metric free (contravariant Piola); all nonlinear terms use one quadrature
  rule with precomputed metric terms. The velocity mass must use the same rule, so that
  $\Pi_U(c\,v) = c\,v$ also on non-affine kites.

## Extension to the 3D compressible Euler equations (`KiteNonhydrostatic`)

A direct product of the 2D spaces with the vertical layers, Lorenz staggering:

| field | space |
|---|---|
| ρ, ρθ | `KiteRadau` ⊗ DG0 (layer values) |
| w | `KiteRadau` ⊗ CG1 (interface values) |
| u_h | velocity space ⊗ DG0 |

Every Radau node carries one vertical column. Per layer the horizontal terms are those of shallow
water with ρ in place of h, θ = ρθ/ρ and the Exner pressure π:

$$
F = \Pi_U(\rho u),\quad F_\theta = \Pi_U(\theta F),\quad
M_U\dot u = -\int\frac{\zeta+f}{\rho}(\hat r\times F)\cdot\phi_j - B^TK/a - c_pM_U\Pi_U(\theta M_U^{-1}B^T\pi/a),
$$
$$
M_H\dot\rho = BF/a,\qquad M_H\dot{(\rho\theta)} = BF_\theta/a,
$$

plus the vertical coupling of the vector-invariant form (−ηw in the u equation, η·ṽ − ∂K/∂z in the
w equation). The pressure pair is energy consistent: $\pi^TBM_U^{-1}\Theta F = F^T\Theta M_U^{-1}B^T\pi$
with the symmetric $\Theta = (\int\theta\,\phi_i\cdot\phi_j)$.

The vertical terms (acoustic, buoyancy, gravity, vertical ρ and ρθ fluxes) are implicit per
column with the kernels of `TRiSKNonhydrostatic`: one tridiagonal system for w per column,
linearized around the Exner pressure of the base state, consistent CG1 mass for w. Time stepping:
Wicker–Skamarock RK3 with a HEVI split, horizontal mass fluxes forward–backward.

**What changes from 2D to 3D.** In shallow water no field is transported with the mass flux. In 3D,
θ is, and θ is broken across the primal faces. The central flux $B\,\Pi_U(\theta F)$ never sees the
jump [θ], and node-scale θ modes grow on the jet (faster with resolution). The upwind trace
(`horizontal_theta=:upwind`), $-\int_e\eta_i|F\cdot n|/2\,(\theta^K-\theta^E)$ on every primal face,
removes that growth on the jet, but not yet on the thermal-wind state. The open problems are listed
in [KITE3D_PLAN.md](KITE3D_PLAN.md).
