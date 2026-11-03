# Discretization of `KiteNonhydrostatic` (as implemented, 2026-11-03)

This is the state of `src/Nonhydrostatic/KiteNonhydrostatic.jl` after steps 1–3 of
[KITE3D_PLAN.md](KITE3D_PLAN.md): serial, lowest vertical order, no vertical advection. The open
sub-cell instability is described at the end of the plan.

## Equations

Dry compressible Euler equations, shallow atmosphere over flat ground, in height coordinates, with
prognostic variables $u_h$ (horizontal velocity), $w$, $\rho$ and $\rho\theta$, and the Exner
pressure $\pi = (R\rho\theta/p_0)^{R/c_v}$:

$$
\partial_t u_h = -(\zeta + f)\,\hat r \times u_h - \nabla_h K - c_p\theta\nabla_h\pi,\qquad
\partial_t w = -c_p\theta\,\partial_z\pi - g,
$$

$$
\partial_t \rho = -\nabla_h\cdot(\rho u_h) - \partial_z(\rho w),\qquad
\partial_t(\rho\theta) = -\nabla_h\cdot(\theta\rho u_h) - \partial_z(\theta\rho w).
$$

Here $K = |u_h|^2/2$ and $\zeta = \hat r\cdot\nabla\times u_h$. Not yet included: vertical
advection of $u_h$, all advection of $w$, and $w$ in the kinetic energy. Rayleigh damping of $w$
acts above `damping_height`.

## Mesh and spaces

**Horizontal.** The kite mesh of the primal mesh (`create_kite_mesh`) lies on the unit sphere, and
every horizontal operator is scaled by the radius $a$. Each primal cell is split into kites (primal
vertex, edge midpoint, centre, edge midpoint); the kites around a primal vertex form its dual cell.
The 2D spaces are those of `NonlinearKiteShallowWater(primal, 1)`:

- $H$ = `KiteRadau(2)`: $Q_2$ on every kite with tensor Gauss–Radau nodes, three per direction,
  including the end point at the centre side. The nodes lie inside the kites, on the centre–edge
  kite edges and at the centre, never on primal edges. A function is continuous inside a primal cell
  and broken across primal edges. There are 37 dofs per hexagon.
- $U$ = `KiteBrokenRT(1)`: RTCF$_1$ on every kite (contravariant Piola map), with normal continuity
  across the halves of primal edges (inside a dual cell) and none across the centre–edge kite edges
  (the dual-cell boundaries).
- $Z$ = `Lagrange(2)` on the kites, continuous, for the relative vorticity.

**Vertical.** The grid has layers $k = 1..n$ of thickness $\Delta z_k$ with centres $z_k$, and
interfaces $j = 1..n+1$ at heights $z_{j-1/2}$. Let $\Delta z_j^h = z_j - z_{j-1}$ (the distance of
the layer centres around interface $j$), and let
$\varphi_j = w^{low}_j\varphi_{j-1} + (1-w^{low}_j)\varphi_j$ be the linear interpolation to the
interface.

**Fields** (arrays of horizontal dofs × levels):

| field | space | values |
|---|---|---|
| $\rho$, $\rho\theta$ | $H\otimes DG_0$ | nodal values per layer, $(n_H, n)$ |
| $w$ | $H\otimes CG_1$ | nodal values per interface, $(n_H, n+1)$, $w_1 = w_{n+1} = 0$ |
| $u_h$ | $U\otimes DG_0$ | RT dofs per layer, $(n_U, n)$ |

Every height dof $i$ (a Radau node $x_i$, `column_points`) carries one column of $\rho$, $\rho\theta$
and $w$.

## Mass matrices

- $M_H$: **diagonal**. It uses the Radau nodes as quadrature (`lumped=true`),
  $M_{H,ii} = \sum_{K\ni x_i} \omega_i\,|J_K(x_i)|$. This is exact only for affine kites.
- $M_U$: exact quadrature, **block diagonal** with one block per dual cell (the kites around a
  primal vertex), inverted exactly.
- $M_Z$: GLL lumped, diagonal.
- Vertical, $\rho$ and $\rho\theta$ ($DG_0$): $\mathrm{diag}(\Delta z_k)$, exact.
- Vertical, $w$ ($CG_1$): **consistent** by default (`vertical_mass=:consistent`), tridiagonal, with
  row $j$ equal to $(\Delta z_{j-1}, 2(\Delta z_{j-1}+\Delta z_j), \Delta z_j)/6$. Its bandwidth is
  that of the coupling through $\rho$ and $\rho\theta$, so the column system stays tridiagonal.
  `:lumped` gives $\mathrm{diag}(\Delta z_j^h)$, pointwise at the interfaces as in
  `TRiSKNonhydrostatic`. The rows sum to $\Delta z^h_j$ in both cases, so gravity and the rest state
  are the same.

The 3D mass matrices are the tensor products: $M_H\otimes\mathrm{diag}(\Delta z)$ for $\rho$ and
$\rho\theta$, $M_H\otimes\mathrm{diag}(\Delta z^h)$ for $w$, and $M_U\otimes\mathrm{diag}(\Delta z)$
for $u_h$.

## Horizontal operators (per layer)

**Kite divergence** $B: U\to H$ (`assemble_kite_divergence`, degree 6):

$$
B_{ij} = \sum_K \int_K \phi_j\cdot\nabla\eta_i \;-\; \sum_K\sum_{F\subset\partial K\cap\text{primal edges}} \int_F \eta_i^{K}\,(\phi_j\cdot n_K),
$$

with $\mathbf 1^T B = 0$. Then $-\nabla\cdot F \approx M_H^{-1}BF/a$, and the gradient is the
adjoint, $\nabla p \approx M_U^{-1}B^T p/a$.

**Weighted projection** (one 3×3 Gauss rule per kite, `KiteQuadrature`, degree 4): for $h\in H$
and $v\in U$,

$$
\Pi_U(h v) = M_U^{-1}\Big(\int h\,v\cdot\phi_j\Big)_j .
$$

**Vorticity and kinetic energy:**

$$
M_Z\zeta = -\frac1a\Big(\int u\cdot\mathrm{rot}\,\gamma_i\Big)_i,\qquad
M_H K = \Big(\int \tfrac12|u|^2\eta_i\Big)_i .
$$

The kinetic energy uses the same 3×3 rule.

## Horizontal tendencies (`horizontal_tendency!`, per layer)

With $\theta = \rho\theta/\rho$ and $\pi = \pi(\rho\theta)$ at the nodes:

$$
F = \Pi_U(\rho u),\qquad F_\theta = \Pi_U(\theta F),
$$

$$
\dot u = M_U^{-1}\Big(-\tfrac1a B^T K - \int \frac{\zeta + f}{\rho}\,(\hat r\times F)\cdot\phi_j\Big) - c_p\,\Pi_U\big(\theta\, M_U^{-1}B^T\pi/a\big),
$$

$$
\dot\rho = M_H^{-1}BF/a,\qquad \dot{(\rho\theta)} = M_H^{-1}BF_\theta/a .
$$

The PV term uses $\zeta$, $\rho$ and $f$ at the quadrature points.

**Energy:** with $E = \sum_k\Delta z_k\big[\int\rho|u|^2/2 + \sum_i M_{H,ii}\,(c_v\rho\theta\,\pi + \rho g z_k)_i\big]$,
we have $\partial E/\partial u = M_U F$, $\partial E/\partial\rho = M_H(K + g z)$ and
$\partial E/\partial(\rho\theta) = M_H c_p\pi$. The horizontal terms then give

$$
\sum_k\Delta z_k\big(F^TM_U\dot u + (K+gz)^TM_H\dot\rho + c_p\pi^TM_H\dot{(\rho\theta)}\big) = 0 .
$$

The PV term vanishes pointwise. The $K$ and $gz$ terms cancel through $B^T$/$B$ and
$\mathbf 1^TB = 0$. The pressure pair cancels through
$\pi^TBM_U^{-1}\Theta F - F^T\Theta M_U^{-1}B^T\pi = 0$, where $\Theta = (\int\theta\,\phi_i\cdot\phi_j)$
is symmetric. This is tested to 1e-14 of the summed products.

## Vertical terms (per column $i$, the kernels of `TRiSKNonhydrostatic`)

$$
\dot w_j = -c_p\,\theta_j\,\frac{\pi_j - \pi_{j-1}}{\Delta z^h_j} - g - r_j w_j,\qquad j = 2..n,
$$

$$
\dot\rho_k = -\frac{M_{k+1} - M_k}{\Delta z_k},\qquad
\dot{(\rho\theta)}_k = -\frac{\tilde\theta_{k+1}M_{k+1} - \tilde\theta_kM_k}{\Delta z_k},\qquad M_j = \rho_j w_j .
$$

- $\rho_j$ is the linear interpolation of $\rho$ of the stage state.
- $\theta_j$ in the pressure gradient is the linear interpolation of $\theta$ of the stage state.
- $\tilde\theta_j$ in the flux depends on `vertical_theta`:
  - `:upwind` (default): the third-order upwind interface value of $\theta$ after a Crank–Nicolson
    implicit vertical advection of $\theta$ of the base state over $\tau/2$ (ICON-IAP
    `ltheta_up_vert`; one pentadiagonal system per column);
  - `:centred`: $\theta_j$;
  - `:iap`: the upwind value also in the pressure gradient.

## Time stepping

Wicker–Skamarock RK3 from the base state $y^0$ with stages $\tau = \Delta t/3, \Delta t/2, \Delta t$.
Each stage computes $y = y^0 + \tau\,T(y^s)$ (`stage!`):

1. **Horizontal momentum, explicit:** $u = u^0 + \tau\,\dot u(y^s)$.
2. **Horizontal mass and $\rho\theta$ fluxes, forward–backward:** with the new $u$ and $\rho^s$,
   $\theta^s$, take $F = \Pi_U(\rho^s u)$ and $F_\theta = \Pi_U(\theta^sF)$. Then
   $\rho^{ex} = \rho^0 + \tau M_H^{-1}BF/a$ and $(\rho\theta)^{ex} = (\rho\theta)^0 + \tau M_H^{-1}BF_\theta/a$.
3. **Vertical, implicit per column:**
   - $M_j = \rho^s_j(\beta w_j + (1-\beta)w^0_j)$ with $\beta = 0.65$;
   - $\rho = \rho^{ex} - \tau\delta_z M$ and $\rho\theta = (\rho\theta)^{ex} - \tau\delta_z(\tilde\theta M)$;
   - $\pi$ linearized around the base state: $\pi \approx \pi^0 + \gamma^0(\rho\theta - (\rho\theta)^0)$
     with $\gamma^0 = \kappa\pi^0/(\rho\theta)^0$ and $\kappa = R/c_v$;
   - the $w$ equation:

$$
\textstyle\sum_m m_{jm}(w_m - w^0_m) + \tau r_j w_j = - \tau g - \tau c_p\theta_j\,\frac{(1-\beta)(\pi^0_j - \pi^0_{j-1}) + \beta(\pi_j - \pi_{j-1})}{\Delta z^h_j}.
$$

   Here $m_{jm}$ is the vertical mass row $j$ divided by $\Delta z^h_j$. Eliminating $\rho$ and
   $\rho\theta$ gives one tridiagonal system in $w$ per column.

The explicit horizontal time-step limit on hexagon 3 is between 250 and 300 s.

## Points to check

- The diagonal $M_H$ (Radau quadrature) is inexact on non-affine kites, while $B$ and the
  projections use exact Gauss quadrature. This differs between the cubed sphere (affine kites,
  stable) and the hexagon and triangle meshes (non-affine, unstable).
- The vertical $w$ mass: the consistent and the lumped $CG_1$ mass give the same growth.
- $\theta$ at the quadrature points is the $Q_2$ interpolation of the nodal $\theta$, used in
  $F_\theta$ and in the pressure gradient. Horizontally, $\theta$ is never upwinded.
- $\rho$ in the vertical flux and $\theta_j$ in the vertical pressure gradient are nodal values of
  the column, while the horizontal pressure gradient sees $\pi$ only through $B^T$. A sub-cell
  pressure pattern in a layer is therefore weighted differently horizontally and vertically.
