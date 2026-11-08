# Discretization of `KiteNonhydrostatic` (as implemented, 2026-11-04)

This is the state of `src/Nonhydrostatic/Kite/KiteNonhydrostatic.jl` after steps 1–3 of
[KITE3D_PLAN.md](KITE3D_PLAN.md): serial, lowest vertical order, with the vertical coupling of the vector-invariant form. The jet
instability and the upwind θ trace that removes it are described at the end of the plan.

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

Here $\zeta = \hat r\cdot\nabla\times u_h$. With `vertical_advection=:vector_invariant` (default)
the momentum advection is the complete vector-invariant form $\omega\times u + \nabla K$ of the shallow
atmosphere: $K = |u_h|^2/2 + w^2/2$, the horizontal vorticity $\eta = \partial_z u_h - \nabla_h w$, the term
$-\eta w$ in the $u_h$ equation and $\eta\cdot u_h - \partial_z K$ in the $w$ equation (exactly adjoint, see
below). The equations above show only the terms without it. Rayleigh damping of $w$ acts above
`damping_height`.

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
- $M_U$: **block diagonal** with one block per dual cell (the kites around a primal vertex),
  inverted exactly. It uses the same 3×3 Gauss rule as the projections (`mass_degree`), so that
  $\Pi_U(c\,v) = c\,v$ holds exactly for constants also on non-affine kites. With a different rule
  the $\rho\theta$ flux of a layer at rest creates $\theta'$, which caused the sub-cell instability.
- $M_Z$: GLL lumped, diagonal.
- Vertical, $\rho$ and $\rho\theta$ ($DG_0$): $\mathrm{diag}(\Delta z_k)$, exact.
- Vertical, $w$ ($CG_1$): **lumped** by the trapezoidal rule of every layer (the Gauss–Lobatto rule
  of $CG_1$), $\mathrm{diag}(\Delta z^h_j)$ with $\Delta z^h_j = (\Delta z_{j-1} + \Delta z_j)/2$. All
  vertical inner products use this rule, so the vertical mass flux and the kinetic energy of $w$ use
  the same weights (see the vertical terms).

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

**Vertical coupling of the vector-invariant form** (`vertical_advection=:vector_invariant`):

$$
\eta_j = \frac{u_j - u_{j-1}}{\Delta z^h_j} - M_U^{-1}B^Tw_j/a,\qquad
(\eta w)_k = \frac{\Delta z^h_k\,\Pi_U(w_k\eta_k) + \Delta z^h_{k+1}\,\Pi_U(w_{k+1}\eta_{k+1})}{2\Delta z_k},
$$

$$
\dot u_k \mathrel{-}= (\eta w)_k,\qquad
t^w_j = M_H^{-1}\Big(\int \eta_j\cdot\tfrac{u_{j-1}+u_j}{2}\,\psi_i\Big) - \frac{K_j - K_{j-1}}{\Delta z^h_j},
$$

with $K_k$ including $(\Delta z^h_k w_k^2 + \Delta z^h_{k+1}w_{k+1}^2)/4\Delta z_k$. The two η terms are
exactly adjoint, $\sum_k\Delta z_k u_k^TM_U(\eta w)_k = \sum_j\Delta z^h_j w_j^TM_H(\eta\cdot\tilde u)_j$
(tested to 1e-10), so the exchange of horizontal and vertical kinetic energy is exact. $t^w$ is the
explicit $w$ tendency of the column solver.

**Upwind θ trace** (`horizontal_theta=:upwind`). $\theta$ is broken across the primal edges, and
$B\,\Pi_U(\theta F)$ pairs the traces of the test function from both sides with the single-valued
normal component of $\Pi_U(\theta F)$: a central flux, in which the jump $[\theta]$ never enters.
The upwind option adds the DG upwind correction on every primal face $e$ (half of a primal edge)
between the kites $K$ and $E$:

$$
M_H\dot{(\rho\theta)} \mathrel{+}= -\frac1a\sum_e\int_e \eta_i^K\,\frac{|F\cdot n|}{2}\,(\theta^K - \theta^E)
\;-\;\frac1a\sum_e\int_e \eta_i^E\,\frac{|F\cdot n|}{2}\,(\theta^E - \theta^K),
$$

with the 1D Gauss rule of the face, $F\cdot n\,ds = \hat F\cdot\hat\nu\,dt$ (contravariant
Piola, metric free) and the two sides paired by the physical position of their face points
(`upwind_faces`). It vanishes for $\theta$ continuous across the face, conserves $\rho\theta$
(the two sides cancel), and dissipates the θ variance, $\theta^T(\cdot) = -\frac1a\sum_e\int_e
\frac{|F\cdot n|}{2}[\theta]^2 \le 0$. Together with $B\,\Pi_U(\theta F)$ it is the upwind
flux $\theta^{up}F\cdot n$ up to the projection. The energy exchange of the pressure pair is no
longer exact: the correction removes energy at the rate of the jumps. The momentum equation and
the mass flux are unchanged.

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

## Vertical terms (per column $i$, `KiteColumn.jl`)

Galerkin terms of $\rho, \rho\theta \in DG_0$ and $w \in CG_1$, tested with the layer indicators
$\chi_k$ and the hat functions $\phi_j$, every vertical integral by the trapezoidal rule of each
layer. The horizontal mass $M_{H,ii}$ is a common factor of all of them and cancels. With this rule the
projection of a $DG_0$ field $\rho$ times $w$ into $CG_1$ is $\bar\rho_j w_j$, with the
thickness-weighted interface mean
$$
\bar\rho_j = \frac{\Delta z_{j-1}\rho_{j-1} + \Delta z_j\rho_j}{\Delta z_{j-1} + \Delta z_j}
$$
(`interface_weights`; not the linear interpolation to the interface, which weights the layers the
other way round on stretched grids). The terms:

$$
F_j = \bar\rho_j w_j,\qquad G_j = \bar\theta_j F_j,\qquad
\Delta z_k\dot\rho_k = -(F_{k+1} - F_k),\qquad \Delta z_k\dot{(\rho\theta)}_k = -(G_{k+1} - G_k),
$$

$$
\Delta z^h_j\,\dot w_j = -c_p\,\bar\theta_j\,(\pi_j - \pi_{j-1}) - g\,\Delta z^h_j - \Delta z^h_j r_j w_j,
\qquad j = 2..n,\quad w_1 = w_{n+1} = 0 .
$$

$F$ is $\Pi_{CG_1}(\rho w)$ and $G$ is $\Pi_{CG_1}(\theta F)$, as $F = \Pi_U(\rho u)$ and
$\Pi_U(\theta F)$ horizontally; the pressure gradient is $\bar\theta$ times the adjoint of the vertical
divergence. With the kinetic energy $\sum_j \Delta z^h_j\bar\rho_j w_j^2/2$ of the same rule the exchange
with the internal and potential energy is exact (summation by parts, $\Delta z^h_j = z_j - z_{j-1}$):
$$
\sum_j \Delta z^h_j F_j\dot w_j + \sum_k \Delta z_k\bigl(c_p\pi_k\dot{(\rho\theta)}_k + g z_k\dot\rho_k\bigr) = 0
$$
(`vertical_tendency!`, tested). `vertical_theta = :upwind` (default) adds the upwind penalty of the
stage state, $P_j = |F_j|(\theta_{j-1} - \theta_j)/2$, to $G_j$: the upwind flux of $DG_0$, which
damps the variance of $\theta$; `:central` leaves it out. The pressure gradient keeps $\bar\theta_j$ in
both cases, so the penalty is the only non-conservative part. The passive tracer has the same flux
and penalty, so $q = 1$ gives the equation of $\rho$.

## Time stepping

Wicker–Skamarock RK3 from the base state $y^0$ with stages $\tau = \Delta t/3, \Delta t/2, \Delta t$.
Each stage computes $y = y^0 + \tau\,T(y^s)$ (`stage!`):

1. **Horizontal momentum, explicit:** $u = u^0 + \tau\,\dot u(y^s)$.
2. **Horizontal mass and $\rho\theta$ fluxes, forward–backward:** with the new $u$ and $\rho^s$,
   $\theta^s$, take $F = \Pi_U(\rho^s u)$ and $F_\theta = \Pi_U(\theta^sF)$. Then
   $\rho^{ex} = \rho^0 + \tau M_H^{-1}BF/a$ and $(\rho\theta)^{ex} = (\rho\theta)^0 + \tau M_H^{-1}BF_\theta/a$.
3. **Vertical, implicit per column:**
   - $F_j = \bar\rho^s_j(\beta w_j + (1-\beta)w^0_j)$ with $\beta = 0.65$, $\bar\rho^s$ and $\bar\theta^s$ the interface means of the stage state;
   - $\rho = \rho^{ex} - \tau\delta_z F$ and $\rho\theta = (\rho\theta)^{ex} - \tau\delta_z(\bar\theta F + P)$;
   - $\pi$ linearized around the base state: $\pi \approx \pi^0 + \gamma^0(\rho\theta - (\rho\theta)^0)$
     with $\gamma^0 = \kappa\pi^0/(\rho\theta)^0$ and $\kappa = R/c_v$;
   - the $w$ equation:

$$
(1 + \tau r_j)\,w_j = w^0_j + \tau\,\dot w^{ex}_j - \tau g - \tau c_p\bar\theta_j\,\frac{(1-\beta)(\pi^0_j - \pi^0_{j-1}) + \beta(\pi_j - \pi_{j-1})}{\Delta z^h_j}.
$$

   Eliminating $\rho$ and $\rho\theta$ gives one tridiagonal system in $w$ per column
   (`kite_column_kernel!`).

The explicit horizontal time-step limit on hexagon 3 is between 250 and 300 s.

## Points to check

- Solved: the velocity mass and the projections must use the same rule (see $M_U$). $B$ and $R$
  have no metric (contravariant Piola) and are exact.
- $\theta$ at the quadrature points is the $Q_2$ interpolation of the nodal $\theta$, used in
  $F_\theta$ and in the pressure gradient. With `horizontal_theta=:central` it is never upwinded
  horizontally; `:upwind` adds the face term above.
- $\rho$ in the vertical flux and $\theta_j$ in the vertical pressure gradient are nodal values of
  the column, while the horizontal pressure gradient sees $\pi$ only through $B^T$. A sub-cell
  pressure pattern in a layer is therefore weighted differently horizontally and vertically.
