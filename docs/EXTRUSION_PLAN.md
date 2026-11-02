# Extruded meshes and a nonhydrostatic C-grid model

Plan for the third dimension in dmplex (October 2026). First target: the dry baroclinic wave of
Ullrich et al. (2014, DCMIP 2016) without orography, with the horizontal C-grid operators that
already exist (TRiSK on Voronoi meshes as MPAS and the hexagonal ICON-IAP, the ICON triangle
operators) and staggered finite differences in the vertical. The finite elements follow later on
the same extruded topology.

## References

- PETSc: `DMPlexExtrude` / `DMPlexTransformExtrude` (`src/dm/impls/plex/plexextrude.c`,
  `transform/impls/extrude`). Extrusion is a `DMPlexTransform`: every point `p` of the surface
  mesh produces the points `p × {level}` and `p × {layer}`. With `tensor = true` the new cells are
  tensor cells (`DM_POLYTOPE_TRI_PRISM_TENSOR`, `DM_POLYTOPE_QUAD_PRISM_TENSOR`,
  `DM_POLYTOPE_SEG_PRISM_TENSOR` for the lateral faces): the cone is (bottom, top, lateral faces),
  which keeps the product structure visible to the numbering.
- Lange, Mitchell, Knepley, Gorman (2016), *Efficient mesh management in Firedrake using
  PETSc-DMPlex*, SISC 38, arXiv:1506.07749: the base mesh is a `DMPlex`, the `PetscSection` gives
  the dof layout per point, the extruded mesh is not stored as a 3D plex.
- Bercea, McRae, Ham, Mitchell, Rathgeber, Nardi, Luporini, Kelly (2016), *A structure-exploiting
  numbering algorithm for finite elements on extruded meshes, and its performance evaluation in
  Firedrake*, GMD 9, arXiv:1604.05937: the dofs of a column are numbered contiguously, one
  indirection per column, the vertical neighbours by offsets.
- McRae, Bercea, Mitchell, Ham, Cotter (2016), *Automated generation and symbolic manipulation of
  tensor product finite elements*, SISC, arXiv:1411.2940: the compatible spaces on prisms as sums
  of products (for the later FEM part).
- Gassmann (2013), *A global hexagonal C-grid non-hydrostatic dynamical core (ICON-IAP) designed
  for energetic consistency*, QJRMS 139; Gassmann and Herzog (2008), the energy-consistent vertical
  implicit scheme; the ICON-IAP code on this machine, `~/EigeneProgramme/ICON-IAP-MOD`
  (`atm_dyn_iconam/mo_divergent_modes.f90`, `mo_org_momentum_adv.f90`, `mo_nh_stepping.f90`).
- Zängl, Reinert, Rípodas, Baldauf (2015), the ICON nonhydrostatic core (QJRMS 141), for the
  triangle variant; the ported ICON 2024.07 operators are already in `ICONShallowWater`.
- Ullrich, Melvin, Jablonowski, Staniforth (2014), the deep/shallow baroclinic wave in height
  coordinates; implemented in CGDycore.jl (`BaroWaveDryExample`), the reference solution for us.

## 1. Extruded topology (`src/Topology/Extruded.jl`)

`ExtrudedPlex(base::Plex, nlayers)` stores only the surface plex and the number of layers, as
Firedrake does; nothing of the 3D DAG is materialized.

- Points are pairs `(p, k)`: a base point `p` and either a layer `k ∈ 1:nlayers` (`:layer`, the
  open interval) or a level `k ∈ 1:nlayers+1` (`:level`). The product gives the strata
  - cell × layer → 3D cell (tensor prism), cell × level → horizontal face,
  - edge × layer → lateral face (quad), edge × level → horizontal edge,
  - vertex × layer → vertical edge, vertex × level → 3D vertex.
- Cones as in `DMPlexTransformExtrude` with tensor cells: `cone((c, k)) = ((c, k), (c, k+1),
  (e, k) for e in cone(c))` with the base orientations; the top face has the reversed orientation
  as in PETSc. `support` likewise. Functions, not stored arrays, so they work on the GPU too.
- Levels are numbered from the bottom (`k = 1` the ground), unlike ICON (top to bottom); the
  conversion is one line in the ICON-IAP port.
- A flat point number for compatibility with `Section` and the output:
  `point(p, k, :layer|:level)` and its inverse, with the column of `p` contiguous (Bercea et al.).
- `ExtrudedSection`: dofs per (base stratum, layer|level). The global index of a field is
  `offset(p) + k` (vertical fastest, Bercea) or `i + (k-1) n` (horizontal fastest). The model
  below uses the second layout: arrays `(n_local, nlev)` in Julia's column-major order, so one GPU
  thread per column accesses consecutive entries of neighbouring columns (coalesced) in the
  horizontal stencils and in the vertical solver. This is ICON's `(nproma, nlev)` layout too.
- Parallel: columns are never split. The 2D distribution (`TRiSKDistribution`, equal-area
  partition, ghost rings) is reused unchanged; the halo dof lists are repeated for every level
  (`d + (k-1) n`), so one message per neighbour carries all levels.
- Output: 3D cells as VTK wedges (13) for triangles, hexahedra (12) for quads, polyhedra or
  extruded polygons for the Voronoi cells (VTX writer with a level loop); horizontal slices and
  columns as cheaper alternatives.
- Tests: Euler characteristic of the 3D complex, cone sizes per cell type, the orientation
  identity (boundary of the boundary is zero) for every prism, the divergence theorem per prism
  with the geometric areas, halo exchange of `(n, nlev)` arrays bitwise on 1/3/4 ranks.

## 2. Vertical grid and metric (`src/Geometry/VerticalGrid.jl`)

- Height coordinate, flat ground for now: interface heights `z_half[k]`, `k = 1:nlev+1`, from
  `z = 0` to `z_top`, stretched as in CGDycore (`--Stretch`, the ICON `vct_a` formula) or uniform.
  Layer heights `z_full[k]` as the interface mean (ICON) or the mass-weighted centre.
- Shallow atmosphere first: horizontal areas and lengths of the unit sphere times `a²` resp. `a`,
  independent of `k`. Deep atmosphere later: `(a + z)²` and `a + z` per level (the TRiSK weights
  stay valid because the geometry is a scaled copy per level).
- Later, orography: terrain-following `z(x, k)` (SLEVE), the metric terms `∂z/∂x_n` at the edges
  (ICON `ddxn_z_full`) and the contravariant correction of the vertical flux, as in ICON-IAP.

## 3. Nonhydrostatic model on the extruded C grid (`src/Nonhydrostatic/`)

Lorenz staggering as in ICON and ICON-IAP:

| variable | location | array |
|---|---|---|
| `vn` normal velocity | lateral faces (edge × layer) | `(ne, nlev)` |
| `w` vertical velocity | horizontal faces (cell × level) | `(nc, nlev+1)` |
| `ρ`, `ρθ_v` | cells (cell × layer) | `(nc, nlev)` |
| `π = (R ρθ_v / p₀)^(R/c_v)` | cells, diagnostic | `(nc, nlev)` |

Equations (shallow, no orography; `Δz_k` layer thickness, `Δz_{k+1/2}` between layer centres):

    ∂vn/∂t = − [(ζ + f) r̂ × v]_n − [w ∂v/∂z]_n − ∂_n (K + Φ) − c_p θ_e ∂_n π
    ∂w/∂t  = − [vertical vorticity part] − ∂_z (K_h + Φ) − c_p θ_{k+1/2} ∂_z π
    ∂ρ/∂t  = − ∇_h ⋅ (ρ_e vn) − ∂_z (ρ_{k+1/2} w)
    ∂ρθ/∂t = − ∇_h ⋅ (ρ_e vn θ_e) − ∂_z (ρ_{k+1/2} w θ_{k+1/2})

- Horizontal per layer: the TRiSK operators of `TRiSKShallowWater` with the layer mass flux
  `ρ_e vn` (PV flux `q = (ζ + f)/ρ` per layer; energy-conserving pairing), divergence, gradient,
  cell kinetic energy `e_inn_c`. Kernels run over `(edge, level)` resp. `(cell, level)`; the
  neighbour tables are the 2D ones. Alternatively the ICON triangle operators
  (`ICONShallowWater`: RBF tangential velocity, `coeff_gradekin`, centered `∇K` on refined grids).
- Vertical: centred differences on the staggered grid; `θ` at the interfaces by the mass-weighted
  mean (later third-order upwind, as `advection_scal_edges` in ICON-IAP); the vertical advection of
  `vn` and the horizontal vorticity terms of ICON-IAP `slow_modes` (`ddt_vn_vort`, `ddt_w_vort`).
- Time stepping, HEVI: horizontally explicit, vertically implicit for `w`, `ρ`, `ρθ_v` (the
  vertical acoustic and gravity waves). Version 1: ICON-IAP's two-stage scheme
  (`slow_modes` + `divergent_modes`), with the off-centered Exner pressure and a tridiagonal system
  for the vertical mass flux `ρ w` (Gassmann and Herzog 2008); the 5-band system of ICON-IAP
  (implicit vertical advection of `w`) follows if needed. Version 2: the IMEX/Rosenbrock tables of
  CGDycore (SSP-KnothL) with the same vertical Jacobian, for comparison with CGSEM.
- Damping: Rayleigh damping of `w` in the upper layers (ICON-IAP `rw_updamp`, CGDycore
  `DampingW`), fourth-order divergence damping of `vn` (`ICONShallowWater` already has it),
  optionally the horizontal diffusion of ICON-IAP.
- GPU: one kernel per operator over `(n, nlev)`; the vertical solver one thread per column (Thomas
  algorithm in registers or local memory). MPI: one halo exchange of `[vn; w; ρ; ρθ]` per stage.

## 4. Baroclinic wave

- Initial state: Ullrich et al. (2014) shallow, `z_top = 30 km`, port of
  `BaroWaveDryExample` (CGDycore `src/Examples/initial.jl`): `ρ`, `θ` at the cell centres of every
  layer (the hydrostatic balance discretely corrected in every column: integrate the discrete
  `w` equation for `π` from the analytic ground pressure, so the rest state is balanced to
  round-off), `vn` at the lateral faces from the analytic wind plus the perturbation, `w = 0`.
- Grids: hexagon 5/6 (MPAS-like SCVT or spring), MPAS x1.40962 (120 km), ICON R02B04/R02B05 for
  the triangle variant; 30 levels, stretched.
- Diagnostics: minimum surface pressure over 15 days, `ps` and `T850` on day 8 and 10 against
  CGDycore (CGSEM, `nPanel` 32) and the DCMIP results; total mass to round-off, total energy drift.
  ADIOS2/VTX output of the lowest layer and of 3D wedges.

## Status (30 October 2026)

- `ExtrudedPlex` (`src/Topology/Extruded.jl`): points, cones with incidence signs, supports, strata,
  boundary matrices. Tested on icosahedral, hexagonal and cubed-sphere meshes: `∂∂ = 0`, Euler
  characteristic 2, point numbering invertible, supports inverse to cones (`test/extruded.jl`).
  Not yet: `ExtrudedSection`, the extruded halo exchange, output of 3D cells.
- `VerticalGrid` and `TRiSKNonhydrostatic` (`src/Nonhydrostatic/`): version 1 of section 3,
  serial, KernelAbstractions kernels on the CPU. TRiSK per layer with `ρ` as the thickness, centred
  vertical differences, `θ` centred horizontally and vertically, WS-RK3 with a tridiagonal HEVI
  solve per stage (`β = 0.65`), Rayleigh damping of `w` above 22 km. No diffusion, no divergence
  damping.
- Results (hexagon 5, 10242 cells ≈ 240 km, 30 levels of 1 km, `dt = 300 s`, 4 threads, 0.1 s per
  step): a horizontally homogeneous rest state stays at rest to round-off (`max|w|` 1.6e-14 after
  20 steps); the unperturbed baroclinic wave stays steady (`max|vn - vn₀|` 0.8 m/s after 3 days);
  with the perturbation the minimum surface pressure is 992.3 / 983.8 / 966.0 / 946.6 / 918.7 hPa
  on day 6 / 7 / 8 / 9 / 10, smooth fields; mass conserved to round-off.
- Hexagon 6 (40962 cells ≈ 120 km) without dissipation becomes unstable from day 4 in the jet core
  (11–13 km, ±46°, both hemispheres, so also the unperturbed jet; doubling time ≈ 6 h), with
  `dt = 300` and `150 s` alike. The mixed kinetic energy of MPAS (`kinetic_energy = :mixed`) makes
  it worse (day 4 instead of 5.5), so it is not the classical Hollingsworth case. With fourth-order
  hyperdiffusion of `vn` (`hyperdiffusion = 1e15` m⁴/s, the C-grid vector Laplacian
  `∇δ + k × ∇ζ` twice, once per step) it is stable: p_s min 992.7 / 982.6 / 963.2 / 938.5 /
  920.3 hPa on day 6–10, smooth fields (60 min for 10 days on 8 threads). The version-1 scheme is not
  energy consistent (advective vertical advection of `vn`, centred `θ`); the energy-consistent
  form of ICON-IAP is the next step.
- Vertical advection in the 3D vector-invariant form as in CGSEM (`vertical_advection =
  :vector_invariant`, `η = ∂vn/∂z - ∂w/∂n`, adjoint averages, the `vn`–`w` exchange cancels to
  8e-16). Hexagon 6 without dissipation: unstable from day 3.5 (p_s min 989.8 hPa on day 3), with
  the mixed kinetic energy from day 4.5; with `hyperdiffusion = 1e15` stable, p_s min 994.2 /
  987.3 / 979.4 / 962.6 / 942.1 / 910.8 hPa on day 5–10 (advective: 996.7 / 992.7 / 982.6 /
  963.2 / 938.5 / 920.3). The default stays `:advective`, until the kinetic energy and the
  averages follow ICON-IAP (Gassmann's treatment of the Hollingsworth instability).
- Correction (2 November 2026): the `η v̄n` term of the `w` equation had half its weight
  (`l_e d_e / 4A_i` instead of the cell inner product `l_e d_{e,i}/A_i`, ICON `e_inn_c`); the
  energy check used the edge area `l_e d_e/2`, but `½vn²` carries `l_e d_e`. The vector-invariant
  results above are those of the wrong scheme.
- ICON-IAP form (`pv_flux = :iap`, `vertical_advection = :iap`, now the defaults): PV usage of
  `i_cori_method = 4` (`trisk_pv_edges`, rhombus PV, symmetric, no work), the vorticity term with
  the mass fluxes `M_e/ρ_e` and `ρ_e vn` as in `vorticity_tendencies`, `K_w` with the `dzh`
  weights as in `kinetic_energy`; PV flux work and the `vn`–`w` exchange vanish to round-off
  (`test/extruded.jl`). Not (yet): the implicit vertical advection of `vn` (`impl_vert_adv_vn`),
  the vertical diffusion (the Smagorinsky diffusion and the upwind vertical `θ` follow below).
  Without dissipation, p_s min on day 6 / 7 / 8 / 9 / 10 (hPa):

  | mesh | `pv_flux` | vertical | p_s min | roughness p_s / T day 6 |
  |---|---|---|---|---|
  | hex 5 | `:iap` | `:iap` | 991.6 / 982.6 / 967.4 / 954.6 / 938.8 | |
  | hex 6 | `:iap` | `:iap` | 989.3 / 976.0 / 955.3 / 956.4 / 944.1 | 0.23 hPa / 0.50 K |
  | hex 6 | `:trisk` | `:vector_invariant` | 992.5 / 982.4 / 963.8 / 950.2 / 935.9 | 0.05 hPa / 0.16 K |
  | hex 6 | `:iap` | `:advective` | NaN on day 4 | |

  With the corrected weight the vector-invariant form is stable on hexagon 6 without
  dissipation. The ICON-IAP PV usage is noisier at the grid scale (as `:gassmann` in the
  shallow-water model); ICON-IAP itself runs with Smagorinsky diffusion.
- ICON-IAP standard (2 November 2026, defaults `trisk_weights = :iap`, `smagorinsky = 0.15`,
  `ν4 = 0`):
  - TRiSK weights of `compute_cori_coeff_idx` (`iap_trisk_weights`): planar kite fractions
    `R_{i,v} = Σ d_{e,i} l_{e,v} / 2A_i`; on hexagon 6 they differ from the exact weights by 1e-5
    and are antisymmetric to 1e-5 (second order). `trisk_weights = :exact` keeps the old ones.
  - Smagorinsky diffusion with the ICON-IAP coefficient `K_e = c · 8πa²/n_e · |D|`, `c = 0.15`,
    tapered to zero in the sponge as `kmh_bg`, and the deformation
    `|D|² = (n·∂_n u − t·∂_t u)² + (n·∂_t u + t·∂_n u)²` from Perot vectors at the cells and
    barycentric vectors at the vertices (zero to 2e-5 for linear fields, 0.005 Ω for solid-body
    rotation on hexagon 5). The operator is the δ–ζ form `(1/ρ_e)[∇(ρKδ) − k×∇(ρKζ)]·n`, not
    Gassmann's least-squares strain projections; it dissipates kinetic energy and returns it as
    heat `(ρK_c δ² + Σ_v K_iv/A_i ρK_v ζ_v²)/(c_p π)` (`ldiss_heat`), exactly in the discrete
    energy balance (`test/extruded.jl`). Evaluated once per step on the old state.

  | mesh | `pv_flux` | weights | p_s min day 6 / 7 / 8 / 9 / 10 | roughness p_s / T day 6 / 8 / 10 |
  |---|---|---|---|---|
  | hex 5 | `:iap` | `:iap` | 992.7 / 983.7 / 968.2 / 947.6 / 922.4 | |
  | hex 6 | `:iap` | `:iap` | 991.9 / 982.6 / 963.1 / 941.4 / 924.8 | 0.04/0.10, 0.14/0.31, 0.25/0.61 |
  | hex 6 | `:trisk` | `:exact` | 992.8 / 983.4 / 965.0 / 940.4 / 925.7 | 0.04/0.10, 0.16/0.47, 0.27/0.77 |

  Without dissipation (runs above) hexagon 6 with `:iap` has the roughness 0.23/0.50, 0.32/0.79,
  0.59/1.50, with `:trisk` 0.05/0.16, 0.25/0.68, 0.51/1.28. The Smagorinsky diffusion halves the
  grid-scale noise and removes the difference between the PV usages; hexagon 5 and 6 differ by
  at most 6 hPa (day 9) and by 2.4 hPa on day 10.
- Upwind vertical `θ` (2 November 2026, default `vertical_theta = :upwind`): ICON-IAP's
  `ltheta_up_vert` (`impl_vert_adv_scalar_5diag`). `θ` of the old state is advected vertically
  with the mass flux of the stage state over half the stage, Crank–Nicolson, one pentadiagonal
  system per column (`theta_interface_kernel!`); the interface values of the result are linear
  plus 1/6 of the curvature at the upwind cell (third order, `theta_stencil`). On stretched
  grids the weight of the upwind cell is set so that linear profiles are exact (ICON-IAP's `2`
  keeps constants only on uniform grids). ICON-IAP uses these values in the `ρθ` flux and in the
  vertical pressure gradient (`:iap`); there the stencil switches with the sign of round-off `w`
  and a column at rest gets `w` of 0.16 m/s after five steps. `:upwind` therefore uses them in
  the flux only and keeps the linear interpolation in `∂π/∂z` (rest to round-off, the exchange
  with `K_w` then not exact). p_s min day 6 / 7 / 8 / 9 / 10 (hPa), Smagorinsky 0.15:

  | mesh | `vertical_theta` | p_s min | roughness p_s / T day 6 / 8 / 10 |
  |---|---|---|---|
  | hex 5 | `:centred` | 992.7 / 983.7 / 968.2 / 947.6 / 922.4 | |
  | hex 5 | `:upwind` | 992.6 / 983.5 / 968.4 / 946.9 / 921.8 | |
  | hex 5 | `:iap` | 993.1 / 984.8 / 970.8 / 945.5 / 925.6 | |
  | hex 6 | `:centred` | 991.9 / 982.6 / 963.1 / 941.4 / 924.8 | 0.04/0.10, 0.14/0.31, 0.25/0.61 |
  | hex 6 | `:upwind` | 991.7 / 982.3 / 963.1 / 940.4 / 925.1 | 0.04/0.10, 0.14/0.28, 0.23/0.48 |

- Next: GPU runs with KernelAbstractions (`examples/baroclinic_wave.jl --gpu`,
  `examples/gpu/baroclinic_wave.sbatch`, one A100 on Levante), then the distribution over ranks.

## 5. Steps

1. `ExtrudedPlex`, `ExtrudedSection`, extruded halo exchange, tests (topology, MPI).
2. Vertical grid, metric, and the extruded `TRiSKGrid` view (areas per level, volume of the prisms).
3. Explicit nonhydrostatic model, RK4 with a small time step: hydrostatic rest atmosphere
   (stays at rest), a gravity wave on a small planet (Skamarock–Klemp on the reduced sphere, as in
   DCMIP 2012 test 3-1) to check the vertical operators against CGDycore.
4. HEVI: tridiagonal vertical solver (ICON-IAP two-stage), then the baroclinic wave on hexagon 5
   (TRiSK) and on ICON R02B04 (triangles); compare with CGDycore.
5. GPU (KernelAbstractions) and MPI runs on Levante.
6. Finite elements in 3D: tensor-product spaces on the extruded mesh (McRae et al. 2016:
   `RT_k ⊗ DG ⊕ DG_k ⊗ CG` for H(div), `DG_k ⊗ DG` for L2, the `θ` space on the `w` nodes as in
   Gusto), quadrature as tensor products, the `ExtrudedSection` with the Bercea numbering;
   the CGSEM column structure of CGDycore as a special case.
