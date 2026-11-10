# Orography for the nonhydrostatic dycores

All three dycores have flat ground: the levels are one `VerticalGrid` for every column (`dz[k]`,
`dzh[j]`, `wlow[j]`). With orography the heights depend on the column.

## Common

- **Levels.** `terrain_following(vgrid, h)`: the interfaces `z = ζ + h b(ζ)` of every column over the
  reference levels `ζ` of `vgrid`.
  - `:sleve`: `b = sinh((F/s)^n - (ζ/s)^n)/sinh((F/s)^n)` below `flat_height` `F`, zero above
    (Schär et al. 2002, one scale `s` of 4 km, `n = 1.35` as ICON).
  - `:basic`: `b = 1 - ζ/z_top` (Gal-Chen and Somerville 1975).
  - Explicit heights, such as ICON's `z_ifc`, can also be passed.
- **Column metrics.** `column_metrics(z_half)` gives `z_full`, `dz`, `dzh` and `wlow` per column (cells × levels).
  `VerticalGrid` accepts a ground above zero, so a column is a `VerticalGrid` and `balanced_column` works unchanged.
- **Reference levels.** Rayleigh damping, damping and diffusion profiles stay on the reference
  levels `vgrid`, as ICON does with `vct_a`.
- **Test cases.**
  - `jw_surface_height`: the orography of Jablonowski and Williamson (2006) that balances their steady
    state. The steady state (`up = 0`) must stay steady (u error at day 9), and the wave runs on it.
  - Resting atmosphere over a steep mountain (DCMIP 2012 test 2-0): spurious `w`.
  - Schär mountain waves on the reduced planet (DCMIP 2012 test 2-1, X = 500).
  - DCMIP 2025 mesoscale mountain tests (Andrews et al. 2026, egusphere-2026-2293), on the small
    planet `small_planet(20)` (a/20, 20Ω or no rotation). There are two orographies:
    - `dcmip2025_gap_height`: a mountain chain with a gap.
    - `dcmip2025_vortex_height`: an isolated Gaussian mountain for vortex shedding.

    The initial state `dcmip2025_mountain_flow` is isothermal, 288 K, with u = 10 cos φ. The
    levels `dcmip2025_levels` have 57 layers up to 20 km. The paper's Rayleigh friction of u and v
    towards the initial wind above 10 km (τ = 100 s) is not in the models yet; they damp only w.
- **Planet and rotation.** `radius` and `omega` are keywords of every model. `coriolis(x)` gives
  the Coriolis parameter at the point `x` of the unit sphere (default `2Ω x₃`), so a run can have
  no rotation or use an f-plane.

## ICON (`ICONNonhydrostatic`): first, checked against Fortran ICON `jabw`

The equations along the terrain-following levels, as in Zängl et al. (2015).

- **Mass and ρθ fluxes.**
  - The horizontal fluxes are multiplied by the layer thickness `Δz_e` at the edges, and their
    divergence is divided by `Δz_i` of the cell.
  - The vertical flux uses the contravariant velocity `w - w_c` with the correction
    `w_c = vn ∂z/∂n + vt ∂z/∂t`. It is formed at the edges (full levels) from the slopes of the levels
    (`∂z/∂t` from the vertex heights), averaged to the cells with the bilinear weights and interpolated to the interfaces.
  - Ground: `w = w_c`, extrapolated quadratically; no mass flux.
- **Horizontal pressure gradient.** `c_p θ_e [∂π'/∂n - (∂z/∂n) ∂π'/∂z]` with `π' = π - π_ref(z)`. The
  reference part has no horizontal gradient on z surfaces, which keeps the truncation error small
  over slopes. ICON's `igradp_method = 3` (Zängl 2012) is an option for steep slopes later.
- **Velocity advection.**
  - Along the levels as now.
  - The vertical advection of `vn` and `w` uses `w - w_c`.
- **Column metrics.** `dz`, `dzh`, `wlow`, `∂w/∂z` coefficients, `wgtfacq` and the reference state at
  the real heights, per cell, plus `Δz_e` and `wlow_e` at the edges.
- **Not included.**
  - The metric correction of the divergence damping and of the θ diffusion over steep slopes
    (`l_zdiffu_t`).
  - The truncated pressure gradient.
- **Checks.**
  - Identical to the flat model with `h = 0` (all tests).
  - JW steady state over its orography.
  - JW wave against Fortran ICON `jabw` with ICON's `z_ifc`.

## MPAS-Gassmann

The same terms on the hexagonal C grid with the operators of ICON-IAP, plus the five-band column
solve with the contravariant mass flux.

## Kite

Mapped extruded elements: the height `z(x, ζ)` in the Jacobian of the prisms. Quadrature with the
metric in the mass matrices, the horizontal gradient along `ζ` corrected by `∂z/∂x ∂/∂z`, and the
contravariant vertical velocity in the column coupling.
