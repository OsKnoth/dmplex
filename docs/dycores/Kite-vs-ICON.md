# Kite and ICON: accuracy, effective resolution and cost

Dry baroclinic wave (Ullrich et al. 2014), model top 30 km, sponge above 22 km, on one A100 of
Levante. ICON (`ICONNonhydrostatic`) on the R2B grids with its namelist defaults; Kite
(`KiteNonhydrostatic`) with `horizontal_theta=:dg_upwind`, `horizontal_advection=:dg`. Each model is
diagnosed on its own points and quadrature weights: spherical-harmonic coefficients of p_s
(l ≤ 128) and the kinetic-energy spectrum from ζ and δ (l ≤ 256), averaged over the layers with
centres between 2.4 and 9.75 km.

Resolution is given as the mean spacing `Δx = sqrt(4π a²/N)` of the N mass points (columns) per layer:

| run | columns | Δx | dt |
|---|---|---|---|
| ICON R2B4 | 20480 | 158 km | 216 s |
| ICON R2B5 | 81920 | 79 km | 108 s |
| ICON R2B6 | 327680 | 39 km | 54 s |
| ICON R2B7 | 1310720 | 20 km | 27 s |
| Kite RT₁×Q₂ / Wess Q₂², hexagon 3 | 23682 | 147 km | 150 s |
| Kite RT₁×Q₂ / Wess Q₂², hexagon 4 | 94722 | 73 km | 75 s |

## Effective resolution (30 layers)

`l_eff` is the wavenumber where the smoothed ratio of the KE spectrum to the ICON R2B7 spectrum
drops below 0.5, only below the grid Nyquist `l < πa/Δx`.

| run | l_eff | in Δx |
|---|---|---|
| ICON R2B4 | ≈ 50 | ≈ 5 Δx |
| ICON R2B5 | 70–77 | ≈ 7 Δx |
| ICON R2B6 | 125–130 | ≈ 8 Δx |
| Kite RT₁×Q₂ hexagon 3 | 55–95 | |
| Kite Wess Q₂² hexagon 3 | 100–118 | ≈ 2.5 Δx |
| Kite RT₁×Q₂ hexagon 4 | 160–180 | ≈ 3.3 Δx |
| Kite Wess Q₂² hexagon 4 | above R2B7 at all l | – |

Kite resolves per column two to three times finer scales than ICON. Wess Q₂² on hexagon 4 lies above
the R2B7 spectrum with a large divergent fraction, which points to too little dissipation at the
small scales; RT₁×Q₂ on hexagon 3 has an uptick at l > 140.

## p_s against ICON R2B7 with 60 layers

Error of the p_s coefficients, l = 1–85, in % of the R2B7 L60 signal. `upwind` and `central` are
Kite's `vertical_theta`.

| run | day 8, L30 | L60 | L120 | day 10, L30 | L60 | L120 |
|---|---|---|---|---|---|---|
| ICON R2B6 | 32.2 | 5.7 | 10.3 | 47.8 | 13.0 | 13.1 |
| Kite RT₁×Q₂ hexagon 4, upwind | 39.6 | 21.7 | 12.7 | 50.6 | 31.8 | 20.0 |
| Kite RT₁×Q₂ hexagon 4, central | 28.9 | **5.7** | 9.7 | 42.5 | **12.9** | 13.4 |
| Kite Wess Q₂² hexagon 3, upwind | 44.3 | 26.5 | – | 52.6 | 32.5 | – |

Vertical convergence, difference between successive layer numbers at day 8:

| run | L30 → L60 | L60 → L120 |
|---|---|---|
| ICON R2B6 | 32.3 | 7.4 |
| Kite hexagon 4, upwind | 29.4 | 12.3 |
| Kite hexagon 4, central | 29.1 | 6.7 |

- With 30 layers (Δz = 1 km) the vertical error dominates in both models (30–40 %), so 30-layer
  runs do not compare horizontal accuracy.
- ICON converges vertically at about second order. Kite with the upwind θ in the vertical ρθ flux
  (DG0 upwind penalty, first order) converges more slowly; with the central θ it converges like ICON.
- With the central θ, Kite on hexagon 4 (73 km) is as close to R2B7 as ICON R2B6 (39 km) with a
  quarter of the columns: 5.7 % at day 8 with 60 layers for both, 8.3 % between the two with 120 layers.

Minimum p_s at day 10 (hPa):

| run | L30 | L60 | L120 |
|---|---|---|---|
| ICON R2B6 | 916.82 | 927.63 | 927.83 |
| ICON R2B7 | 914.86 | 926.81 | – |
| Kite hexagon 4, upwind | 923.09 | 921.52 | 924.43 |
| Kite hexagon 4, central | 917.94 | 925.09 | 926.16 |
| Kite Wess Q₂² hexagon 3, upwind | 921.97 | 920.07 | – |

## Cost

Milliseconds per time step on one A100:

| run | L30 | L60 | L120 |
|---|---|---|---|
| ICON R2B4 | 4.9 | – | – |
| ICON R2B5 | 7.3 | – | – |
| ICON R2B6 | 22.0 | 43.8 | 84.3 |
| ICON R2B7 | 83.8 | 141 | – |
| Kite RT₁×Q₂ hexagon 3 | 26.7 | – | – |
| Kite Wess Q₂² hexagon 3 | 55.3 | 92.1 | – |
| Kite RT₁×Q₂ hexagon 4 | 85–87 | 163–168 | 331–342 |
| Kite Wess Q₂² hexagon 4 | 183.6 | – | – |

At equal column count Kite costs 5–8 times ICON per simulated day. At equal accuracy (Kite hexagon
4 central against ICON R2B6, both 60 layers, with dt 75 s and 54 s) Kite takes about 2.7 times
ICON's time per simulated day.
