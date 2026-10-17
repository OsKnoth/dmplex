# Zonal equal-area partition of the sphere (Leopardi 2006, "A partition of the
# unit sphere into regions of equal area and small diameter"), as used by the
# EqualRegionsPartitioner of Atlas (ECMWF) and by CGDycore: a north and a south
# polar cap and collars in between, each collar split into regions in longitude.

area_of_cap(s_cap) = 4π * sin(s_cap / 2)^2
sradius_of_cap(area) = 2asin(sqrt(area / π) / 2)

"""
    equal_area_zones(N) -> regions per zone

Number of regions in each zone (north cap, collars, south cap) of the zonal
equal-area partition of the sphere into `N` regions; the sum is `N`.
"""
function equal_area_zones(N::Integer)
    N >= 1 || throw(ArgumentError("need at least one region"))
    N == 1 && return [1]
    ideal_area = 4π / N
    c_polar = N == 2 ? π / 2 : sradius_of_cap(ideal_area)
    a_ideal = sqrt(ideal_area)
    n_collars = N > 2 ? max(1, round(Int, (π - 2c_polar) / a_ideal)) : 1
    a_fitting = (π - 2c_polar) / n_collars
    ideal = [1.0; [(area_of_cap(c_polar + k * a_fitting) - area_of_cap(c_polar + (k - 1) * a_fitting)) / ideal_area
                   for k in 1:n_collars]; 1.0]
    regions = zeros(Int, length(ideal))
    discrepancy = 0.0
    for (z, r) in enumerate(ideal)  # round to naturals, carrying the discrepancy
        regions[z] = round(Int, r + discrepancy)
        discrepancy += r - regions[z]
    end
    return filter(>(0), regions)
end

"Split `n` items into `parts` counts that differ by at most one."
balanced_counts(n, parts) = [n ÷ parts + (p <= n % parts ? 1 : 0) for p in 1:parts]

"""
    equal_area_partition(dm, nparts; centers=nothing) -> part per cell

Partition the cells of `dm` into `nparts` parts of (nearly) equal cell count
whose shapes follow the zonal equal-area regions: cells are sorted from north
to south and handed out zone by zone, each collar sorted from west to east
(Atlas `EqualRegionsPartitioner`, CGDycore `DecomposeEqualArea`). On a plane
the zones are bands in `y` split in `x`. Returns a vector of part numbers
`1:nparts`, indexed by `cell_index`.
"""
function equal_area_partition(dm::Plex, nparts::Integer; centers=nothing)
    nc = num_cells(dm)
    nparts <= nc || throw(ArgumentError("more parts than cells"))
    x = centers === nothing ? compute_mesh_metrics(dm).cell_centers : centers
    if dm.surface == :sphere
        north_south = [asin(clamp(p[3], -1, 1)) for p in x]   # latitude, sorted descending
        west_east = [mod(atan(p[2], p[1]), 2π) for p in x]  # longitude in [0, 2π)
        zones = equal_area_zones(nparts)
    else
        north_south, west_east = [p[2] for p in x], [p[1] for p in x]
        bands = clamp(round(Int, sqrt(nparts * (dm.period[2] > 0 ? dm.period[2] / dm.period[1] : 1))), 1, nparts)
        zones = balanced_counts(nparts, bands)
    end
    counts = balanced_counts(nc, nparts)
    order = sortperm(1:nc; lt=(a, b) -> north_south[a] > north_south[b] ||
                                        (north_south[a] == north_south[b] && west_east[a] < west_east[b]))
    part = zeros(Int, nc)
    region, start = 0, 0
    for nzone in zones
        zone_count = sum(counts[(region + 1):(region + nzone)])
        cells = order[(start + 1):(start + zone_count)]
        sort!(cells; lt=(a, b) -> west_east[a] < west_east[b] ||
                                  (west_east[a] == west_east[b] && north_south[a] > north_south[b]))
        offset = 0
        for r in 1:nzone
            part[cells[(offset + 1):(offset + counts[region + r])]] .= region + r
            offset += counts[region + r]
        end
        region += nzone
        start += zone_count
    end
    return part
end
