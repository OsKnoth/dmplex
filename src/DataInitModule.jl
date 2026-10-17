module DataInitModule

using LinearAlgebra
using StaticArrays
using ..DMPlexModule
using ..MetricModule

export init_edge_velocity, init_tracer

"Solid-body rotation about a slightly tilted axis."
function analytical_wind_field(pos::SVector{3, Float64})::SVector{3, Float64}
    omega = SVector(0.1, 0.0, 1.0)
    return cross(omega, pos)
end

"""
    init_edge_velocity(dm, metrics)

Normal velocity of the analytical wind at every edge midpoint, positive from
the left to the right cell; indexed with `edge_index`.
"""
init_edge_velocity(dm::Plex, metrics::MeshMetrics) =
    [dot(analytical_wind_field(metrics.edge_midpoints[i]), metrics.edge_normals[i])
     for i in 1:num_edges(dm)]

"""
    init_tracer(dm, metrics)

Cell-wise constant cosine bell of radius 0.5 centered at `(1, 0, 0)`; indexed
with `cell_index`.
"""
function init_tracer(dm::Plex, metrics::MeshMetrics)
    return map(metrics.cell_centers) do center
        dist = acos(clamp(dot(center, SVector(1.0, 0.0, 0.0)), -1.0, 1.0))
        dist < 0.5 ? 0.5 * (1.0 + cos(π * dist / 0.5)) : 0.0
    end
end

end # module
