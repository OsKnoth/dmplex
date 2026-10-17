"""
    analytical_cube_point(i, j, n, panel)

Equiangular gnomonic projection (Sadourny) of grid node `(i, j)` of a panel
with `n × n` cells onto the unit sphere.
"""
function analytical_cube_point(i::Int, j::Int, n::Int, panel::Int)::SVector{3, Float64}
    a = tan(i * π / (2 * n) - 0.25 * π)
    b = tan(j * π / (2 * n) - 0.25 * π)
    # 1=West, 2=East, 3=South, 4=North, 5=Bottom, 6=Top
    raw = panel == 1 ? SVector(-1.0, a, b) :
          panel == 2 ? SVector( 1.0, a, b) :
          panel == 3 ? SVector(a, -1.0, b) :
          panel == 4 ? SVector(a,  1.0, b) :
          panel == 5 ? SVector(a, b, -1.0) :
                       SVector(a, b,  1.0)
    return normalize(raw)
end

"""
    create_cubed_sphere_mesh(n)

Equiangular cubed-sphere quadrilateral mesh with `n × n` cells per panel.
Nodes on panel edges and corners are shared between panels.
"""
function create_cubed_sphere_mesh(n::Int)
    n > 0 || throw(ArgumentError("cubed-sphere resolution must be positive"))

    vertices = SVector{3, Float64}[]
    vertex_ids = Dict{NTuple{3, Float64}, Int}()
    panel_grids = [Matrix{Int}(undef, n + 1, n + 1) for _ in 1:6]
    for panel in 1:6, j in 0:n, i in 0:n
        point = analytical_cube_point(i, j, n, panel)
        key = Tuple(round.(point, digits=12) .+ 0.0)  # + 0.0 maps -0.0 to 0.0 for hashing
        panel_grids[panel][i + 1, j + 1] = get!(vertex_ids, key) do
            push!(vertices, point)
            length(vertices)
        end
    end

    quads = Vector{Int}[]
    for panel in 1:6, j in 1:n, i in 1:n
        grid = panel_grids[panel]
        push!(quads, [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
    end

    return plex_from_polygons(quads, vertices)
end
