using Test
using dmplex

function check_spherical_mesh(builder, resolution)
    dm, vertices = builder(resolution)
    metrics = compute_mesh_metrics(dm, vertices)
    tracer = init_tracer(dm, vertices)
    velocity = init_edge_velocity(dm, metrics, vertices)
    rhs = zeros(length(tracer))
    compute_advection_rhs!(rhs, tracer, dm, metrics, velocity)

    @test length(vertices) == dm.vertex_end - dm.vertex_start + 1
    @test all(cell -> !isempty(get_cell_vertices(dm, cell)), dm.cell_start:dm.cell_end)
    @test all(edge -> begin
        left, right = get_support_cells(dm, edge)
        left != 0 && right != 0 && left != right
    end, dm.edge_start:dm.edge_end)
    @test all(isfinite, values(metrics.cell_volumes))
    @test all(>(0.0), values(metrics.cell_volumes))
    @test sum(values(metrics.cell_volumes)) ≈ 4π atol=1e-10

    mass_rhs = sum(
        rhs[cell - dm.cell_start + 1] * metrics.cell_volumes[cell]
        for cell in dm.cell_start:dm.cell_end
    )
    @test mass_rhs ≈ 0.0 atol=1e-12
end

@testset "Spherical mesh topology and upwind advection" begin
    for resolution in 0:2
        check_spherical_mesh(create_spherical_mesh, resolution)
        check_spherical_mesh(create_hexagon_mesh, resolution)
    end
    for resolution in 1:3
        check_spherical_mesh(create_cubed_sphere_mesh, resolution)
    end
end

@testset "DMPlex labels, adjacency, and point fields" begin
    dm, _ = create_spherical_mesh(0)
    cell = dm.cell_start
    closure = get_transitive_closure(dm, cell)
    @test first(closure) == cell
    @test length(closure) == 7
    @test all(point -> point in closure, get_cone(dm, cell) .|> abs)
    @test length(get_cell_neighbors(dm, cell)) == 3
    @test all(neighbor -> neighbor in get_adjacency(dm, cell), get_cell_neighbors(dm, cell))

    set_label!(dm, "region", cell, 4)
    set_label!(dm, "region", cell + 1, 4)
    set_label!(dm, "boundary", first(get_cone(dm, cell)) |> abs, 1)
    @test get_label_value(dm, "region", cell) == 4
    @test get_label_value(dm, "region", dm.cell_end) === nothing
    @test get_label_stratum(dm, "region", 4) == [cell, cell + 1]
    @test get_label_names(dm) == ["boundary", "region"]
    clear_label!(dm, "boundary")
    @test get_label_names(dm) == ["region"]

    set_field!(dm, "solution", cell, [1.0, 2.0])
    set_field!(dm, "solution", cell + 1, [3.0, 4.0])
    values = get_field(dm, "solution", cell)
    values[1] = -1.0
    @test get_field(dm, "solution", cell) == [1.0, 2.0]
    @test get_field_names(dm) == ["solution"]
    @test_throws ArgumentError set_field!(dm, "solution", cell, [1, 2])
    remove_field!(dm, "solution")
    @test isempty(get_field_names(dm))
    @test_throws KeyError get_field(dm, "missing", cell)
end
