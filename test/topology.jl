@testset "Mesh topology: $name" for (name, builder, resolutions) in MESHES
    for resolution in resolutions
        dm = builder(resolution)
        @test check_plex(dm)
        @test get_chart(dm) == 1:(num_cells(dm) + num_edges(dm) + num_vertices(dm))
        @test get_height_stratum(dm, 0) == cell_range(dm) == get_depth_stratum(dm, 2)
        @test all(e -> all(!=(0), get_edge_cells(dm, e)), edge_range(dm))
        # discrete de Rham sequence: the boundary of a boundary vanishes
        @test iszero(boundary_matrix(dm, 1) * boundary_matrix(dm, 2))
    end
end

@testset "Closure, adjacency and labels" begin
    dm = create_spherical_mesh(0)
    cell = first(cell_range(dm))
    points, orientations = get_closure(dm, cell)
    @test length(points) == 7
    @test points[1] == cell
    @test points[2:4] == get_cone(dm, cell)
    @test points[5:7] == get_cell_vertices(dm, cell)
    @test orientations[2:4] == get_cone_orientation(dm, cell)
    @test get_closure(dm, first(edge_range(dm)))[1][2:3] == get_cone(dm, first(edge_range(dm)))

    @test length(get_cell_neighbors(dm, cell)) == 3
    @test get_adjacency(dm, cell; use_closure=false) == sort([cell; get_cell_neighbors(dm, cell)])
    # FEM adjacency: all cells sharing a vertex (valence 5: 3*5 - 3*2 + 1 = 10 cells)
    @test count(in(cell_range(dm)), get_adjacency(dm, cell)) == 10
    v = first(vertex_range(dm))
    @test count(in(cell_range(dm)), get_star(dm, v)) == 5

    set_label!(dm, "region", cell, 4)
    set_label!(dm, "region", cell + 1, 4)
    set_label!(dm, "boundary", first(get_cone(dm, cell)), 1)
    @test get_label_value(dm, "region", cell) == 4
    @test get_label_value(dm, "region", last(cell_range(dm))) === nothing
    @test get_label_stratum(dm, "region", 4) == [cell, cell + 1]
    @test get_label_names(dm) == ["boundary", "region"]
    clear_label!(dm, "boundary")
    @test get_label_names(dm) == ["region"]
end
