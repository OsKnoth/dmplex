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

@testset "Hilbert order" begin
    for dm in (create_hexagon_mesh(2), create_cubed_sphere_mesh(3))
        set_label!(dm, "mark", first(cell_range(dm)) + 5, 7)
        set_label!(dm, "mark", first(edge_range(dm)) + 3, 8)
        set_label!(dm, "mark", first(vertex_range(dm)) + 9, 9)
        new = hilbert_order(dm)
        @test check_plex(new)
        @test (num_cells(new), num_edges(new), num_vertices(new)) == (num_cells(dm), num_edges(dm), num_vertices(dm))
        perm = hilbert_permutation(dm)
        @test all(sum(cell_coordinates(new, first(cell_range(new)) + c - 1)) ≈
                  sum(cell_coordinates(dm, first(cell_range(dm)) + perm[c] - 1)) for c in eachindex(perm))
        # labels follow their points
        corners(d, p) = get_depth(d, p) == 0 ? [p] : get_depth(d, p) == 1 ? get_cone(d, p) : get_cell_vertices(d, p)
        point(d, p) = sum(get_coordinates.(Ref(d), corners(d, p)))
        for value in 7:9
            p, q = only(get_label_stratum(dm, "mark", value)), only(get_label_stratum(new, "mark", value))
            @test point(dm, p) ≈ point(new, q)
        end
    end
end
