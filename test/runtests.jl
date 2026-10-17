using LinearAlgebra
using Test
using dmplex

const MESHES = [
    ("icosahedral", create_spherical_mesh, 0:2),
    ("hexagonal", create_hexagon_mesh, 0:2),
    ("cubed sphere", create_cubed_sphere_mesh, 1:3),
]

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

@testset "Metrics and upwind advection: $name" for (name, builder, resolutions) in MESHES
    for resolution in resolutions
        dm = builder(resolution)
        metrics = compute_mesh_metrics(dm)
        tracer = init_tracer(dm, metrics)
        velocity = init_edge_velocity(dm, metrics)
        rhs = compute_advection_rhs!(similar(tracer), tracer, dm, metrics, velocity)

        @test all(>(0.0), metrics.cell_areas)
        @test sum(metrics.cell_areas) ≈ 4π atol=1e-10
        @test all(n -> norm(n) ≈ 1, metrics.edge_normals)
        @test all(i -> abs(dot(metrics.edge_normals[i], metrics.edge_midpoints[i])) < 1e-12, eachindex(metrics.edge_normals))
        # the edge normal points from the left cell towards the right cell
        @test all(edge_range(dm)) do e
            left, right = get_edge_cells(dm, e)
            i = edge_index(dm, e)
            dot(metrics.edge_normals[i],
                metrics.cell_centers[cell_index(dm, right)] - metrics.cell_centers[cell_index(dm, left)]) > 0
        end
        @test sum(rhs .* metrics.cell_areas) ≈ 0.0 atol=1e-12
        @test all(c -> 0 <= c <= 1, tracer)
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

@testset "Section dof layout" begin
    tri = create_spherical_mesh(1)
    quad = create_cubed_sphere_mesh(2)
    for dm in (tri, quad)
        nv, ne, nc = num_vertices(dm), num_edges(dm), num_cells(dm)
        k = get_cone_size(dm, first(cell_range(dm)))
        @test storage_size(Section(dm, lagrange(1))) == nv
        @test storage_size(Section(dm, lagrange(2))) == nv + ne + (k == 4 ? nc : 0)
        @test storage_size(Section(dm, lagrange(3))) == nv + 2ne + (k == 4 ? 4nc : nc)
        @test storage_size(Section(dm, discontinuous(0))) == nc
        @test storage_size(Section(dm, discontinuous(1))) == (k == 4 ? 4nc : 3nc)
        @test storage_size(Section(dm, raviart_thomas(0))) == ne
        @test storage_size(Section(dm, raviart_thomas(1))) == 2ne + (k == 4 ? 4nc : 2nc)
        @test storage_size(Section(dm, lagrange(1; ncomp=3))) == 3nv
    end

    dm = tri
    s = Section(dm, raviart_thomas(0; name="u"), discontinuous(0; name="h"))
    @test field_names(s) == ["u", "h"]
    @test storage_size(s) == num_edges(dm) + num_cells(dm)
    # point ordering: every dof index is used exactly once
    used = sort!(reduce(vcat, [collect(dof_range(s, p)) for p in get_chart(dm)]))
    @test used == 1:storage_size(s)
    @test get_field_dof(s, first(cell_range(dm)), field_index(s, "h")) == 1
    @test_throws ArgumentError field_range(s, 1)

    sf = Section(dm, raviart_thomas(0; name="u"), discontinuous(0; name="h"); ordering=:field)
    @test field_range(sf, 1) == 1:num_edges(dm)
    @test field_range(sf, 2) == num_edges(dm) + 1:storage_size(sf)
    @test all(c -> dof_range(sf, c, 2) ⊆ field_range(sf, 2), cell_range(dm))
    @test_throws ArgumentError get_offset(sf, 1)
    @test_throws KeyError field_index(sf, "missing")
end

@testset "Orientation of closure dofs" begin
    for dm in (create_spherical_mesh(1), create_cubed_sphere_mesh(2))
        k = get_cone_size(dm, first(cell_range(dm)))

        # RT0: shared edge dofs enter the two cells with opposite signs, so the
        # assembled divergence integrates to zero over the closed sphere
        s = Section(dm, raviart_thomas(0))
        divergence = zeros(num_cells(dm), storage_size(s))
        for c in cell_range(dm)
            idx, sgn = closure_dofs(dm, s, c, 1)
            @test length(idx) == k
            divergence[cell_index(dm, c), idx] .= sgn
        end
        @test all(iszero, sum(divergence; dims=1))
        @test divergence' == boundary_matrix(dm, 2)

        # P3 and RT1: the two cells sharing an edge see its nodes in opposite order
        for (layout, nodes) in ((lagrange(3), 2), (raviart_thomas(1), 2))
            s = Section(dm, layout)
            sign_flip = layout.space == :Hdiv ? -1 : 1
            for e in edge_range(dm)
                left, right = get_edge_cells(dm, e)
                local_nodes(c) = begin
                    idx, sgn = closure_dofs(dm, s, c, 1)
                    pos = findfirst(==(e), get_cone(dm, c))
                    interior = layout.space == :H1 ? (k == 4 ? 4 : 1) : (k == 4 ? 4 : 2)
                    range = interior + (pos - 1) * nodes .+ (1:nodes)
                    idx[range], sgn[range]
                end
                (li, ls), (ri, rs) = local_nodes(left), local_nodes(right)
                @test li == reverse(ri) == collect(dof_range(s, e))
                @test all(==(1), ls) && all(==(sign_flip), rs)
            end
        end
    end
end
