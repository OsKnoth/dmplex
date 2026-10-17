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
