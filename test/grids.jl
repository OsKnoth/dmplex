using NCDatasets

xyz_to_lonlat(x) = (atan(x[2], x[1]), asin(clamp(x[3], -1, 1)))

"ICON-like grid file written from an icosahedral mesh (same variable layout as ICON)."
function write_icon_file(filename, dm)
    centers = compute_mesh_metrics(dm).cell_centers
    vertex_of_cell = permutedims(reduce(hcat, [vertex_index.(Ref(dm), get_cell_vertices(dm, c)) for c in cell_range(dm)]))
    cells_of_vertex = zeros(Int32, num_vertices(dm), 6)
    for v in vertex_range(dm)
        ring = [cell_index(dm, c) for c in get_star(dm, v) if c in cell_range(dm)]
        cells_of_vertex[vertex_index(dm, v), 1:length(ring)] .= ring
    end
    NCDataset(filename, "c") do ds
        defVar(ds, "vlon", first.(xyz_to_lonlat.(dm.coordinates)), ("vertex",))
        defVar(ds, "vlat", last.(xyz_to_lonlat.(dm.coordinates)), ("vertex",))
        defVar(ds, "clon", first.(xyz_to_lonlat.(centers)), ("cell",))
        defVar(ds, "clat", last.(xyz_to_lonlat.(centers)), ("cell",))
        defVar(ds, "vertex_of_cell", Int32.(vertex_of_cell), ("cell", "nv"))
        defVar(ds, "cells_of_vertex", cells_of_vertex, ("vertex", "ne"))
    end
end

"MPAS-like grid file written from a hexagonal mesh, padded like MPAS-Ocean files."
function write_mpas_file(filename, dm; radius=6.371229e6)
    nv = num_vertices(dm)
    centers = compute_mesh_metrics(dm).cell_centers
    vertices_on_cell = fill(Int32(nv + 1), 6, num_cells(dm))
    n_edges_on_cell = zeros(Int32, num_cells(dm))
    for c in cell_range(dm)
        ring = [vertex_index(dm, v) for v in get_cell_vertices(dm, c)]
        vertices_on_cell[1:length(ring), cell_index(dm, c)] .= ring
        n_edges_on_cell[cell_index(dm, c)] = length(ring)
    end
    cells_on_vertex = zeros(Int32, 3, nv)
    for v in vertex_range(dm)
        cells_on_vertex[:, vertex_index(dm, v)] .= [cell_index(dm, c) for c in get_star(dm, v) if c in cell_range(dm)]
    end
    NCDataset(filename, "c") do ds
        ds.attrib["on_a_sphere"] = "YES"
        for (name, points, dim) in (("Vertex", dm.coordinates, "nVertices"), ("Cell", centers, "nCells"))
            for (k, axis) in enumerate(("x", "y", "z"))
                defVar(ds, axis * name, radius .* getindex.(points, k), (dim,))
            end
        end
        defVar(ds, "verticesOnCell", vertices_on_cell, ("maxEdges", "nCells"))
        defVar(ds, "nEdgesOnCell", n_edges_on_cell, ("nCells",))
        defVar(ds, "cellsOnVertex", cells_on_vertex, ("vertexDegree", "nVertices"))
    end
end

total_area(dm) = sum(compute_mesh_metrics(dm).cell_areas)

@testset "ICON and MPAS readers" begin
    mktempdir() do dir
        icosahedral = create_spherical_mesh(2)
        icon = joinpath(dir, "icon.nc")
        write_icon_file(icon, icosahedral)
        dm = read_grid(icon)
        @test check_plex(dm)
        @test (num_cells(dm), num_edges(dm), num_vertices(dm)) ==
              (num_cells(icosahedral), num_edges(icosahedral), num_vertices(icosahedral))
        @test dm.coordinates ≈ icosahedral.coordinates
        dual = read_icon_grid(icon; dual=true)
        @test check_plex(dual)
        @test num_cells(dual) == num_vertices(icosahedral)
        @test count(c -> get_cone_size(dual, c) == 5, cell_range(dual)) == 12
        @test total_area(dual) ≈ 4π

        hexagonal = create_hexagon_mesh(2)
        mpas = joinpath(dir, "mpas.nc")
        write_mpas_file(mpas, hexagonal)
        dm = read_grid(mpas)
        @test check_plex(dm)
        @test num_cells(dm) == num_cells(hexagonal)
        @test all(x -> norm(x) ≈ 1, dm.coordinates)
        delaunay = read_mpas_grid(mpas; dual=true)
        @test check_plex(delaunay)
        @test num_cells(delaunay) == num_vertices(hexagonal)
        @test all(c -> get_cone_size(delaunay, c) == 3, cell_range(delaunay))
        @test total_area(delaunay) ≈ 4π

        other = joinpath(dir, "other.nc")
        NCDataset(ds -> defVar(ds, "temperature", zeros(3), ("x",)), other, "c")
        @test_throws ArgumentError read_grid(other)
    end
end

@testset "HEALPix" begin
    for nside in (1, 2, 4)
        dm = create_healpix_mesh(nside)
        @test check_plex(dm)
        @test num_cells(dm) == 12nside^2
        @test num_vertices(dm) == 12nside^2 + 2
        areas = compute_mesh_metrics(dm).cell_areas
        @test sum(areas) ≈ 4π
        @test maximum(areas) / minimum(areas) < 1.6
    end
    # quadrilateral FE spaces work unchanged on HEALPix meshes
    dm = create_healpix_mesh(2)
    @test isposdef(Symmetric(Matrix(assemble_mass(FESpace(dm, RaviartThomas(1))))))
end

const GRID_DIR = joinpath(@__DIR__, "..", "grid")
@testset "Grid files in grid/" begin
    files = ["icon_grid_0010_R02B04_G.nc", "x1.2562.grid.nc", "QU240.nc"]
    available = filter(f -> isfile(joinpath(GRID_DIR, f)), files)
    isempty(available) && @info "no grid files found in $GRID_DIR, skipping"
    for f in available, dual in (false, true)
        dm = read_grid(joinpath(GRID_DIR, f); dual)
        closed = all(e -> get_support_size(dm, e) == 2, edge_range(dm))
        @test check_plex(dm; closed)
        closed && @test total_area(dm) ≈ 4π
    end
end

@testset "Kite grids" begin
    for primal in (create_spherical_mesh(1), create_cubed_sphere_mesh(3), create_hexagon_mesh(1), create_healpix_mesh(2))
        dm = create_kite_mesh(primal)
        @test check_plex(dm)
        @test all(c -> get_cone_size(dm, c) == 4, cell_range(dm))
        @test num_cells(dm) == sum(get_cone_size(primal, c) for c in cell_range(primal))
        @test num_vertices(dm) == num_vertices(primal) + num_edges(primal) + num_cells(primal)
        @test total_area(dm) ≈ 4π
        # canonical local numbering: vertex 1 primal vertex, vertex 3 center,
        # edges 1 and 4 halves of primal edges, edges 2 and 3 midpoint–center
        vtype(v) = get_label_value(dm, "kite_vertex_type", v)
        etype(e) = get_label_value(dm, "kite_edge_type", e)
        @test all(cell_range(dm)) do c
            vertices, edges = get_cell_vertices(dm, c), get_cone(dm, c)
            vtype.(vertices) == [0, 1, 2, 1] && etype.(edges) == [1, 2, 2, 1]
        end
        # every primal edge is split into two half edges; every primal cell has n interior edges
        @test count(e -> etype(e) == 1, edge_range(dm)) == 2 * num_edges(primal)
        @test count(e -> etype(e) == 2, edge_range(dm)) == num_edges(dm) - 2 * num_edges(primal)
        # kites of one dual cell share their primal vertex; their number is the vertex valence
        for v in vertex_range(primal)
            kites = get_label_stratum(dm, "kite_dual_cell", vertex_index(primal, v))
            @test length(kites) == count(in(cell_range(primal)), get_star(primal, v))
        end
    end
end

@testset "Cell centers and orthogonal kites" begin
    icosahedral = create_spherical_mesh(2)
    hexagonal = create_hexagon_mesh(2)
    # triangles: circumcenters, equidistant from the three vertices
    centers = cell_centers(icosahedral)
    @test all(cell_range(icosahedral)) do c
        d = [dot(centers[cell_index(icosahedral, c)], get_coordinates(icosahedral, v)) for v in get_cell_vertices(icosahedral, c)]
        maximum(d) - minimum(d) < 1e-14
    end
    # Voronoi mesh: the generators are recovered (the vertices of the dual triangles)
    generators = cell_centers(hexagonal)
    @test orthogonality_defect(hexagonal, generators) < 1e-12
    @test maximum(norm.(generators .- icosahedral.coordinates)) < 1e-12
    @test cell_centers(hexagonal; method=:voronoi) ≈ generators
    # non-orthogonal meshes fall back to centroids
    cubed = create_cubed_sphere_mesh(4)
    @test cell_centers(cubed) == cell_centers(cubed; method=:centroid)
    @test_throws ArgumentError cell_centers(cubed; method=:circumcenter)

    # on the Voronoi mesh the kites of a dual cell form exactly the dual triangle:
    # midpoint–center kite edges lie on the great circle through two centers
    kites = create_kite_mesh(hexagonal)
    @test check_plex(kites)
    @test sum(compute_mesh_metrics(kites).cell_areas) ≈ 4π
    for e in edge_range(hexagonal)
        left, right = get_edge_cells(hexagonal, e)
        m = kites.coordinates[num_vertices(hexagonal) + edge_index(hexagonal, e)]
        gl, gr = generators[cell_index(hexagonal, left)], generators[cell_index(hexagonal, right)]
        @test abs(dot(normalize(cross(gl, gr)), m)) < 1e-12
    end
    kite_areas = compute_mesh_metrics(kites).cell_areas
    dual_areas = zeros(num_vertices(hexagonal))
    for (k, c) in enumerate(cell_range(kites))
        dual_areas[get_label_value(kites, "kite_dual_cell", c)] += kite_areas[k]
    end
    @test dual_areas ≈ compute_mesh_metrics(icosahedral).cell_areas
end
