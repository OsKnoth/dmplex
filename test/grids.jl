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

@testset "Spring dynamics" begin
    plain, optimized = create_spherical_mesh(4), create_spherical_mesh(4; spring_beta=1.2)
    @test num_cells(optimized) == num_cells(plain) && optimized.coordinates[1:12] == plain.coordinates[1:12]
    @test all(x -> abs(norm(x) - 1) < 1e-14, optimized.coordinates)
    @test total_area(optimized) ≈ 4π
    areas(dm) = compute_mesh_metrics(dm).cell_areas
    @test maximum(areas(optimized)) / minimum(areas(optimized)) < 1.2 < maximum(areas(plain)) / minimum(areas(plain))
    # equilibrium: no tangential spring force on the free vertices
    x = copy(optimized.coordinates)
    edges = [Tuple(sort(vertex_index.(Ref(optimized), get_cone(optimized, e)))) for e in edge_range(optimized)]
    @test spring_dynamics!(x, edges; beta=1.2, mean_length=sqrt(4π / (5sqrt(3))) / 16, fixed=1:12) < 10
    @test maximum(norm.(x .- optimized.coordinates)) < 1e-9
    hexagons = create_hexagon_mesh(4; spring_beta=1.2)
    @test orthogonality_defect(hexagons, optimized.coordinates) < 1e-10
end

@testset "Centroidal Voronoi tessellation (Lloyd)" begin
    scvt = create_spherical_mesh(4; centroidal=true)
    @test scvt.coordinates[1:12] == create_spherical_mesh(0).coordinates
    @test all(x -> abs(norm(x) - 1) < 1e-14, scvt.coordinates) && total_area(scvt) ≈ 4π
    # a fixed point: the vertices are the centroids of their Voronoi cells
    x = copy(scvt.coordinates)
    triangles = [Tuple(vertex_index.(Ref(scvt), get_cell_vertices(scvt, c))) for c in cell_range(scvt)]
    @test lloyd!(x, triangles; fixed=1:12) < 3 && maximum(norm.(x .- scvt.coordinates)) < 1e-9
    # Anderson mixing converges to the same tessellation as plain Lloyd
    y, z = copy(create_spherical_mesh(4).coordinates), copy(create_spherical_mesh(4).coordinates)
    lloyd!(y, triangles; fixed=1:12)
    lloyd!(z, triangles; fixed=1:12, history=0)
    @test maximum(norm.(y .- z)) < 1e-7
    # TRiSK tangential reconstruction on the hexagonal dual: far more consistent than on plain bisection
    tangential_error(opts) = begin
        g = TRiSKGrid(create_hexagon_mesh(4; opts...); centers=create_spherical_mesh(4; opts...).coordinates)
        @test orthogonality_defect(g.dm, g.centers) < 1e-10
        ω = SVector(0.3, -0.2, 1.0)
        u = [dot(cross(ω, x), n) for (x, n) in zip(g.edge_points, g.edge_normals)]
        maximum(1:num_edges(g)) do e
            vt = sum(g.trisk_weights[k, e] * g.primal_lengths[g.trisk_edges[k, e]] * u[g.trisk_edges[k, e]]
                     for k in axes(g.trisk_edges, 1)) / g.dual_lengths[e]
            abs(vt - dot(cross(ω, g.edge_points[e]), cross(g.edge_points[e], g.edge_normals[e])))
        end
    end
    @test tangential_error((; centroidal=true)) < 0.1 * tangential_error((;))
end

@testset "Relaxed triangle meshes" begin
    # a perturbed icosahedral mesh (with obtuse triangles), cleaned up by spring dynamics
    rng = Random.MersenneTwister(1)
    dm = create_spherical_mesh(3)
    h = 0.25 * sqrt(4π / num_cells(dm))
    x = [normalize(p + h * normalize(cross(p, randn(rng, SVector{3, Float64})))) for p in dm.coordinates]
    perturbed = plex_from_polygons([collect(vertex_index.(Ref(dm), get_cell_vertices(dm, c))) for c in cell_range(dm)], x)
    function max_angle(m)
        maximum(cell_range(m)) do c
            v = [get_coordinates(m, p) for p in get_cell_vertices(m, c)]
            maximum(1:3) do k
                p, q, r = v[k], v[mod1(k + 1, 3)], v[mod1(k + 2, 3)]
                acosd(clamp(dot(normalize(q - dot(q, p)p), normalize(r - dot(r, p)p)), -1, 1))
            end
        end
    end
    relaxed = relax_triangles(perturbed)
    @test max_angle(perturbed) > 90 > 80 > max_angle(relaxed)
    @test total_area(relaxed) ≈ 4π
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

@testset "Periodic plane" begin
    for cells in (:quadrilateral, :triangle)
        dm = create_periodic_plane(6, 5; Lx=4π, Ly=2π, cells)
        @test dm.surface == :plane && dm.period == SVector(4π, 2π)
        @test check_plex(dm)  # closed torus, Euler characteristic 0
        @test num_cells(dm) == (cells == :quadrilateral ? 30 : 60)
        @test total_area(dm) ≈ 8π^2
        # cells across the seam are unwrapped: all cell areas positive and equal per type
        areas = compute_mesh_metrics(dm).cell_areas
        @test all(a -> a ≈ areas[1], areas)
        kites = create_kite_mesh(dm)
        @test check_plex(kites)
        @test total_area(kites) ≈ 8π^2
        @test all(x -> 0 <= x[1] < 4π && 0 <= x[2] < 2π, kites.coordinates)
    end
    @test_throws ArgumentError create_periodic_plane(2, 5)
end
