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
