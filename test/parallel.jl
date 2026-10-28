using KernelAbstractions

@testset "Block solver (CPU loop and kernel)" begin
    V = FESpace(create_kite_mesh(create_hexagon_mesh(1)), KiteBrokenRT(1))
    S = mass_solver(V)
    B = block_solver(S)
    Random.seed!(3)
    b = randn(num_dofs(V))
    x = ldiv!(zeros(num_dofs(V)), B, b)
    @test x ≈ S \ b
    # the device kernel, run on the CPU backend
    y = zeros(num_dofs(V))
    dmplex.FEM.block_solve_kernel!(CPU())(y, b, B.indices, B.first, B.count, B.coefficient, B.values;
                                          ndrange=length(B.indices))
    KernelAbstractions.synchronize(CPU())
    @test y ≈ x
end

@testset "Row matrix and host solver" begin
    V = FESpace(create_cubed_sphere_mesh(2), RaviartThomas(1))
    S = mass_solver(V)
    A = S.matrix[1:2:end, :]
    R = RowMatrix(sparse(A'), CPU())
    Random.seed!(4)
    x = randn(num_dofs(V))
    @test row_mul!(zeros(size(A, 1)), R, x, -2.0) ≈ -2 * (A * x)
    # the device kernel, run on the CPU backend
    y = zeros(size(A, 1))
    dmplex.FEM.row_mul_kernel!(CPU())(y, R.colptr, R.rowval, R.nzval, x, -2.0; ndrange=size(A, 1))
    KernelAbstractions.synchronize(CPU())
    @test y ≈ -2 * (A * x)
    @test ldiv!(zeros(num_dofs(V)), HostSolver(S, num_dofs(V)), x) ≈ S \ x
    @test device_solver(S, CPU()) === S
end

@testset "Halo pack and unpack kernels" begin
    x, dofs = collect(1.0:10.0), [7, 2, 9]
    buffer = zeros(3)
    dmplex.Parallel.pack_kernel!(CPU())(buffer, x, dofs; ndrange=3)
    KernelAbstractions.synchronize(CPU())
    @test buffer == [7.0, 2.0, 9.0]
    dmplex.Parallel.unpack_kernel!(CPU())(x, -buffer, dofs; ndrange=3)
    KernelAbstractions.synchronize(CPU())
    @test x[dofs] == [-7.0, -2.0, -9.0]
end

@testset "Row distribution of classical spaces" begin
    dm = create_cubed_sphere_mesh(3)
    nranks = 3
    part = equal_area_partition(dm, nranks)
    for family in (RaviartThomas(1), Lagrange(2), Discontinuous(1))
        V = FESpace(dm, family)
        M = assemble_mass(V)
        distributions = [DofDistribution(V, CellDistribution(dm, part, nranks, rank)) for rank in 0:nranks-1]
        owner = last(first(distributions))
        # every dof is owned exactly once, and ghosts are owned elsewhere
        owned_ids = reduce(vcat, (d.global_ids[owned(d)] for (d, _) in distributions))
        @test sort(owned_ids) == 1:num_dofs(V)
        @test all(owner[i] == p for (p, (d, _)) in enumerate(distributions) for i in d.global_ids[owned(d)])
        # the distributed upper rows assemble to the upper triangle of the renumbered matrix
        number, offsets = contiguous_numbering(owner)
        @test offsets[end] == num_dofs(V) && issorted(offsets)
        blocks = [distributed_rows(M, d, owner, rank) for (rank, (d, _)) in zip(0:nranks-1, distributions)]
        @test [first_row for (_, first_row) in blocks] == offsets[1:end-1]
        order = invperm(number)
        @test reduce(hcat, (T for (T, _) in blocks)) == sparse(triu(M[order, order])')
    end
end

@testset "VTK piece for VTX output" begin
    kites = create_kite_mesh(create_hexagon_mesh(1))
    cells = 5:20
    p = VTKPiece(kites, cells)
    @test p.celltype == 9
    @test size(p.connectivity) == (5, length(cells))
    @test all(p.connectivity[1, :] .== 4)
    @test minimum(p.connectivity[2:end, :]) == 0 && maximum(p.connectivity[2:end, :]) == size(p.points, 2) - 1
    @test all(c -> isapprox(sum(abs2, c), 1; atol=1e-12), eachcol(p.points))
    @test occursin("<DataArray Name=\"vorticity\" />", dmplex.Outputs.vtx_schema(["vorticity"]))
end
