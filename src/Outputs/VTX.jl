"""
    VTKPiece

Unstructured-grid piece of a set of cells in VTK layout with local vertex
numbering: `points` (3 × vertices), `connectivity` (1 + nodes per cell ×
cells, the node count followed by 0-based local vertex indices, as legacy VTK)
and the VTK cell type shared by all cells.
"""
struct VTKPiece
    points::Matrix{Float64}
    connectivity::Matrix{Int64}
    celltype::UInt32
end

"VTK piece of the cells `cells` (cell indices) of `dm`; all cells must have the same number of vertices."
function VTKPiece(dm::Plex, cells::AbstractVector{<:Integer}=1:num_cells(dm))
    first_cell = first(cell_range(dm)) - 1
    vertices = [[vertex_index(dm, v) for v in get_cell_vertices(dm, first_cell + c)] for c in cells]
    nodes = isempty(vertices) ? 4 : length(first(vertices))
    all(v -> length(v) == nodes, vertices) ||
        throw(ArgumentError("VTX output needs a single cell type (all cells with the same number of vertices)"))
    used = sort!(unique!(reduce(vcat, vertices; init=Int[])))
    to_local = Dict(v => i - 1 for (i, v) in enumerate(used))
    connectivity = Matrix{Int64}(undef, nodes + 1, length(cells))
    for (j, v) in enumerate(vertices)
        connectivity[1, j] = nodes
        connectivity[2:end, j] .= [to_local[i] for i in v]
    end
    points = reduce(hcat, (Vector{Float64}(dm.coordinates[v]) for v in used); init=zeros(3, 0))
    celltype = nodes == 3 ? 5 : nodes == 4 ? 9 : 7   # VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON
    return VTKPiece(points, connectivity, UInt32(celltype))
end

"VTX data model (`vtk.xml` attribute) of an unstructured grid with the given cell-data names."
function vtx_schema(cell_data)
    arrays = join(("        <DataArray Name=\"$name\" />\n" for name in cell_data))
    return """
    <VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
      <UnstructuredGrid>
        <Piece NumberOfPoints="NumberOfNodes" NumberOfCells="NumberOfCells">
          <Points>
            <DataArray Name="geometry" />
          </Points>
          <Cells>
            <DataArray Name="connectivity" />
            <DataArray Name="types" />
          </Cells>
          <CellData>
    $(arrays)        <DataArray Name="TIME">step</DataArray>
          </CellData>
        </Piece>
      </UnstructuredGrid>
    </VTKFile>
    """
end

"""
    writer = vtx_writer(filename, dm; cells=1:num_cells(dm), cell_data=(:vorticity,), comm=nothing, engine="BP5")
    write_vtx!(writer, time; vorticity=values)
    close(writer)

ADIOS2 output of cell data on the unstructured grid of `dm` in the VTX data
model, read by ParaView's ADIOS2 VTX reader (open `filename`, a `.bp`
directory). With an MPI communicator every rank writes its own block: the
cells `cells` (cell indices, e.g. the owned kites `model.distribution.kite_ids[owned_kites(d)]`)
with local vertex numbering, and one value per cell for every name in
`cell_data`. Values may be host arrays or GPU arrays (`CuArray`); device data
is handed to ADIOS2 directly, without a copy to the host, which needs an
ADIOS2 library built with CUDA (`JULIA_ADIOS2_PATH`, then
`Pkg.build("ADIOS2")`). Needs `using ADIOS2` (package extension).
"""
function vtx_writer end

"Write one output step; see `vtx_writer`."
function write_vtx! end
