"""
    write_vtkhdf(filename, dm; cell_data...)

Write the surface mesh as VTKHDF unstructured grid (triangles, quads or
polygons) with cell-wise data given as keyword arguments, e.g.
`write_vtkhdf("out", dm; tracer=C)`.
"""
function write_vtkhdf(filename::AbstractString, dm::Plex; cell_data...)
    points = reduce(hcat, dm.coordinates)
    cells = map(cell_range(dm)) do c
        nodes = [vertex_index(dm, v) for v in get_cell_vertices(dm, c)]
        cell_type = length(nodes) == 3 ? VTKCellTypes.VTK_TRIANGLE :
                    length(nodes) == 4 ? VTKCellTypes.VTK_QUAD :
                    VTKCellTypes.VTK_POLYGON
        MeshCell(cell_type, nodes)
    end
    vtkhdf_grid(filename, points, cells) do vtk
        for (name, values) in cell_data
            vtk[String(name), VTKCellData()] = values
        end
    end
    return filename
end
