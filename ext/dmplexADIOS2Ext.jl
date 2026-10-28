"ADIOS2 output in the VTX data model (unstructured grid with cell data), from host or GPU arrays."
module dmplexADIOS2Ext

using ADIOS2
using KernelAbstractions
using MPI
using dmplex
using dmplex.Outputs: VTKPiece, vtx_schema
import dmplex.Outputs: vtx_writer, write_vtx!

const MEMORY_SPACE_HOST = Cint(1)
const MEMORY_SPACE_GPU = Cint(2)   # adios2_memory_space_gpu, only in ADIOS2 builds with GPU support

mutable struct VTXWriter
    adios::Adios
    io::AIO
    engine::Engine
    piece::VTKPiece
    variables::Dict{Symbol, Variable}
    memory_space::Dict{Symbol, Cint}
    keep::Vector{Any}         # arrays handed to ADIOS2 by pointer, alive until the end of the step
    open::Bool
end

function vtx_writer(filename::AbstractString, dm::Plex; cells::AbstractVector{<:Integer}=1:num_cells(dm),
                    cell_data=(:vorticity,), comm=nothing, engine::AbstractString="BP5")
    piece = VTKPiece(dm, cells)
    adios = comm === nothing ? adios_init_serial() : adios_init_mpi(comm)
    io = declare_io(adios, "dmplex-vtx")
    set_engine(io, engine)
    define_attribute(io, "vtk.xml", vtx_schema(String.(cell_data)))
    variables = Dict{Symbol, Variable}(
        :NumberOfNodes => define_variable(io, "NumberOfNodes", UInt32, LocalValue()),
        :NumberOfCells => define_variable(io, "NumberOfCells", UInt32, LocalValue()),
        :types => define_variable(io, "types", UInt32),
        :step => define_variable(io, "step", Float64),
        # local arrays (count only): one block per rank, Julia sizes are reversed to C order
        :geometry => define_variable(io, "geometry", Float64, nothing, nothing, size(piece.points)),
        :connectivity => define_variable(io, "connectivity", Int64, nothing, nothing, size(piece.connectivity)))
    for name in cell_data
        variables[Symbol(name)] = define_variable(io, String(name), Float64, nothing, nothing, (length(cells),))
    end
    return VTXWriter(adios, io, open(io, filename, mode_write), piece, variables, Dict{Symbol, Cint}(), Any[], true)
end

"Hand `data` to ADIOS2: host arrays as they are, device arrays by their device pointer."
function put_data!(w::VTXWriter, name::Symbol, data::AbstractVector{Float64})
    variable = w.variables[name]
    length(data) == count(variable)[1] ||
        throw(ArgumentError("$name: expected $(count(variable)[1]) values, got $(length(data))"))
    host = KernelAbstractions.get_backend(data) isa KernelAbstractions.CPU
    space = host ? MEMORY_SPACE_HOST : MEMORY_SPACE_GPU
    if get(w.memory_space, name, Cint(-1)) != space
        err = ccall((:adios2_set_memory_space, ADIOS2.libadios2_c), Cint, (Ptr{Cvoid}, Cint), variable.ptr, space)
        err == 0 || error("ADIOS2 cannot take $(host ? "host" : "GPU") buffers (error $err); GPU output needs " *
                          "an ADIOS2 library built with CUDA (set JULIA_ADIOS2_PATH and run Pkg.build(\"ADIOS2\"))")
        w.memory_space[name] = space
    end
    push!(w.keep, data)
    if host
        put!(w.engine, variable, data)
    else
        put!(w.engine, variable, reinterpret(Ptr{Float64}, pointer(data)))
    end
    return nothing
end

function write_vtx!(w::VTXWriter, time::Real; cell_data...)
    w.open || throw(ArgumentError("the writer is closed"))
    p = w.piece
    begin_step(w.engine)
    put!(w.engine, w.variables[:step], Float64(time))
    put!(w.engine, w.variables[:types], p.celltype)
    put!(w.engine, w.variables[:NumberOfNodes], UInt32(size(p.points, 2)))
    put!(w.engine, w.variables[:NumberOfCells], UInt32(size(p.connectivity, 2)))
    put!(w.engine, w.variables[:geometry], p.points)
    put!(w.engine, w.variables[:connectivity], p.connectivity)
    for (name, values) in cell_data
        haskey(w.variables, name) || throw(ArgumentError("$name was not declared in cell_data of vtx_writer"))
        put_data!(w, name, values)
    end
    end_step(w.engine)
    empty!(w.keep)
    return w
end

function Base.close(w::VTXWriter)
    w.open || return nothing
    close(w.engine)
    adios_finalize(w.adios)
    w.open = false
    return nothing
end

end # module
