"Spherical mesh generators and grid-file readers; all return a `Plex` on the unit sphere."
module Grids

using LinearAlgebra
using NCDatasets: NCDataset
using SparseArrays
using StaticArrays
using ..Topology

export create_spherical_mesh, create_cubed_sphere_mesh, create_hexagon_mesh, create_healpix_mesh,
    create_periodic_plane
export read_grid, read_icon_grid, read_mpas_grid
export create_kite_mesh, cell_centers, orthogonality_defect, spring_dynamics!, lloyd!

include("Icosahedral.jl")
include("CubedSphere.jl")
include("FileGrids.jl")
include("CellCenters.jl")
include("Hexagonal.jl")
include("HEALPix.jl")
include("Plane.jl")
include("Kite.jl")

end # module
