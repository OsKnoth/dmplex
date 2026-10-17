"""
Finite elements of arbitrary order on triangles and quadrilaterals: Lagrange,
discontinuous, Raviart–Thomas and Nédélec families, reference elements built
from a Legendre prime basis, function spaces on a `Plex`, and assembly.
"""
module FEM

using LinearAlgebra
using SparseArrays
using StaticArrays
using ..Topology
using ..Geometry

export ElementFamily, Lagrange, Discontinuous, RaviartThomas, Nedelec, KiteRadau, KiteBrokenRT, function_space, value_size,
    field, lagrange, discontinuous, raviart_thomas, nedelec
export QuadratureRule, quadrature_rule, gauss_legendre, gauss_lobatto, gauss_radau, legendre
export ReferenceElement, DofFunctional, num_dofs, tabulate, entity_dofs
export FESpace, reference_element, cell_dofs, evaluate, Tabulation, cell_basis, precompute_basis
export MassSolver, mass_solver, block_size
export assemble_kite_divergence, assemble_kite_gradient
export assemble_mass, assemble_bilinear, assemble_divergence, assemble_coriolis, assemble_load, project

include("Polynomials.jl")
include("Quadrature.jl")
include("Families.jl")
include("ReferenceElement.jl")
include("Lagrange.jl")
include("EdgeElements.jl")
include("KiteElements.jl")
include("FESpace.jl")
include("Assembly.jl")
include("MassSolver.jl")
include("KiteOperators.jl")

end # module
