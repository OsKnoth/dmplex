# Polyhedral finite volumes: description and documentation

- [polyhedral.tex](polyhedral.tex): the method of POLYEDER as it is carried over to
  `src/Polyhedral/` (paper draft): cell complexes, the exact integration of monomials over
  polyhedral cells, the geometry, the polynomial reconstruction, the face gradient from interaction
  regions, the shallow-water equations and cut cells.
- [polyhedral_doc.tex](polyhedral_doc.tex): user documentation of the submodule, how the functions
  are used, the tests and the examples, with figures of the test examples (`figures/`): meshes,
  solid-body rotation, Galewsky, cut cells, and flow over the Gallus–Klemp hill on a Cartesian
  cut-cell grid.

```sh
latexmk -pdf polyhedral.tex
latexmk -pdf polyhedral_doc.tex
```

The numbers of Table 1 of `polyhedral.tex` (volumes of a spherical shell) come from `PolyGeometry` on
`extruded_complex(ExtrudedPlex(mesh, 1), [0.0, 0.1]; radius=1.0)`; the tests are in
`test/polyhedral.jl`. The figures of `polyhedral_doc.tex` are made by the scripts in
`examples/polyhedral/`: `figure_data.jl` (meshes, rotation, cut cells), `galewsky_poly.jl` (with
`OUT=file`), `mountain.jl` (hourly `gk_HHh.bin`), plotted by `plot_doc.py` and `plot_mountain.py`
(matplotlib, on Levante).
