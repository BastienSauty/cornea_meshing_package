"""
Command line:  python -m cornea params.json [-o output_dir]

Builds the geometry, writes cornea.msh, then reads it back to compute F and plot
cornea_mesh.png, cornea_fields.png and cornea_tags.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import meshio

from .params import CorneaParams
from .geometry import CorneaGeometry
from .mesh_io import build_nodes, write_msh
from .jacobian import element_jacobians, pushed_fields, diagnostics
from .plots import plot_mesh, plot_fields, plot_tags


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cornea", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", help="JSON file with R_v, R_a, Q_a, R_p, Q_p, h, Nx, Ny")
    ap.add_argument("-o", "--output", default="output", help="output directory (default: output)")
    args = ap.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    p = CorneaParams.from_json(args.params)

    # 1. geometry and mesh
    geom = CorneaGeometry(p)
    print(geom.summary())
    pts, uv = build_nodes(geom, p.Nx, p.Ny)
    write_msh(out / "cornea.msh", pts, uv)
    plot_mesh(pts, out / "cornea_mesh.png")

    # 2. Jacobian recovered from the written mesh only (nodes + nodal u, v)
    mesh = meshio.read(out / "cornea.msh")
    centroids, uc, F = element_jacobians(mesh)
    fibre, normal = pushed_fields(F)
    print(diagnostics(F, geom.F_exact(uc[:, 0], uc[:, 1]), fibre, normal))
    plot_fields(mesh, centroids, fibre, normal, out / "cornea_fields.png")
    plot_tags(mesh, out / "cornea_tags.png")

    print(f"written to {out}/: cornea.msh, cornea_mesh.png, cornea_fields.png, cornea_tags.png")


if __name__ == "__main__":
    main()
