"""
Command line:  python -m cornea params.json [-o output_dir]

Builds the geometry, writes cornea.msh (mesh, tagged boundaries, and the
per-element circ/normal orientation as element data), then reads it back
with mesh_io.read_msh to plot cornea_mesh.png, cornea_fields.png and
cornea_tags.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from .params import CorneaParams
from .geometry import CorneaGeometry
from .mesh_io import build_nodes, element_frames, write_msh, read_msh
from .jacobian import read_frames, diagnostics
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

    # 1. geometry, mesh, and the per-element local frame (circ, normal) from the exact map
    geom = CorneaGeometry(p)
    print(geom.summary())
    pts, uv = build_nodes(geom, p.Nx, p.Ny)
    circ, normal = element_frames(geom, uv)
    write_msh(out / "cornea.msh", pts, circ, normal)
    plot_mesh(pts, out / "cornea_mesh.png")

    # 2. read the mesh back: circ/normal come straight from element data, no Jacobian recovery
    mesh = read_msh(out / "cornea.msh")
    centroids, circ_r, normal_r = read_frames(mesh)
    uc = 0.25 * (uv[:-1, :-1] + uv[:-1, 1:] + uv[1:, :-1] + uv[1:, 1:]).reshape(-1, 2)
    print(diagnostics(circ_r, normal_r, geom.F_exact(uc[:, 0], uc[:, 1])))
    plot_fields(mesh, centroids, circ_r, normal_r, out / "cornea_fields.png")
    plot_tags(mesh, out / "cornea_tags.png")

    print(f"written to {out}/: cornea.msh, cornea_mesh.png, cornea_fields.png, cornea_tags.png")


if __name__ == "__main__":
    main()