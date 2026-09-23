"""Structured quad mesh of the cornea section and Gmsh MSH 2.2 (ASCII) export."""
import numpy as np

# physical groups: name -> (dimension, tag)
PHYS = {
    "anterior":     (1, 1),
    "posterior":    (1, 2),
    "limbus":       (1, 3),
    "central_line": (1, 4),
    "cornea":       (2, 10),
}


def build_nodes(geom, Nx, Ny):
    """
    pts : node coordinates (x, z), shape (Ny, Nx, 2), with pts = phi(uv)
    uv  : reference coordinates (u, v),  shape (Ny, Nx, 2)
    i -> radial (u), j -> thickness (v, v=0 posterior, v=1 anterior).
    """
    U, V = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
    return geom.phi(U, V), np.stack([U, V], axis=-1)


def _quad_area(p):
    return 0.5 * sum(p[k - 1][0] * p[k][1] - p[k][0] * p[k - 1][1] for k in range(4))


def write_msh(filename, pts, uv):
    """Write nodes, quads, tagged boundary lines and the nodal fields u, v."""
    Ny, Nx, _ = pts.shape
    nid = lambda i, j: j * Nx + i + 1          # 1-based node id

    quads = []
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]   # counter-clockwise
            if _quad_area([pts[b, a] for a, b in q]) <= 0:
                raise ValueError(f"inverted element at (i, j) = ({i}, {j})")
            quads.append([nid(a, b) for a, b in q])

    lines = {
        "posterior":    [(nid(i, 0), nid(i + 1, 0)) for i in range(Nx - 1)],
        "anterior":     [(nid(i, Ny - 1), nid(i + 1, Ny - 1)) for i in range(Nx - 1)],
        "central_line": [(nid(0, j), nid(0, j + 1)) for j in range(Ny - 1)],
        "limbus":       [(nid(Nx - 1, j), nid(Nx - 1, j + 1)) for j in range(Ny - 1)],
    }

    with open(filename, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        f.write(f"$PhysicalNames\n{len(PHYS)}\n")
        for name, (dim, tag) in PHYS.items():
            f.write(f'{dim} {tag} "{name}"\n')
        f.write("$EndPhysicalNames\n")

        f.write(f"$Nodes\n{Nx * Ny}\n")
        for j in range(Ny):
            for i in range(Nx):
                f.write(f"{nid(i, j)} {pts[j, i, 0]:.12g} {pts[j, i, 1]:.12g} 0\n")
        f.write("$EndNodes\n")

        f.write(f"$Elements\n{sum(map(len, lines.values())) + len(quads)}\n")
        eid = 1
        for name, segs in lines.items():             # type 1 = 2-node line
            tag = PHYS[name][1]
            for a, b in segs:
                f.write(f"{eid} 1 2 {tag} {tag} {a} {b}\n")
                eid += 1
        tag = PHYS["cornea"][1]
        for q in quads:                               # type 3 = 4-node quad
            f.write(f"{eid} 3 2 {tag} {tag} {' '.join(map(str, q))}\n")
            eid += 1
        f.write("$EndElements\n")

        for k, name in enumerate(["u", "v"]):         # nodal fields
            f.write(f'$NodeData\n1\n"{name}"\n1\n0.0\n3\n0\n1\n{Nx * Ny}\n')
            for j in range(Ny):
                for i in range(Nx):
                    f.write(f"{nid(i, j)} {uv[j, i, k]:.12g}\n")
            f.write("$EndNodeData\n")
