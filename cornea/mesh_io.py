"""Structured quad mesh of the cornea section and Gmsh MSH 2.2 (ASCII) export,
carrying the per-element local orientation frame (circ, normal) as element
data in the same file.

Both write_msh and read_msh are written out explicitly rather than delegating
to meshio.write / meshio.read, because the installed meshio (checked on
5.3.5, the latest at time of writing) has two independent bugs that make the
plain meshio round-trip fail on this mesh:

  1. Write side, numpy>=2 dependent: meshio's ASCII writer formats each data
     value with repr(). With numpy>=2, repr(np.float64(0.0)) is the string
     "np.float64(0.0)" instead of "0.0", so mesh.write(...) corrupts the file
     it writes -- meshio.read() then fails with "could not be read to its
     end due to unmatched data".
  2. Read side, independent of numpy: meshio's gmsh2.2 reader reconstructs
     cell_data by calling cell_data_from_raw(cells, cell_data_raw), where
     `cells` is still a list of plain (type_str, ndarray) tuples. That
     function does `cs = np.cumsum([len(c) for c in cells])[:-1]` --
     `len(c)` on a 2-tuple is always 2, regardless of how many elements are
     actually in that cell block -- so any mesh with more than one cell type
     (this one always has line + quad) gets split at the wrong point, and
     meshio.read() raises "Incompatible cell data ... has length N, but
     corresponding cell data item has length 2." (Confirmed correct in
     meshio 5.0.0, broken in 5.3.5 -- a version regression, not a fluke.)

write_msh below writes plain, explicitly-formatted floats (no repr()), so
bug 1 never arises regardless of numpy version. read_msh parses
$Nodes/$Elements/$ElementData itself and hands meshio.Mesh already-correctly
-split arrays, so bug 2's buggy split function is never called. The returned
object is a completely normal meshio.Mesh (points, cells, cell_data,
field_data, cell_data_dict, get_cells_type, ...), so downstream code
(plot_mesh, plot_fields, plot_tags) is unaffected.

Storage format for circ/normal: Gmsh's $ElementData only allows 1, 3 or 9
field components per view (scalar, vector, tensor -- this is a hard rule of
the file format itself, not a meshio restriction), so a literal 2-component
vector isn't a legal Gmsh data block. circ and normal are each written as a
single 3-component ("vector") field, "circ" and "normal", with a dummy 0 for
the out-of-plane component: (circ_x, circ_z, 0) and (normal_x, normal_z, 0).
This keeps the file's vector fields dimension-agnostic -- exactly like the
node coordinates, which are always (x, y, z) even here where y is always
0 -- and lets Gmsh's own viewer recognise and draw them as vectors, rather
than as two unrelated scalar colour maps.
"""
import numpy as np

from .jacobian import push_vector, push_normal

# physical groups: name -> (dimension, tag)
PHYS = {
    "anterior":     (1, 1),
    "posterior":    (1, 2),
    "limbus":       (1, 3),
    "central_line": (1, 4),
    "cornea":       (2, 10),
}

_GMSH_ELEM_TYPE = {1: "line", 3: "quad"}   # the only two element types write_msh emits


def build_nodes(geom, Nx, Ny):
    """
    pts : node coordinates (x, z), shape (Ny, Nx, 2), with pts = phi(uv)
    uv  : reference coordinates (u, v),  shape (Ny, Nx, 2)
    i -> radial (u), j -> thickness (v, v=0 posterior, v=1 anterior).
    """
    U, V = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
    return geom.phi(U, V), np.stack([U, V], axis=-1)


def element_frames(geom, uv):
    """
    Local referencing system (circ, normal) at each quad's centroid, from the
    exact map: circ = unit F.e_u (circumferential direction), normal = unit
    F^-T.e_v (through-thickness direction, posterior -> anterior). Each is
    (n_elem, 2), in the same (i fast, j slow) element order as the quads
    built in write_msh.
    """
    uc = 0.25 * (uv[:-1, :-1] + uv[:-1, 1:] + uv[1:, :-1] + uv[1:, 1:])  # (Ny-1, Nx-1, 2)
    uc = uc.reshape(-1, 2)
    F = geom.F_exact(uc[:, 0], uc[:, 1])
    return push_vector(F, [1.0, 0.0]), push_normal(F, [0.0, 1.0])


def _quad_area(p):
    return 0.5 * sum(p[k - 1][0] * p[k][1] - p[k][0] * p[k - 1][1] for k in range(4))


def write_msh(filename, pts, circ, normal):
    """Write nodes, quads, tagged boundary lines, and the per-element local
    frame as two 3-component element-data vector fields, "circ" and "normal"
    -- (circ_x, circ_z, 0) and (normal_x, normal_z, 0), the out-of-plane
    component always 0 (see the module docstring for why 3, not 2,
    components).

    Every element carries a value for each field -- boundary lines get a
    dummy (0, 0, 0), quads their real vector -- because Gmsh's ElementData is
    one flat, file-order array per field, later split back into per-cell-type
    blocks; see read_msh for why that split is redone by hand there.
    """
    Ny, Nx, _ = pts.shape
    nid = lambda i, j: j * Nx + i + 1          # 1-based node id

    quads = []
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]   # counter-clockwise
            if _quad_area([pts[b, a] for a, b in q]) <= 0:
                raise ValueError(f"inverted element at (i, j) = ({i}, {j})")
            quads.append([nid(a, b) for a, b in q])

    if len(quads) != len(circ) or len(quads) != len(normal):
        raise ValueError("circ/normal must have one entry per quad element")

    lines = {
        "posterior":    [(nid(i, 0), nid(i + 1, 0)) for i in range(Nx - 1)],
        "anterior":     [(nid(i, Ny - 1), nid(i + 1, Ny - 1)) for i in range(Nx - 1)],
        "central_line": [(nid(0, j), nid(0, j + 1)) for j in range(Ny - 1)],
        "limbus":       [(nid(Nx - 1, j), nid(Nx - 1, j + 1)) for j in range(Ny - 1)],
    }
    n_lines = sum(map(len, lines.values()))

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

        f.write(f"$Elements\n{n_lines + len(quads)}\n")
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

        fields = {"circ": circ, "normal": normal}
        for name, vecs in fields.items():             # 3 components: (x, z, 0)
            f.write(f'$ElementData\n1\n"{name}"\n1\n0.0\n3\n0\n3\n{n_lines + len(quads)}\n')
            for eid_line in range(1, n_lines + 1):
                f.write(f"{eid_line} 0 0 0\n")
            for k, (vx, vz) in enumerate(vecs):
                f.write(f"{n_lines + k + 1} {vx:.12g} {vz:.12g} 0\n")
            f.write("$EndElementData\n")


def _iter_blocks(text, name):
    """Yield the (start, end) character span of the content of every
    $name ... $EndName block in text (start/end exclude the tag lines)."""
    tag_start, tag_end = f"${name}\n", f"$End{name}\n"
    pos = 0
    while True:
        s = text.find(tag_start, pos)
        if s == -1:
            return
        s += len(tag_start)
        e = text.find(tag_end, s)
        yield s, e
        pos = e + len(tag_end)


def _read_element_data(text, name):
    """One named $ElementData block: an (num_items, num_components) array
    over the file's whole element range (1-based ids, file order), regardless
    of cell type."""
    for s, e in _iter_blocks(text, "ElementData"):
        block_lines = text[s:e].splitlines()
        pos = 0
        n_string_tags = int(block_lines[pos]); pos += 1
        tags = block_lines[pos:pos + n_string_tags]; pos += n_string_tags
        if tags[0].strip().strip('"') != name:
            continue
        n_real_tags = int(block_lines[pos]); pos += 1 + n_real_tags
        n_int_tags = int(block_lines[pos]); pos += 1
        int_tags = [int(x) for x in block_lines[pos:pos + n_int_tags]]; pos += n_int_tags
        num_components = int_tags[1]
        num_items = int_tags[2]
        data = np.zeros((num_items, num_components))
        for line in block_lines[pos:pos + num_items]:
            parts = line.split()
            eid = int(parts[0])
            data[eid - 1] = [float(v) for v in parts[1:1 + num_components]]
        return data
    raise ValueError(f'$ElementData block "{name}" not found in the mesh file')


def read_msh(filename):
    """Read back a mesh written by write_msh -- points, tagged cells, and the
    per-element (circ, normal) local frame -- as a meshio.Mesh, built by hand
    rather than via meshio.read() (see the module docstring for why).
    """
    import meshio

    with open(filename) as f:
        text = f.read()

    n_start, n_end = next(_iter_blocks(text, "Nodes"))
    node_lines = text[n_start:n_end].splitlines()
    n_nodes = int(node_lines[0])
    points = np.array([[float(x) for x in ln.split()[1:]] for ln in node_lines[1:1 + n_nodes]])

    field_data = {}
    phys_block = next(_iter_blocks(text, "PhysicalNames"), None)
    if phys_block is not None:
        p_start, p_end = phys_block
        phys_lines = text[p_start:p_end].splitlines()
        for ln in phys_lines[1:1 + int(phys_lines[0])]:
            dim_str, tag_str, name = ln.split(None, 2)
            field_data[name.strip().strip('"')] = np.array([int(tag_str), int(dim_str)])

    e_start, e_end = next(_iter_blocks(text, "Elements"))
    elem_lines = text[e_start:e_end].splitlines()
    n_elems = int(elem_lines[0])

    cell_types, cell_idx, cell_phys = [], [], []
    for ln in elem_lines[1:1 + n_elems]:
        parts = [int(x) for x in ln.split()]
        etype, ntags = parts[1], parts[2]
        phys = parts[3] if ntags > 0 else 0
        node_ids0 = [n - 1 for n in parts[3 + ntags:]]
        t = _GMSH_ELEM_TYPE[etype]
        if not cell_types or cell_types[-1] != t:            # new block on type change
            cell_types.append(t)
            cell_idx.append([])
            cell_phys.append([])
        cell_idx[-1].append(node_ids0)
        cell_phys[-1].append(phys)

    cells = [(t, np.array(idx, dtype=int)) for t, idx in zip(cell_types, cell_idx)]
    phys_arrays = [np.array(p, dtype=int) for p in cell_phys]
    counts = [len(idx) for idx in cell_idx]
    cs = np.cumsum(counts)[:-1]                              # correct per-block split points

    cell_data = {"gmsh:physical": phys_arrays, "gmsh:geometrical": phys_arrays}
    for name in ("circ", "normal"):
        raw = _read_element_data(text, name)                 # (n_elems, 3)
        cell_data[name] = np.split(raw, cs)

    return meshio.Mesh(points, cells, cell_data=cell_data, field_data=field_data)