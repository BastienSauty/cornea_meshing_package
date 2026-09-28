"""Local orientation frame (circ, normal) per element: built from the exact
map in mesh_io.element_frames, stored in the mesh as two 3-component vector
fields ("circ", "normal", out-of-plane component 0), and reassembled here --
no Jacobian recovery from nodal fields."""
import numpy as np


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def push_vector(F, A):
    """Material direction A (defined in the square) -> unit vector F.A."""
    return _unit(np.einsum("eij,j->ei", F, A))


def push_normal(F, N):
    """Normal / gradient direction N (defined in the square) -> unit vector F^-T.N."""
    return _unit(np.einsum("eji,j->ei", np.linalg.inv(F), N))


def read_frames(mesh):
    """Element centroids (x, z), and the (circ, normal) local frame, one pair
    per quad, taken as the in-plane (x, z) part of the "circ"/"normal"
    3-component element data (the third, out-of-plane component is always
    0), in the mesh's quad cell order."""
    xz = mesh.points[:, :2]
    quads = mesh.get_cells_type("quad")
    centroids = xz[quads].mean(axis=1)

    cd = mesh.cell_data_dict
    circ = cd["circ"]["quad"][:, :2]
    normal = cd["normal"]["quad"][:, :2]
    return centroids, circ, normal


def diagnostics(circ, normal, F_ref=None):
    """Sanity checks on the stored local frame (orthogonality, orientation),
    and, if F_ref is given (e.g. geom.F_exact at the element centroids), its
    angular error against the exact map -- which now only measures the
    mesh's text I/O round-trip precision, since circ/normal are written from
    that same F_ref."""
    dot = np.abs((circ * normal).sum(1))
    cross = circ[:, 0] * normal[:, 1] - circ[:, 1] * normal[:, 0]
    msg = (f"{len(circ)} elements\n"
           f"max |circ . normal| = {dot.max():.1e} (0 = orthogonal)\n"
           f"min (circ x normal) = {cross.min():.3f} (> 0: orientation preserved)")
    if F_ref is not None:
        circ_ref = push_vector(F_ref, [1.0, 0.0])
        normal_ref = push_normal(F_ref, [0.0, 1.0])
        ang_c = np.degrees(np.arccos(np.clip(np.abs((circ * circ_ref).sum(1)), 0, 1)))
        ang_n = np.degrees(np.arccos(np.clip(np.abs((normal * normal_ref).sum(1)), 0, 1)))
        msg += (f"\nvs exact map (I/O round-trip check): "
                f"max angle error circ {ang_c.max():.1e} deg, normal {ang_n.max():.1e} deg")
    return msg