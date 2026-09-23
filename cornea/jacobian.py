"""Recover F = d(x,z)/d(u,v) from a mesh with nodal fields u, v, and push fields forward."""
import numpy as np

# bilinear shape-function derivatives at the element centre (xi = eta = 0),
# node order of a 4-node quad: (-1,-1), (1,-1), (1,1), (-1,1)
_dN_dxi = 0.25 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)


def element_jacobians(mesh):
    """For a meshio mesh: element centroids (x,z), centroid (u,v), and F per quad."""
    xz = mesh.points[:, :2]
    uv = np.stack([mesh.point_data["u"], mesh.point_data["v"]], axis=-1)
    quads = np.vstack([c.data for c in mesh.cells if c.type == "quad"])
    Xe, Ue = xz[quads], uv[quads]                      # (n_elem, 4, 2)
    Jx = np.einsum("eai,aj->eij", Xe, _dN_dxi)         # d(x,z)/d(xi,eta)
    Ju = np.einsum("eai,aj->eij", Ue, _dN_dxi)         # d(u,v)/d(xi,eta)
    return Xe.mean(axis=1), Ue.mean(axis=1), Jx @ np.linalg.inv(Ju)


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def push_vector(F, A):
    """Material direction A (defined in the square) -> unit vector F.A."""
    return _unit(np.einsum("eij,j->ei", F, A))


def push_normal(F, N):
    """Normal / gradient direction N (defined in the square) -> unit vector F^-T.N."""
    return _unit(np.einsum("eji,j->ei", np.linalg.inv(F), N))


def pushed_fields(F):
    """Fibre along u (F.e_u) and normal to v = const, posterior -> anterior (F^-T.e_v)."""
    return push_vector(F, [1.0, 0.0]), push_normal(F, [0.0, 1.0])


def diagnostics(F, F_ref, fibre, normal):
    rel = np.linalg.norm(F - F_ref, axis=(1, 2)) / np.linalg.norm(F_ref, axis=(1, 2))
    grid = push_vector(F, [0.0, 1.0])
    ang = np.degrees(np.arccos(np.clip(np.abs((grid * normal).sum(1)), 0, 1)))
    return (f"{len(F)} elements, relative error of F vs exact map: "
            f"median {np.median(rel):.1e}, max {rel.max():.1e}\n"
            f"min det F = {np.linalg.det(F).min():.3f} (> 0: orientation preserved)\n"
            f"max |fibre . normal| = {np.abs((fibre * normal).sum(1)).max():.1e}\n"
            f"max angle between v-grid line and true normal = {ang.max():.1f} deg")
