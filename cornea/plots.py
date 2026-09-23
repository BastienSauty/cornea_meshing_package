"""Figures. Each function draws on a given axis, or creates a figure and saves it."""
import numpy as np
import matplotlib.pyplot as plt


def _finish(fig, ax, title, filename):
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    if fig is not None:
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)


def _new(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
        return fig, ax
    return None, ax


def plot_mesh(pts, filename="cornea_mesh.png", ax=None):
    """Structured mesh from node coordinates pts of shape (Ny, Nx, 2)."""
    fig, ax = _new(ax)
    ax.plot(pts[..., 0].T, pts[..., 1].T, "k-", lw=0.6)   # constant v
    ax.plot(pts[..., 0], pts[..., 1], "k-", lw=0.6)       # constant u
    _finish(fig, ax, "Square mesh mapped onto the cornea section", filename)
    return ax


def plot_fields(mesh, centroids, fibre, normal, filename="cornea_fields.png", ax=None, every=2):
    """Fibre (F.e_u) and normal (F^-T.e_v) at element centroids, over the mesh read from .msh."""
    fig, ax = _new(ax)
    for c in mesh.cells:
        if c.type == "quad":
            for q in c.data:
                ax.fill(*mesh.points[q, :2].T, fc="none", ec="0.8", lw=0.4)
    k = slice(None, None, every)
    ax.quiver(*centroids[k].T, *fibre[k].T, color="tab:blue", scale=40, width=0.002,
              label="fibre = F·(1,0)")
    ax.quiver(*centroids[k].T, *normal[k].T, color="tab:red", scale=40, width=0.002,
              label="normal = F⁻ᵀ·(0,1)")
    ax.legend(loc="upper right")
    _finish(fig, ax, "Orientations pushed forward from the square", filename)
    return ax


def plot_tags(mesh, filename="cornea_tags.png", ax=None):
    """Boundary physical groups read back from the .msh."""
    fig, ax = _new(ax)
    for c in mesh.cells:
        if c.type == "quad":
            for q in c.data:
                ax.fill(*mesh.points[q, :2].T, fc="none", ec="0.8", lw=0.4)
    names = {v[0]: k for k, v in mesh.field_data.items() if v[1] == 1}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for c, phys in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if c.type != "line":
            continue
        for n, tag in enumerate(np.unique(phys)):
            segs = c.data[phys == tag]
            for s in segs:
                ax.plot(*mesh.points[s, :2].T, color=colors[tag % len(colors)], lw=2.5)
            ax.plot([], [], color=colors[tag % len(colors)], lw=2.5, label=f"{names[tag]} ({tag})")
    ax.legend(loc="upper right")
    _finish(fig, ax, "Physical groups", filename)
    return ax
