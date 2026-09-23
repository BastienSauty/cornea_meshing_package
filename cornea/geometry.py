"""
Cornea geometry and the map phi: unit square (u, v) -> cornea section (x, z).

Reference square:  u in [0,1] -> radial   (u=0 central line, u=1 limbus)
                   v in [0,1] -> thickness (v=0 posterior,   v=1 anterior)
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


def f_conic(x, R, Q):
    """Conic sag f(x) for radius R and asphericity Q."""
    return x**2 / R / (1 + np.sqrt(1 - (1 + Q) * x**2 / R**2))


def df_conic(x, R, Q):
    """Slope f'(x) of the conic sag."""
    return x / np.sqrt(R**2 - (1 + Q) * x**2)


def _arclength_curve(f, x_end, n_dense=2000):
    """C(u), u in [0,1]: point on (x, f(x)) at fraction u of the arc length (smooth spline)."""
    xd = np.linspace(0.0, x_end, n_dense)
    zd = f(xd)
    sd = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xd), np.diff(zd)))])
    x_of_s = CubicSpline(sd / sd[-1], xd)

    def C(u):
        x = x_of_s(np.clip(u, 0.0, 1.0))
        return np.stack([x, f(x)], axis=-1)
    return C


class CorneaGeometry:
    def __init__(self, p):
        self.p = p
        # translation so that the posterior end point sits at z = 0
        self.trans = p.h + f_conic(p.R_v, p.R_p, p.Q_p)

        # limbus edge: normal to the posterior face at x = R_v, up to the anterior face
        self.P = np.array([p.R_v, self.z_post(p.R_v)])
        s = self.dz_post(p.R_v)
        self.tangent = np.array([1.0, s]) / np.hypot(1.0, s)
        self.normal = np.array([-s, 1.0]) / np.hypot(1.0, s)   # posterior -> anterior
        g = lambda t: self.z_ant(self.P[0] + t * self.normal[0]) - (self.P[1] + t * self.normal[1])
        self.edge_length = brentq(g, 0.0, 5.0 * (self.z_ant(p.R_v) - self.P[1]) + 1.0)
        self.A = self.P + self.edge_length * self.normal
        self.x_a_end = self.A[0]

        # the four boundaries of the Coons patch
        self.C_post = _arclength_curve(self.z_post, p.R_v)       # v = 0
        self.C_ant = _arclength_curve(self.z_ant, self.x_a_end)  # v = 1
        self.P00, self.P10 = self.C_post(0.0), self.C_post(1.0)
        self.P01, self.P11 = self.C_ant(0.0), self.C_ant(1.0)

    # surfaces in their final position (apex up, opening downwards)
    def z_ant(self, x):
        return self.trans - f_conic(x, self.p.R_a, self.p.Q_a)

    def z_post(self, x):
        return self.trans - self.p.h - f_conic(x, self.p.R_p, self.p.Q_p)

    def dz_post(self, x):
        return -df_conic(x, self.p.R_p, self.p.Q_p)

    def phi(self, u, v):
        """Coons map (x, z) = phi(u, v); u, v arrays of the same shape."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        U, V = u[..., None], v[..., None]
        D_axis = (1 - V) * self.P00 + V * self.P01       # u = 0 edge
        D_limb = (1 - V) * self.P10 + V * self.P11       # u = 1 edge
        return ((1 - V) * self.C_post(u) + V * self.C_ant(u)
                + (1 - U) * D_axis + U * D_limb
                - ((1 - U) * (1 - V) * self.P00 + U * (1 - V) * self.P10
                   + (1 - U) * V * self.P01 + U * V * self.P11))

    def F_exact(self, u, v, eps=1e-6):
        """F = d(x,z)/d(u,v) of the exact map, by central differences. Shape (..., 2, 2)."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        dphi_du = (self.phi(u + eps, v) - self.phi(u - eps, v)) / (2 * eps)
        dphi_dv = (self.phi(u, v + eps) - self.phi(u, v - eps)) / (2 * eps)
        return np.stack([dphi_du, dphi_dv], axis=-1)

    def displacement(self, X, Y, L=1.0, H=1.0):
        """Displacement turning nodes (X, Y) of the square [0,L]x[0,H] into the cornea section."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        return self.phi(X / L, Y / H) - np.stack([X, Y], axis=-1)

    def summary(self):
        return (f"anterior end x = {self.x_a_end:.4f}, limbus edge length = {self.edge_length:.4f}\n"
                f"cos(limbus edge, posterior tangent) = {self.normal @ self.tangent:.1e} (0 = normal)\n"
                f"apex: posterior z = {self.z_post(0):.3f}, anterior z = {self.z_ant(0):.3f}")
