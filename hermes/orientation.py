import numpy as np

import quaternion
from numba import jit
from poliastro.core.util import cross


def cart2polar(xyz):
    r = np.linalg.norm(xyz, axis=1)
    # r_xy = np.linalg.norm(xyz[:, 0:1], axis=1)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    theta = np.arccos(xyz[:, 2] / r)

    return r, phi, theta


def rot_quat(v1, v2):
    # From: http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    # Can be optimized a couple of steps further if required
    # ToDo support matrices

    w, x, y, z = wxyz(v1, v2)

    return np.quaternion(w, x, y, z).normalized()


@jit
def wxyz(v1, v2):
    """
    Calculates w, x, y, z coefficients for quaternion from v1 to v2
    Parameters
    ----------
    v1 :
    v2 :
    """

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    u1 = v1 / n1
    u2 = v2 / n2

    xyz = cross(u1, u2)
    w = np.dot(u1, u2) + 1

    return w, xyz[0], xyz[1], xyz[2]
