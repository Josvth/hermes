import numpy as np
from numba import jit, njit, prange
from numpy.core.umath_tests import inner1d
from astropy import time, units as u

from hermes.util import norm_along_rows


def body_angle_to_slant(psi, r, Rbody):
    b = r
    c = Rbody
    alpha = psi
    return np.sqrt(b ** 2 + c ** 2 - 2 * b * c * np.cos(alpha))


def slant_to_fov(slant, r, Rbody):
    a = slant
    b = r
    c = Rbody
    return np.arccos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b))


def elevation_to_slant(el, r, Rbody):
    """ Computes the slant range from the elevation angle"""
    beta = el + 90 * u.deg

    b = r
    c = Rbody

    # Solutions of cosine formula: a^2 - 2*c*cos(theta)*a - b^2 + c^2 = 0
    a1 = c * np.cos(beta) + np.sqrt(b ** 2 - c ** 2 * np.sin(beta) ** 2)
    a2 = c * np.cos(beta) - np.sqrt(b ** 2 - c ** 2 * np.sin(beta) ** 2)

    return np.maximum(a1, a2)  # Todo: is this always the maximum?


def fov_edge_range(r, theta, Rbody):
    """ Computes the range to the edge of the FOV circle"""

    b = r
    c = Rbody

    # Solutions of cosine formula: a^2 - 2*b*cos(theta)*a + b^2 - c^2 = 0
    a1 = b * np.cos(theta) + np.sqrt(c ** 2 - b ** 2 * np.sin(theta) ** 2)
    a2 = b * np.cos(theta) - np.sqrt(c ** 2 - b ** 2 * np.sin(theta) ** 2)

    return np.minimum(a1, a2)  # Todo: is this always the minimum?


@jit
def point_inside_cone(r, ttip, ttheta, pphi=None):
    # ToDo Implement ttheta and pphi properly (depends on pointing/orientation)
    """
    Checks if point r lies inside the cones starting with
    Parameters

    based on: https://stackoverflow.com/questions/12826117/how-can-i-detect-if-a-point-is-inside-a-cone-or-not-in-3d-space
    ----------
    r :
    ttip :
    """
    # ttip = np.atleast_2d(ttip)

    # For now we assume that the direction vectors are pointing towards (0, 0, 0)
    ddir = -ttip / norm_along_rows(ttip).reshape(-1, 1)

    # First take the differenc of our point and the tips of the cones (speed-up in non-numba)
    ddiff = r - ttip

    # First we find the distance along the axis of the cone
    ccone_dist = np.sum(ddiff * ddir, axis=1)

    # Then we find the radius of the cone at that distance
    ccone_radius = np.tan(ttheta) * ccone_dist

    # Then calculate the orthogonal distance
    oorth_dist = norm_along_rows(ddiff - (ddir.T * ccone_dist).T)

    # Check if inside
    insd = oorth_dist <= ccone_radius

    return insd


@jit
def point_inside_cone_audacy(r, ttip, ttheta, pphi=None):
    # ToDo Implement ttheta and pphi properly (depends on pointing/orientation)
    """
    Checks if point r lies inside the cones starting with
    Parameters

    based on: https://stackoverflow.com/questions/12826117/how-can-i-detect-if-a-point-is-inside-a-cone-or-not-in-3d-space
    ----------
    r :
    ttip :
    """
    # ttip = np.atleast_2d(ttip)

    # For now we assume that the direction vectors are pointing towards (0, 0, 0)
    ddir = -ttip / norm_along_rows(ttip).reshape(-1, 1)

    # First take the difference of our point and the tips of the cones (speed-up in non-numba)
    ddiff = r - ttip

    # First we find the distance along the axis of the cone
    ccone_dist = np.sum(ddiff * ddir, axis=1)

    # Then we find the radius of the cone at that distance
    ccone_radius = np.tan(ttheta) * ccone_dist  # outer radius of outer ring
    ccone_radius2 = np.tan(np.repeat(18.29 * np.pi / 180, len(ttheta))) * ccone_dist  # inner radius of outer ring
    ccone_radius3 = np.tan(np.repeat(16.55 * np.pi / 180, len(ttheta))) * ccone_dist  # outer radius of inner ring

    # Then calculate the orthogonal distance
    oorth_dist = norm_along_rows(ddiff - (ddir.T * ccone_dist).T)

    # Check if inside
    insd = oorth_dist <= ccone_radius3
    insd = insd + (oorth_dist >= ccone_radius2) * (oorth_dist <= ccone_radius)

    return insd


@jit
def line_intersects_sphere(pos_1, pos_2, pos_3, radius):
    """
    Checks if the shortest line spanning between xyz positions in pos_1 and xyz positions in pos_2 intersects with
    the body.
    Based of http://paulbourke.net/geometry/circlesphere/index.html#linesphere
    Intersection of a Line and a Sphere (or circle)

    Parameters
    ----------
    pos_3 : np.array
    pos_2 : np.array
    pos_1 : np.array

    Returns
    -------
    object
    """
    # Check if intersects a sphere
    #

    diff2_1 = pos_2 - pos_1
    diff3_1 = pos_3 - pos_1

    # num = np.dot(diff2_1, diff3_1.T).squeeze()
    num = np.sum(diff2_1 * diff3_1, axis=1)
    denum = np.sum(np.square(diff2_1), axis=1)

    uu = num.T / denum  # u vector

    p = pos_1 + (diff2_1.T * uu).T  # closest positions

    diffp = p - pos_3
    # dp = np.linalg.norm(diffp, axis=1)  # distances to sphere center
    dp = norm_along_rows(diffp)  # distances to sphere center

    isct = (0 < uu) * (uu < 1) * (dp < radius)  # check

    return isct

    # THIS IS NOT FASTER!!
    # isct = np.zeros(uu.size)
    #
    # for i, u in enumerate(uu):
    #
    #     # Check if closest point of line to the sphere is in between points
    #     isct[i] = (0 < u) * (u < 1)
    #
    #     if isct[i]:
    #         # If so check the distance
    #         p = pos_1 + diff2_1[i] * u  # closest positions
    #
    #         diffp = np.subtract(p, pos_3)
    #         dp = np.linalg.norm(diffp)  # distances to sphere center
    #
    #         isct[i] = dp <r
    #
    # return isct


@jit
def spherical_to_cartesian(R, theta, phi):
    z = R * np.cos(theta)
    y = R * np.sin(theta) * np.sin(phi)
    x = R * np.sin(theta) * np.cos(phi)

    return x, y, z

@njit
def normalize(v):
    return v / np.linalg.norm(v)

### Based on: https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
#@njit
def orthogonal(v):
    """ Returns a vector orhogonal to v"""
    x, y, z = np.abs(v)

    if x < y:
        if x < z:
            o = np.array([1, 0, 0]) # X
        else:
            o = np.array([0, 0, 1]) # Z
    else:
        if y < z:
            o = np.array([0, 1, 0]) # Y
        else:
            o = np.array([0, 0, 1]) # Z

    return np.cross(v, o)

#@njit
def find_quaternion(v1, v2):
    """ Finds a quaternion that rotates v1 to v2 """
    x1, y1, z1 = v1
    x2, y2, z2 = v2

    dot_v1v2 = np.dot(v1, v2)
    sqrt_v1v2 = np.sqrt((x1 ** 2 + y1 ** 2 + z1 ** 2) * (x2 ** 2 + y2 ** 2 + z2 ** 2))

    if dot_v1v2 / sqrt_v1v2 == -1:
        q = np.zeros(4)
        q[1:] = normalize(orthogonal(v1))
        return q
    else:
        qw = sqrt_v1v2 + dot_v1v2
        qx, qy, qz = np.cross(v1, v2)
        return normalize(np.array([qw, qx, qy, qz]))

def find_quaternion_v2(u, v):
    """ Finds a quaternion that rotates u to v

    Based on: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final

    """
    norm_u_norm_v = np.sqrt(np.dot(u, u) * np.dot(v, v))
    qw = norm_u_norm_v + np.dot(u, v)

    if qw < 1.0e-6 * norm_u_norm_v:
        qw = 0.0
        ux, uy, uz = u
        if np.abs(ux) > np.abs(uz):
            qx, qy, qz = np.array([-uy, ux, 0.0])
        else:
            qx, qy, qz = np.array([0.0, -uz, uy])
    else:
        qx, qy, qz = np.cross(u, v)

    return normalize(np.array([qw, qx, qy, qz]))

#@njit(parallel=True)
def find_quaternions(u, v):
    qq = np.zeros((u.shape[0], 4))

    for i in prange(u.shape[0]):
        qq[i] = find_quaternion(u[i, :], v[i, :])

    return qq

def rotate_vectors(v, q):

    import quaternionic as qn

    v = np.atleast_2d(v)

    v0 = np.zeros((v.shape[0], 4))
    v0[:, 1:] = v
    q0 = q * qn.array(v0) * q.inverse

    return q0.vector

@njit(parallel=False)
def bulk_rotate(u, R_x, R_y, R_z):
    v = np.zeros(u.shape)
    for i in prange(v.shape[0]):
        v[i, 0] = np.dot(R_x[i, :], u[i, :])
        v[i, 1] = np.dot(R_y[i, :], u[i, :])
        v[i, 2] = np.dot(R_z[i, :], u[i, :])
    return v

