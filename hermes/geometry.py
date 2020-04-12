import numpy as np
from numba import jit
from numpy.core.umath_tests import inner1d

from hermes.util import norm_along_rows

@jit
def point_inside_cone(r, ttip, ttheta, pphi = None):
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
def point_inside_cone_audacy(r, ttip, ttheta, pphi = None):
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
    ccone_radius = np.tan(ttheta) * ccone_dist      # outer radius of outer ring
    ccone_radius2 = np.tan(np.repeat(18.29 * np.pi / 180, len(ttheta))) * ccone_dist     # inner radius of outer ring
    ccone_radius3 = np.tan(np.repeat(16.55 * np.pi / 180, len(ttheta))) * ccone_dist     # outer radius of inner ring

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
    dp = norm_along_rows(diffp) # distances to sphere center

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
