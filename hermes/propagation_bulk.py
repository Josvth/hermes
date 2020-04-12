from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from numba import jit, njit

from astropy import time, units as u

from poliastro.core.angles import nu_to_M, _kepler_equation, _kepler_equation_prime, E_to_nu, nu_to_F, F_to_M, nu_to_E, \
    E_to_M

from poliastro.core.elements import coe2rv, rv2coe

import numpy as np


@jit
def markley_bulk(k, tof, pp, eecc, iinc, rraan, aargp, nnu0):
    """ Solves the kepler problem by a non iterative method. Relative error is
    around 1e-18, only limited by machine double-precission errors.

    Parameters
    ----------
    k : float
        Standar Gravitational parameter
    r0 : array
        Initial position vector wrt attractor center.
    v0 : array
        Initial velocity vector.
    tof : float
        Time of flight.

    Returns
    -------
    rf: array
        Final position vector
    vf: array
        Final velocity vector

    Note
    ----
    The following algorithm was taken from http://dx.doi.org/10.1007/BF00691917.
    """

    # Solve first for eccentricity and mean anomaly
    # p, ecc, inc, raan, argp, nu = rv2coe(k, r0, v0)

    # check of all orbits are elipses
    # if eecc.max() >= 1:
    # raise ValueError("All eccentricities must be smaller than 1")

    EE = nu_to_E(nnu0, eecc)
    MM0 = E_to_M(EE, eecc)

    aa = pp / (1 - eecc ** 2)
    nn = np.sqrt(k / aa ** 3)
    MM = MM0 + nn * tof

    # Range between -pi and pi
    MM = (MM + np.pi) % (2 * np.pi) - np.pi
    # MM = MM % (2 * np.pi)
    # for i, MM_elmt in enumerate(MM):
    #     if MM_elmt > np.pi:
    #         MM[i] = -(2 * np.pi - MM_elmt)

    # Equation (20)
    aalpha = (3 * np.pi ** 2 + 1.6 * (np.pi - np.abs(MM)) / (1 + eecc)) / (np.pi ** 2 - 6)

    # Equation (5)
    dd = 3 * (1 - eecc) + aalpha * eecc

    # Equation (9)
    qq = 2 * aalpha * dd * (1 - eecc) - MM ** 2

    # Equation (10)
    rr = 3 * aalpha * dd * (dd - 1 + eecc) * MM + MM ** 3

    # Equation (14)
    ww = (np.abs(rr) + np.sqrt(qq ** 3 + rr ** 2)) ** (2 / 3)

    # Equation (15)
    EE = (2 * rr * ww / (ww ** 2 + ww * qq + qq ** 2) + MM) / dd

    # Equation (26)
    f0 = _kepler_equation(EE, MM, eecc)
    f1 = _kepler_equation_prime(EE, MM, eecc)
    f2 = eecc * np.sin(EE)
    f3 = eecc * np.cos(EE)
    f4 = -f2

    # Equation (22)
    delta3 = -f0 / (f1 - 0.5 * f0 * f2 / f1)
    delta4 = -f0 / (f1 + 0.5 * delta3 * f2 + 1 / 6 * delta3 ** 2 * f3)
    delta5 = -f0 / (
            f1 + 0.5 * delta4 * f2 + 1 / 6 * delta4 ** 2 * f3 + 1 / 24 * delta4 ** 3 * f4
    )

    EE += delta5
    nnu = E_to_nu(EE, eecc)

    return nnu
