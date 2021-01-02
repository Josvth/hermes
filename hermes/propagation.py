from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from numba import jit, njit
from numpy.core.umath import cos, sin, sqrt

from astropy import time, units as u

from poliastro.core.angles import _kepler_equation, _kepler_equation_prime, E_to_nu, nu_to_E, E_to_M

import numpy as np


@njit
def secular_rates_J2(k, p, ecc, inc, raan, argp, nu, J2, Rbody):
    """ Returns the first order secular rates of change dΩ/dt, dω/dt and dθ/dt by J2 perturbation """

    a = p / (1 - ecc ** 2)
    n = np.sqrt(k / a ** 3)

    # From Howard, Curtis (eq 10.93b-d) or AE4874 (eq. 23.35) and AE4874 (eq. 23.27)
    draan = -3 / 2 * J2 * n * (Rbody / p) ** 2 * np.cos(inc)
    dargp = 3 / 2 * J2 * (n / 2) * (Rbody / p) ** 2 * (5 * cos(inc) ** 2 - 1)
    dnu = n * (1 + 3 / 4 * J2 * (Rbody / a) ** 2 * (1 - ecc**2) ** (-3/2) * (3 * cos(inc) ** 2 - 1))

    return draan, dargp, dnu


@njit
def markley_coe(k, p, ecc, inc, raan, argp, nu, tof):
    M0 = E_to_M(nu_to_E(nu, ecc), ecc)
    a = p / (1 - ecc ** 2)
    n = np.sqrt(k / a ** 3)
    M = M0 + n * tof

    # Range between -pi and pi
    M = (M + np.pi) % (2 * np.pi) - np.pi

    # Equation (20)
    alpha = (3 * np.pi ** 2 + 1.6 * (np.pi - np.abs(M)) / (1 + ecc)) / (np.pi ** 2 - 6)

    # Equation (5)
    d = 3 * (1 - ecc) + alpha * ecc

    # Equation (9)
    q = 2 * alpha * d * (1 - ecc) - M ** 2

    # Equation (10)
    r = 3 * alpha * d * (d - 1 + ecc) * M + M ** 3

    # Equation (14)
    w = (np.abs(r) + np.sqrt(q ** 3 + r ** 2)) ** (2 / 3)

    # Equation (15)
    E = (2 * r * w / (w ** 2 + w * q + q ** 2) + M) / d

    # Equation (26)
    f0 = _kepler_equation(E, M, ecc)
    f1 = _kepler_equation_prime(E, M, ecc)
    f2 = ecc * np.sin(E)
    f3 = ecc * np.cos(E)
    f4 = -f2

    # Equation (22)
    delta3 = -f0 / (f1 - 0.5 * f0 * f2 / f1)
    delta4 = -f0 / (f1 + 0.5 * delta3 * f2 + 1 / 6 * delta3 ** 2 * f3)
    delta5 = -f0 / (
            f1 + 0.5 * delta4 * f2 + 1 / 6 * delta4 ** 2 * f3 + 1 / 24 * delta4 ** 3 * f4
    )

    E += delta5
    nu = E_to_nu(E, ecc)

    return nu


@jit
def rv_pqw(k, p, ecc, nu):
    r"""Returns r and v vectors in perifocal frame.

    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    nu: float
        True anomaly (rad).

    Returns
    -------

    r: ndarray
        Position. Dimension 3 vector
    v: ndarray
        Velocity. Dimension 3 vector

    Notes
    -----
    These formulas can be checked at Curtis 3rd. Edition, page 110. Also the
    example proposed is 2.11 of Curtis 3rd Edition book.

    .. math::

        \vec{r} = \frac{h^2}{\mu}\frac{1}{1 + e\cos(\theta)}\begin{bmatrix}
        \cos(\theta)\\
        \sin(\theta)\\
        0
        \end{bmatrix} \\\\\\

        \vec{v} = \frac{h^2}{\mu}\begin{bmatrix}
        -\sin(\theta)\\
        e+\cos(\theta)\\
        0
        \end{bmatrix}

    Examples
    --------
    >>> from poliastro.constants import GM_earth
    >>> k = GM_earth.value  # Earth gravitational parameter
    >>> ecc = 0.3  # Eccentricity
    >>> h = 60000e6  # Angular momentum of the orbit (m**2 / s)
    >>> nu = np.deg2rad(120)  # True Anomaly (rad)
    >>> p = h**2 / k  # Parameter of the orbit
    >>> r, v = rv_pqw(k, p, ecc, nu)
    >>> # Printing the results
    r = [[-5312706.25105345  9201877.15251336    0]] [m]
    v = [[-5753.30180931 -1328.66813933  0]] [m]/[s]

    Also works on vector inputs:
    >>> r, v = rv_pqw(np.full(4,k), np.full(4,p), np.full(4,ecc), np.full(4,nu))
    r = [[-5312706.25105345  9201877.15251336    0]
         [-5312706.25105345  9201877.15251336    0]
         [-5312706.25105345  9201877.15251336    0]
         [-5312706.25105345  9201877.15251336    0]] [m]
    v = [[-5753.30180931 -1328.66813933  0]
         [-5753.30180931 -1328.66813933  0]
         [-5753.30180931 -1328.66813933  0]
         [-5753.30180931 -1328.66813933  0]] [m]/[s]
    """

    # Unit vectors in pqw (this is done to circumvent a numba issue: https://github.com/numba/numba/issues/4470)
    u_p = np.array([[1, 0, 0]]).T
    u_q = np.array([[0, 1, 0]]).T
    u_w = np.array([[0, 0, 1]]).T

    r_pf = (p / (1 + ecc * cos(nu)))  # length of r-vector (N,)
    r_p = r_pf * cos(nu)
    r_w = r_pf * sin(nu)
    r_q = np.zeros_like(nu)
    r = (u_p * r_p + u_q * r_w + u_w * r_q).T

    v_pf = sqrt(k / p)  # length of v-vector (N,)
    v_p = v_pf * -sin(nu)
    v_w = v_pf * (ecc + cos(nu))
    v_q = np.zeros_like(nu)
    v = (u_p * v_p + u_q * v_w + u_w * v_q).T

    return r, v


@njit
def pqw_to_ijk_vectors(inc, raan, argp):
    cos_argp = cos(argp)
    cos_raan = cos(raan)
    sin_argp = sin(argp)
    sin_raan = sin(raan)
    cos_inc = cos(inc)
    sin_inc = sin(inc)

    pi = cos_argp * cos_raan - sin_argp * sin_raan * cos_inc  # Eq. 11.32
    pj = cos_argp * sin_raan + sin_argp * cos_raan * cos_inc  # Eq. 11.32
    pk = sin_argp * sin_inc  # Eq. 11.32
    qi = -sin_argp * cos_raan - cos_argp * sin_raan * cos_inc  # Eq. 11.32
    qj = -sin_argp * sin_raan + cos_argp * cos_raan * cos_inc  # Eq. 11.32
    qk = cos_argp * sin_inc  # Eq. 11.32
    wi = sin_raan * sin_inc
    wj = -cos_raan * sin_inc
    wk = cos_inc

    return pi, pj, pk, qi, qj, qk, wi, wj, wk


@njit
def pqw_to_eci(r, v, k, p, ecc, nu, pi, pj, pk, qi, qj, qk, wi, wj, wk):
    # Position vectors
    r_pqw_norm = p / (1 + ecc * cos(nu))
    r_p = r_pqw_norm * cos(nu)
    r_q = r_pqw_norm * sin(nu)
    r_w = 0

    r_x = r_p * pi + r_q * qi + r_w * wi
    r_y = r_p * pj + r_q * qj + r_w * wj
    r_z = r_p * pk + r_q * qk + r_w * wk

    r[:, 0] = r_x
    r[:, 1] = r_y
    r[:, 2] = r_z

    # Velocity vectors
    v_pqw_norm = np.sqrt(k / p)
    v_p = v_pqw_norm * -sin(nu)
    v_q = v_pqw_norm * (ecc + cos(nu))
    v_w = 0

    v_x = v_p * pi + v_q * qi + v_w * wi
    v_y = v_p * pj + v_q * qj + v_w * wj
    v_z = v_p * pk + v_q * qk + v_w * wk

    v[:, 0] = v_x
    v[:, 1] = v_y
    v[:, 2] = v_z
