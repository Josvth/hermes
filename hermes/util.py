import numpy as np
from astropy import time, units as u
from numba import jit, njit, prange
from numpy import cos, sin, multiply, square, divide

def wrap(angle):
    """ Wraps an angle to (-pi, pi) or (-180, 180)"""
    angle_rad = angle.to(u.rad).value
    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
    return (angle_rad * u.rad).to(angle.unit)

def hex2rgb(color):
    color = color[1:]   # Strip hex sybol #
    return tuple(float(int(color[i:i + 2], 16)) / 255.0 for i in (0, 2, 4))

def fast_get_positions(sats, ff, aarg_pe, rraan, ii, aa, ee):
    # Iterate to set up variables
    for i, s in enumerate(sats):
        # Slow #TODO speed up by improving 'true_anomaly_from_eccentric' function
        ff[i] = s.f

    ## Bulk calculate U vectors
    uu = aarg_pe + ff
    sin_uu = sin(uu)
    cos_uu = cos(uu)
    sin_rraan = sin(rraan)
    cos_rraan = cos(rraan)
    sin_ii = sin(ii)
    cos_ii = cos(ii)

    sin_uu_cos_i = multiply(sin_uu, cos_ii)

    UU = np.array(
        [multiply(cos_uu, cos_rraan) - multiply(sin_uu_cos_i, sin_rraan),
         multiply(cos_uu, sin_rraan) + multiply(sin_uu_cos_i, cos_rraan),
         multiply(sin_uu, sin_ii)])

    UU = np.squeeze(np.swapaxes(UU, 0, 1))

    ## Bulk calculate orbital radii
    cos_ff = cos(ff)

    rradius = divide(multiply(aa, 1 - square(ee)), 1 + multiply(ee, cos_ff))

    return multiply(UU.T, rradius.T).T


def generate_time_vector(epoch, value):
    """Generates a TimeDelta time vector.

    Parameters
    ----------
    epoch : ~astropy.time.Time
        Epoch of simulation
    value : ~astropy.units.Quantity, ~astropy.time.Time, ~astropy.time.TimeDelta
        Scalar time to propagate.

    Returns
    -------
    Orbit : ~astropy.time.TimeDelta
        Time vector

    """

    if isinstance(value, time.Time) and not isinstance(value, time.TimeDelta):
        time_of_flight = value - epoch
    else:
        # Works for both Quantity and TimeDelta objects
        time_of_flight = time.TimeDelta(value)

    # Use the highest precision we can afford
    # np.atleast_1d does not work directly on TimeDelta objects
    jd1 = np.atleast_1d(time_of_flight.jd1)
    jd2 = np.atleast_1d(time_of_flight.jd2)
    return time.TimeDelta(jd1, jd2, format="jd", scale=time_of_flight.scale)

@jit
def calc_lmn(iinc, rraan, aargp):

    cos_aargp = cos(aargp)
    cos_rraan = cos(rraan)
    sin_aargp = sin(aargp)
    sin_rraan = sin(rraan)
    cos_iinc = cos(iinc)
    sin_iinc = sin(iinc)

    ll1 = cos_aargp * cos_rraan - sin_aargp * sin_rraan * cos_iinc     # Eq. 11.32
    mm1 = cos_aargp * sin_rraan + sin_aargp * cos_rraan * cos_iinc     # Eq. 11.32
    nn1 = sin_aargp * sin_iinc                                            # Eq. 11.32
    ll2 = -sin_aargp * cos_rraan - cos_aargp * sin_rraan * cos_iinc    # Eq. 11.32
    mm2 = -sin_aargp * sin_rraan + cos_aargp * cos_rraan * cos_iinc    # Eq. 11.32
    nn2 = cos_aargp * sin_iinc                                            # Eq. 11.32

    return ll1, mm1, nn1, ll2, mm2, nn2


@jit
def coe2xyz_fast(xyz, pp, eecc, ll1, mm1, nn1, ll2, mm2, nn2, nnu):

    cos_nnu = cos(nnu)
    sin_nnu = sin(nnu)

    rr = pp / (1 + eecc*cos_nnu)   # Eq. 11.33

    xyz[:, 0] = rr * (ll1 * cos_nnu + ll2 * sin_nnu)
    xyz[:, 1] = rr * (mm1 * cos_nnu + mm2 * sin_nnu)
    xyz[:, 2] = rr * (nn1 * cos_nnu + nn2 * sin_nnu)

@njit
def norm_along_rows(x):
    return np.sum(np.abs(x)**2,axis=-1)**(1./2)


#@njit(parallel=True)
def row_cross(va1, va2):
    """Computes the cross product of each row in a Nx3 vector array"""

    va = np.zeros(va1.shape)
    for i in prange(va1.shape[0]):
        va[i,:] = np.cross(va1[i,:], va2[i,:])

    return va


def generate_time_vector(start, stop, delta):

    """Generates time vectors for simulation

    Parameters
    ----------
    start : ~astropy.time.Time
        Start time
    stop : ~astropy.time.Time
        End time.
    """

    # Create time vector
    dt = stop - start

    t = np.arange(0, dt.to(u.s).value, delta.to(u.s).value)
    tof_vector = time.TimeDelta(t * u.s)

    # Use the highest precision we can afford
    # np.atleast_1d does not work directly on TimeDelta objects
    jd1 = np.atleast_1d(tof_vector.jd1)
    jd2 = np.atleast_1d(tof_vector.jd2)

    return time.TimeDelta(jd1, jd2, format="jd", scale=tof_vector.scale)
