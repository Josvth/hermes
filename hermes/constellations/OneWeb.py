from hermes.objects import Satellite, Earth, Constellation, SatGroup, SatPlane

import numpy as np
from astropy import time, units as u

from hermes.util import wrap, hex2rgb

# Based off SAT-LOI-20160428-00041
def _OneWeb_00041():

    # OneWeb initial LEO constellation
    # SAT-LOI-20160428-00041
    # Orbital planes 1 to 18
    # 1200 km 87.9 degree
    # RAAN[0] = 0, delta_RAAN = 10.2
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 4.5 delta_NNU = 9
    n_planes = 18  # number of planes
    n_sats = 40  # number of satellites per plane
    set = SatGroup.as_set(Earth.poli_body,
                          a=Earth.poli_body.R_mean + 1200 * u.km, ecc=0 * u.one, inc=87.9 * u.deg,
                          rraan=np.arange(0, n_planes) * 10.2 * u.deg,
                          aargp=np.repeat(0 * u.deg, n_planes),
                          nnnu=np.split(wrap(np.mod(
                                     np.tile(np.arange(0, n_sats) * 9 * u.deg, (n_planes, 1)) +
                                     np.tile(np.arange(0, n_planes) * 4.5 * u.deg, (n_sats, 1)).T, 360 * u.deg)),
                                     n_planes)
                          )
    set.color = hex2rgb("#0074D9")  # Blue
    set.fov = 28.86 * u.deg

    constellation = Constellation()
    constellation.append(set)

    return constellation


OneWeb_00041 = _OneWeb_00041()