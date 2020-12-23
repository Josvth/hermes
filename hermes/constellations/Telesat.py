from hermes.objects import Satellite, Earth, Constellation, SatGroup, SatPlane

import numpy as np
from astropy import time, units as u

# Based off SAT-MPL-20200526-00053 and
from hermes.util import hex2rgb


def _Telesat_00053():

    J2017 = time.Time('J2017', scale='tt')

    # SAT-MPL-20200526-00053
    # Orbital plane 1 - 20 & 37 - 43 (polar orbits)
    # 27 planes (@ 1015 km, 99.0 degree, 13 sats/plane)
    # RAAN[0] = 0, delta_RAAN = 360 / 27 * 2
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 1.025 delta_NNU = 360 / 13
    # Index of NNNU[plane#][sat#]
    # Note: to let the code make more sense indexes match schedule planes as:
    # [37, 39, ..., 43, 2, 4, ..., 42, 1, 3, ..., 19]
    num_plane = 27  # number of planes
    num_sat = 13  # number of satellites per plane
    set_polar = SatGroup.as_set(Earth.poli_body,
                                a=Earth.poli_body.R_mean + 1015 * u.km, ecc=0 * u.one, inc=99.0 * u.deg,
                                rraan=np.mod(np.arange(0, num_plane) * 360.0 / num_plane * 2, 360.0) * u.deg,
                                aargp=np.repeat(0 * u.deg, num_plane),
                                nnnu=np.split(
                                    np.mod(np.tile(np.arange(0, num_sat) * 360.0 / num_sat, (num_plane, 1)) +
                                                np.tile(np.arange(0, num_plane) * 1.025, (num_sat, 1)).T, 360),
                                    num_plane) * u.deg,
                                epoch=J2017)
    set_polar.color = hex2rgb("#0074D9")  # Blue
    #set_polar.set_fov(44.85 * u.deg)

    # 40 planes (@ 1325 km, 50.88 degree, 33 sats/plane)
    # RAAN[0] = 0, delta_RAAN = 360 / 40
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 360 / 33 / 5 delta_NNU = 360 / 33  --- 0.0:1.09:13.1 with some weird ordering per plane
    # Index of NNNU[plane#][sat#]
    # Note: to let the code make more sense indexes match schedule planes as:
    num_plane = 40  # number of planes
    num_sat = 33  # number of satellites per plane
    set_inc = SatGroup.as_set(Earth.poli_body,
                              a=Earth.poli_body.R_mean + 1325 * u.km, ecc=0 * u.one, inc=50.88 * u.deg,
                              rraan=np.mod(np.arange(0, num_plane) * 360.0 / num_plane * 2, 360.0) * u.deg,
                              aargp=np.repeat(0 * u.deg, num_plane),
                              nnnu=np.split(
                                  np.mod(np.tile(np.arange(0, num_sat) * 360.0 / num_sat, (num_plane, 1)) +
                                         np.tile(np.array([0, 3, 1, 4, 2] * 8) * 2.18, (num_sat, 1)).T, 360),
                                  num_plane) * u.deg,
                              epoch=J2017)
    set_inc.color = hex2rgb("#FF4136")  # Blue
    #set_inc.set_fov(44.85 * u.deg)

    constellation = Constellation()
    constellation.append(set_polar)
    constellation.append(set_inc)

    return constellation

Telesat_00053 = _Telesat_00053()