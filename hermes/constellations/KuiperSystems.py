from hermes.objects import Satellite, Earth, Constellation, SatSet, SatPlane

import numpy as np
from astropy import time, units as u

# Based off SAT-LOA-20190704-00057
def _Kuiper_00057():
    J2020 = time.Time('J2020', scale='tt')

    # The excel sheet attached to the application provides the details of each sat.

    # 630 km 51.9 degree shell
    # RAAN[0] = 0, delta_RAAN = 360/34
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 360 / 34 / 34 * 3 delta_NNU = 360 / 34
    np_630 = 34  # number of planes
    ns_630 = 34  # number of satellites per plane
    set_630 = SatSet.as_set(Earth.poli_body,
                                a=Earth.poli_body.R_mean + 630 * u.km, ecc=0 * u.one, inc=51.9 * u.deg,
                                rraan=np.arange(0, np_630) * 360 / np_630 * u.deg,
                                aargp=np.repeat(0 * u.deg, np_630),
                                nnnu=np.split(np.mod(
                                    np.tile(np.arange(0, ns_630) * 360 / ns_630 * u.deg, (np_630, 1)) +
                                    np.tile(np.arange(0, np_630) * 360 / np_630 / ns_630 * 3 * u.deg, (ns_630, 1)).T, 360 * u.deg),
                                    np_630),
                                epoch=J2020)
    set_630.set_color("#0074D9")  # Blue
    set_630.set_fov(48.2 * u.deg)

    # 610 km 42 degree shell
    # RAAN[0] = 0, delta_RAAN = 360 / 36
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 360/36/36*3 delta_NNU = 360/36
    np_610 = 36  # number of planes
    ns_610 = 36  # number of satellites per plane
    set_610 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 610 * u.km, ecc=0 * u.one, inc=42.0 * u.deg,
                                 rraan=np.arange(0, np_610) * 360 / np_610 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_610),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_610) * 360 / ns_610 * u.deg, (np_610, 1)) +
                                     np.tile(np.arange(0, np_610) * 360 / ns_610 / np_610 * 3 * u.deg, (ns_610, 1)).T, 360 * u.deg),
                                     np_610),
                                 epoch=J2020)
    set_610.set_color("#FF851B")  # Orange
    set_610.set_fov(48.2 * u.deg)

    # 590 km 33 degree shell
    # RAAN[0] = 0, delta_RAAN = 360 / 28
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 360 / 28 / 28 delta_NNU = 360 / 28
    np_590 = 28  # number of planes
    ns_590 = 28  # number of satellites per plane
    set_590 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 590 * u.km, ecc=0 * u.one, inc=33.0 * u.deg,
                                 rraan=np.arange(0, np_590) * 360 / np_590 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_590),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_590) * 360 / ns_590 * u.deg, (np_590, 1)) +
                                     np.tile(np.arange(0, np_590) * 360 / ns_590 / np_590 * u.deg, (ns_590, 1)).T, 360 * u.deg),
                                     np_590),
                                 epoch=J2020)
    set_590.set_color("#2ECC40")  # Green
    set_590.set_fov(48.2 * u.deg)

    constellation = Constellation()
    constellation.append(set_630)
    constellation.append(set_610)
    constellation.append(set_590)

    return constellation

Kuiper_00057 = _Kuiper_00057()