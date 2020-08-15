from hermes.objects import Satellite, Earth, Constellation, SatSet, SatPlane

import numpy as np
from astropy import time, units as u

# Based off SAT-MOD-20190830-00087 and
def _SpaceX_00087():

    J2015 = time.Time('J2015', scale='tt')

    # SAT-MOD-20190830-00087
    # Orbital plane 1
    # 550 km 53 degree
    # RAAN[0] = 0, delta_RAAN = 5
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 0 ?? delta_NNU = 360 / 72
    np_550_530 = 72  # number of planes
    ns_550_530 = 22  # number of satellites per plane
    set_550_530 = SatSet.as_set(Earth.poli_body,
                                a=Earth.poli_body.R_mean + 550 * u.km, ecc=0 * u.one, inc=53.0 * u.deg,
                                rraan=np.arange(0, np_550_530) * 5 * u.deg,
                                aargp=np.repeat(0 * u.deg, np_550_530),
                                nnnu=np.split(np.mod(
                                    np.tile(np.arange(0, ns_550_530) * 360 / ns_550_530 * u.deg, (np_550_530, 1)) +
                                    np.tile(np.arange(0, np_550_530) * 0 * u.deg, (ns_550_530, 1)).T, 360 * u.deg),
                                    np_550_530),
                                epoch=J2015)
    set_550_530.set_color("#0074D9")  # Blue
    set_550_530.set_fov(44.85 * u.deg)

    # Sat-LOA-20161115-00118
    # Orbital planes 33 to 64
    # 1110 km 53.8 degree
    # RAAN[0] = 5.6, delta_RAAN = 11.25
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 3.375 delta_NNU = 7.2
    np_1110_538 = 32  # number of planes
    ns_1110_538 = 50  # number of satellites per plane
    set_1110_538 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 1110 * u.km, ecc=0 * u.one, inc=53.8 * u.deg,
                                 rraan=np.arange(5.6, 360, 11.25) * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_1110_538
                                                 ),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, 360, 7.2) * u.deg, (np_1110_538, 1)) +
                                     np.tile(np.arange(0, 105, 3.375) * u.deg, (ns_1110_538, 1)).T, 360 * u.deg),
                                     np_1110_538),
                                 epoch=J2015)
    set_1110_538.set_color("#FF851B")  # Orange
    set_1110_538.set_fov(40.72 * u.deg)

    # Sat-LOA-20161115-00118
    # Orbital planes 65 to 72
    # 1130 km 74 degree
    # RAAN[0] = 0, delta_RAAN = 45
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 7.2 delta_NNU = 7.2
    np_1130_740 = 8  # number of planes
    ns_1130_740 = 50  # number of satellites per plane
    set_1130_740 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 1130 * u.km, ecc=0 * u.one, inc=74.0 * u.deg,
                                 rraan=np.arange(0, 360, 45) * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_1130_740),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, 360, 7.2) * u.deg, (np_1130_740, 1)) +
                                     np.tile(np.arange(0, 55, 7.2) * u.deg, (ns_1130_740, 1)).T, 360 * u.deg),
                                     np_1130_740),
                                 epoch=J2015)
    set_1130_740.set_color("#2ECC40")  # Green
    set_1130_740.set_fov(40.59 * u.deg)

    # Sat-LOA-20161115-00118
    # Orbital planes 73 to 78
    # 1325 km 70 degree
    # RAAN[0] = 0, delta_RAAN = 60
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 0.8 delta_NNU = 4.8
    np_1325_700 = 6  # number of planes
    ns_1325_700 = 75  # number of satellites per plane
    set_1325_700 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean+ 1325 * u.km, ecc=0 * u.one, inc=70.0 * u.deg,
                                 rraan=np.arange(0, np_1325_700) * 60 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_1325_700),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_1325_700) * 4.8 * u.deg, (np_1325_700, 1)) +
                                     np.tile(np.arange(0, np_1325_700) * 0.8 * u.deg, (ns_1325_700, 1)).T, 360 * u.deg),
                                     np_1325_700),
                                 epoch=J2015)
    set_1325_700.set_color("#B10DC9")  # Purple
    set_1325_700.set_fov(39.67 * u.deg)

    # Sat-LOA-20161115-00118
    # Orbital planes 79 to 83
    # 1275 km 81 degree
    # RAAN[0] = 0, delta_RAAN = 72.0
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 7.2 delta_NNU = 7.2
    np_1275_810 = 5  # number of planes
    ns_1275_810 = 75  # number of satellites per plane
    set_1275_810 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 1275 * u.km, ecc=0 * u.one, inc=81.0 * u.deg,
                                 rraan=np.arange(0, np_1275_810) * 72.0 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_1275_810),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_1275_810) * 4.8 * u.deg, (np_1275_810, 1)) +
                                     np.tile(np.arange(0, np_1275_810) * 0.8 * u.deg, (ns_1275_810, 1)).T, 360 * u.deg),
                                     np_1275_810),
                                 epoch=J2015)
    set_1275_810.set_color("#FF4136")  # Red
    set_1275_810.set_fov(39.36 * u.deg)

    constellation = Constellation()
    constellation.append(set_550_530)
    constellation.append(set_1110_538)
    constellation.append(set_1130_740)
    constellation.append(set_1325_700)
    constellation.append(set_1275_810)

    return constellation

SpaceX_00087 = _SpaceX_00087()