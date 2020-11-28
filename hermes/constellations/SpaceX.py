from hermes.objects import Satellite, Earth, Constellation, SatSet, SatPlane

import numpy as np
from astropy import time, units as u

# Based off SAT-MOD-20190830-00087
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
                                 a=Earth.poli_body.R_mean + 1325 * u.km, ecc=0 * u.one, inc=70.0 * u.deg,
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


# Based off SAT-MOD-20200417-00037
def _SpaceX_00037():

    J2015 = time.Time('J2015', scale='tt')

    # Shell 1
    # 550 km 53 degree
    # RAAN[0] = 0, delta_RAAN = 5
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 8.8 delta_NNU = 360 / 22
    np_550_530 = 72  # number of planes
    ns_550_530 = 22  # number of satellites per plane
    set_550_530 = SatSet.as_set(Earth.poli_body,
                                a=Earth.poli_body.R_mean + 550 * u.km, ecc=0 * u.one, inc=53.0 * u.deg,
                                rraan=np.arange(0, np_550_530) * 5 * u.deg,
                                aargp=np.repeat(0 * u.deg, np_550_530),
                                nnnu=np.split(np.mod(
                                    np.tile(np.arange(0, ns_550_530) * 360 / 22 * u.deg, (np_550_530, 1)) +
                                    np.tile(np.arange(0, np_550_530) * 8.8 * u.deg, (ns_550_530, 1)).T, 360 * u.deg),
                                    np_550_530),
                                epoch=J2015)
    set_550_530.set_color("#0074D9")  # Blue
    set_550_530.set_fov(44.85 * u.deg)

    # Shell 2
    # 540 km 53.2 degree
    # RAAN[0] = 2.5, delta_RAAN = 5
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 12.35 delta_NNU = 360 / 22
    np_540_532 = 72  # number of planes
    ns_540_532 = 22  # number of satellites per plane
    set_540_532 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 540 * u.km, ecc=0 * u.one, inc=53.2 * u.deg,
                                 rraan=np.arange(0, np_540_532) * 5 * u.deg + 2.5 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_540_532),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_540_532) * 360 / 22 * u.deg, (np_540_532, 1)) +
                                     np.tile(np.arange(0, np_540_532) * 12.35 * u.deg, (ns_540_532, 1)).T, 360 * u.deg),
                                     np_540_532),
                                 epoch=J2015)
    set_540_532.set_color("#FF851B")  # Orange
    set_540_532.set_fov(40.72 * u.deg)

    # Shell 3
    # 570 km 70 degree
    # RAAN[0] = 0, delta_RAAN = 10
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 7.65 delta_NNU = 360 / 20
    np_570_70 = 36  # number of planes
    ns_570_70 = 20  # number of satellites per plane
    set_570_70 = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 570.0 * u.km, ecc=0 * u.one, inc=70.0 * u.deg,
                                 rraan=np.arange(0, np_570_70) * 10 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_570_70),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_570_70) * 360 / 20 * u.deg, (np_570_70, 1)) +
                                     np.tile(np.arange(0, np_570_70) * 7.65 * u.deg, (ns_570_70, 1)).T, 360 * u.deg),
                                     np_570_70),
                                 epoch=J2015)
    set_570_70.set_color("#2ECC40")  # Green
    set_570_70.set_fov(40.59 * u.deg)

    # Shell 4
    # 560 km 97.6 degree
    # RAAN[0] = 63.7, delta_RAAN = 60
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 0.8 delta_NNU = 4.8
    np_560_976a = 6  # number of planes
    ns_560_976a = 58  # number of satellites per plane
    set_560_976a = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 560.0 * u.km, ecc=0 * u.one, inc=97.6 * u.deg,
                                 rraan=np.arange(0, np_560_976a) * 60 * u.deg + 63.7 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_560_976a),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_560_976a) * 360 / ns_560_976a * u.deg, (np_560_976a, 1)) +
                                     np.tile(np.arange(0, np_560_976a) * 1.04 * u.deg, (ns_560_976a, 1)).T, 360 * u.deg),
                                     np_560_976a),
                                 epoch=J2015)
    set_560_976a.set_color("#B10DC9")  # Purple
    set_560_976a.set_fov(97.6 * u.deg)

    # Shell 5
    # 560 km 97.6 degree
    # RAAN[0] = 75.7, delta_RAAN = 12.0
    # ARGP[:] = 0
    # NNNU[0][0] = 0 delta_NNNU = 0.275 delta_NNU = 360 / 43
    np_560_976b = 4  # number of planes
    ns_560_976b = 43  # number of satellites per plane
    set_560_976b = SatSet.as_set(Earth.poli_body,
                                 a=Earth.poli_body.R_mean + 560.0 * u.km, ecc=0 * u.one, inc=97.6 * u.deg,
                                 rraan=np.arange(0, np_560_976b) * 12.0 * u.deg + 75.7 * u.deg,
                                 aargp=np.repeat(0 * u.deg, np_560_976b),
                                 nnnu=np.split(np.mod(
                                     np.tile(np.arange(0, ns_560_976b) * 360 / ns_560_976b * u.deg, (np_560_976b, 1)) +
                                     np.tile(np.arange(0, np_560_976b) * 0.275 * u.deg, (ns_560_976b, 1)).T, 360 * u.deg),
                                     np_560_976b),
                                 epoch=J2015)
    set_560_976b.set_color("#FF4136")  # Red
    set_560_976b.set_fov(97.6 * u.deg)

    constellation = Constellation()
    constellation.append(set_550_530)
    constellation.append(set_540_532)
    constellation.append(set_570_70)
    constellation.append(set_560_976a)
    constellation.append(set_560_976b)

    return constellation


SpaceX_00037 = _SpaceX_00037()