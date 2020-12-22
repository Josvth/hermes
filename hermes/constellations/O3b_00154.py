from hermes.objects import Satellite, Earth, Constellation, SatGroup, SatPlane

import numpy as np
from astropy import time, units as u

# Based off SAT-AMD-2017110900154
def _O3b_00154():
    inc_plane1 = SatPlane.as_plane(
        Earth.poli_body,
        a=Earth.poli_body.R_mean + 8062 * u.km, ecc=0 * u.one, inc=70 * u.deg,
        raan=0 * u.deg,
        argp=0 * u.deg,
        nnu=np.array([3.5, 75.5, 147.5, 219.5, 291.5]) * u.deg
    )
    inc_plane1.set_color("#0074D9")
    equ_plane = SatPlane.as_plane(
        Earth.poli_body,
        a=Earth.poli_body.R_mean + 8062 * u.km, ecc=0 * u.one, inc=0 * u.deg,
        raan=0 * u.deg,
        argp=0 * u.deg,
        nnu=np.linspace(0, 360, 32) * u.deg
    )
    equ_plane.set_color("#FF4136")
    equ_plane.set_fov(51.53 * u.deg)

    inc_plane2 = SatPlane.as_plane(
        Earth.poli_body,
        a=Earth.poli_body.R_mean + 8062 * u.km, ecc=0 * u.one, inc=70 * u.deg,
        raan=180 * u.deg,
        argp=0 * u.deg,
        nnu=np.array([183.5, 225.5, 327.5, 111.5, 39.5]) * u.deg
    )
    inc_plane2.set_color("#0074D9")

    constellation = Constellation()
    constellation.append(inc_plane1)
    constellation.append(equ_plane)
    constellation.append(inc_plane2)

    return constellation

O3b_00154 = _O3b_00154()


