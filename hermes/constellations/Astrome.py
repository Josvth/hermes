from hermes.objects import Satellite, Earth, Constellation, SatGroup, SatPlane

import numpy as np
from astropy import time, units as u

from hermes.util import wrap, hex2rgb

# Based off https://astrome.io/wp-content/uploads/2019/07/Astrome-YellowPaper.pdf
def _Astrome():

    n_planes = 11
    n_sats = 18
    set = SatGroup.as_set(Earth.poli_body,
                          a=Earth.poli_body.R_mean + 1530 * u.km, ecc=0 * u.one, inc=30.0 * u.deg,
                          rraan=np.arange(0, n_planes) * 30 * u.deg,
                          aargp=np.repeat(0 * u.deg, n_planes),
                          nnnu=np.split(np.mod(
                            np.tile(np.linspace(0, 360, n_sats) * u.deg, (n_planes, 1)) +
                            np.tile(np.arange(0, n_planes) * 1.818 * u.deg, (n_sats, 1)).T, 360 * u.deg),
                            n_planes))
    set.color = hex2rgb("#0074D9")
    set.fov = 34.763 * u.deg

    constellation = Constellation()
    constellation.append(set)

    return constellation

Astrome = _Astrome()