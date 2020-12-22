from hermes.objects import Satellite, Earth, Constellation, SatGroup, SatPlane

import numpy as np
from astropy import time, units as u

# Based off [SAT-LOA-20161115-00117]
def _Audacy_00117():

    asat_1 = Satellite.circular(Earth.poli_body, 20270.4 * u.km - Earth.poli_body.R_mean, inc=25 * u.deg, raan=157.64 * u.deg,
                                arglat=0 * u.deg)
    asat_2 = Satellite.circular(Earth.poli_body, 20270.4 * u.km - Earth.poli_body.R_mean, inc=25 * u.deg, raan=217.64 * u.deg,
                                arglat=180 * u.deg)
    asat_3 = Satellite.circular(Earth.poli_body, 20270.4 * u.km - Earth.poli_body.R_mean, inc=25 * u.deg, raan=277.64 * u.deg,
                                arglat=0 * u.deg)
    constellation = Constellation()
    constellation.append(asat_1)
    constellation.append(asat_2)
    constellation.append(asat_3)
    constellation.set_color("#0074D9")
    constellation.set_fov(21.22 * u.deg)

    return constellation

Audacy_00117 = _Audacy_00117()