from astropy import time, units as u
from mayavi import mlab

from hermes.analysis import LOSAnalysis
from hermes.objects import Earth, Satellite, SatGroup
from hermes.scenario import Scenario
from hermes.util import hex2rgb
from hermes.visualisation import Visualisation3DGIL, Visualisation3DGILChristmas

import numpy as np

start = time.Time('2019-09-01 10:00:00.000', scale='tt')
stop = time.Time('2019-09-01 12:00:00.000', scale='tt')
step = 10 * u.s

point = time.Time('2019-09-01 8:00:00.000', scale='tt')

scenario = Scenario(start, stop, step, Earth)

r_p = Earth.poli_body.R_mean + 500 * u.km
r_a = Earth.poli_body.R_mean + 300000 * u.km

a = 0.5 * (r_p + r_a)
ecc = (r_a - r_p) / (r_a + r_p)

# Stars
# Make a new Satellite object and give it a cyan color
star1 = Satellite.from_classical(Earth.poli_body, a, ecc, inc=50 * u.deg, raan=270 * u.deg, argp=270 * u.deg,
                                 nu=250 * u.deg)
star1.color = hex2rgb('#F8B229')

star2 = Satellite.from_classical(Earth.poli_body, a, ecc, inc=45 * u.deg, raan=120 * u.deg, argp=0 * u.deg,
                                 nu=0 * u.deg)
star2.color = hex2rgb('#F8B229')

star3 = Satellite.from_classical(Earth.poli_body, a, ecc, inc=45 * u.deg, raan=240 * u.deg, argp=0 * u.deg,
                                 nu=0 * u.deg)
star3.color = hex2rgb('#F8B229')

# Red and Green
n_planes = 20  # number of planes
n_sats = 40  # number of satellites per plane
red = SatGroup.as_set(Earth.poli_body,
                      a=Earth.poli_body.R_mean + 2000 * u.km, ecc=0 * u.one, inc=60 * u.deg,
                      rraan=np.arange(0, n_planes) * (360./n_planes * 2) * u.deg,
                      aargp=np.repeat(0 * u.deg, n_planes),
                      nnnu=np.split(np.mod(
                          np.tile(np.arange(0, n_sats) * 9 * u.deg, (n_planes, 1)) +
                          np.tile(np.arange(0, n_planes) * 4.5 * u.deg, (n_sats, 1)).T, 360 * u.deg),
                          n_planes)
                      )
red.color = hex2rgb("#EA4630")

green = SatGroup.as_set(Earth.poli_body,
                        a=Earth.poli_body.R_mean + 2000 * u.km, ecc=0 * u.one, inc=60 * u.deg,
                        rraan=np.arange(0, n_planes) * (360./n_planes * 2) * u.deg,
                        aargp=np.repeat(0 * u.deg, n_planes),
                        nnnu=np.split(np.mod(
                            np.tile(np.arange(0, n_sats) * 9 * u.deg, (n_planes, 1)) +
                            np.tile(np.arange(0, n_planes) * 4.5 * u.deg, (n_sats, 1)).T, 360 * u.deg) + 180 / n_sats * u.deg,
                            n_planes)
                        )
green.color = hex2rgb("#146B3A")

#
# sat2 = Satellite.from_classical(Earth.poli_body, a, ecc, inc=165 * u.deg, raan=30 * u.deg, argp=30 * u.deg,
#                                 nu=0 * u.deg)
# sat2.color = hex2rgb('#B10DC9')
#
# sat3 = Satellite.circular(Earth.poli_body, 2500 * u.km, inc=60 * u.deg, raan=10 * u.deg, arglat=70 * u.deg)
# sat3.color = hex2rgb('#FFDC00')
#
# sat4 = Satellite.circular(Earth.poli_body, 3000 * u.km, inc=45 * u.deg, raan=110 * u.deg, arglat=210 * u.deg)
# sat4.color = hex2rgb('#001f3f')
#
# sat5 = Satellite.circular(Earth.poli_body, 3500 * u.km, inc=110 * u.deg, raan=70 * u.deg, arglat=0 * u.deg)
# sat5.color = hex2rgb('#FF851B')  # 0074D9
#
# group = SatGroup()
# group.append((sat2, sat3, sat4, sat5))

# And we add it to the scenario
scenario.add_satellite(star1)
# scenario.add_satellite(star2)
# scenario.add_satellite(star3)
scenario.add_satellite(red)
scenario.add_satellite(green)

# Initialize
scenario.initialize()

#scenario.step_to(point)

figure = mlab.figure(size=(1200, 1200), bgcolor=hex2rgb('#2E3336'))
vis = Visualisation3DGILChristmas(scenario, figure=figure)

vis.visualise()
mlab.view(azimuth=90, elevation=90, distance=6*Earth.poli_body.R_mean.to(u.km).value, focalpoint=(0,0,0))
figure.scene.movie_maker.record = True
vis.run()
#vis.show()
