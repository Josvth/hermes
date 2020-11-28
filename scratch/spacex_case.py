from hermes.simulation import Scenario
from hermes.constellations.O3b_00154 import O3b
from hermes.objects import Earth, Satellite, SatSet, Constellation
from hermes.analysis import AccessAnalysis
import numpy as np
from astropy import time, units as u
from mayavi import mlab

J2019 = time.Time('J2019', scale='tt')

start = time.Time('2019-09-01 10:00:00.000', scale='tt')        # Start time of simulation
stop = time.Time('2019-09-01 11:00:00.000', scale='tt')         # Stop time of simulation
step = 5 * u.s

fig = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))   # Make a new figure (similar to MATLAB)

# Start by making a scenario we will add our simulation objects to
scenario = Scenario(Earth, start, stop, step, figure=fig)

# Also add the constellation
n_plane = 72  # number of planes
n_sat = 20  # number of satellites per plane

# assuming 11/20 phase offset
set = SatSet.as_set(Earth.poli_body,
                            a=Earth.poli_body.R_mean + 550 * u.km, ecc=0 * u.one, inc=53.0 * u.deg,
                            rraan=np.arange(0, n_plane) * 360 / n_plane * u.deg,
                            aargp=np.repeat(0 * u.deg, n_plane),
                            nnnu=np.split(np.mod(
                                np.tile(np.arange(0, n_sat) * 360 / n_sat * u.deg, (n_plane, 1)) +
                                np.tile(np.arange(0, n_plane) * (0) * u.deg, (n_sat, 1)).T, 360 * u.deg),
                                n_plane),
                            epoch=J2019)
set.set_color("#0074D9")    # Blue

constellation = Constellation()
constellation.append(set)

#constellation.set_fov(44.85 * u.deg)

scenario.add_satellite(constellation)

# Initizalize scenario
scenario.initialize()

# Start animation
scenario.draw_scenario()
scenario.step()  # do one step to let numba compile
fig.scene.movie_maker.record = True
scenario.animate(scenario)
mlab.show()
