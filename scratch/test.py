from hermes.simulation import Scenario
from hermes.objects import Earth, Satellite

from astropy import time, units as u
from mayavi import mlab

#mlab.init_notebook('ipy')

#%%

from hermes.constellations.Telesat import Telesat_00053

start = time.Time('2019-09-01 10:00:00.000', scale='tt')        # Start time of simulation
stop = time.Time('2019-09-01 11:00:00.000', scale='tt')         # Stop time of simulation
step = 1 * u.s

fig = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))   # Make a new figure (similar to MATLAB)

# Start by making a scenario we will add our simulation objects to
scenario = Scenario(Earth, start, stop, step, figure=fig)

# Also add the constellation
constellation = Telesat_00053
scenario.add_satellite(constellation)

# Initizalize scenario
scenario.initialize()

# Start animation
scenario.draw_scenario()

point = time.Time('2019-09-01 10:14:00.000', scale='tt')
scenario.step_to(point, True)

# scenario.draw_frame()
scenario.step(True)  # do one step to let numba compile
mlab.show()