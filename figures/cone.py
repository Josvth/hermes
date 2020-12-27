from hermes.analysis import LOSAnalysis
from hermes.scenario import Scenario
from hermes.objects import Earth, Satellite, SatGroup

from astropy import time, units as u
from mayavi import mlab
from tvtk.api import tvtk
from hermes.util import hex2rgb
from hermes.visualisation import Visualisation3DGIL, draw_vector, draw_satellite_fov

figures_dir = 'D:/git/thesis_report_ae/figures'

start = time.Time('2019-09-01 10:00:00.000', scale='tt')        # Start time of simulation
stop = time.Time('2019-09-02 10:00:00.000', scale='tt')         # Stop time of simulation
step = 10 * u.s

point = time.Time('2019-09-01 8:00:00.000', scale='tt')

scenario = Scenario(start, stop, step, Earth)

# Make a new Satellite object and give it a cyan color
sat1 = Satellite.circular(Earth.poli_body, 2000 * u.km, inc=45 * u.deg, raan=90 * u.deg, arglat=30 * u.deg)
sat1.color = hex2rgb('#00ffff')
sat1.fov_3D_show = True

sat2 = Satellite.circular(Earth.poli_body, 2000 * u.km, inc=165 * u.deg, raan=30 * u.deg, arglat=30 * u.deg)
sat2.color = hex2rgb('#B10DC9')
sat2.fov_3D_show = True

# And we add it to the scenario
scenario.add_satellite(sat1)
scenario.add_satellite(sat2)

scenario.initialize()

vis = Visualisation3DGIL(scenario)

mlab.view(45, 60)

vis.figure.scene.movie_maker.record = True
vis.run()