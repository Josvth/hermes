
from hermes.analysis import LOSAnalysis
from hermes.scenario import Scenario
from hermes.objects import Earth, Satellite, SatGroup

from astropy import time, units as u
from mayavi import mlab
from tvtk.api import tvtk

from hermes.simulation import Simulation
from hermes.util import hex2rgb
from hermes.visualisation import Visualisation3DGIL

#mlab.init_notebook('ipy')

# PDF exporter defaults
ex = tvtk.GL2PSExporter()
ex.file_format = 'pdf'
ex.sort = 'bsp'
ex.compress = 1
#ex.edit_traits(kind='livemodal')

figures_dir = 'D:/git/thesis_report_ae/figures'

start = time.Time('2019-09-01 10:00:00.000', scale='tt')        # Start time of simulation
stop = time.Time('2019-09-07 10:00:00.000', scale='tt')         # Stop time of simulation
step = 1 * u.s

scenario = Scenario(start, stop, step, Earth)

# Make a new Satellite object and give it a cyan color
sat1 = Satellite.circular(Earth.poli_body, 550 * u.km, inc=40 * u.deg, raan=90 * u.deg, arglat=100 * u.deg)
sat1.color = hex2rgb('#00ffff')

sat2 = Satellite.circular(Earth.poli_body, 2000 * u.km, inc=75 * u.deg, raan=30 * u.deg, arglat=135 * u.deg)
sat2.color = hex2rgb('#ffff00')
sat2.fov_3D_show = True
sat2.fov = 30 * u.deg

# And we add it to the scenario
scenario.add_satellite(sat1)
scenario.add_satellite(sat2)

# Add line-of-sight analysis
los_analysis = LOSAnalysis(scenario, sat1, sat2)
los_analysis.show_nolos = True

scenario.add_analysis(los_analysis)

# Initialise the scenario
scenario.initialise()

simulation = Simulation(scenario, False)
simulation.run()

#vis = Visualisation3DGIL(scenario)

#vis.run()

#scenario.save()



