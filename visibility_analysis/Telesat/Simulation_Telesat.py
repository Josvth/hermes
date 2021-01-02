
from hermes.analysis import LOSAnalysis
from hermes.constellations.Telesat import Telesat_00053
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

scenario = Scenario(start, stop, step, Earth, name='Test')

# Make a new Satellite object and give it a cyan color
sat_400_51 = Satellite.circular(Earth.poli_body, 400 * u.km, inc=51.6 * u.deg, raan=0 * u.deg, arglat=0 * u.deg)
sat_400_51.color = hex2rgb('#00ffff')
sat_400_51.name = 'sat_400km_51deg'
#sat_400_51.J2_perturbation = True

sat_500_97 = Satellite.circular(Earth.poli_body, 500 * u.km, inc=97.5 * u.deg, raan=0 * u.deg, arglat=0 * u.deg)
sat_500_97.color = hex2rgb('#00ffff')
sat_500_97.name = 'sat_500km_97deg'
#sat_500_97.J2_perturbation = True

# And we add it to the scenario
scenario.add_satellite(sat_400_51)
scenario.add_satellite(sat_500_97)

# scenario.add_satellite(sat3)

# Constellation
constellation = Telesat_00053
scenario.add_satellite(constellation)

# Add line-of-sight analysis
scenario.add_analysis(LOSAnalysis(scenario, sat_400_51, constellation))
scenario.add_analysis(LOSAnalysis(scenario, sat_500_97, constellation))
#scenario.add_analysis(LOSAnalysis(scenario, sat_300_45, constellation))
#scenario.add_analysis(LOSAnalysis(scenario, sat_500_45, constellation))


# Initialise the scenario
scenario.initialise()

simulation = Simulation(scenario, show_3d=False)
simulation.run()



