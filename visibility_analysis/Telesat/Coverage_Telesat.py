
from hermes.analysis import LOSAnalysis, CoverageAnalysis
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

# Constellation
constellation = Telesat_00053
scenario.add_satellite(constellation)

# Add line-of-sight analysis
#analysis = CoverageAnalysis(scenario, constellation, name='COVAnalysis_1300km', altitude=1300 * u.km)
#analysis = CoverageAnalysis(scenario, constellation, name='COVAnalysis_1000km', altitude=1000 * u.km)
analysis = CoverageAnalysis(scenario, constellation, name='COVAnalysis_400km', altitude=400 * u.km)
scenario.add_analysis(analysis)

# Initialise the scenario
scenario.initialise()
writer = analysis.generate_writer()
writer.initialise()
writer.flush()


