import hermes
from hermes.analysis import LOSAnalysis
from hermes.constellations.O3b import O3b_00154
from hermes.scenario import Scenario
from hermes.objects import Earth, Satellite, SatGroup

from astropy import time, units as u
from mayavi import mlab

from hermes.simulation import Simulation
from hermes.util import hex2rgb

##
start = time.Time('2019-09-01 10:00:00.000', scale='tt')  # Start time of simulation
stop = time.Time('2019-09-01 12:00:00.000', scale='tt')  # Stop time of simulation
step = 30 * u.s

scenario = Scenario(start, stop, step, hermes.objects.Earth, name='O3b_00154')

sat_500_97 = Satellite.circular(hermes.objects.Earth.poli_body, 500 * u.km, inc=97.5 * u.deg, raan=0 * u.deg,
                                arglat=0 * u.deg)
sat_500_97.color = hex2rgb('#00ffff')
sat_500_97.name = 'sat_500km_97deg'
# sat_500_97.J2_perturbation = True
sat_500_97.plane_3D_show = True

# And we add it to the scenario
scenario.add_satellite(sat_500_97)

# Add the constellation
constellation = O3b_00154
scenario.add_satellite(constellation)

# Add line-of-sight analysis
scenario.add_analysis(LOSAnalysis(scenario, sat_500_97, constellation))

# Initialise the scenario
scenario.initialise()

simulation = Simulation(scenario, show_3d=True)
vis = simulation.vis_3D
vis.visualise(next(vis.state_generator))
mlab.view(distance=70000.0)
fig = simulation.vis_3D.figure
fig.scene.movie_maker.record = True
simulation.run()



