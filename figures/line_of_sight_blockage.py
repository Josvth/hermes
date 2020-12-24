from astropy import time, units as u

from hermes.analysis import LOSAnalysis
from hermes.objects import Earth, Satellite, SatGroup
from hermes.scenario import Scenario
from hermes.util import hex2rgb
from hermes.visualisation import Visualisation3DGIL

start = time.Time('2019-09-01 10:00:00.000', scale='tt')
stop = time.Time('2019-09-07 10:00:00.000', scale='tt')
step = 10 * u.s

point = time.Time('2019-09-01 8:00:00.000', scale='tt')

scenario = Scenario(start, stop, step, Earth)

# Make a new Satellite object and give it a cyan color
sat1 = Satellite.circular(Earth.poli_body, 2000 * u.km, inc=45 * u.deg, raan=90 * u.deg, arglat=30 * u.deg)
sat1.color = hex2rgb('#00ffff')

sat2 = Satellite.circular(Earth.poli_body, 2000 * u.km, inc=165 * u.deg, raan=30 * u.deg, arglat=30 * u.deg)
sat2.color = hex2rgb('#B10DC9')

sat3 = Satellite.circular(Earth.poli_body, 2500 * u.km, inc=60 * u.deg, raan=10 * u.deg, arglat=70 * u.deg)
sat3.color = hex2rgb('#FFDC00')

sat4 = Satellite.circular(Earth.poli_body, 3000 * u.km, inc=45 * u.deg, raan=110 * u.deg, arglat=210 * u.deg)
sat4.color = hex2rgb('#001f3f')

sat5 = Satellite.circular(Earth.poli_body, 3500 * u.km, inc=110 * u.deg, raan=70 * u.deg, arglat=0 * u.deg)
sat5.color = hex2rgb('#FF851B') #0074D9

group = SatGroup()
group.append((sat2, sat3, sat4, sat5))

# And we add it to the scenario
scenario.add_satellite(sat1)
scenario.add_satellite(group)

# Add line-of-sight analysis
los_analysis = LOSAnalysis(scenario, sat1, group)
los_analysis.check_fov = False

scenario.add_analysis(los_analysis)

# Initialize
scenario.initialize()

scenario.step_to(point)

vis = Visualisation3DGIL(scenario)
vis.show()