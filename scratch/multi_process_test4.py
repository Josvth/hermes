import multiprocessing as mp
from mayavi import mlab
import numpy as np
from astropy import time, units as u

from hermes.objects import Earth, Satellite
from hermes.scenario import Scenario
from hermes.util import hex2rgb
from hermes.visualisation import Visualisation3D, Visualisation3DGIL


def simulation_worker(sec, queue):

    queue.put(sec.initialize())

    for state in sec.run():
        queue.put(state)
        #print("SIM!")

    print("SIM DONE!")


def visualize_worker(vis3D):
    an = vis3D.animate()
    mlab.show()

def mlab_worker():
    mlab.show()
    while True:
        pass

if __name__ == '__main__':

    start = time.Time('2019-09-01 10:00:00.000', scale='tt')
    stop = time.Time('2019-09-07 10:00:00.000', scale='tt')
    step = 10 * u.s

    scenario = Scenario(start, stop, step, Earth)

    # Make a new Satellite object and give it a cyan color
    sat1 = Satellite.circular(Earth.poli_body, 500 * u.km, inc=90 * u.deg, raan=0 * u.deg, arglat=0 * u.deg)
    sat1.color = hex2rgb('#00ffff')

    # Also add the constellation
    from hermes.constellations.Telesat import Telesat_00053
    constellation = Telesat_00053
    scenario.add_satellite(constellation)

    # And we add it to the scenario
    scenario.add_satellite(sat1)

    vis = Visualisation3DGIL(scenario)
    vis.run()

    pass

