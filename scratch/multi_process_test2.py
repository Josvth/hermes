import multiprocessing as mp
from mayavi import mlab
import numpy as np
from astropy import time, units as u

from hermes.objects import Earth, Satellite
from hermes.scenario import Scenario
from hermes.visualisation import Visualisation3D


def simulation_worker(sec, queue):

    queue.put(sec.initialise())

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
    sat1.color = '#00ffff'

    # Also add the constellation
    from hermes.constellations.Telesat import Telesat_00053
    constellation = Telesat_00053
    scenario.add_satellite(constellation)

    # And we add it to the scenario
    scenario.add_satellite(sat1)

    import time as ptime

    q = mp.Queue(100000)
    #simulation_process = mp.Process(target=simulation_worker, args=(scenario,q,), daemon=True)
    #ptime.sleep(10)
    #simulation_process.deamon = True
    #simulation_process.join()

    visualiser = Visualisation3D(q)
    visualise_process = mp.Process(target=visualize_worker, args=(visualiser,), daemon=True)
    #ptime.sleep(10)
    visualise_process.start()
    ptime.sleep(2)
    print("Starting in:")
    t = 20
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        ptime.sleep(1)
        t -= 1

    from timeit import default_timer as timer

    q.put(scenario.initialise())
    t_sim_step = timer()
    for state in scenario.run():
        t_sim_step = timer()
        #q.put(state)
        print("t=%2.1f s (%5.3f ms)" % (0, (timer() - t_sim_step) * 1000))
        #print("SIM!")

    #simulation_process.start()
    #visualise_process.deamon = True
    #simulation_process.join()
    visualise_process.join()
    pass

