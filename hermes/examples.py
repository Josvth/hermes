import os


def O3b_example():

    from hermes.simulation import Scenario
    from hermes.constellations.O3b_00154 import O3b
    from hermes.objects import Earth, Satellite
    from hermes.analysis import AccessAnalysis

    from astropy import time, units as u
    from mayavi import mlab

    J2019 = time.Time('J2019', scale='tt')

    start = time.Time('2019-09-01 10:00:00.000', scale='tt')        # Start time of simulation
    stop = time.Time('2019-09-08 10:00:00.000', scale='tt')         # Stop time of simulation
    step = 10 * u.s

    fig = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))   # Make a new figure (similar to MATLAB)

    # Start by making a scenario we will add our simulation objects to
    scenario = Scenario(Earth, start, stop, step, figure=fig)

    # Make a new Satellite object and give it a cyan color
    sat1 = Satellite.circular(Earth.poli_body, 500 * u.km, inc=90 * u.deg, raan=0 * u.deg, arglat=0 * u.deg)
    sat1.set_color('#00ffff')

    # And we add it to the scenario
    scenario.add_satellite(sat1)

    # Also add the constellation
    constellation = O3b
    scenario.add_satellite(constellation)

    # Add analysis
    an = AccessAnalysis(scenario, sat1, constellation)
    scenario.add_analysis(an)

    # Initizalize scenario
    scenario.initialize()

    # Start animation
    scenario.draw_scenario()
    scenario.step()  # do one step to let numba compile
    scenario.animate(scenario)
    mlab.show()


def Telesat_example():

    from hermes.simulation import Scenario
    from hermes.constellations.Telesat import Telesat_00053
    from hermes.objects import Earth, Satellite
    from hermes.analysis import AccessAnalysis

    from astropy import time, units as u
    from mayavi import mlab

    J2019 = time.Time('J2019', scale='tt')

    start = time.Time('2019-09-01 10:00:00.000', scale='tt')        # Start time of simulation
    stop = time.Time('2019-09-08 10:00:00.000', scale='tt')         # Stop time of simulation
    step = 10 * u.s

    fig = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))   # Make a new figure (similar to MATLAB)

    # Start by making a scenario we will add our simulation objects to
    scenario = Scenario(Earth, start, stop, step, figure=fig)

    # Make a new Satellite object and give it a cyan color
    sat1 = Satellite.circular(Earth.poli_body, 500 * u.km, inc=90 * u.deg, raan=0 * u.deg, arglat=0 * u.deg)
    sat1.set_color('#00ffff')

    # And we add it to the scenario
    scenario.add_satellite(sat1)

    # Also add the constellation
    constellation = Telesat_00053
    #scenario.add_satellite(constellation)

    # Add analysis
    #an = AccessAnalysis(scenario, sat1, constellation)
    #scenario.add_analysis(an)

    # Initizalize scenario
    scenario.initialize()

    # Start animation
    scenario.draw_scenario()
    #scenario.draw_frame()

    import numpy as np

    """Generates a pretty set of lines."""
    n_mer, n_long = 6, 11
    dphi = np.pi / 1000.0
    phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5) * 10000
    y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5) * 10000
    z = np.sin(n_long * mu / n_mer) * 0.5 * 10000

    l = mlab.plot3d(x, y, z, np.sin(mu), tube_radius=25, colormap='Spectral')

    scenario.step()  # do one step to let numba compile
    # from tvtk.tools import visual
    # visual.set_viewer(fig)

    # from pyface.timer.api import Timer

    #animator = scenario.simulate()
    #t = Timer(250, animator.next)

    for value in scenario.simulate(animate=False):
        pass

    scenario.animate()
    mlab.show()
    #import time
    #time.sleep(3)

    # import multiprocessing
    #
    # print('before')
    # process = multiprocessing.Process(target=mlab.show())
    # process.start()
    # print('after')
    #
    # def animation_worker():
    #
    # import time
    # t = time.time()
    # rate = 1.
    # for value in scenario.simulate(animate=True):
    #     wait_time = (t + rate) - time.time()
    #     if wait_time > 0:
    #         time.sleep(wait_time)
    #     t = time.time()
    #     fig.scene.render()


def eccentric_example():
    from hermes.simulation import Scenario
    from hermes.objects import Earth, Satellite

    from astropy import time, units as u
    from mayavi import mlab

    start = time.Time('2019-09-01 10:00:00.000', scale='tt')  # Start time of simulation
    stop = time.Time('2019-09-08 10:00:00.000', scale='tt')  # Stop time of simulation
    step = 10 * u.s

    fig = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)

    # Start by making a scenario we will add our simulation objects to
    scenario = Scenario(Earth, start, stop, step, figure=fig)

    # Make a new Satellite object and give it a cyan color
    sat1 = Satellite.from_classical(Earth, Earth.R_mean + 500 * u.km, inc=90 * u.deg, raan=0 * u.deg, arglat=0 * u.deg)
    sat1.set_color('#00ffff')

    # And we add it to the scenario
    scenario.add_satellite(sat1)

    # Initizalize scenario
    scenario.initialize()

    # Start animation
    scenario.draw_scenario()
    scenario.step()  # do one step to let numba compile
    scenario.animate(scenario)
    mlab.show()