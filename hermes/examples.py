def O3b_example():

    from hermes.simulation import Scenario
    from hermes.constellations.O3b_constellation import O3b
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