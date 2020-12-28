from astropy import time, units as u

from hermes.objects import SatGroup
from hermes.util import generate_time_vector

class Scenario(object):


    class State(object):

        def __init__(self, start, stop, step, attractor):
            self.start = start
            self.stop = stop
            self.step = step
            self.time = start
            self.tof = time.TimeDelta(0 * u.s)
            self.attractor = attractor
            self.satellites = SatGroup()
            self.analyses = []

    def __init__(self, start, stop, step, attractor):
        # Scenario state
        self.state = Scenario.State(start, stop, step, attractor)

    def add_satellite(self, satellite):
        self.state.satellites.append(satellite)

    def add_analysis(self, analysis):
        self.state.analyses.append(analysis)

    ##
    def propagate_to(self, t):
        """Propagates the simulation to time t

        Parameters
        ----------
        t: ~astropy.time.Time
            Time to propagate to.
        """

        dt = t - self.state.start

        tof = dt.to(u.s).value

        # Propagates attractor
        self.state.attractor.propagate_to(tof)

        # Propagate all satellite objects
        self.state.satellites.propagate_to(tof)

        # Update tof
        self.state.time = t
        self.state.tof = dt

    def analyze(self):
        """Runs the analysis for the current state"""
        for i, an in enumerate(self.state.analyses):
            an.run(self.state)

    ## Simulation
    def initialise(self):

        print("Initializing attractor")
        self.state.attractor.initialise()

        print("Initializing %d satellites..." % len(self.state.satellites))
        self.state.satellites.initialize()

        for i, an in enumerate(self.state.analyses):
            print("Initializing analysis %d of %d..." % (i + 1, len(self.state.analyses)))
            an.initialise()

        return self.state

    def run(self):
        while self.state.time <= self.state.stop:
            yield self.step()

    def step(self):
        """Steps the simulation to the next time step"""
        return self.step_to(self.state.time + self.state.step)

    def step_to(self, t):
        """Steps the simulation to an arbitrary point in time

        Parameters
        ----------
        t: ~astropy.time.Time
            Time to propagate to.
        """
        self.propagate_to(t)
        self.analyze()
        return self.state

    def stop(self):
        pass

    def save(self):
        """Runs the analysis for the current state"""
        for i, an in enumerate(self.state.analyses):
            an.stop()