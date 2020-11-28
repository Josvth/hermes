import sys
import traceback
from builtins import object, enumerate
from timeit import default_timer as timer

import astropy
from mayavi import mlab

from hermes.objects import SatGroup

import numpy as np
from astropy import units as u

class Scenario(object):

    def __init__(self, body, t_start, t_stop, t_step, epoch=None, figure=None):

        self.body = body

        self.t_start = t_start
        self.t_stop = t_stop
        self.t_step = t_step

        if epoch is None:
            self.epoch = t_start

        self._generate_time_vector(self.t_start, self.t_stop, self.t_step)
        self.tof_current = self._tof_vector[0]

        self.figure = figure

        self.objects = [body]
        self.sat_group = SatGroup()
        self.analyses = []

    def draw_scenario(self):
        # mlab.figure(self.figure)
        for ob in self.objects:
            ob.draw(self.figure)
        for an in self.analyses:
            an.draw(self.figure)
        self.sat_group.draw(self.figure)

    def add_object(self, sc_object):
        self.objects.append(sc_object)

    def add_satellite(self, satellite):
        self.sat_group.append(satellite)

    def add_analysis(self, analysis):
        return self.analyses.append(analysis)

    def step(self, draw_update=False):
        self.propagate_to(self.tof_current, draw_update)
        self.analyze(self.tof_current, draw_update)

    def initialize(self):

        for i, ob in enumerate(self.objects):
            print("Initializing object %d of %d..." % (i + 1, len(self.objects)))
            ob.initialize()

        print("Initializing %d satellites..." % len(self.sat_group))
        self.sat_group.initialize()

        for i, an in enumerate(self.analyses):
            print("Initializing analysis %d of %d..." % (i + 1, len(self.analyses)))
            an.initialize()

    def propagate_to(self, t, draw_update=False):
        """Propagates the simulation to time t

        Parameters
        ----------
        draw_update : bool
            True if updates should be drawn (only use when animating)
        t: ~astropy.time.TimeDelta
            Time to propagate to.
        """

        # Propagates all non satellite objects (now only the attractor... which has no propagation...)
        for i, ob in enumerate(self.objects):
            # print("Propagating object %d of %d" % (i + 1, len(self.objects)))
            ob.propagate_to(t)
            if draw_update:
                ob.draw_update(self.figure)

        # Propagate all satellite objects
        self.sat_group.propagate_to(t)
        if draw_update:
            self.sat_group.draw(self.figure)

    def analyze(self, t, draw_update=False):
        for i, an in enumerate(self.analyses):
            an.run(t)
            if draw_update:
                an.draw_update()

    def simulate(self, t_start=None, t_stop=None, t_step=None, animate=False, follow=None):

        if t_start is not None: self.t_start = t_start
        if t_stop is not None: self.t_stop = t_stop
        if t_step is not None: self.t_step = t_step

        self._generate_time_vector(self.t_start, self.t_stop, self.t_step)

        t_sim_start = timer()  # simulation start time

        for i, tof in enumerate(self._tof_vector):
            t_sim_step = timer()

            self.figure.scene.disable_render = True
            self.tof_current = tof
            self.step(animate)
            self.figure.scene.disable_render = False
            print("t=%2.1f s (%5.3f ms)" % (self.t[i], (timer() - t_sim_step) * 1000))

            # if follow is not None:
            #     mlab.view(azimuth=np.mod(
            #         -20 + np.arctan2(follow._xyz[1].to(u.km).value, follow._xyz[0].to(u.km).value) * 180 / np.pi, 360))

            yield
            # if animate:
            #     yield

        print("Total simulation time: %5.5f s" % (timer() - t_sim_start))

    @mlab.animate(delay=20, ui=True)
    def animate(self, follow=None, t_start=None, t_stop=None, t_delta=None):
        try:
            yield from self.simulate(t_start, t_stop, t_delta, animate=True, follow=follow)
        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)

    def animate_notebook(self, follow=None, t_start=None, t_stop=None, t_delta=None):
        import time
        t = time.time()
        rate = 0.02
        for value in self.simulate(t_start, t_stop, t_delta, animate=True, follow=follow):
            wait_time = (t + rate) - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            t = time.time()

    def _generate_time_vector(self, start, stop, delta):

        """Generates time vectors for simulation

        Parameters
        ----------
        start : ~astropy.time.Time
            Start time
        stop : ~astropy.time.Time
            End time.
        """

        # Create time vector
        dt = stop - start

        self.t = np.arange(0, dt.to(u.s).value, delta.to(u.s).value)
        self._tof_vector = astropy.time.TimeDelta(self.t * u.s)

        # Use the highest precision we can afford
        # np.atleast_1d does not work directly on TimeDelta objects
        jd1 = np.atleast_1d(self._tof_vector.jd1)
        jd2 = np.atleast_1d(self._tof_vector.jd2)
        self._tof_vector = astropy.time.TimeDelta(jd1, jd2, format="jd", scale=self._tof_vector.scale)

    def draw_frame(self):

        mag = 9500 * u.km
        xyz = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        uvw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * mag

        mlab.quiver3d(xyz[0, :], xyz[1, :], xyz[2, :], uvw[0, :], uvw[1, :], uvw[2, :], reset_zoom=False,
                      mode='2darrow', scale_mode='vector', scale_factor=1)

        pass

    def step_to(self, time, draw_update):
        """Quick method to step to a point in time

        Parameters
        ----------
        time : ~astropy.time.Time
            Time to step to.
        """

        tof = time - self.t_start
        self.tof_current = tof
        self.step(draw_update)

    def stop(self):

        for an in self.analyses:
            an.stop()
