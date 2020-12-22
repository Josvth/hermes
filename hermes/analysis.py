import csv
from builtins import property, object, enumerate, range

import numpy as np

from astropy import time, units as u

from mayavi import mlab

from abc import ABC, abstractmethod

from hermes.geometry import line_intersects_sphere, point_inside_cone, point_inside_cone_audacy
from hermes.objects import Satellite

class Analysis(ABC):

    def __init__(self):
        self.csv_name = None
        self._csv_file = None
        self._csv_writer = None

    def initialize(self):
        if self.csv_name is not None:
            self._csv_file = open(self.csv_name, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            self._csv_writer.writerow(['id','TimeUTCG', 'tof',
                                       'xyz_a_x', 'xyz_a_y', 'xyz_a_z',
                                       'xyz_b_x', 'xyz_b_y', 'xyz_b_z',
                                       'StrandName'])

    @abstractmethod
    def run(self, t):
        pass

    @abstractmethod
    def draw_update(self):
        pass

    @abstractmethod
    def stop(self):
        pass

class AccessInstant(object):

    def __init__(self, obj_a_snap, obj_b_snap, time_delta, strand_name):
        """

        Parameters
        ----------

        """
        # Uncomment to also store xyz (takes a lot of memory!)
        self.obj_a_snap = obj_a_snap
        self.obj_b_snap = obj_b_snap
        self.tof = time_delta
        self.strand_name = strand_name

    def to_csv_row(self, epoch):
        return ['not_implemented', epoch + self.tof, self.tof.to(u.s).value,
                self.obj_a_snap[0], self.obj_a_snap[1], self.obj_a_snap[2],
                self.obj_b_snap[0], self.obj_b_snap[1], self.obj_b_snap[2],
                self.strand_name]


class AccessAnalysis(Analysis):

    def __init__(self, scenario, obj_a, obj_b, audacy = False):

        self.scenario = scenario
        self.obj_a = obj_a
        self.obj_b = obj_b

        self.fovs = None
        self.r_a = None
        self.rr_b = None
        self.audacy = audacy

        self.current_access_instants = []

        super().__init__()

    @property
    def obj_a(self):
        """First simulation object"""
        return self._obj_a

    @property
    def obj_b(self):
        """Second simulation object"""
        return self._obj_b

    @obj_a.setter
    def obj_a(self, value):
        if value.__len__() != 1:
            raise ValueError("Object a should have only one position.")
        self._obj_a_slice = self.get_slice(value)
        self._obj_a = value

    @obj_b.setter
    def obj_b(self, value):
        self._obj_b_slice = self.get_slice(value)
        self._obj_b = value

    def get_slice(self, value):

        # ToDo this probably breaks when using a satellite inside a group
        # If satellite return index of satellite
        if isinstance(value, Satellite):
            return self.scenario.sat_group.index(value)

        # If a group return indices of group elements
        i_group = self.scenario.sat_group.index(value)
        return slice(i_group, i_group + len(value), 1)

    def get_positions(self, indices):
        return self.scenario.sat_group.rr[indices]

    def initialize(self):

        self.fovs = np.zeros((self._obj_b_slice.stop - self._obj_b_slice.start,))

        for i, s in enumerate(self._obj_b):
            self.fovs[i] = s.fov.to(u.rad).value

        self.r_a = self.get_positions(self._obj_a_slice)
        self.rr_b = self.get_positions(self._obj_b_slice)

        super(AccessAnalysis, self).initialize()

    def find_accesses(self, tof):

        # Check if line of sight is within fov and does not intersect
        itsc = line_intersects_sphere(self.r_a, self.rr_b, self.scenario.body.xyz,
                                      self.scenario.body.poli_body.R_mean.to(u.m).value)
        if self.audacy:
            insd = point_inside_cone_audacy(self.r_a, self.rr_b, self.fovs)
        else:
            insd = point_inside_cone(self.r_a, self.rr_b, self.fovs)

        # Get satellites for which satellite b is in line of side and in front of the Earth
        los = insd * (1 - itsc) > 0

        # print(sum(los))

        for i in [i for i, x in enumerate(los) if x]:
            strand_name = "%d to %d" % (self._obj_a_slice, self._obj_b_slice.start + i)
            ai = AccessInstant(self.r_a, self.rr_b[i], tof, strand_name)

            # Append to current access list
            self.current_access_instants.append(ai)

    def run(self, tof):
        """

        Parameters
        ----------
        time_delta : ~astropy.time.TimeDelta
        """

        # Clear all accesses we had in this window
        self.current_access_instants = []

        # Find new accesses and append to our list of accesses
        self.find_accesses(tof)

        # Export to CSV
        if self._csv_file is not None:
            for ai in self.current_access_instants:
                self._csv_writer.writerow(ai.to_csv_row(self.scenario.epoch))

    def draw(self, figure):

        color = "#01FF70"[1:]
        color = tuple(float(int(color[i:i + 2], 16)) / 255.0 for i in (0, 2, 4))

        x = np.array([0, 0])
        y = np.array([0, 0])
        z = np.array([0, 0])

        self.mlab_points = mlab.plot3d(x, y, z, tube_radius=25., color=color, figure=figure)

    def draw_update(self):

        x = np.array([0, 0])
        y = np.array([0, 0])
        z = np.array([0, 0])

        if len(self.current_access_instants) > 0:
            x[0] = self.current_access_instants[0].obj_a_snap[0] / 1000
            y[0] = self.current_access_instants[0].obj_a_snap[1] / 1000
            z[0] = self.current_access_instants[0].obj_a_snap[2] / 1000

            x[1] = self.current_access_instants[0].obj_b_snap[0] / 1000
            y[1] = self.current_access_instants[0].obj_b_snap[1] / 1000
            z[1] = self.current_access_instants[0].obj_b_snap[2] / 1000

        self.mlab_points.mlab_source.trait_set(x=x, y=y, z=z)

    def stop(self):
        self._csv_file.close()
