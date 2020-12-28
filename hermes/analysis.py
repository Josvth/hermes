import csv
from builtins import property, object, enumerate, range

import numpy as np
import pandas as pd

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

    def initialise(self):
        if self.csv_name is not None:
            self._csv_file = open(self.csv_name, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            self._csv_writer.writerow(['id', 'TimeUTCG', 'tof',
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

    def __init__(self, scenario, obj_a, obj_b, audacy=False):

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

    def initialise(self):

        self.fovs = np.zeros((self._obj_b_slice.stop - self._obj_b_slice.start,))

        for i, s in enumerate(self._obj_b):
            self.fovs[i] = s.fov.to(u.rad).value

        self.r_a = self.get_positions(self._obj_a_slice)
        self.rr_b = self.get_positions(self._obj_b_slice)

        super(AccessAnalysis, self).initialise()

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


class LOSAnalysis(Analysis):

    # Settings
    check_block = True
    check_fov = True

    # State variables
    previous_los = []
    los = []

    # Visualisation variables
    show_los = True
    show_nolos = False

    # Storage
    contact_instances = []
    pass_counter = 0

    # Constants
    BUFFER_SIZE = 1000

    def __init__(self, scenario, obj_a, obj_b):

        self.scenario = scenario
        self.obj_a = obj_a
        self.obj_b = obj_b

        self.ffov = None    # Field-of-views [rad]
        self.R_body = scenario.state.attractor.poli_body.R_mean.to(u.m).value  # Radius of attractor [m]
        self.r_a = None   # This is a 'pointer' to the state vectors in the simulations SatGroup [m]
        self.rr_b = None  # This is a 'pointer' to the state vectors in the simulations SatGroup [m]

        # Build storage
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.store = pd.HDFStore("%s.h5" % timestamp)

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
            return slice(self.scenario.state.satellites.index(value), self.scenario.state.satellites.index(value) + 1)

        # If a group return indices of group elements
        i_group = self.scenario.state.satellites.index(value)
        return slice(i_group, i_group + len(value), 1)

    def get_positions(self, indices):
        return self.scenario.state.satellites.rr[indices]

    def initialise(self):

        # Reset storage
        self.contact_instances = []
        self.pass_counter = 0

        # Generate a list of FOVs
        self.ffov = np.zeros((self._obj_b_slice.stop - self._obj_b_slice.start,))
        for i, s in enumerate(self._obj_b):
            self.ffov[i] = s.fov.to(u.rad).value

        # Grab the pointers to the satellite positions
        self.r_a = self.get_positions(self._obj_a_slice)[0, :]
        self.rr_b = self.get_positions(self._obj_b_slice)

        # Find access at initial time point
        self.find_los()
        self.previous_los = [False] * len(self.los)

        # Store line-of-sights
        self.generate_instances()

        super().initialise()

    def find_los(self):

        # Check if line of sight is within fov and does not intersect
        if self.check_block:
            itsc = line_intersects_sphere(self.r_a, self.rr_b, self.scenario.state.attractor.xyz,
                                          self.R_body)
        else:
            itsc = np.ones(len(self.obj_b))

        if self.check_fov:
            insd = point_inside_cone(self.r_a, self.rr_b, self.ffov)
        else:
            insd = np.ones(len(self.obj_b))

        # Get satellites for which satellite b is in line of side and in front of the Earth
        self.los = insd * (1 - itsc) > 0

        return self.los

        # print(sum(los))

        # for i in [i for i, x in enumerate(self.los) if x]:
        #     strand_name = "%d to %d" % (self._obj_a_slice, self._obj_b_slice.start + i)
        #     ai = AccessInstant(self.r_a, self.rr_b[i], tof, strand_name)
        #
        #     # Append to current access list
        #     self.current_access_instants.append(ai)

    def generate_instances(self):

        for i in range(len(self.los)):

            started = not self.previous_los[i] and self.los[i]
            in_pass = self.los[i]
            ended = self.previous_los[i] and not self.los[i]

            if started:
                self.pass_counter = self.pass_counter + 1

            if in_pass:
                # Store instance
                strand_name = "sat-%d to sat-%d" % (self._obj_a_slice.start, self._obj_b_slice.start + i)

                tof_s = self.scenario.state.tof_s
                timestamp = str(self.scenario.state.time)

                r_a_x, r_a_y, r_a_z = self.r_a      # Decompose because its easier to append in Pandas/HDF5/CSV
                r_b_x, r_b_y, r_b_z = self.rr_b[i]  # Decompose because its easier to append in Pandas/HDF5/CSV

                # Todo store velocities

                contact_instance = {
                    'strand_name': strand_name,
                    'tof': tof_s,
                    'r_a_x': r_a_x, 'r_a_y': r_a_y, 'r_a_z': r_a_z,
                    'r_b_x': r_b_x, 'r_b_y': r_b_y, 'r_b_z': r_b_z,
                    'time': timestamp,
                }

                self.contact_instances.append(contact_instance)

                if len(self.contact_instances) >= self.BUFFER_SIZE:
                    self.store_instances()
                    self.contact_instances = []

            if ended:
                pass

    def store_instances(self):
        #print("Making data frame... (this might take some time)")
        df_contact_instances = pd.DataFrame(self.contact_instances)
        #print("Appending data frame to hdf5... (this might take some time)")
        self.store.append('contact_instances', df_contact_instances)

    def run(self, tof):
        """

        Parameters
        ----------
        time_delta : ~astropy.time.TimeDelta
        """

        # Find new accesses
        self.previous_los = self.los
        self.find_los()

        # Store line-of-sights
        self.generate_instances()

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
        self.store_instances()


        # print("Making data frame... (this might take some time)")
        # df_contact_instances = pd.DataFrame(self.contact_instances)
        #
        # from datetime import datetime
        # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #
        # print("Writing data frame to pickle... (this might take some time)")
        # df_contact_instances.to_pickle("%s.pickle" % timestamp)

        # for opportunity in self.current_opportunities:
        #     opportunity['stop_tof'].append(self.scenario.state.tof.to(u.s).value)
        #
        #     # Append passes to file
        #     self.opportunities.append(opportunity)
        #
        # opportunities_df = pd.DataFrame(self.opportunities)
        # opportunities_df.to_pickle('test.pickle')
