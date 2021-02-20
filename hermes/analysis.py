import csv
import os
from builtins import property, object, enumerate, range

import numpy as np
import pandas as pd

from astropy import time, units as u

from mayavi import mlab

from abc import ABC, abstractmethod

from hermes.geometry import line_intersects_sphere, point_inside_cone, point_inside_cone_audacy, spherical_to_cartesian
from hermes.objects import Satellite


class Analysis(ABC):

    name = 'Unnamed-Analysis'

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
    pass_number = []
    instance_counters = []  # Counts the number of instances (n) in a pass for each of the pairs

    # Visualisation variables
    show_los = True
    show_nolos = False

    # Storage
    contact_instances = {}  # Dict having key (p, n) and the contact instance
    pass_counter = 0  # Counts the number of passes (p)

    # Constants
    BUFFER_SIZE = 1000

    # HACKS
    audacy = False

    def __init__(self, scenario, obj_a, obj_b, name='LOSAnalysis'):

        self.scenario = scenario
        self.obj_a = obj_a
        self.obj_b = obj_b

        self.ffov = None  # Field-of-views [rad]
        self.R_body = scenario.state.attractor.poli_body.R_mean.to(u.m).value  # Radius of attractor [m]
        self.r_a = None  # This is a 'pointer' to the state vectors in the simulations SatGroup [m]
        self.v_a = None  # This is a 'pointer' to the state vectors in the simulations SatGroup [m]
        self.rr_b = None  # This is a 'pointer' to the state vectors in the simulations SatGroup [m]
        self.vv_b = None  # This is a 'pointer' to the state vectors in the simulations SatGroup [m]

        self.name = name

        super().__init__()

    def _generate_name(self):
        return "LOSAnalysis_%s_to_%s" % (self.obj_a.prefixed_name(), self.obj_b.prefixed_name())

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

    def get_velocities(self, indices):
        return self.scenario.state.satellites.vv[indices]

    def initialise(self):

        # Generate storage file
        self.name = self._generate_name()

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.store = pd.HDFStore("%s_%s_%s.h5" % (timestamp, self.scenario.name, self.name))

        # Reset storage
        self.contact_instances = {}
        self.pass_counter = 0

        # Generate a list of FOVs
        self.ffov = np.zeros((self._obj_b_slice.stop - self._obj_b_slice.start,))
        for i, s in enumerate(self._obj_b):
            self.ffov[i] = s.fov.to(u.rad).value

        # Grab the pointers to the satellite positions
        self.r_a = self.get_positions(self._obj_a_slice)[0, :]
        self.v_a = self.get_velocities(self._obj_a_slice)[0, :]
        self.rr_b = self.get_positions(self._obj_b_slice)
        self.vv_b = self.get_velocities(self._obj_b_slice)

        # Find access at initial time point
        self.find_los()
        self.previous_los = [False] * len(self.los)
        self.pass_number = [-1] * (self._obj_b_slice.stop - self._obj_b_slice.start)
        self.instance_counters = [0] * (self._obj_b_slice.stop - self._obj_b_slice.start)

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
            if self.audacy:
                insd = point_inside_cone_audacy(self.r_a, self.rr_b, self.ffov)
            else:
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
                self.pass_counter = self.pass_counter + 1  # Increment pass counter
                self.pass_number[i] = self.pass_counter  # Set the number of the pass counter of this pair
                self.instance_counters[i] = 0  # Set the instance counter for this pair to zero

            if in_pass:

                # Store instance
                strand_name = "%s to %s" % (self._obj_a_slice.start, self._obj_b_slice.start + i)

                tof_s = self.scenario.state.tof_s
                timestamp = str(self.scenario.state.time)

                r_a_x, r_a_y, r_a_z = self.r_a      # Decompose because its easier to append in Pandas/HDF5/CSV
                v_a_x, v_a_y, v_a_z = self.v_a      # Decompose because its easier to append in Pandas/HDF5/CSV
                r_b_x, r_b_y, r_b_z = self.rr_b[i]  # Decompose because its easier to append in Pandas/HDF5/CSV
                v_b_x, v_b_y, v_b_z = self.vv_b[i]  # Decompose because its easier to append in Pandas/HDF5/CSV

                # Todo store velocities
                contact_instance = {
                    'strand_name': strand_name,
                    # 'p': self.pass_number[i],
                    # 'n': self.instance_counters[i],
                    'tof': tof_s,
                    'r_a_x': r_a_x, 'r_a_y': r_a_y, 'r_a_z': r_a_z,
                    'v_a_x': r_a_x, 'v_a_y': v_a_y, 'v_a_z': v_a_z,
                    'r_b_x': r_b_x, 'r_b_y': r_b_y, 'r_b_z': r_b_z,
                    'v_b_x': v_b_x, 'v_b_y': r_b_y, 'v_b_z': r_b_z,
                    'time': timestamp
                }

                self.contact_instances[(self.pass_number[i], self.instance_counters[i])] = contact_instance

                if len(self.contact_instances) >= self.BUFFER_SIZE:
                    self.store_instances()
                    self.contact_instances = {}

                self.instance_counters[i] = self.instance_counters[i] + 1  # Increment instance counter

            if ended:
                pass

    def store_instances(self):
        # print("Making data frame... (this might take some time)")
        mix = pd.MultiIndex.from_tuples(self.contact_instances.keys(), names=('p', 'n'))
        df_contact_instances = pd.DataFrame(list(self.contact_instances.values()), index=mix)
        # print("Appending data frame to hdf5... (this might take some time)")
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
        pass

    def draw_update(self):
        pass

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


class CoverageAnalysis(Analysis):

    class CoverageAnalysisWriter(object):
        store = None

        def __init__(self, analysis, directory=None):
            self.analysis = analysis
            self.directory = directory

            self.file_name = None
            self.file_path = None

        def _generate_file_name(self, analysis):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            return '%s_%s.h5' % (timestamp, analysis.name)

        def initialise(self):
            self.file_name = self._generate_file_name(self.analysis)
            self.file_path = self.directory + os.path.sep + self.file_name if self.directory is not None else self.file_name
            self.store = pd.HDFStore(self.file_path)

        def store_coverage(self):
            tof = self.analysis.scenario.state.tof_s
            r_samp = self.analysis.r_samp
            num_los = self.analysis.los.sum(axis=1)

            coverage_dict = {'tof': tof,
                             'r_x': r_samp[:, 0], 'r_y': r_samp[:, 1], 'r_z': r_samp[:, 2],
                             'num_los': num_los}

            coverage_df = pd.DataFrame(coverage_dict)
            self.store.append('coverage', coverage_df)

        def flush(self):
            self.store_coverage()

    def __init__(self, scenario, sim_ob, name=None, altitude=500 * u.km, dtheta=2 * u.deg, dphi=2 * u.deg):
        super().__init__()

        self.name = name if name is not None else "COVAnalysis-%dkm" % altitude.to(u.km).value
        self.scenario = scenario
        self.altitude = altitude

        self.ffov_rad = None  # Field-of-views [rad]
        self.R_body = scenario.state.attractor.poli_body.R_mean.to(u.m).value  # Radius of attractor [m]

        self.shell_radius_m = self.R_body + altitude.to(u.m).value
        self.sim_ob = sim_ob

        # Generate sampling vectors
        theta, phi = np.meshgrid(np.arange(0, np.pi + dtheta.to(u.rad).value, dtheta.to(u.rad).value),
                                 np.arange(0, 2 * np.pi, dphi.to(u.rad).value))
        x, y, z = spherical_to_cartesian(self.shell_radius_m, theta, phi)

        self.r_samp = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    def generate_writer(self, *args, **kwargs):
        return self.CoverageAnalysisWriter(self, *args, **kwargs)

    def find_los(self):
        rr_b = np.stack([sat.rr for sat in self.sim_ob])

        # Map to arrays of size N = N_samp * N_ngso_s
        rr_a = np.repeat(self.r_samp, len(rr_b), axis=0)
        rr_b = np.tile(rr_b, (len(self.r_samp), 1))
        ffov = np.tile(self.ffov_rad, (len(self.r_samp),))

        itsc = line_intersects_sphere(rr_a, rr_b, self.scenario.state.attractor.xyz, self.R_body)
        insd = point_inside_cone(rr_a, rr_b, ffov)

        self.los = insd * (1 - itsc) > 0
        self.los = self.los.reshape((len(self.r_samp), len(self.sim_ob)))

    def initialise(self):
        # Generate a ndarray of FOVs
        self.ffov_rad = np.stack([s.fov.to(u.rad).value for s in self.sim_ob])

        # Find access at initial time point
        self.find_los()

        super().initialise()

    def run(self, t):
        pass

    def draw_update(self):
        pass

    def stop(self):
        pass
