from abc import ABC, abstractmethod
from itertools import chain

from mayavi import mlab
import poliastro
from poliastro.frames import Planes
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import *
from scipy.constants import kilo

from astropy import time, units as u
import numpy as np

from hermes.propagation_bulk import markley_bulk
from hermes.util import calc_lmn, coe2xyz_fast, hex2rgb
from collections import MutableSequence

SCALE_FACTOR = 150
TUBE_RADIUS = 15.

J2000 = time.Time('J2000', scale='utc')


# PROP_METHOD = mean_motion   # Takes about 84s to do 6000 seconds
# PROP_METHOD = markley  # Takes about 75s to do 6000 seconds


class ScenarioObject(ABC):

    def __init__(self):
        super().__init__()
        self._xyz_in_m = np.array([0, 0, 0])

    @abstractmethod
    def __len__(self):
        pass

    @property
    def xyz(self):
        """Current current Cartesian coordinates for fast access"""
        return self._xyz_in_m

    @abstractmethod
    def initialize(self):
        """Allows the object to initialize before simulation/propagation is ran."""
        pass

    @abstractmethod
    def propagate_to(self, t):
        """Propagates object to a specific time. Should update _xyz"""
        pass

    # Plotting
    @abstractmethod
    def draw(self, figure):
        pass

    @abstractmethod
    def draw_update(self, figure):
        pass


class CelestialBody(ScenarioObject):

    def __init__(self, body):
        self.tof_last = None
        self.sphere_actor = None
        self.rotation = 0 * u.deg
        self.poli_body = body
        super().__init__()

    # Todo implement
    @property
    def xyz(self):
        return np.array([0, 0, 0])

    def __len__(self):
        return 1

    # Todo implement
    def initialize(self):
        pass

    def propagate_to(self, t):
        if self.tof_last is None:
            self.tof_last = t
            return

        dt = t - self.tof_last
        self.tof_last = t
        drot = dt.to(u.s) * 360 * u.deg / (86164.09 * u.s)  # Todo, make less hardcoded
        self.rotation = (self.rotation + drot) % (360 * u.deg)

    def draw(self, figure):
        from tvtk.api import tvtk

        import urllib.request
        urllib.request.urlretrieve("https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg", "blue_marble.jpg")

        img = tvtk.JPEGReader()
        img.file_name = "blue_marble.jpg"

        texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)

        # use a TexturedSphereSource, a.k.a. getting our hands dirty
        Nrad = 180

        # create the sphere source with a given radius and angular resolution
        sphere = tvtk.TexturedSphereSource(radius=self.poli_body.R_mean.to(u.km).value, theta_resolution=Nrad,
                                           phi_resolution=Nrad)

        # assemble rest of the pipeline, assign texture
        sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
        self.sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
        figure.scene.add_actor(self.sphere_actor)

    def draw_update(self, figure):
        self.sphere_actor.orientation = [0, 0, self.rotation.to(u.deg).value]
        pass


class _EarthObject(CelestialBody):

    plot_color = '#0e4a5b'

    def __init__(self):
        super().__init__(poliastro.bodies.Earth)


Earth = _EarthObject()


class ObjectOrbit(Orbit, ScenarioObject):

    def __init__(self, state, epoch, plane):
        self.plot_color = hex2rgb('#ffffff')
        self.orbit_points = None
        super().__init__(state, epoch, plane)

    # Abstract implementations
    def __len__(self):
        return 1

    def initialize(self):
        pass

    def propagate_to(self, t):
        # Todo it would be real nice if this could be done pointer wise
        self._xyz = propagate(self, t)._xyz[0]

        # k = self.attractor.k.to(u.m ** 3 / u.s ** 2).value
        # tt = t.to(u.s).value
        # p = self.p.value
        # ecc = self.ecc.value
        # inc = self.inc.to(u.rad).value,
        # raan = self.raan.to(u.rad).value,
        # argp = self.argp.to(u.rad).value
        # nu0 = self.nu.to(u.rad).value

        # test_nu = markley_bulk(k, tt, p, ecc, inc, raan, argp, nu0)
        # test_xyz, vv = coe2rv(k, p, ecc, inc, raan, argp, test_nu)

        # assert np.array_equal(self._xyz._xyz.value, test_xyz)

        pass

    def draw(self, figure):
        if self.orbit_points is None:
            pos = self.sample()

            self.orbit_points = mlab.plot3d(pos.x, pos.y, pos.z, color=self.plot_color, tube_radius=TUBE_RADIUS)

    def draw_update(self, figure):
        pass

    def set_color(self, color):
        self.plot_color = hex2rgb(color)


class GroupNode(ABC):
    parent = None

    # Todo, re-introduce update methods for speed?

    @abstractmethod
    def draw_orbits(self, figure):
        pass

    @abstractmethod
    def draw_sats(self, figure):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class Satellite(ObjectOrbit, GroupNode):
    fov = 45 * u.deg  # Nadir pointing FOV

    def __init__(self, state, epoch, plane):
        super().__init__(state, epoch, plane)
        self._xyz = self.r  # set the initial position
        self.pos_points = None

    def draw_sats(self, figure):

        # ToDo somehow get position from parent if this is called
        if self.pos_points is None:
            x, y, z = self.r[0].to(u.km), self.r[1].to(u.km), self.r[2].to(u.km)
            self.pos_points = mlab.points3d(x, y, z, color=self.plot_color, scale_factor=SCALE_FACTOR,
                                            figure=figure)
        else:
            x, y, z = self.r[0].to(u.km), self.r[1].to(u.km), self.r[2].to(u.km)
            self.pos_points.mlab_source.trait_set(x=x, y=y, z=z)

    def draw_orbits(self, figure):
        # Call ObjectOrbit draw
        super().draw(figure)

    def __iter__(self):
        yield self


class SatGroup(GroupNode, MutableSequence):

    # ToDo add a parent to this tree structure?
    def __init__(self, *children: list, group_type="group"):
        super().__init__()
        self._children = list(children)

        self._group_type = group_type

        #  Empty fields
        self.xyz_in_m = 0
        self.k = 0
        self.pp = 0
        self.eecc = 0
        self.iinc = 0
        self.rraan = 0
        self.aargp = 0
        self.nnu0 = 0

        self.ll1 = 0
        self.mm1 = 0
        self.nn1 = 0
        self.ll2 = 0
        self.mm2 = 0
        self.nn2 = 0

        self.colors = []

        self.sat_points = None

    def __len__(self):
        # Return length of whole tree
        # Todo maybe use __iter__ here?
        return sum([len(child) for child in self._children])

    def __getitem__(self, index):
        return self._children[index]

    def __delitem__(self, index):
        del self._children[index]

    def insert(self, index, value):
        # if not isinstance(value, Satellite) and not isinstance(value, SatGroup) and not issubclass(value.ge, SatGroup):
        #     warnings.warn("Type %s is not supported." % value, UserWarning)
        self._children.insert(index, value)

    def __setitem__(self, index, value):
        self._children[index] = value

    def __iter__(self):

        # Iterate depth first
        for child in chain(*map(iter, self._children)):
            yield child

    def initialize(self):

        # Store satellite parameters
        length = len(self)
        self.xyz_in_m = np.zeros((length, 3))
        # Todo make Earth less hard coded
        self.k = Earth.poli_body.k.to(u.m ** 3 / u.s ** 2).value

        self.pp = np.zeros(length)
        self.eecc = np.zeros(length)
        self.iinc = np.zeros(length)
        self.rraan = np.zeros(length)
        self.aargp = np.zeros(length)
        self.nnu0 = np.zeros(length)

        self.colors = np.zeros((length, 4))

        # generate vectors
        for i, s in enumerate(self):
            self.pp[i] = s.p.value
            self.eecc[i] = s.ecc.value
            self.iinc[i] = s.inc.to(u.rad).value
            self.rraan[i] = s.raan.to(u.rad).value
            self.aargp[i] = s.argp.to(u.rad).value
            self.nnu0[i] = s.nu.to(u.rad).value
            self.xyz_in_m[i] = s.r.to(u.m).value
            self.colors[i, :] = np.array(
                [s.plot_color[0], s.plot_color[1], s.plot_color[2], 1]) * 255  # Add no transparency

        # precalculate lmn
        self.ll1, self.mm1, self.nn1, self.ll2, self.mm2, self.nn2 = calc_lmn(self.iinc, self.rraan, self.aargp)

    def propagate_to(self, t):

        # propagate at once
        nnu = markley_bulk(self.k, t.to(u.s).value, self.pp, self.eecc, self.iinc, self.rraan, self.aargp, self.nnu0)

        import numpy.ctypeslib as nc
        # self._xyz, vv = coe2rv(self.k, self.pp, self.eecc, self.iinc, self.rraan, self.aargp, nnu)
        # self._xyz = self._xyz * u.m

        # self._xyz = coe2xyz(self.k, self.pp, self.eecc, self.iinc, self.rraan, self.aargp, nnu) * u.m
        # self.xyz_in_m = coe2xyz_fast(self.pp, self.eecc, self.ll1, self.mm1, self.nn1, self.ll2, self.mm2, self.nn2,
        #                              nnu)
        coe2xyz_fast(self.xyz_in_m, self.pp, self.eecc, self.ll1, self.mm1, self.nn1, self.ll2, self.mm2, self.nn2,
                     nnu)

    def draw(self, figure):

        # Check to see if we have drawn before
        if self.sat_points is None:
            # Let children draw but not the actual satellites as we draw them as one point cloud
            # Very important here is to use 'self._children' and NOT 'self' because it will then also iterate over
            # individual satellites that are in planes
            for child in self._children:
                child.draw_orbits(figure)

        # Draw all the sats in a figure
        self.draw_sats(figure)

    def draw_orbits(self, figure):

        # Let children draw but not the actual satellites
        # Very important here is to use 'self._children' and NOT 'self' because it will then also iterate over
        # individual satellites that are in planes
        for child in self._children:
            child.draw_orbits(figure)

    def draw_sats(self, figure):

        # Draws all satellites as a single plot3D

        if self.sat_points is None:
            x, y, z = self.get_mayavi_xyz()
            self.sat_points = mlab.points3d(x, y, z, np.arange(len(self)), figure=figure, scale_mode='none',
                                            scale_factor=SCALE_FACTOR)
            # self.sat_points.glyph.color_mode = 'color_by_scalar'  # Color by scalar
            self.sat_points.module_manager.scalar_lut_manager.lut.number_of_colors = len(self)
            self.sat_points.module_manager.scalar_lut_manager.lut.table = self.colors
            mlab.draw()
        else:
            x, y, z = self.get_mayavi_xyz()
            self.sat_points.mlab_source.trait_set(x=x, y=y, z=z)

    def get_mayavi_xyz(self):
        # # ToDo I assume all points are in u.km
        # stacked = np.stack(self._xyz_in_m)
        # x, y, z = stacked[:, 0], stacked[:, 1], stacked[:, 2]
        x, y, z = self.xyz_in_m[:, 0] / 1000, self.xyz_in_m[:, 1] / 1000, self.xyz_in_m[:, 2] / 1000
        return x, y, z

    # Satellite mutations
    def set_color(self, color):
        for child in self._children:
            child.set_color(color)

    def set_fov(self, fov):
        for sat in self._children:
            sat.fov = fov

    @classmethod
    @u.quantity_input(a=u.m, ecc=u.one, inc=u.rad, raan=u.rad, argp=u.rad, nu=u.rad)
    def as_plane(cls, attractor, a, ecc, inc, raan, argp, nnu, epoch=J2000, plane=Planes.EARTH_EQUATOR):
        """

        Parameters
        ----------
        attractor : Body
            Main attractor.
        a : ~astropy.units.Quantity
            Semi-major axis.
        ecc : ~astropy.units.Quantity
            Eccentricity.
        inc : ~astropy.units.Quantity
            Inclination
        raan : ~astropy.units.Quantity
            Right ascension of the ascending node.
        argp : ~astropy.units.Quantity
            Argument of the pericenter.
        nnu : np.array of ~astropy.units.Quantity
            1D np.array of length n_sats specifing the mean anomaly of each of the satellites
        epoch : ~astropy.time.Time, optional
            Epoch time
        plane : ~poliastro.frames.Planes, optional
            Fundamental plane of the frame.
        """

        group = SatPlane()
        group.ref_orbit = ObjectOrbit.from_classical(attractor, a, ecc, inc, raan, argp, 0 * u.deg, epoch, plane)

        n_sat = len(nnu)

        for i in range(n_sat):
            # Create new satellite with different Mean Anomaly M
            sat = Satellite.from_classical(attractor, a, ecc, inc, raan, argp, nnu[i], epoch, plane)
            sat.parent = group
            # Add satellite to group
            group.append(sat)

        return group

    @classmethod
    @u.quantity_input(a=u.m, ecc=u.one, inc=u.rad, rraan=u.rad, argp=u.rad)
    def as_set(cls, attractor, a, ecc, inc, rraan, aargp, nnnu, epoch=J2000, plane=Planes.EARTH_EQUATOR):
        """

        Parameters
        ----------
        attractor : Body
            Main attractor.
        a : ~astropy.units.Quantity
            Semi-major axis of the orbital set.
        ecc : ~astropy.units.Quantity
            Eccentricity of the orbital set.
        inc : ~astropy.units.Quantity
            Inclination of the orbital set
        rraan : ~astropy.units.Quantity array
            1-N np.array with the right ascension of the ascending node of each plane.
        aargp : ~astropy.units.Quantity
            1-N np.array with the argument of the pericenter of each plane.
        nnnu : list 
            n_plane sized list with 1-N np.arrays specifing the mean anomaly of each of the satellites inside the plane.
        epoch : ~astropy.time.Time, optional
            Epoch time
        plane : ~poliastro.frames.Planes, optional
            Fundamental plane of the frame.
        """

        # Todo add check for sizes rraan, aargp and nnnu
        if len({len(rraan), len(aargp), len(nnnu)}) != 1:
            raise ValueError("Size of rraan, aargp and nnnu should be identical. Currently (%d,%d,%d)" % (
                len(rraan), len(aargp), len(nnnu)))

        n_plane = len(rraan)

        group = SatSet()
        group._group_type = "set"

        for i_p in range(n_plane):
            group_plane = cls.as_plane(attractor, a, ecc, inc, rraan[i_p], aargp[i_p], np.squeeze(nnnu[i_p]), epoch,
                                       plane)
            group_plane.parent = group
            # Todo do parenting in append?
            group.append(group_plane)

        return group


class SatPlane(SatGroup):
    group_type = "plane"
    ref_orbit = None

    # def draw_orbit(self, figure):
    #     if self.ref_orbit is not None:
    #         self.ref_orbit.draw(figure)

    def draw_orbits(self, figure):
        if self.ref_orbit is not None:
            self.ref_orbit.draw(figure)
        self.ref_orbit.draw(figure)

    # def draw(self, figure, draw_sats=True):
    #     self.draw_orbit(figure)
    #     if draw_sats:
    #         self.draw_sats(figure)

    def set_color(self, color):
        if self.ref_orbit is not None:
            self.ref_orbit.set_color(color)
        super().set_color(color)

    # ToDo add .circular constructor


class SatSet(SatGroup):
    group_type = "set"

    # ToDo add .circular constructor


class Constellation(SatGroup):
    group_type = "constellation"
