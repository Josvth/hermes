from builtins import object

from astropy import units as u
from mayavi import mlab

import numpy as np
from numpy import concatenate as cat

from hermes.objects import Earth, SatPlane, Satellite, SatGroup, SatGroup, _EarthObject

SCALE_UNIT = u.km
SCALE_FACTOR = 100
TUBE_RADIUS = 10.
TRAILING = 10000
TRAILING_OPACITY = np.linspace(0, 1, TRAILING)


def draw_satellite(figure, satellite, m_data=None):
    if m_data is None:  # Make new line
        pos = satellite.sample()
        x = pos.x.to(SCALE_UNIT)
        y = pos.y.to(SCALE_UNIT)
        z = pos.z.to(SCALE_UNIT)
        return mlab.plot3d(x, y, z, figure=figure, color=satellite.color, tube_radius=TUBE_RADIUS)
    else:  # We don't update planes
        return m_data
    # # Starts an orbit trace
    # x, y, z = satellite.parent.r_of(satellite) * (u.m.to(SCALE_UNIT))
    #
    # if m_data is None:  # Make new line
    #     return mlab.plot3d(x, y, z, 0, figure=figure, color=satellite.color, tube_radius=TUBE_RADIUS, transparent=True)
    # else:  # Update line
    #     xx = np.insert(m_data.mlab_source.x, 0, x)
    #     yy = np.insert(m_data.mlab_source.y, 0, y)
    #     zz = np.insert(m_data.mlab_source.z, 0, z)
    #     if len(xx) == TRAILING + 1:
    #         # Roll trail
    #         m_data.mlab_source.trait_set(x=xx[:-1], y=yy[:-1], z=zz[:-1])
    #     else:
    #
    #         # Set lookup table
    #         #m_data.module_manager.scalar_lut_manager.lut.number_of_colors = 2
    #         # TODO:
    #         #m_data.module_manager.scalar_lut_manager.lut.table = np.array([list(satellite.color) + [0], [1,1,1,1]])
    #
    #         m_data.mlab_source.reset(x=xx, y=yy, z=zz, scalars=TRAILING_OPACITY[0:len(xx)])
    #
    #     return m_data


def draw_satellite_plane(figure, plane, m_data=None):
    """Draws a satellite plane if not drawn before"""
    if m_data is None:  # Make new line
        pos = plane.ref_orbit.sample()
        x = pos.x.to(SCALE_UNIT).value
        y = pos.y.to(SCALE_UNIT).value
        z = pos.z.to(SCALE_UNIT).value
        return mlab.plot3d(x, y, z, figure=figure, color=plane.color, tube_radius=TUBE_RADIUS)
    else:  # We don't update planes
        return m_data


def draw_satellite_group(figure, group, m_data=None):
    """"Draws all satellites in a SatGroup"""

    x, y, z = group.rr.T * (u.m.to(u.km))  # Get position vectors in km

    if m_data is None:  # Make new points cloud
        # Make a list of colors
        colors = np.ones((len(group), 4))
        for i, ob in enumerate(group):
            colors[i, 0:3] = ob.color

        # If we have more than 1 satellite color we need to set the color look up table
        if len(colors) > 1:
            m_data = mlab.points3d(x, y, z, np.arange(len(group)), figure=figure, scale_mode='none',
                                   scale_factor=SCALE_FACTOR)
            m_data.module_manager.scalar_lut_manager.lut.number_of_colors = colors.shape[0]
            m_data.module_manager.scalar_lut_manager.lut.table = colors * 255
        else:
            m_data = mlab.points3d(x, y, z, np.arange(len(group)), figure=figure, scale_mode='none',
                                   scale_factor=SCALE_FACTOR, color=tuple(colors[1, 0:3]))

        return m_data
    else:  # Update points cloud data
        m_data.mlab_source.trait_set(x=x, y=y, z=z)
        return m_data


def draw_earth(attractor, actor=None, figure=None):
    if actor is None:

        from tvtk.api import tvtk

        import urllib.request
        from pathlib import Path

        local_filename = Path("/hermes_temp/blue_marble.jpg")
        if not local_filename.is_file():

            local_filename.parent.mkdir(parents=True, exist_ok=True)

            print("Downloading Earth")

            from tqdm import tqdm
            dbar = tqdm(leave=False)

            def download_bar(count, block_size, total_size):
                dbar.total = total_size
                dbar.update(block_size)

            local_filename, headers = urllib.request.urlretrieve(
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg",
                "/hermes_temp/blue_marble.jpg",
                reporthook=download_bar)
        else:
            local_filename = str(local_filename)

        img = tvtk.JPEGReader()
        img.file_name = local_filename

        texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)

        # use a TexturedSphereSource, a.k.a. getting our hands dirty
        Nrad = 180

        # create the sphere source with a given radius and angular resolution
        sphere = tvtk.TexturedSphereSource(radius=attractor.poli_body.R_mean.to(SCALE_UNIT).value,
                                           theta_resolution=Nrad,
                                           phi_resolution=Nrad)

        # assemble rest of the pipeline, assign texture
        sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
        actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)

        if figure is None:
            figure = mlab.gcf()

        figure.scene.add_actor(actor)

    else:
        actor.orientation = [0, 0, attractor.rotation.to(u.deg).value]

    return actor


def draw_satellites(satellites, m_data_list=None, figure=None):
    if figure is None:
        figure = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)

    if m_data_list is None:
        m_data_list = []

    n = 0  # We don't know how many objects need drawing

    # # Draw top level
    # m_data = self.m_data[n] if len(self.m_data) > n else self.m_data.append(None)
    # self.m_data[n] = draw_satellite_group(self.figure, satellites, m_data=m_data)
    # n = n + 1

    for ob in satellites.iter_all():

        if isinstance(ob, SatPlane):
            # We always draw the planes
            m_data = m_data_list[n] if len(m_data_list) > n else m_data_list.append(None)
            m_data_list[n] = draw_satellite_plane(figure, ob, m_data=m_data)
            n = n + 1
            pass
        elif isinstance(ob, SatGroup):
            if ob.parent is None:
                # If this SatGroup has no parent it is a top-level set and we draw the satellites
                m_data = m_data_list[n] if len(m_data_list) > n else m_data_list.append(None)
                m_data_list[n] = draw_satellite_group(figure, ob, m_data=m_data)
                n = n + 1
                pass
        elif isinstance(ob, Satellite):
            if not isinstance(ob.parent, SatPlane):
                # If the satellite is not in a plane we draw its orbit traced (see update_satellite)
                m_data = m_data_list[n] if len(m_data_list) > n else m_data_list.append(None)
                m_data_list[n] = draw_satellite(figure, ob, m_data=m_data)
                n = n + 1

        else:
            print("Cannot draw: %s" % type(ob).__name__)

    return m_data_list


class Visualisation3DGIL(object):

    def __init__(self, scenario):
        self.scenario = scenario
        self.attractor_actor = None
        self.m_data_list = []
        self.figure = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)
        # self.figure = fig if (fig is not None) else mlab.figure()
        # self.figure = None

    def initialize(self, state):
        """"Initialises the visualisation by drawing objects"""
        self.draw_attractor(state.attractor)
        self.m_data_list = draw_satellites(state.satellites, m_data_list=self.m_data_list, figure=self.figure)
        pass

    # @mlab.show
    @mlab.animate(delay=50, ui=True)
    def animate(self):

        import time as ptime

        print("Starting in:")
        t = 5
        while t:
            mins, secs = divmod(t, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            print(timer, end="\r")
            ptime.sleep(1)
            t -= 1

        # self.figure = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)

        self.figure.scene.disable_render = True
        self.initialize(self.scenario.initialize())
        self.figure.scene.disable_render = False

        for state in self.scenario.run():
            self.figure.scene.disable_render = True
            # print("VIS!")
            self.draw_attractor(state.attractor)
            self.m_data_list = draw_satellites(state.satellites, m_data_list=self.m_data_list, figure=self.figure)
            self.figure.scene.disable_render = False
            yield
            # import time
            # time.sleep(.1)

    def run(self):
        an = self.animate()
        mlab.show()  # Blocking

    # Attractor visualisation
    def draw_attractor(self, attractor):
        if isinstance(attractor, _EarthObject):
            self.attractor_actor = draw_earth(attractor, self.attractor_actor, self.figure)
        else:
            print("Cannot draw body: %s" % type(attractor).__name__)


class Visualisation3D(object):

    def __init__(self, state_queue):
        self.queue = state_queue
        self.attractor_actor = None
        self.m_data = []
        self.figure = None
        # self.figure = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)
        # self.figure = fig if (fig is not None) else mlab.figure()
        # self.figure = None

    def initialise(self, state):
        """"Initialises the visualisation by drawing objects"""
        self.draw_attractor(state.attractor)
        draw_satellites(state.satellites)
        pass

    # @mlab.show
    @mlab.animate(delay=50, ui=True)
    def animate(self):

        self.figure = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)

        self.figure.scene.disable_render = True
        self.initialise(self.queue.get())
        self.figure.scene.disable_render = False

        while True:
            self.figure.scene.disable_render = True
            # print("VIS!")
            state = self.queue.get()
            self.draw_attractor(state.attractor)
            draw_satellites(state.satellites)
            self.figure.scene.disable_render = False
            yield
            # import time
            # time.sleep(.1)

    # Attractor visualisation
    def draw_attractor(self, attractor):
        if isinstance(attractor, _EarthObject):
            self.attractor_actor = draw_earth(attractor, self.attractor_actor, self.figure)
        else:
            print("Cannot draw body: %s" % type(attractor).__name__)
