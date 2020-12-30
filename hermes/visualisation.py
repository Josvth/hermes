from builtins import object

from astropy import units as u
from mayavi import mlab

import numpy as np
from numpy import concatenate as cat

from hermes.analysis import LOSAnalysis
from hermes.geometry import fov_edge_range
from hermes.objects import Earth, SatPlane, Satellite, SatGroup, SatGroup, _EarthObject
from hermes.util import hex2rgb

SCALE_UNIT = u.km
SCALE_FACTOR = 100
TUBE_RADIUS = 10.
TRAILING = 50
TRAILING_OPACITY = np.linspace(0, 1, TRAILING)


def draw_vector(figure, vector, origin=np.array([0, 0, 0]), m_data=None):
    vector = np.atleast_2d(vector * u.m.to(SCALE_UNIT))
    origin = np.atleast_2d(origin * u.m.to(SCALE_UNIT))
    m_data = mlab.quiver3d(origin[:, 0], origin[:, 1], origin[:, 2], vector[:, 0], vector[:, 1], vector[:, 2],
                           reset_zoom=False,
                           mode='2darrow', scale_mode='vector', scale_factor=1, line_width=1.0)
    return m_data


def draw_satellite_fov(figure, satellite, m_data=None):
    r = satellite.parent.r_of(satellite) * u.m.to(SCALE_UNIT)  # position vector in meters
    r_norm = np.linalg.norm(r)

    fov = satellite.fov.to(u.rad).value
    R_body = satellite.attractor.R_mean.to(SCALE_UNIT).value

    # Calculate the height and radius of the cone
    a = fov_edge_range(r_norm, fov, R_body)
    h_cone = a * np.cos(fov)

    origin = r * (r_norm - h_cone) / r_norm
    tip = r - origin

    if m_data is None:
        m_data = mlab.quiver3d(origin[0], origin[1], origin[2], tip[0], tip[1], tip[2],
                               figure=figure, reset_zoom=False, mode='cone',
                               color=satellite.color[0:3],
                               scale_mode='vector', scale_factor=1, resolution=64)
        m_data.glyph.glyph_source.glyph_source.angle = satellite.fov.to(u.deg).value
        m_data.actor.actor.property.opacity = satellite.fov_3D_opacity
    else:
        m_data.mlab_source.trait_set(x=origin[0], y=origin[1], z=origin[2],
                                     u=tip[0], v=tip[1], w=tip[2])
        # m_data.glyph.glyph_source.glyph_source.angle = satellite.fov.to(u.deg).value  # Reset these just in case
        # m_data.actor.actor.property.opacity = satellite.fov_3D_opacity  # Reset these just in case

    return m_data


def draw_satellite(figure, satellite, m_data=None):
    m_data_plane = m_data['plane'] if m_data is not None and 'plane' in m_data else None
    m_data_trace = m_data['trace'] if m_data is not None and 'trace' in m_data else None
    m_data_fov = m_data['fov'] if m_data is not None and 'fov' in m_data else None

    if satellite.plane_3D_show:
        m_data_plane = draw_orbit(figure, satellite, satellite.color, m_data=m_data_plane)
    elif m_data_plane is not None:
        m_data_plane.remove()
        m_data_plane = None

    if satellite.trace_3D_show:
        m_data_trace = draw_satellite_trace(figure, satellite, m_data=m_data_trace)
    elif m_data_trace is not None:
        m_data_trace.remove()
        m_data_trace = None

    if satellite.fov_3D_show:
        m_data_fov = draw_satellite_fov(figure, satellite, m_data=m_data_fov)
    elif m_data_fov is not None:
        m_data_fov.remove()
        m_data_fov = None

    m_data = {'plane': m_data_plane, 'trace': m_data_trace, 'fov': m_data_fov}

    return m_data


def draw_satellite_trace(figure, satellite, m_data=None):
    # Starts an orbit trace
    x, y, z = satellite.parent.r_of(satellite) * (u.m.to(SCALE_UNIT))

    if m_data is None:  # Make new line
        return mlab.plot3d(x, y, z, 0, figure=figure, tube_radius=TUBE_RADIUS)
    else:  # Update line
        xx = np.insert(m_data.mlab_source.x, 0, x)
        yy = np.insert(m_data.mlab_source.y, 0, y)
        zz = np.insert(m_data.mlab_source.z, 0, z)
        if len(xx) == satellite.trace_3D_length + 1:
            # Roll trail
            m_data.mlab_source.trait_set(x=xx[:-1], y=yy[:-1], z=zz[:-1])
        else:

            # Set lookup table
            r = np.linspace(satellite.color[0], 1, satellite.trace_3D_length)
            g = np.linspace(satellite.color[1], 1, satellite.trace_3D_length)
            b = np.linspace(satellite.color[2], 1, satellite.trace_3D_length)
            a = np.linspace(1, 0, satellite.trace_3D_length)
            colors = np.array([r, g, b, a]).T
            m_data.module_manager.scalar_lut_manager.lut.number_of_colors = len(xx)
            m_data.module_manager.scalar_lut_manager.lut.table = colors[-len(xx):, :] * 255

            ss = np.arange(len(xx))

            m_data.mlab_source.reset(x=xx, y=yy, z=zz, scalars=ss)

        return m_data


def draw_orbit(figure, orbit, color, m_data=None):
    """Draws a satellite plane if not drawn before"""
    if m_data is None:  # Make new line
        pos = orbit.sample(values=1000)
        x = pos.x.to(SCALE_UNIT).value
        y = pos.y.to(SCALE_UNIT).value
        z = pos.z.to(SCALE_UNIT).value
        return mlab.plot3d(x, y, z, figure=figure, color=color, tube_radius=TUBE_RADIUS)
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
                                   scale_factor=SCALE_FACTOR, color=tuple(colors[0, 0:3]))

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
        actor.orientation = [0, 0, attractor.rotation_deg]

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
            m_data_list[n] = draw_orbit(figure, ob.ref_orbit, ob.color, m_data=m_data)
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


def draw_satellites_christmas(satellites, m_data_list=None, figure=None):
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
            m_data_list[n] = draw_orbit(figure, ob, ob.color, m_data=m_data)
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
                m_data_list[n] = draw_satellite_trace(figure, ob, m_data=m_data)
                n = n + 1

        else:
            print("Cannot draw: %s" % type(ob).__name__)

    return m_data_list


def draw_analysis(analyses, figure=None):
    if figure is None:
        figure = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)

    m_data = []

    for analysis in analyses:
        if isinstance(analysis, LOSAnalysis):
            # draw line-of-sight vectors
            if analysis.show_los:
                for r_b in analysis.rr_b[analysis.los, :]:
                    m_data.append(draw_los_vector(analysis.r_a, r_b, color=hex2rgb('#01FF70'), figure=figure))

            # draw no line-of-sight vectors
            if analysis.show_nolos:
                for r_b in analysis.rr_b[~analysis.los, :]:
                    m_data.append(draw_los_vector(analysis.r_a, r_b, color=hex2rgb('#c3312f'), figure=figure))

    return m_data


def draw_los_vector(r_a, r_b, color=None, figure=None):
    if figure is None:
        figure = mlab.figure(size=(1200, 1200), bgcolor=(1.0, 1.0, 1.0))  # Make a new figure (similar to MATLAB)

    if color is None:
        color = hex2rgb('#01FF70')

    x = np.array([0, 0])
    y = np.array([0, 0])
    z = np.array([0, 0])

    x[0] = r_a[0] * u.m.to(SCALE_UNIT)
    y[0] = r_a[1] * u.m.to(SCALE_UNIT)
    z[0] = r_a[2] * u.m.to(SCALE_UNIT)

    x[1] = r_b[0] * u.m.to(SCALE_UNIT)
    y[1] = r_b[1] * u.m.to(SCALE_UNIT)
    z[1] = r_b[2] * u.m.to(SCALE_UNIT)

    return mlab.plot3d(x, y, z, tube_radius=TUBE_RADIUS, color=color, figure=figure, reset_zoom=False)


class Visualisation3DGIL(object):

    def __init__(self, state_generator=None, figure=None):

        self.state_generator = state_generator
        self.attractor_actor = None

        self.m_data_sats = []
        self.m_data_analyses = []
        self.figure = figure if (figure is not None) else mlab.figure(size=(1200, 1200), bgcolor=(
            1.0, 1.0, 1.0))

    # @mlab.show
    @mlab.animate(delay=50, ui=True)
    def animate(self):

        import time as ptime

        print("Starting in: ", end="")
        t = 5
        while t:
            mins, secs = divmod(t, 60)
            timer = 'Starting in: {:02d}:{:02d}'.format(mins, secs)
            print(timer, end="\r")
            ptime.sleep(1)
            t -= 1
        print(timer)

        for state in self.state_generator:
            self.figure.scene.disable_render = True
            self.visualise(state)  # Draw initial picture
            self.figure.scene.disable_render = False
            yield

    def run(self):
        an = self.animate()
        mlab.show()  # Blocking

    def visualise(self, state, show=False):

        self.draw_attractor(state.attractor)
        self.m_data_sats = draw_satellites(state.satellites, m_data_list=self.m_data_sats,
                                           figure=self.figure)

        for m_data_analysis in self.m_data_analyses:
            m_data_analysis.remove()
        self.m_data_analyses = draw_analysis(state.analyses, figure=self.figure)

        if show:
            self.show()

    def show(self):
        mlab.show()

    # Attractor visualisation
    def draw_attractor(self, attractor):
        if isinstance(attractor, _EarthObject):
            self.attractor_actor = draw_earth(attractor, self.attractor_actor, self.figure)
        else:
            print("Cannot draw body: %s" % type(attractor).__name__)


class Visualisation3DGILChristmas(Visualisation3DGIL):

    def __init__(self, scenario, figure=None):
        super().__init__(scenario, figure)

    def visualise(self, show=False):
        self.draw_attractor(self.scenario.state.attractor)
        self.m_data_sats = draw_satellites_christmas(self.scenario.state.satellites, m_data_list=self.m_data_sats,
                                                     figure=self.figure)

        for m_data_analysis in self.m_data_analyses:
            m_data_analysis.remove()
        self.m_data_analyses = draw_analysis(self.scenario.state.analyses, figure=self.figure)


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
