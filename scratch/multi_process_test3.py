import multiprocessing as mp
from mayavi import mlab
import numpy as np

class Data(object):

    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.mu = 0

def worker(q):

    n_mer, n_long = 6, 11
    pi = np.pi
    dphi = pi / 1000.0
    phi = np.arange(0.0, 2 * pi + 0.5 * dphi, dphi, 'd')
    mu = phi * n_mer
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    z = np.sin(n_long * mu / n_mer) * 0.5

    data = Data()
    data.x = x
    data.y = y
    data.z = z
    data.scalars = np.sin(mu)
    q.put(data)

    i = 0
    while i < 100000000:
        data.x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer +
                                     np.pi * (i + 1) / 5.) * 0.5)
        data.scalars = np.sin(mu + np.pi * (i + 1) / 5)

        q.put(data)
        i = i + 1
        import time
        time.sleep(.1)


class Visualiser(object):

    def __init__(self, q):
        self.mlab_data = None
        self.q = q

    #@mlab.show
    @mlab.animate
    def visualise(self):

        if self.mlab_data is None:  # First render
            data = self.q.get()
            # View it
            self.mlab_data = mlab.plot3d(data.x, data.y, data.z, data.scalars, tube_radius=0.025, colormap='Spectral')

        while True:
            data = self.q.get()
            self.mlab_data.mlab_source.reset(x=data.x, y=data.y, z=data.z, scalars=data.scalars)
            #self.mlab_data.mlab_source.trait_set(x=data.x, y=data.y, z=data.z)
            yield


def visualize_target(visualiser):
    visualiser.visualise()

def mlab_thread():
    mlab.show()

if __name__ == '__main__':
    q = mp.Queue()
    work_process = mp.Process(target=worker, args=(q,), daemon = True)
    work_process.start()
    #work_process.deamon = True

    #mlab_process = mp.Process(target=mlab_thread, args=(), daemon = True)
    #mlab_process.start()

    visualiser = Visualiser(q)
    visualise_process = mp.Process(target=visualize_target, args=(visualiser,), daemon = True)
    visualise_process.start()
    #visualise_process.deamon = True
    visualise_process.join()