from analysis import AccessAnalysis
from objects import OrbitalPlaneSet, Satellite, Earth, OrbitalPlane, SatGroup

from simulation import *

## Main script

fig = None
# fig = mlab.figure(size=(1200, 1200))

J2015 = time.Time('J2015', scale='tt')

start = time.Time('2019-09-01 10:00:00.000', scale='tt')
stop = time.Time('2019-09-08 10:00:00.000', scale='tt')
step = 10 * u.s

scenario = Scenario(Earth, start, stop, step, figure=fig)

constellation = SatGroup()
constellation.append(Satellite.circular(Earth, 500 * u.km, inc=75 * u.deg, raan=0*u.deg))
constellation.append(SatGroup.as_plane(Earth, (1500 + 6371) * u.km, 0 * u.one, 0 * u.deg, 0 * u.deg, 0 * u.deg, np.array([0, 45, 90, 135, 180]) * u.deg))


pass
#
# plane_set_1150 = OrbitalPlaneSet(
#     Earth, n_planes=32, n_ssats=50, p_aa=1150 * u.km, i=53.0 * u.deg, raan_st_deg=0 * u.deg,
#     nu_st_deg=[0.0, 1.9, 3.8, 5.7, 0.5, 2.4, 4.3, 6.2,
#                0.9, 2.8, 4.7, 6.6, 1.4, 3.3, 5.2, 7.1,
#                1.8, 3.7, 5.6, 0.3, 2.3, 4.2, 6.1, 0.8,
#                2.7, 4.6, 6.5, 1.2, 3.2, 5.1, 7.0, 1.7] * u.deg,
#     epoch=J2015)
#
# # ToDo this function slows down initialization a lot
# plane_set_1150.generate_planes()
#
# sat = Satellite.circular(Earth, 500 * u.km, inc=75 * u.deg, raan=0*u.deg)
# sat2 = Satellite.circular(Earth, 1500 * u.km, inc=90 * u.deg, arglat=45 * u.deg)
# test_plane = OrbitalPlane.from_classical(Earth, Earth.R_mean + 1500 * u.km, 0 * u.one, 45 * u.deg, 0 * u.deg, 0 * u.deg, 0 * u.deg)
# test_plane.generate_sats(50)
#
# # print(sat.T)
#
# # Add objects
# scenario.add_object(sat)
# # scenario.add_object(sat2)
# scenario.add_object(plane_set_1150)
# # scenario.add_object(test_plane)
#
# # Test propagation
# #tt = scenario.epoch + 1000
# #propagate(sat, tt, method=kepler)
#
# # Add analyses
# scenario.add_analysis(AccessAnalysis(scenario, sat, plane_set_1150))
# # if fig is None:
# # scenario.add_analysis(AccessAnalysis(scenario, sat, sat2))
#
# # tt = 1000
# # scenario.propagate_to(generate_time_vector(scenario.epoch, tt * u.s))
# #scenario.draw_scenario()
#
# # Run simulation
# #pos = plane_set_1150.planes[1].sample(50 + 1)
#
# if fig is None:
#     scenario.step()  # do one step to let numba compile
#     for step in scenario.simulate():
#         pass
#
#     # for i, an in enumerate(scenario.analyses):
#     #     num_access = len(an.accesses)
#     #     print("Analysis %d found %d" % (i, num_access))
#     #     if num_access:
#     #         for j, ac in enumerate(an.accesses):
#     #             print("Access %d had duration %2.1f s" % (j, ac.duration.to(u.s).value))
#
# # animation
# if not(fig is None):
#     scenario.draw_scenario()
#     scenario.step() # do one step to let numba compile
#     scenario.animate(follow=sat)
#     mlab.show()


