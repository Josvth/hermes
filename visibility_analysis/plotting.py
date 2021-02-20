import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import GCRS, CartesianRepresentation, ITRS, SphericalRepresentation
from astropy import units as u, time
import cartopy.crs as ccrs


def plot_coverage_spatial(coverage_df):
    # Calculate lat longs
    epoch = time.Time('J2017', scale='tt')
    obs_times = epoch + coverage_df['tof'].values * u.s
    gcrs_xyz = GCRS(x=coverage_df['r_x'], y=coverage_df['r_y'], z=coverage_df['r_z'],
                    obstime=obs_times, representation_type=CartesianRepresentation)
    itrs_xyz = gcrs_xyz.transform_to(ITRS(obstime=obs_times))
    itrs_latlon = itrs_xyz.represent_as(SphericalRepresentation)

    coverage_df['r_lat'] = itrs_latlon.lat.to(u.deg)
    coverage_df['r_lon'] = ((itrs_latlon.lon + np.pi * u.rad) % (2 * np.pi * u.rad) - np.pi * u.rad).to(u.deg)

    ## Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes([0.1, 0.1, .9, .9], projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    step = 1
    llon = np.arange(-180, 180 + step, step)
    llat = np.arange(-90, 90 + step, step)

    # Set zero to NaN so it isn't accounted in the contour plot
    coverage_df.loc[coverage_df['num_los'] == 0, 'num_los'] = np.nan

    from scipy.interpolate import griddata
    val = griddata((coverage_df['r_lon'], coverage_df['r_lat']), coverage_df['num_los'],
                   (llon[None, :], llat[:, None]), method='linear')
    val = np.rint(val)

    # define the bins and normalize
    import copy
    cmap = copy.copy(plt.get_cmap('coolwarm'))  # define the colormap
    cmap.set_bad(color=(.5, .5, .5), alpha=1)

    im = ax.contourf(llon, llat, val, cmap=cmap)

    cax, kw = matplotlib.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.7)

    max_val = np.nanmax(val)

    if max_val < 8:
        bounds = np.linspace(0, max_val, int(max_val) + 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        out = matplotlib.colorbar.ColorbarBase(cax,
                                               cmap=cmap,
                                               norm=norm,
                                               ticks=bounds,
                                               spacing='proportional',
                                               label='Satellites in view',
                                               **kw)
    else:
        out = fig.colorbar(im,
                           cax=cax,
                           # spacing='proportional',
                           # ticks=bounds,
                           format='%1i',
                           label='Satellites in view',
                           **kw)

    ax.set_ylim((-90, 90))
