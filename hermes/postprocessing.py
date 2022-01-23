import pandas as pd
import numpy as np
from tqdm import tqdm

from astropy import units as u, time
from astropy.coordinates import GCRS, CartesianRepresentation, ITRS, SphericalRepresentation

from hermes.geometry import find_quaternions, rotate_vectors, bulk_rotate

# Functions to quickly grab state-vectors as numpy arrays
def get_r_a(instances_df):
    return np.array([instances_df['r_a_x'].values, instances_df['r_a_y'].values, instances_df['r_a_z'].values]).T


def get_v_a(instances_df):
    return np.array([instances_df['v_a_x'].values, instances_df['v_a_y'].values, instances_df['v_a_z'].values]).T


def get_rv_a(instances_df):
    r = get_r_a(instances_df)
    v = get_v_a(instances_df)
    return r, v


def get_r_b(instances_df):
    return np.array([instances_df['r_b_x'].values, instances_df['r_b_y'].values, instances_df['r_b_z'].values]).T


def get_v_b(instances_df):
    return np.array([instances_df['v_b_x'].values, instances_df['v_b_y'].values, instances_df['v_b_z'].values]).T


def get_rv_b(instances_df):
    r = get_r_b(instances_df)
    v = get_v_b(instances_df)
    return r, v


def get_r_ab_sff(instances_df):
    return np.array([instances_df['r_ab_sff_x'].values,
                     instances_df['r_ab_sff_y'].values,
                     instances_df['r_ab_sff_z'].values]).T

def get_begin_tof(instances_df):
    pass

def get_end_tof(instances_df):
    pass

## Functions that add to the instances dataframe
def add_range(instances_df):
    """ Adds the range to the data-frame

    Parameters
    ----------
    instances_df : pandas.DataFrame
        A data frame with contact instances.

    Returns
    -------
    instances_df
        A copy of the original instances data frame with new 'd' column being the range.

    """
    instances_df['d'] = np.sqrt((instances_df.r_a_x - instances_df.r_b_x) ** 2 +
                                (instances_df.r_a_y - instances_df.r_b_y) ** 2 +
                                (instances_df.r_a_z - instances_df.r_b_z) ** 2)

    return instances_df

def add_sff(instances_df):
    r_a, v_a = get_rv_a(instances_df)
    r_b = get_r_b(instances_df)

    # SFF in ECIF (attitude determination step)
    # Pointing of satelite in ecif (x - towards velocity, z - towards zenith, y = cross(x, -z)
    x_sff_in_ecif = v_a / np.linalg.norm(v_a, axis=1)[:, np.newaxis]
    z_sff_in_ecif = r_a / np.linalg.norm(r_a, axis=1)[:, np.newaxis]
    y_sff_in_ecif = np.cross(x_sff_in_ecif, -z_sff_in_ecif, axis=1)

    # Obtain line-of-sight vector
    r_ab = r_b - r_a

    # Rotate line-of-sight vector to sff
    r_ab_sff = bulk_rotate(r_ab, x_sff_in_ecif, y_sff_in_ecif, z_sff_in_ecif)

    # Expand to store in df
    r_ab_sff_x = r_ab_sff[:, 0]
    r_ab_sff_y = r_ab_sff[:, 1]
    r_ab_sff_z = r_ab_sff[:, 2]

    instances_df['r_ab_sff_x'] = r_ab_sff_x
    instances_df['r_ab_sff_y'] = r_ab_sff_y
    instances_df['r_ab_sff_z'] = r_ab_sff_z

    return instances_df

def add_latlon(instances_df):

    epoch = time.Time('J2017', scale='tt')
    obs_times = epoch + instances_df['tof'].values * u.s

    gcrs_xyz = GCRS(x=instances_df['r_a_x'], y=instances_df['r_a_y'], z=instances_df['r_a_z'],
                    obstime=obs_times, representation_type=CartesianRepresentation)
    itrs_xyz = gcrs_xyz.transform_to(ITRS(obstime=obs_times))
    itrs_latlon = itrs_xyz.represent_as(SphericalRepresentation)

    instances_df['r_a_lat'] = itrs_latlon.lat.to(u.deg)
    instances_df['r_a_lon'] = itrs_latlon.lon.to(u.deg)

    return instances_df

### Variables from grouped passes dataframe to lists
def generate_pass_range_list(passes_df):

    range_m_list = [None] * len(passes_df)

    for i, value in enumerate(passes_df):
        name, pass_df = value
        range_m_list[i] = pass_df.d.values

    return range_m_list

def generate_pass_tof_list(passes_df):

    tof_s_list = [None] * len(passes_df)

    for i, value in enumerate(passes_df):
        name, pass_df = value
        tof_s_list[i] = pass_df.tof.values

    return tof_s_list

def generate_pass_r_ab_list(passes_df):

    r_ab_list = [None] * len(passes_df)

    for i, value in enumerate(passes_df):
        name, pass_df = value
        r_ab_list[i] = get_r_ab_sff(pass_df)

    return r_ab_list

### Generate single line per pass dataframes
def generate_grouped_passed_df(instances_df):
    passes_df = instances_df.groupby(['p'])
    return passes_df

def generate_passes_df_reduced(instances_df):
    
    begins = instances_df.groupby(['p', 'strand_name'], as_index=False).first(1)
    begins.index.name = 'p'
    begins.rename(columns={'tof': 'start_tof'}, inplace=True)
    ends = instances_df.groupby(['p', 'strand_name'], as_index=False).last(1)
    ends.index.name = 'p'
    ends.rename(columns={'tof': 'end_tof'}, inplace=True)

    passes_df = pd.concat([begins, ends.end_tof], axis=1)
    passes_df['tof'] = passes_df['start_tof'] # Set new time of flight as start of pass
    passes_df['duration'] = passes_df['end_tof'] - passes_df['start_tof']

    return passes_df

def generate_pass_df(instance_df):
    import warnings
    warnings.warn("Deprecated. Use 'generate_pass_df_reduced' with 'add_latlon'.", DeprecationWarning)

    starts = instance_df.groupby(['p', 'strand_name'], as_index=False).first(1)
    starts.index.name = 'p'
    starts.rename(columns={'tof': 'start_tof'}, inplace=True)
    ends = instance_df.groupby(['p', 'strand_name'], as_index=False).last(1)
    ends.index.name = 'p'
    ends.rename(columns={'tof': 'end_tof'}, inplace=True)

    pass_df = pd.concat([starts, ends.end_tof], axis=1)
    pass_df['duration'] = pass_df['end_tof'] - pass_df['start_tof']

    pass_df['r_b_norm'] = np.sqrt(pass_df['r_b_x'] ** 2 + pass_df['r_b_y'] ** 2 + pass_df['r_b_z'] ** 2)
    pass_df['tof'] = pass_df['start_tof']

    pass_df = add_latlon(pass_df)

    # epoch = time.Time('J2017', scale='tt')
    # obs_times = epoch + pass_df['start_tof'].values * u.s
    #
    # gcrs_xyz = GCRS(x=pass_df['r_a_x'], y=pass_df['r_a_y'], z=pass_df['r_a_z'],
    #                 obstime=obs_times, representation_type=CartesianRepresentation)
    # itrs_xyz = gcrs_xyz.transform_to(ITRS(obstime=obs_times))
    # itrs_latlon = itrs_xyz.represent_as(SphericalRepresentation)
    #
    # pass_df['r_a_lat'] = itrs_latlon.lat.to(u.deg)
    # pass_df['r_a_lon'] = itrs_latlon.lon.to(u.deg)

    return pass_df


def generate_gap_df(instances_df, step_duration=1.0, tof_start=0.0, tof_end=None):
    if tof_end is None:
        tof_end = max(instances_df['tof'])

    unique_tofs = np.unique(np.concatenate([[tof_start], instances_df['tof'].to_numpy(), [tof_end]]))
    gap_indexes = np.squeeze(np.argwhere(np.diff(unique_tofs) > 1.5 * step_duration))

    start_tofs = unique_tofs[gap_indexes]
    end_tofs = unique_tofs[gap_indexes + 1]
    duration = end_tofs - start_tofs
    gaps_dict = {'start_tof': start_tofs,
                 'end_tof': end_tofs,
                 'duration': duration}

    gaps_df = pd.DataFrame(gaps_dict, index=list(range(len(start_tofs))) if isinstance(start_tofs, np.ndarray) else [0])
    gaps_df.index.name = 'g'
    return gaps_df