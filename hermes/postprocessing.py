import pandas as pd
import numpy as np
from tqdm import tqdm

from astropy import units as u, time
from astropy.coordinates import GCRS, CartesianRepresentation, ITRS, SphericalRepresentation

# Functions to quickly grab state-vectors as numpy arrays
from hermes.geometry import find_quaternions, rotate_vectors, bulk_rotate
from hermes.util import row_cross


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


def add_sff_2(instances_df):
    r_a, v_a = get_rv_a(instances_df)
    r_b = get_r_b(instances_df)

    xyz_eci = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

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


def add_sff(instances_df):
    import quaternionic as qn

    # State vector of satellite a
    r_a, v_a = get_rv_a(instances_df)
    r_a = r_a[0:10, :]
    v_a = v_a[0:10, :]

    N = r_a.shape[0]

    # ECIF frame
    x_ecif = np.tile(np.array([1, 0, 0]), (N, 1))
    y_ecif = np.tile(np.array([0, 1, 0]), (N, 1))
    z_ecif = np.tile(np.array([0, 0, 1]), (N, 1))

    # Find the quaternion that aligns x
    q_x = qn.array(find_quaternions(x_sff_in_ecif, x_ecif))
    x_test = rotate_vectors(x_sff_in_ecif, q_x)

    # Rotate other axes to follow
    y_int = rotate_vectors(y_sff_in_ecif, q_x)
    z_int = rotate_vectors(z_sff_in_ecif, q_x)

    # Find the quaternion that aligns z
    q_z = qn.array(find_quaternions(z_int, z_ecif))
    z_test = rotate_vectors(z_int, q_z)

    y_test1 = rotate_vectors(y_int, q_z)
    y_test2 = rotate_vectors(y_sff_in_ecif, q_z * q_x)
    y_test3 = rotate_vectors(y_sff_in_ecif, q_z)

    x_test2 = rotate_vectors(x_sff_in_ecif, q_z * q_x)
    z_test2 = rotate_vectors(z_sff_in_ecif, q_z * q_x)
    pass

    # # Find quaternion that rotates x to the right place
    # q_x = qn.array(find_quaternions(x_ecif, x_sff_in_ecif))
    #
    # # Then rotate the z to this intermediate frame
    # z_int = rotate_vectors(z_ecif, q_x)
    #
    # # Then find the vector that rotate this intermediate  that rotates z to the right place
    # q_z = qn.array(find_quaternions(z_int, z_sff_in_ecif))
    #
    # q_ecif_sff = q_x * q_z
    #
    # # Test
    # x_test = rotate_vectors(x_ecif, q_ecif_sff)

    return instances_df


### Generate single line per pass dataframes
def generate_pass_df(instance_df):
    starts = instance_df.groupby(['p', 'strand_name'], as_index=False).first(1)
    starts.index.name = 'p'
    starts.rename(columns={'tof': 'start_tof'}, inplace=True)
    ends = instance_df.groupby(['p', 'strand_name'], as_index=False).last(1)
    ends.index.name = 'p'
    ends.rename(columns={'tof': 'end_tof'}, inplace=True)

    pass_df = pd.concat([starts, ends.end_tof], axis=1)
    pass_df['duration'] = pass_df['end_tof'] - pass_df['start_tof']

    pass_df['r_b_norm'] = np.sqrt(pass_df['r_b_x'] ** 2 + pass_df['r_b_y'] ** 2 + pass_df['r_b_z'] ** 2)

    epoch = time.Time('J2017', scale='tt')
    obs_times = epoch + pass_df['start_tof'].values * u.s

    gcrs_xyz = GCRS(x=pass_df['r_a_x'], y=pass_df['r_a_y'], z=pass_df['r_a_z'],
                    obstime=obs_times, representation_type=CartesianRepresentation)
    itrs_xyz = gcrs_xyz.transform_to(ITRS(obstime=obs_times))
    itrs_latlon = itrs_xyz.represent_as(SphericalRepresentation)

    pass_df['r_a_lat'] = itrs_latlon.lat.to(u.deg)
    pass_df['r_a_lon'] = itrs_latlon.lon.to(u.deg)

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


### OLD DO NOT USE
def instances_to_passes_fast(instances_df, step_duration=1.0):
    # Sort in chronological order (if not already)
    # instances_df = instances_df.sort_values(by= 'tof')

    # Combine xyz positions
    r_a = np.vstack(
        (instances_df.xyz_a_x.to_numpy().T, instances_df.xyz_a_y.to_numpy().T, instances_df.xyz_a_z.to_numpy().T)).T
    r_b = np.vstack(
        (instances_df.xyz_b_x.to_numpy().T, instances_df.xyz_b_y.to_numpy().T, instances_df.xyz_b_z.to_numpy().T)).T

    instances_df = instances_df.assign(r_a=pd.Series(np.vsplit(r_a, len(r_a))))
    instances_df = instances_df.assign(r_b=pd.Series(np.vsplit(r_b, len(r_b))))
    instances_df = instances_df.drop(columns=['xyz_a_x', 'xyz_a_y', 'xyz_a_z', 'xyz_b_x', 'xyz_b_y', 'xyz_b_z'])

    # Group by strand
    grouped_df = instances_df.groupby('StrandName')

    pp = []

    pass_id = 0

    with tqdm(total=len(grouped_df), desc='Group') as group_bar:
        for name, group in grouped_df:

            # Calculate delta time between instances
            group = group.assign(
                delta_tof=group['tof'] - np.concatenate(([group['tof'].iloc[0]], group['tof'].iloc[0:-1].values)))

            i_l = group.index[0]
            for i in group.index[group['delta_tof'] > 2 * step_duration]:

                # Select instances between indexes
                pass_instances = group.loc[i_l:i - 1]

                if pass_instances.tof.values[1] == pass_instances.tof.values[0]:
                    pass

                # Create pass df
                p = {"StartTof": pass_instances.iloc[0].tof,
                     "StopTof": pass_instances.iloc[-1].tof,
                     "ttof": [pass_instances.tof.values],
                     "rr_a": [np.concatenate(pass_instances.r_a.values)],
                     "rr_b": [np.concatenate(pass_instances.r_b.values)],
                     "rrange": [np.linalg.norm(
                         np.concatenate(pass_instances.r_b.values) - np.concatenate(pass_instances.r_a.values),
                         axis=1)],
                     'StrandName': pass_instances.iloc[0].StrandName}

                # Create pass df and add to list
                pass_df = pd.DataFrame.from_dict(p)
                pp.append(pass_df)

                # # plot pass
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots()
                # ax.plot(pass_df.ttof[0], pass_df.rrange[0])
                # fig.show()

                i_l = i

            group_bar.update(1)

        passes_df = pd.concat(pp, ignore_index=True)
        passes_df['Duration'] = passes_df['StopTof'] - passes_df['StartTof']
        # return passes_df
        return passes_df.sort_values(by='StartTof').reset_index(drop=True)


def instances_to_passes(instances_df, step_duration=1.0):
    # Sort in chronological order (if not already)
    instances_df = instances_df.sort_values(by='tof')

    # Combine xyz positions
    r_a = np.vstack(
        (instances_df.xyz_a_x.to_numpy().T, instances_df.xyz_a_y.to_numpy().T, instances_df.xyz_a_z.to_numpy().T)).T
    r_b = np.vstack(
        (instances_df.xyz_b_x.to_numpy().T, instances_df.xyz_b_y.to_numpy().T, instances_df.xyz_b_z.to_numpy().T)).T

    instances_df = instances_df.assign(r_a=pd.Series(np.vsplit(r_a, len(r_a))))
    instances_df = instances_df.assign(r_b=pd.Series(np.vsplit(r_b, len(r_b))))
    instances_df = instances_df.drop(columns=['xyz_a_x', 'xyz_a_y', 'xyz_a_z', 'xyz_b_x', 'xyz_b_y', 'xyz_b_z'])

    # Group by strand
    grouped_df = instances_df.groupby('StrandName')

    pp = []
    passes_df = pd.DataFrame()

    pass_id = 0

    with tqdm(total=len(grouped_df), desc='Group') as group_bar:
        for name, group in grouped_df:

            p = {"ID": pass_id,
                 "StartTof": group.iloc[0].tof,
                 "StopTof": group.iloc[0].tof,
                 "ttof": [group.iloc[0].tof],
                 "rr_a": [group.iloc[0].r_a],
                 "rr_b": [group.iloc[0].r_b],
                 'StrandName': group.iloc[0].StrandName}

            for index, instance in group.iterrows():

                last_tof = p["StopTof"]  # last found tof in this pass
                instance_tof = instance.tof  # tof of this new instance

                if instance_tof - last_tof <= 1.5 * step_duration:  # check if this step is part of the previous period
                    # if so the stop time is extended
                    p["StopTof"] = instance_tof

                    # and the tof and xyz coordinates are added to the lists
                    p["ttof"].append(instance_tof)
                    p["rr_a"].append(instance.r_a)
                    p["rr_b"].append(instance.r_b)

                else:
                    # if not then a pass was finished, add it to the passes_df and continue
                    p["ttof"] = [np.hstack(p["ttof"])]
                    p["rr_a"] = [np.vstack(p["rr_a"])]
                    p["rr_b"] = [np.vstack(p["rr_b"])]

                    pass_df = pd.DataFrame.from_dict(p)
                    pass_df.set_index("ID")
                    pp.append(pass_df)

                    pass_id = pass_id + 1  # increment pass id

                    # initialize the next pass df
                    p = {"ID": pass_id,
                         "StartTof": instance.tof,
                         "StopTof": instance.tof,
                         "ttof": [instance.tof],
                         "rr_a": [instance.r_a],
                         "rr_b": [instance.r_b],
                         'StrandName': instance.StrandName}

            group_bar.update(1)

    # Finalize the last row
    pass_df = pd.DataFrame.from_dict(p)
    pass_df.set_index("ID")
    pp.append(pass_df)

    # Create dataframe of passes
    passes_df = pd.concat(pp)
    passes_df.set_index("ID")
    # Calculate total duration of passes
    passes_df['Duration'] = passes_df['StopTof'] - passes_df['StartTof']

    return passes_df
