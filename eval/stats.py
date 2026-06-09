#!/usr/bin/env python3

# Copyright 2024 Jacob Hartzer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
A collection of functions for calculating statistics from the multi-IMU, multi-Camera simulation.

Typical usage is:
```
python3 eval/stats.py config/example.yaml
```

To get help:
```
python3 eval/stats.py --help
```
"""

import os

import h5py
from input_parser import InputParser
import numpy as np
from utilities import (calculate_rotation_errors, find_and_read_data_frames,
                       generate_mc_lists, interpolate_error,
                       interpolate_quat_error, lists_to_rot)


def RMSE_from_vectors(x_list, y_list, z_list):
    """Calculate the root mean square errors from list of vector elements."""
    x_err = np.array(x_list)
    y_err = np.array(y_list)
    z_err = np.array(z_list)
    rmse = np.sqrt(np.mean(x_err * x_err + y_err * y_err + z_err * z_err))
    return rmse


def body_err_pos(body_state_dfs, body_truth_dfs):
    """Calculate the body state position error."""
    RMSE_list = []
    for body_state, body_truth in zip(body_state_dfs, body_truth_dfs):
        true_time = body_truth['time'].to_list()
        true_pos_0 = body_truth['body_pos_0'].to_list()
        true_pos_1 = body_truth['body_pos_1'].to_list()
        true_pos_2 = body_truth['body_pos_2'].to_list()

        est_time = body_state['time'].to_list()
        est_pos_0 = body_state['body_pos_0'].to_list()
        est_pos_1 = body_state['body_pos_1'].to_list()
        est_pos_2 = body_state['body_pos_2'].to_list()

        err_pos_0 = interpolate_error(true_time, true_pos_0, est_time, est_pos_0)
        err_pos_1 = interpolate_error(true_time, true_pos_1, est_time, est_pos_1)
        err_pos_2 = interpolate_error(true_time, true_pos_2, est_time, est_pos_2)
        RMSE_list.append(RMSE_from_vectors(err_pos_0, err_pos_1, err_pos_2))
    return RMSE_list


def body_err_vel(body_state_dfs, body_truth_dfs):
    """Calculate the body state velocity error."""
    RMSE_list = []
    for body_state, body_truth in zip(body_state_dfs, body_truth_dfs):
        true_time = body_truth['time'].to_list()
        true_vel_0 = body_truth['body_vel_0'].to_list()
        true_vel_1 = body_truth['body_vel_1'].to_list()
        true_vel_2 = body_truth['body_vel_2'].to_list()

        est_time = body_state['time'].to_list()
        est_vel_0 = body_state['body_vel_0'].to_list()
        est_vel_1 = body_state['body_vel_1'].to_list()
        est_vel_2 = body_state['body_vel_2'].to_list()

        err_vel_0 = interpolate_error(true_time, true_vel_0, est_time, est_vel_0)
        err_vel_1 = interpolate_error(true_time, true_vel_1, est_time, est_vel_1)
        err_vel_2 = interpolate_error(true_time, true_vel_2, est_time, est_vel_2)
        RMSE_list.append(RMSE_from_vectors(err_vel_0, err_vel_1, err_vel_2))
    return RMSE_list


def body_err_acc(body_state_dfs, body_truth_dfs):
    """Calculate the body state acceleration error."""
    RMSE_list = []
    for body_state, body_truth in zip(body_state_dfs, body_truth_dfs):
        true_time = body_truth['time'].to_list()
        true_acc_0 = body_truth['body_acc_0'].to_list()
        true_acc_1 = body_truth['body_acc_1'].to_list()
        true_acc_2 = body_truth['body_acc_2'].to_list()

        est_time = body_state['time'].to_list()
        est_acc_0 = body_state['body_acc_0'].to_list()
        est_acc_1 = body_state['body_acc_1'].to_list()
        est_acc_2 = body_state['body_acc_2'].to_list()

        err_acc_0 = interpolate_error(true_time, true_acc_0, est_time, est_acc_0)
        err_acc_1 = interpolate_error(true_time, true_acc_1, est_time, est_acc_1)
        err_acc_2 = interpolate_error(true_time, true_acc_2, est_time, est_acc_2)
        RMSE_list.append(RMSE_from_vectors(err_acc_0, err_acc_1, err_acc_2))
    return RMSE_list


def body_err_ang_pos(body_state_dfs, body_truth_dfs):
    """Calculate the body state angular error."""
    # TODO: Calculate quaternion error
    RMSE_list = []
    for body_state, body_truth in zip(body_state_dfs, body_truth_dfs):
        true_time = body_truth['time'].to_list()
        true_ang_pos_0 = body_truth['body_ang_pos_0'].to_list()
        true_ang_pos_1 = body_truth['body_ang_pos_1'].to_list()
        true_ang_pos_2 = body_truth['body_ang_pos_2'].to_list()
        # true_ang_pos_3 = body_truth['body_ang_pos_3'].to_list()

        est_time = body_state['time'].to_list()
        est_ang_pos_0 = body_state['body_ang_pos_0'].to_list()
        est_ang_pos_1 = body_state['body_ang_pos_1'].to_list()
        est_ang_pos_2 = body_state['body_ang_pos_2'].to_list()
        # est_ang_pos_3 = body_state['body_ang_pos_3'].to_list()

        err_ang_vel_0 = interpolate_error(true_time, true_ang_pos_0, est_time, est_ang_pos_0)
        err_ang_vel_1 = interpolate_error(true_time, true_ang_pos_1, est_time, est_ang_pos_1)
        err_ang_vel_2 = interpolate_error(true_time, true_ang_pos_2, est_time, est_ang_pos_2)
        RMSE_list.append(RMSE_from_vectors(err_ang_vel_0, err_ang_vel_1, err_ang_vel_2))
    return RMSE_list


def body_err_ang_vel(body_state_dfs, body_truth_dfs):
    """Calculate the body state angular velocity error."""
    RMSE_list = []
    for body_state, body_truth in zip(body_state_dfs, body_truth_dfs):
        true_time = body_truth['time'].to_list()
        true_ang_vel_0 = body_truth['body_ang_vel_0'].to_list()
        true_ang_vel_1 = body_truth['body_ang_vel_1'].to_list()
        true_ang_vel_2 = body_truth['body_ang_vel_2'].to_list()

        est_time = body_state['time'].to_list()
        est_ang_vel_0 = body_state['body_ang_vel_0'].to_list()
        est_ang_vel_1 = body_state['body_ang_vel_1'].to_list()
        est_ang_vel_2 = body_state['body_ang_vel_2'].to_list()

        err_ang_vel_0 = interpolate_error(true_time, true_ang_vel_0, est_time, est_ang_vel_0)
        err_ang_vel_1 = interpolate_error(true_time, true_ang_vel_1, est_time, est_ang_vel_1)
        err_ang_vel_2 = interpolate_error(true_time, true_ang_vel_2, est_time, est_ang_vel_2)
        RMSE_list.append(RMSE_from_vectors(err_ang_vel_0, err_ang_vel_1, err_ang_vel_2))
    return RMSE_list


def body_err_ang_acc(body_state_dfs, body_truth_dfs):
    """Calculate the body state angular acceleration error."""
    RMSE_list = []
    for body_state, body_truth in zip(body_state_dfs, body_truth_dfs):
        true_time = body_truth['time'].to_list()
        true_ang_acc_0 = body_truth['body_ang_acc_0'].to_list()
        true_ang_acc_1 = body_truth['body_ang_acc_1'].to_list()
        true_ang_acc_2 = body_truth['body_ang_acc_2'].to_list()

        est_time = body_state['time'].to_list()
        est_ang_acc_0 = body_state['body_ang_acc_0'].to_list()
        est_ang_acc_1 = body_state['body_ang_acc_1'].to_list()
        est_ang_acc_2 = body_state['body_ang_acc_2'].to_list()

        err_ang_acc_0 = interpolate_error(true_time, true_ang_acc_0, est_time, est_ang_acc_0)
        err_ang_acc_1 = interpolate_error(true_time, true_ang_acc_1, est_time, est_ang_acc_1)
        err_ang_acc_2 = interpolate_error(true_time, true_ang_acc_2, est_time, est_ang_acc_2)
        RMSE_list.append(RMSE_from_vectors(err_ang_acc_0, err_ang_acc_1, err_ang_acc_2))
    return RMSE_list


def body_states(body_state_dfs):
    """Calculate the body state position error."""
    state_sizes = []
    for body_state in body_state_dfs:
        state_sizes.append(np.mean(body_state['state_size'].to_list()))
    return state_sizes


def sensor_err_pos(sensor_dfs, body_truth_dfs_dict, prefix):
    """Calculate the sensor position error."""
    RMSE_list = []
    for sensor_state, body_truth in zip(sensor_dfs, body_truth_dfs_dict):
        sensor_id = sensor_state.attrs['id']
        true_time = body_truth['time'].to_list()
        true_pos_0 = body_truth[f'{prefix}_pos_{sensor_id}_0'].to_list()
        true_pos_1 = body_truth[f'{prefix}_pos_{sensor_id}_1'].to_list()
        true_pos_2 = body_truth[f'{prefix}_pos_{sensor_id}_2'].to_list()

        est_time = sensor_state['time'].to_list()
        est_pos_0 = sensor_state[f'{prefix}_pos_0'].to_list()
        est_pos_1 = sensor_state[f'{prefix}_pos_1'].to_list()
        est_pos_2 = sensor_state[f'{prefix}_pos_2'].to_list()

        err_pos_0 = interpolate_error(true_time, true_pos_0, est_time, est_pos_0)
        err_pos_1 = interpolate_error(true_time, true_pos_1, est_time, est_pos_1)
        err_pos_2 = interpolate_error(true_time, true_pos_2, est_time, est_pos_2)
        RMSE_list.append(RMSE_from_vectors(err_pos_0, err_pos_1, err_pos_2))
    return RMSE_list


def sensor_err_ang(sensor_dfs, body_truth_dfs_dict, prefix):
    """Calculate the sensor angular position error."""
    RMSE_list = []
    for sensor_state, body_truth in zip(sensor_dfs, body_truth_dfs_dict):
        sensor_id = sensor_state.attrs['id']
        true_time = body_truth['time'].to_list()
        true_ang_vel_0 = body_truth[f'{prefix}_ang_pos_{sensor_id}_0'].to_list()
        true_ang_vel_1 = body_truth[f'{prefix}_ang_pos_{sensor_id}_1'].to_list()
        true_ang_vel_2 = body_truth[f'{prefix}_ang_pos_{sensor_id}_2'].to_list()

        est_time = sensor_state['time'].to_list()
        est_ang_vel_0 = sensor_state[f'{prefix}_ang_pos_0'].to_list()
        est_ang_vel_1 = sensor_state[f'{prefix}_ang_pos_1'].to_list()
        est_ang_vel_2 = sensor_state[f'{prefix}_ang_pos_2'].to_list()

        err_ang_vel_0 = interpolate_error(true_time, true_ang_vel_0, est_time, est_ang_vel_0)
        err_ang_vel_1 = interpolate_error(true_time, true_ang_vel_1, est_time, est_ang_vel_1)
        err_ang_vel_2 = interpolate_error(true_time, true_ang_vel_2, est_time, est_ang_vel_2)
        RMSE_list.append(RMSE_from_vectors(err_ang_vel_0, err_ang_vel_1, err_ang_vel_2))
    return RMSE_list


def imu_err_bias(imu_dfs, body_truth_dfs_dict, bias_type):
    """Calculate the imu bias error."""
    RMSE_list = []
    for imu_state, body_truth in zip(imu_dfs, body_truth_dfs_dict):
        sensor_id = imu_state.attrs['id']
        true_time = body_truth['time'].to_list()
        true_bias_0 = body_truth[f'imu_{bias_type}_bias_{sensor_id}_0'].to_list()
        true_bias_1 = body_truth[f'imu_{bias_type}_bias_{sensor_id}_1'].to_list()
        true_bias_2 = body_truth[f'imu_{bias_type}_bias_{sensor_id}_2'].to_list()

        est_time = imu_state['time'].to_list()
        est_bias_0 = imu_state[f'imu_{bias_type}_bias_0'].to_list()
        est_bias_1 = imu_state[f'imu_{bias_type}_bias_1'].to_list()
        est_bias_2 = imu_state[f'imu_{bias_type}_bias_2'].to_list()

        err_bias_0 = interpolate_error(true_time, true_bias_0, est_time, est_bias_0)
        err_bias_1 = interpolate_error(true_time, true_bias_1, est_time, est_bias_1)
        err_bias_2 = interpolate_error(true_time, true_bias_2, est_time, est_bias_2)
        RMSE_list.append(RMSE_from_vectors(err_bias_0, err_bias_1, err_bias_2))
    return RMSE_list


def imu_duration(imu_dfs):
    """Calculate the imu bias error."""
    mean_list = []
    for imu_state in imu_dfs:
        durations = np.array(imu_state['duration_0'].to_list())
        mean_list.append(np.mean(durations))
    return mean_list


def gps_err_pos(gps_dfs, body_truth_dfs):
    """Calculate the gps position error."""
    RMSE_list = []
    for gps_state, body_truth in zip(gps_dfs, body_truth_dfs):
        true_lat = body_truth['ref_lat'][0]
        true_lon = body_truth['ref_lon'][0]
        true_alt = body_truth['ref_alt'][0]

        index = next((i + 1 for i, x in enumerate(gps_state['is_initialized']) if x), None)
        if index and index < len(gps_state['is_initialized']):
            est_lat = gps_state['ref_lat'][index]
            est_lon = gps_state['ref_lon'][index]
            est_alt = gps_state['ref_alt'][index]

            err_lat = est_lat - true_lat
            err_lon = est_lon - true_lon
            err_alt = est_alt - true_alt
            RMSE_list.append(RMSE_from_vectors([err_lat], [err_lon], [err_alt]))
    return RMSE_list


def gps_err_ang(gps_dfs, body_truth_dfs):
    """Calculate the gps angular position error."""
    error_list = []
    for gps_state, body_truth in zip(gps_dfs, body_truth_dfs):
        true_hdg = body_truth['ref_heading'][0]

        index = next((i + 1 for i, x in enumerate(gps_state['is_initialized']) if x), None)
        if index and index < len(gps_state['is_initialized']):
            est_hdg = gps_state['ref_heading'][index]
            error_list.append(est_hdg - true_hdg)

    return error_list


def gps_init_count(gps_dfs):
    """Calculate the gps initialization count."""
    time_list = []
    for gps_state in gps_dfs:
        time_list.append(np.sum(gps_state['is_initialized'] == 0))
    return time_list


def write_summary(directory, stats):
    """Write the error summary statistics to a file."""
    with open(os.path.join(directory, 'stats.txt'), 'w') as f:
        f.write('Statistic,RMSE-Mean,RMSE-StdDev\n')
        for key in stats:
            vals = np.array(stats[key])
            mu = np.mean(vals)
            sig = np.std(vals)
            f.write('{}: {:0.4f}, {:0.4f}\n'.format(key, mu, sig))
    return


def save_errors_to_hdf5(data_dirs, body_state_dfs_dict, body_truth_dfs_dict,
                        imu_dfs_dict, mskcf_dfs_dict, gps_dfs_dict):
    """Calculate and write error time series back to the HDF5 file."""
    # Find the single merged HDF5 file in the parent directory
    single_h5_path = None
    if data_dirs:
        parent_dir = os.path.dirname(data_dirs[0].rstrip(os.sep))
        grandparent_dir = os.path.dirname(parent_dir)
        if os.path.exists(grandparent_dir):
            h5_files = [f for f in os.listdir(grandparent_dir) if f.endswith('.h5')]
            if h5_files:
                single_h5_path = os.path.join(grandparent_dir, h5_files[0])

    if not single_h5_path or not os.path.exists(single_h5_path):
        return

    try:
        with h5py.File(single_h5_path, 'a') as f:
            for i, data_dir in enumerate(data_dirs):
                run_name = os.path.basename(data_dir.rstrip(os.sep))
                import re
                matches = re.findall(r'[0-9]+$', run_name)
                group_name = f'run_{int(matches[-1])}' if matches else run_name

                if group_name not in f:
                    continue
                run_group = f[group_name]
                errors_group = run_group.require_group('errors')

                # 1. Body Errors
                if 0 in body_state_dfs_dict and 0 in body_truth_dfs_dict:
                    body_state = body_state_dfs_dict[0][i]
                    body_truth = body_truth_dfs_dict[0][i]
                    true_time = body_truth['time'].to_list()
                    est_time = body_state['time'].to_list()

                    # Helper to stack and write a 3D error dataset
                    def write_3d_error(name, true_cols, est_cols):
                        errs = [
                            interpolate_error(
                                true_time, body_truth[t_col].to_list(),
                                est_time, body_state[e_col].to_list()
                            )
                            for t_col, e_col in zip(true_cols, est_cols)
                        ]
                        data = np.column_stack((est_time, errs[0], errs[1], errs[2]))
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(name, data=data)
                        ds.attrs['column_names'] = 'time,x,y,z'

                    write_3d_error(
                        'body_pos_err',
                        ['body_pos_0', 'body_pos_1', 'body_pos_2'],
                        ['body_pos_0', 'body_pos_1', 'body_pos_2']
                    )
                    write_3d_error(
                        'body_vel_err',
                        ['body_vel_0', 'body_vel_1', 'body_vel_2'],
                        ['body_vel_0', 'body_vel_1', 'body_vel_2']
                    )
                    write_3d_error(
                        'body_acc_err',
                        ['body_acc_0', 'body_acc_1', 'body_acc_2'],
                        ['body_acc_0', 'body_acc_1', 'body_acc_2']
                    )
                    write_3d_error(
                        'body_ang_vel_err',
                        ['body_ang_vel_0', 'body_ang_vel_1', 'body_ang_vel_2'],
                        ['body_ang_vel_0', 'body_ang_vel_1', 'body_ang_vel_2']
                    )
                    write_3d_error(
                        'body_ang_acc_err',
                        ['body_ang_acc_0', 'body_ang_acc_1', 'body_ang_acc_2'],
                        ['body_ang_acc_0', 'body_ang_acc_1', 'body_ang_acc_2']
                    )

                    # Body Angular Error (Quaternion/rotation error in radians)
                    true_w = body_truth['body_ang_pos_0'].to_list()
                    true_x = body_truth['body_ang_pos_1'].to_list()
                    true_y = body_truth['body_ang_pos_2'].to_list()
                    true_z = body_truth['body_ang_pos_3'].to_list()

                    est_w = body_state['body_ang_pos_0'].to_list()
                    est_x = body_state['body_ang_pos_1'].to_list()
                    est_y = body_state['body_ang_pos_2'].to_list()
                    est_z = body_state['body_ang_pos_3'].to_list()

                    interp_w = np.interp(est_time, true_time, true_w)
                    interp_x = np.interp(est_time, true_time, true_x)
                    interp_y = np.interp(est_time, true_time, true_y)
                    interp_z = np.interp(est_time, true_time, true_z)

                    interp_r = lists_to_rot(interp_w, interp_x, interp_y, interp_z)
                    est_ang_pos_r = lists_to_rot(est_w, est_x, est_y, est_z)

                    err_ax, err_ay, err_az = calculate_rotation_errors(est_ang_pos_r, interp_r)
                    data = np.column_stack((est_time, err_ax, err_ay, err_az))
                    name = 'body_ang_err'
                    if name in errors_group:
                        del errors_group[name]
                    ds = errors_group.create_dataset(name, data=data)
                    ds.attrs['column_names'] = 'time,x,y,z'

                # 2. IMU Errors
                body_truth = body_truth_dfs_dict[0][i]
                true_time = body_truth['time'].to_list()
                for sensor_id in imu_dfs_dict.keys():
                    imu_df = imu_dfs_dict[sensor_id][i]
                    est_time = imu_df['time'].to_list()

                    # IMU Position Error
                    if 'imu_pos_0' in imu_df:
                        true_pos_0 = body_truth[f'imu_pos_{sensor_id}_0'].to_list()
                        true_pos_1 = body_truth[f'imu_pos_{sensor_id}_1'].to_list()
                        true_pos_2 = body_truth[f'imu_pos_{sensor_id}_2'].to_list()
                        err_x = interpolate_error(
                            true_time, true_pos_0, est_time, imu_df['imu_pos_0'].to_list())
                        err_y = interpolate_error(
                            true_time, true_pos_1, est_time, imu_df['imu_pos_1'].to_list())
                        err_z = interpolate_error(
                            true_time, true_pos_2, est_time, imu_df['imu_pos_2'].to_list())
                        name = f'imu_pos_err_{sensor_id}'
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(
                            name, data=np.column_stack((est_time, err_x, err_y, err_z)))
                        ds.attrs['column_names'] = 'time,x,y,z'

                    # IMU Angular Error (in radians)
                    if 'imu_ang_pos_0' in imu_df:
                        true_w = body_truth[f'imu_ang_pos_{sensor_id}_0'].to_list()
                        true_x = body_truth[f'imu_ang_pos_{sensor_id}_1'].to_list()
                        true_y = body_truth[f'imu_ang_pos_{sensor_id}_2'].to_list()
                        true_z = body_truth[f'imu_ang_pos_{sensor_id}_3'].to_list()

                        est_w = imu_df['imu_ang_pos_0'].to_list()
                        est_x = imu_df['imu_ang_pos_1'].to_list()
                        est_y = imu_df['imu_ang_pos_2'].to_list()
                        est_z = imu_df['imu_ang_pos_3'].to_list()

                        err_ax, err_ay, err_az = interpolate_quat_error(
                            true_time, true_w, true_x, true_y, true_z,
                            est_time, est_w, est_x, est_y, est_z
                        )
                        # Convert to radians since interpolate_quat_error returns milliradians
                        err_ax = err_ax / 1e3
                        err_ay = err_ay / 1e3
                        err_az = err_az / 1e3
                        name = f'imu_ang_err_{sensor_id}'
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(
                            name, data=np.column_stack((est_time, err_ax, err_ay, err_az)))
                        ds.attrs['column_names'] = 'time,x,y,z'

                    # IMU Bias Errors
                    if 'imu_acc_bias_0' in imu_df:
                        true_acc_b0 = body_truth[f'imu_acc_bias_{sensor_id}_0'].to_list()
                        true_acc_b1 = body_truth[f'imu_acc_bias_{sensor_id}_1'].to_list()
                        true_acc_b2 = body_truth[f'imu_acc_bias_{sensor_id}_2'].to_list()
                        err_abx = interpolate_error(
                            true_time, true_acc_b0, est_time,
                            imu_df['imu_acc_bias_0'].to_list())
                        err_aby = interpolate_error(
                            true_time, true_acc_b1, est_time,
                            imu_df['imu_acc_bias_1'].to_list())
                        err_abz = interpolate_error(
                            true_time, true_acc_b2, est_time,
                            imu_df['imu_acc_bias_2'].to_list())
                        name = f'imu_acc_bias_err_{sensor_id}'
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(
                            name, data=np.column_stack((est_time, err_abx, err_aby, err_abz)))
                        ds.attrs['column_names'] = 'time,x,y,z'

                    if 'imu_gyr_bias_0' in imu_df:
                        true_gyr_b0 = body_truth[f'imu_gyr_bias_{sensor_id}_0'].to_list()
                        true_gyr_b1 = body_truth[f'imu_gyr_bias_{sensor_id}_1'].to_list()
                        true_gyr_b2 = body_truth[f'imu_gyr_bias_{sensor_id}_2'].to_list()
                        err_gbx = interpolate_error(
                            true_time, true_gyr_b0, est_time,
                            imu_df['imu_gyr_bias_0'].to_list())
                        err_gby = interpolate_error(
                            true_time, true_gyr_b1, est_time,
                            imu_df['imu_gyr_bias_1'].to_list())
                        err_gbz = interpolate_error(
                            true_time, true_gyr_b2, est_time,
                            imu_df['imu_gyr_bias_2'].to_list())
                        name = f'imu_gyr_bias_err_{sensor_id}'
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(
                            name, data=np.column_stack((est_time, err_gbx, err_gby, err_gbz)))
                        ds.attrs['column_names'] = 'time,x,y,z'

                # 3. MSCKF Errors
                for sensor_id in mskcf_dfs_dict.keys():
                    cam_df = mskcf_dfs_dict[sensor_id][i]
                    est_time = cam_df['time'].to_list()

                    # Camera Position Error
                    if 'cam_pos_0' in cam_df:
                        true_pos_0 = body_truth[f'cam_pos_{sensor_id}_0'].to_list()
                        true_pos_1 = body_truth[f'cam_pos_{sensor_id}_1'].to_list()
                        true_pos_2 = body_truth[f'cam_pos_{sensor_id}_2'].to_list()
                        err_x = interpolate_error(
                            true_time, true_pos_0, est_time, cam_df['cam_pos_0'].to_list())
                        err_y = interpolate_error(
                            true_time, true_pos_1, est_time, cam_df['cam_pos_1'].to_list())
                        err_z = interpolate_error(
                            true_time, true_pos_2, est_time, cam_df['cam_pos_2'].to_list())
                        name = f'cam_pos_err_{sensor_id}'
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(
                            name, data=np.column_stack((est_time, err_x, err_y, err_z)))
                        ds.attrs['column_names'] = 'time,x,y,z'

                    # Camera Angular Error (in radians)
                    if 'cam_ang_pos_0' in cam_df:
                        true_w = body_truth[f'cam_ang_pos_{sensor_id}_0'].to_list()
                        true_x = body_truth[f'cam_ang_pos_{sensor_id}_1'].to_list()
                        true_y = body_truth[f'cam_ang_pos_{sensor_id}_2'].to_list()
                        true_z = body_truth[f'cam_ang_pos_{sensor_id}_3'].to_list()

                        est_w = cam_df['cam_ang_pos_0'].to_list()
                        est_x = cam_df['cam_ang_pos_1'].to_list()
                        est_y = cam_df['cam_ang_pos_2'].to_list()
                        est_z = cam_df['cam_ang_pos_3'].to_list()

                        err_ax, err_ay, err_az = interpolate_quat_error(
                            true_time, true_w, true_x, true_y, true_z,
                            est_time, est_w, est_x, est_y, est_z
                        )
                        # Convert to radians since interpolate_quat_error returns milliradians
                        err_ax = err_ax / 1e3
                        err_ay = err_ay / 1e3
                        err_az = err_az / 1e3
                        name = f'cam_ang_err_{sensor_id}'
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(
                            name, data=np.column_stack((est_time, err_ax, err_ay, err_az)))
                        ds.attrs['column_names'] = 'time,x,y,z'

                # 4. GPS Errors
                for sensor_id in gps_dfs_dict.keys():
                    gps_df = gps_dfs_dict[sensor_id][i]

                    # GPS Errors
                    if 'ant_pos_0' in gps_df:
                        est_time = gps_df['time'].to_list()
                        true_pos_0 = body_truth[f'gps_pos_{sensor_id}_0'].to_list()
                        true_pos_1 = body_truth[f'gps_pos_{sensor_id}_1'].to_list()
                        true_pos_2 = body_truth[f'gps_pos_{sensor_id}_2'].to_list()
                        err_x = interpolate_error(
                            true_time, true_pos_0, est_time, gps_df['ant_pos_0'].to_list())
                        err_y = interpolate_error(
                            true_time, true_pos_1, est_time, gps_df['ant_pos_1'].to_list())
                        err_z = interpolate_error(
                            true_time, true_pos_2, est_time, gps_df['ant_pos_2'].to_list())
                        name = f'gps_pos_err_{sensor_id}'
                        if name in errors_group:
                            del errors_group[name]
                        ds = errors_group.create_dataset(
                            name, data=np.column_stack((est_time, err_x, err_y, err_z)))
                        ds.attrs['column_names'] = 'time,x,y,z'

    except Exception as e:
        print(f'Warning: Failed to save error time series: {e}')


def save_stats_to_hdf5(data_dirs, stats):
    """Save calculated RMSE statistics back to each run's HDF5 file."""
    # Try to find a single/merged HDF5 file in the parent directory
    single_h5_path = None
    if data_dirs:
        parent_dir = os.path.dirname(data_dirs[0].rstrip(os.sep))
        grandparent_dir = os.path.dirname(parent_dir)
        if os.path.exists(grandparent_dir):
            h5_files = [f for f in os.listdir(grandparent_dir) if f.endswith('.h5')]
            if h5_files:
                single_h5_path = os.path.join(grandparent_dir, h5_files[0])

    if single_h5_path and os.path.exists(single_h5_path):
        try:
            with h5py.File(single_h5_path, 'a') as f:
                for i, data_dir in enumerate(data_dirs):
                    run_name = os.path.basename(data_dir.rstrip(os.sep))
                    import re
                    matches = re.findall(r'[0-9]+$', run_name)
                    if matches:
                        group_name = f'run_{int(matches[-1])}'
                    else:
                        group_name = run_name

                    if group_name in f:
                        run_group = f[group_name]
                        stats_group = run_group.require_group('statistics')
                        for key, val_list in stats.items():
                            if isinstance(val_list, list) and len(val_list) == len(data_dirs):
                                val = val_list[i]
                                if key in stats_group:
                                    del stats_group[key]
                                stats_group.create_dataset(key, data=val)
        except Exception as e:
            print(f'Warning: Failed to save statistics to {single_h5_path}: {e}')
    else:
        # Fallback to individual files
        for i, data_dir in enumerate(data_dirs):
            if os.path.exists(data_dir):
                h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
            else:
                h5_files = []
            if h5_files:
                h5_path = os.path.join(data_dir, h5_files[0])
            else:
                h5_path = os.path.join(data_dir, 'simulation_data.h5')
            if not os.path.exists(h5_path):
                continue
            try:
                with h5py.File(h5_path, 'a') as f:
                    stats_group = f.require_group('statistics')
                    for key, val_list in stats.items():
                        if isinstance(val_list, list) and len(val_list) == len(data_dirs):
                            val = val_list[i]
                            if key in stats_group:
                                del stats_group[key]
                            stats_group.create_dataset(key, data=val)
            except Exception as e:
                print(f'Warning: Failed to save statistics to {h5_path}: {e}')


# TODO(jhartzer): Split for loop into thread pool
def calc_sim_stats(config_sets, args):
    """Top level function to plot simulation results from sets of config files."""
    for config_set in config_sets:

        data_dirs = [config.split('.yaml')[0] for config in config_set]
        if len(config_set) > 1:
            stat_dir = os.path.dirname(os.path.dirname(config_set[0]))
        else:
            stat_dir = data_dirs[0]

        stats = {}
        body_state_dfs_dict = find_and_read_data_frames(data_dirs, 'body_state')
        body_truth_dfs_dict = find_and_read_data_frames(data_dirs, 'body_truth')
        for key in body_state_dfs_dict:
            body_state_dfs = body_state_dfs_dict[key]
            body_truth_dfs = body_truth_dfs_dict[key]
            stats[f'body_{key}_err_pos'] = body_err_pos(body_state_dfs, body_truth_dfs)
            stats[f'body_{key}_err_vel'] = body_err_vel(body_state_dfs, body_truth_dfs)
            stats[f'body_{key}_err_acc'] = body_err_acc(body_state_dfs, body_truth_dfs)
            stats[f'body_{key}_err_ang_pos'] = body_err_ang_pos(body_state_dfs, body_truth_dfs)
            stats[f'body_{key}_err_ang_vel'] = body_err_ang_vel(body_state_dfs, body_truth_dfs)
            stats[f'body_{key}_err_ang_acc'] = body_err_ang_acc(body_state_dfs, body_truth_dfs)
            stats[f'body_{key}_states'] = body_states(body_state_dfs)

        imu_dfs_dict = find_and_read_data_frames(data_dirs, 'imu')
        body_truth_dfs = body_truth_dfs_dict[0]
        for key in sorted(imu_dfs_dict.keys()):
            imu_dfs = imu_dfs_dict[key]
            stats[f'imu_{key}_err_pos'] = sensor_err_pos(imu_dfs, body_truth_dfs, 'imu')
            stats[f'imu_{key}_err_ang'] = sensor_err_ang(imu_dfs, body_truth_dfs, 'imu')
            stats[f'imu_{key}_err_acc_bias'] = imu_err_bias(imu_dfs, body_truth_dfs, 'acc')
            stats[f'imu_{key}_err_gyr_bias'] = imu_err_bias(imu_dfs, body_truth_dfs, 'gyr')
            stats[f'imu_{key}_duration'] = imu_duration(imu_dfs)

        mskcf_dfs_dict = find_and_read_data_frames(data_dirs, 'msckf')
        for key in sorted(mskcf_dfs_dict.keys()):
            mskcf_dfs = mskcf_dfs_dict[key]
            stats[f'mskcf_{key}_err_pos'] = sensor_err_pos(mskcf_dfs, body_truth_dfs, 'cam')
            stats[f'mskcf_{key}_err_ang'] = sensor_err_ang(mskcf_dfs, body_truth_dfs, 'cam')

        fiducial_dfs_dict = find_and_read_data_frames(data_dirs, 'fiducial')
        for key in sorted(fiducial_dfs_dict.keys()):
            pass  # TODO(jhartzer): Use a separate function to read board static positions
            # fiducial_dfs = fiducial_dfs_dict[key]
            # stats[f'fiducial_{key}_err_pos'] = fid_err_pos(fiducial_dfs, body_truth_dfs, 'fid')
            # stats[f'fiducial_{key}_err_ang'] = fid_err_ang(fiducial_dfs, body_truth_dfs, 'fid')

        gps_dfs_dict = find_and_read_data_frames(data_dirs, 'gps')
        for key in sorted(gps_dfs_dict.keys()):
            gps_dfs = gps_dfs_dict[key]
            stats[f'gps_{key}_err_init_pos'] = gps_err_pos(gps_dfs, body_truth_dfs)
            stats[f'gps_{key}_err_init_ang'] = gps_err_ang(gps_dfs, body_truth_dfs)
            stats[f'gps_{key}_init_count'] = gps_init_count(gps_dfs)

        save_errors_to_hdf5(
            data_dirs, body_state_dfs_dict, body_truth_dfs_dict,
            imu_dfs_dict, mskcf_dfs_dict, gps_dfs_dict
        )
        save_stats_to_hdf5(data_dirs, stats)
        write_summary(stat_dir, stats)


# TODO(jhartzer): Write tests
# TODO(jhartzer): Compress functions into vector and quaternion errors
if __name__ == '__main__':
    parser = InputParser()
    args = parser.parse_args()

    config_files = generate_mc_lists(args)
    calc_sim_stats(config_files, args)
