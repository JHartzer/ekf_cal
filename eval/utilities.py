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

import collections
import glob
import math
import os
import re

from bokeh.plotting import figure
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml


def get_colors(args):
    """Get color palette based on the theme setting."""
    if args.light:
        return ['red', 'green', 'blue', 'magenta']
    else:
        return ['cyan', 'yellow', 'magenta', 'white']


def calculate_alpha(line_count: int):
    """Calculate transparency value from number of plots."""
    alpha = 1.0 / math.pow(line_count, 0.5)
    return alpha


def interpolate_error(true_t, true_x, estimate_t, estimate_x):
    """Calculate an interpolated error using truth and estimate points."""
    interp_x = np.interp(estimate_t, true_t, true_x)
    errors = [estimate - interp for estimate, interp in zip(estimate_x, interp_x)]
    return np.array(errors)


def interpolate_quat_error(
        true_t,
        true_w,
        true_x,
        true_y,
        true_z,
        estimate_t,
        estimate_w,
        estimate_x,
        estimate_y,
        estimate_z):
    """Calculate an interpolated quaternion error using truth and estimates."""
    interp_w = np.interp(estimate_t, true_t, true_w)
    interp_x = np.interp(estimate_t, true_t, true_x)
    interp_y = np.interp(estimate_t, true_t, true_y)
    interp_z = np.interp(estimate_t, true_t, true_z)

    err_x = []
    err_y = []
    err_z = []
    for i in range(len(interp_w)):
        iw = interp_w[i]
        ix = interp_x[i]
        iy = interp_y[i]
        iz = interp_z[i]
        ew = estimate_w[i]
        ex = estimate_x[i]
        ey = estimate_y[i]
        ez = estimate_z[i]
        qi = Rotation.from_quat([iw, ix, iy, iz], scalar_first=True)
        qe = Rotation.from_quat([ew, ex, ey, ez], scalar_first=True)
        q_err = qi * qe.inv()
        error_euler = q_err.as_euler('XYZ')
        err_x.append(error_euler[0] * 1e3)
        err_y.append(error_euler[1] * 1e3)
        err_z.append(error_euler[2] * 1e3)

    return np.array(err_x), np.array(err_y), np.array(err_z)


def lists_to_rot(w_list, x_list, y_list, z_list):
    """Convert lists of quaternion elements to a list of scipy rotations."""
    r_list = []
    for w, x, y, z in zip(w_list, x_list, y_list, z_list):
        r = Rotation.from_quat([w, x, y, z], scalar_first=True)
        r_list.append(r)
    return r_list


def calculate_rotation_errors(estimate, truth):
    """Calculate errors from two lists of quaternions."""
    x = []
    y = []
    z = []
    for est, true in zip(estimate, truth):
        rotation = est * true.inv()
        error_eul = rotation.as_euler('XYZ')
        x.append(error_eul[0])
        y.append(error_eul[1])
        z.append(error_eul[2])

    return np.array(x), np.array(y), np.array(z)


def RMSE_from_vectors(x_list, y_list, z_list):
    """Calculate the root mean square errors from list of vector elements."""
    x_err = np.array(x_list)
    y_err = np.array(y_list)
    z_err = np.array(z_list)
    rmse = np.sqrt(np.mean(x_err * x_err + y_err * y_err + z_err * z_err))
    return rmse


def plot_update_timing(data_frames, rate=None):
    """Plot histogram of update execution durations."""
    df_prefix = data_frames[0].attrs['prefix']
    df_id = str(data_frames[0].attrs['id'])
    fig = figure(width=800, height=300, x_axis_label='time [us]',
                 y_axis_label='Count', title=f'{df_prefix} {df_id} Update Time')
    durations = np.array([])
    for df in data_frames:
        durations = np.append(durations, df['duration_0'] / 1e3)
    hist, edges = np.histogram(durations)
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], legend_label='Duration [us]')
    if rate:
        pass
        # TODO(jhartzer): Add max duration line
        # axs.axvline(x=1000.0 / rate, color='red', linestyle='--')
    return fig


def format_prefix(prefix):
    """Generate formatted prefix from string."""
    if (prefix == 'imu'):
        return 'IMU'
    elif (prefix == 'camera'):
        return 'Camera'
    elif (prefix == 'body_state'):
        return 'Body'
    elif (prefix == 'fiducial'):
        return 'Fiducial'
    elif (prefix == 'msckf'):
        return 'MSCKF'
    else:
        return ''


def parse_yaml(config):
    """Collect sensor configuration data from input yaml."""
    config_data = {}
    config_data['imu_rates'] = {}
    config_data['camera_rates'] = {}
    with open(config, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            imu_list = yaml_dict['/EkfCalNode']['ros__parameters']['imu_list']
            cam_list = yaml_dict['/EkfCalNode']['ros__parameters']['camera_list']
            id_counter = 1
            if imu_list:
                imu_dict = yaml_dict['/EkfCalNode']['ros__parameters']['imu']
                for imu_name in imu_list:
                    config_data['imu_rates'][id_counter] = imu_dict[imu_name]['rate']
                    id_counter += 1
            if cam_list:
                cam_dict = yaml_dict['/EkfCalNode']['ros__parameters']['camera']
                for cam_name in cam_list:
                    config_data['camera_rates'][id_counter] = cam_dict[cam_name]['rate']
                    id_counter += 1

        except yaml.YAMLError as exc:
            print(exc)

    return config_data


_h5_dataset_cache = {}


def get_matching_datasets(h5_path, run_group_name, prefix):
    mtime = os.path.getmtime(h5_path)
    cache_key = (h5_path, mtime)
    if cache_key not in _h5_dataset_cache:
        paths = []
        with h5py.File(h5_path, 'r') as f:
            def visit(name, node):
                if isinstance(node, h5py.Dataset):
                    paths.append(name)
            f.visititems(visit)
        _h5_dataset_cache[cache_key] = paths

    all_paths = _h5_dataset_cache[cache_key]
    matched_paths = []
    
    # If run_group_name is provided, filter datasets under that run group.
    # Otherwise, match truth/root level.
    prefix_to_match = f"{run_group_name}/" if run_group_name else ""
    
    for path in all_paths:
        if prefix_to_match:
            if not path.startswith(prefix_to_match):
                continue
            rel_path = path[len(prefix_to_match):]
        else:
            rel_path = path
            
        matched = False
        if prefix == 'body_truth':
            matched = (rel_path == 'truth/body')
        elif prefix == 'board_truth':
            matched = (rel_path == 'truth/board')
        elif prefix == 'feature_points':
            matched = (rel_path == 'truth/feature_points')
        else:
            base = os.path.basename(rel_path)
            matched = (
                base == prefix or
                re.match(rf'^{prefix}_[0-9]+$', base)
            )
        if matched:
            matched_paths.append(path)
            
    matched_paths.sort()
    return matched_paths


def find_and_read_data_frames(directories, prefix):
    """
    Find matching dataframes and read using pandas.

    Supports single HDF5, run-specific HDF5, and CSV.
    """
    data_frame_sets = collections.defaultdict(list)

    # 1. Try to find a single/merged HDF5 file in the parent directory
    single_h5_path = None
    if directories:
        # Get parent directory of runs (e.g. config/example/runs)
        parent_dir = os.path.dirname(directories[0].rstrip(os.sep))
        # Get grandparent directory (e.g. config/example)
        grandparent_dir = os.path.dirname(parent_dir)
        if os.path.exists(grandparent_dir):
            h5_files = [f for f in os.listdir(grandparent_dir) if f.endswith('.h5')]
            if h5_files:
                single_h5_path = os.path.join(grandparent_dir, h5_files[0])

    # If the single merged HDF5 file exists, read from it
    if single_h5_path and os.path.exists(single_h5_path):
        with h5py.File(single_h5_path, 'r') as f:
            if prefix in ['body_truth', 'board_truth', 'feature_points']:
                # Read truth directly from root /truth group
                ds_name_map = {
                    'body_truth': 'truth/body',
                    'board_truth': 'truth/board',
                    'feature_points': 'truth/feature_points'
                }
                ds_path = ds_name_map[prefix]
                if ds_path in f:
                    dataset = f[ds_path]
                    data = dataset[:]
                    cols_attr = dataset.attrs.get('column_names', '')
                    if isinstance(cols_attr, bytes):
                        cols_attr = cols_attr.decode('utf-8')
                    cols = cols_attr.split(',') if cols_attr else None

                    df = pd.DataFrame(data, columns=cols)
                    df.dropna(inplace=True)
                    df.attrs['prefix'] = format_prefix(prefix)
                    df.attrs['id'] = 0

                    # Replicate the truth dataset for each run directory
                    for _ in range(len(directories)):
                        data_frame_sets[0].append(df.copy())
            else:
                for directory in directories:
                    # Extract run ID from directory name
                    run_name = os.path.basename(directory.rstrip(os.sep))
                    matches = re.findall(r'[0-9]+$', run_name)
                    if matches:
                        run_group_name = f'run_{int(matches[-1])}'
                    else:
                        run_group_name = run_name

                    ds_paths = get_matching_datasets(single_h5_path, run_group_name, prefix)
                    for ds_path in ds_paths:
                        dataset = f[ds_path]
                        data = dataset[:]

                        cols_attr = dataset.attrs.get('column_names', '')
                        if isinstance(cols_attr, bytes):
                            cols_attr = cols_attr.decode('utf-8')
                        cols = cols_attr.split(',') if cols_attr else None

                        df = pd.DataFrame(data, columns=cols)
                        df.dropna(inplace=True)
                        df.attrs['prefix'] = format_prefix(prefix)

                        # Extract sensor ID (not run ID!)
                        base = os.path.basename(ds_path)
                        sensor_id_matches = re.findall(r'_[0-9]+$', base)
                        if sensor_id_matches:
                            sensor_id = int(sensor_id_matches[0].replace('_', ''))
                        else:
                            sensor_id = 0
                        df.attrs['id'] = sensor_id

                        data_frame_sets[sensor_id].append(df)

    else:
        # 2. Fallback to individual run HDF5 files or CSVs
        for directory in directories:
            # Find any .h5 file in the directory
            if os.path.exists(directory):
                h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
            else:
                h5_files = []
            if h5_files:
                h5_path = os.path.join(directory, h5_files[0])
            else:
                h5_path = os.path.join(directory, 'simulation_data.h5')

            if os.path.exists(h5_path):
                ds_paths = get_matching_datasets(h5_path, None, prefix)
                with h5py.File(h5_path, 'r') as f:
                    for ds_path in ds_paths:
                        dataset = f[ds_path]
                        data = dataset[:]

                        cols_attr = dataset.attrs.get('column_names', '')
                        if isinstance(cols_attr, bytes):
                            cols_attr = cols_attr.decode('utf-8')
                        cols = cols_attr.split(',') if cols_attr else None

                        df = pd.DataFrame(data, columns=cols)
                        df.dropna(inplace=True)
                        df.attrs['prefix'] = format_prefix(prefix)

                        base = os.path.basename(ds_path)
                        matches = re.findall(r'_[0-9]+$', base)
                        if matches:
                            file_id = int(matches[0].replace('_', ''))
                        else:
                            file_id = 0
                        df.attrs['id'] = file_id
                        data_frame_sets[file_id].append(df)
            else:
                file_paths_id = glob.glob(os.path.join(directory, prefix + '.csv'))
                file_paths_id.extend(glob.glob(os.path.join(directory, prefix + '_[0-9].csv')))
                for file_path in file_paths_id:
                    file_name = os.path.basename(file_path)
                    df = pd.read_csv(file_path).dropna()
                    df.attrs['prefix'] = format_prefix(prefix)
                    matches = re.findall(r'_[0-9]*\.csv', file_name)
                    if matches:
                        file_id = int(matches[0].split('_')[1].split('.csv')[0])
                    else:
                        file_id = 0
                    df.attrs['id'] = file_id
                    data_frame_sets[file_id].append(df)

    if prefix in ['body_truth', 'board_truth', 'feature_points']:
        for file_id in list(data_frame_sets.keys()):
            dfs = data_frame_sets[file_id]
            if len(dfs) > 0 and len(dfs) < len(directories):
                first_truth = dfs[0]
                while len(data_frame_sets[file_id]) < len(directories):
                    data_frame_sets[file_id].append(first_truth.copy())

    return data_frame_sets


def merge_mc_hdf5_files(config_set, input_file):
    """Merge individual run HDF5 files into a single HDF5 file with shared truth at root."""
    if len(config_set) <= 1:
        return

    top_name = os.path.basename(input_file).split('.yaml')[0]
    top_dir = input_file.split('.yaml')[0]
    output_h5_path = os.path.join(top_dir, f'{top_name}.h5')

    # Ensure directory of output file exists
    os.makedirs(top_dir, exist_ok=True)

    # Open target file
    with h5py.File(output_h5_path, 'w') as out_f:
        for config_path in config_set:
            run_dir = config_path.split('.yaml')[0]
            if not os.path.exists(run_dir):
                continue

            run_name = os.path.basename(run_dir.rstrip(os.sep))
            # Find any .h5 file in the directory
            h5_files = [f for f in os.listdir(run_dir) if f.endswith('.h5')]
            if not h5_files:
                continue

            src_h5_path = os.path.join(run_dir, h5_files[0])
            try:
                with h5py.File(src_h5_path, 'r') as in_f:
                    # Group name like run_0, run_1, etc.
                    matches = re.findall(r'[0-9]+$', run_name)
                    if matches:
                        group_name = f'run_{int(matches[-1])}'
                    else:
                        group_name = run_name

                    out_f.create_group(group_name)

                    # Copy all datasets and groups from source
                    def copy_item(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            if name.startswith('truth/'):
                                # Copy truth datasets directly to root /truth
                                dest_path = name
                                if dest_path not in out_f:
                                    out_f.copy(obj, dest_path)
                            else:
                                # Copy other datasets under run group
                                dest_path = f'{group_name}/{name}'
                                out_f.copy(obj, dest_path)
                    in_f.visititems(copy_item)

                # Delete individual HDF5 file after copy
                os.remove(src_h5_path)
            except Exception as e:
                print(f'Warning: Failed to copy or remove temporary file {src_h5_path}: {e}')


def generate_mc_lists(args):
    """Generate sets of yaml configuration files to plot."""
    mc_lists = []
    for input_file in args.inputs:
        yaml_files = []
        with open(input_file, 'r') as input_stream:
            try:
                top_yaml = yaml.safe_load(input_stream)
                sim_yaml = top_yaml['/EkfCalNode']['ros__parameters']['sim_params']
                num_runs = sim_yaml['number_of_runs']
                if (args.runs):
                    num_runs = args.runs
                if (num_runs > 1):
                    top_name = os.path.basename(input_file).split('.yaml')[0]
                    yaml_dir = input_file.split('.yaml')[0] + os.sep
                    runs_dir = os.path.join(yaml_dir, 'runs')
                    n_digits = math.ceil(math.log10(num_runs))
                    for i in range(num_runs):
                        sub_file = os.path.join(
                            runs_dir, '{}_{:0{:d}.0f}.yaml'.format(top_name, i, n_digits))
                        yaml_files.append(sub_file)
                else:
                    yaml_files.append(input_file)
            except yaml.YAMLError as exc:
                print(exc)
        mc_lists.append(yaml_files)
    return mc_lists
