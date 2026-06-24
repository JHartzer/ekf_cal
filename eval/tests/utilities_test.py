#!/usr/bin/env python3

# Copyright 2026 Jacob Hartzer
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

import h5py
import numpy as np
import pandas as pd
import utilities
import yaml


def test_calculate_alpha_and_format_prefix():
    assert utilities.calculate_alpha(4) == 0.5
    assert utilities.format_prefix('imu') == 'IMU'
    assert utilities.format_prefix('camera') == 'Camera'
    assert utilities.format_prefix('unknown') == ''


def test_interpolate_error_returns_estimate_minus_interpolated_truth():
    errors = utilities.interpolate_error(
        true_t=[0.0, 1.0, 2.0],
        true_x=[0.0, 2.0, 4.0],
        estimate_t=[0.5, 1.5],
        estimate_x=[1.5, 2.5],
    )
    np.testing.assert_allclose(errors, [0.5, -0.5])


def test_interpolate_quat_error_is_zero_for_matching_quaternions():
    err_x, err_y, err_z = utilities.interpolate_quat_error(
        true_t=[0.0, 1.0],
        true_w=[1.0, 1.0],
        true_x=[0.0, 0.0],
        true_y=[0.0, 0.0],
        true_z=[0.0, 0.0],
        estimate_t=[0.25, 0.75],
        estimate_w=[1.0, 1.0],
        estimate_x=[0.0, 0.0],
        estimate_y=[0.0, 0.0],
        estimate_z=[0.0, 0.0],
    )
    np.testing.assert_allclose(err_x, [0.0, 0.0])
    np.testing.assert_allclose(err_y, [0.0, 0.0])
    np.testing.assert_allclose(err_z, [0.0, 0.0])


def test_lists_to_rot_and_calculate_rotation_errors_track_known_rotation():
    truth = utilities.lists_to_rot([1.0], [0.0], [0.0], [0.0])
    estimate = utilities.lists_to_rot(
        [np.cos(np.pi / 4.0)], [np.sin(np.pi / 4.0)], [0.0], [0.0]
    )

    err_x, err_y, err_z = utilities.calculate_rotation_errors(estimate, truth)

    np.testing.assert_allclose(err_x, [np.pi / 2.0], atol=1e-7)
    np.testing.assert_allclose(err_y, [0.0], atol=1e-7)
    np.testing.assert_allclose(err_z, [0.0], atol=1e-7)


def test_parse_yaml_extracts_sensor_rates_in_sequence(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config = {
        '/EkfCalNode': {
            'ros__parameters': {
                'imu_list': ['imu_a', 'imu_b'],
                'camera_list': ['cam_a'],
                'imu': {
                    'imu_a': {'rate': 100.0},
                    'imu_b': {'rate': 200.0},
                },
                'camera': {
                    'cam_a': {'rate': 30.0},
                },
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

    parsed = utilities.parse_yaml(str(config_path))

    assert parsed['imu_rates'] == {1: 100.0, 2: 200.0}
    assert parsed['camera_rates'] == {3: 30.0}


def _write_dataset(group, name, data, columns):
    dataset = group.create_dataset(name, data=np.asarray(data, dtype=float))
    dataset.attrs['column_names'] = ','.join(columns)
    return dataset


def test_get_matching_datasets_filters_truth_and_run_sensor_datasets(tmp_path):
    h5_path = tmp_path / 'sample.h5'
    with h5py.File(h5_path, 'w') as h5_file:
        truth = h5_file.create_group('truth')
        _write_dataset(truth, 'body', [[0.0, 1.0]], ['time', 'body_pos_0'])
        run_0 = h5_file.create_group('run_0')
        _write_dataset(run_0, 'imu_1', [[0.0, 1.0]], ['time', 'imu_pos_0'])
        _write_dataset(run_0, 'imu_2', [[0.0, 2.0]], ['time', 'imu_pos_0'])
        _write_dataset(
            run_0,
            'imu_1_timing',
            [[0.0, 0.0, 0.1]],
            ['time_used', 'time_offset_sample', 'time_offset_min'])
        _write_dataset(run_0, 'body_state', [[0.0, 3.0]], ['time', 'body_pos_0'])

    assert utilities.get_matching_datasets(str(h5_path), None, 'body_truth') == ['truth/body']
    assert utilities.get_matching_datasets(str(h5_path), 'run_0', 'imu') == [
        'run_0/imu_1', 'run_0/imu_2']
    assert utilities.get_matching_datasets(str(h5_path), 'run_0', 'imu_timing') == [
        'run_0/imu_1_timing']
    assert utilities.get_matching_datasets(str(h5_path), 'run_0', 'body_state') == [
        'run_0/body_state']


def test_find_and_read_data_frames_reads_merged_hdf5_and_replicates_truth(tmp_path):
    base_dir = tmp_path / 'config'
    runs_dir = base_dir / 'runs'
    run0_dir = runs_dir / 'example_0'
    run1_dir = runs_dir / 'example_1'
    run0_dir.mkdir(parents=True)
    run1_dir.mkdir(parents=True)

    merged_h5 = base_dir / 'example.h5'
    with h5py.File(merged_h5, 'w') as h5_file:
        truth = h5_file.create_group('truth')
        _write_dataset(
            truth,
            'body',
            [[0.0, 10.0], [1.0, 11.0]],
            ['time', 'body_pos_0'],
        )
        run_0 = h5_file.create_group('run_0')
        run_1 = h5_file.create_group('run_1')
        _write_dataset(run_0, 'imu_1', [[0.0, 1.0], [1.0, np.nan]], ['time', 'imu_pos_0'])
        _write_dataset(run_1, 'imu_1', [[0.0, 2.0], [1.0, 3.0]], ['time', 'imu_pos_0'])
        _write_dataset(
            run_0,
            'imu_1_timing',
            [[0.0, 0.1, 0.1], [1.0, 0.2, 0.1]],
            ['time_used', 'time_offset_sample', 'time_offset_min'])
        _write_dataset(
            run_1,
            'imu_1_timing',
            [[0.0, 0.3, 0.2], [1.0, 0.4, 0.2]],
            ['time_used', 'time_offset_sample', 'time_offset_min'])

    truth_sets = utilities.find_and_read_data_frames([str(run0_dir), str(run1_dir)], 'body_truth')
    imu_sets = utilities.find_and_read_data_frames([str(run0_dir), str(run1_dir)], 'imu')
    imu_timing_sets = utilities.find_and_read_data_frames(
        [str(run0_dir), str(run1_dir)], 'imu_timing')

    assert len(truth_sets[0]) == 2
    assert truth_sets[0][0].attrs['id'] == 0
    np.testing.assert_allclose(truth_sets[0][1]['body_pos_0'], [10.0, 11.0])

    assert len(imu_sets[1]) == 2
    assert imu_sets[1][0].attrs['prefix'] == 'IMU'
    assert imu_sets[1][0].attrs['id'] == 1
    # NaN rows should be dropped on readback.
    np.testing.assert_allclose(imu_sets[1][0]['time'], [0.0])
    np.testing.assert_allclose(imu_sets[1][1]['imu_pos_0'], [2.0, 3.0])

    assert len(imu_timing_sets[1]) == 2
    assert imu_timing_sets[1][0].attrs['prefix'] == 'IMU'
    assert imu_timing_sets[1][0].attrs['id'] == 1
    np.testing.assert_allclose(imu_timing_sets[1][0]['time_offset_min'], [0.1, 0.1])


def test_find_and_read_data_frames_preserves_timing_rows_with_optional_nan_columns(tmp_path):
    base_dir = tmp_path / 'config'
    runs_dir = base_dir / 'runs'
    run0_dir = runs_dir / 'example_0'
    run0_dir.mkdir(parents=True)

    merged_h5 = base_dir / 'example.h5'
    with h5py.File(merged_h5, 'w') as h5_file:
        run_0 = h5_file.create_group('run_0')
        _write_dataset(
            run_0,
            'camera_3_timing',
            [[0.0, 0.0, 0.1, np.nan, 0.1, 0.1, np.nan]],
            [
                'time_used',
                'time_measured',
                'time_received',
                'time_true',
                'time_offset_sample',
                'time_offset_min',
                'time_alignment_error',
            ])

    timing_sets = utilities.find_and_read_data_frames([str(run0_dir)], 'camera_timing')

    assert list(timing_sets.keys()) == [3]
    assert len(timing_sets[3][0]) == 1
    np.testing.assert_allclose(timing_sets[3][0]['time_offset_min'], [0.1])


def test_find_and_read_data_frames_falls_back_to_csv(tmp_path):
    run_dir = tmp_path / 'single_run'
    run_dir.mkdir()
    pd.DataFrame({'time': [0.0, 1.0], 'imu_pos_0': [1.0, 2.0]}).to_csv(
        run_dir / 'imu_1.csv', index=False
    )

    imu_sets = utilities.find_and_read_data_frames([str(run_dir)], 'imu')

    assert list(imu_sets.keys()) == [1]
    np.testing.assert_allclose(imu_sets[1][0]['imu_pos_0'], [1.0, 2.0])
    assert imu_sets[1][0].attrs['prefix'] == 'IMU'


def test_merge_mc_hdf5_files_moves_truth_to_root_and_deletes_run_files(tmp_path):
    input_file = tmp_path / 'example.yaml'
    input_file.write_text('dummy: true\n', encoding='utf-8')
    top_dir = tmp_path / 'example'
    runs_dir = top_dir / 'runs'
    run0_dir = runs_dir / 'example_0'
    run1_dir = runs_dir / 'example_1'
    run0_dir.mkdir(parents=True)
    run1_dir.mkdir(parents=True)

    config_set = [
        str(runs_dir / 'example_0.yaml'),
        str(runs_dir / 'example_1.yaml'),
    ]
    for idx, run_dir in enumerate([run0_dir, run1_dir]):
        with h5py.File(run_dir / f'run_{idx}.h5', 'w') as h5_file:
            truth = h5_file.create_group('truth')
            _write_dataset(truth, 'body', [[0.0, 10.0 + idx]], ['time', 'body_pos_0'])
            _write_dataset(h5_file, 'imu_1', [[0.0, float(idx)]], ['time', 'imu_pos_0'])

    utilities.merge_mc_hdf5_files(config_set, str(input_file))

    merged_h5 = top_dir / 'example.h5'
    assert merged_h5.exists()
    assert not any(run0_dir.glob('*.h5'))
    assert not any(run1_dir.glob('*.h5'))

    with h5py.File(merged_h5, 'r') as h5_file:
        assert 'truth/body' in h5_file
        assert 'run_0/imu_1' in h5_file
        assert 'run_1/imu_1' in h5_file
        np.testing.assert_allclose(h5_file['truth/body'][:, 1], [10.0])


def test_generate_mc_lists_expands_multi_run_inputs(tmp_path):
    input_file = tmp_path / 'config.yaml'
    top_dir = tmp_path / 'config'
    runs_dir = top_dir / 'runs'
    runs_dir.mkdir(parents=True)
    config = {
        '/EkfCalNode': {
            'ros__parameters': {
                'sim_params': {
                    'number_of_runs': 3,
                }
            }
        }
    }
    input_file.write_text(yaml.safe_dump(config), encoding='utf-8')

    class Args:
        inputs = [str(input_file)]
        runs = None

    mc_lists = utilities.generate_mc_lists(Args())

    assert mc_lists == [[
        str(runs_dir / 'config_0.yaml'),
        str(runs_dir / 'config_1.yaml'),
        str(runs_dir / 'config_2.yaml'),
    ]]
