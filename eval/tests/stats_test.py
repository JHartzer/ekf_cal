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

import numpy as np
import pandas as pd

import stats


def _df(data, sensor_id=None):
    df = pd.DataFrame(data)
    if sensor_id is not None:
        df.attrs['id'] = sensor_id
    return df


def test_body_err_pos_interpolates_truth_before_rmse():
    body_state = _df({
        'time': [0.5, 1.5],
        'body_pos_0': [1.5, 2.5],
        'body_pos_1': [0.0, 0.0],
        'body_pos_2': [0.0, 0.0],
    })
    body_truth = _df({
        'time': [0.0, 1.0, 2.0],
        'body_pos_0': [0.0, 2.0, 4.0],
        'body_pos_1': [0.0, 0.0, 0.0],
        'body_pos_2': [0.0, 0.0, 0.0],
    })

    rmse = stats.body_err_pos([body_state], [body_truth])

    np.testing.assert_allclose(rmse, [0.5])


def test_sensor_err_pos_and_imu_err_bias_use_sensor_id_columns():
    sensor_df = _df({
        'time': [0.0, 1.0],
        'imu_pos_0': [1.0, 2.0],
        'imu_pos_1': [0.0, 0.0],
        'imu_pos_2': [0.0, 0.0],
        'imu_acc_bias_0': [0.2, 0.2],
        'imu_acc_bias_1': [0.0, 0.0],
        'imu_acc_bias_2': [0.0, 0.0],
    }, sensor_id=1)
    truth_df = _df({
        'time': [0.0, 1.0],
        'imu_pos_1_0': [1.0, 1.0],
        'imu_pos_1_1': [0.0, 0.0],
        'imu_pos_1_2': [0.0, 0.0],
        'imu_acc_bias_1_0': [0.1, 0.1],
        'imu_acc_bias_1_1': [0.0, 0.0],
        'imu_acc_bias_1_2': [0.0, 0.0],
    })

    pos_rmse = stats.sensor_err_pos([sensor_df], [truth_df], 'imu')
    bias_rmse = stats.imu_err_bias([sensor_df], [truth_df], 'acc')

    np.testing.assert_allclose(pos_rmse, [np.sqrt(0.5)])
    np.testing.assert_allclose(bias_rmse, [0.1])


def test_gps_error_helpers_and_init_count_use_first_initialized_sample_after_transition():
    gps_df = _df({
        'is_initialized': [0, 1, 1],
        'ref_lat': [100.0, 100.2, 100.3],
        'ref_lon': [200.0, 199.8, 199.7],
        'ref_alt': [300.0, 300.5, 300.6],
        'ref_heading': [0.0, 0.4, 0.5],
    })
    truth_df = _df({
        'ref_lat': [100.0],
        'ref_lon': [200.0],
        'ref_alt': [300.0],
        'ref_heading': [0.1],
    })

    pos_rmse = stats.gps_err_pos([gps_df], [truth_df])
    ang_err = stats.gps_err_ang([gps_df], [truth_df])
    init_count = stats.gps_init_count([gps_df])

    expected_rmse = np.sqrt(0.3 ** 2 + (-0.3) ** 2 + 0.6 ** 2)
    np.testing.assert_allclose(pos_rmse, [expected_rmse])
    np.testing.assert_allclose(ang_err, [0.4])
    assert init_count == [1]


def test_write_summary_outputs_mean_and_stddev(tmp_path):
    stats_path = tmp_path / 'stats.csv'
    stats.write_summary(str(tmp_path), {'metric': [1.0, 3.0]})

    content = stats_path.read_text(encoding='utf-8').splitlines()

    assert content[0] == 'Statistic,RMSE-Mean,RMSE-StdDev'
    assert content[1] == 'metric,2.0000,1.0000'


def test_calc_errors_for_single_run_generates_zero_body_errors_and_triangulation_errors():
    body_truth = _df({
        'time': [0.0, 1.0],
        'body_pos_0': [1.0, 2.0], 'body_pos_1': [0.0, 0.0], 'body_pos_2': [0.0, 0.0],
        'body_vel_0': [0.5, 0.5], 'body_vel_1': [0.0, 0.0], 'body_vel_2': [0.0, 0.0],
        'body_acc_0': [0.0, 0.0], 'body_acc_1': [0.0, 0.0], 'body_acc_2': [0.0, 0.0],
        'body_ang_pos_0': [1.0, 1.0], 'body_ang_pos_1': [0.0, 0.0],
        'body_ang_pos_2': [0.0, 0.0], 'body_ang_pos_3': [0.0, 0.0],
        'body_ang_vel_0': [0.0, 0.0], 'body_ang_vel_1': [0.0, 0.0], 'body_ang_vel_2': [0.0, 0.0],
        'body_ang_acc_0': [0.0, 0.0], 'body_ang_acc_1': [0.0, 0.0], 'body_ang_acc_2': [0.0, 0.0],
    })
    body_state = body_truth.copy()
    for idx in range(18):
        body_state[f'body_cov_{idx}'] = 1.0

    aug_state = _df({
        'time': [0.0, 1.0],
        'aug_ang_0': [1.0, 1.0],
        'aug_ang_1': [0.0, 0.0],
        'aug_ang_2': [0.0, 0.0],
        'aug_ang_3': [0.0, 0.0],
    })
    feat_df = _df({'x': [1.0], 'y': [2.0], 'z': [3.0]})
    tri_df = _df({'time': [0.0], 'feature': [99], 'x': [1.0], 'y': [2.0], 'z': [3.0]})

    run_args = (
        0,
        'run_dir',
        body_state,
        body_truth,
        {},
        {},
        {},
        {},
        None,
        aug_state,
        {1: tri_df},
        {0: feat_df},
    )

    _, _, run_errors = stats._calc_errors_for_single_run(run_args)

    for key in [
        'body_pos_err', 'body_vel_err', 'body_acc_err', 'body_ang_err',
        'body_ang_vel_err', 'body_ang_acc_err', 'body_nees', 'body_euler',
        'aug_euler', 'triangulation_err_1',
    ]:
        assert key in run_errors

    np.testing.assert_allclose(run_errors['body_pos_err'][:, 1:], 0.0)
    np.testing.assert_allclose(run_errors['body_ang_err'][:, 1:], 0.0)
    np.testing.assert_allclose(run_errors['body_nees'][:, 1], 0.0)
    np.testing.assert_allclose(run_errors['triangulation_err_1'][:, 1:], 0.0)
    np.testing.assert_allclose(run_errors['body_euler'][:, 1:], 0.0)
    np.testing.assert_allclose(run_errors['aug_euler'][:, 1:], 0.0)
