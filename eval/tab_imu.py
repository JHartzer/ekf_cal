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

from bokeh.layouts import layout
from bokeh.models import Range1d, Spacer, TabPanel
from bokeh.plotting import figure
from scipy.stats.distributions import chi2
from utilities import calculate_alpha, get_colors, plot_update_timing


class tab_imu:
    """Class for plotting IMU data."""

    def __init__(self, imu_dfs, body_truth_dfs, args, err_dfs=None):
        """Initialize the tab_imu class for plotting IMU information."""
        self.imu_dfs = imu_dfs
        self.body_truth_dfs = body_truth_dfs
        self.is_extrinsic = 'imu_ext_cov_0' in self.imu_dfs[0].keys()
        self.is_intrinsic = 'imu_int_cov_0' in self.imu_dfs[0].keys()
        self.alpha = calculate_alpha(len(self.imu_dfs))
        self.colors = get_colors(args)
        self.err_dfs = err_dfs if err_dfs is not None else {}

    def plot_acc_measurements(self):
        """Plot acceleration measurements."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Acceleration [m/s/s]',
            title='Acceleration Measurements')
        for imu_df in self.imu_dfs:
            t_imu = imu_df['time']
            acc_0 = imu_df['acc_0']
            acc_1 = imu_df['acc_1']
            acc_2 = imu_df['acc_2']
            fig.line(t_imu, acc_0, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(t_imu, acc_1, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(t_imu, acc_2, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_omg_measurements(self):
        """Plot angular rate measurements."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Angular Rate [rad/s]',
            title='Angular Rate Measurements')
        for imu_df in self.imu_dfs:
            t_imu = imu_df['time']
            omg_0 = imu_df['omg_0']
            omg_1 = imu_df['omg_1']
            omg_2 = imu_df['omg_2']
            fig.line(t_imu, omg_0, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(t_imu, omg_1, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(t_imu, omg_2, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_acc_residuals(self):
        """Plot acceleration residuals."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Acceleration [m/s/s]',
            title='Acceleration Residuals')
        for imu_df in self.imu_dfs:
            t_imu = imu_df['time']
            res_0 = imu_df['residual_0']
            res_1 = imu_df['residual_1']
            res_2 = imu_df['residual_2']
            fig.line(t_imu, res_0, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(t_imu, res_1, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(t_imu, res_2, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_omg_residuals(self):
        """Plot angular rate residuals."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Angular Rate [rad/s]',
            title='Angular Rate Residuals')
        for imu_df in self.imu_dfs:
            time = imu_df['time']
            res_3 = imu_df['residual_3']
            res_4 = imu_df['residual_4']
            res_5 = imu_df['residual_5']
            fig.line(time, res_3, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, res_4, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, res_5, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_ext_pos_err(self):
        """Plot the extrinsic position error."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Error [mm]',
            title='Extrinsic Position Error')

        imu_pos_err_list = self.err_dfs.get('imu_pos_err', [])
        for err_df in imu_pos_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'] * 1e3, alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'] * 1e3, alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'] * 1e3, alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')

        return fig

    def plot_ext_ang_err(self):
        """Plot the extrinsic angular error."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Angle Error [mrad]',
            title='Extrinsic Angle Error')
        imu_ang_err_list = self.err_dfs.get('imu_ang_err', [])
        for err_df in imu_ang_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'] * 1e3, alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'] * 1e3, alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'] * 1e3, alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_acc_bias_err(self):
        """Plot the intrinsic accelerometer bias error."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Bias Error [m]',
            title='Accelerometer Bias Error')
        imu_acc_bias_err_list = self.err_dfs.get('imu_acc_bias_err', [])
        for err_df in imu_acc_bias_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], color=self.colors[0], alpha=self.alpha)
            fig.line(time, err_df['y'], color=self.colors[1], alpha=self.alpha)
            fig.line(time, err_df['z'], color=self.colors[2], alpha=self.alpha)

        return fig

    def plot_omg_bias_err(self):
        """Plot the intrinsic gyroscope bias error."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Bias Error [m]',
            title='Gyroscope Bias Error')
        imu_gyr_bias_err_list = self.err_dfs.get('imu_gyr_bias_err', [])
        for err_df in imu_gyr_bias_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], color=self.colors[0], alpha=self.alpha)
            fig.line(time, err_df['y'], color=self.colors[1], alpha=self.alpha)
            fig.line(time, err_df['z'], color=self.colors[2], alpha=self.alpha)

        return fig

    def plot_imu_ext_pos_cov(self):
        """Plot extrinsic position covariance."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Covariance [m]',
            title='Position Covariance')
        for imu_df in self.imu_dfs:
            time = imu_df['time']
            imu_ext_cov_0 = imu_df['imu_ext_cov_0']
            imu_ext_cov_1 = imu_df['imu_ext_cov_1']
            imu_ext_cov_2 = imu_df['imu_ext_cov_2']
            fig.line(time, imu_ext_cov_0, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, imu_ext_cov_1, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, imu_ext_cov_2, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_imu_ext_ang_cov(self):
        """Plot extrinsic angle covariance."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Angle Covariance [m]',
            title='Angle Covariance')
        for imu_df in self.imu_dfs:
            time = imu_df['time']
            imu_ext_cov_3 = imu_df['imu_ext_cov_3']
            imu_ext_cov_4 = imu_df['imu_ext_cov_4']
            imu_ext_cov_5 = imu_df['imu_ext_cov_5']
            fig.line(time, imu_ext_cov_3, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, imu_ext_cov_4, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, imu_ext_cov_5, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_imu_int_pos_cov(self):
        """Plot intrinsic accelerometer bias covariance."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Accelerometer Bias Covariance [m/s/s]',
            title='Accelerometer Bias Covariance')
        for imu_df in self.imu_dfs:
            time = imu_df['time']
            imu_int_cov_0 = imu_df['imu_int_cov_0']
            imu_int_cov_1 = imu_df['imu_int_cov_1']
            imu_int_cov_2 = imu_df['imu_int_cov_2']
            fig.line(time, imu_int_cov_0, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, imu_int_cov_1, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, imu_int_cov_2, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_imu_int_ang_cov(self):
        """Plot intrinsic gyroscope bias covariance."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Gyroscope Bias Covariance [rad/s]',
            title='Gyroscope Bias Covariance')
        for imu_df in self.imu_dfs:
            time = imu_df['time']
            imu_int_cov_3 = imu_df['imu_int_cov_3']
            imu_int_cov_4 = imu_df['imu_int_cov_4']
            imu_int_cov_5 = imu_df['imu_int_cov_5']
            fig.line(time, imu_int_cov_3, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, imu_int_cov_4, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, imu_int_cov_5, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_stationary(self):
        """Plot is stationary update being performed."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Is Stationary',
            title='Is Stationary')
        for imu_df in self.imu_dfs:
            t_imu = imu_df['time']
            is_stationary = imu_df['stationary']
            score = imu_df['score']
            fig.line(
                t_imu,
                is_stationary,
                alpha=self.alpha,
                color=self.colors[0],
                legend_label='Is Stationary')
            fig.line(
                t_imu,
                score,
                alpha=self.alpha,
                color=self.colors[1],
                legend_label='Chi^2 Score')
        fig.y_range = Range1d(0, 1)
        return fig

    def plot_imu_nees(self):
        """Plot IMU normalized estimation error squared."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='NEES', title='Normalized Estimation Error Squared')
        imu_nees_list = self.err_dfs.get('imu_nees', [])

        for err_df in imu_nees_list:
            fig.line(err_df['time'], err_df['nees'], alpha=self.alpha, color=self.colors[0])

        dof = 0
        if self.is_extrinsic:
            dof += 6
        if self.is_intrinsic:
            dof += 6

        fig.hspan(y=chi2.ppf(0.025, df=dof), line_color='red')
        fig.hspan(y=chi2.ppf(0.975, df=dof), line_color='red')
        fig.y_range = Range1d(0, 40)

        return fig

    def get_tab(self):
        """Generate the Bokeh TabPanel containing all IMU plots."""
        layout_plots = [
            [self.plot_acc_measurements(), self.plot_omg_measurements()],
            [self.plot_acc_residuals(), self.plot_omg_residuals()],
        ]

        if self.is_extrinsic:
            layout_plots.append([self.plot_imu_ext_pos_cov(), self.plot_imu_ext_ang_cov()])
            layout_plots.append([self.plot_ext_pos_err(), self.plot_ext_ang_err()])

        if self.is_intrinsic:
            layout_plots.append([self.plot_imu_int_pos_cov(), self.plot_imu_int_ang_cov()])
            layout_plots.append([self.plot_acc_bias_err(), self.plot_omg_bias_err()])

        layout_plots.append([plot_update_timing(self.imu_dfs), self.plot_stationary()])

        if self.is_extrinsic or self.is_intrinsic:
            layout_plots.append([self.plot_imu_nees(), Spacer()])

        tab_layout = layout(layout_plots, sizing_mode='stretch_width')
        tab = TabPanel(child=tab_layout,
                       title=f"IMU {self.imu_dfs[0].attrs['id']}")

        return tab
