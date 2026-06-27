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


import numpy as np
from bokeh.layouts import layout
from bokeh.models import Range1d, Spacer, TabPanel
from bokeh.plotting import figure
from scipy.stats.distributions import chi2
from utilities import calculate_alpha, get_colors, plot_update_timing


class tab_body:
    """Class for plotting body data."""

    def __init__(self, body_state_dfs, aug_state_dfs, body_truth_dfs, args, err_dfs=None):
        """Initialize the tab_body class for plotting body state information."""
        self.body_state_dfs = body_state_dfs
        self.aug_state_dfs = aug_state_dfs
        self.body_truth_dfs = body_truth_dfs
        self.alpha = calculate_alpha(len(self.body_state_dfs))
        self.colors = get_colors(args)
        self.err_dfs = err_dfs if err_dfs is not None else {}

    def plot_body_pos(self):
        """Plot body position."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Position [m]', title='Body Position')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            pos_x = body_df['body_pos_0']
            pos_y = body_df['body_pos_1']
            pos_z = body_df['body_pos_2']
            fig.line(time, pos_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, pos_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, pos_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_vel(self):
        """Plot body velocity."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Velocity [m/s]', title='Body Velocity')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            vel_x = body_df['body_vel_0']
            vel_y = body_df['body_vel_1']
            vel_z = body_df['body_vel_2']
            fig.line(time, vel_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, vel_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, vel_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_acc(self):
        """Plot body acceleration."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Acceleration [m/s/s]', title='Body Acceleration')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            acc_x = body_df['body_acc_0']
            acc_y = body_df['body_acc_1']
            acc_z = body_df['body_acc_2']
            fig.line(time, acc_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, acc_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, acc_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_ang(self):
        """Plot body angular position."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angle [rad]', title='Body Angle')
        body_euler_list = self.err_dfs.get('body_euler', [])
        for err_df in body_euler_list:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_ang_vel(self):
        """Plot body angular velocity."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angular Velocity', title='Body Angular Velocity')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            ang_vel_x = body_df['body_ang_vel_0']
            ang_vel_y = body_df['body_ang_vel_1']
            ang_vel_z = body_df['body_ang_vel_2']
            fig.line(time, ang_vel_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, ang_vel_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, ang_vel_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_ang_acc(self):
        """Plot body angular acceleration."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angular Acceleration', title='Body Angular Acceleration')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            ang_acc_x = body_df['body_ang_acc_0']
            ang_acc_y = body_df['body_ang_acc_1']
            ang_acc_z = body_df['body_ang_acc_2']
            fig.line(time, ang_acc_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, ang_acc_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, ang_acc_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_err_pos(self):
        """Plot the body state position error."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Position Error [m]', title='Body Position Error')
        body_pos_err_list = self.err_dfs.get('body_pos_err', [])
        for err_df in body_pos_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_err_vel(self):
        """Plot the body state velocity error."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Velocity Error [m/s]', title='Body Velocity Error')
        body_vel_err_list = self.err_dfs.get('body_vel_err', [])
        for err_df in body_vel_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_err_acc(self):
        """Plot the body state acceleration error."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Acceleration Error [m/s/s]', title='Body Acceleration Error')
        body_acc_err_list = self.err_dfs.get('body_acc_err', [])
        for err_df in body_acc_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_err_ang(self):
        """Plot the body state angular error."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angular Error', title='Body Angular Error')
        body_ang_err_list = self.err_dfs.get('body_ang_err', [])
        for err_df in body_ang_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_err_ang_vel(self):
        """Plot the body state angular velocity error."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angular Velocity Error [rad/s]',
                     title='Body Angular Velocity Error')
        body_ang_vel_err_list = self.err_dfs.get('body_ang_vel_err', [])
        for err_df in body_ang_vel_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_err_ang_acc(self):
        """Plot the body state angular acceleration error."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angular Acceleration Error [rad/s/s]',
                     title='Body Angular Acceleration Error')
        body_ang_acc_err_list = self.err_dfs.get('body_ang_acc_err', [])
        for err_df in body_ang_acc_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_pos_cov(self):
        """Plot body covariances for body position."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Position [m]', title='Body Position Covariance')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            cov_x = body_df['body_cov_0']
            cov_y = body_df['body_cov_1']
            cov_z = body_df['body_cov_2']
            fig.line(time, cov_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cov_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cov_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_vel_cov(self):
        """Plot body covariances for body velocity."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Velocity [m/s]', title='Body Velocity Covariance')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            cov_x = body_df['body_cov_3']
            cov_y = body_df['body_cov_4']
            cov_z = body_df['body_cov_5']
            fig.line(time, cov_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cov_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cov_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_acc_cov(self):
        """Plot body covariances for body acceleration."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Acceleration [m/s/s]', title='Body Acceleration Covariance')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            is_reduced = (body_df['body_cov_9'] == 0.0).all()
            if is_reduced:
                cov_x = np.zeros_like(time)
                cov_y = np.zeros_like(time)
                cov_z = np.zeros_like(time)
            else:
                cov_x = body_df['body_cov_6']
                cov_y = body_df['body_cov_7']
                cov_z = body_df['body_cov_8']
            fig.line(time, cov_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cov_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cov_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_ang_cov(self):
        """Plot body covariances for body angles."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angle [rad]', title='Body Angular Covariance')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            is_reduced = (body_df['body_cov_9'] == 0.0).all()
            if is_reduced:
                cov_x = body_df['body_cov_6']
                cov_y = body_df['body_cov_7']
                cov_z = body_df['body_cov_8']
            else:
                cov_x = body_df['body_cov_9']
                cov_y = body_df['body_cov_10']
                cov_z = body_df['body_cov_11']
            fig.line(time, cov_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cov_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cov_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_ang_vel_cov(self):
        """Plot body covariances for body angular rate."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angular Rate [rad/s]', title='Body Angular Rate Covariance')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            cov_x = body_df['body_cov_12']
            cov_y = body_df['body_cov_13']
            cov_z = body_df['body_cov_14']
            fig.line(time, cov_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cov_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cov_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_ang_acc_cov(self):
        """Plot body covariances for body angular acceleration."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Angular Acceleration [rad/s/s]',
                     title='Body Angular Acceleration Covariance')
        for body_df in self.body_state_dfs:
            time = body_df['time']
            cov_x = body_df['body_cov_15']
            cov_y = body_df['body_cov_16']
            cov_z = body_df['body_cov_17']
            fig.line(time, cov_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cov_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cov_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_body_nees(self):
        """Plot body normalized estimation error squared."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='NEES', title='Normalized Estimation Error Squared')
        body_nees_list = self.err_dfs.get('body_nees', [])

        for err_df in body_nees_list:
            fig.line(err_df['time'], err_df['nees'], alpha=self.alpha, color=self.colors[0])

        df = 18
        if self.body_state_dfs:
            is_reduced = (self.body_state_dfs[0]['body_cov_9'] == 0.0).all()
            if is_reduced:
                df = 9

        fig.hspan(y=chi2.ppf(0.025, df=df), line_color='red')
        fig.hspan(y=chi2.ppf(0.975, df=df), line_color='red')
        fig.y_range = Range1d(0, 2 * df + 4)

        return fig

    def plot_aug_pos(self):
        """Plot augmented state position."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Augmented Position [m]',
                     title='Augmented State Position')
        for aug_df in self.aug_state_dfs:
            time = aug_df['time']
            fig.scatter(
                time,
                aug_df['aug_pos_0'],
                alpha=self.alpha,
                color=self.colors[0],
                legend_label='X',
                size=1)
            fig.scatter(
                time,
                aug_df['aug_pos_1'],
                alpha=self.alpha,
                color=self.colors[1],
                legend_label='Y',
                size=1)
            fig.scatter(
                time,
                aug_df['aug_pos_2'],
                alpha=self.alpha,
                color=self.colors[2],
                legend_label='Z',
                size=1)

        return fig

    def plot_aug_ang(self):
        """Plot augmented state orientation."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='Augmented Orientation [rad]',
                     title='Augmented State Orientation')

        aug_euler_list = self.err_dfs.get('aug_euler', [])
        for err_df in aug_euler_list:
            time = err_df['time']
            fig.scatter(
                time,
                err_df['x'],
                alpha=self.alpha,
                color=self.colors[0],
                legend_label='X',
                size=1)
            fig.scatter(
                time,
                err_df['y'],
                alpha=self.alpha,
                color=self.colors[1],
                legend_label='Y',
                size=1)
            fig.scatter(
                time,
                err_df['z'],
                alpha=self.alpha,
                color=self.colors[2],
                legend_label='Z',
                size=1)

        return fig

    def plot_state_size(self):
        """Plot total state size."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='State Size',
                     title='State Size')
        for body_df in self.body_state_dfs:

            fig.line(
                body_df['time'],
                body_df['state_size'],
                alpha=self.alpha,
                color=self.colors[0])

        return fig

    def get_tab(self):
        """Generate the Bokeh TabPanel containing all body state plots."""
        is_reduced = False
        if self.body_state_dfs:
            is_reduced = (self.body_state_dfs[0]['body_cov_9'] == 0.0).all()

        layout_plots = [
            [
                self.plot_body_pos(),
                self.plot_body_ang()
            ],
            [
                self.plot_body_err_pos(),
                self.plot_body_err_ang()
            ],
            [
                self.plot_body_pos_cov(),
                self.plot_body_ang_cov()
            ],
            [
                self.plot_body_vel(),
                self.plot_body_ang_vel()
            ],
            [
                self.plot_body_err_vel(),
                self.plot_body_err_ang_vel()
            ],
            [
                self.plot_body_vel_cov(),
                Spacer() if is_reduced else self.plot_body_ang_vel_cov()
            ],
            [
                self.plot_body_acc(),
                Spacer() if is_reduced else self.plot_body_ang_acc()
            ],
            [
                self.plot_body_err_acc(),
                Spacer() if is_reduced else self.plot_body_err_ang_acc()
            ]
        ]

        if not is_reduced:
            layout_plots.append([self.plot_body_acc_cov(), self.plot_body_ang_acc_cov()])

        if self.aug_state_dfs:
            layout_plots.append([self.plot_aug_pos(), self.plot_aug_ang()])

        layout_plots.append([plot_update_timing(self.body_state_dfs), self.plot_state_size()])
        layout_plots.append([self.plot_body_nees(), Spacer()])

        tab_layout = layout(layout_plots, sizing_mode='stretch_width')
        tab = TabPanel(child=tab_layout, title='Body')

        return tab
