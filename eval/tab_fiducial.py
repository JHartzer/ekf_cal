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


class tab_fiducial:
    """Class for plotting fiducial data."""

    def __init__(self, fiducial_dfs, board_truth_dfs, body_truth_dfs, args, err_dfs=None):
        """Initialize the tab_fiducial class for plotting fiducial information."""
        self.fiducial_dfs = fiducial_dfs
        self.board_truth_dfs = board_truth_dfs
        self.body_truth_dfs = body_truth_dfs
        self.err_dfs = err_dfs if err_dfs is not None else {}
        self.is_fid_extrinsic = 'fid_cov_0' in self.fiducial_dfs[0].keys()
        self.is_cam_extrinsic = 'cam_cov_0' in self.fiducial_dfs[0].keys()

        self.alpha = calculate_alpha(len(self.fiducial_dfs))
        self.colors = get_colors(args)

    def plot_camera_pos(self):
        """Plot camera position offsets."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position [m]',
            title='Camera Position')
        for fiducial_df in self.fiducial_dfs:
            time = fiducial_df['time']
            pos_x = fiducial_df['cam_pos_0']
            pos_y = fiducial_df['cam_pos_1']
            pos_z = fiducial_df['cam_pos_2']
            fig.line(time, pos_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, pos_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, pos_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_camera_ang(self):
        """Plot camera angular offsets."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Orientation',
            title='Camera Orientation')
        for fiducial_df in self.fiducial_dfs:
            time = fiducial_df['time']
            ang_x = fiducial_df['cam_ang_pos_0']
            ang_y = fiducial_df['cam_ang_pos_1']
            ang_z = fiducial_df['cam_ang_pos_2']
            fig.line(time, ang_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, ang_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, ang_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_cam_pos_err(self):
        """Plot camera extrinsic position errors."""
        fig = figure(
            width=400,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Error [mm]',
            title='Camera Extrinsic Position Error')
        fid_pos_err_list = self.err_dfs.get('fiducial_pos_err', [])
        for err_df in fid_pos_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'] * 1e3, alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'] * 1e3, alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'] * 1e3, alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_cam_ang_err(self):
        """Plot the camera extrinsic angular error."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Angle Error [mrad]',
            title='Camera Extrinsic Angle Error')
        fid_ang_err_list = self.err_dfs.get('fiducial_ang_err', [])
        for err_df in fid_ang_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'] * 1e3, alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'] * 1e3, alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'] * 1e3, alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_cam_pos_cov(self):
        """Plot extrinsic position covariance."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Covariance [m]',
            title='Camera Position Covariance')
        for fiducial_df in self.fiducial_dfs:
            time = fiducial_df['time']
            cam_cov_0 = fiducial_df['cam_cov_0']
            cam_cov_1 = fiducial_df['cam_cov_1']
            cam_cov_2 = fiducial_df['cam_cov_2']
            fig.line(time, cam_cov_0, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cam_cov_1, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cam_cov_2, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_cam_ang_cov(self):
        """Plot extrinsic angle covariance."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Angle Covariance [m]',
            title='Camera Angle Covariance')
        for fiducial_df in self.fiducial_dfs:
            time = fiducial_df['time']
            cam_cov_3 = fiducial_df['cam_cov_3']
            cam_cov_4 = fiducial_df['cam_cov_4']
            cam_cov_5 = fiducial_df['cam_cov_5']
            fig.line(time, cam_cov_3, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cam_cov_4, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cam_cov_5, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_fiducial_error_pos(self):
        """Plot fiducial position in Camera Frame."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position [m]',
            title='Fiducial Position in Camera Frame')

        for fiducial_df in self.fiducial_dfs:
            time = fiducial_df['time']
            board_px = fiducial_df['board_meas_pos_0']
            board_py = fiducial_df['board_meas_pos_1']
            board_pz = fiducial_df['board_meas_pos_2']
            fig.line(time, board_px, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, board_py, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, board_pz, alpha=self.alpha, color=self.colors[2], legend_label='Z')

        return fig

    def plot_fiducial_error_ang(self):
        """Plot fiducial angle in camera frame."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Angular',
            title='Fiducial Angle in Camera Frame')

        fiducial_meas_euler = self.err_dfs.get('fiducial_meas_euler', [])
        for err_df in fiducial_meas_euler:
            time = err_df['time']
            fig.line(time, err_df['x'], alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'], alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'], alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_fid_nees(self):
        """Plot fiducial normalized estimation error squared."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='NEES', title='Normalized Estimation Error Squared')
        fid_nees_list = self.err_dfs.get('fiducial_nees', [])
        for err_df in fid_nees_list:
            fig.line(err_df['time'], err_df['nees'], alpha=self.alpha, color=self.colors[0])

        fig.hspan(y=chi2.ppf(0.025, df=6), line_color='red')
        fig.hspan(y=chi2.ppf(0.975, df=6), line_color='red')
        fig.y_range = Range1d(0, 40)

        return fig

    def get_tab(self):
        """Generate the Bokeh TabPanel containing all fiducial plots."""
        layout_plots = [[self.plot_fiducial_error_pos(), self.plot_fiducial_error_ang()]]

        if self.is_cam_extrinsic:
            layout_plots.append([self.plot_camera_pos(), self.plot_camera_ang()])
            layout_plots.append([self.plot_cam_pos_err(), self.plot_cam_ang_err()])
            layout_plots.append([self.plot_cam_pos_cov(), self.plot_cam_ang_cov()])

        layout_plots.append([plot_update_timing(self.fiducial_dfs), Spacer()])

        if self.is_fid_extrinsic:
            layout_plots.append([self.plot_fid_nees(), Spacer()])

        tab_layout = layout(layout_plots, sizing_mode='stretch_width')
        tab = TabPanel(child=tab_layout, title=f"Fiducial {self.fiducial_dfs[0].attrs['id']}")

        return tab
