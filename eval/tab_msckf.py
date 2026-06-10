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

from bokeh.layouts import layout
from bokeh.models import Band, Range1d, Spacer, TabPanel
from bokeh.plotting import ColumnDataSource, figure
import numpy as np
from scipy.stats.distributions import chi2
from utilities import calculate_alpha, get_colors, plot_update_timing


class tab_msckf:
    """Class for plotting camera MSCKF data."""

    def __init__(self, msckf_dfs, tri_dfs, feat_dfs, body_truth_dfs, args, err_dfs=None):
        """Initialize the tab_msckf class for plotting camera MSCKF information."""
        self.msckf_dfs = msckf_dfs
        self.tri_dfs = tri_dfs
        self.feat_dfs = feat_dfs
        self.body_truth_dfs = body_truth_dfs
        self.is_extrinsic = 'cam_cov_0' in self.msckf_dfs[0].keys()
        self.alpha = calculate_alpha(len(self.msckf_dfs))
        self.colors = get_colors(args)
        self.err_dfs = err_dfs if err_dfs is not None else {}

    def plot_cam_pos(self):
        """Plot camera position offsets."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position [m]',
            title='Camera Position')
        for msckf_df in self.msckf_dfs:
            time = msckf_df['time']
            pos_x = msckf_df['cam_pos_0']
            pos_y = msckf_df['cam_pos_1']
            pos_z = msckf_df['cam_pos_2']
            fig.line(time, pos_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, pos_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, pos_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_cam_ang(self):
        """Plot camera angular offsets."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Orientation',
            title='Camera Orientation')
        for msckf_df in self.msckf_dfs:
            time = msckf_df['time']
            ang_x = msckf_df['cam_ang_pos_0']
            ang_y = msckf_df['cam_ang_pos_1']
            ang_z = msckf_df['cam_ang_pos_2']
            fig.line(time, ang_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, ang_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, ang_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_cam_pos_err(self):
        """Plot camera extrinsic position error."""
        fig = figure(
            width=400,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Error [mm]',
            title='Camera Extrinsic Position Error')
        cam_pos_err_list = self.err_dfs.get('cam_pos_err', [])
        for err_df in cam_pos_err_list:
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
        cam_ang_err_list = self.err_dfs.get('cam_ang_err', [])
        for err_df in cam_ang_err_list:
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
            title='Position Covariance')
        for msckf_df in self.msckf_dfs:
            time = msckf_df['time']
            cam_cov_0 = msckf_df['cam_cov_0']
            cam_cov_1 = msckf_df['cam_cov_1']
            cam_cov_2 = msckf_df['cam_cov_2']
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
            title='Angle Covariance')
        for msckf_df in self.msckf_dfs:
            time = msckf_df['time']
            cam_cov_3 = msckf_df['cam_cov_3']
            cam_cov_4 = msckf_df['cam_cov_4']
            cam_cov_5 = msckf_df['cam_cov_5']
            fig.line(time, cam_cov_3, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(time, cam_cov_4, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(time, cam_cov_5, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_triangulation_error(self):
        """Plot MSCKF feature point triangulation error."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Triangulation Error [m]',
            title='MSCKF Triangulation Error')

        err_x = collections.defaultdict(list)
        err_y = collections.defaultdict(list)
        err_z = collections.defaultdict(list)

        for tri_df, feat_df in zip(self.tri_dfs, self.feat_dfs):
            time = tri_df['time']
            feature = tri_df['feature']
            feat_x = tri_df['x']
            feat_y = tri_df['y']
            feat_z = tri_df['z']

            true_x = feat_df['x']
            true_y = feat_df['y']
            true_z = feat_df['z']

            for (t, f, x, y, z) in zip(time, feature, feat_x, feat_y, feat_z):
                err_x[t].append(x - true_x[int(f)])
                err_y[t].append(y - true_y[int(f)])
                err_z[t].append(z - true_z[int(f)])
        times = []
        mean_x = []
        mean_y = []
        mean_z = []
        std_x = []
        std_y = []
        std_z = []

        for time in err_x:
            times.append(time)
            mean_x.append(np.mean(err_x[time]))
            mean_y.append(np.mean(err_y[time]))
            mean_z.append(np.mean(err_z[time]))
            std_x.append(np.std(err_x[time]))
            std_y.append(np.std(err_y[time]))
            std_z.append(np.std(err_z[time]))

        times = np.array(times)
        mean_x = np.array(mean_x)
        mean_y = np.array(mean_y)
        mean_z = np.array(mean_z)
        std_x = np.array(std_x)
        std_y = np.array(std_y)
        std_z = np.array(std_z)

        t_indices = times.argsort()
        times = times[t_indices]
        mean_x = mean_x[t_indices]
        mean_y = mean_y[t_indices]
        mean_z = mean_z[t_indices]
        std_x = std_x[t_indices]
        std_y = std_y[t_indices]
        std_z = std_z[t_indices]

        fig.line(times, mean_x, color=self.colors[0], legend_label='X')
        fig.line(times, mean_y, color=self.colors[1], legend_label='Y')
        fig.line(times, mean_z, color=self.colors[2], legend_label='Z')

        cds_x = ColumnDataSource({'base': times, 'lower': mean_x - std_x, 'upper': mean_x + std_x})
        cds_y = ColumnDataSource({'base': times, 'lower': mean_y - std_y, 'upper': mean_y + std_y})
        cds_z = ColumnDataSource({'base': times, 'lower': mean_z - std_z, 'upper': mean_z + std_z})

        fig.add_layout(
            Band(
                base='base',
                lower='lower',
                upper='upper',
                fill_alpha=0.3,
                source=cds_x,
                fill_color=self.colors[0],
                line_color=self.colors[0]))
        fig.add_layout(
            Band(
                base='base',
                lower='lower',
                upper='upper',
                fill_alpha=0.3,
                source=cds_y,
                fill_color=self.colors[1],
                line_color=self.colors[1]))
        fig.add_layout(
            Band(
                base='base',
                lower='lower',
                upper='upper',
                fill_alpha=0.3,
                source=cds_z,
                fill_color=self.colors[2],
                line_color=self.colors[2]))

        return fig

    def plot_track_count(self):
        """Plot number of tracks."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Number of Tracks',
            title='Track Counts')
        for msckf_df in self.msckf_dfs:
            t_cam = msckf_df['time']
            track_count = msckf_df['FeatureTracks']
            fig.line(
                t_cam,
                track_count,
                alpha=self.alpha,
                color=self.colors[0],
                legend_label='Track Count')
        return fig

    def plot_cam_nees(self):
        """Plot camera MSCKF normalized estimation error squared."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='NEES', title='Normalized Estimation Error Squared')
        cam_nees_list = self.err_dfs.get('cam_nees', [])
        for err_df in cam_nees_list:
            fig.line(err_df['time'], err_df['nees'], alpha=self.alpha, color=self.colors[0])

        fig.hspan(y=chi2.ppf(0.025, df=6), line_color='red')
        fig.hspan(y=chi2.ppf(0.975, df=6), line_color='red')
        fig.y_range = Range1d(0, 40)

        return fig

    def get_tab(self):
        """Generate the Bokeh TabPanel containing all camera MSCKF plots."""
        layout_plots = [[self.plot_track_count(), self.plot_triangulation_error()]]

        if self.is_extrinsic:
            layout_plots.append([self.plot_cam_pos(), self.plot_cam_ang()])
            layout_plots.append([self.plot_cam_pos_err(), self.plot_cam_ang_err()])
            layout_plots.append([self.plot_cam_pos_cov(), self.plot_cam_ang_cov()])

        layout_plots.append([plot_update_timing(self.msckf_dfs), Spacer()])

        if self.is_extrinsic:
            layout_plots.append([self.plot_cam_nees(), Spacer()])

        tab_layout = layout(layout_plots, sizing_mode='stretch_width')
        tab = TabPanel(child=tab_layout, title=f"MSCKF {self.msckf_dfs[0].attrs['id']}")

        return tab
