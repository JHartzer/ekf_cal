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
import numpy as np
from scipy.stats.distributions import chi2
from utilities import (
    calculate_alpha,
    get_colors,
    plot_timing_alignment_error,
    plot_timing_offsets,
    plot_update_timing,
)


class tab_gps:
    """Class for plotting GPS data."""

    def __init__(self, gps_dfs, body_truth_dfs, args, err_dfs=None, rate=None, timing_dfs=None):
        """Initialize the tab_gps class for plotting GPS information."""
        self.gps_dfs = gps_dfs
        self.body_truth_dfs = body_truth_dfs
        self.is_extrinsic = 'gps_cov_0' in self.gps_dfs[0].keys()
        self.alpha = calculate_alpha(len(self.gps_dfs))
        self.colors = get_colors(args)
        self.err_dfs = err_dfs if err_dfs is not None else {}
        self.rate = rate
        self.timing_dfs = timing_dfs if timing_dfs is not None else []

    def plot_gps_measurements(self):
        """Plot camera GPS measurements."""
        fig = figure(
            width=400,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position [m]',
            title='GPS Measurements')
        for gps_df in self.gps_dfs:
            t_gps = gps_df['time']
            fig.line(t_gps, gps_df['x'], alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(t_gps, gps_df['y'], alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(t_gps, gps_df['z'], alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_gps_residuals(self):
        """Plot camera GPS residuals."""
        fig = figure(
            width=400,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Residual [m]',
            title='GPS Residuals')
        for gps_df in self.gps_dfs:
            t_gps = gps_df['time']
            res_x = gps_df['residual_0']
            res_y = gps_df['residual_1']
            res_z = gps_df['residual_2']
            fig.line(t_gps, res_x, alpha=self.alpha, color=self.colors[0], legend_label='X')
            fig.line(t_gps, res_y, alpha=self.alpha, color=self.colors[1], legend_label='Y')
            fig.line(t_gps, res_z, alpha=self.alpha, color=self.colors[2], legend_label='Z')
        return fig

    def plot_ant_pos_error(self):
        """Plot camera GPS residuals."""
        fig = figure(
            width=400,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Error [mm]',
            title='GPS Antenna Position Error')
        gps_pos_err_list = self.err_dfs.get('gps_pos_err', [])
        for err_df in gps_pos_err_list:
            time = err_df['time']
            fig.line(time, err_df['x'] * 1e3, alpha=self.alpha,
                     color=self.colors[0], legend_label='X')
            fig.line(time, err_df['y'] * 1e3, alpha=self.alpha,
                     color=self.colors[1], legend_label='Y')
            fig.line(time, err_df['z'] * 1e3, alpha=self.alpha,
                     color=self.colors[2], legend_label='Z')
        return fig

    def plot_gps_cov(self):
        """Plot GPS antenna position covariance."""
        fig = figure(
            width=800,
            height=300,
            x_axis_label='Time [s]',
            y_axis_label='Position Covariance [mm]',
            title='GPS Antenna Position Covariance')
        for gps_df in self.gps_dfs:
            t_gps = gps_df['time']
            gps_int_cov_3 = np.array(gps_df['gps_cov_0']) * 1e3
            gps_int_cov_4 = np.array(gps_df['gps_cov_1']) * 1e3
            gps_int_cov_5 = np.array(gps_df['gps_cov_2']) * 1e3
            fig.line(
                t_gps,
                gps_int_cov_3,
                alpha=self.alpha,
                color=self.colors[0],
                legend_label='X')
            fig.line(
                t_gps,
                gps_int_cov_4,
                alpha=self.alpha,
                color=self.colors[1],
                legend_label='Y')
            fig.line(
                t_gps,
                gps_int_cov_5,
                alpha=self.alpha,
                color=self.colors[2],
                legend_label='Z')
        return fig

    def plot_gps_nees(self):
        """Plot GPS normalized estimation error squared."""
        fig = figure(width=800, height=300, x_axis_label='Time [s]',
                     y_axis_label='NEES', title='Normalized Estimation Error Squared')
        gps_nees_list = self.err_dfs.get('gps_nees', [])
        for err_df in gps_nees_list:
            fig.line(err_df['time'], err_df['nees'], alpha=self.alpha, color=self.colors[0])

        fig.hspan(y=chi2.ppf(0.025, df=3), line_color='red')
        fig.hspan(y=chi2.ppf(0.975, df=3), line_color='red')
        fig.y_range = Range1d(0, 15)

        return fig

    def get_tab(self):
        """Generate the Bokeh TabPanel containing all GPS plots."""
        layout_plots = [[self.plot_gps_measurements(), self.plot_gps_residuals()]]

        if self.is_extrinsic:
            layout_plots.append([self.plot_ant_pos_error(), self.plot_gps_cov()])

        layout_plots.append([plot_update_timing(self.gps_dfs, self.rate), Spacer()])

        if self.timing_dfs:
            layout_plots.append([
                plot_timing_offsets(self.timing_dfs),
                plot_timing_alignment_error(self.timing_dfs)])

        if self.is_extrinsic:
            layout_plots.append([self.plot_gps_nees(), Spacer()])

        tab_layout = layout(layout_plots, sizing_mode='stretch_width')
        tab = TabPanel(child=tab_layout,
                       title=f"GPS {self.gps_dfs[0].attrs['id']}")

        return tab
