// Copyright 2025 Jacob Hartzer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef TRACKERS__FIDUCIALS__APRIL_GRID_TRACKER_HPP_
#define TRACKERS__FIDUCIALS__APRIL_GRID_TRACKER_HPP_

#include "trackers/fiducial_tracker.hpp"

#include <aprilgrid.hpp>
#include <opencv2/aruco.hpp>

///
/// @class AprilGridTracker
/// @brief AprilGridTracker Tracker Class
///
class AprilGridTracker : public FiducialTracker
{
public:
  explicit AprilGridTracker(FiducialTracker::Parameters params);

  AprilGrid m_board;

private:
  bool EstimatePoseBoard(
    const cv::Mat & img_in,
    cv::Mat & img_out,
    cv::Mat & camera_matrix,
    cv::Mat & dist_coefficients,
    cv::Vec3d & r_vec,
    cv::Vec3d & t_vec) const;
};

#endif  // TRACKERS__FIDUCIALS__APRIL_GRID_TRACKER_HPP_
