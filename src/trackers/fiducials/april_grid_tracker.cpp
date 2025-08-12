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

#include "trackers/fiducials/april_grid_tracker.hpp"

#include <vector>

#include <aprilgrid.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include "trackers/fiducial_tracker.hpp"


AprilGridTracker::AprilGridTracker(FiducialTracker::Parameters params)
: FiducialTracker(params),
  m_board(
    cv::Size(params.squares_x, params.squares_y),
    params.marker_length,
    params.border_bits,
    params.separation_bits,
    params.predefined_dict,
    params.starting_id
  ) {}

bool AprilGridTracker::EstimatePoseBoard(
  const cv::Mat & img_in,
  cv::Mat & img_out,
  cv::Mat & camera_matrix,
  cv::Mat & dist_coefficients,
  cv::Vec3d & r_vec,
  cv::Vec3d & t_vec
) const
{

  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  m_board.detectAprilTags(img_in, corners, ids);

  if (ids.size() > 0) {

    std::vector<cv::Point3f> obj_points;
    std::vector<cv::Point2f> img_points;
    m_board.matchImagePoints(corners, ids, obj_points, img_points);

    bool valid = cv::solvePnP(
      obj_points, img_points, camera_matrix, dist_coefficients, r_vec, t_vec);

    if (valid) {
      cv::cvtColor(img_in, img_out, cv::COLOR_GRAY2BGR);
      AprilGrid::drawDetectedTags(img_out, ids, img_points);
      AprilGrid::drawReprojectionErrors(
        img_out, ids, obj_points, img_points, r_vec, t_vec, camera_matrix, dist_coefficients);
      cv::drawFrameAxes(img_out, camera_matrix, dist_coefficients, r_vec, t_vec, 5.0);
      return true;
    }
  }
  return false;
}
