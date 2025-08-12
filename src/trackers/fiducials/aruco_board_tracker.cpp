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

#include "trackers/fiducials/aruco_board_tracker.hpp"

#include <vector>

#include <opencv2/aruco.hpp>

#include "trackers/fiducial_tracker.hpp"


ArucoBoardTracker::ArucoBoardTracker(FiducialTracker::Parameters params)
: FiducialTracker(params)
{
  m_board = cv::aruco::GridBoard::create(
    static_cast<int>(params.squares_x),
    static_cast<int>(params.squares_y),
    static_cast<float>(params.square_length),
    static_cast<float>(params.marker_length),
    m_dict
  );
}

bool ArucoBoardTracker::EstimatePoseBoard(
  const cv::Mat & img_in,
  cv::Mat & img_out,
  cv::InputArray & camera_matrix,
  cv::InputArray & dist_coefficients,
  cv::Vec3d & r_vec,
  cv::Vec3d & t_vec
) const
{
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  cv::aruco::detectMarkers(img_in, m_dict, marker_corners, marker_ids);
  // if at least one marker detected
  if (marker_ids.size() > 0) {
    cv::aruco::drawDetectedMarkers(img_out, marker_corners, marker_ids);
    int valid = estimatePoseBoard(
      marker_corners, marker_ids, m_board, camera_matrix, dist_coefficients, r_vec, t_vec);
    // if at least one board marker detected
    if (valid > 0) {
      cv::drawFrameAxes(img_out, camera_matrix, dist_coefficients, r_vec, t_vec, 0.1);
      return true;
    }
  }
  return false;
}
