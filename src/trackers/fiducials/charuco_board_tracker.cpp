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

#include "trackers/fiducials/charuco_board_tracker.hpp"

#include <vector>

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>

#include "trackers/fiducial_tracker.hpp"


CharucoBoardTracker::CharucoBoardTracker(FiducialTracker::Parameters params)
: FiducialTracker(params)
{
  m_board = cv::aruco::CharucoBoard::create(
    static_cast<int>(params.squares_x),
    static_cast<int>(params.squares_y),
    static_cast<float>(params.square_length),
    static_cast<float>(params.marker_length),
    m_dict
  );
}

bool CharucoBoardTracker::EstimatePoseBoard(
  const cv::Mat & img_in,
  cv::Mat & img_out,
  cv::Mat & camera_matrix,
  cv::Mat & dist_coefficients,
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
    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    cv::aruco::interpolateCornersCharuco(
      marker_corners, marker_ids, img_in, m_board, charuco_corners, charuco_ids, camera_matrix,
      dist_coefficients);

    // if at least one charuco corner detected
    if (charuco_ids.size() > 0) {
      cv::Scalar color = cv::Scalar(255, 0, 0);
      cv::aruco::drawDetectedCornersCharuco(img_out, charuco_corners, charuco_ids, color);
      bool valid = cv::aruco::estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, m_board,
        camera_matrix, dist_coefficients, r_vec, t_vec);
      // if charuco pose is valid
      if (valid) {
        cv::drawFrameAxes(img_out, camera_matrix, dist_coefficients, r_vec, t_vec, 0.1f);
        return true;
      }
    }
  }
  return false;
}
