// Copyright 2023 Jacob Hartzer
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

#include "trackers/fiducial_tracker.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/opencv.hpp>

#include "trackers/tracker.hpp"
#include "utility/type_helper.hpp"

FiducialTracker::FiducialTracker(FiducialTracker::Parameters params)
: Tracker(params),
  m_fiducial_updater(
    params.id,
    params.camera_id,
    params.is_extrinsic,
    params.is_cam_extrinsic,
    params.log_directory,
    params.data_log_rate,
    params.logger
  ),
  m_detector_type(params.detector_type)
{
  m_pos_error = Eigen::Vector3d::Ones() * params.variance.pos;
  m_ang_error = Eigen::Vector3d::Ones() * params.variance.ang;

  auto dict_name = static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(params.predefined_dict);
  m_dict = cv::aruco::getPredefinedDictionary(dict_name);

  FidState fid_state;
  fid_state.SetIsExtrinsic(params.is_extrinsic);
  fid_state.pos_f_in_l = params.pos_f_in_l;
  fid_state.ang_f_to_l = params.ang_f_to_l;
  fid_state.id = params.id;
  Eigen::MatrixXd covariance(g_fid_extrinsic_state_size, g_fid_extrinsic_state_size);
  covariance.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * params.variance.pos;
  covariance.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * params.variance.ang;

  m_ekf->RegisterFiducial(fid_state, covariance);
}

void FiducialTracker::Track(
  double time,
  unsigned int frame_id,
  const cv::Mat & img_in,
  cv::Mat & img_out)
{
  cv::Ptr<cv::aruco::DetectorParameters> params = cv::makePtr<cv::aruco::DetectorParameters>();
  std::vector<int> marker_ids;
  cv::Mat camera_matrix = m_ekf->m_state.cam_states[m_camera_id].intrinsics.ToCameraMatrix();
  cv::Mat distortion = m_ekf->m_state.cam_states[m_camera_id].intrinsics.ToDistortionVector();

  cv::Vec3d r_vec, t_vec;
  bool valid = EstimatePoseBoard(
    img_in, img_out, camera_matrix, distortion, r_vec, t_vec);

  // if marker pose is valid
  if (valid) {
    cv::drawFrameAxes(img_out, camera_matrix, distortion, r_vec, t_vec, 0.5);

    Eigen::Vector3d pos_f_in_c = CvVectorToEigen(t_vec);
    Eigen::Quaterniond ang_f_to_c = RodriguesToQuat(r_vec);

    BoardDetection board_detection;
    board_detection.frame_id = frame_id;
    board_detection.pos_f_in_c = pos_f_in_c;
    board_detection.ang_f_to_c = ang_f_to_c;
    board_detection.pos_error = m_pos_error;
    board_detection.ang_error = m_ang_error;

    m_fiducial_updater.UpdateEKF(*m_ekf, time, board_detection);
  }
}
