// Copyright 2022 Jacob Hartzer
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

#include "sensors/sim/sim_camera.hpp"

#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "infrastructure/debug_logger.hpp"
#include "infrastructure/sim/truth_engine.hpp"
#include "sensors/camera.hpp"
#include "sensors/sim/sim_camera_message.hpp"
#include "ekf/types.hpp"
#include "sensors/sim/sim_sensor.hpp"
#include "trackers/sim/sim_feature_tracker.hpp"
#include "trackers/sim/sim_fiducial_tracker.hpp"
#include "utility/sim/sim_rng.hpp"
#include "utility/type_helper.hpp"


SimCamera::SimCamera(
  Parameters params,
  std::shared_ptr<TruthEngine> truth_engine)
: Camera(params.cam_params), SimSensor(params)
{
  m_pos_error = params.pos_error;
  m_ang_error = params.ang_error;
  m_truth = truth_engine;
  m_room_size = params.room_size;
  m_generate_video = params.generate_video;
  if (!params.cam_params.log_directory.empty()) {
    m_video_path = params.cam_params.log_directory + "/" + params.cam_params.name + ".avi";
  } else {
    m_video_path = params.cam_params.name + ".avi";
  }

  // Set true camera values
  Eigen::Vector3d pos_c_in_b_true;
  Eigen::Quaterniond ang_c_to_b_true;
  if (m_no_errors) {
    pos_c_in_b_true = params.cam_params.pos_c_in_b;
    ang_c_to_b_true = params.cam_params.ang_c_to_b;
  } else {
    pos_c_in_b_true = SimRNG::VecNormRand(params.cam_params.pos_c_in_b, params.pos_error);
    ang_c_to_b_true = SimRNG::QuatNormRand(params.cam_params.ang_c_to_b, params.ang_error);
  }

  truth_engine->SetCameraPosition(m_id, pos_c_in_b_true);
  truth_engine->SetCameraAngularPosition(m_id, ang_c_to_b_true);
  truth_engine->SetCameraIntrinsics(m_id, params.cam_params.intrinsics);

  if (m_generate_video) {
    cv::Size frame_size(
      static_cast<int>(params.cam_params.intrinsics.width),
      static_cast<int>(params.cam_params.intrinsics.height));
    m_video_writer.open(
      m_video_path,
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
      std::max(m_rate, 1.0),
      frame_size,
      true);
    if (!m_video_writer.isOpened()) {
      std::stringstream msg;
      msg << "Failed to open simulated video output: " << m_video_path;
      m_logger->Log(LogLevel::WARN, msg.str());
      m_generate_video = false;
    }
  }
}

std::vector<std::shared_ptr<SimCameraMessage>> SimCamera::GenerateMessages()
{
  std::vector<std::shared_ptr<SimCameraMessage>> messages;
  std::vector<double> measurement_times = GenerateMeasurementTimes(m_rate);
  std::map<unsigned int, cv::Mat> frame_buffer;
  unsigned int max_track_length = 0;
  for (const auto & tracker_iter : m_trackers) {
    max_track_length = std::max(max_track_length, tracker_iter.second->GetMaxTrackLength());
  }

  m_logger->Log(
    LogLevel::INFO, "Generating " + std::to_string(measurement_times.size()) + " Camera frames");

  for (double measurement_time : measurement_times) {
    unsigned int frame_id = GenerateFrameID();
    cv::Mat blank_img;
    if (m_generate_video) {
      frame_buffer[frame_id] = RenderFrame(measurement_time);
    }
    auto cam_msg = std::make_shared<SimCameraMessage>(blank_img);
    cam_msg->sensor_id = m_id;
    cam_msg->sensor_type = SensorType::Camera;
    cam_msg->time = measurement_time;
    cam_msg->frame_id = frame_id;

    // Tracker Messages
    for (auto const & trk_iter : m_trackers) {
      auto trk_msg = m_trackers[trk_iter.first]->GenerateMessage(measurement_time, frame_id);
      cam_msg->feature_track_messages.push_back(trk_msg);
      if (m_generate_video) {
        for (const auto & feature_track : trk_msg->feature_tracks) {
          OverlayFeatureTrack(frame_buffer, feature_track, trk_msg->tracker_id);
        }
      }
    }

    // Fiducial Messages
    for (auto const & fid_iter : m_fiducials) {
      auto fid_msg = m_fiducials[fid_iter.first]->GenerateMessage(measurement_time, frame_id);
      cam_msg->fiducial_track_messages.push_back(fid_msg);
    }

    messages.push_back(cam_msg);
    FlushReadyFrames(frame_buffer, frame_id, max_track_length, false);
  }

  FlushReadyFrames(frame_buffer, 0, max_track_length, true);
  if (m_video_writer.isOpened()) {
    m_video_writer.release();
  }
  return messages;
}

void SimCamera::AddTracker(std::shared_ptr<SimFeatureTracker> tracker)
{
  m_trackers[tracker->GetID()] = tracker;
}

void SimCamera::AddFiducial(std::shared_ptr<SimFiducialTracker> fiducial)
{
  m_fiducials[fiducial->GetID()] = fiducial;
}

void SimCamera::Callback(const SimCameraMessage & sim_camera_message)
{
  double local_time = m_ekf->CalculateLocalTime(sim_camera_message.time);
  m_ekf->PredictModel(local_time);
  m_ekf->AugmentStateIfNeeded(m_id, sim_camera_message.frame_id);

  for (auto feature_track_message : sim_camera_message.feature_track_messages) {
    if (!feature_track_message->feature_tracks.empty()) {
      m_trackers[feature_track_message->tracker_id]->Callback(
        sim_camera_message.time, *feature_track_message);
    }
  }
  for (auto fiducial_track_message : sim_camera_message.fiducial_track_messages) {
    m_fiducials[fiducial_track_message->tracker_id]->Callback(
      sim_camera_message.time, *fiducial_track_message);
  }
}

cv::Scalar SimCamera::GetTrackColor(unsigned int tracker_id, unsigned int feature_id)
{
  unsigned int seed = tracker_id * 2654435761U + feature_id * 2246822519U;
  return cv::Scalar(
    64 + static_cast<int>(seed % 192U),
    64 + static_cast<int>((seed / 7U) % 192U),
    64 + static_cast<int>((seed / 17U) % 192U));
}

bool SimCamera::ShouldShowTrack(unsigned int feature_id)
{
  return SHOW_NTH_TRACK <= 1 || (feature_id % SHOW_NTH_TRACK) == 0;
}

void SimCamera::OverlayFeatureTrack(
  std::map<unsigned int, cv::Mat> & frame_buffer,
  const FeatureTrack & feature_track,
  unsigned int tracker_id
) const
{
  if (feature_track.empty()) {
    return;
  }

  unsigned int feature_id = 0;
  if (feature_track.front().key_point.class_id >= 0) {
    feature_id = static_cast<unsigned int>(feature_track.front().key_point.class_id);
  }
  if (!ShouldShowTrack(feature_id)) {
    return;
  }
  cv::Scalar color = GetTrackColor(tracker_id, feature_id);

  std::vector<cv::Point> history_points;
  history_points.reserve(feature_track.size());
  for (const auto & feature_point : feature_track) {
    history_points.emplace_back(
      cvRound(feature_point.key_point.pt.x), cvRound(feature_point.key_point.pt.y));
  }

  for (unsigned int idx = 0; idx < feature_track.size(); ++idx) {
    auto frame_iter = frame_buffer.find(feature_track[idx].frame_id);
    if (frame_iter == frame_buffer.end()) {
      continue;
    }

    std::vector<cv::Point> partial_history(
      history_points.begin(), history_points.begin() + static_cast<long>(idx + 1));
    if (partial_history.size() > 1) {
      cv::polylines(frame_iter->second, partial_history, false, color, 1, cv::LINE_AA);
    }
    for (const auto & point : partial_history) {
      cv::circle(frame_iter->second, point, 2, color, cv::FILLED, cv::LINE_AA);
    }
  }
}

void SimCamera::FlushReadyFrames(
  std::map<unsigned int, cv::Mat> & frame_buffer,
  unsigned int current_frame_id,
  unsigned int max_track_length,
  bool flush_all
)
{
  if (!m_generate_video || !m_video_writer.isOpened()) {
    return;
  }

  while (!frame_buffer.empty()) {
    auto frame_iter = frame_buffer.begin();
    bool should_flush = flush_all;
    if (!flush_all) {
      if (max_track_length == 0) {
        should_flush = true;
      } else if (current_frame_id < max_track_length) {
        should_flush = false;
      } else {
        should_flush = frame_iter->first <= current_frame_id - max_track_length;
      }
    }

    if (!should_flush) {
      break;
    }

    m_video_writer.write(frame_iter->second);
    frame_buffer.erase(frame_iter);
  }
}

cv::Mat SimCamera::RenderFrame(double time) const
{
  Intrinsics intrinsics = m_truth->GetCameraIntrinsics(m_id);
  cv::Mat image = cv::Mat::zeros(
    static_cast<int>(intrinsics.height),
    static_cast<int>(intrinsics.width),
    CV_8UC3);

  Eigen::Vector3d pos_b_in_l = m_truth->GetBodyPosition(time);
  Eigen::Quaterniond ang_b_to_l = m_truth->GetBodyAngularPosition(time);
  Eigen::Vector3d pos_c_in_b = m_truth->GetCameraPosition(m_id);
  Eigen::Quaterniond ang_c_to_b = m_truth->GetCameraAngularPosition(m_id);
  Eigen::Matrix3d rot_l_to_c = (ang_b_to_l * ang_c_to_b).toRotationMatrix().transpose();

  cv::Mat ang_l_to_c_cv(3, 3, cv::DataType<double>::type);
  EigenMatrixToCv(rot_l_to_c, ang_l_to_c_cv);

  cv::Mat r_vec(3, 1, cv::DataType<double>::type);
  cv::Rodrigues(ang_l_to_c_cv, r_vec);

  Eigen::Vector3d pos_l_in_c = rot_l_to_c * (-(pos_b_in_l + ang_b_to_l * pos_c_in_b));

  cv::Mat t_vec(3, 1, cv::DataType<double>::type);
  t_vec.at<double>(0) = pos_l_in_c[0];
  t_vec.at<double>(1) = pos_l_in_c[1];
  t_vec.at<double>(2) = pos_l_in_c[2];

  double half_width = static_cast<double>(m_room_size) / 2.0;
  double half_height = static_cast<double>(m_room_size) / 4.0;
  double grid_spacing = 0.5;

  std::vector<Eigen::Vector3d> room_corners = {
    {-half_width, -half_width, -half_height},
    {half_width, -half_width, -half_height},
    {half_width, half_width, -half_height},
    {-half_width, half_width, -half_height},
    {-half_width, -half_width, half_height},
    {half_width, -half_width, half_height},
    {half_width, half_width, half_height},
    {-half_width, half_width, half_height}};
  const std::vector<std::pair<int, int>> room_edges = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}};

  for (double value = -half_width; value <= half_width + 1.0e-9; value += grid_spacing) {
    DrawWorldLine(
      image,
      Eigen::Vector3d(value, -half_width, -half_height),
      Eigen::Vector3d(value, half_width, -half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(-half_width, value, -half_height),
      Eigen::Vector3d(half_width, value, -half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(value, -half_width, half_height),
      Eigen::Vector3d(value, half_width, half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(-half_width, value, half_height),
      Eigen::Vector3d(half_width, value, half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(value, -half_width, -half_height),
      Eigen::Vector3d(value, -half_width, half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(value, half_width, -half_height),
      Eigen::Vector3d(value, half_width, half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(-half_width, value, -half_height),
      Eigen::Vector3d(-half_width, value, half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(half_width, value, -half_height),
      Eigen::Vector3d(half_width, value, half_height),
      r_vec,
      t_vec,
      intrinsics,
      1);
  }
  for (double value = -half_height; value <= half_height + 1.0e-9; value += grid_spacing) {
    DrawWorldLine(
      image,
      Eigen::Vector3d(-half_width, -half_width, value),
      Eigen::Vector3d(half_width, -half_width, value),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(-half_width, half_width, value),
      Eigen::Vector3d(half_width, half_width, value),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(-half_width, -half_width, value),
      Eigen::Vector3d(-half_width, half_width, value),
      r_vec,
      t_vec,
      intrinsics,
      1);
    DrawWorldLine(
      image,
      Eigen::Vector3d(half_width, -half_width, value),
      Eigen::Vector3d(half_width, half_width, value),
      r_vec,
      t_vec,
      intrinsics,
      1);
  }

  for (const auto & edge : room_edges) {
    DrawWorldLine(
      image,
      room_corners[edge.first],
      room_corners[edge.second],
      r_vec,
      t_vec,
      intrinsics,
      2);
  }

  for (const auto & fid_iter : m_fiducials) {
    std::vector<cv::Point3d> corners = fid_iter.second->GetBoardCornersInLocalFrame();
    if (corners.size() < 4) {
      continue;
    }
    for (unsigned int i = 0; i < corners.size(); ++i) {
      DrawProjectedLine(
        image,
        corners[i],
        corners[(i + 1) % corners.size()],
        r_vec,
        t_vec,
        intrinsics,
        cv::Scalar(255, 255, 255),
        2);
    }
  }

  return image;
}

void SimCamera::DrawWorldLine(
  cv::Mat & image,
  const Eigen::Vector3d & start,
  const Eigen::Vector3d & end,
  const cv::Mat & r_vec,
  const cv::Mat & t_vec,
  const Intrinsics & intrinsics,
  int thickness
) const
{
  DrawProjectedLine(
    image,
    cv::Point3d(start.x(), start.y(), start.z()),
    cv::Point3d(end.x(), end.y(), end.z()),
    r_vec,
    t_vec,
    intrinsics,
    cv::Scalar(255, 255, 255),
    thickness);
}

bool SimCamera::DrawProjectedLine(
  cv::Mat & image,
  const cv::Point3d & point_1,
  const cv::Point3d & point_2,
  const cv::Mat & r_vec,
  const cv::Mat & t_vec,
  const Intrinsics & intrinsics,
  const cv::Scalar & color,
  int thickness
) const
{
  std::vector<cv::Point3d> world_points{point_1, point_2};
  cv::Mat rotation_matrix;
  cv::Rodrigues(r_vec, rotation_matrix);

  std::vector<cv::Point3d> camera_points;
  camera_points.reserve(world_points.size());
  for (const auto & world_point : world_points) {
    cv::Mat point_mat = (cv::Mat_<double>(3, 1) << world_point.x, world_point.y, world_point.z);
    cv::Mat camera_point = rotation_matrix * point_mat + t_vec;
    camera_points.emplace_back(
      camera_point.at<double>(0),
      camera_point.at<double>(1),
      camera_point.at<double>(2));
  }

  if (camera_points[0].z <= 0.0 || camera_points[1].z <= 0.0) {
    return false;
  }

  std::vector<cv::Point2d> projected_points;
  cv::projectPoints(
    world_points,
    r_vec,
    t_vec,
    intrinsics.ToCameraMatrix(),
    intrinsics.ToDistortionVector(),
    projected_points);
  if (projected_points.size() != 2) {
    return false;
  }

  const cv::Point point_1_px(cvRound(projected_points[0].x), cvRound(projected_points[0].y));
  const cv::Point point_2_px(cvRound(projected_points[1].x), cvRound(projected_points[1].y));
  cv::line(image, point_1_px, point_2_px, color, thickness, cv::LINE_AA);
  return true;
}
