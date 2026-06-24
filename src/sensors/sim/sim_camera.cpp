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

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>

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
  std::vector<TimingSample> measurement_times = GenerateMeasurementTimes(m_rate);
  std::map<unsigned int, cv::Mat> frame_buffer;
  unsigned int max_track_length = 0;
  for (const auto & tracker_iter : m_trackers) {
    max_track_length = std::max(max_track_length, tracker_iter.second->GetMaxTrackLength());
  }

  m_logger->Log(
    LogLevel::INFO, "Generating " + std::to_string(measurement_times.size()) + " Camera frames");

  for (const auto & measurement_time : measurement_times) {
    unsigned int frame_id = GenerateFrameID();
    cv::Mat blank_img;
    if (m_generate_video) {
      frame_buffer[frame_id] = RenderFrame(measurement_time.time_true);
    }
    auto cam_msg = std::make_shared<SimCameraMessage>(blank_img);
    cam_msg->sensor_id = m_id;
    cam_msg->sensor_type = SensorType::Camera;
    cam_msg->time_true = measurement_time.time_true;
    cam_msg->time_measured = measurement_time.time_measured;
    cam_msg->time_received = measurement_time.time_received;
    cam_msg->frame_id = frame_id;

    // Tracker Messages
    for (auto const & trk_iter : m_trackers) {
      auto trk_msg = m_trackers[trk_iter.first]->GenerateMessage(
        measurement_time.time_true,
        frame_id);
      cam_msg->feature_track_messages.push_back(trk_msg);
      if (m_generate_video) {
        unsigned int tracker_max_track_length = m_trackers[trk_iter.first]->GetMaxTrackLength();
        for (const auto & feature_track : trk_msg->feature_tracks) {
          OverlayBufferedFeatureTrack(
            frame_buffer,
            feature_track,
            trk_msg->tracker_id,
            tracker_max_track_length);
        }
      }
    }

    // Fiducial Messages
    for (auto const & fid_iter : m_fiducials) {
      auto fid_msg = m_fiducials[fid_iter.first]->GenerateMessage(
        measurement_time.time_true,
        frame_id);
      cam_msg->fiducial_track_messages.push_back(fid_msg);
    }

    messages.push_back(cam_msg);
    FlushReadyFrames(frame_buffer, frame_id, max_track_length, false);
  }

  if (m_generate_video) {
    for (const auto & tracker_iter : m_trackers) {
      FeatureTracks active_feature_tracks = tracker_iter.second->GetActiveFeatureTracks();
      unsigned int tracker_max_track_length = tracker_iter.second->GetMaxTrackLength();
      for (const auto & feature_track : active_feature_tracks) {
        OverlayBufferedFeatureTrack(
          frame_buffer,
          feature_track,
          tracker_iter.second->GetID(),
          tracker_max_track_length);
      }
    }
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
  BufferMessage(
    sim_camera_message,
    m_message_buffer,
    [this](const SimCameraMessage & buffered_message) {ExecuteCallback(buffered_message);});
}

void SimCamera::ExecuteCallback(const SimCameraMessage & sim_camera_message)
{
  LogTiming(sim_camera_message);
  double local_time = m_ekf->CalculateLocalTime(sim_camera_message.time_used);
  m_ekf->PredictModel(local_time);

  for (auto fiducial_track_message : sim_camera_message.fiducial_track_messages) {
    m_fiducials[fiducial_track_message->tracker_id]->Callback(
      sim_camera_message.time_used, *fiducial_track_message);
  }

  m_ekf->AugmentStateIfNeeded(m_id, sim_camera_message.frame_id);

  for (auto feature_track_message : sim_camera_message.feature_track_messages) {
    if (!feature_track_message->feature_tracks.empty()) {
      m_trackers[feature_track_message->tracker_id]->Callback(
        sim_camera_message.time_used, *feature_track_message);
    }
  }
}

void SimCamera::Flush()
{
  FlushBufferedMessages(
    m_message_buffer,
    [this](const SimCameraMessage & buffered_message) {ExecuteCallback(buffered_message);});
}

bool SimCamera::HasBufferedMeasurements() const
{
  return HasBufferedMessages(m_message_buffer);
}

double SimCamera::GetNextBufferedMeasurementTime() const
{
  return GetNextBufferedMessageTime(m_message_buffer);
}

bool SimCamera::FlushNextMeasurement()
{
  return FlushNextBufferedMessage(
    m_message_buffer,
    [this](const SimCameraMessage & buffered_message) {ExecuteCallback(buffered_message);});
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

double SimCamera::GetTrackAlpha(unsigned int age, unsigned int max_track_length)
{
  if (max_track_length == 0) {
    return 1.0;
  }

  double alpha = 1.0 - (static_cast<double>(age) / static_cast<double>(max_track_length));
  return std::max(0.0, std::min(alpha, 1.0));
}

bool SimCamera::GetFeatureDepth(double time, int feature_id, double & depth) const
{
  if (feature_id < 0) {
    return false;
  }

  cv::Point3d feature_point = m_truth->GetFeature(feature_id);
  Eigen::Vector3d pos_b_in_l = m_truth->GetBodyPosition(time);
  Eigen::Quaterniond ang_b_to_l = m_truth->GetBodyAngularPosition(time);
  Eigen::Vector3d pos_c_in_b = m_truth->GetCameraPosition(m_id);
  Eigen::Quaterniond ang_c_to_b = m_truth->GetCameraAngularPosition(m_id);
  Eigen::Matrix3d rot_l_to_c = (ang_b_to_l * ang_c_to_b).toRotationMatrix().transpose();
  Eigen::Vector3d pos_l_in_c = rot_l_to_c * (-(pos_b_in_l + ang_b_to_l * pos_c_in_b));
  Eigen::Vector3d point_in_l(feature_point.x, feature_point.y, feature_point.z);
  Eigen::Vector3d point_in_c = rot_l_to_c * point_in_l + pos_l_in_c;

  depth = point_in_c.z();
  return depth > 0.0;
}

int SimCamera::GetFeatureRadius(double depth)
{
  constexpr int min_feature_radius = 2;
  constexpr int max_feature_radius = 8;
  if (depth <= 0.0) {
    return min_feature_radius;
  }

  double scaled_radius = static_cast<double>(max_feature_radius) / depth;
  int radius = static_cast<int>(std::round(scaled_radius));
  return std::max(min_feature_radius, std::min(radius, max_feature_radius));
}

void SimCamera::BlendCircle(
  cv::Mat & image,
  const cv::Point & center,
  int radius,
  const cv::Scalar & color,
  double alpha
) const
{
  if (alpha <= 0.0 || radius <= 0) {
    return;
  }

  int padding = radius + 2;
  cv::Rect roi(
    std::max(0, center.x - padding),
    std::max(0, center.y - padding),
    std::min(image.cols, center.x + padding + 1) - std::max(0, center.x - padding),
    std::min(image.rows, center.y + padding + 1) - std::max(0, center.y - padding));
  if (roi.width <= 0 || roi.height <= 0) {
    return;
  }

  cv::Mat overlay = image(roi).clone();
  cv::circle(
    overlay,
    cv::Point(center.x - roi.x, center.y - roi.y),
    radius,
    color,
    cv::FILLED,
    cv::LINE_AA);
  cv::addWeighted(overlay, alpha, image(roi), 1.0 - alpha, 0.0, image(roi));
}

void SimCamera::BlendLine(
  cv::Mat & image,
  const cv::Point & point_1,
  const cv::Point & point_2,
  const cv::Scalar & color,
  double alpha,
  int thickness
) const
{
  if (alpha <= 0.0) {
    return;
  }

  int padding = thickness + 2;
  int min_x = std::max(0, std::min(point_1.x, point_2.x) - padding);
  int min_y = std::max(0, std::min(point_1.y, point_2.y) - padding);
  int max_x = std::min(image.cols - 1, std::max(point_1.x, point_2.x) + padding);
  int max_y = std::min(image.rows - 1, std::max(point_1.y, point_2.y) + padding);
  cv::Rect roi(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
  if (roi.width <= 0 || roi.height <= 0) {
    return;
  }

  cv::Mat overlay = image(roi).clone();
  cv::line(
    overlay,
    cv::Point(point_1.x - roi.x, point_1.y - roi.y),
    cv::Point(point_2.x - roi.x, point_2.y - roi.y),
    color,
    thickness,
    cv::LINE_AA);
  cv::addWeighted(overlay, alpha, image(roi), 1.0 - alpha, 0.0, image(roi));
}

void SimCamera::OverlayBufferedFeatureTrack(
  std::map<unsigned int, cv::Mat> & frame_buffer,
  const FeatureTrack & feature_track,
  unsigned int tracker_id,
  unsigned int max_track_length
) const
{
  if (feature_track.empty()) {
    return;
  }

  int class_id = feature_track.back().key_point.class_id;
  if (class_id < 0) {
    return;
  }

  unsigned int feature_id = static_cast<unsigned int>(class_id);
  if (!ShouldShowTrack(feature_id)) {
    return;
  }

  cv::Scalar color = GetTrackColor(tracker_id, feature_id);
  for (unsigned int idx = 0; idx < feature_track.size(); ++idx) {
    auto frame_iter = frame_buffer.find(feature_track[idx].frame_id);
    if (frame_iter == frame_buffer.end()) {
      continue;
    }

    cv::Mat & image = frame_iter->second;
    for (unsigned int seg_idx = 1; seg_idx <= idx; ++seg_idx) {
      unsigned int age = idx - seg_idx;
      double alpha = GetTrackAlpha(age, max_track_length);
      cv::Point point_1(
        cvRound(feature_track[seg_idx - 1].key_point.pt.x),
        cvRound(feature_track[seg_idx - 1].key_point.pt.y));
      cv::Point point_2(
        cvRound(feature_track[seg_idx].key_point.pt.x),
        cvRound(feature_track[seg_idx].key_point.pt.y));
      BlendLine(image, point_1, point_2, color, alpha, 2);
    }

    double depth = 0.0;
    if (GetFeatureDepth(feature_track[idx].frame_time, class_id, depth)) {
      cv::Point center(
        cvRound(feature_track[idx].key_point.pt.x),
        cvRound(feature_track[idx].key_point.pt.y));
      BlendCircle(image, center, GetFeatureRadius(depth), color, 1.0);
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
