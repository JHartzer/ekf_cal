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

#include "infrastructure/sim/truth_engine.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ekf/constants.hpp"
#include "ekf/types.hpp"
#include "infrastructure/data_logger.hpp"
#include "infrastructure/debug_logger.hpp"
#include "utility/sim/sim_rng.hpp"
#include "utility/string_helper.hpp"

TruthEngine::TruthEngine(double max_time, std::shared_ptr<DebugLogger> logger)
: m_max_time(max_time), m_logger(logger)
{
  GenerateGridFeatures();
}

Eigen::Vector3d TruthEngine::GetImuPosition(unsigned int sensor_id)
{
  return m_imu_pos[sensor_id];
}

Eigen::Quaterniond TruthEngine::GetImuAngularPosition(unsigned int sensor_id)
{
  return m_imu_ang_pos[sensor_id];
}

Eigen::Vector3d TruthEngine::GetImuAccelerometerBias(unsigned int sensor_id)
{
  return m_imu_acc_bias[sensor_id];
}

Eigen::Vector3d TruthEngine::GetImuGyroscopeBias(unsigned int sensor_id)
{
  return m_imu_gyro_bias[sensor_id];
}

Eigen::Vector3d TruthEngine::GetCameraPosition(unsigned int sensor_id)
{
  return m_cam_pos[sensor_id];
}

Eigen::Quaterniond TruthEngine::GetCameraAngularPosition(unsigned int sensor_id)
{
  return m_cam_ang_pos[sensor_id];
}

Eigen::Vector3d TruthEngine::GetGpsPosition(unsigned int sensor_id)
{
  return m_gps_pos[sensor_id];
}
Eigen::Vector3d TruthEngine::GetLocalPosition() const
{
  return m_lla_reference;
}
double TruthEngine::GetLocalHeading() const
{
  return m_heading;
}

void TruthEngine::SetImuPosition(unsigned int sensor_id, const Eigen::Vector3d & imu_pos)
{
  m_imu_pos[sensor_id] = imu_pos;
}

void TruthEngine::SetRoomSize(int room_size)
{
  m_room_size = static_cast<double>(room_size);
  m_feature_points.clear();
  m_feature_points_map.clear();
  GenerateGridFeatures();
}

void TruthEngine::SetImuAngularPosition(
  unsigned int sensor_id,
  const Eigen::Quaterniond & imu_ang_pos)
{
  m_imu_ang_pos[sensor_id] = imu_ang_pos;
}

void TruthEngine::SetImuAccelerometerBias(
  unsigned int sensor_id,
  const Eigen::Vector3d & imu_acc_bias)
{
  m_imu_acc_bias[sensor_id] = imu_acc_bias;
}

void TruthEngine::SetImuGyroscopeBias(unsigned int sensor_id, const Eigen::Vector3d & imu_gyro_bias)
{
  m_imu_gyro_bias[sensor_id] = imu_gyro_bias;
}

void TruthEngine::SetCameraPosition(unsigned int sensor_id, const Eigen::Vector3d & cam_pos)
{
  m_cam_pos[sensor_id] = cam_pos;
}

void TruthEngine::SetCameraAngularPosition(
  unsigned int sensor_id,
  const Eigen::Quaterniond & cam_ang_pos)
{
  m_cam_ang_pos[sensor_id] = cam_ang_pos;
}

void TruthEngine::SetBoardPosition(unsigned int board_id, const Eigen::Vector3d & board_position)
{
  m_board_pos[board_id] = board_position;
}

void TruthEngine::SetGpsPosition(unsigned int sensor_id, const Eigen::Vector3d & gps_position)
{
  m_gps_pos[sensor_id] = gps_position;
}

void TruthEngine::SetLocalPosition(const Eigen::Vector3d & lla_reference)
{
  m_lla_reference = lla_reference;
}

void TruthEngine::SetLocalHeading(double heading)
{
  m_heading = heading;
}

Eigen::Vector3d TruthEngine::GetBoardPosition(unsigned int board_id)
{
  return m_board_pos[board_id];
}

void TruthEngine::SetBoardOrientation(
  unsigned int board_id,
  const Eigen::Quaterniond & board_orientation)
{
  m_board_ang[board_id] = board_orientation;
}

Eigen::Quaterniond TruthEngine::GetBoardOrientation(unsigned int board_id)
{
  return m_board_ang[board_id];
}

/// @todo Eventually remove this function as it's unecessary
void TruthEngine::GenerateGridFeatures()
{
  double room_half_width = m_room_size / 2.0;
  double room_half_height = m_room_size / 4.0;
  for (int i = 0; i < m_grid_size; ++i) {
    for (int j = 0; j < m_grid_size; ++j) {
      double grid_size_double = static_cast<double>(m_grid_size);
      double grid_xy =
        (static_cast<double>(i) / grid_size_double * m_room_size) - room_half_width;
      double grid_z =
        (static_cast<double>(j) / grid_size_double * (2.0 * room_half_height)) - room_half_height;
      double grid_y =
        (static_cast<double>(j) / grid_size_double * m_room_size) - room_half_width;
      m_feature_points.emplace_back(room_half_width, grid_xy, grid_z);
      m_feature_points.emplace_back(-room_half_width, grid_xy, grid_z);
      m_feature_points.emplace_back(grid_xy, room_half_width, grid_z);
      m_feature_points.emplace_back(grid_xy, -room_half_width, grid_z);
      m_feature_points.emplace_back(grid_xy, grid_y, room_half_height);
      m_feature_points.emplace_back(grid_xy, grid_y, -room_half_height);
    }
  }

  for (unsigned int i = 0; i < m_feature_points.size(); ++i) {
    m_feature_points_map[i] = m_feature_points[i];
  }
}

std::vector<cv::Point3d> TruthEngine::GenerateVisibleFeatures(
  double time,
  unsigned int camera_id,
  unsigned int new_feature_count
)
{
  Eigen::Vector3d pos_b_in_l = GetBodyPosition(time);
  Eigen::Quaterniond ang_b_to_l = GetBodyAngularPosition(time);
  Eigen::Vector3d pos_c_in_b = GetCameraPosition(camera_id);
  Eigen::Quaterniond ang_c_to_b = GetCameraAngularPosition(camera_id);
  Eigen::Quaterniond ang_c_to_l = ang_b_to_l * ang_c_to_b;
  Eigen::Vector3d pos_c_in_l = pos_b_in_l + ang_b_to_l * pos_c_in_b;
  Intrinsics intrinsics = GetCameraIntrinsics(camera_id);
  double room_half_width = m_room_size / 2.0;
  double room_half_height = m_room_size / 4.0;
  const double eps = 1.0e-9;

  for (unsigned int i = 0; i < new_feature_count; ++i) {
    auto c_x = static_cast<unsigned int>(SimRNG::UniRand(0, intrinsics.width));
    auto c_y = static_cast<unsigned int>(SimRNG::UniRand(0, intrinsics.height));

    Eigen::Vector3d dir_f_in_c{
      (c_x - intrinsics.width / 2) / (intrinsics.f_x / intrinsics.pixel_size),
      (c_y - intrinsics.height / 2) / (intrinsics.f_y / intrinsics.pixel_size),
      1.0};

    Eigen::Vector3d dir_f_in_l = ang_c_to_l * dir_f_in_c.normalized();
    double min_t = std::numeric_limits<double>::max();

    if (std::abs(dir_f_in_l.x()) > eps) {
      double t_x_pos = (room_half_width - pos_c_in_l.x()) / dir_f_in_l.x();
      double t_x_neg = (-room_half_width - pos_c_in_l.x()) / dir_f_in_l.x();
      if (t_x_pos > eps) {min_t = std::min(min_t, t_x_pos);}
      if (t_x_neg > eps) {min_t = std::min(min_t, t_x_neg);}
    }
    if (std::abs(dir_f_in_l.y()) > eps) {
      double t_y_pos = (room_half_width - pos_c_in_l.y()) / dir_f_in_l.y();
      double t_y_neg = (-room_half_width - pos_c_in_l.y()) / dir_f_in_l.y();
      if (t_y_pos > eps) {min_t = std::min(min_t, t_y_pos);}
      if (t_y_neg > eps) {min_t = std::min(min_t, t_y_neg);}
    }
    if (std::abs(dir_f_in_l.z()) > eps) {
      double t_z_pos = (room_half_height - pos_c_in_l.z()) / dir_f_in_l.z();
      double t_z_neg = (-room_half_height - pos_c_in_l.z()) / dir_f_in_l.z();
      if (t_z_pos > eps) {min_t = std::min(min_t, t_z_pos);}
      if (t_z_neg > eps) {min_t = std::min(min_t, t_z_neg);}
    }

    Eigen::Vector3d pos_f_in_l_eig = pos_c_in_l + (min_t * dir_f_in_l);
    cv::Point3d pos_f_in_l{pos_f_in_l_eig.x(), pos_f_in_l_eig.y(), pos_f_in_l_eig.z()};

    m_feature_points.push_back(pos_f_in_l);
    m_feature_points_map[static_cast<unsigned int>(m_feature_points_map.size())] = pos_f_in_l;
  }

  return m_feature_points;
}

std::vector<cv::Point3d> TruthEngine::GetFeatures()
{
  return m_feature_points;
}

cv::Point3d TruthEngine::GetFeature(int feature_id)
{
  return m_feature_points_map[static_cast<unsigned int>(feature_id)];
}

TruthEngine::~TruthEngine() {}

void TruthEngine::WriteTruthData(
  double data_log_rate,
  const std::string & log_directory)
{
  DataLogger truth_logger(log_directory, "body_truth");
  truth_logger.EnableLogging();

  std::stringstream header;
  header << "time";
  header << EnumerateHeader("body_pos", 3);
  header << EnumerateHeader("body_vel", 3);
  header << EnumerateHeader("body_acc", 3);
  header << EnumerateHeader("body_ang_pos", 4);
  header << EnumerateHeader("body_ang_vel", 3);
  header << EnumerateHeader("body_ang_acc", 3);

  unsigned int sensor_count{0};
  for (unsigned int i = 0; i < m_imu_pos.size(); ++i) {
    ++sensor_count;
    header << EnumerateHeader(std::string("imu_pos_") + std::to_string(sensor_count), 3);
    header << EnumerateHeader(std::string("imu_ang_pos_") + std::to_string(sensor_count), 4);
    header << EnumerateHeader(std::string("imu_acc_bias_") + std::to_string(sensor_count), 3);
    header << EnumerateHeader(std::string("imu_gyr_bias_") + std::to_string(sensor_count), 3);
  }
  for (unsigned int i = 0; i < m_cam_pos.size(); ++i) {
    ++sensor_count;
    header << EnumerateHeader(std::string("cam_pos_") + std::to_string(sensor_count), 3);
    header << EnumerateHeader(std::string("cam_ang_pos_") + std::to_string(sensor_count), 4);
  }
  for (unsigned int i = 0; i < m_gps_pos.size(); ++i) {
    ++sensor_count;
    header << EnumerateHeader(std::string("gps_pos_") + std::to_string(sensor_count), 3);
  }
  if (!m_gps_pos.empty()) {
    header << ",ref_lat,ref_lon,ref_alt,ref_heading";
  }

  truth_logger.DefineHeader(header.str());

  auto num_measurements = static_cast<unsigned int>(std::floor((m_max_time + 1.0) * data_log_rate));
  for (unsigned int i = 0; i < num_measurements; ++i) {
    sensor_count = 0;
    std::stringstream msg;
    double time = static_cast<double>(i) / data_log_rate;
    msg << time;
    msg << VectorToCommaString(GetBodyPosition(time));
    msg << VectorToCommaString(GetBodyVelocity(time));
    msg << VectorToCommaString(GetBodyAcceleration(time) + g_gravity);
    msg << QuaternionToCommaString(GetBodyAngularPosition(time));
    msg << VectorToCommaString(GetBodyAngularRate(time));
    msg << VectorToCommaString(GetBodyAngularAcceleration(time));
    for (unsigned int j = 0; j < m_imu_pos.size(); ++j) {
      ++sensor_count;
      msg << VectorToCommaString(GetImuPosition(sensor_count));
      msg << QuaternionToCommaString(GetImuAngularPosition(sensor_count));
      msg << VectorToCommaString(GetImuAccelerometerBias(sensor_count));
      msg << VectorToCommaString(GetImuGyroscopeBias(sensor_count));
    }
    for (unsigned int j = 0; j < m_cam_pos.size(); ++j) {
      ++sensor_count;
      msg << VectorToCommaString(GetCameraPosition(sensor_count));
      msg << QuaternionToCommaString(GetCameraAngularPosition(sensor_count));
    }
    for (unsigned int j = 0; j < m_gps_pos.size(); ++j) {
      ++sensor_count;
      msg << VectorToCommaString(GetGpsPosition(sensor_count));
    }
    if (!m_gps_pos.empty()) {
      msg << VectorToCommaString(GetLocalPosition());
      msg << "," << GetLocalHeading();
    }

    truth_logger.Log(msg.str());
  }

  DataLogger board_logger(log_directory, "board_truth");
  board_logger.EnableLogging();
  board_logger.DefineHeader("board,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z");

  for (auto const & board : m_board_pos) {
    std::stringstream msg;
    msg << std::to_string(board.first);
    msg << VectorToCommaString(m_board_pos[board.first]);
    msg << QuaternionToCommaString(m_board_ang[board.first]);
    board_logger.Log(msg.str());
  }

  DataLogger feature_logger(log_directory, "feature_points");
  feature_logger.EnableLogging();
  feature_logger.DefineHeader("Feature,x,y,z");

  for (unsigned int i = 0; i < m_feature_points.size(); ++i) {
    std::stringstream msg;
    msg << std::to_string(i);
    msg << "," << m_feature_points[i].x;
    msg << "," << m_feature_points[i].y;
    msg << "," << m_feature_points[i].z;
    feature_logger.Log(msg.str());
  }
}

Intrinsics TruthEngine::GetCameraIntrinsics(unsigned int sensor_id)
{
  return m_cam_intrinsics[sensor_id];
}

void TruthEngine::SetCameraIntrinsics(unsigned int sensor_id, const Intrinsics & intrinsics)
{
  m_cam_intrinsics[sensor_id] = intrinsics;
}
