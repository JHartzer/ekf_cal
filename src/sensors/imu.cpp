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

#include "sensors/imu.hpp"

#include <Eigen/Core>

#include <memory>
#include <string>

#include "ekf/types.hpp"
#include "infrastructure/debug_logger.hpp"
#include "sensors/imu_message.hpp"
#include "sensors/sensor.hpp"


IMU::IMU(IMU::Parameters params)
: Sensor(params), m_ekf(params.ekf),
  m_imu_updater(m_id, params.is_extrinsic, params.is_intrinsic,
    params.log_directory, params.data_log_rate, params.logger)
{
  m_is_extrinsic = params.is_extrinsic;
  m_is_intrinsic = params.is_intrinsic;
  m_rate = params.rate;

  ImuState imu_state;
  imu_state.SetIsExtrinsic(params.is_extrinsic);
  imu_state.SetIsIntrinsic(params.is_intrinsic);
  imu_state.pos_stability = params.pos_stability;
  imu_state.ang_stability = params.ang_stability;
  imu_state.acc_bias_stability = params.acc_bias_stability;
  imu_state.omg_bias_stability = params.omg_bias_stability;
  imu_state.pos_i_in_b = params.pos_i_in_b;
  imu_state.ang_i_to_b = params.ang_i_to_b;
  imu_state.acc_bias = params.acc_bias;
  imu_state.omg_bias = params.omg_bias;

  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(imu_state.size, imu_state.size);
  if (imu_state.GetIsExtrinsic() && imu_state.GetIsIntrinsic()) {
    cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * params.variance.pos;
    cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * params.variance.ang;
    cov.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * params.variance.acc_bias;
    cov.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * params.variance.gyr_bias;
  } else if (imu_state.GetIsExtrinsic()) {
    cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * params.variance.pos;
    cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * params.variance.ang;
  } else if (imu_state.GetIsIntrinsic()) {
    cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * params.variance.acc_bias;
    cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * params.variance.gyr_bias;
  }

  m_ekf->RegisterIMU(m_id, imu_state, cov);
}

void IMU::Callback(const ImuMessage & imu_message)
{
  BufferMessage(
    imu_message,
    m_message_buffer,
    [this](const ImuMessage & buffered_message) {ExecuteCallback(buffered_message);});
}

void IMU::ExecuteCallback(const ImuMessage & imu_message)
{
  m_logger->Log(
    LogLevel::DEBUG,
    "IMU \"" + m_name + "\" callback at used time " + std::to_string(imu_message.time_used));
  m_imu_updater.UpdateEKF(
    *m_ekf,
    imu_message.time_used,
    imu_message.acceleration,
    imu_message.acceleration_covariance,
    imu_message.angular_rate,
    imu_message.angular_rate_covariance);
  m_logger->Log(LogLevel::DEBUG, "IMU \"" + m_name + "\" callback complete");
}

void IMU::Flush()
{
  FlushBufferedMessages(
    m_message_buffer,
    [this](const ImuMessage & buffered_message) {ExecuteCallback(buffered_message);});
}
