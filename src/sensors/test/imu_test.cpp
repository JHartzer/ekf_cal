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

#include <gtest/gtest.h>

#include <memory>

#include "ekf/constants.hpp"
#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "infrastructure/debug_logger.hpp"
#include "sensors/imu_message.hpp"
#include "sensors/imu.hpp"


TEST(test_IMU, Constructor) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  IMU::Parameters imu_params1;
  imu_params1.ekf = ekf;
  imu_params1.logger = ekf_params.debug_logger;
  imu_params1.is_intrinsic = false;
  imu_params1.is_extrinsic = false;
  IMU imu1(imu_params1);

  IMU::Parameters imu_params2;
  imu_params2.ekf = ekf;
  imu_params2.logger = ekf_params.debug_logger;
  imu_params2.is_intrinsic = true;
  imu_params2.is_extrinsic = false;
  IMU imu2(imu_params2);

  IMU::Parameters imu_params3;
  imu_params3.ekf = ekf;
  imu_params3.logger = ekf_params.debug_logger;
  imu_params3.is_intrinsic = false;
  imu_params3.is_extrinsic = true;
  IMU imu3(imu_params3);

  IMU::Parameters imu_params4;
  imu_params4.ekf = ekf;
  imu_params4.logger = ekf_params.debug_logger;
  imu_params4.is_intrinsic = true;
  imu_params4.is_extrinsic = true;
  IMU imu4(imu_params4);
}

TEST(test_IMU, ID) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  IMU::Parameters imu_params1;
  imu_params1.ekf = ekf;
  imu_params1.logger = ekf_params.debug_logger;

  IMU imu1(imu_params1);

  IMU::Parameters imu_params2;
  imu_params2.ekf = ekf;
  imu_params2.logger = ekf_params.debug_logger;

  IMU imu2(imu_params2);

  unsigned int id_one = imu1.GetId();
  unsigned int id_two = id_one + 1;

  EXPECT_EQ(imu1.GetId(), id_one);
  EXPECT_EQ(imu2.GetId(), id_two);
}

TEST(test_IMU, IMU_message) {
  ImuMessage imu_message;
}

TEST(test_IMU, Callback) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  BodyState body_state;
  body_state.pos_b_in_l = Eigen::Vector3d(0, 0, 0);
  body_state.vel_b_in_l = Eigen::Vector3d(0, 0, 0);
  body_state.acc_b_in_l = Eigen::Vector3d(0, 0, 0) + g_gravity;
  body_state.ang_b_to_l = Eigen::Quaterniond::Identity();
  body_state.ang_vel_b_in_l = Eigen::Vector3d(0, 0, 0);
  body_state.ang_acc_b_in_l = Eigen::Vector3d(0, 0, 0);
  ekf->Initialize(1.0, body_state);
  ekf->InitializeGravity();

  IMU::Parameters imu_params;
  imu_params.ekf = ekf;
  imu_params.logger = ekf_params.debug_logger;
  imu_params.is_intrinsic = true;
  imu_params.is_extrinsic = true;
  imu_params.filter_sensor_time = false;
  IMU imu(imu_params);

  ImuMessage imu_message;
  imu_message.time_measured = 1.1;
  imu_message.time_received = 9.9;
  imu_message.acceleration = Eigen::Vector3d(0.1, 0.2, 9.8);
  imu_message.acceleration_covariance = Eigen::Matrix3d::Identity() * 0.01;
  imu_message.angular_rate = Eigen::Vector3d(0.01, 0.02, 0.03);
  imu_message.angular_rate_covariance = Eigen::Matrix3d::Identity() * 0.001;

  EXPECT_NO_THROW(imu.Callback(imu_message));
  EXPECT_DOUBLE_EQ(ekf->GetCurrentTime(), 1.1);
}

TEST(test_IMU, Callback_FilterSensorTime_ReordersWithinWindow) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  BodyState body_state;
  body_state.pos_b_in_l = Eigen::Vector3d(0, 0, 0);
  body_state.vel_b_in_l = Eigen::Vector3d(0, 0, 0);
  body_state.acc_b_in_l = Eigen::Vector3d(0, 0, 0) + g_gravity;
  body_state.ang_b_to_l = Eigen::Quaterniond::Identity();
  body_state.ang_vel_b_in_l = Eigen::Vector3d(0, 0, 0);
  body_state.ang_acc_b_in_l = Eigen::Vector3d(0, 0, 0);
  ekf->Initialize(1.0, body_state);
  ekf->InitializeGravity();

  IMU::Parameters imu_params;
  imu_params.ekf = ekf;
  imu_params.logger = ekf_params.debug_logger;
  imu_params.is_intrinsic = true;
  imu_params.is_extrinsic = true;
  imu_params.filter_sensor_time = true;
  imu_params.measurement_time_reorder_window = 1.0;
  IMU imu(imu_params);

  ImuMessage imu_message_1;
  imu_message_1.time_measured = 1.0;
  imu_message_1.time_received = 2.0;
  imu_message_1.acceleration = Eigen::Vector3d(0.1, 0.2, 9.8);
  imu_message_1.acceleration_covariance = Eigen::Matrix3d::Identity() * 0.01;
  imu_message_1.angular_rate = Eigen::Vector3d(0.01, 0.02, 0.03);
  imu_message_1.angular_rate_covariance = Eigen::Matrix3d::Identity() * 0.001;

  ImuMessage imu_message_2 = imu_message_1;
  imu_message_2.time_measured = 2.1;
  imu_message_2.time_received = 2.6;

  imu.Callback(imu_message_1);
  EXPECT_DOUBLE_EQ(ekf->GetCurrentTime(), 1.0);

  imu.Callback(imu_message_2);
  EXPECT_DOUBLE_EQ(ekf->GetCurrentTime(), 1.0);

  imu.Flush();
  EXPECT_DOUBLE_EQ(ekf->GetCurrentTime(), 2.6);
}
