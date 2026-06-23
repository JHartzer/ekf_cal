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
#include <vector>

#include "ekf/ekf.hpp"
#include "infrastructure/debug_logger.hpp"
#include "infrastructure/sim/truth_engine_cyclic.hpp"
#include "sensors/imu.hpp"
#include "sensors/sim/sim_imu.hpp"
#include "sensors/sim/sim_imu_message.hpp"
#include "utility/sim/sim_rng.hpp"

TEST(test_SimIMU, Constructor) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  Eigen::Vector3d pos_frequency{1, 2, 3};
  Eigen::Vector3d ang_frequency{4, 5, 6};
  Eigen::Vector3d pos_offset{1, 2, 3};
  Eigen::Vector3d ang_offset{0.1, 0.2, 0.3};
  double pos_amplitude = 1.0;
  double ang_amplitude = 0.1;
  double stationary_time{0.0};
  double max_time{1.0};

  auto truthEngine = std::make_shared<TruthEngineCyclic>(
    pos_frequency,
    ang_frequency,
    pos_offset,
    ang_offset,
    pos_amplitude,
    ang_amplitude,
    stationary_time,
    max_time,
    ekf_params.debug_logger
  );

  IMU::Parameters imu_params;
  imu_params.rate = 100.0;
  imu_params.ekf = ekf;
  imu_params.logger = ekf_params.debug_logger;

  SimIMU::Parameters sim_imu_params;
  sim_imu_params.imu_params = imu_params;

  SimIMU sim_imu(sim_imu_params, truthEngine);
  SimRNG::SetSeed(1);
  std::vector<std::shared_ptr<SimImuMessage>> imu_messages = sim_imu.GenerateMessages();
}

TEST(test_SimIMU, TimingSemantics) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  Eigen::Vector3d pos_frequency{1, 2, 3};
  Eigen::Vector3d ang_frequency{4, 5, 6};
  Eigen::Vector3d pos_offset{1, 2, 3};
  Eigen::Vector3d ang_offset{0.1, 0.2, 0.3};
  double pos_amplitude = 1.0;
  double ang_amplitude = 0.1;
  double stationary_time{0.0};
  double max_time{1.0};

  auto truth_engine = std::make_shared<TruthEngineCyclic>(
    pos_frequency,
    ang_frequency,
    pos_offset,
    ang_offset,
    pos_amplitude,
    ang_amplitude,
    stationary_time,
    max_time,
    ekf_params.debug_logger
  );

  IMU::Parameters imu_params;
  imu_params.rate = 100.0;
  imu_params.ekf = ekf;
  imu_params.logger = ekf_params.debug_logger;

  SimIMU::Parameters sim_imu_params;
  sim_imu_params.imu_params = imu_params;
  sim_imu_params.time_jitter = 0.01;
  sim_imu_params.clock_bias = 0.25;
  sim_imu_params.acc_error = Eigen::Vector3d::Zero();
  sim_imu_params.omg_error = Eigen::Vector3d::Zero();
  sim_imu_params.pos_error = Eigen::Vector3d::Zero();
  sim_imu_params.ang_error = Eigen::Vector3d::Zero();
  sim_imu_params.acc_bias_error = Eigen::Vector3d::Zero();
  sim_imu_params.omg_bias_error = Eigen::Vector3d::Zero();

  SimIMU sim_imu(sim_imu_params, truth_engine);
  SimRNG::SetSeed(1);
  std::vector<std::shared_ptr<SimImuMessage>> imu_messages = sim_imu.GenerateMessages();

  ASSERT_FALSE(imu_messages.empty());
  for (const auto & imu_message : imu_messages) {
    EXPECT_DOUBLE_EQ(imu_message->time_measured - imu_message->time_true, 0.25);
    EXPECT_GE(imu_message->time_received, imu_message->time_true);
  }
}
