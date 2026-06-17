// Copyright 2024 Jacob Hartzer
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

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <memory>

#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "ekf/update/updater.hpp"
#include "infrastructure/debug_logger.hpp"

TEST(test_updater, constructor) {
  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  Updater updater(0, logger);
}

TEST(test_updater, root_covariance) {
  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = logger;
  ekf_params.use_root_covariance = true;
  EKF ekf(ekf_params);

  BodyState body_state;
  ekf.Initialize(0.0, body_state);

  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(1, 18);
  jacobian(0, 0) = 1.0;
  Eigen::VectorXd residual = Eigen::VectorXd::Zero(1);
  residual[0] = 0.5;
  Eigen::MatrixXd measurement_noise = Eigen::MatrixXd::Identity(1, 1);
  measurement_noise(0, 0) = 0.04;

  Updater::KalmanUpdate(ekf, jacobian, residual, measurement_noise);

  EXPECT_NEAR(ekf.m_state.body_state.pos_b_in_l[0], 0.0, 1.0);
}
