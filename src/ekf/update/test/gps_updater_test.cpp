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
#include <string>

#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "ekf/update/gps_updater.hpp"
#include "infrastructure/debug_logger.hpp"
#include "utility/custom_assertions.hpp"
#include "utility/gps_helper.hpp"

TEST(test_gps_updater, update) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  double time_init = 0.0;
  BodyState body_state;
  body_state.vel_b_in_l = Eigen::Vector3d::Ones();
  ekf->Initialize(time_init, body_state);

  unsigned int gps_id{0};
  bool is_extrinsic{false};

  GpsState gps_state;
  gps_state.SetIsExtrinsic(is_extrinsic);
  Eigen::Matrix3d gps_cov = Eigen::Matrix3d::Zero(3, 3);
  ekf->RegisterGPS(gps_id, gps_state, gps_cov);

  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  GpsUpdater gps_updater(gps_id, is_extrinsic, "", 0.0, logger);
  Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Identity() * 1e-9;

  State state = ekf->m_state;
  EXPECT_NEAR(state.body_state.pos_b_in_l[0], 0, 1e-2);
  EXPECT_NEAR(state.body_state.pos_b_in_l[1], 0, 1e-2);
  EXPECT_NEAR(state.body_state.pos_b_in_l[2], 0, 1e-2);

  double time = time_init + 1;
  Eigen::Vector3d ref_lla{0, 0, 0};
  Eigen::Vector3d antenna_enu{1, 1, 1};
  Eigen::Vector3d gps_lla = enu_to_lla(antenna_enu, ref_lla);
  gps_updater.UpdateEKF(*ekf, time, gps_lla, pos_cov);

  state = ekf->m_state;
  EXPECT_NEAR(state.body_state.pos_b_in_l[0], 1.0, 1e-2);
  EXPECT_NEAR(state.body_state.pos_b_in_l[1], 1.0, 1e-2);
  EXPECT_NEAR(state.body_state.pos_b_in_l[2], 1.0, 1e-2);

  time += 1;
  antenna_enu = Eigen::Vector3d{2, 2, 2};
  gps_lla = enu_to_lla(antenna_enu, ref_lla);
  gps_updater.UpdateEKF(*ekf, time, gps_lla, pos_cov);

  state = ekf->m_state;
  EXPECT_NEAR(state.body_state.pos_b_in_l[0], 1.5, 1e-2);
  EXPECT_NEAR(state.body_state.pos_b_in_l[1], 1.5, 1e-2);
  EXPECT_NEAR(state.body_state.pos_b_in_l[2], 1.5, 1e-2);
}

TEST(test_gps_updater, jacobian) {
  unsigned int gps_id{0};
  bool is_extrinsic{true};
  GpsState gps_state;
  gps_state.pos_a_in_b = Eigen::Vector3d{1, 2, 3};
  gps_state.SetIsExtrinsic(is_extrinsic);

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();

  auto debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = debug_logger;
  EKF ekf(ekf_params);
  ekf.RegisterGPS(gps_id, gps_state, covariance);

  auto gps_updater = GpsUpdater(gps_id, is_extrinsic, "log_file_directory", 0.0, debug_logger);

  Eigen::VectorXd base_state = ekf.m_state.ToVector();
  Eigen::MatrixXd jac_analytical = gps_updater.GetMeasurementJacobian(ekf);
  Eigen::Vector3d base_meas = gps_updater.PredictMeasurement(ekf);

  double delta = 1.0e-6;
  unsigned int jac_size = base_state.size();
  Eigen::MatrixXd jac_numerical = Eigen::MatrixXd::Zero(3, jac_size);
  for (unsigned int i = 0; i < jac_size; ++i) {
    Eigen::VectorXd delta_vec = base_state;
    delta_vec[i] += delta;
    ekf.m_state.SetState(delta_vec);
    jac_numerical.block<3, 1>(0, i) = (gps_updater.PredictMeasurement(ekf) - base_meas) / delta;
  }

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(jac_analytical, jac_numerical, 1e-3));
}

TEST(test_gps_updater, multi_update) {
  auto debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = debug_logger;
  ekf_params.use_root_covariance = true;
  EKF ekf(ekf_params);

  BodyState body_state;
  ekf.Initialize(0.0, body_state);

  // Register GPS
  unsigned int gps_id{0};
  bool is_extrinsic{true};
  GpsState gps_state;
  gps_state.SetIsExtrinsic(is_extrinsic);
  Eigen::Matrix3d gps_cov = Eigen::Matrix3d::Identity() * 1e-3;
  ekf.RegisterGPS(gps_id, gps_state, gps_cov);

  GpsUpdater gps_updater(gps_id, is_extrinsic, "", 0.0, debug_logger);

  // Populate GPS vectors in EKF
  for (int i = 0; i < 4; ++i) {
    double time = static_cast<double>(i + 1);
    Eigen::Vector3d gps_lla = gps_state.pos_a_in_b + Eigen::Vector3d(0.0001 * i, 0.0001 * i, 0.0);
    ekf.AttemptGpsInitialization(time, gps_lla);
  }

  // Set reference GPS LLA and angle manually so MultiUpdateEKF has non-zero references
  ekf.SetGpsReference(Eigen::Vector3d(0.0, 0.0, 0.0), 0.1);

  // Perform UpdateEKF to test the root covariance block in output logging of UpdateEKF
  // and trigger the log block at line 139.
  Eigen::Vector3d single_gps_lla{0.0001, 0.0001, 0.0};
  gps_updater.UpdateEKF(ekf, 5.0, single_gps_lla, gps_cov);

  // Perform MultiUpdateEKF
  gps_updater.MultiUpdateEKF(ekf);
}
