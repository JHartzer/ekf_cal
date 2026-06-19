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

#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <memory>

#include "ekf/ekf.hpp"
#include "infrastructure/debug_logger.hpp"
#include "utility/custom_assertions.hpp"
#include "utility/type_helper.hpp"
#include "trackers/fiducial_tracker.hpp"
#include "trackers/sim/sim_fiducial_tracker.hpp"
#include "infrastructure/sim/truth_engine_cyclic.hpp"

TEST(test_fiducial_tracker, constructor) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  Eigen::Vector3d pos_frequency {1, 2, 3};
  Eigen::Vector3d ang_frequency {1, 2, 3};
  Eigen::Vector3d pos_offset {0, 0, 0};
  Eigen::Vector3d ang_offset {0, 0, 0};
  double pos_amplitude {1.0};
  double ang_amplitude {0.1};
  double stationary_time {1.0};
  double max_time {1.0};

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

  FiducialTracker::Parameters fiducial_params;
  fiducial_params.ekf = ekf;
  fiducial_params.is_extrinsic = true;
  fiducial_params.pos_f_in_l = Eigen::Vector3d{5, 0, 0};
  fiducial_params.ang_f_to_l = Eigen::Quaterniond{1, 0, 0, 0};
  SimFiducialTracker::Parameters sim_params;
  sim_params.fiducial_params = fiducial_params;
  SimFiducialTracker sim_fiducial_tracker(sim_params, truth_engine);
}

TEST(test_fiducial_tracker, generate_message_matches_truth_geometry) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  auto truth_engine = std::make_shared<TruthEngineCyclic>(
    Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(),
    0.0,
    0.0,
    0.0,
    1.0,
    ekf_params.debug_logger
  );

  const unsigned int camera_id = 7;
  const Eigen::Vector3d pos_c_in_b_true{0.1, -0.2, 0.3};
  const Eigen::Quaterniond ang_c_to_b_true = EigVecToQuat(Eigen::Vector3d{0.2, -0.1, 0.3});
  truth_engine->SetCameraPosition(camera_id, pos_c_in_b_true);
  truth_engine->SetCameraAngularPosition(camera_id, ang_c_to_b_true);

  FiducialTracker::Parameters fiducial_params;
  fiducial_params.ekf = ekf;
  fiducial_params.logger = ekf_params.debug_logger;
  fiducial_params.camera_id = camera_id;
  fiducial_params.pos_f_in_l = Eigen::Vector3d{2.0, 1.0, 4.0};
  fiducial_params.ang_f_to_l = EigVecToQuat(Eigen::Vector3d{-0.2, 0.4, 0.1});

  SimFiducialTracker::Parameters sim_params;
  sim_params.no_errors = true;
  sim_params.fiducial_params = fiducial_params;
  SimFiducialTracker sim_fiducial_tracker(sim_params, truth_engine);

  const double message_time = 0.5;
  auto msg = sim_fiducial_tracker.GenerateMessage(message_time, 42);

  ASSERT_TRUE(msg->is_board_visible);

  Eigen::Vector3d pos_b_in_l = truth_engine->GetBodyPosition(message_time);
  Eigen::Quaterniond ang_b_to_l = truth_engine->GetBodyAngularPosition(message_time);
  Eigen::Matrix3d rot_l_to_b = ang_b_to_l.toRotationMatrix().transpose();
  Eigen::Matrix3d rot_b_to_c = ang_c_to_b_true.toRotationMatrix().transpose();

  Eigen::Vector3d expected_pos =
    rot_b_to_c * (rot_l_to_b * (fiducial_params.pos_f_in_l - pos_b_in_l) - pos_c_in_b_true);
  Eigen::Quaterniond expected_ang =
    ang_c_to_b_true.inverse() * ang_b_to_l.inverse() * fiducial_params.ang_f_to_l;

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(msg->board_detection.pos_f_in_c, expected_pos, 1e-12));
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(
    QuatToRotVec(msg->board_detection.ang_f_to_c * expected_ang.conjugate()),
    Eigen::Vector3d::Zero(),
    1e-12));
}
