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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "ekf/constants.hpp"
#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "infrastructure/debug_logger.hpp"
#include "utility/custom_assertions.hpp"
#include "utility/gps_helper.hpp"


TEST(test_EKF, get_counts) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  EXPECT_EQ(ekf->GetImuCount(), 0);
  EXPECT_EQ(ekf->GetCamCount(), 0);

  ImuState imu_state;
  imu_state.SetIsIntrinsic(true);
  imu_state.SetIsExtrinsic(false);
  Eigen::MatrixXd imu_covariance(6, 6);
  ekf->RegisterIMU(0, imu_state, imu_covariance);

  CamState cam_state;
  cam_state.SetIsExtrinsic(true);
  Eigen::MatrixXd cam_covariance(6, 6);
  ekf->RegisterCamera(1, cam_state, cam_covariance);
  ekf->AugmentStateIfNeeded(1, 0);
  ekf->AugmentStateIfNeeded(1, 1);

  EXPECT_EQ(ekf->GetImuCount(), 1);
  EXPECT_EQ(ekf->GetCamCount(), 1);

  EXPECT_EQ(ekf->m_state.imu_states[0].index, 18);
  EXPECT_EQ(ekf->m_state.cam_states[1].index, 24);
  EXPECT_EQ(ekf->GetAugState(1, 0, 0).index, 30);
  EXPECT_EQ(ekf->GetAugState(1, 1, 0).index, 36);
}

TEST(test_EKF, duplicate_sensors) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  EXPECT_EQ(ekf->GetImuCount(), 0);
  EXPECT_EQ(ekf->GetCamCount(), 0);

  ImuState imu_state;
  imu_state.SetIsIntrinsic(true);
  imu_state.SetIsExtrinsic(false);
  Eigen::MatrixXd imu_covariance(g_imu_intrinsic_state_size, g_imu_intrinsic_state_size);
  ekf->RegisterIMU(0, imu_state, imu_covariance);
  ekf->RegisterIMU(0, imu_state, imu_covariance);

  GpsState gps_state;
  gps_state.SetIsExtrinsic(true);
  Eigen::MatrixXd gps_cov(g_gps_extrinsic_state_size, g_gps_extrinsic_state_size);
  ekf->RegisterGPS(1, gps_state, gps_cov);
  ekf->RegisterGPS(1, gps_state, gps_cov);

  CamState cam_state;
  cam_state.SetIsExtrinsic(true);
  Eigen::MatrixXd cam_covariance(g_cam_extrinsic_state_size, g_cam_extrinsic_state_size);
  ekf->RegisterCamera(2, cam_state, cam_covariance);
  ekf->RegisterCamera(2, cam_state, cam_covariance);

  FidState fid_state;
  fid_state.SetIsExtrinsic(true);
  fid_state.id = 3;
  Eigen::MatrixXd fid_cov(g_fid_extrinsic_state_size, g_fid_extrinsic_state_size);
  ekf->RegisterFiducial(fid_state, fid_cov);
  ekf->RegisterFiducial(fid_state, fid_cov);

  ekf->AugmentStateIfNeeded(2, 0);
  ekf->AugmentStateIfNeeded(2, 1);

  EXPECT_EQ(ekf->GetImuCount(), 1);
  EXPECT_EQ(ekf->GetGpsCount(), 1);
  EXPECT_EQ(ekf->GetCamCount(), 1);

  EXPECT_EQ(ekf->m_state.imu_states[0].index, 18);
  EXPECT_EQ(ekf->m_state.gps_states[1].index, 24);
  EXPECT_EQ(ekf->m_state.cam_states[2].index, 27);
  EXPECT_EQ(ekf->GetAugState(2, 0, 0).index, 39);
  EXPECT_EQ(ekf->GetAugState(2, 1, 0).index, 45);

  EXPECT_EQ(ekf->GetImuStateStart(), 18);
  EXPECT_EQ(ekf->GetGpsStateStart(), 24);
  EXPECT_EQ(ekf->GetCamStateStart(), 27);
  EXPECT_EQ(ekf->GetFidStateStart(), 33);
  EXPECT_EQ(ekf->GetAugStateStart(), 39);
}

TEST(test_EKF, SetBodyProcessNoise) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  Eigen::VectorXd process_noise = Eigen::VectorXd::Ones(18);
  ekf->SetBodyProcessNoise(process_noise);
}

TEST(test_EKF, MatchState) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  AugState aug_state = ekf->GetAugState(0, 0, 0);

  Eigen::Quaterniond zero_quat {1, 0, 0, 0};
  Eigen::Vector3d zero_vec {0, 0, 0};

  EXPECT_EQ(aug_state.frame_id, 0);
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(aug_state.ang_b_to_l, zero_quat, 1e-6));
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(aug_state.pos_b_in_l, zero_vec, 1e-6));

  EXPECT_EQ(ekf->GetAugState(0, 0, 0).index, 0);
}

TEST(test_EKF, SetMaxTrackLength) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  ekf->SetMaxTrackLength(20);
}

TEST(test_EKF, SetGpsReference) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  Eigen::Vector3d pos_e_in_g {0, 0, 0};
  double ang_l_to_e {0};

  ekf->SetGpsReference(pos_e_in_g, ang_l_to_e);
  EXPECT_TRUE(ekf->IsLlaInitialized());
}

TEST(test_EKF, AugmentCovariance) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  Eigen::VectorXd in_vec(12);
  in_vec << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  Eigen::MatrixXd in_cov = in_vec.asDiagonal();

  Eigen::MatrixXd out_cov = ekf->AugmentCovariance(in_cov, 12);

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(out_cov.block<12, 12>(0, 0), in_cov, 1e-6));

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(out_cov.block<3, 3>(12, 12), in_cov.block<3, 3>(0, 0), 1e-6));
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(out_cov.block<3, 3>(15, 15), in_cov.block<3, 3>(9, 9), 1e-6));

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(out_cov.block<3, 3>(0, 12), in_cov.block<3, 3>(0, 0), 1e-6));
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(out_cov.block<3, 3>(9, 15), in_cov.block<3, 3>(9, 9), 1e-6));
}

TEST(test_EKF, PredictModelRK4) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  ekf_params.use_rk4 = true;
  auto ekf = std::make_shared<EKF>(ekf_params);
  ekf->InitializeGravity();

  BodyState body_state_init;
  body_state_init.pos_b_in_l = Eigen::Vector3d(1.0, 2.0, 3.0);
  body_state_init.vel_b_in_l = Eigen::Vector3d(0.1, 0.2, 0.3);
  body_state_init.acc_b_in_l = Eigen::Vector3d(1.0, 2.0, 3.0) + g_gravity;
  body_state_init.ang_b_to_l = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  body_state_init.ang_vel_b_in_l = Eigen::Vector3d(0.1, 0.2, 0.3);
  body_state_init.ang_acc_b_in_l = Eigen::Vector3d(0.01, 0.02, 0.03);

  ekf->Initialize(0.0, body_state_init);
  ekf->SetZeroAcceleration(false);

  ekf->PredictModel(1.0);

  Eigen::Vector3d expected_pos(2.1, 4.2, 6.3);
  Eigen::Vector3d expected_vel(1.1, 2.2, 3.3);
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(ekf->m_state.body_state.pos_b_in_l, expected_pos, 1e-6));
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(ekf->m_state.body_state.vel_b_in_l, expected_vel, 1e-6));

  // Orientation RK4: expected w,x,y,z is roughly [0.980768, 0.052162, 0.104325, 0.156487]
  Eigen::Quaterniond expected_ang(0.980768, 0.052162, 0.104325, 0.156487);
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(ekf->m_state.body_state.ang_b_to_l, expected_ang, 1e-5));
}

TEST(test_EKF, LogBodyStateAndGetters) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  ekf_params.data_log_rate = 10.0;
  ekf_params.log_directory = "/temp/";
  auto ekf = std::make_shared<EKF>(ekf_params);

  ekf->LogBodyStateIfNeeded(0);
  ekf->LogBodyStateIfNeeded(1);

  ImuState imu_state;
  imu_state.pos_stability = 0.1;
  ekf->RegisterIMU(3, imu_state, Eigen::MatrixXd::Identity(6, 6));

  GpsState gps_state;
  gps_state.pos_stability = 0.2;
  ekf->RegisterGPS(4, gps_state, Eigen::Matrix3d::Identity());

  CamState cam_state;
  cam_state.pos_stability = 0.3;
  ekf->RegisterCamera(5, cam_state, Eigen::MatrixXd::Identity(6, 6));

  ImuState imu_ret = ekf->GetImuState(3);
  GpsState gps_ret = ekf->GetGpsState(4);
  CamState cam_ret = ekf->GetCamState(5);

  EXPECT_EQ(imu_ret.pos_stability, 0.1);
  EXPECT_EQ(gps_ret.pos_stability, 0.2);
  EXPECT_EQ(cam_ret.pos_stability, 0.3);

  EXPECT_EQ(ekf->GetAugStateSize(), 0);
}

TEST(test_EKF, AugmentStateIfNeededAndPrune) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  ekf_params.augmenting_type = AugmentationType::TIME;
  ekf_params.augmenting_delta_time = 0.5;
  auto ekf = std::make_shared<EKF>(ekf_params);
  ekf->InitializeGravity();

  BodyState body_state;
  ekf->Initialize(0.0, body_state);

  // 1st augmentation at t=0.0 (empty list triggers line 491 and line 552)
  ekf->AugmentStateIfNeeded();
  EXPECT_EQ(ekf->GetAugStateSize(), 6);

  // Set frame received since last aug to true:
  ekf->AugmentStateIfNeeded(0, 1);

  // 2nd augmentation at t=1.0:
  ekf->PredictModel(1.0);
  EXPECT_EQ(ekf->GetAugStateSize(), 12);

  // Set frame received since last aug to true:
  ekf->AugmentStateIfNeeded(0, 2);

  // 3rd augmentation at t=2.0 (triggers pruning condition line 530 and removes first state)
  ekf->PredictModel(2.0);
  EXPECT_EQ(ekf->GetAugStateSize(), 12);  // should remain 12 because 1 was added and 1 was pruned
}

TEST(test_EKF, GetAugStateBranch) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  ekf_params.augmenting_type = AugmentationType::PRIMARY;
  auto ekf = std::make_shared<EKF>(ekf_params);
  ekf->InitializeGravity();

  BodyState body_state;
  ekf->Initialize(0.0, body_state);

  ekf->AugmentStateIfNeeded(0, 10);

  // Call GetAugState with camera_id = 1 (not m_primary_camera_id which is 0) to hit line 641
  AugState retrieved = ekf->GetAugState(1, 10, 0.0);
  EXPECT_EQ(retrieved.index, 18);
}

TEST(test_EKF, AttemptGpsInitialization) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  ekf_params.gps_init_type = GpsInitType::BASELINE_DIST;
  ekf_params.gps_init_baseline_dist = 0.5;
  auto ekf = std::make_shared<EKF>(ekf_params);

  Eigen::Vector3d ref_lla(37.7749, -122.4194, 0.0);

  ekf->m_state.body_state.pos_b_in_l = Eigen::Vector3d(0, 0, 0);
  ekf->AttemptGpsInitialization(
    0.0, enu_to_lla(Eigen::Vector3d(0, 0, 0), ref_lla), ekf->m_state.body_state.pos_b_in_l);

  ekf->m_state.body_state.pos_b_in_l = Eigen::Vector3d(10, 0, 0);
  ekf->AttemptGpsInitialization(
    1.0, enu_to_lla(Eigen::Vector3d(10, 0, 0), ref_lla), ekf->m_state.body_state.pos_b_in_l);

  ekf->m_state.body_state.pos_b_in_l = Eigen::Vector3d(20, 0, 0);
  ekf->AttemptGpsInitialization(
    2.0, enu_to_lla(Eigen::Vector3d(20, 0, 0), ref_lla), ekf->m_state.body_state.pos_b_in_l);

  ekf->m_state.body_state.pos_b_in_l = Eigen::Vector3d(30, 0, 0);
  ekf->AttemptGpsInitialization(
    3.0, enu_to_lla(Eigen::Vector3d(30, 0, 0), ref_lla), ekf->m_state.body_state.pos_b_in_l);

  EXPECT_TRUE(ekf->IsLlaInitialized());
  EXPECT_EQ(ekf->GetGpsTimeVector().size(), 4);
  EXPECT_EQ(ekf->GetGpsEcefVector().size(), 4);
  EXPECT_EQ(ekf->GetGpsXyzVector().size(), 4);
}

TEST(test_EKF, AttemptGpsInitializationSignedHeading) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  ekf_params.gps_init_type = GpsInitType::BASELINE_DIST;
  ekf_params.gps_init_baseline_dist = 1.0;
  auto ekf = std::make_shared<EKF>(ekf_params);

  const Eigen::Vector3d ref_lla(37.7749, -122.4194, 15.0);
  const double heading = -0.25;
  const std::vector<Eigen::Vector3d> local_positions{
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {2.0, 0.5, 0.0},
    {3.0, 1.0, 0.0},
  };

  for (unsigned int i = 0; i < local_positions.size(); ++i) {
    ekf->m_state.body_state.pos_b_in_l = local_positions[i];
    const Eigen::Vector3d gps_enu = local_to_enu(local_positions[i], heading);
    ekf->AttemptGpsInitialization(
      static_cast<double>(i),
      enu_to_lla(gps_enu, ref_lla),
      ekf->m_state.body_state.pos_b_in_l);
  }

  EXPECT_TRUE(ekf->IsLlaInitialized());
  EXPECT_NEAR(ekf->GetReferenceAngle(), heading, 1e-3);
  EXPECT_TRUE(EXPECT_EIGEN_NEAR(ekf->GetReferenceLLA(), ref_lla, 1e-3));
}

TEST(test_EKF, ReducedState9State) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  ekf_params.use_reduced_state = true;
  ekf_params.use_rk4 = true;
  auto ekf = std::make_shared<EKF>(ekf_params);

  EXPECT_EQ(ekf->GetStateSize(), 9);
  EXPECT_EQ(ekf->GetOrientationStateIndex(), 6);
  EXPECT_EQ(ekf->m_state.body_state.size, 9);

  BodyState body_state_init;
  body_state_init.size = 9;
  body_state_init.pos_b_in_l = Eigen::Vector3d::Zero();
  body_state_init.vel_b_in_l = Eigen::Vector3d(1.0, 2.0, 3.0);
  body_state_init.ang_b_to_l = Eigen::Quaterniond::Identity();

  ekf->Initialize(0.0, body_state_init);
  ekf->InitializeGravity();

  ekf->m_state.body_state.acc_b_in_l = Eigen::Vector3d(0.0, 0.0, 9.80665);
  ekf->m_state.body_state.ang_vel_b_in_l = Eigen::Vector3d(0.1, 0.2, 0.3);

  ekf->PredictModel(1.0);

  EXPECT_NEAR(ekf->m_state.body_state.pos_b_in_l.x(), 1.0, 1e-6);
  EXPECT_NEAR(ekf->m_state.body_state.pos_b_in_l.y(), 2.0, 1e-6);
  EXPECT_NEAR(ekf->m_state.body_state.pos_b_in_l.z(), 3.0, 1e-6);
  EXPECT_FALSE(ekf->m_state.body_state.ang_b_to_l.isApprox(Eigen::Quaterniond::Identity(), 1e-4));
}
