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
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "ekf/update/msckf_updater.hpp"
#include "infrastructure/debug_logger.hpp"
#include "utility/custom_assertions.hpp"
#include "utility/math_helper.hpp"
#include "utility/type_helper.hpp"


TEST(test_msckf_updater, projection_jacobian) {
  Eigen::Vector3d base_pos {1, 2, 3};
  Eigen::MatrixXd jac_analytic = Eigen::MatrixXd::Zero(2, 3);
  MsckfUpdater::ProjectionJacobian(base_pos, jac_analytic);

  double delta = 1.0e-6;
  Eigen::MatrixXd jac_numerical = Eigen::MatrixXd::Zero(2, 3);
  Eigen::Vector2d base_meas = MsckfUpdater::Project(base_pos);
  for (unsigned int i = 0; i < 3; ++i) {
    Eigen::Vector3d delta_pos = base_pos;
    delta_pos[i] += delta;
    jac_numerical.block<2, 1>(0, i) = (MsckfUpdater::Project(delta_pos) - base_meas) / delta;
  }

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(jac_analytic, jac_numerical, 1e-3));
}


TEST(test_msckf_updater, distortion_jacobian) {
  Eigen::Vector2d xy_norm {1, 2};
  Intrinsics intrinsics;
  Eigen::MatrixXd jac_analytic = Eigen::MatrixXd::Zero(2, 2);
  MsckfUpdater::DistortionJacobian(xy_norm, intrinsics, jac_analytic);

  double delta = 1.0e-6;
  Eigen::MatrixXd jac_numerical = Eigen::MatrixXd::Zero(2, 2);
  Eigen::Vector2d base_meas = MsckfUpdater::Distort(xy_norm, intrinsics);
  for (unsigned int i = 0; i < 2; ++i) {
    Eigen::Vector2d delta_pos = xy_norm;
    delta_pos[i] += delta;
    jac_numerical.block<2, 1>(0, i) =
      (MsckfUpdater::Distort(delta_pos, intrinsics) - base_meas) / delta;
  }

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(jac_analytic, jac_numerical, 1e-3));
}


TEST(test_msckf_updater, update) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  EKF ekf(ekf_params);
  BodyState body_state;
  body_state.vel_b_in_l = Eigen::Vector3d{0, 5, 0};
  ekf.Initialize(0.0, body_state);

  unsigned int cam_id{1};

  CamState cam_state;
  Eigen::MatrixXd cam_cov = Eigen::MatrixXd::Zero(6, 6);
  ekf.RegisterCamera(cam_id, cam_state, cam_cov);

  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto msckf_updater = MsckfUpdater(1, false, "", 0.0, 1.0, 100.0, logger);

  double time {0.3};
  cv::KeyPoint point_1;
  cv::KeyPoint point_2;
  cv::KeyPoint point_3;
  point_1.pt.x = 320;
  point_1.pt.y = 240;
  point_2.pt.x = 220;
  point_2.pt.y = 240;
  point_3.pt.x = 120;
  point_3.pt.y = 240;

  FeaturePoint feature_point_1;
  FeaturePoint feature_point_2;
  FeaturePoint feature_point_3;
  feature_point_1.frame_id = 1;
  feature_point_1.key_point = point_1;
  feature_point_2.frame_id = 2;
  feature_point_2.key_point = point_2;
  feature_point_3.frame_id = 3;
  feature_point_3.key_point = point_3;

  FeatureTrack feature_points;
  feature_points.push_back(feature_point_1);
  feature_points.push_back(feature_point_2);
  feature_points.push_back(feature_point_3);

  FeatureTracks feature_tracks;
  feature_tracks.push_back(feature_points);

  ekf.PredictModel(0.1);
  ekf.AugmentStateIfNeeded(cam_id, feature_point_1.frame_id);
  ekf.PredictModel(0.2);
  ekf.AugmentStateIfNeeded(cam_id, feature_point_2.frame_id);
  ekf.PredictModel(0.3);
  ekf.AugmentStateIfNeeded(cam_id, feature_point_3.frame_id);

  msckf_updater.UpdateEKF(ekf, time, feature_tracks, 1e-3);
}


static Eigen::Vector2d predict_feature_projection(
  const Eigen::Vector3d & pos_f_in_l,
  const Eigen::Vector3d & pos_b_in_l,
  const Eigen::Quaterniond & ang_b_to_l,
  const Eigen::Vector3d & pos_c_in_b,
  const Eigen::Quaterniond & ang_c_to_b,
  const Intrinsics & intrinsics,
  bool use_distortion
)
{
  Eigen::Matrix3d rot_ci_to_b = ang_c_to_b.toRotationMatrix();
  Eigen::Matrix3d rot_bi_to_l = ang_b_to_l.toRotationMatrix();

  Eigen::Vector3d pos_f_in_bi = rot_bi_to_l.transpose() * (pos_f_in_l - pos_b_in_l);
  Eigen::Vector3d pos_f_in_ci = rot_ci_to_b.transpose() * (pos_f_in_bi - pos_c_in_b);

  Eigen::Vector2d xy_norm = MsckfUpdater::Project(pos_f_in_ci);
  if (use_distortion) {
    return MsckfUpdater::Distort(xy_norm, intrinsics);
  } else {
    return xy_norm;
  }
}

TEST(test_msckf_updater, state_and_feature_jacobian_correctness) {
  // Test both with and without distortion
  for (bool use_distortion : {false, true}) {
    Eigen::Vector3d pos_f_in_l{2.0, 3.5, 5.0};
    Eigen::Vector3d pos_b_in_l{0.5, -0.2, 1.1};
    Eigen::Quaterniond ang_b_to_l = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()) *
                                                       Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()) *
                                                       Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));

    Eigen::Vector3d pos_c_in_b{0.15, -0.05, 0.03};
    Eigen::Quaterniond ang_c_to_b = Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()) *
                                                      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                                                      Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitZ()));

    Intrinsics intrinsics;
    intrinsics.f_x = 0.01;
    intrinsics.f_y = 0.01;
    intrinsics.k_1 = -0.15;
    intrinsics.k_2 = 0.08;
    intrinsics.p_1 = 0.002;
    intrinsics.p_2 = -0.001;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.pixel_size = 5e-6;

    double delta = 1e-7;

    // Base prediction
    Eigen::Vector2d base_pred = predict_feature_projection(
      pos_f_in_l, pos_b_in_l, ang_b_to_l, pos_c_in_b, ang_c_to_b, intrinsics, use_distortion);

    // Compute base coordinates for analytical Jacobians
    Eigen::Matrix3d rot_ci_to_b = ang_c_to_b.toRotationMatrix();
    Eigen::Matrix3d rot_bi_to_l = ang_b_to_l.toRotationMatrix();
    Eigen::Matrix3d rot_b_to_ci = rot_ci_to_b.transpose();
    Eigen::Matrix3d rot_l_to_ci = rot_b_to_ci * rot_bi_to_l.transpose();

    Eigen::Vector3d pos_f_in_bi = rot_bi_to_l.transpose() * (pos_f_in_l - pos_b_in_l);
    Eigen::Vector3d pos_f_in_ci = rot_ci_to_b.transpose() * (pos_f_in_bi - pos_c_in_b);
    Eigen::Vector2d xy_norm = MsckfUpdater::Project(pos_f_in_ci);

    // Analytical H_p (Projection Jacobian)
    Eigen::MatrixXd H_p(2, 3);
    MsckfUpdater::ProjectionJacobian(pos_f_in_ci, H_p);

    // Analytical H_d (Distortion Jacobian)
    Eigen::MatrixXd H_d = Eigen::MatrixXd::Identity(2, 2);
    if (use_distortion) {
      // Evaluate H_d at xy_norm
      MsckfUpdater::DistortionJacobian(xy_norm, intrinsics, H_d);
    }

    // Feature Position Jacobian H_f
    Eigen::MatrixXd H_f_analytical = H_d * H_p * rot_l_to_ci;
    Eigen::MatrixXd H_f_numerical = Eigen::MatrixXd::Zero(2, 3);
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d pos_f_perturbed = pos_f_in_l;
      pos_f_perturbed[i] += delta;
      Eigen::Vector2d pred = predict_feature_projection(
        pos_f_perturbed, pos_b_in_l, ang_b_to_l, pos_c_in_b, ang_c_to_b, intrinsics, use_distortion);
      H_f_numerical.col(i) = (pred - base_pred) / delta;
    }
    EXPECT_TRUE(EXPECT_EIGEN_NEAR(H_f_analytical, H_f_numerical, 1e-3));

    // Body/Augmented Position Jacobian
    Eigen::MatrixXd H_pos_analytical = -H_d * H_p * rot_l_to_ci;
    Eigen::MatrixXd H_pos_numerical = Eigen::MatrixXd::Zero(2, 3);
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d pos_b_perturbed = pos_b_in_l;
      pos_b_perturbed[i] += delta;
      Eigen::Vector2d pred = predict_feature_projection(
        pos_f_in_l, pos_b_perturbed, ang_b_to_l, pos_c_in_b, ang_c_to_b, intrinsics, use_distortion);
      H_pos_numerical.col(i) = (pred - base_pred) / delta;
    }
    EXPECT_TRUE(EXPECT_EIGEN_NEAR(H_pos_analytical, H_pos_numerical, 1e-3));

    // Body/Augmented Orientation Jacobian
    Eigen::MatrixXd H_t_body = rot_b_to_ci * SkewSymmetric(rot_bi_to_l.transpose() * (pos_f_in_l - pos_b_in_l));
    Eigen::MatrixXd H_rot_analytical_body = H_d * H_p * H_t_body;

    Eigen::MatrixXd H_rot_numerical = Eigen::MatrixXd::Zero(2, 3);
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d rot_perturb = Eigen::Vector3d::Zero();
      rot_perturb[i] = delta;
      Eigen::Quaterniond ang_b_perturbed = ang_b_to_l * RotVecToQuat(rot_perturb);
      Eigen::Vector2d pred = predict_feature_projection(
        pos_f_in_l, pos_b_in_l, ang_b_perturbed, pos_c_in_b, ang_c_to_b, intrinsics, use_distortion);
      H_rot_numerical.col(i) = (pred - base_pred) / delta;
    }
    EXPECT_TRUE(EXPECT_EIGEN_NEAR(H_rot_analytical_body, H_rot_numerical, 1e-3));

    // Camera Extrinsic Position Jacobian
    Eigen::MatrixXd H_cam_pos_analytical = -H_d * H_p * rot_b_to_ci;
    Eigen::MatrixXd H_cam_pos_numerical = Eigen::MatrixXd::Zero(2, 3);
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d pos_c_perturbed = pos_c_in_b;
      pos_c_perturbed[i] += delta;
      Eigen::Vector2d pred = predict_feature_projection(
        pos_f_in_l, pos_b_in_l, ang_b_to_l, pos_c_perturbed, ang_c_to_b, intrinsics, use_distortion);
      H_cam_pos_numerical.col(i) = (pred - base_pred) / delta;
    }
    EXPECT_TRUE(EXPECT_EIGEN_NEAR(H_cam_pos_analytical, H_cam_pos_numerical, 1e-3));

    // Camera Extrinsic Orientation Jacobian
    Eigen::MatrixXd H_cam_rot_analytical_body = H_d * H_p * SkewSymmetric(pos_f_in_ci);

    Eigen::MatrixXd H_cam_rot_numerical = Eigen::MatrixXd::Zero(2, 3);
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d rot_perturb = Eigen::Vector3d::Zero();
      rot_perturb[i] = delta;
      Eigen::Quaterniond ang_c_perturbed = ang_c_to_b * RotVecToQuat(rot_perturb);
      Eigen::Vector2d pred = predict_feature_projection(
        pos_f_in_l, pos_b_in_l, ang_b_to_l, pos_c_in_b, ang_c_perturbed, intrinsics, use_distortion);
      H_cam_rot_numerical.col(i) = (pred - base_pred) / delta;
    }
    EXPECT_TRUE(EXPECT_EIGEN_NEAR(H_cam_rot_analytical_body, H_cam_rot_numerical, 1e-3));
  }
}
