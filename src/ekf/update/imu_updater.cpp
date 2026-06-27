// Copyright 2023 Jacob Hartzer
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

#include "ekf/update/imu_updater.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "ekf/constants.hpp"
#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "ekf/update/updater.hpp"
#include "infrastructure/debug_logger.hpp"
#include "utility/math_helper.hpp"
#include "utility/string_helper.hpp"


ImuUpdater::ImuUpdater(
  unsigned int imu_id,
  bool is_extrinsic,
  bool is_intrinsic,
  const std::string & log_file_directory,
  double data_log_rate,
  std::shared_ptr<DebugLogger> logger
)
: Updater(imu_id, logger),
  m_is_extrinsic(is_extrinsic),
  m_is_intrinsic(is_intrinsic),
  m_data_logger(log_file_directory, "imu_" + std::to_string(imu_id))
{
  std::stringstream header;
  header << "time,stationary,score";
  header << EnumerateHeader("imu_pos", 3);
  header << EnumerateHeader("imu_ang_pos", 4);
  header << EnumerateHeader("imu_acc_bias", 3);
  header << EnumerateHeader("imu_gyr_bias", 3);
  if (m_is_extrinsic) {header << EnumerateHeader("imu_ext_cov", g_imu_extrinsic_state_size);}
  if (m_is_intrinsic) {header << EnumerateHeader("imu_int_cov", g_imu_intrinsic_state_size);}
  header << EnumerateHeader("acc", 3);
  header << EnumerateHeader("omg", 3);
  header << EnumerateHeader("residual", 6);
  header << EnumerateHeader("duration", 1);

  m_data_logger.DefineHeader(header.str());
  if (data_log_rate != 0.0) {m_data_logger.EnableLogging();}
  m_data_logger.SetLogRate(data_log_rate);
}

void ImuUpdater::UpdateEKF(
  EKF & ekf,
  const double time,
  const Eigen::Vector3d & acceleration,
  const Eigen::Matrix3d & acceleration_covariance,
  const Eigen::Vector3d & angular_rate,
  const Eigen::Matrix3d & angular_rate_covariance
)
{
  double local_time = ekf.CalculateLocalTime(time);

  // Check for zero velocity
  if (ZeroAccelerationUpdate(
      ekf,
      local_time,
      acceleration,
      acceleration_covariance,
      angular_rate,
      angular_rate_covariance))
  {
    ekf.SetZeroAcceleration(true);
    return;
  }

  ekf.SetZeroAcceleration(false);

  auto t_start = std::chrono::high_resolution_clock::now();

  // Perform Update
  Eigen::Quaterniond ang_b_to_l = ekf.m_state.body_state.ang_b_to_l;
  Eigen::Quaterniond ang_i_to_b = ekf.m_state.imu_states[m_id].ang_i_to_b;
  Eigen::Vector3d acc_bias = ekf.m_state.imu_states[m_id].acc_bias;
  Eigen::Vector3d omg_bias = ekf.m_state.imu_states[m_id].omg_bias;
  Eigen::VectorXd resid = Eigen::VectorXd::Zero(6);

  if (ekf.m_state.body_state.size == g_body_state_min_size) {
    ekf.m_state.body_state.acc_b_in_l = ang_b_to_l * ang_i_to_b * (acceleration - acc_bias);
    ekf.m_state.body_state.ang_vel_b_in_l = ang_b_to_l * ang_i_to_b * (angular_rate - omg_bias);
    ekf.m_state.body_state.ang_acc_b_in_l = Eigen::Vector3d::Zero();
  } else {
    Eigen::VectorXd measurement(acceleration.size() + angular_rate.size());
    measurement.segment<3>(0) = acceleration;
    measurement.segment<3>(3) = angular_rate;

    Eigen::VectorXd pred_measurement = PredictMeasurement(ekf);
    resid = measurement - pred_measurement;

    Eigen::MatrixXd jacobian = GetMeasurementJacobian(ekf);

    Eigen::MatrixXd meas_noise = Eigen::MatrixXd::Zero(6, 6);
    meas_noise.block<3, 3>(0, 0) = acceleration_covariance;
    meas_noise.block<3, 3>(3, 3) = angular_rate_covariance;

    // Apply Kalman update
    KalmanUpdate(ekf, jacobian, resid, meas_noise, "IMU");
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  auto t_execution = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

  ekf.PredictModel(local_time);

  // Write outputs
  std::stringstream msg;

  msg << local_time;
  msg << ",0," << ekf.GetMotionDetectionChiSquared();
  msg << VectorToCommaString(ekf.m_state.imu_states[m_id].pos_i_in_b);
  msg << QuaternionToCommaString(ekf.m_state.imu_states[m_id].ang_i_to_b);
  msg << VectorToCommaString(ekf.m_state.imu_states[m_id].acc_bias);
  msg << VectorToCommaString(ekf.m_state.imu_states[m_id].omg_bias);
  if (m_is_extrinsic || m_is_intrinsic) {
    unsigned int imu_index = ekf.m_state.imu_states[m_id].index;
    unsigned int imu_size = ekf.m_state.imu_states[m_id].size;
    Eigen::VectorXd cov_diag =
      ekf.m_cov.block(imu_index, imu_index, imu_size, imu_size).diagonal();
    if (ekf.GetUseRootCovariance()) {
      cov_diag = cov_diag.cwiseProduct(cov_diag);
    }
    msg << VectorToCommaString(cov_diag);
  }
  msg << VectorToCommaString(acceleration);
  msg << VectorToCommaString(angular_rate);
  msg << VectorToCommaString(resid);
  msg << "," << t_execution.count();
  m_data_logger.RateLimitedLog(msg.str(), local_time);

  ekf.LogBodyStateIfNeeded(static_cast<int>(t_execution.count()));
}

Eigen::MatrixXd ImuUpdater::GetZeroAccelerationJacobian(EKF & ekf) const
{
  unsigned int meas_size = m_is_intrinsic ? 6 : 3;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(meas_size, ekf.GetStateSize());

  Eigen::Quaterniond ang_i_to_b = ekf.m_state.imu_states[m_id].ang_i_to_b;
  Eigen::Quaterniond ang_b_to_l = ekf.m_state.body_state.ang_b_to_l;

  jacobian.block<3, 3>(
    0,
    ekf.GetOrientationStateIndex()) = -SkewSymmetric(
    ang_i_to_b.conjugate() *
    ang_b_to_l.conjugate() * g_gravity) *
    QuaternionJacobian(ang_b_to_l).transpose();

  if (m_is_extrinsic) {
    unsigned int index_extrinsic = ekf.m_state.imu_states[m_id].index_extrinsic;
    jacobian.block<3, 3>(0, index_extrinsic) = -SkewSymmetric(
      ang_i_to_b.conjugate() * ang_b_to_l.conjugate() * g_gravity
    );
  }

  if (m_is_intrinsic) {
    unsigned int index_intrinsic = ekf.m_state.imu_states[m_id].index_intrinsic;
    jacobian.block<3, 3>(0, index_intrinsic + 0) = -Eigen::Matrix3d::Identity();
    jacobian.block<3, 3>(3, index_intrinsic + 3) = -Eigen::Matrix3d::Identity();
  }

  return jacobian;
}

bool ImuUpdater::ZeroAccelerationUpdate(
  EKF & ekf,
  double local_time,
  const Eigen::Vector3d & acceleration,
  const Eigen::Matrix3d & acceleration_covariance,
  const Eigen::Vector3d & angular_rate,
  const Eigen::Matrix3d & angular_rate_covariance
)
{
  auto t_start = std::chrono::high_resolution_clock::now();

  if (m_initial_motion_detected) {
    return false;
  }

  unsigned int meas_size = m_is_intrinsic ? 6 : 3;
  Eigen::Quaterniond ang_i_to_b = ekf.m_state.imu_states[m_id].ang_i_to_b;
  Eigen::Quaterniond ang_b_to_l = ekf.m_state.body_state.ang_b_to_l;

  Eigen::MatrixXd jacobian = GetZeroAccelerationJacobian(ekf);

  Eigen::MatrixXd meas_noise = Eigen::MatrixXd::Zero(meas_size, meas_size);
  meas_noise.block<3, 3>(0, 0) = acceleration_covariance;
  if (m_is_intrinsic) {
    meas_noise.block<3, 3>(3, 3) = angular_rate_covariance;
  }

  Eigen::Vector3d bias_a = ekf.m_state.imu_states[m_id].acc_bias;
  Eigen::Vector3d bias_g = ekf.m_state.imu_states[m_id].omg_bias;

  Eigen::VectorXd resid = Eigen::VectorXd::Zero(meas_size);
  resid.segment<3>(0) = -(acceleration - bias_a -
    ang_i_to_b.conjugate() *
    ang_b_to_l.conjugate() * g_gravity);
  if (m_is_intrinsic) {
    resid.segment<3>(3) = -(angular_rate - bias_g);
  }

  Eigen::MatrixXd score_mat = resid.transpose() *
    (jacobian * ekf.m_cov * jacobian.transpose() + ekf.GetImuNoiseScaleFactor() *
    meas_noise).inverse() * resid;

  double score = std::abs(score_mat(0, 0));
  if (score > ekf.GetMotionDetectionChiSquared() && ekf.IsGravityInitialized()) {
    m_initial_motion_detected = true;
    return false;
  } else if (score < ekf.GetMotionDetectionChiSquared()) {
    ekf.InitializeGravity();
  }

  /// @todo Test stationary rotation
  // AngularUpdate(ekf, angular_rate, angular_rate_covariance);

  ekf.PredictModel(local_time);

  // Update Jacobian
  jacobian = GetZeroAccelerationJacobian(ekf);

  // Apply Kalman update
  // Eigen::Quaterniond ang_b_to_l_pre = ekf.m_state.body_state.ang_b_to_l;
  KalmanUpdate(ekf, jacobian, resid, meas_noise);

  ekf.m_state.body_state.acc_b_in_l = g_gravity;
  ekf.m_state.body_state.ang_vel_b_in_l = Eigen::Vector3d::Zero();
  ekf.m_state.body_state.ang_acc_b_in_l = Eigen::Vector3d::Zero();

  // Prevent unintentional rotation about the vertical axis
  // if (m_correct_heading_rotation) {
  //   Eigen::Vector3d x_axis_body_pre = ang_b_to_l_pre.inverse() * Eigen::Vector3d::UnitX();
  //   Eigen::Vector3d x_axis_body = ekf.m_state.body_state.ang_b_to_l.inverse() *
  //     Eigen::Vector3d::UnitX();
  //   Eigen::Vector3d plane_normal = g_gravity.cross(x_axis_body_pre) / g_gravity.norm();
  //   double angle = M_PI / 2 -
  //     std::acos(x_axis_body.dot(plane_normal) / x_axis_body.norm() / plane_normal.norm());
  //   Eigen::Vector3d rotation_axis = x_axis_body.cross(plane_normal);
  //   auto correction = Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_axis));
  //   ekf.m_state.body_state.ang_b_to_l = ekf.m_state.body_state.ang_b_to_l * correction;
  // }

  auto t_end = std::chrono::high_resolution_clock::now();
  auto t_execution = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

  // Write outputs
  std::stringstream msg;

  msg << local_time;
  msg << ",1," << score;
  msg << VectorToCommaString(ekf.m_state.imu_states[m_id].pos_i_in_b);
  msg << QuaternionToCommaString(ekf.m_state.imu_states[m_id].ang_i_to_b);
  msg << VectorToCommaString(ekf.m_state.imu_states[m_id].acc_bias);
  msg << VectorToCommaString(ekf.m_state.imu_states[m_id].omg_bias);

  if (m_is_extrinsic) {
    unsigned int index_extrinsic = ekf.m_state.imu_states[m_id].index_extrinsic;
    Eigen::VectorXd ex_diag = ekf.m_cov.block(index_extrinsic, index_extrinsic, 6, 6).diagonal();
    msg << VectorToCommaString(ex_diag.cwiseProduct(ex_diag));
  }

  if (m_is_intrinsic) {
    unsigned int index_intrinsic = ekf.m_state.imu_states[m_id].index_intrinsic;
    Eigen::VectorXd in_diag = ekf.m_cov.block(index_intrinsic, index_intrinsic, 6, 6).diagonal();
    msg << VectorToCommaString(in_diag.cwiseProduct(in_diag));
  }

  msg << VectorToCommaString(acceleration);
  msg << VectorToCommaString(angular_rate);
  msg << VectorToCommaString(resid);

  if (!m_is_intrinsic) {
    msg << VectorToCommaString(Eigen::Vector3d::Zero());
  }

  msg << "," << t_execution.count();
  m_data_logger.RateLimitedLog(msg.str(), local_time);

  ekf.LogBodyStateIfNeeded(static_cast<int>(t_execution.count()));

  return true;
}

Eigen::VectorXd ImuUpdater::PredictMeasurement(EKF & ekf) const
{
  Eigen::Vector3d pos_i_in_b = ekf.m_state.imu_states[m_id].pos_i_in_b;
  Eigen::Quaterniond ang_i_to_b = ekf.m_state.imu_states[m_id].ang_i_to_b;
  Eigen::Vector3d acc_bias = ekf.m_state.imu_states[m_id].acc_bias;
  Eigen::Vector3d omg_bias = ekf.m_state.imu_states[m_id].omg_bias;
  Eigen::Quaterniond ang_b_to_l = ekf.m_state.body_state.ang_b_to_l;

  Eigen::VectorXd predicted_measurement(6);

  Eigen::Vector3d body_acc_b = ang_b_to_l.conjugate() * ekf.m_state.body_state.acc_b_in_l;
  Eigen::Vector3d body_ang_vel_b = ang_b_to_l.conjugate() * ekf.m_state.body_state.ang_vel_b_in_l;
  Eigen::Vector3d body_ang_acc_b = ang_b_to_l.conjugate() * ekf.m_state.body_state.ang_acc_b_in_l;

  Eigen::Vector3d imu_acc_b =
    body_acc_b +
    body_ang_acc_b.cross(pos_i_in_b) +
    body_ang_vel_b.cross(body_ang_vel_b.cross(pos_i_in_b));

  Eigen::Vector3d imu_omg_b = body_ang_vel_b;

  // Rotate measurements in place
  predicted_measurement.segment<3>(0) = acc_bias + ang_i_to_b.conjugate() * imu_acc_b;
  predicted_measurement.segment<3>(3) = omg_bias + ang_i_to_b.conjugate() * imu_omg_b;

  return predicted_measurement;
}

Eigen::MatrixXd ImuUpdater::GetMeasurementJacobian(EKF & ekf) const
{
  Eigen::Vector3d pos_i_in_b = ekf.m_state.imu_states[m_id].pos_i_in_b;
  Eigen::Quaterniond ang_i_to_b = ekf.m_state.imu_states[m_id].ang_i_to_b;
  Eigen::Quaterniond ang_b_to_l = ekf.m_state.body_state.ang_b_to_l;
  Eigen::Matrix3d rot_l_to_b = ang_b_to_l.conjugate().toRotationMatrix();
  Eigen::Matrix3d rot_b_to_i = ang_i_to_b.conjugate().toRotationMatrix();
  Eigen::Vector3d body_acc_b = rot_l_to_b * ekf.m_state.body_state.acc_b_in_l;
  Eigen::Vector3d body_ang_vel_b = rot_l_to_b * ekf.m_state.body_state.ang_vel_b_in_l;
  Eigen::Vector3d body_ang_acc_b = rot_l_to_b * ekf.m_state.body_state.ang_acc_b_in_l;
  Eigen::Matrix3d omega_jacobian_b =
    SkewSymmetric(body_ang_vel_b) * SkewSymmetric(pos_i_in_b).transpose() +
    SkewSymmetric(body_ang_vel_b.cross(pos_i_in_b)).transpose();
  Eigen::Matrix3d body_orientation_acc_jacobian =
    SkewSymmetric(body_acc_b) -
    SkewSymmetric(pos_i_in_b) * SkewSymmetric(body_ang_acc_b) +
    omega_jacobian_b * SkewSymmetric(body_ang_vel_b);
  Eigen::Vector3d imu_acc_i = rot_b_to_i * (
    body_acc_b +
    body_ang_acc_b.cross(pos_i_in_b) +
    body_ang_vel_b.cross(body_ang_vel_b.cross(pos_i_in_b)));
  Eigen::Vector3d imu_omg_i = rot_b_to_i * body_ang_vel_b;

  Eigen::MatrixXd measurement_jacobian = Eigen::MatrixXd::Zero(6, ekf.GetStateSize());

  // Body Acceleration
  measurement_jacobian.block<3, 3>(0, 6) = rot_b_to_i * rot_l_to_b;

  // Body Orientation
  measurement_jacobian.block<3, 3>(0, 9) = rot_b_to_i * body_orientation_acc_jacobian;

  // Body Angular Velocity
  measurement_jacobian.block<3, 3>(0, 12) = rot_b_to_i * omega_jacobian_b * rot_l_to_b;

  // Body Angular Acceleration
  measurement_jacobian.block<3, 3>(0, 15) = -rot_b_to_i * SkewSymmetric(pos_i_in_b) * rot_l_to_b;

  // Body Orientation
  measurement_jacobian.block<3, 3>(3, 9) = rot_b_to_i * SkewSymmetric(body_ang_vel_b);

  // Body Angular Velocity
  measurement_jacobian.block<3, 3>(3, 12) = rot_b_to_i * rot_l_to_b;

  if (m_is_extrinsic) {
    unsigned int index_extrinsic = ekf.m_state.imu_states[m_id].index_extrinsic;

    // IMU Positional Offset
    measurement_jacobian.block<3, 3>(0, index_extrinsic + 0) =
      rot_b_to_i * (SkewSymmetric(body_ang_acc_b) +
      SkewSymmetric(body_ang_vel_b) * SkewSymmetric(body_ang_vel_b)
      );

    // IMU Angular Offset
    measurement_jacobian.block<3, 3>(0, index_extrinsic + 3) = SkewSymmetric(imu_acc_i);

    // IMU Angular Offset
    measurement_jacobian.block<3, 3>(3, index_extrinsic + 3) = SkewSymmetric(imu_omg_i);
  }

  if (m_is_intrinsic) {
    unsigned int index_intrinsic = ekf.m_state.imu_states[m_id].index_intrinsic;
    measurement_jacobian.block<3, 3>(0, index_intrinsic + 0) = Eigen::Matrix3d::Identity();
    measurement_jacobian.block<3, 3>(3, index_intrinsic + 3) = Eigen::Matrix3d::Identity();
  }

  return measurement_jacobian;
}

void ImuUpdater::AngularUpdate(
  EKF & ekf,
  const Eigen::Vector3d & angular_rate,
  const Eigen::Matrix3d & angular_rate_covariance
) const
{
  Eigen::Quaterniond ang_b_to_l = ekf.m_state.body_state.ang_b_to_l;
  Eigen::Quaterniond ang_i_to_b = ekf.m_state.imu_states[m_id].ang_i_to_b;
  Eigen::Vector3d omg_bias = ekf.m_state.imu_states[m_id].omg_bias;

  if (ekf.m_state.body_state.size == g_body_state_min_size) {
    ekf.m_state.body_state.ang_vel_b_in_l = ang_b_to_l * ang_i_to_b * (angular_rate - omg_bias);
    ekf.m_state.body_state.ang_acc_b_in_l = Eigen::Vector3d::Zero();
  } else {
    Eigen::VectorXd measurement(3);
    measurement.segment<3>(0) = angular_rate;

    Eigen::VectorXd pred_measurement =
      ang_i_to_b.conjugate() *
      ang_b_to_l.conjugate() * ekf.m_state.body_state.ang_vel_b_in_l + omg_bias;
    Eigen::VectorXd resid = measurement - pred_measurement;

    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 9);
    jacobian.block<3, 3>(0, 3) = Eigen::MatrixXd::Identity(3, 3);

    Eigen::MatrixXd meas_noise = Eigen::MatrixXd::Zero(3, 3);
    meas_noise.block<3, 3>(0, 0) = angular_rate_covariance;

    // Apply Kalman update
    KalmanUpdate(ekf, jacobian, resid, meas_noise, "IMU");
  }
}
