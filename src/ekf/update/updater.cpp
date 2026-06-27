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

#include "ekf/update/updater.hpp"

#include <memory>

#include "ekf/ekf.hpp"
#include "infrastructure/debug_logger.hpp"
#include "utility/math_helper.hpp"

Updater::Updater(
  unsigned int sensor_id,
  std::shared_ptr<DebugLogger> logger,
  SensorType sensor_type)
: m_id(sensor_id), m_logger(logger), m_sensor_type(sensor_type) {}

bool Updater::KalmanUpdate(
  EKF & ekf,
  const Eigen::MatrixXd & jacobian,
  const Eigen::VectorXd & residual,
  const Eigen::MatrixXd & measurement_noise
) const
{
  double chi2_threshold = ekf.GetChi2Threshold();
  Eigen::MatrixXd S;
  Eigen::MatrixXd observation_noise;

  if (ekf.GetUseRootCovariance()) {
    Eigen::MatrixXd innovation;
    observation_noise = measurement_noise.cwiseSqrt();
    innovation = QR_r(ekf.m_cov * jacobian.transpose(), observation_noise);
    S = innovation.transpose() * innovation;
  } else {
    observation_noise = measurement_noise;
    S = jacobian * ekf.m_cov * jacobian.transpose() + observation_noise;
  }

  if (m_sensor_type != SensorType::IMU && chi2_threshold > 0.0) {
    double mahalanobis_dist_sq = residual.dot(S.ldlt().solve(residual));
    if (mahalanobis_dist_sq > chi2_threshold) {
      if (ekf.GetDebugLogger()) {
        std::stringstream msg;
        msg << ToString(m_sensor_type) << " measurement rejected! Mahalanobis distance squared: "
            << mahalanobis_dist_sq << " exceeds threshold: " << chi2_threshold;
        ekf.GetDebugLogger()->Log(LogLevel::INFO, msg.str());
      }
      return false;
    }
  }

  // Calculate Kalman gain
  Eigen::MatrixXd gain;
  if (ekf.GetUseRootCovariance()) {
    Eigen::MatrixXd rhs = jacobian * ekf.m_cov.transpose() * ekf.m_cov;
    gain = S.ldlt().solve(rhs).transpose();
  } else {
    gain = S.ldlt().solve(jacobian * ekf.m_cov).transpose();
  }

  Eigen::VectorXd update = gain * residual;
  ekf.m_state += update;

  unsigned int rows = static_cast<unsigned int>(ekf.m_cov.rows());
  unsigned int cols = static_cast<unsigned int>(ekf.m_cov.cols());
  if (ekf.GetUseRootCovariance()) {
    ekf.m_cov = QR_r(
      ekf.m_cov * (Eigen::MatrixXd::Identity(rows, cols) -
      gain * jacobian).transpose(), observation_noise * gain.transpose());
  } else {
    ekf.m_cov =
      (Eigen::MatrixXd::Identity(rows, cols) - gain * jacobian) * ekf.m_cov *
      (Eigen::MatrixXd::Identity(rows, cols) - gain * jacobian).transpose() +
      gain * observation_noise * gain.transpose();
  }
  return true;
}
