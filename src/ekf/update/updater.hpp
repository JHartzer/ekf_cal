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

#ifndef EKF__UPDATE__UPDATER_HPP_
#define EKF__UPDATE__UPDATER_HPP_

#include <memory>

#include "ekf/ekf.hpp"
#include "infrastructure/debug_logger.hpp"

///
/// @class Updater
/// @brief Base class for EKF updater classes
///
class Updater
{
public:
  ///
  /// @brief EKF Updater constructor
  /// @param sensor_id Sensor ID
  /// @param logger Debug logger pointer
  /// @param sensor_type Sensor type used to control updater behavior
  ///
  explicit Updater(
    unsigned int sensor_id,
    std::shared_ptr<DebugLogger> logger,
    SensorType sensor_type = SensorType::Sensor);

  bool KalmanUpdate(
    EKF & ekf,
    const Eigen::MatrixXd & jacobian,
    const Eigen::VectorXd & residual,
    const Eigen::MatrixXd & measurement_noise_input) const;

protected:
  unsigned int m_id;                      ///< @brief Associated sensor ID
  std::shared_ptr<DebugLogger> m_logger;  ///< @brief Debug logger
  SensorType m_sensor_type;               ///< @brief Sensor type for updater behavior
};

#endif  // EKF__UPDATE__UPDATER_HPP_
