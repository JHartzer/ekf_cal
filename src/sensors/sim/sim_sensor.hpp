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


#ifndef SENSORS__SIM__SIM_SENSOR_HPP_
#define SENSORS__SIM__SIM_SENSOR_HPP_

#include <memory>
#include <vector>

#include "infrastructure/sim/truth_engine.hpp"

///
/// @brief SimSensor namespace
///
class SimSensor
{
public:
  struct TimingSample
  {
    double time_true {0.0};      ///< @brief True sensing time
    double time_measured {0.0};  ///< @brief Sensor clock timestamp
    double time_received {0.0};  ///< @brief Local receive timestamp
  };

  ///
  /// @brief Sim IMU initialization parameters structure
  ///
  typedef struct Parameters
  {
    bool no_errors {false};        ///< @brief Perfect measurements flag
    double time_jitter {0.0};      ///< @brief Exponential delay mean (1 / lambda)
    double clock_bias {0.0};       ///< @brief Constant sensor clock bias
  } Parameters;

  ///
  /// @brief SimSensor constructor
  ///
  explicit SimSensor(Parameters params);

  ///
  /// @brief Generate list of true measurement times
  /// @param m_rate Sensor rate
  /// @return List of sensor measurement times
  ///
  std::vector<TimingSample> GenerateMeasurementTimes(double m_rate) const;

  ///
  /// @brief Apply sensor clock bias to true time
  /// @param true_time True measurement time
  /// @return Sensor clock timestamp
  ///
  double ApplyTimeBias(double true_time) const;

  ///
  /// @brief Apply transmission delay, if necessary, to true time
  /// @param true_time True measurement time
  /// @return Time with positive-only delay
  ///
  double ApplyTimeDelay(double true_time) const;

protected:
  bool m_no_errors {false};              ///< @brief Flag to remove measurement errors
  double m_time_jitter {0.0};            ///< @brief Exponential delay mean (1 / lambda)
  double m_clock_bias {0.0};             ///< @brief Constant sensor clock bias
  std::shared_ptr<TruthEngine> m_truth;  ///< @brief Truth engine pointer
};

#endif  // SENSORS__SIM__SIM_SENSOR_HPP_
