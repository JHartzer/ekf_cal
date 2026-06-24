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

#include <cmath>
#include <vector>

#include "sensors/sim/sim_sensor.hpp"
#include "utility/sim/sim_rng.hpp"


SimSensor::SimSensor(Parameters params)
: m_no_errors(params.no_errors)
{
  assert(params.time_jitter >= 0.0 && "Delay jitter must be positive");
  if (m_no_errors) {
    m_time_jitter = 0.0;
    m_time_bias = 0.0;
  } else {
    m_time_jitter = params.time_jitter;
    m_time_bias = (params.time_bias_error == 0.0) ?
      0.0 :
      SimRNG::NormRand(0.0, params.time_bias_error);
  }
}

std::vector<SimSensor::TimingSample> SimSensor::GenerateMeasurementTimes(double m_rate) const
{
  auto num_measurements = static_cast<unsigned int>(std::floor(m_truth->m_max_time * m_rate));
  double time_init = m_no_errors ? 0 : SimRNG::UniRand(0.0, 1.0 / m_rate);

  std::vector<TimingSample> message_times;
  for (unsigned int i = 0; i < num_measurements; ++i) {
    const double true_time = static_cast<double>(i) / m_rate + time_init;
    message_times.push_back(
      TimingSample {
      true_time,
      ApplyTimeBias(true_time),
      ApplyTimeDelay(true_time)});
  }
  return message_times;
}

double SimSensor::ApplyTimeBias(double true_time) const
{
  return true_time + m_time_bias;
}

double SimSensor::ApplyTimeDelay(double true_time) const
{
  if (m_no_errors || m_time_jitter == 0.0) {
    return true_time;
  }
  return true_time + SimRNG::ExpRand(1.0 / m_time_jitter);
}
