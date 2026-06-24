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

#include "sensors/sensor.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "infrastructure/data_logger.hpp"
#include "infrastructure/debug_logger.hpp"
#include "sensors/sensor_message.hpp"

// Initialize static variable
unsigned int Sensor::m_sensor_count = 0;

Sensor::Sensor(Parameters params)
: m_id(++m_sensor_count)
{
  m_name = params.name;
  m_logger = params.logger;
  m_filter_sensor_time = params.filter_sensor_time;
  m_measurement_time_reorder_window = std::max(0.0, params.measurement_time_reorder_window);
}

unsigned int Sensor::GetId() const
{
  return m_id;
}

std::string Sensor::GetName() const
{
  return m_name;
}

bool MessageCompare(std::shared_ptr<SensorMessage> l_msg, std::shared_ptr<SensorMessage> r_msg)
{
  return l_msg->time_received < r_msg->time_received;
}

void Sensor::InitializeTimingLogger(
  const std::string & log_prefix,
  const std::string & log_directory,
  double data_log_rate)
{
  m_timing_logger.SetOutputDirectory(log_directory);
  m_timing_logger.SetName(log_prefix + "_" + std::to_string(m_id) + "_timing");
  m_timing_logger.DefineHeader(
    "time_used,time_measured,time_received,time_true,time_offset_sample,time_offset_min,"
    "time_alignment_error");
  if (data_log_rate != 0.0) {
    m_timing_logger.EnableLogging();
  }
  m_timing_logger.SetLogRate(data_log_rate);
}

void Sensor::UpdateTimeFilter(const SensorMessage & sensor_message)
{
  if (!m_filter_sensor_time) {
    return;
  }

  const double offset_sample = sensor_message.time_received - sensor_message.time_measured;
  if (!m_min_offset_initialized || offset_sample < m_min_offset) {
    m_min_offset = offset_sample;
    m_min_offset_initialized = true;
  }
}

double Sensor::GetTimeUsed(const SensorMessage & sensor_message) const
{
  if (!m_filter_sensor_time || !m_min_offset_initialized) {
    return sensor_message.time_measured;
  }
  return sensor_message.time_measured + m_min_offset;
}

void Sensor::LogTiming(const SensorMessage & sensor_message)
{
  const double true_time = sensor_message.GetTimeTrue();
  const double offset_sample = sensor_message.time_received - sensor_message.time_measured;
  const double offset_min = m_min_offset_initialized ? m_min_offset : offset_sample;
  double alignment_error = std::numeric_limits<double>::quiet_NaN();
  if (!std::isnan(true_time)) {
    alignment_error = sensor_message.time_used - true_time;
  }

  m_timing_logger.RateLimitedLog(
    std::vector<double> {
    sensor_message.time_used,
    sensor_message.time_measured,
    sensor_message.time_received,
    true_time,
    offset_sample,
    offset_min,
    alignment_error},
    sensor_message.time_used);
}

void Sensor::Callback(const SensorMessage sensor_message) const
{
  std::stringstream msg;
  msg << "Base Sensor callback invoked at measured time " << sensor_message.time_measured;
  msg << ", received time " << sensor_message.time_received;
  msg << " for sensor " << sensor_message.sensor_id;
  msg << " of type " << static_cast<unsigned int>(sensor_message.sensor_type);
  m_logger->Log(LogLevel::INFO, msg.str());
  m_logger->Log(LogLevel::INFO, "Base Sensor callback invoked");
}

void Sensor::Flush() {}

bool Sensor::HasBufferedMeasurements() const
{
  return false;
}

double Sensor::GetNextBufferedMeasurementTime() const
{
  return 0.0;
}

bool Sensor::FlushNextMeasurement()
{
  return false;
}
