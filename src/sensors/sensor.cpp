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

#include <memory>
#include <sstream>
#include <string>

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
