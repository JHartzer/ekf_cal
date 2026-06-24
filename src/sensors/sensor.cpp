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
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "infrastructure/data_logger.hpp"
#include "infrastructure/debug_logger.hpp"
#include "sensors/sensor_message.hpp"

// Initialize static variable
unsigned int Sensor::m_sensor_count = 0;

Sensor::MeasurementScheduler::MeasurementScheduler(double measurement_time_reorder_window)
: m_measurement_time_reorder_window(std::max(0.0, measurement_time_reorder_window))
{}

unsigned int Sensor::MeasurementScheduler::Enqueue(
  unsigned int sensor_id,
  double time_used,
  double time_received,
  std::function<void()> execute_fn)
{
  PendingMeasurement measurement;
  measurement.sequence_id = m_next_sequence_id++;
  measurement.sensor_id = sensor_id;
  measurement.time_used = time_used;
  measurement.time_received = time_received;
  measurement.execute_fn = std::move(execute_fn);
  m_pending_measurements.push_back(std::move(measurement));
  m_latest_time_received = std::max(m_latest_time_received, time_received);
  SortPending();
  return ReleaseReadyMeasurements(false);
}

unsigned int Sensor::MeasurementScheduler::FlushAll()
{
  return ReleaseReadyMeasurements(true);
}

unsigned int Sensor::MeasurementScheduler::FlushSensor(unsigned int sensor_id)
{
  unsigned int executed_count = 0;
  auto iter = m_pending_measurements.begin();
  while (iter != m_pending_measurements.end()) {
    if (iter->sensor_id != sensor_id) {
      ++iter;
      continue;
    }

    iter->execute_fn();
    iter = m_pending_measurements.erase(iter);
    executed_count++;
  }
  return executed_count;
}

bool Sensor::MeasurementScheduler::HasPending(unsigned int sensor_id) const
{
  for (const auto & measurement : m_pending_measurements) {
    if (measurement.sensor_id == sensor_id) {
      return true;
    }
  }
  return false;
}

double Sensor::MeasurementScheduler::GetNextPendingTime(unsigned int sensor_id) const
{
  for (const auto & measurement : m_pending_measurements) {
    if (measurement.sensor_id == sensor_id) {
      return measurement.time_used;
    }
  }
  return 0.0;
}

bool Sensor::MeasurementScheduler::FlushNext(unsigned int sensor_id)
{
  for (auto iter = m_pending_measurements.begin(); iter != m_pending_measurements.end(); ++iter) {
    if (iter->sensor_id != sensor_id) {
      continue;
    }

    iter->execute_fn();
    m_pending_measurements.erase(iter);
    return true;
  }
  return false;
}

void Sensor::MeasurementScheduler::SortPending()
{
  std::stable_sort(
    m_pending_measurements.begin(),
    m_pending_measurements.end(),
    [](const PendingMeasurement & left, const PendingMeasurement & right) {
      if (left.time_used == right.time_used) {
        if (left.time_received == right.time_received) {
          return left.sequence_id < right.sequence_id;
        }
        return left.time_received < right.time_received;
      }
      return left.time_used < right.time_used;
    });
}

unsigned int Sensor::MeasurementScheduler::ReleaseReadyMeasurements(bool flush_all)
{
  unsigned int executed_count = 0;
  while (!m_pending_measurements.empty()) {
    const auto & next_measurement = m_pending_measurements.front();
    if (!flush_all &&
      (m_latest_time_received - next_measurement.time_used) < m_measurement_time_reorder_window)
    {
      break;
    }

    next_measurement.execute_fn();
    m_pending_measurements.erase(m_pending_measurements.begin());
    executed_count++;
  }
  return executed_count;
}

Sensor::Sensor(Parameters params)
: m_id(++m_sensor_count)
{
  m_name = params.name;
  m_logger = params.logger;
  m_filter_sensor_time = params.filter_sensor_time;
  m_measurement_time_reorder_window = std::max(0.0, params.measurement_time_reorder_window);
  m_measurement_scheduler = params.measurement_scheduler;
  if (!m_measurement_scheduler) {
    m_measurement_scheduler =
      std::make_shared<MeasurementScheduler>(m_measurement_time_reorder_window);
  }
}

unsigned int Sensor::GetId() const
{
  return m_id;
}

std::string Sensor::GetName() const
{
  return m_name;
}

std::shared_ptr<Sensor::MeasurementScheduler> Sensor::GetMeasurementScheduler() const
{
  return m_measurement_scheduler;
}

std::uint64_t Sensor::GetExecutionCount() const
{
  return m_execution_count;
}

void Sensor::RecordMeasurementExecution(double time_used)
{
  m_execution_count++;
  (void)time_used;
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

double Sensor::GetTimeUsed(const SensorMessage & sensor_message)
{
  if (!m_filter_sensor_time) {
    return sensor_message.time_measured;
  }

  const double offset_sample = sensor_message.time_received - sensor_message.time_measured;
  if (!m_min_offset_initialized || offset_sample < m_min_offset) {
    m_min_offset = offset_sample;
    m_min_offset_initialized = true;
  }

  double time_used = sensor_message.time_measured + m_min_offset;
  time_used = std::min(time_used, sensor_message.time_received);
  if (m_last_assigned_time_used_initialized) {
    time_used = std::max(time_used, m_last_assigned_time_used);
  }

  m_last_assigned_time_used = time_used;
  m_last_assigned_time_used_initialized = true;
  return time_used;
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

void Sensor::Flush()
{
  if (m_measurement_scheduler) {
    m_measurement_scheduler->FlushSensor(m_id);
  }
}

bool Sensor::HasBufferedMeasurements() const
{
  return m_measurement_scheduler && m_measurement_scheduler->HasPending(m_id);
}

double Sensor::GetNextBufferedMeasurementTime() const
{
  if (!m_measurement_scheduler) {
    return 0.0;
  }
  return m_measurement_scheduler->GetNextPendingTime(m_id);
}

bool Sensor::FlushNextMeasurement()
{
  return m_measurement_scheduler && m_measurement_scheduler->FlushNext(m_id);
}
