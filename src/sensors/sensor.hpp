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

#ifndef SENSORS__SENSOR_HPP_
#define SENSORS__SENSOR_HPP_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ekf/ekf.hpp"
#include "infrastructure/data_logger.hpp"
#include "infrastructure/debug_logger.hpp"
#include "sensors/sensor_message.hpp"

///
/// @class Sensor
/// @brief Pure base sensor class
/// @todo Function for checking callback rate
///
class Sensor
{
public:
  class MeasurementScheduler
  {
public:
    explicit MeasurementScheduler(double measurement_time_reorder_window);

    unsigned int Enqueue(
      unsigned int sensor_id,
      double time_used,
      double time_received,
      std::function<void()> execute_fn);
    unsigned int FlushAll();
    unsigned int FlushSensor(unsigned int sensor_id);
    bool HasPending(unsigned int sensor_id) const;
    double GetNextPendingTime(unsigned int sensor_id) const;
    bool FlushNext(unsigned int sensor_id);

private:
    struct PendingMeasurement
    {
      std::uint64_t sequence_id {0};
      unsigned int sensor_id {0};
      double time_used {0.0};
      double time_received {0.0};
      std::function<void()> execute_fn;
    };

    void SortPending();
    unsigned int ReleaseReadyMeasurements(bool flush_all);

    double m_measurement_time_reorder_window {0.0};
    double m_latest_time_received {-std::numeric_limits<double>::infinity()};
    std::uint64_t m_next_sequence_id {0};
    std::vector<PendingMeasurement> m_pending_measurements;
  };

  ///
  /// @brief Sensor parameter structure
  ///
  typedef struct Parameters
  {
    std::string name {""};                ///< @brief Name
    std::string topic {""};               ///< @brief Topic
    double rate{1.0};                     ///< @brief Update rate
    double data_log_rate {0.0};           ///< @brief Data logging rate
    std::string log_directory {""};       ///< @brief Data logging directory
    bool filter_sensor_time {true};       ///< @brief Enable min-delay timestamp filtering
    double measurement_time_reorder_window {1.0};  ///< @brief Reorder window in seconds
    std::shared_ptr<DebugLogger> logger;  ///< @brief Debug logger
    std::shared_ptr<EKF> ekf;             ///< @brief EKF to update
    std::shared_ptr<MeasurementScheduler> measurement_scheduler;  ///< @brief Shared filter queue
  } Parameters;

  ///
  /// @brief Sensor class constructor
  /// @param params Sensor parameters
  ///
  explicit Sensor(Parameters params);

  virtual ~Sensor() = default;

  ///
  /// @brief Sensor ID getter method
  /// @return Sensor ID
  ///
  unsigned int GetId() const;

  ///
  /// @brief Sensor name getter method
  /// @return Sensor name
  ///
  std::string GetName() const;

  ///
  /// @brief Sensor callback function
  /// @param sensor_message callback message
  ///
  void Callback(const SensorMessage sensor_message) const;

  ///
  /// @brief Flush buffered measurements
  ///
  virtual void Flush();

  ///
  /// @brief Check whether buffered measurements remain
  /// @return True when buffered measurements remain
  ///
  virtual bool HasBufferedMeasurements() const;

  ///
  /// @brief Get the next buffered measurement time used
  /// @return Time used for the next buffered measurement
  ///
  virtual double GetNextBufferedMeasurementTime() const;

  ///
  /// @brief Flush the next buffered measurement only
  /// @return True when a buffered measurement was processed
  ///
  virtual bool FlushNextMeasurement();

  std::shared_ptr<MeasurementScheduler> GetMeasurementScheduler() const;

protected:
  void InitializeTimingLogger(
    const std::string & log_prefix,
    const std::string & log_directory,
    double data_log_rate);
  void LogTiming(const SensorMessage & sensor_message);

  template<typename MessageT, typename ExecuteFn>
  unsigned int BufferMessage(
    const MessageT & sensor_message,
    ExecuteFn execute_fn)
  {
    if (!m_filter_sensor_time) {
      MessageT immediate_message = sensor_message;
      immediate_message.time_used = immediate_message.time_measured;
      RecordMeasurementExecution(immediate_message.time_used);
      execute_fn(immediate_message);
      return 1;
    }

    MessageT buffered_message = sensor_message;
    buffered_message.time_used = GetTimeUsed(sensor_message);
    return m_measurement_scheduler->Enqueue(
      m_id,
      buffered_message.time_used,
      buffered_message.time_received,
      [this, execute_fn, buffered_message]() {
        RecordMeasurementExecution(buffered_message.time_used);
        execute_fn(buffered_message);
      });
  }

  std::uint64_t GetExecutionCount() const;
  void RecordMeasurementExecution(double time_used);

  double m_rate;                          ///< @brief Sensor measurement rate
  unsigned int m_id;                      ///< @brief Sensor id
  std::string m_name;                     ///< @brief Sensor name
  std::shared_ptr<DebugLogger> m_logger;  ///< @brief Debug logger
  bool m_is_initialized{false};           ///< @brief Sensor initialization flag
  bool m_filter_sensor_time {false};      ///< @brief Sensor time filter enable
  double m_measurement_time_reorder_window {1.0};  ///< @brief Reordering window
  std::shared_ptr<MeasurementScheduler> m_measurement_scheduler;  ///< @brief Shared queue
  DataLogger m_timing_logger;             ///< @brief Timing observability logger

private:
  double GetTimeUsed(const SensorMessage & sensor_message);

  static unsigned int m_sensor_count;     ///< @brief Static sensor count
  bool m_min_offset_initialized {false};  ///< @brief Min offset initialized flag
  double m_min_offset {0.0};              ///< @brief Running minimum time offset
  bool m_last_assigned_time_used_initialized {false};  ///< @brief Sensor time clamp initialized
  double m_last_assigned_time_used {0.0};  ///< @brief Last adjusted sensor timestamp
  std::uint64_t m_execution_count {0};    ///< @brief Executed measurement count
};

bool MessageCompare(std::shared_ptr<SensorMessage> l_msg, std::shared_ptr<SensorMessage> r_msg);

#endif  // SENSORS__SENSOR_HPP_
