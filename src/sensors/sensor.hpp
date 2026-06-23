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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ekf/ekf.hpp"
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
    bool filter_sensor_time {false};      ///< @brief Enable min-delay timestamp filtering
    double measurement_time_reorder_window {1.0};  ///< @brief Reorder window in seconds
    std::shared_ptr<DebugLogger> logger;  ///< @brief Debug logger
    std::shared_ptr<EKF> ekf;             ///< @brief EKF to update
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

protected:
  template<typename MessageT, typename ExecuteFn>
  unsigned int BufferMessage(
    const MessageT & sensor_message,
    std::vector<MessageT> & message_buffer,
    ExecuteFn execute_fn)
  {
    if (!m_filter_sensor_time) {
      MessageT immediate_message = sensor_message;
      immediate_message.time_used = immediate_message.time_measured;
      execute_fn(immediate_message);
      return 1;
    }

    UpdateTimeFilter(sensor_message);
    message_buffer.push_back(sensor_message);
    RefreshBufferedMessageTimes(message_buffer);
    return ReleaseBufferedMessages(message_buffer, execute_fn, false);
  }

  template<typename MessageT, typename ExecuteFn>
  unsigned int FlushBufferedMessages(
    std::vector<MessageT> & message_buffer,
    ExecuteFn execute_fn)
  {
    RefreshBufferedMessageTimes(message_buffer);
    return ReleaseBufferedMessages(message_buffer, execute_fn, true);
  }

  double m_rate;                          ///< @brief Sensor measurement rate
  unsigned int m_id;                      ///< @brief Sensor id
  std::string m_name;                     ///< @brief Sensor name
  std::shared_ptr<DebugLogger> m_logger;  ///< @brief Debug logger
  bool m_is_initialized{false};           ///< @brief Sensor initialization flag
  bool m_filter_sensor_time {false};      ///< @brief Sensor time filter enable
  double m_measurement_time_reorder_window {1.0};  ///< @brief Reordering window

private:
  void UpdateTimeFilter(const SensorMessage & sensor_message);
  double GetTimeUsed(const SensorMessage & sensor_message) const;

  template<typename MessageT>
  void RefreshBufferedMessageTimes(std::vector<MessageT> & message_buffer)
  {
    if (message_buffer.empty()) {
      return;
    }

    for (auto & buffered_message : message_buffer) {
      buffered_message.time_used = GetTimeUsed(buffered_message);
    }

    std::stable_sort(
      message_buffer.begin(),
      message_buffer.end(),
      [](const MessageT & left, const MessageT & right) {
        if (left.time_used == right.time_used) {
          return left.time_received < right.time_received;
        }
        return left.time_used < right.time_used;
      });
  }

  template<typename MessageT, typename ExecuteFn>
  unsigned int ReleaseBufferedMessages(
    std::vector<MessageT> & message_buffer,
    ExecuteFn execute_fn,
    bool flush_all)
  {
    if (message_buffer.empty()) {
      return 0;
    }

    unsigned int executed_count = 0;
    const double latest_time_used = message_buffer.back().time_used;
    while (!message_buffer.empty()) {
      if (!flush_all &&
        (latest_time_used - message_buffer.front().time_used) < m_measurement_time_reorder_window)
      {
        break;
      }

      execute_fn(message_buffer.front());
      message_buffer.erase(message_buffer.begin());
      executed_count++;
    }
    return executed_count;
  }

  static unsigned int m_sensor_count;     ///< @brief Static sensor count
  bool m_min_offset_initialized {false};  ///< @brief Min offset initialized flag
  double m_min_offset {0.0};              ///< @brief Running minimum time offset
};

bool MessageCompare(std::shared_ptr<SensorMessage> l_msg, std::shared_ptr<SensorMessage> r_msg);

#endif  // SENSORS__SENSOR_HPP_
