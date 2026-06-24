// Copyright 2024 Jacob Hartzer
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

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "infrastructure/debug_logger.hpp"
#include "sensors/camera_message.hpp"
#include "sensors/imu_message.hpp"
#include "sensors/sensor_message.hpp"
#include "sensors/sensor.hpp"
#include "sensors/sim/sim_imu_message.hpp"


class TestSensor : public Sensor
{
public:
  explicit TestSensor(
    const Sensor::Parameters & params,
    std::vector<unsigned int> * execution_log = nullptr)
  : Sensor(params), m_shared_execution_log(execution_log) {}

  void Record(const SensorMessage & sensor_message)
  {
    BufferMessage(
      sensor_message,
      [this](const SensorMessage & buffered_message) {
        m_execution_order.push_back(buffered_message.sensor_id);
        if (m_shared_execution_log != nullptr) {
          m_shared_execution_log->push_back(buffered_message.sensor_id);
        }
        m_used_times.push_back(buffered_message.time_used);
        m_measured_times.push_back(buffered_message.time_measured);
      });
  }

  void Record(const SimImuMessage & sensor_message)
  {
    BufferMessage(
      sensor_message,
      [this](const SimImuMessage & buffered_message) {
        m_execution_order.push_back(buffered_message.sensor_id);
        if (m_shared_execution_log != nullptr) {
          m_shared_execution_log->push_back(buffered_message.sensor_id);
        }
        m_used_times.push_back(buffered_message.time_used);
        m_alignment_errors.push_back(buffered_message.time_used - buffered_message.time_true);
      });
  }

  void FlushAll()
  {
    GetMeasurementScheduler()->FlushAll();
  }

  std::vector<double> m_used_times;
  std::vector<double> m_measured_times;
  std::vector<double> m_alignment_errors;
  std::vector<unsigned int> m_execution_order;

private:
  std::vector<unsigned int> * m_shared_execution_log {nullptr};
};


TEST(test_sensor, Constructor) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  Sensor sensor(sensor_params);
}

TEST(test_sensor, ParametersDefaults) {
  Sensor::Parameters sensor_params;
  EXPECT_TRUE(sensor_params.filter_sensor_time);
  EXPECT_DOUBLE_EQ(sensor_params.measurement_time_reorder_window, 1.0);
}

TEST(test_sensor, MessageCompare) {
  cv::Mat cam_img = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
  auto camera_message = std::make_shared<CameraMessage>(cam_img);
  camera_message->time_measured = 10.0;
  camera_message->time_received = 0.0;

  auto imu_message = std::make_shared<ImuMessage>();
  imu_message->time_measured = 0.0;
  imu_message->time_received = 0.1;

  EXPECT_TRUE(MessageCompare(camera_message, imu_message));
  EXPECT_FALSE(MessageCompare(imu_message, camera_message));
}

TEST(test_sensor, Callback) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  Sensor sensor(sensor_params);

  SensorMessage sensor_message;
  sensor_message.time_measured = 1.0;
  sensor_message.time_received = 1.0;
  sensor.Callback(sensor_message);
}

TEST(test_sensor, GetName) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  sensor_params.name = "test_sensor_name";
  Sensor sensor(sensor_params);

  EXPECT_EQ(sensor.GetName(), "test_sensor_name");
}

TEST(test_sensor, FilterDisabled_UsesMeasuredTime) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  sensor_params.filter_sensor_time = false;
  TestSensor sensor(sensor_params);

  SensorMessage sensor_message;
  sensor_message.time_measured = 1.25;
  sensor_message.time_received = 9.75;
  sensor.Record(sensor_message);

  ASSERT_EQ(sensor.m_used_times.size(), 1U);
  ASSERT_EQ(sensor.m_measured_times.size(), 1U);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[0], 1.25);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[0], sensor.m_measured_times[0]);
}

TEST(test_sensor, FilterEnabled_UsesRunningMinimumOffset) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  sensor_params.filter_sensor_time = true;
  sensor_params.measurement_time_reorder_window = 0.0;
  TestSensor sensor(sensor_params);

  SensorMessage message_1;
  message_1.time_measured = 1.0;
  message_1.time_received = 2.0;
  sensor.Record(message_1);

  SensorMessage message_2;
  message_2.time_measured = 2.0;
  message_2.time_received = 2.3;
  sensor.Record(message_2);

  SensorMessage message_3;
  message_3.time_measured = 3.0;
  message_3.time_received = 3.1;
  sensor.Record(message_3);

  ASSERT_EQ(sensor.m_used_times.size(), 3U);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[0], 2.0);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[1], 2.3);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[2], 3.1);
}

TEST(test_sensor, FilterEnabled_MinimumOffsetIsMonotoneNonIncreasing) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  sensor_params.filter_sensor_time = true;
  sensor_params.measurement_time_reorder_window = 0.0;
  TestSensor sensor(sensor_params);

  SensorMessage message_1;
  message_1.time_measured = 1.0;
  message_1.time_received = 1.8;
  sensor.Record(message_1);

  SensorMessage message_2;
  message_2.time_measured = 2.0;
  message_2.time_received = 2.6;
  sensor.Record(message_2);

  SensorMessage message_3;
  message_3.time_measured = 3.0;
  message_3.time_received = 3.4;
  sensor.Record(message_3);

  ASSERT_EQ(sensor.m_used_times.size(), 3U);
  const std::vector<double> offsets {
    sensor.m_used_times[0] - 1.0,
    sensor.m_used_times[1] - 2.0,
    sensor.m_used_times[2] - 3.0};
  EXPECT_GE(offsets[0], offsets[1]);
  EXPECT_GE(offsets[1], offsets[2]);
  EXPECT_DOUBLE_EQ(offsets[0], 0.8);
  EXPECT_DOUBLE_EQ(offsets[1], 0.6);
  EXPECT_DOUBLE_EQ(offsets[2], 0.4);
}

TEST(test_sensor, SimAlignmentErrorConvergesTowardZeroWithSmallerDelays) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  sensor_params.filter_sensor_time = true;
  sensor_params.measurement_time_reorder_window = 0.0;
  TestSensor sensor(sensor_params);

  SimImuMessage message_1;
  message_1.time_true = 1.0;
  message_1.time_measured = 1.25;
  message_1.time_received = 1.8;
  sensor.Record(message_1);

  SimImuMessage message_2;
  message_2.time_true = 2.0;
  message_2.time_measured = 2.25;
  message_2.time_received = 2.4;
  sensor.Record(message_2);

  SimImuMessage message_3;
  message_3.time_true = 3.0;
  message_3.time_measured = 3.25;
  message_3.time_received = 3.1;
  sensor.Record(message_3);

  ASSERT_EQ(sensor.m_alignment_errors.size(), 3U);
  EXPECT_DOUBLE_EQ(sensor.m_alignment_errors[0], 0.8);
  EXPECT_DOUBLE_EQ(sensor.m_alignment_errors[1], 0.4);
  EXPECT_NEAR(sensor.m_alignment_errors[2], 0.1, 1e-12);
  EXPECT_GT(std::abs(sensor.m_alignment_errors[0]), std::abs(sensor.m_alignment_errors[1]));
  EXPECT_GT(std::abs(sensor.m_alignment_errors[1]), std::abs(sensor.m_alignment_errors[2]));
}

TEST(test_sensor, FlushNextMeasurementProcessesBufferedMessagesInTimeUsedOrder) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  sensor_params.filter_sensor_time = true;
  sensor_params.measurement_time_reorder_window = 10.0;
  TestSensor sensor(sensor_params);

  SensorMessage message_1;
  message_1.time_measured = 1.0;
  message_1.time_received = 1.2;
  sensor.Record(message_1);

  SensorMessage message_2;
  message_2.time_measured = 2.0;
  message_2.time_received = 2.5;
  sensor.Record(message_2);

  ASSERT_TRUE(sensor.HasBufferedMeasurements());
  EXPECT_DOUBLE_EQ(sensor.GetNextBufferedMeasurementTime(), 1.2);
  EXPECT_TRUE(sensor.FlushNextMeasurement());
  ASSERT_EQ(sensor.m_used_times.size(), 1U);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[0], 1.2);
  EXPECT_TRUE(sensor.HasBufferedMeasurements());
  EXPECT_DOUBLE_EQ(sensor.GetNextBufferedMeasurementTime(), 2.2);
  EXPECT_TRUE(sensor.FlushNextMeasurement());
  ASSERT_EQ(sensor.m_used_times.size(), 2U);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[1], 2.2);
  EXPECT_FALSE(sensor.HasBufferedMeasurements());
  EXPECT_FALSE(sensor.FlushNextMeasurement());
}

TEST(test_sensor, FilterEnabled_PreservesPerSensorCausalityWhenDelayShrinks) {
  Sensor::Parameters sensor_params;
  sensor_params.logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  sensor_params.filter_sensor_time = true;
  sensor_params.measurement_time_reorder_window = 0.0;
  TestSensor sensor(sensor_params);

  SensorMessage message_1;
  message_1.time_measured = 1.0;
  message_1.time_received = 5.0;
  sensor.Record(message_1);

  SensorMessage message_2;
  message_2.time_measured = 1.5;
  message_2.time_received = 5.01;
  sensor.Record(message_2);

  ASSERT_EQ(sensor.m_used_times.size(), 2U);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[0], 5.0);
  EXPECT_DOUBLE_EQ(sensor.m_used_times[1], 5.01);
  EXPECT_GE(sensor.m_used_times[1], sensor.m_used_times[0]);
}

TEST(test_sensor, SharedScheduler_ReordersAcrossSensors) {
  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto scheduler = std::make_shared<Sensor::MeasurementScheduler>(10.0);
  std::vector<unsigned int> execution_log;

  Sensor::Parameters sensor_1_params;
  sensor_1_params.logger = logger;
  sensor_1_params.filter_sensor_time = true;
  sensor_1_params.measurement_time_reorder_window = 10.0;
  sensor_1_params.measurement_scheduler = scheduler;
  TestSensor sensor_1(sensor_1_params, &execution_log);

  Sensor::Parameters sensor_2_params = sensor_1_params;
  TestSensor sensor_2(sensor_2_params, &execution_log);

  SensorMessage sensor_2_init_message;
  sensor_2_init_message.sensor_id = sensor_2.GetId();
  sensor_2_init_message.time_measured = 0.0;
  sensor_2_init_message.time_received = 0.1;
  sensor_2.Record(sensor_2_init_message);

  SensorMessage slow_message;
  slow_message.sensor_id = sensor_1.GetId();
  slow_message.time_measured = 1.0;
  slow_message.time_received = 5.0;
  sensor_1.Record(slow_message);

  SensorMessage fast_message;
  fast_message.sensor_id = sensor_2.GetId();
  fast_message.time_measured = 4.0;
  fast_message.time_received = 5.1;
  sensor_2.Record(fast_message);

  scheduler->FlushAll();

  ASSERT_EQ(execution_log.size(), 3U);
  EXPECT_EQ(execution_log[0], sensor_2.GetId());
  EXPECT_EQ(execution_log[1], sensor_2.GetId());
  EXPECT_EQ(execution_log[2], sensor_1.GetId());
  ASSERT_EQ(sensor_1.m_used_times.size(), 1U);
  ASSERT_EQ(sensor_2.m_used_times.size(), 2U);
  EXPECT_DOUBLE_EQ(sensor_1.m_used_times[0], 5.0);
  EXPECT_DOUBLE_EQ(sensor_2.m_used_times[0], 0.1);
  EXPECT_DOUBLE_EQ(sensor_2.m_used_times[1], 4.1);
}
