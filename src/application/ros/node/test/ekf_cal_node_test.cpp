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

#include <gtest/gtest.h>
#include <stddef.h>

#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/header.hpp>

#include "application/ros/node/ekf_cal_node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensors/ros/ros_gps_message.hpp"
#include "sensors/ros/ros_imu_message.hpp"

///
/// @class EkfCalNode_test
/// @brief Testing class for EkfCalNode
///
class EkfCalNode_test : public ::testing::Test
{
protected:
  virtual void SetUp() {rclcpp::init(0, nullptr);}  ///< @brief EKF CAL Test node set up method
  virtual void TearDown() {rclcpp::shutdown();}     ///< @brief EKF CAL Test node tear down method
};

class TestEkfCalNode : public EkfCalNode
{
public:
  void SetCurrentRosTime(double current_ros_time) {m_current_ros_time = current_ros_time;}

  double GetCurrentRosTime() const override {return m_current_ros_time;}

  void OnImuMessageStamped(const RosImuMessage & ros_imu_message) const override
  {
    m_last_imu_time_measured = ros_imu_message.time_measured;
    m_last_imu_time_received = ros_imu_message.time_received;
  }

  void OnGpsMessageStamped(const RosGpsMessage & ros_gps_message) const override
  {
    m_last_gps_time_measured = ros_gps_message.time_measured;
    m_last_gps_time_received = ros_gps_message.time_received;
  }

  double GetLastImuTimeMeasured() const {return m_last_imu_time_measured;}
  double GetLastImuTimeReceived() const {return m_last_imu_time_received;}
  double GetLastGpsTimeMeasured() const {return m_last_gps_time_measured;}
  double GetLastGpsTimeReceived() const {return m_last_gps_time_received;}

private:
  double m_current_ros_time {0.0};
  mutable double m_last_imu_time_measured {0.0};
  mutable double m_last_imu_time_received {0.0};
  mutable double m_last_gps_time_measured {0.0};
  mutable double m_last_gps_time_received {0.0};
};

TEST_F(EkfCalNode_test, hello_world)
{
  TestEkfCalNode node;

  node.set_parameter(rclcpp::Parameter("debug_log_level", 1));
  node.set_parameter(rclcpp::Parameter("measurement_time_reorder_window", 0.5));
  node.set_parameter(rclcpp::Parameter("imu_list", std::vector<std::string> {"imu_1"}));
  node.set_parameter(rclcpp::Parameter("camera_list", std::vector<std::string> {"cam_2"}));
  node.set_parameter(rclcpp::Parameter("tracker_list", std::vector<std::string> {"tracker_3"}));
  node.set_parameter(rclcpp::Parameter("gps_list", std::vector<std::string> {"gps_4"}));
  node.set_parameter(rclcpp::Parameter("fiducial_list", std::vector<std::string> {"fiducial_5"}));

  node.Initialize();
  node.DeclareSensors();

  node.set_parameter(rclcpp::Parameter("imu.imu_1.is_extrinsic", false));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.is_intrinsic", false));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.rate", 400.0));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.topic", "/ImuTopic"));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.filter_sensor_time", true));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.pos", 0.1));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.ang", 0.1));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.acc_bias", 1e-3));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.gyr_bias", 1e-3));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.pos_i_in_b", std::vector<double> {0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.ang_i_to_b", std::vector<double> {1.0, 0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.acc_bias", std::vector<double> {0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.omg_bias", std::vector<double> {0.0, 0.0, 0.0}));

  node.set_parameter(rclcpp::Parameter("camera.cam_2.rate", 5.0));
  node.set_parameter(rclcpp::Parameter("camera.cam_2.topic", "/CameraTopic"));
  node.set_parameter(rclcpp::Parameter("camera.cam_2.filter_sensor_time", false));
  node.set_parameter(
    rclcpp::Parameter("camera.cam_2.pos_c_in_b", std::vector<double> {0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("camera.cam_2.ang_c_to_b", std::vector<double> {1.0, 0.0, 0.0, 0.0}));
  node.set_parameter(rclcpp::Parameter("camera.cam_2.variance.pos", 0.1));
  node.set_parameter(rclcpp::Parameter("camera.cam_2.variance.ang", 0.1));
  node.set_parameter(rclcpp::Parameter("camera.cam_2.tracker", "tracker_3"));
  node.set_parameter(
    rclcpp::Parameter("camera.cam_2.fiducials", std::vector<std::string> {"fiducial_5"}));

  node.set_parameter(rclcpp::Parameter("tracker.tracker_3.feature_detector", 4));
  node.set_parameter(rclcpp::Parameter("tracker.tracker_3.descriptor_extractor", 0));
  node.set_parameter(rclcpp::Parameter("tracker.tracker_3.descriptor_matcher", 0));
  node.set_parameter(rclcpp::Parameter("tracker.tracker_3.detector_threshold", 10));

  node.set_parameter(rclcpp::Parameter("gps.gps_4.topic", "/gps1"));
  node.set_parameter(rclcpp::Parameter("gps.gps_4.rate", 10.0));
  node.set_parameter(rclcpp::Parameter("gps.gps_4.filter_sensor_time", false));
  node.set_parameter(rclcpp::Parameter("gps.gps_4.variance.pos", 0.1));
  node.set_parameter(
    rclcpp::Parameter("gps.gps_4.pos_a_in_b", std::vector<double> {0.0, 0.0, 0.0}));

  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.fiducial_type", 1));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.squares_x", 5));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.squares_y", 7));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.square_length", 0.04));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.marker_length", 0.02));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.min_track_length", 0));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.max_track_length", 1));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.is_extrinsic", false));
  node.set_parameter(
    rclcpp::Parameter("fiducial.fiducial_5.pos_f_in_l", std::vector<double> {0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("fiducial.fiducial_5.ang_f_to_l", std::vector<double> {1.0, 0.0, 0.0, 0.0}));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.variance.pos", 0.1));
  node.set_parameter(rclcpp::Parameter("fiducial.fiducial_5.variance.pos", 0.1));

  node.LoadSensors();
  const auto imu_ids = node.GetImuIds();
  const auto camera_ids = node.GetCameraIds();
  const auto gps_ids = node.GetGpsIds();
  ASSERT_EQ(imu_ids.size(), 1U);
  ASSERT_EQ(camera_ids.size(), 1U);
  ASSERT_EQ(gps_ids.size(), 1U);

  auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
  imu_msg->header.stamp.sec = 0;
  imu_msg->header.stamp.nanosec = 0;
  imu_msg->linear_acceleration.x = 0.0;
  imu_msg->linear_acceleration.y = 0.0;
  imu_msg->linear_acceleration.z = 0.0;
  imu_msg->angular_velocity.x = 0.0;
  imu_msg->angular_velocity.y = 0.0;
  imu_msg->angular_velocity.z = 0.0;
  imu_msg->linear_acceleration_covariance.fill(0);
  imu_msg->angular_velocity_covariance.fill(0);

  node.ImuCallback(imu_msg, imu_ids.front());

  imu_msg->header.stamp.nanosec = 500000000;
  node.ImuCallback(imu_msg, imu_ids.front());

  auto cam_msg = std::make_shared<sensor_msgs::msg::Image>();
  cv::Mat cv_image = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
  cam_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC1", cv_image).toImageMsg();

  node.CameraCallback(cam_msg, camera_ids.front());

  auto gps_msg = std::make_shared<sensor_msgs::msg::NavSatFix>();
  gps_msg->altitude = 0.0;
  gps_msg->latitude = 0.0;
  gps_msg->longitude = 0.0;

  node.GpsCallback(gps_msg, gps_ids.front());

  node.PublishState();

  EXPECT_TRUE(true);
}

TEST_F(EkfCalNode_test, imu_callback_uses_receipt_time)
{
  TestEkfCalNode node;

  node.set_parameter(rclcpp::Parameter("debug_log_level", 1));
  node.set_parameter(rclcpp::Parameter("measurement_time_reorder_window", 0.5));
  node.set_parameter(rclcpp::Parameter("imu_list", std::vector<std::string> {"imu_1"}));

  node.Initialize();
  node.DeclareSensors();

  node.set_parameter(rclcpp::Parameter("imu.imu_1.is_extrinsic", false));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.is_intrinsic", false));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.rate", 400.0));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.topic", "/ImuTopic"));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.filter_sensor_time", true));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.pos", 0.1));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.ang", 0.1));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.acc_bias", 1e-3));
  node.set_parameter(rclcpp::Parameter("imu.imu_1.variance.gyr_bias", 1e-3));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.pos_i_in_b", std::vector<double> {0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.ang_i_to_b", std::vector<double> {1.0, 0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.acc_bias", std::vector<double> {0.0, 0.0, 0.0}));
  node.set_parameter(
    rclcpp::Parameter("imu.imu_1.omg_bias", std::vector<double> {0.0, 0.0, 0.0}));

  node.LoadSensors();
  const auto imu_ids = node.GetImuIds();
  ASSERT_EQ(imu_ids.size(), 1U);
  const auto imu_id = imu_ids.front();

  auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
  imu_msg->header.stamp.sec = 10;
  imu_msg->header.stamp.nanosec = 0;
  imu_msg->linear_acceleration_covariance.fill(0.0);
  imu_msg->angular_velocity_covariance.fill(0.0);

  node.SetCurrentRosTime(20.0);
  node.ImuCallback(imu_msg, imu_id);
  EXPECT_DOUBLE_EQ(node.GetLastImuTimeMeasured(), 10.0);
  EXPECT_DOUBLE_EQ(node.GetLastImuTimeReceived(), 20.0);

  imu_msg->header.stamp.sec = 11;
  node.SetCurrentRosTime(21.0);
  node.ImuCallback(imu_msg, imu_id);

  EXPECT_DOUBLE_EQ(node.GetLastImuTimeMeasured(), 11.0);
  EXPECT_DOUBLE_EQ(node.GetLastImuTimeReceived(), 21.0);
}

TEST_F(EkfCalNode_test, gps_callback_uses_receipt_time)
{
  TestEkfCalNode node;

  node.set_parameter(rclcpp::Parameter("debug_log_level", 1));
  node.set_parameter(rclcpp::Parameter("measurement_time_reorder_window", 0.5));
  node.set_parameter(rclcpp::Parameter("gps_list", std::vector<std::string> {"gps_1"}));

  node.Initialize();
  node.DeclareSensors();

  node.set_parameter(rclcpp::Parameter("gps.gps_1.topic", "/GpsTopic"));
  node.set_parameter(rclcpp::Parameter("gps.gps_1.rate", 10.0));
  node.set_parameter(rclcpp::Parameter("gps.gps_1.filter_sensor_time", true));
  node.set_parameter(rclcpp::Parameter("gps.gps_1.variance.pos", 0.1));
  node.set_parameter(
    rclcpp::Parameter("gps.gps_1.pos_a_in_b", std::vector<double> {0.0, 0.0, 0.0}));

  node.LoadSensors();
  const auto gps_ids = node.GetGpsIds();
  ASSERT_EQ(gps_ids.size(), 1U);
  const auto gps_id = gps_ids.front();

  auto gps_msg = std::make_shared<sensor_msgs::msg::NavSatFix>();
  gps_msg->header.stamp.sec = 30;
  gps_msg->header.stamp.nanosec = 0;
  gps_msg->latitude = 0.0;
  gps_msg->longitude = 0.0;
  gps_msg->altitude = 0.0;

  node.SetCurrentRosTime(50.0);
  node.GpsCallback(gps_msg, gps_id);
  EXPECT_DOUBLE_EQ(node.GetLastGpsTimeMeasured(), 30.0);
  EXPECT_DOUBLE_EQ(node.GetLastGpsTimeReceived(), 50.0);

  gps_msg->header.stamp.sec = 31;
  node.SetCurrentRosTime(51.0);
  node.GpsCallback(gps_msg, gps_id);

  EXPECT_DOUBLE_EQ(node.GetLastGpsTimeMeasured(), 31.0);
  EXPECT_DOUBLE_EQ(node.GetLastGpsTimeReceived(), 51.0);
}
