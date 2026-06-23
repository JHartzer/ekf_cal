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

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include "infrastructure/debug_logger.hpp"
#include "infrastructure/sim/truth_engine_smoother.hpp"

// Forward declarations of helpers defined in truth_engine_smoother.cpp
std::vector<Eigen::Vector3d> SlidingWindowFilter(
  std::vector<Eigen::Vector3d> & data,
  unsigned int buffer_size);

double MaxAcceleration(std::vector<double> times, std::vector<Eigen::Vector3d> & data);

TEST(test_TruthEngineSmoother, Oscillating) {
  double stationary_time {0.0};
  double max_time {6.0};
  std::vector<double> times {0, 1, 2, 3, 4, 5, 6};
  std::vector<double> positions{
    0, 0, 0,
    1, 0, 0,
    1, 1, 1,
    1, 0, 1,
    0, 0, 1,
    0, 0, 0,
    0, 0, 0};
  std::vector<double> angles{
    0.0, 0.0, 0.0,
    0.1, 0.0, 0.0,
    0.1, 0.1, 0.1,
    0.1, 0.0, 0.1,
    0.0, 0.0, 0.1,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0};

  auto pos_errs = std::vector<double> {0.0, 0.0, 0.0};
  auto ang_errs = std::vector<double> {0.0, 0.0, 0.0};

  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");

  TruthEngineSmoother truth_engine_spline(
    times, positions, angles, pos_errs, ang_errs, stationary_time, max_time, 10.0, logger);

  for (unsigned int i = 0; i < 70; ++i) {
    double time = static_cast<double>(i) / 10.0;
    auto pos = truth_engine_spline.GetBodyPosition(time);
    std::cout << time << " " << pos.transpose() << std::endl;
  }
}

TEST(test_TruthEngineSmoother, Getters) {
  double stationary_time {1.0};
  double max_time {6.0};
  std::vector<double> times {0, 1, 2, 3, 4, 5, 6};
  std::vector<double> positions{
    0, 0, 0,
    1, 0, 0,
    1, 1, 1,
    1, 0, 1,
    0, 0, 1,
    0, 0, 0,
    0, 0, 0};
  std::vector<double> angles{
    0.0, 0.0, 0.0,
    0.1, 0.0, 0.0,
    0.1, 0.1, 0.1,
    0.1, 0.0, 0.1,
    0.0, 0.0, 0.1,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0};

  auto pos_errs = std::vector<double> {0.0, 0.0, 0.0};
  auto ang_errs = std::vector<double> {0.0, 0.0, 0.0};

  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");

  TruthEngineSmoother truth_engine_spline(
    times, positions, angles, pos_errs, ang_errs, stationary_time, max_time, 10.0, logger);

  // Time invalid: negative time
  {
    double time = -0.5;
    EXPECT_EQ(truth_engine_spline.GetBodyVelocity(time), Eigen::Vector3d::Zero());
    EXPECT_EQ(truth_engine_spline.GetBodyAcceleration(time), Eigen::Vector3d::Zero());
    EXPECT_EQ(truth_engine_spline.GetBodyAngularPosition(time).w(), 1.0);
    EXPECT_EQ(truth_engine_spline.GetBodyAngularRate(time), Eigen::Vector3d::Zero());
    EXPECT_EQ(truth_engine_spline.GetBodyAngularAcceleration(time), Eigen::Vector3d::Zero());
  }

  // Time invalid: past max_time
  {
    double time = 10.0;
    EXPECT_EQ(truth_engine_spline.GetBodyVelocity(time), Eigen::Vector3d::Zero());
    EXPECT_EQ(truth_engine_spline.GetBodyAcceleration(time), Eigen::Vector3d::Zero());
    EXPECT_EQ(truth_engine_spline.GetBodyAngularPosition(time).w(), 1.0);
    EXPECT_EQ(truth_engine_spline.GetBodyAngularRate(time), Eigen::Vector3d::Zero());
    EXPECT_EQ(truth_engine_spline.GetBodyAngularAcceleration(time), Eigen::Vector3d::Zero());
  }

  // Valid time
  {
    double time = 3.0;
    EXPECT_NO_THROW(truth_engine_spline.GetBodyVelocity(time));
    EXPECT_NO_THROW(truth_engine_spline.GetBodyAcceleration(time));
    EXPECT_NO_THROW(truth_engine_spline.GetBodyAngularPosition(time));
    EXPECT_NO_THROW(truth_engine_spline.GetBodyAngularRate(time));
    EXPECT_NO_THROW(truth_engine_spline.GetBodyAngularAcceleration(time));
  }
}

TEST(test_TruthEngineSmoother_Helpers, SlidingWindowFilter) {
  std::vector<Eigen::Vector3d> data = {
    Eigen::Vector3d(1.0, 2.0, 3.0),
    Eigen::Vector3d(4.0, 5.0, 6.0),
    Eigen::Vector3d(7.0, 8.0, 9.0)
  };

  std::vector<Eigen::Vector3d> result = SlidingWindowFilter(data, 1);
  ASSERT_EQ(result.size(), 3);
  EXPECT_NEAR(result[0].x(), 5.0 / 3.0, 1e-6);
  EXPECT_NEAR(result[1].x(), 4.0, 1e-6);
  EXPECT_NEAR(result[2].x(), 11.0 / 3.0, 1e-6);
}

TEST(test_TruthEngineSmoother_Helpers, MaxAcceleration) {
  std::vector<double> times = {0.0, 1.0, 2.0, 3.0};

  std::vector<Eigen::Vector3d> data(10, Eigen::Vector3d::Zero());
  data[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
  data[1] = Eigen::Vector3d(1.0, 1.0, 1.0);
  data[2] = Eigen::Vector3d(4.0, 4.0, 4.0);
  data[3] = Eigen::Vector3d(9.0, 9.0, 9.0);
  data.resize(3);

  double max_acc = MaxAcceleration(times, data);
  EXPECT_NEAR(max_acc, 2.0, 1e-6);
}
