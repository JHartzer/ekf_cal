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

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <H5Cpp.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "ekf/ekf.hpp"
#include "infrastructure/debug_logger.hpp"
#include "infrastructure/hdf5_log_manager.hpp"
#include "infrastructure/sim/truth_engine.hpp"
#include "sensors/sim/sim_gps.hpp"
#include "sensors/gps.hpp"
#include "infrastructure/sim/truth_engine_cyclic.hpp"
#include "utility/custom_assertions.hpp"
#include "utility/gps_helper.hpp"
#include "utility/sim/sim_rng.hpp"

namespace
{
std::string GetHdf5Filename(const std::string & directory)
{
  std::string dir = directory;
  while (!dir.empty() && dir.back() == '/') {
    dir.pop_back();
  }
  const std::size_t pos = dir.find_last_of('/');
  const std::string base = (pos == std::string::npos) ? dir : dir.substr(pos + 1);
  return base + ".h5";
}

std::vector<double> ReadHdf5Column(
  const std::string & directory,
  const std::string & dataset_path,
  std::size_t column_index)
{
  std::string h5_filename = GetHdf5Filename(directory);
  std::shared_ptr<H5::H5File> file = Hdf5LogManager::GetFile(directory, h5_filename);
  H5::DataSet dataset = file->openDataSet(dataset_path);
  H5::DataSpace file_space = dataset.getSpace();

  hsize_t dims[2] = {0, 0};
  file_space.getSimpleExtentDims(dims);
  std::vector<double> values(dims[0] * dims[1], 0.0);
  dataset.read(values.data(), H5::PredType::NATIVE_DOUBLE);

  std::vector<double> column;
  column.reserve(dims[0]);
  for (hsize_t row = 0; row < dims[0]; ++row) {
    column.push_back(values[row * dims[1] + column_index]);
  }
  return column;
}
}  // namespace

TEST(test_SimIMU, Constructor) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  GPS::Parameters gps_params;
  gps_params.name = "GPS_1";
  gps_params.topic = "GPS_1";
  gps_params.rate = 5.0;
  gps_params.pos_a_in_b = Eigen::Vector3d{0, 0, 0};
  gps_params.variance.pos = 5.0;
  gps_params.ekf = ekf;
  gps_params.logger = ekf_params.debug_logger;

  SimGPS::Parameters sim_gps_params;
  sim_gps_params.lla_error = Eigen::Vector3d{5.0, 5.0, 5.0};
  sim_gps_params.pos_e_in_g_err = Eigen::Vector3d{0.0, 0.0, 0.0};
  sim_gps_params.ang_l_to_e_err = 0.0;
  sim_gps_params.gps_params = gps_params;

  Eigen::Vector3d pos_frequency{1, 2, 3};
  Eigen::Vector3d ang_frequency{1, 2, 3};
  Eigen::Vector3d pos_offset{0, 0, 0};
  Eigen::Vector3d ang_offset{0, 0, 0};
  double pos_amplitude{1.0};
  double ang_amplitude{0.1};
  double stationary_time{1.0};
  double max_time{1.0};

  auto truth_engine_cyclic = std::make_shared<TruthEngineCyclic>(
    pos_frequency,
    ang_frequency,
    pos_offset,
    ang_offset,
    pos_amplitude,
    ang_amplitude,
    stationary_time,
    max_time,
    ekf_params.debug_logger
  );

  auto truth_engine = std::static_pointer_cast<TruthEngine>(truth_engine_cyclic);
  SimGPS sim_gps(sim_gps_params, truth_engine);
  truth_engine->SetGpsPosition(sim_gps.GetId(), Eigen::Vector3d {0, 0, 0});
  truth_engine->SetLocalPosition(Eigen::Vector3d {0, 0, 0});
  truth_engine->SetLocalHeading(0.0);

  SimRNG::SetSeed(1);
  auto gps_msgs = sim_gps.GenerateMessages();

  ASSERT_EQ(gps_msgs.size(), 5U);

  EXPECT_NEAR(gps_msgs[1]->time_measured - gps_msgs[0]->time_measured, 0.2, 1e-3);
  EXPECT_NEAR(gps_msgs[2]->time_measured - gps_msgs[1]->time_measured, 0.2, 1e-3);
  EXPECT_NEAR(gps_msgs[3]->time_measured - gps_msgs[2]->time_measured, 0.2, 1e-3);
  EXPECT_NEAR(gps_msgs[4]->time_measured - gps_msgs[3]->time_measured, 0.2, 1e-3);

  Eigen::Vector3d lla_ref = Eigen::Vector3d::Zero();

  Eigen::Vector3d enu_0 = lla_to_enu(gps_msgs[0]->gps_lla, lla_ref);
  Eigen::Vector3d enu_1 = lla_to_enu(gps_msgs[1]->gps_lla, lla_ref);
  Eigen::Vector3d enu_2 = lla_to_enu(gps_msgs[2]->gps_lla, lla_ref);
  Eigen::Vector3d enu_3 = lla_to_enu(gps_msgs[3]->gps_lla, lla_ref);
  Eigen::Vector3d enu_4 = lla_to_enu(gps_msgs[4]->gps_lla, lla_ref);

  Eigen::Matrix<double, 3, 5> actual_enu;
  actual_enu.col(0) = enu_0;
  actual_enu.col(1) = enu_1;
  actual_enu.col(2) = enu_2;
  actual_enu.col(3) = enu_3;
  actual_enu.col(4) = enu_4;

  Eigen::Matrix<double, 3, 5> expected_enu;
  expected_enu <<
    -0.740003721992806, -3.624393920094640, -4.190170337710447, -1.370910371261179,
    5.834538949661810,
    -0.311266717001466, 7.129809294825257, -0.588358196053966, -5.420695545198459,
    3.286366287520096,
    0.622802860441859, -2.069844207723619, -4.642202243644744, 0.973152515827472,
    -6.216173492137504;

  EXPECT_TRUE(EXPECT_EIGEN_NEAR(actual_enu, expected_enu, 1e-5));
}

TEST(test_SimGPS, TimingSemantics) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  GPS::Parameters gps_params;
  gps_params.name = "GPS_1";
  gps_params.topic = "GPS_1";
  gps_params.rate = 5.0;
  gps_params.pos_a_in_b = Eigen::Vector3d{0, 0, 0};
  gps_params.variance.pos = 5.0;
  gps_params.ekf = ekf;
  gps_params.logger = ekf_params.debug_logger;

  SimGPS::Parameters sim_gps_params;
  sim_gps_params.lla_error = Eigen::Vector3d::Zero();
  sim_gps_params.pos_a_in_b_err = Eigen::Vector3d::Zero();
  sim_gps_params.pos_e_in_g_err = Eigen::Vector3d::Zero();
  sim_gps_params.ang_l_to_e_err = 0.0;
  sim_gps_params.gps_params = gps_params;
  sim_gps_params.time_jitter = 0.01;
  sim_gps_params.time_bias_error = 0.125;

  Eigen::Vector3d pos_frequency{1, 2, 3};
  Eigen::Vector3d ang_frequency{1, 2, 3};
  Eigen::Vector3d pos_offset{0, 0, 0};
  Eigen::Vector3d ang_offset{0, 0, 0};
  double pos_amplitude{1.0};
  double ang_amplitude{0.1};
  double stationary_time{1.0};
  double max_time{1.0};

  auto truth_engine_cyclic = std::make_shared<TruthEngineCyclic>(
    pos_frequency,
    ang_frequency,
    pos_offset,
    ang_offset,
    pos_amplitude,
    ang_amplitude,
    stationary_time,
    max_time,
    ekf_params.debug_logger
  );

  SimRNG::SetSeed(1);
  const double expected_time_bias = SimRNG::NormRand(0.0, sim_gps_params.time_bias_error);
  SimRNG::SetSeed(1);
  auto truth_engine = std::static_pointer_cast<TruthEngine>(truth_engine_cyclic);
  SimGPS sim_gps(sim_gps_params, truth_engine);
  truth_engine->SetGpsPosition(sim_gps.GetId(), Eigen::Vector3d {0, 0, 0});
  truth_engine->SetLocalPosition(Eigen::Vector3d {0, 0, 0});
  truth_engine->SetLocalHeading(0.0);

  auto gps_msgs = sim_gps.GenerateMessages();

  ASSERT_FALSE(gps_msgs.empty());
  for (const auto & gps_msg : gps_msgs) {
    EXPECT_DOUBLE_EQ(gps_msg->time_measured - gps_msg->time_true, expected_time_bias);
    EXPECT_GE(gps_msg->time_received, gps_msg->time_true);
  }
}

TEST(test_SimGPS, CallbackPreservesTrueTimeForTimingLogs) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);

  GPS::Parameters gps_params;
  gps_params.name = "GPS_1";
  gps_params.topic = "GPS_1";
  gps_params.rate = 5.0;
  gps_params.pos_a_in_b = Eigen::Vector3d{0, 0, 0};
  gps_params.variance.pos = 5.0;
  gps_params.data_log_rate = 5.0;
  gps_params.log_directory = "/tmp/sim_gps_timing_test";
  gps_params.filter_sensor_time = true;
  gps_params.measurement_time_reorder_window = 0.0;
  gps_params.ekf = ekf;
  gps_params.logger = ekf_params.debug_logger;

  SimGPS::Parameters sim_gps_params;
  sim_gps_params.lla_error = Eigen::Vector3d::Zero();
  sim_gps_params.pos_a_in_b_err = Eigen::Vector3d::Zero();
  sim_gps_params.pos_e_in_g_err = Eigen::Vector3d::Zero();
  sim_gps_params.ang_l_to_e_err = 0.0;
  sim_gps_params.gps_params = gps_params;
  sim_gps_params.time_jitter = 0.01;
  sim_gps_params.time_bias_error = 0.125;

  Eigen::Vector3d pos_frequency{1, 2, 3};
  Eigen::Vector3d ang_frequency{1, 2, 3};
  Eigen::Vector3d pos_offset{0, 0, 0};
  Eigen::Vector3d ang_offset{0, 0, 0};
  double pos_amplitude{1.0};
  double ang_amplitude{0.1};
  double stationary_time{1.0};
  double max_time{1.0};

  auto truth_engine_cyclic = std::make_shared<TruthEngineCyclic>(
    pos_frequency,
    ang_frequency,
    pos_offset,
    ang_offset,
    pos_amplitude,
    ang_amplitude,
    stationary_time,
    max_time,
    ekf_params.debug_logger
  );

  auto truth_engine = std::static_pointer_cast<TruthEngine>(truth_engine_cyclic);
  std::system(("rm -rf \"" + gps_params.log_directory + "\"").c_str());

  SimRNG::SetSeed(1);
  SimGPS sim_gps(sim_gps_params, truth_engine);
  truth_engine->SetGpsPosition(sim_gps.GetId(), Eigen::Vector3d {0, 0, 0});
  truth_engine->SetLocalPosition(Eigen::Vector3d {0, 0, 0});
  truth_engine->SetLocalHeading(0.0);

  auto gps_msgs = sim_gps.GenerateMessages();

  ASSERT_FALSE(gps_msgs.empty());
  for (const auto & gps_msg : gps_msgs) {
    sim_gps.Callback(*gps_msg);
  }
  sim_gps.Flush();

  const std::string dataset_path = "sensors/gps_" + std::to_string(sim_gps.GetId()) + "_timing";
  std::vector<double> alignment_errors =
    ReadHdf5Column(gps_params.log_directory, dataset_path, 6);

  ASSERT_FALSE(alignment_errors.empty());
  EXPECT_FALSE(std::isnan(alignment_errors.back()));
}
