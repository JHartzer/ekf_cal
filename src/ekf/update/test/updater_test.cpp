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
#include <stdlib.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "ekf/constants.hpp"
#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "ekf/update/fiducial_updater.hpp"
#include "ekf/update/gps_updater.hpp"
#include "ekf/update/imu_updater.hpp"
#include "ekf/update/updater.hpp"
#include "infrastructure/debug_logger.hpp"
#include "infrastructure/hdf5_log_manager.hpp"
#include "utility/gps_helper.hpp"

namespace
{
std::string MakeUniqueLogDirectory(const std::string & prefix)
{
  std::string dir_template = "/tmp/ekf_cal_" + prefix + "_XXXXXX";
  std::vector<char> mutable_template(dir_template.begin(), dir_template.end());
  mutable_template.push_back('\0');

  char * created_dir = mkdtemp(mutable_template.data());
  if (created_dir == nullptr) {
    return "";
  }

  return std::string(created_dir);
}

std::string GetHdf5Filename(const std::string & directory)
{
  std::string dir = directory;
  while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) {
    dir.pop_back();
  }

  size_t pos = dir.find_last_of("/\\");
  std::string name = (pos == std::string::npos) ? dir : dir.substr(pos + 1);
  if (name.empty()) {
    name = "simulation_data";
  }
  return name + ".h5";
}

std::vector<double> ReadLoggedDurations(
  const std::string & log_directory,
  const std::string & dataset_path)
{
  std::shared_ptr<H5::H5File> file =
    Hdf5LogManager::GetFile(log_directory, GetHdf5Filename(log_directory));
  if (!file) {
    return {};
  }

  file->flush(H5F_SCOPE_GLOBAL);
  H5::DataSet dataset = file->openDataSet(dataset_path);
  H5::DataSpace dataspace = dataset.getSpace();

  hsize_t dims[2] = {0, 0};
  dataspace.getSimpleExtentDims(dims);
  if (dims[0] == 0 || dims[1] == 0) {
    return {};
  }

  std::vector<double> data(static_cast<size_t>(dims[0] * dims[1]), 0.0);
  dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);

  std::vector<double> durations(static_cast<size_t>(dims[0]), 0.0);
  for (size_t row = 0; row < static_cast<size_t>(dims[0]); ++row) {
    durations[row] = data[row * static_cast<size_t>(dims[1]) + static_cast<size_t>(dims[1] - 1)];
  }

  return durations;
}

void ExpectDurationsBelow(const std::vector<double> & durations, double max_duration_us)
{
  ASSERT_FALSE(durations.empty());
  for (double duration_us : durations) {
    EXPECT_LT(duration_us, max_duration_us);
  }
}
}  // namespace

TEST(test_updater, constructor) {
  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  Updater updater(0, logger);
}

TEST(test_updater, root_covariance) {
  auto logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = logger;
  ekf_params.use_root_covariance = true;
  EKF ekf(ekf_params);

  BodyState body_state;
  ekf.Initialize(0.0, body_state);

  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(1, 18);
  jacobian(0, 0) = 1.0;
  Eigen::VectorXd residual = Eigen::VectorXd::Zero(1);
  residual[0] = 0.5;
  Eigen::MatrixXd measurement_noise = Eigen::MatrixXd::Identity(1, 1);
  measurement_noise(0, 0) = 0.04;

  Updater updater(0, logger);
  updater.KalmanUpdate(ekf, jacobian, residual, measurement_noise);

  EXPECT_NEAR(ekf.m_state.body_state.pos_b_in_l[0], 0.0, 1.0);
}

TEST(test_updater, imu_update_duration_stays_below_10khz_budget) {
  std::string log_directory = MakeUniqueLogDirectory("imu_update_timing");
  ASSERT_FALSE(log_directory.empty());

  auto logger = std::make_shared<DebugLogger>(LogLevel::WARN, "");
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = logger;
  EKF ekf(ekf_params);

  BodyState body_state;
  body_state.vel_b_in_l = Eigen::Vector3d::Ones();
  ekf.Initialize(0.0, body_state);
  ekf.InitializeGravity();

  ImuState imu_state;
  imu_state.SetIsExtrinsic(false);
  imu_state.SetIsIntrinsic(false);
  ekf.RegisterIMU(0, imu_state, Eigen::MatrixXd::Zero(0, 0));

  ImuUpdater imu_updater(0, false, false, log_directory, 1.0e6, logger);

  const Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 1e-3;
  for (int i = 1; i <= 3; ++i) {
    imu_updater.UpdateEKF(
      ekf,
      0.001 * static_cast<double>(i),
      2.0 * g_gravity,
      covariance,
      Eigen::Vector3d::Ones(),
      covariance);
  }

  ExpectDurationsBelow(ReadLoggedDurations(log_directory, "sensors/imu_0"), 100.0);
}

TEST(test_updater, gps_update_duration_stays_below_100hz_budget) {
  std::string log_directory = MakeUniqueLogDirectory("gps_update_timing");
  ASSERT_FALSE(log_directory.empty());

  auto logger = std::make_shared<DebugLogger>(LogLevel::WARN, "");
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = logger;
  EKF ekf(ekf_params);

  BodyState body_state;
  ekf.Initialize(0.0, body_state);

  GpsState gps_state;
  gps_state.SetIsExtrinsic(false);
  ekf.RegisterGPS(0, gps_state, Eigen::Matrix3d::Zero());
  ekf.SetGpsReference(Eigen::Vector3d::Zero(), 0.0);

  GpsUpdater gps_updater(0, false, log_directory, 1.0e6, logger);
  const Eigen::Matrix3d pos_covariance = Eigen::Matrix3d::Identity() * 1e-9;
  const Eigen::Vector3d ref_lla = Eigen::Vector3d::Zero();

  for (int i = 1; i <= 3; ++i) {
    Eigen::Vector3d antenna_enu = Eigen::Vector3d::Constant(static_cast<double>(i));
    gps_updater.UpdateEKF(
      ekf,
      static_cast<double>(i),
      enu_to_lla(antenna_enu, ref_lla),
      pos_covariance);
  }

  ExpectDurationsBelow(ReadLoggedDurations(log_directory, "sensors/gps_0"), 10000.0);
}

TEST(test_updater, camera_update_duration_stays_below_1000hz_budget) {
  std::string log_directory = MakeUniqueLogDirectory("fiducial_update_timing");
  ASSERT_FALSE(log_directory.empty());

  auto logger = std::make_shared<DebugLogger>(LogLevel::WARN, "");
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = logger;
  EKF ekf(ekf_params);

  BodyState body_state;
  ekf.Initialize(0.0, body_state);

  CamState cam_state;
  cam_state.SetIsExtrinsic(false);
  ekf.RegisterCamera(0, cam_state, Eigen::MatrixXd::Zero(0, 0));

  FidState fid_state;
  fid_state.id = 0;
  fid_state.SetIsExtrinsic(false);
  ekf.RegisterFiducial(fid_state, Eigen::MatrixXd::Zero(0, 0));

  FiducialUpdater fiducial_updater(0, 0, false, false, log_directory, 1.0e6, logger);

  BoardDetection board_detection;
  board_detection.frame_id = 0;
  board_detection.pos_f_in_c = Eigen::Vector3d{5.0, 0.0, 0.0};
  board_detection.ang_f_to_c = Eigen::Quaterniond{1.0, 0.0, 0.0, 0.0};
  board_detection.pos_error = Eigen::Vector3d::Constant(0.1);
  board_detection.ang_error = Eigen::Vector3d::Constant(0.1);

  fiducial_updater.UpdateEKF(ekf, 0.1, board_detection);

  ExpectDurationsBelow(ReadLoggedDurations(log_directory, "sensors/fiducial_0_0"), 1000.0);
}
