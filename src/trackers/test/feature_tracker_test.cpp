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

#include <memory>
#include <string>
#include <vector>

#include "ekf/ekf.hpp"
#include "ekf/types.hpp"
#include "infrastructure/debug_logger.hpp"
#include "sensors/camera_message.hpp"
#include "trackers/feature_tracker.hpp"
#include "sensors/imu.hpp"
#include "sensors/camera.hpp"

struct FeatureTrackerTestAccess
{
  static int GenerateFeatureID()
  {
    return FeatureTracker::GenerateFeatureID();
  }

  static unsigned int PrevFrameId(const FeatureTracker & tracker)
  {
    return tracker.m_prev_frame_id;
  }

  static std::vector<cv::KeyPoint> & PrevKeyPoints(FeatureTracker & tracker)
  {
    return tracker.m_prev_key_points;
  }
};

TEST(test_feature_tracker, initialization) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");

  FeatureTracker::Parameters params;
  params.ekf = std::make_shared<EKF>(ekf_params);
  params.camera_id = 1;

  params.detector = Detector::BRISK;
  params.descriptor = Descriptor::ORB;
  params.matcher = Matcher::BRUTE_FORCE;
  FeatureTracker feature_tracker_1 {params};

  params.detector = Detector::FAST;
  FeatureTracker feature_tracker_2 {params};

  params.detector = Detector::GFTT;
  FeatureTracker feature_tracker_3 {params};

  params.detector = Detector::MSER;
  FeatureTracker feature_tracker_4 {params};

  params.detector = Detector::ORB;
  FeatureTracker feature_tracker_5 {params};

  params.detector = Detector::SIFT;
  FeatureTracker feature_tracker_6 {params};

  params.descriptor = Descriptor::SIFT;
  FeatureTracker feature_tracker_7 {params};

  params.matcher = Matcher::FLANN;
  FeatureTracker feature_tracker_8 {params};

  EXPECT_EQ(feature_tracker_1.GetID(), 1);
}

TEST(test_feature_tracker, track) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  BodyState body_state_init;
  body_state_init.vel_b_in_l[1] = 0.1;
  ekf->Initialize(0.0, body_state_init);

  IMU::Parameters imu_params;
  imu_params.ekf = ekf;
  imu_params.logger = ekf_params.debug_logger;
  IMU imu(imu_params);

  Camera::Parameters cam_params;
  cam_params.ang_c_to_b = Eigen::Quaterniond{0.5, -0.5, 0.5, -0.5};
  cam_params.intrinsics.height = 555;
  cam_params.intrinsics.width = 641;
  cam_params.intrinsics.f_x = 512.8;
  cam_params.intrinsics.f_y = 512.8;
  cam_params.ekf = ekf;
  cam_params.logger = ekf_params.debug_logger;
  Camera cam(cam_params);

  FeatureTracker::Parameters tracker_params;
  tracker_params.px_error = 0.1;
  tracker_params.max_track_length = 2;
  tracker_params.camera_id = cam.GetId();
  tracker_params.ekf = ekf;
  tracker_params.min_feat_dist = 0.1;
  tracker_params.logger = ekf_params.debug_logger;
  tracker_params.detector = Detector::FAST;

  auto feature_tracker = std::make_shared<FeatureTracker>(tracker_params);
  cam.AddTracker(feature_tracker);

  std::string source_file = __FILE__;
  std::string source_dir = source_file.substr(0, source_file.find_last_of("/\\"));
  cv::Mat img_1 =
    cv::imread(source_dir + "/images/tsukuba_l.png", cv::IMREAD_GRAYSCALE);
  cv::Mat img_2 =
    cv::imread(source_dir + "/images/tsukuba_r.png", cv::IMREAD_GRAYSCALE);

  ASSERT_FALSE(img_1.empty());
  ASSERT_FALSE(img_2.empty());

  CameraMessage cam_msg_1(img_1);
  CameraMessage cam_msg_2(img_2);

  cam_msg_1.time_measured = 0.0;
  cam_msg_1.time_received = 0.0;
  cam_msg_2.time_measured = 1.0;
  cam_msg_2.time_received = 1.0;

  cam.Callback(cam_msg_1);
  cam.Callback(cam_msg_2);

  cv::imwrite(source_dir + "/images/feature_track.png", cam.m_out_img);

  EXPECT_NEAR(ekf->m_state.body_state.pos_b_in_l[0], 0.0, 1e-1);
  EXPECT_NEAR(ekf->m_state.body_state.pos_b_in_l[1], 0.1, 1e-1);
  EXPECT_NEAR(ekf->m_state.body_state.pos_b_in_l[2], 0.0, 1e-1);
}

TEST(test_feature_tracker, GenerateFeatureID) {
  int id1 = FeatureTrackerTestAccess::GenerateFeatureID();
  int id2 = FeatureTrackerTestAccess::GenerateFeatureID();
  int id3 = FeatureTrackerTestAccess::GenerateFeatureID();
  EXPECT_EQ(id2, id1 + 1);
  EXPECT_EQ(id3, id2 + 1);
}

TEST(test_feature_tracker, GridFeatures) {
  std::vector<cv::KeyPoint> key_points;
  // kp1: in bounds, first in grid cell (0,0)
  key_points.emplace_back(cv::Point2f(5.0f, 5.0f), 1.0f);
  // kp2: in bounds, second in grid cell (0,0) -> should be discarded
  key_points.emplace_back(cv::Point2f(6.0f, 6.0f), 1.0f);
  // kp3: in bounds, first in grid cell (1,0) -> should be kept
  key_points.emplace_back(cv::Point2f(15.0f, 5.0f), 1.0f);
  // kp4: out of bounds (negative x) -> should be discarded
  key_points.emplace_back(cv::Point2f(-5.0f, 5.0f), 1.0f);
  // kp5: out of bounds (x >= cols) -> should be discarded
  key_points.emplace_back(cv::Point2f(55.0f, 5.0f), 1.0f);

  FeatureTracker::GridFeatures(key_points, 50, 50);

  ASSERT_EQ(key_points.size(), 2);
  EXPECT_FLOAT_EQ(key_points[0].pt.x, 5.0f);
  EXPECT_FLOAT_EQ(key_points[0].pt.y, 5.0f);
  EXPECT_FLOAT_EQ(key_points[1].pt.x, 15.0f);
  EXPECT_FLOAT_EQ(key_points[1].pt.y, 5.0f);
}

TEST(test_feature_tracker, RatioTest) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  FeatureTracker::Parameters params;
  params.ekf = std::make_shared<EKF>(ekf_params);
  params.camera_id = 1;
  FeatureTracker tracker{params};

  // Create matches
  std::vector<std::vector<cv::DMatch>> matches;

  // 1. Match with 1 neighbor -> should be cleared
  std::vector<cv::DMatch> m1 = {cv::DMatch(0, 0, 10.0f)};
  matches.push_back(m1);

  // 2. Match with 2 neighbors, ratio > 0.7 (e.g., 8.0/10.0 = 0.8) -> should be cleared
  std::vector<cv::DMatch> m2 = {cv::DMatch(1, 1, 8.0f), cv::DMatch(1, 2, 10.0f)};
  matches.push_back(m2);

  // 3. Match with 2 neighbors, ratio <= 0.7 (e.g., 5.0/10.0 = 0.5) -> should be kept
  std::vector<cv::DMatch> m3 = {cv::DMatch(2, 3, 5.0f), cv::DMatch(2, 4, 10.0f)};
  matches.push_back(m3);

  tracker.RatioTest(matches);

  ASSERT_EQ(matches.size(), 3);
  EXPECT_TRUE(matches[0].empty());
  EXPECT_TRUE(matches[1].empty());
  EXPECT_FALSE(matches[2].empty());
  EXPECT_EQ(matches[2].size(), 2);
  EXPECT_EQ(matches[2][0].trainIdx, 3);
}

TEST(test_feature_tracker, SymmetryTest) {
  std::vector<std::vector<cv::DMatch>> matches_forward;
  std::vector<std::vector<cv::DMatch>> matches_backward;
  std::vector<cv::DMatch> matches_out;

  // 1. Symmetric match:
  // Forward: feature 0 matches to 1 (best) and 2 (second)
  // Backward: feature 1 matches to 0 (best) and 3 (second)
  // Both must have size 2 to be processed.
  std::vector<cv::DMatch> f1 = {cv::DMatch(0, 1, 1.0f), cv::DMatch(0, 2, 2.0f)};
  std::vector<cv::DMatch> b1 = {cv::DMatch(1, 0, 1.0f), cv::DMatch(1, 3, 2.0f)};
  matches_forward.push_back(f1);
  matches_backward.push_back(b1);

  // 2. Asymmetric match:
  // Forward: feature 4 matches to 5 and 6
  // Backward: feature 5 matches to 7 and 8 (not 4)
  std::vector<cv::DMatch> f2 = {cv::DMatch(4, 5, 1.0f), cv::DMatch(4, 6, 2.0f)};
  std::vector<cv::DMatch> b2 = {cv::DMatch(5, 7, 1.0f), cv::DMatch(5, 8, 2.0f)};
  matches_forward.push_back(f2);
  matches_backward.push_back(b2);

  FeatureTracker::SymmetryTest(matches_forward, matches_backward, matches_out);

  ASSERT_EQ(matches_out.size(), 1);
  EXPECT_EQ(matches_out[0].queryIdx, 0);
  EXPECT_EQ(matches_out[0].trainIdx, 1);
}

TEST(test_feature_tracker, RANSAC) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  FeatureTracker::Parameters params;
  params.ekf = std::make_shared<EKF>(ekf_params);
  params.camera_id = 1;

  // Register camera with non-zero distortion parameters to verify undistortion code path
  CamState cam_state;
  cam_state.intrinsics.f_x = 0.01;
  cam_state.intrinsics.f_y = 0.01;
  cam_state.intrinsics.width = 640.0;
  cam_state.intrinsics.height = 480.0;
  cam_state.intrinsics.pixel_size = 5.0e-6;
  cam_state.intrinsics.k_1 = -0.1;
  cam_state.intrinsics.k_2 = 0.02;
  cam_state.intrinsics.p_1 = 0.001;
  cam_state.intrinsics.p_2 = 0.001;
  params.ekf->RegisterCamera(1, cam_state, Eigen::MatrixXd::Zero(6, 6));

  FeatureTracker tracker{params};

  // Populate m_prev_key_points
  FeatureTrackerTestAccess::PrevKeyPoints(tracker).clear();
  std::vector<cv::KeyPoint> curr_key_points;
  std::vector<cv::DMatch> matches_in;

  // Create 10 inliers and 1 outlier
  for (int i = 0; i < 11; ++i) {
    float x_prev = static_cast<float>(i % 3) * 50.0f;
    float y_prev = static_cast<float>(i / 3.0) * 50.0f;
    FeatureTrackerTestAccess::PrevKeyPoints(tracker).emplace_back(
      cv::Point2f(x_prev, y_prev), 1.0f);

    float x_curr, y_curr;
    if (i < 10) {
      // Inlier: simple shift by (5.0, 5.0)
      x_curr = x_prev + 5.0f;
      y_curr = y_prev + 5.0f;
    } else {
      // Outlier: different shift
      x_curr = x_prev + 100.0f;
      y_curr = y_prev - 100.0f;
    }
    curr_key_points.emplace_back(cv::Point2f(x_curr, y_curr), 1.0f);
    matches_in.emplace_back(i, i, 1.0f);
  }

  std::vector<cv::DMatch> matches_out;
  tracker.RANSAC(matches_in, curr_key_points, matches_out);

  // We expect at least the 10 inliers to pass, and the outlier to be rejected.
  EXPECT_GE(matches_out.size(), 7);
  for (const auto & match : matches_out) {
    EXPECT_LT(match.queryIdx, 10);
    EXPECT_LT(match.trainIdx, 10);
  }

  // Test RANSAC with less than 10 matches
  std::vector<cv::DMatch> small_matches_in(5, cv::DMatch(0, 0, 1.0f));
  std::vector<cv::DMatch> small_matches_out;
  tracker.RANSAC(small_matches_in, curr_key_points, small_matches_out);
  EXPECT_TRUE(small_matches_out.empty());
}

TEST(test_feature_tracker, DistanceTest) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  FeatureTracker::Parameters params;
  params.ekf = std::make_shared<EKF>(ekf_params);
  params.camera_id = 1;
  FeatureTracker tracker{params};

  FeatureTrackerTestAccess::PrevKeyPoints(tracker).clear();
  std::vector<cv::KeyPoint> curr_key_points;
  std::vector<cv::DMatch> matches_in;

  // Match 0: dist = 10.0
  FeatureTrackerTestAccess::PrevKeyPoints(tracker).emplace_back(cv::Point2f(0.0f, 0.0f), 1.0f);
  curr_key_points.emplace_back(cv::Point2f(10.0f, 0.0f), 1.0f);
  matches_in.emplace_back(0, 0, 1.0f);

  // Match 1: dist = 10.0
  FeatureTrackerTestAccess::PrevKeyPoints(tracker).emplace_back(cv::Point2f(10.0f, 10.0f), 1.0f);
  curr_key_points.emplace_back(cv::Point2f(20.0f, 10.0f), 1.0f);
  matches_in.emplace_back(1, 1, 1.0f);

  // Match 2: dist = 10.0
  FeatureTrackerTestAccess::PrevKeyPoints(tracker).emplace_back(cv::Point2f(20.0f, 20.0f), 1.0f);
  curr_key_points.emplace_back(cv::Point2f(30.0f, 20.0f), 1.0f);
  matches_in.emplace_back(2, 2, 1.0f);

  // Match 3: dist = 100.0 (outlier)
  FeatureTrackerTestAccess::PrevKeyPoints(tracker).emplace_back(cv::Point2f(30.0f, 30.0f), 1.0f);
  curr_key_points.emplace_back(cv::Point2f(130.0f, 30.0f), 1.0f);
  matches_in.emplace_back(3, 3, 1.0f);

  std::vector<cv::DMatch> matches_out;
  tracker.DistanceTest(matches_in, curr_key_points, matches_out);

  ASSERT_EQ(matches_out.size(), 3);
  EXPECT_EQ(matches_out[0].queryIdx, 0);
  EXPECT_EQ(matches_out[1].queryIdx, 1);
  EXPECT_EQ(matches_out[2].queryIdx, 2);
}

TEST(test_feature_tracker, TrackDirect) {
  EKF::Parameters ekf_params;
  ekf_params.debug_logger = std::make_shared<DebugLogger>(LogLevel::DEBUG, "");
  auto ekf = std::make_shared<EKF>(ekf_params);
  BodyState body_state_init;
  body_state_init.vel_b_in_l[1] = 0.1;
  ekf->Initialize(0.0, body_state_init);

  FeatureTracker::Parameters tracker_params;
  tracker_params.px_error = 0.1;
  tracker_params.max_track_length = 2;
  tracker_params.camera_id = 1;
  tracker_params.ekf = ekf;
  tracker_params.min_feat_dist = 0.1;
  tracker_params.logger = ekf_params.debug_logger;
  tracker_params.detector = Detector::FAST;
  tracker_params.descriptor = Descriptor::ORB;
  tracker_params.matcher = Matcher::BRUTE_FORCE;
  tracker_params.down_sample = true;
  tracker_params.down_sample_height = 240;
  tracker_params.down_sample_width = 320;

  FeatureTracker feature_tracker{tracker_params};

  std::string source_file = __FILE__;
  std::string source_dir = source_file.substr(0, source_file.find_last_of("/\\"));
  cv::Mat img_1 =
    cv::imread(source_dir + "/images/tsukuba_l.png", cv::IMREAD_GRAYSCALE);
  cv::Mat img_2 =
    cv::imread(source_dir + "/images/tsukuba_r.png", cv::IMREAD_GRAYSCALE);

  ASSERT_FALSE(img_1.empty());
  ASSERT_FALSE(img_2.empty());

  cv::Mat img_out_1 = cv::Mat::zeros(img_1.size(), CV_8UC3);
  cv::Mat img_out_2 = cv::Mat::zeros(img_2.size(), CV_8UC3);

  feature_tracker.Track(0.0, 1, img_1, img_out_1);

  EXPECT_EQ(FeatureTrackerTestAccess::PrevFrameId(feature_tracker), 1);
  EXPECT_GT(FeatureTrackerTestAccess::PrevKeyPoints(feature_tracker).size(), 0);

  feature_tracker.Track(1.0, 2, img_2, img_out_2);

  EXPECT_EQ(FeatureTrackerTestAccess::PrevFrameId(feature_tracker), 2);
}
