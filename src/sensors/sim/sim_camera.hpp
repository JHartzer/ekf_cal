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

#ifndef SENSORS__SIM__SIM_CAMERA_HPP_
#define SENSORS__SIM__SIM_CAMERA_HPP_

#include <Eigen/Core>

#include <string>
#include <map>
#include <memory>
#include <vector>

#include "ekf/types.hpp"
#include "infrastructure/sim/truth_engine.hpp"
#include "sensors/camera.hpp"
#include "sensors/sim/sim_camera_message.hpp"
#include "sensors/sim/sim_sensor.hpp"
#include "trackers/sim/sim_feature_tracker.hpp"
#include "trackers/sim/sim_fiducial_tracker.hpp"


///
/// @class SimCamera
/// @brief Simulated Camera Sensor Class
///
class SimCamera : public Camera, public SimSensor
{
public:
  ///
  /// @brief Sim IMU initialization parameters structure
  ///
  typedef struct Parameters : public SimSensor::Parameters
  {
    Eigen::Vector3d pos_error {0.0, 0.0, 0.0};  ///< @brief Position offset error
    Eigen::Vector3d ang_error {0.0, 0.0, 0.0};  ///< @brief Angular offset error
    int room_size {4};                          ///< @brief Room width/depth
    bool generate_video {false};                ///< @brief Render simulated video feed
    Camera::Parameters cam_params;              ///< @brief Camera sensor parameters
  } Parameters;

  ///
  /// @brief Simulation IMU constructor
  /// @param params Simulation IMU parameters
  /// @param truth_engine Truth engine
  ///
  SimCamera(SimCamera::Parameters params, std::shared_ptr<TruthEngine> truth_engine);

  ///
  /// @brief Method to add tracker object to camera sensor
  /// @param tracker Tracker pointer for camera to use during callbacks
  ///
  void AddTracker(std::shared_ptr<SimFeatureTracker> tracker);

  ///
  /// @brief Method to add fiducial object to camera sensor
  /// @param fiducial Fiducial pointer for camera to use during callbacks
  ///
  void AddFiducial(std::shared_ptr<SimFiducialTracker> fiducial);

  ///
  /// @brief Callback method for simulated camera
  /// @param sim_camera_message Simulated camera data message
  ///
  void Callback(const SimCameraMessage & sim_camera_message);

  ///
  /// @brief Generate simulated IMU messages
  /// @return Generated camera messages
  ///
  std::vector<std::shared_ptr<SimCameraMessage>> GenerateMessages();

private:
  static constexpr unsigned int SHOW_NTH_TRACK = 10;

  static cv::Scalar GetTrackColor(unsigned int tracker_id, unsigned int feature_id);
  static bool ShouldShowTrack(unsigned int feature_id);
  static double GetTrackAlpha(unsigned int age, unsigned int max_track_length);
  bool GetFeatureDepth(double time, int feature_id, double & depth) const;
  static int GetFeatureRadius(double depth);
  void BlendCircle(
    cv::Mat & image,
    const cv::Point & center,
    int radius,
    const cv::Scalar & color,
    double alpha
  ) const;
  void BlendLine(
    cv::Mat & image,
    const cv::Point & point_1,
    const cv::Point & point_2,
    const cv::Scalar & color,
    double alpha,
    int thickness
  ) const;
  void OverlayBufferedFeatureTrack(
    std::map<unsigned int, cv::Mat> & frame_buffer,
    const FeatureTrack & feature_track,
    unsigned int tracker_id,
    unsigned int max_track_length
  ) const;
  void FlushReadyFrames(
    std::map<unsigned int, cv::Mat> & frame_buffer,
    unsigned int current_frame_id,
    unsigned int max_track_length,
    bool flush_all
  );
  cv::Mat RenderFrame(double time) const;
  void DrawWorldLine(
    cv::Mat & image,
    const Eigen::Vector3d & start,
    const Eigen::Vector3d & end,
    const cv::Mat & r_vec,
    const cv::Mat & t_vec,
    const Intrinsics & intrinsics,
    int thickness
  ) const;
  bool DrawProjectedLine(
    cv::Mat & image,
    const cv::Point3d & point_1,
    const cv::Point3d & point_2,
    const cv::Mat & r_vec,
    const cv::Mat & t_vec,
    const Intrinsics & intrinsics,
    const cv::Scalar & color,
    int thickness
  ) const;
  void ExecuteCallback(const SimCameraMessage & sim_camera_message);

  Eigen::Vector3d m_pos_error;
  Eigen::Vector3d m_ang_error;
  int m_room_size {4};
  bool m_generate_video {false};
  std::string m_video_path;
  mutable cv::VideoWriter m_video_writer;

  std::map<unsigned int, std::shared_ptr<SimFeatureTracker>> m_trackers;
  std::map<unsigned int, std::shared_ptr<SimFiducialTracker>> m_fiducials;
};


#endif  // SENSORS__SIM__SIM_CAMERA_HPP_
