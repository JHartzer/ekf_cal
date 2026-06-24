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

#include "sensors/gps.hpp"

#include <Eigen/Core>

#include <memory>
#include <string>

#include "ekf/types.hpp"
#include "infrastructure/debug_logger.hpp"
#include "sensors/gps_message.hpp"
#include "sensors/sensor.hpp"


GPS::GPS(GPS::Parameters params)
: Sensor(params),
  m_ekf(params.ekf),
  m_gps_updater(
    m_id,
    params.is_extrinsic,
    params.log_directory,
    params.data_log_rate,
    params.logger)
{
  InitializeTimingLogger("gps", params.log_directory, params.data_log_rate);
  m_rate = params.rate;
  GpsState gps_state;
  gps_state.pos_a_in_b = params.pos_a_in_b;
  gps_state.pos_stability = params.pos_stability;
  gps_state.SetIsExtrinsic(params.is_extrinsic);
  Eigen::Matrix3d gps_cov = Eigen::Matrix3d::Identity() * params.variance.pos;
  m_ekf->RegisterGPS(m_id, gps_state, gps_cov);
}

void GPS::Callback(const GpsMessage & gps_message)
{
  BufferMessage(
    gps_message,
    m_message_buffer,
    [this](const GpsMessage & buffered_message) {ExecuteCallback(buffered_message);});
}

void GPS::ExecuteCallback(const GpsMessage & gps_message)
{
  LogTiming(gps_message);
  m_logger->Log(
    LogLevel::DEBUG,
    "GPS \"" + m_name + "\" callback at used time " + std::to_string(gps_message.time_used));

  m_gps_updater.UpdateEKF(
    *m_ekf, gps_message.time_used, gps_message.gps_lla, gps_message.pos_covariance);

  m_logger->Log(LogLevel::DEBUG, "GPS \"" + m_name + "\" callback complete");
}

void GPS::Flush()
{
  FlushBufferedMessages(
    m_message_buffer,
    [this](const GpsMessage & buffered_message) {ExecuteCallback(buffered_message);});
}

bool GPS::HasBufferedMeasurements() const
{
  return HasBufferedMessages(m_message_buffer);
}

double GPS::GetNextBufferedMeasurementTime() const
{
  return GetNextBufferedMessageTime(m_message_buffer);
}

bool GPS::FlushNextMeasurement()
{
  return FlushNextBufferedMessage(
    m_message_buffer,
    [this](const GpsMessage & buffered_message) {ExecuteCallback(buffered_message);});
}
