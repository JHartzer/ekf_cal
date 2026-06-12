// Copyright 2023 Jacob Hartzer
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

#ifndef INFRASTRUCTURE__DATA_LOGGER_HPP_
#define INFRASTRUCTURE__DATA_LOGGER_HPP_

#include <string>
#include <memory>
#include <vector>
#include <H5Cpp.h>

///
/// @brief DataLogger class
///
class DataLogger
{
public:
  DataLogger() {}

  ///
  /// @brief DataLogger constructor
  /// @param log_directory Output directory for creating data log file
  /// @param name Name of data logger dataset
  ///
  DataLogger(const std::string & log_directory, const std::string & name);

  ///
  /// @brief DataLogger constructor
  /// @param log_directory Output directory for creating data log file
  /// @param name Name of data logger dataset
  /// @param logging_rate Logging rate
  ///
  DataLogger(const std::string & log_directory, const std::string & name, double logging_rate);

  ///
  /// @brief Log message
  /// @param message Message contents of log
  ///
  void Log(const std::string & message);

  ///
  /// @brief Log rate-limited messages
  /// @param message Message contents of log
  /// @param time Message log time for rate-limited logging
  ///
  void RateLimitedLog(const std::string & message, double time);

  ///
  /// @brief Log vector of double values directly
  /// @param values Values to log
  ///
  void Log(const std::vector<double> & values);

  ///
  /// @brief Log rate-limited vector of double values directly
  /// @param values Values to log
  /// @param time Message log time for rate-limited logging
  ///
  void RateLimitedLog(const std::vector<double> & values, double time);

  ///
  /// @brief Function to set the output file header
  /// @param header Header string for output file
  ///
  void DefineHeader(const std::string & header);

  ///
  /// @brief Function to enable logging
  ///
  void EnableLogging();

  ///
  /// @brief Output directory setter
  /// @param log_directory Output directory string
  ///
  void SetOutputDirectory(const std::string & log_directory);

  ///
  /// @brief Data logger name setter
  /// @param name Data logger name
  ///
  void SetName(const std::string & name);

  ///
  /// @brief Data logging rate setter
  /// @param rate Data logging rate
  ///
  void SetLogRate(double rate);

private:
  bool m_initialized{false};
  std::string m_log_header{""};
  std::shared_ptr<H5::H5File> m_h5_file;
  std::unique_ptr<H5::DataSet> m_dataset;
  hsize_t m_current_rows{0};
  hsize_t m_num_cols{0};
  bool m_logging_on {false};
  std::string m_log_directory {""};
  std::string m_name {"default"};
  double m_rate{0.0};
  double m_time_init{0};
  unsigned int m_log_count{0};

  void InitializeHdf5();
};

#endif  // INFRASTRUCTURE__DATA_LOGGER_HPP_
