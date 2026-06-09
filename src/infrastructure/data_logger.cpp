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

#include "infrastructure/data_logger.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include "infrastructure/hdf5_log_manager.hpp"


DataLogger::DataLogger(const std::string & log_directory, const std::string & file_name)
{
  m_log_directory = log_directory;
  m_file_name = file_name;
}

DataLogger::DataLogger(
  const std::string & log_directory, const std::string & file_name,
  double logging_rate)
{
  m_log_directory = log_directory;
  m_file_name = file_name;
  m_rate = logging_rate;
}

static std::string get_hdf5_filename(const std::string & directory)
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

void DataLogger::InitializeHdf5()
{
  if (m_initialized) {return;}

  std::string h5_filename = get_hdf5_filename(m_log_directory);
  m_h5_file = Hdf5LogManager::GetFile(m_log_directory, h5_filename);
  if (!m_h5_file) {
    m_logging_on = false;
    return;
  }

  try {
    std::string name = m_file_name;
    if (name.length() > 4 && name.substr(name.length() - 4) == ".csv") {
      name = name.substr(0, name.length() - 4);
    }
    std::string dataset_path;
    if (name.rfind("gps_", 0) == 0 || name.rfind("imu_", 0) == 0 ||
      name.rfind("camera_", 0) == 0 || name.rfind("msckf_", 0) == 0 ||
      name.rfind("triangulation_", 0) == 0 || name.rfind("fiducial_", 0) == 0)
    {
      dataset_path = "sensors/" + name;
    } else if (name == "body_truth") {
      dataset_path = "truth/body";
    } else if (name == "board_truth") {
      dataset_path = "truth/board";
    } else if (name == "feature_points") {
      dataset_path = "truth/feature_points";
    } else {
      dataset_path = name;
    }

    m_num_cols = 0;
    if (!m_log_header.empty()) {
      std::stringstream ss(m_log_header);
      std::string col;
      while (std::getline(ss, col, ',')) {
        m_num_cols++;
      }
    }

    if (m_num_cols == 0) {
      return;
    }

    H5::DSetCreatPropList cparms;
    hsize_t chunk_dims[2] = {100, m_num_cols};
    cparms.setChunk(2, chunk_dims);

    hsize_t dims[2] = {0, m_num_cols};
    hsize_t maxdims[2] = {H5S_UNLIMITED, m_num_cols};
    H5::DataSpace dataspace(2, dims, maxdims);

    size_t pos = dataset_path.find('/');
    if (pos != std::string::npos) {
      std::string group_name = dataset_path.substr(0, pos);
      try {
        if (H5Lexists(m_h5_file->getId(), group_name.c_str(), H5P_DEFAULT) <= 0) {
          m_h5_file->createGroup(group_name);
        }
      } catch (...) {
      }
    }

    m_dataset = std::make_unique<H5::DataSet>(
      m_h5_file->createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dataspace, cparms));

    H5::StrType str_type(H5::PredType::C_S1, m_log_header.size() + 1);
    H5::Attribute attr = m_dataset->createAttribute("column_names", str_type, H5S_SCALAR);
    attr.write(str_type, m_log_header);

    m_current_rows = 0;
    m_initialized = true;
  } catch (...) {
    m_logging_on = false;
  }
}

void DataLogger::Log(const std::vector<double> & values)
{
  if (m_logging_on) {
    if (!m_initialized) {
      if (m_num_cols == 0) {
        m_num_cols = values.size();
        if (m_log_header.empty()) {
          std::stringstream ss;
          for (size_t i = 0; i < m_num_cols; ++i) {
            ss << "col_" << i;
            if (i + 1 < m_num_cols) {ss << ",";}
          }
          m_log_header = ss.str();
        }
      }
      InitializeHdf5();
    }

    if (m_initialized && m_num_cols == values.size()) {
      try {
        hsize_t new_size[2] = {m_current_rows + 1, m_num_cols};
        m_dataset->extend(new_size);

        H5::DataSpace filespace = m_dataset->getSpace();
        hsize_t offset[2] = {m_current_rows, 0};
        hsize_t count[2] = {1, m_num_cols};
        filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

        H5::DataSpace memspace(2, count);
        m_dataset->write(values.data(), H5::PredType::NATIVE_DOUBLE, memspace, filespace);
        m_current_rows++;
        m_log_count++;
      } catch (...) {
        m_logging_on = false;
      }
    }
  }
}

void DataLogger::Log(const std::string & message)
{
  if (m_logging_on) {
    std::vector<double> values;
    std::stringstream ss(message);
    std::string item;
    while (std::getline(ss, item, ',')) {
      try {
        values.push_back(std::stod(item));
      } catch (...) {
        values.push_back(0.0);
      }
    }
    Log(values);
  }
}

void DataLogger::RateLimitedLog(const std::vector<double> & values, double time)
{
  if (m_logging_on) {
    if (m_time_init != 0.0) {
      double log_count = static_cast<double>(m_log_count);
      double max_count = m_rate * (time - m_time_init);
      if ((m_rate == 0.0) || (log_count < max_count)) {
        Log(values);
      }
    } else {
      m_time_init = time;
      Log(values);
    }
  }
}

void DataLogger::RateLimitedLog(const std::string & message, double time)
{
  if (m_logging_on) {
    if (m_time_init != 0.0) {
      double log_count = static_cast<double>(m_log_count);
      double max_count = m_rate * (time - m_time_init);
      if ((m_rate == 0.0) || (log_count < max_count)) {
        Log(message);
      }
    } else {
      m_time_init = time;
      Log(message);
    }
  }
}

void DataLogger::EnableLogging()
{
  m_logging_on = true;
}

void DataLogger::SetOutputDirectory(const std::string & log_directory)
{
  m_log_directory = log_directory;
}

void DataLogger::SetOutputFileName(const std::string & file_name)
{
  m_file_name = file_name;
}

void DataLogger::DefineHeader(const std::string & header)
{
  m_log_header = header;
}

void DataLogger::SetLogRate(double rate)
{
  m_rate = rate;
}
