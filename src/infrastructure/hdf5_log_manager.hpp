// Copyright 2026 Jacob Hartzer
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

#ifndef INFRASTRUCTURE__HDF5_LOG_MANAGER_HPP_
#define INFRASTRUCTURE__HDF5_LOG_MANAGER_HPP_

#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <H5Cpp.h>

class Hdf5LogManager
{
public:
  static std::shared_ptr<H5::H5File> GetFile(
    const std::string & directory,
    const std::string & filename = "simulation_data.h5")
  {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<H5::H5File>> open_files;

    std::lock_guard<std::mutex> lock(mutex);
    std::string full_path = directory;
    if (!full_path.empty() && full_path.back() != '/') {
      full_path += "/";
    }
    full_path += filename;

    if (open_files.find(full_path) == open_files.end()) {
      try {
        if (!directory.empty()) {
          std::string cmd = "mkdir -p \"" + directory + "\"";
          int ret = std::system(cmd.c_str());
          (void)ret;
        }
        open_files[full_path] = std::make_shared<H5::H5File>(full_path, H5F_ACC_TRUNC);
      } catch (...) {
        open_files[full_path] = nullptr;
      }
    }
    return open_files[full_path];
  }

private:
  Hdf5LogManager() = default;
  ~Hdf5LogManager() = default;
};

#endif  // INFRASTRUCTURE__HDF5_LOG_MANAGER_HPP_
