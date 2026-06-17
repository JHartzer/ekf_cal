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

#include <gtest/gtest.h>

#include <string>
#include "infrastructure/hdf5_log_manager.hpp"


TEST(data_logger, data_logger_constructor_1) {
  DataLogger data_logger;

  data_logger.Log("a1,b1");
  data_logger.SetOutputDirectory("/temp/");
  data_logger.SetName("data");
  data_logger.DefineHeader("col1,col2");
  data_logger.EnableLogging();
  data_logger.Log("a1,b1");
}

TEST(data_logger, data_logger_constructor_2) {
  DataLogger data_logger("/temp/", "data");

  data_logger.Log("a1,b1");
  data_logger.DefineHeader("col1,col2");
  data_logger.EnableLogging();
  data_logger.Log("a1,b1");
}

TEST(data_logger, data_logger_constructor_3) {
  DataLogger data_logger("/temp/", "data", 1.0);

  data_logger.Log("a1,b1");
  data_logger.DefineHeader("col1,col2");
  data_logger.EnableLogging();
  data_logger.Log("a1,b1");
}

TEST(data_logger, rate_limited_log) {
  // Overload 1: string
  {
    DataLogger data_logger("/temp/", "data_str", 10.0);
    data_logger.DefineHeader("col1,col2");
    data_logger.EnableLogging();

    // First log (m_time_init is 0.0, so it sets m_time_init = 1.0 and logs)
    data_logger.RateLimitedLog("1.0,2.0", 1.0);

    // Second log: at 1.1s. max_count = 10.0 * (1.1 - 1.0) = 1.0.
    // log_count is 1.0. log_count < max_count is 1.0 < 1.0 (false), so rate limited!
    data_logger.RateLimitedLog("3.0,4.0", 1.1);

    // Third log: at 1.21s. max_count = 10.0 * (1.21 - 1.0) = 2.1.
    // log_count is 1.0. log_count < max_count is 1.0 < 2.1 (true), so it logs!
    data_logger.RateLimitedLog("5.0,6.0", 1.21);
  }

  // Overload 2: vector<double>
  {
    DataLogger data_logger("/temp/", "data_vec", 5.0);
    data_logger.DefineHeader("col1,col2");
    data_logger.EnableLogging();

    std::vector<double> v1 = {1.0, 2.0};
    std::vector<double> v2 = {3.0, 4.0};

    // First log
    data_logger.RateLimitedLog(v1, 1.0);

    // Second log: at 1.1s
    data_logger.RateLimitedLog(v2, 1.1);

    // Third log: at 1.3s
    data_logger.RateLimitedLog(v2, 1.3);
  }
}

TEST(data_logger, coverage_extra) {
  // 1. Line 80: Name that does not match prefixes
  {
    DataLogger data_logger("/temp/", "custom_name");
    data_logger.DefineHeader("col1");
    data_logger.EnableLogging();
    data_logger.Log(std::vector<double>{1.0});
  }

  // 2. Line 93: m_num_cols == 0
  {
    DataLogger data_logger("/temp/", "empty_cols");
    data_logger.EnableLogging();
    data_logger.Log(std::vector<double>{}); // size is 0
  }

  // 3. Line 125: exception in InitializeHdf5
  {
    DataLogger logger1("/temp/", "dup_dataset");
    logger1.DefineHeader("col1");
    logger1.EnableLogging();
    logger1.Log(std::vector<double>{1.0});

    DataLogger logger2("/temp/", "dup_dataset");
    logger2.DefineHeader("col1");
    logger2.EnableLogging();
    logger2.Log(std::vector<double>{1.0}); // will fail and set m_logging_on = false
  }

  // 4. Line 136: m_log_header.empty()
  {
    DataLogger data_logger("/temp/", "no_header");
    data_logger.EnableLogging();
    data_logger.Log(std::vector<double>{1.0, 2.0});
  }

  // 5. Line 162: exception in Log (dataset extend/write)
  {
    std::string dir = "/temp_close/";
    DataLogger data_logger(dir, "closelog");
    data_logger.DefineHeader("col1");
    data_logger.EnableLogging();
    data_logger.Log(std::vector<double>{1.0});

    // Invalidate/close file
    std::shared_ptr<H5::H5File> file = Hdf5LogManager::GetFile(dir, "temp_close.h5");
    if (file) {
      file->close();
    }
    data_logger.Log(std::vector<double>{2.0}); // throws and sets m_logging_on = false
  }
}
