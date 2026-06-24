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

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "utility/custom_assertions.hpp"
#include "utility/sim/sim_rng.hpp"


TEST(test_SimRNG, RNG) {
  SimRNG::SetSeed(123456789);

  double norm_rand = SimRNG::NormRand(0, 1);
  double uni_rand = SimRNG::UniRand(0, 1);
  double exp_rand = SimRNG::ExpRand(2.0);

  EXPECT_NEAR(norm_rand, -1.2864683, 1e-6);
  EXPECT_NEAR(uni_rand, 0.1366463, 1e-6);
  EXPECT_NEAR(exp_rand, 0.0144863, 1e-6);
  EXPECT_GE(exp_rand, 0.0);

  Eigen::Vector3d vec_mean{0.5, 0.5, 0.5};
  Eigen::Vector3d vec_std_dev{1, 1, 1};
  Eigen::Quaterniond quat_mean{1, 0, 0, 0};
  Eigen::Vector3d quat_std_dev{0.1, 0.1, 0.1};

  Eigen::Vector3d rand_vec = SimRNG::VecNormRand(vec_mean, vec_std_dev);
  Eigen::Quaterniond rand_quat = SimRNG::QuatNormRand(quat_mean, quat_std_dev);

  EXPECT_TRUE(
    EXPECT_EIGEN_NEAR(rand_vec, Eigen::Vector3d {0.1418361, 0.2913381, 1.5366241}, 1e-6));

  EXPECT_TRUE(
    EXPECT_EIGEN_NEAR(
      rand_quat, Eigen::Quaterniond {0.9956954, 0.0105244, -0.0906098, -0.0164267}, 1e-6));
}
