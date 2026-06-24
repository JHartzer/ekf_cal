# ekf_cal

Extended Kalman Filter Calibration and Localization: ekf_cal is a package focused on the simulation and development of a multi-sensor online calibration kalman filter. It combines the architecture of a Multi-State Constraint Kalman Filter (MSCKF) with a multi-sensor calibration filter to provide intrinsic and extrinsic estimates for the following sensors:
- [IMU](docs/filters/imu.md#imu)
- [GPS](docs/filters/gps.md#gps)
- [Cameras](docs/filters/camera.md#camera)
- [Fiducials](docs/filters/fiducial.md#fiducial)

![Frame Graph](docs/doxygen/html/images/setup.svg)

![Experimental Feature Tracking](docs/doxygen/html/images/feature-tracking.gif)

## Quick Start

### Clone the Repository

This guide assumes you have the [ekf_cal](https://github.com/jhartzer/ekf_cal/) repository in a colcon workspace.
```bash
mkdir ekf_cal_ws/
mkdir ekf_cal_ws/src/
cd ekf_cal_ws/src/
git clone git@github.com:JHartzer/ekf_cal.git
cd ../
```

### Dependencies
The ekf_cal package has the following hard dependencies that are required for all compilations:
- [OpenCV](https://opencv.org/)
- [Eigen 3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

The following dependencies are for building the ROS node and simulation, respectively
- [ROS2](https://docs.ros.org/en/rolling/index.html)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)

The following soft dependencies useful for development and documentation
- [Doxygen](https://www.doxygen.nl/index.html)
- [Google Test](https://google.github.io/googletest/)
- [pre-commit](https://pre-commit.com/)

These can be installed by running [rosdep](https://wiki.ros.org/rosdep) in the base directory of the colcon workspace (e.g. `ekf_cal_ws`)
```bash
rosdep install --from-paths src -y --ignore-src
```
or manually via apt
```bash
sudo apt-get install \
  autopep8 \
  cloc \
  cmake \
  doxygen \
  lcov \
  libeigen3-dev \
  libgtest-dev \
  libopencv-dev libopencv-contrib-dev \
  libyaml-cpp-dev \
  pre-commit \
  uncrustify \
  -y
```

### Build

Building can be done simply with the following command:

```bash
colcon build --symlink-install --packages-select ekf_cal --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release
```

#### Docker
Alternatively, a Dockerfile is provided, which can be used either inside a VS Code [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers), or a standalone container.

### Input Files

This repository offers two main ways to utilize the Kalman filter framework: a simulation and ROS2 node. Both the simulation and ROS node are configurable and runnable using identically formatted YAML files. Further documentation on how to configure this YAML file for a particular setup can be found on the [Parameters](docs/parameters.md#parameters) page.

### Simulation

Simulations can be run using a YAML configuration file that extends the base configuration file with additional parameters. See the example [example.yaml](config/example.yaml). Multiple simulations can be run in parallel using the [run.py](eval/run.py). An example using a single input is given below

```bash
python3 eval/run.py config/example.yaml
```

The results of a run can be plotted using [report.py](eval/report.py)
```bash
python3 eval/report.py config/example.yaml
```

To run and plot in sequence, utilize [evaluate.py](eval/evaluate.py)
```bash
python3 eval/evaluate.py config/example.yaml
```

This will generate and run the requested number of simulation runs for the specified run time and produce plots of the Monte Carlo data. For example, the report generates plots of the body acceleration estimates and the true error in those acceleration estimates.

![acceleration](docs/doxygen/html/images/acceleration.png)

![acceleration-error](docs/doxygen/html/images/acceleration-error.png)

### Launch ROS2 Node

For an example of a filter node launch file, see [example.launch.py](launch/example.launch.py)

In particular, note the configuration file [example.yaml](config/example.yaml).

The configuration file specifies which sensor topics should to use and the initialization values. Once built, the ROS node can be started by running the following command

```bash
ros2 launch example.launch
```

Evaluating the output of the ROS node is the same as with the simulations, where reports can be generated using the resultant log files.

## Testing & Static Analysis

Once the package has been built, unit tests and static analysis can be run with the following commands
```bash
colcon test --packages-select ekf_cal --event-handlers console_direct+
```

Install the local formatting and test hooks with:
```bash
python3 -m pip install -r requirements.txt
pre-commit install
```

The configured pre-commit hooks will:
- run `autopep8` on staged Python files
- run `uncrustify` on staged C and C++ files using `uncrustify.cfg`
- configure, build, and run all CTest unit tests in `.precommit-build/RelWithDebInfo` using the repo's `RelWithDebInfo` compiler/build settings

A test code coverage report can be generated using the following commands
```bash
colcon build --symlink-install --packages-select ekf_cal \
   --event-handlers console_cohesion+ \
   --cmake-args -DCMAKE_C_FLAGS='--coverage' -DCMAKE_CXX_FLAGS='--coverage'

colcon test --packages-select ekf_cal --pytest-with-coverage \
   --pytest-args --cov-report=term --event-handlers console_direct+

colcon lcov-result --packages-select ekf_cal --filter '*_test.cpp' '*_main.cpp'
```

For the local non-ROS coverage workflow, configure, build, run the coverage test suite, and generate the HTML report with:
```bash
./tools/report-coverage.sh
```

A performance [flamegraph](https://github.com/brendangregg/FlameGraph) can be generated using the following command

```bash
./tools/generate-flamegraph.sh
```

## Documentation

Documentation can be generated using the following command:
```bash
./tools/build-docs.sh
```

The JOSS paper can be built with:
```bash
./tools/build-joss-paper.sh
```

## Citation

Please cite this software as

```bibtex
@article{2025_EKF_CAL,
  title = {EKF_CAL: Extended Kalman Filter-based Calibration and Localization},
  author = {Hartzer, Jacob and Saripalli, Srikanth},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  url = {https://doi.org/10.21105/joss.07793},
  doi = {10.21105/joss.07793},
  year = {2025},
  volume = {10},
  number = {109},
  pages = {7793},
}
```

## References

For technical reference, please see the following

```bibtex
@inproceedings{2023_Multi_IMU,
  title         = {Online Multi-IMU Calibration Using Visual-Inertial Odometry},
  booktitle     = {2023 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI)},
  author        = {Jacob Hartzer and Srikanth Saripalli},
  year          = {2023},
  doi           = {10.1109/SDF-MFI59545.2023.10361310},
  arxiv         = {2310.12411},
}
```
```bibtex
@inproceedings{2022_Multi_Cam,
  title     = {Online Multi Camera-IMU Calibration},
  booktitle = {2022 IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR)},
  author    = {Hartzer, Jacob and Saripalli, Srikanth},
  year      = {2022},
  pages     = {360-365},
  doi       = {10.1109/SSRR56537.2022.10018692},
  arxiv     = {2209.13821},
}
```
