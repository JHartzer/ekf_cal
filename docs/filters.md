Filter Design {#filter}
=============

`ekf_cal` combines the architecture of a Multi-State Constraint Kalman Filter (MSCKF) with a multi-sensor calibration filter to provide intrinsic and extrinsic estimates for the following sensors:

- @subpage imu
- @subpage multi_imu
- @subpage gps
- @subpage camera
- @subpage fiducial

### Concepts & Architecture
- @subpage coordinate_systems
- @subpage initialization
- @subpage augmenting_states

### Mathematics
- @subpage jacobians

## System Execution Flow

The following sequence diagram outlines the control flow and interactions between the simulation environment (or ROS node), sensors, and the EKF estimation filter:

@startuml
!include design.puml
@enduml

