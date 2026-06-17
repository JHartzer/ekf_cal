Multi-IMU Filter {#multi_imu}
================

Traditional multi-IMU sensor fusion approaches often require tight time synchronization of the IMUs, or interpolate/average the measurements to construct a single "virtual" IMU. However, this has significant limitations when combining IMUs with different update rates or varying noise characteristics.

`ekf_cal` implements an _update-only Multi-IMU filter_ (see reference). Instead of using a single IMU to propagate the EKF state, the EKF state vector is extended to include body-fixed kinematics, and the measurements from all registered IMUs are treated as EKF measurement updates.

## Kinematic State Vector & Propagation

The EKF state vector includes body-fixed motion variables in the local frame \f$ L \f$:

\f[
\boldsymbol{x}_B =
\begin{bmatrix}
    \Pose{B}{L} &
    \Vel{B}{L} &
    \Acc{B}{L} &
    \Quat{B}{L} &
    \AngVel{B}{L} &
    \AngAcc{B}{L}
\end{bmatrix}
\f]

### pagation Kinematics
When multiple IMUs are registered, the EKF propagates the body states forward in time using a kinematic model:

\f{align}{
\Pose{B_k}{L} &= \Pose{B_{k-1}}{L} + \Delta t \Vel{B_{k-1}}{L} \\
\Vel{B_k}{L} &= \Vel{B_{k-1}}{L} + \Delta t \left(\Acc{B_{k-1}}{L} - \boldsymbol{g}\right) \\
\Quat{B_k}{L} &= \text{exp}\left(\AngVel{B_{k-1}}{L} \Delta t\right) \Quat{B_{k-1}}{L} \\
\AngVel{B_k}{L} &= \AngVel{B_{k-1}}{L} + \Delta t \AngAcc{B_{k-1}}{L}
\f}

Where \f$ \boldsymbol{g} \f$ is the gravity vector. The linear acceleration and angular acceleration are propagated via random walk processes driven by process noise.

### ariance Transition
The state transition matrix \f$ \boldsymbol{F} \f$ propagates body error states over \f$ \Delta t \f$:

\f[
\boldsymbol{F} =
\begin{bmatrix}
    \Eye{3} & \Delta t \Eye{3} & \Delta t^2 \Eye{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} \\
    \Zeros{3}{3} & \Eye{3} & \Delta t \Eye{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} \\
    \Zeros{3}{3} & \Zeros{3}{3} & \Eye{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} \\
    \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Eye{3} & \Delta t \QuatRot{\Quat{B}{L}}^T & \Delta t^2 \QuatRot{\Quat{B}{L}}^T \\
    \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Eye{3} & \Delta t \Eye{3} \\
    \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Zeros{3}{3} & \Eye{3}
\end{bmatrix}
\f]

This is implemented in `EKF::GetStateTransition` in [ekf.cpp](file:///home/jacob/proj/ekf_cal_ws/src/ekf_cal/src/ekf/ekf.cpp).

## IMU Measurement Kalman Update

For each IMU measurement, the EKF performs a standard Kalman update to correct the body kinematics and refine the sensor's extrinsic and intrinsic parameters.

### dicted Measurement
The predicted accelerometer and gyroscope measurements (rotated into the IMU frame \f$ I_i \f$) are:

\f{align}{
\hat{\boldsymbol{a}}_{imu} &= \Bias{a} + \QuatRot{\Quat{I_i}{B}}^T \left( \Acc{B}{L} + \AngAcc{B}{L} \times \Pose{I_i}{B} + \AngVel{B}{L} \times (\AngVel{B}{L} \times \Pose{I_i}{B}) + 2 \AngVel{B}{L} \times \Vel{B}{L} \right) \\
\hat{\boldsymbol{\omega}}_{imu} &= \Bias{\omega} + \QuatRot{\Quat{I_i}{B}}^T \QuatRot{\Quat{B}{L}}^T \AngVel{B}{L}
\f}

Where:
- \f$ \Pose{I_i}{B} \f$ and \f$ \Quat{I_i}{B} \f$ are the IMU extrinsic position offset and orientation.
- \f$ \Bias{a} \f$ and \f$ \Bias{\omega} \f$ are the accelerometer and gyroscope intrinsic biases.

The measurement residual is \f$ \boldsymbol{r} = \boldsymbol{z} - \hat{\boldsymbol{z}} \f$, and the state is updated using the IMU observation Jacobians detailed on the [Jacobians](@ref jacobians) page.

## Benefits of the Update-Only Formulation

1. **Different Update Rates**: The EKF processes measurements asynchronously as they arrive. IMUs with different frequencies are handled naturally without downsampling.
2. **Online Calibration**: The extrinsics (\f$ \Pose{I_i}{B} \f$, \f$ \Quat{I_i}{B} \f$) and intrinsics (\f$ \Bias{a} \f$, \f$ \Bias{\omega} \f$) of all secondary IMUs are calibrated online.
3. **No Synchronization Required**: The asynchronous prediction-correction loop naturally aligns measurements to their exact execution times, minimizing synchronization errors.

## Reference

```bibtex
@inproceedings{2023_Multi_IMU,
  title         = {Online Multi-IMU Calibration Using Visual-Inertial Odometry},
  booktitle     = {2023 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI)},
  author        = {Jacob Hartzer and Srikanth Saripalli},
  year          = {2023},
  doi           = {10.1109/SDF-MFI59545.2023.10361310},
  preview       = {multi_imu.png},
  arxiv         = {2310.12411},
}
```
