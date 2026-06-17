Jacobians {#jacobians}
=========

Measurement updates in the Extended Kalman Filter (EKF) require linearizing non-linear measurement models. This page details the measurement Jacobians utilized in `ekf_cal` for each sensor type.

In the EKF, the measurement error residual is linearized as:

\f[
\tilde{\boldsymbol{z}} \approx \boldsymbol{H} \tilde{\boldsymbol{x}} + \boldsymbol{v}
\f]

Where \f$ \boldsymbol{H} \f$ is the measurement Jacobian and \f$ \tilde{\boldsymbol{x}} \f$ is the error state vector.

## GPS Measurement Jacobians

The GPS predicted measurement (in ENU) for antenna \f$ g \f$ is:

\f[
\hat{\boldsymbol{z}} = \Pose{B}{L} + \QuatRot{\Quat{B}{L}} \Pose{A_g}{B}
\f]

The Jacobian \f$ \boldsymbol{H}_{gps} \f$ is computed with respect to the body state and GPS extrinsic state:

- <b>Body Position Error \f$ \PoseTilde{B}{L} \f$</b>:
  \f[
  \boldsymbol{H}_{\Pose{B}{L}} = \QuatRot{\Ang{L}{E}}
  \f]

- <b>Body Orientation Error \f$ \tilde{\boldsymbol{\theta}}_B^L \f$</b>:
  \f[
  \boldsymbol{H}_{\tilde{\boldsymbol{\theta}}_B^L} = -\QuatRot{\Quat{B}{L}} \CrossMat{\Pose{A_g}{B}} \JacR{\Quat{B}{L}}
  \f]

- <b>GPS Antenna Extrinsic Offset \f$ \PoseTilde{A_g}{B} \f$</b>:
  \f[
  \boldsymbol{H}_{\Pose{A_g}{B}} = \QuatRot{\Ang{L}{E}} \QuatRot{\Quat{B}{L}}
  \f]

## IMU Measurement Jacobians

In Multi-IMU mode, the body acceleration \f$ \Acc{B}{L} \f$, angular velocity \f$ \AngVel{B}{L} \f$, and angular acceleration \f$ \AngAcc{B}{L} \f$ are EKF states. The IMU specific force measurements \f$ \boldsymbol{a}_{imu} \f$ and angular velocity measurements \f$ \boldsymbol{\omega}_{imu} \f$ are treated as EKF updates. The Jacobians with respect to these measurement blocks are:

- <b>Body Acceleration \f$ \Acc{B}{L} \f$</b>:
  \f[
  \boldsymbol{H}_{\boldsymbol{a}, \Acc{B}{L}} = \QuatRot{\Quat{I_i}{B}}^T \QuatRot{\Quat{B}{L}}^T, \quad \boldsymbol{H}_{\boldsymbol{\omega}, \Acc{B}{L}} = \Zero
  \f]

- <b>Body Angular Velocity \f$ \AngVel{B}{L} \f$</b>:
  \f[
  \boldsymbol{H}_{\boldsymbol{a}, \AngVel{B}{L}} = \QuatRot{\Quat{I_i}{B}}^T \left( \CrossMat{\AngVel{B}{L}} \CrossMat{\Pose{I_i}{B}}^T + \CrossMat{\AngVel{B}{L} \times \Pose{I_i}{B}}^T \right), \quad \boldsymbol{H}_{\boldsymbol{\omega}, \AngVel{B}{L}} = \QuatRot{\Quat{I_i}{B}}^T \QuatRot{\Quat{B}{L}}^T
  \f]

- <b>Body Angular Acceleration \f$ \AngAcc{B}{L} \f$</b>:
  \f[
  \boldsymbol{H}_{\boldsymbol{a}, \AngAcc{B}{L}} = -\QuatRot{\Quat{I_i}{B}}^T \CrossMat{\Pose{I_i}{B}}, \quad \boldsymbol{H}_{\boldsymbol{\omega}, \AngAcc{B}{L}} = \Zero
  \f]

- <b>IMU Extrinsic Position Offset \f$ \Pose{I_i}{B} \f$</b>:
  \f[
  \boldsymbol{H}_{\boldsymbol{a}, \Pose{I_i}{B}} = \QuatRot{\Quat{I_i}{B}}^T \left( \CrossMat{\AngAcc{B}{L}} + \CrossMat{\AngVel{B}{L}} \CrossMat{\AngVel{B}{L}} \right), \quad \boldsymbol{H}_{\boldsymbol{\omega}, \Pose{I_i}{B}} = \Zero
  \f]

- <b>IMU Extrinsic Orientation Offset \f$ \Quat{I_i}{B} \f$</b>:
  \f[
  \boldsymbol{H}_{\boldsymbol{a}, \Quat{I_i}{B}} = \CrossMat{\QuatRot{\Quat{I_i}{B}}^T \left( \AngAcc{B}{L} \times \Pose{I_i}{B} + \AngVel{B}{L} \times (\AngVel{B}{L} \times \Pose{I_i}{B}) + \QuatRot{\Quat{B}{L}}^T \Acc{B}{L} \right)}, \quad \boldsymbol{H}_{\boldsymbol{\omega}, \Quat{I_i}{B}} = \CrossMat{\QuatRot{\Quat{I_i}{B}}^T \QuatRot{\Quat{B}{L}}^T \AngVel{B}{L}}
  \f]

- <b>IMU Intrinsic Biases (Accel \f$ \boldsymbol{b}_a \f$ and Gyro \f$ \boldsymbol{b}_\omega \f$)</b>:
  \f[
  \boldsymbol{H}_{\boldsymbol{a}, \Bias{a}} = \Eye{3}, \quad \boldsymbol{H}_{\boldsymbol{\omega}, \Bias{\omega}} = \Eye{3}
  \f]

## MSCKF (Camera) Measurement Jacobians

The camera measurement model projects a 3D landmark \f$ \Pose{F}{L} \f$ onto the image plane of augmented camera pose \f$ i \f$. The pixel coordinate Jacobian is decomposed using the chain rule:

\f[
\boldsymbol{H} = \boldsymbol{H}_d \boldsymbol{H}_p \boldsymbol{H}_{geo}
\f]

Where:
- \f$ \boldsymbol{H}_d \f$ is the lens distortion Jacobian (w.r.t. normalized coordinates).
- \f$ \boldsymbol{H}_p \f$ is the perspective projection Jacobian.
- \f$ \boldsymbol{H}_{geo} \f$ is the geometry Jacobian (w.r.t. EKF states).

The geometry Jacobians \f$ \boldsymbol{H}_{geo} \f$ are:

- <b>Augmented Body Position \f$ \Pose{B_i}{L} \f$</b>:
  \f[
  \boldsymbol{H}_{\Pose{B_i}{L}} = -\QuatRot{\Quat{L}{C_i}}
  \f]

- <b>Augmented Body Orientation \f$ \Quat{B_i}{L} \f$</b>:
  \f[
  \boldsymbol{H}_{\Quat{B_i}{L}} = \QuatRot{\Quat{B}{C_i}} \CrossMat{\QuatRot{\Quat{B_i}{L}}^T (\Pose{F}{L} - \Pose{B_i}{L})}
  \f]

- <b>Camera Extrinsic Offset \f$ \Pose{C}{B} \f$</b>:
  \f[
  \boldsymbol{H}_{\Pose{C}{B}} = -\QuatRot{\Quat{B}{C_i}}
  \f]

- <b>Camera Extrinsic Orientation \f$ \Quat{C}{B} \f$</b>:
  \f[
  \boldsymbol{H}_{\Quat{C}{B}} = \CrossMat{\Pose{F}{C_i}}
  \f]

## Fiducial Measurement Jacobians

The fiducial measurement models the relative pose (position and orientation) of a stationary tag in the camera frame. The Jacobians are:

- <b>Body Position Error \f$ \PoseTilde{B}{L} \f$</b>:
  \f[
  \boldsymbol{H}_{pos, \Pose{B}{L}} = -\QuatRot{\Quat{L}{C}}
  \f]

- <b>Body Orientation Error \f$ \tilde{\boldsymbol{\theta}}_B^L \f$</b>:
  \f{align}{
  \boldsymbol{H}_{pos, \tilde{\boldsymbol{\theta}}_B^L} &= \QuatRot{\Quat{L}{C}} \CrossMat{\Pose{F}{L} - \Pose{B}{L}} \JacR{\Quat{B}{L}}^T \\
  \boldsymbol{H}_{rot, \tilde{\boldsymbol{\theta}}_B^L} &= -\QuatRot{\Quat{L}{C}} \QuatRot{\Quat{F}{L}} \JacR{\Quat{B}{L}}^T
  \f}

- <b>Camera Extrinsic Position Offset \f$ \PoseTilde{C}{B} \f$</b>:
  \f[
  \boldsymbol{H}_{pos, \Pose{C}{B}} = -\QuatRot{\Quat{B}{C}}
  \f]

- <b>Camera Extrinsic Orientation Offset \f$ \tilde{\boldsymbol{\theta}}_C^B \f$</b>:
  \f{align}{
  \boldsymbol{H}_{pos, \tilde{\boldsymbol{\theta}}_C^B} &= \QuatRot{\Quat{B}{C}} \CrossMat{\QuatRot{\Quat{B}{L}}^T (\Pose{F}{L} - \Pose{B}{L}) - \Pose{C}{B}} \JacR{\Quat{C}{B}}^T \\
  \boldsymbol{H}_{rot, \tilde{\boldsymbol{\theta}}_C^B} &= -\QuatRot{\Quat{B}{C}} \JacR{\Quat{C}{B}}^T \QuatRot{\Quat{B}{L}}^T \QuatRot{\Quat{F}{L}}
  \f}

- <b>Fiducial Extrinsic Position \f$ \PoseTilde{F}{L} \f$ and Orientation \f$ \tilde{\boldsymbol{\theta}}_F^L \f$</b>:
  \f[
  \boldsymbol{H}_{pos, \Pose{F}{L}} = \Eye{3}, \quad \boldsymbol{H}_{rot, \tilde{\boldsymbol{\theta}}_F^L} = \Eye{3}
  \f]
