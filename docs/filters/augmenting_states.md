Augmenting States {#augmenting_states}
=================

State augmentation is a core mechanism of the Multi-State Constraint Kalman Filter (MSCKF) architecture in `ekf_cal`. It allows the filter to process visual feature measurements from cameras without maintaining 3D landmark coordinates in the EKF state vector.

Instead, the EKF maintains a sliding window of historical body poses (position and orientation) at the timestamps of camera frame captures. These historical poses are called **augmented states**.

## Augmented State Structure

An augmented state \f$ \boldsymbol{x}_{aug, k} \f$ corresponding to the camera frame at time \f$ t_k \f$ contains:

\f[
\boldsymbol{x}_{aug, k} =
\begin{bmatrix}
    \Pose{B_k}{L} \\
    \Quat{B_k}{L}
\end{bmatrix}
\f]

Where:
- \f$ \Pose{B_k}{L} \f$ is the position of the body frame in the local frame at time \f$ t_k \f$.
- \f$ \Quat{B_k}{L} \f$ is the orientation of the body frame in the local frame at time \f$ t_k \f$.

Each augmented state has a size of 6 in the error state vector.

## Covariance Cloning

When state augmentation is triggered, the EKF error covariance matrix \f$ \boldsymbol{P} \f$ is cloned to capture the cross-covariance between the current body state and the new augmented state.

Let the Jacobian of the new augmented state with respect to the current state vector be \f$ \boldsymbol{J}_{aug} \f$:

\f[
\boldsymbol{J}_{aug} =
\begin{bmatrix}
    \boldsymbol{I}_3 & \boldsymbol{0} & \boldsymbol{0} & \boldsymbol{0} & \boldsymbol{0} & \boldsymbol{0} & \dots \\
    \boldsymbol{0} & \boldsymbol{0} & \boldsymbol{0} & \boldsymbol{I}_3 & \boldsymbol{0} & \boldsymbol{0} & \dots
\end{bmatrix}
\f]

This maps the current body position error \f$ \PoseTilde{B}{L} \f$ and orientation error \f$ \tilde{\boldsymbol{\theta}}_B^L \f$ to the new augmented state block. The covariance matrix is then augmented as:

\f[
\boldsymbol{P}_{new} =
\begin{bmatrix}
    \boldsymbol{P} & \boldsymbol{P} \boldsymbol{J}_{aug}^T \\
    \boldsymbol{J}_{aug} \boldsymbol{P} & \boldsymbol{J}_{aug} \boldsymbol{P} \boldsymbol{J}_{aug}^T
\end{bmatrix}
\f]

This is implemented in `EKF::AugmentCovariance`.

## Triggering State Augmentation

The EKF supports several modes of augmentation, configured by the `EKF::Parameters::augmenting_type` parameter:

1. **NONE**: No camera frames are augmented. Typically used for calibration without visual odometry (e.g., fiducial-only calibration).
2. **ALL**: Every camera frame triggers state augmentation.
3. **PRIMARY**: Only frames from the primary camera trigger augmentation.
4. **TIME**: A new state is augmented if the time elapsed since the last augmentation exceeds `EKF::Parameters::augmenting_delta_time`.
5. **ERROR**: A new state is augmented if the vehicle has moved significantly (linear displacement \f$ ||\Delta \Pose{}{}|| > \text{augmenting_pos_error} \f$ or angular rotation \f$ ||\Delta \boldsymbol{\theta}|| > \text{augmenting_ang_error} \f$) since the last augmented frame, or if `EKF::m_max_track_duration` is exceeded.

## State Pruning

To keep the computation bounded, augmented states are managed in a sliding window. Once all features tracked in a historical frame are lost or fully processed, or if the sliding window size exceeds `EKF::m_max_track_length`, the oldest augmented states are pruned using the following process:
- The corresponding pose is removed from the state vector.
- The corresponding rows and columns are deleted from the covariance matrix \f$ \boldsymbol{P} \f$ using `RemoveFromMatrix()`.
- The indices of remaining states are shifted and updated.
