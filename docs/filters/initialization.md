Initialization {#initialization}
==============

The initialization of states in ekf_cal consists of three main parts: filter state initialization, stationary gravity alignment, and GPS local-to-global frame transformation initialization.

## Filter State Initialization
At startup, the EKF is initialized using the `EKF::Initialize` method, which sets the initial body state \f$ \boldsymbol{x}_B \f$ and sets the time.
Sensors (IMUs, Cameras, GPS, and Fiducials) register themselves with the EKF. Depending on configuration flags (e.g., `IMU::Parameters::is_extrinsic` or `IMU::Parameters::is_intrinsic`), registering a sensor extends the EKF state vector \f$ \boldsymbol{x} \f$ and pads the covariance matrix \f$ \boldsymbol{P} \f$ with corresponding uncertainty blocks.

## Stationary Gravity Alignment
For correct navigation, the gravity vector must be properly aligned with the local vertical axis. EKF performs stationary initialization using the IMU:

### Zero-Acceleration Constraint:
Before initial motion is detected, the EKF assumes the vehicle is stationary. It calculates a chi-squared score of the IMU residual:

\f[
\text{score} = \boldsymbol{r}^T \left(\boldsymbol{H}\boldsymbol{P}_{k|k-1}\boldsymbol{H}^T + s\boldsymbol{R}\right)^{-1} \boldsymbol{r}
\f]

where the specific force and angular velocity residuals are:

\f{align}{
\boldsymbol{r}_a &= -\left(\boldsymbol{a}_m - \boldsymbol{b}_a - \mathcal{C}(\Quat{I}{B})^T \mathcal{C}(\Quat{B}{L})^T \boldsymbol{g}\right) \\
\boldsymbol{r}_\omega &= -\left(\boldsymbol{\omega}_m - \boldsymbol{b}_\omega\right)
\f}

### Threshold Check:
If the score is less than `EKF::Parameters::motion_detection_chi_squared`, the system is considered stationary. The filter calls `EKF::InitializeGravity` and performs a Kalman update to align the gravity vector with the local frame.

### Initial Motion Detection:
Once the score exceeds the threshold, the EKF switches `ImuUpdater::m_initial_motion_detected` to true. Standard integration of IMU measurements or kinematic predictions is activated from then on.

##  GPS Local-to-Global Frame Initialization
Because GPS measurements are in the global ECEF/geodetic frame and EKF states are in the local frame, a frame transformation must be initialized. This is done online via the Kabsch-Umeyama algorithm.

- **Data Collection**: As the vehicle moves, EKF collects pairs of GPS coordinates (converted to ENU) and corresponding local EKF body position estimates \f$ \PoseHat{B}{L} \f$.
- **Transformation Estimation**: Once at least 4 coordinate pairs are collected, the EKF applies a 2D Kabsch algorithm to find the translation and rotation (heading) between the ENU GPS points and the EKF local path.
- **Initialization Criteria Options**:
  - **BASELINE_DIST**: Succeeds when the maximum distance between GPS points is greater than gps_init_baseline_dist.
  - **ERROR_THRESHOLD**: Succeeds when the standard deviation of position error is below gps_init_pos_thresh and rotation error is below gps_init_ang_thresh.
- **Applying Reference**: Upon success, EKF sets the global reference point \f$ \Pose{E}{G} \f$ and heading offset \f$ \Ang{L}{E} \f$ using `EKF::SetGpsReference`, and enables online extrinsic calibration.
