Coordinate Systems {#coordinate_systems}
==================

`ekf_cal` operates on multiple coordinate systems and reference frames to perform calibration and estimation. A solid understanding of these frames and their relationships is crucial for configuring and using the filter.

## Reference Frames

The filter utilizes the following coordinate frames:

- <b>Local Frame (\f$L\f$)</b>: The gravity-aligned local reference frame in which the vehicle's navigation is estimated. All body states (position, velocity, acceleration, and orientation) are defined relative to this frame.
- <b>Body Frame (\f$B\f$)</b>: The vehicle-fixed primary reference frame. Typically, the body frame is aligned with the primary IMU. The extrinsics of all other sensors are estimated relative to this frame.
- <b>IMU Frame (\f$I_i\f$)</b>: The coordinate frame of the \f$i\f$-th IMU sensor. Accelerations and angular rates are measured in this frame. The extrinsic parameters are the IMU's position offset \f$ \Pose{I_i}{B} \f$ and orientation \f$ \Quat{I_i}{B} \f$.
- <b>Camera Frame (\f$C_c\f$)</b>: The coordinate frame of the \f$c\f$-th camera. Its origin is at the camera optical center, with the z-axis pointing along the optical axis. Extrinsic parameters are \f$ \Pose{C_c}{B} \f$ and \f$ \Quat{C_c}{B} \f$.
- <b>GPS Antenna Frame (\f$A_g\f$)</b>: The coordinate frame of the \f$g\f$-th GPS antenna phase center. Its position offset relative to the body frame is \f$ \Pose{A_g}{B} \f$.
- <b>Fiducial Frame (\f$F_f\f$)</b>: The coordinate frame of the \f$f\f$-th stationary fiducial marker board. Its position \f$ \Pose{F_f}{L} \f$ and orientation \f$ \Quat{F_f}{L} \f$ are estimated in the local frame.
- <b>ENU Frame (\f$E\f$)</b>: The East-North-Up (ENU) frame, once initialized, allows transformations between global geodetic frames such as LLA or ECEF from global GPS measurements to the local frame \f$L\f$.

## Transform Relationships

The transformations between coordinate frames are defined using translation vectors and quaternions/rotation matrices.

### Body to Local Frame
The body state is represented by its position in the local frame \f$ \Pose{B}{L} \f$ and orientation quaternion \f$ \Quat{B}{L} \f$. The coordinate transformation of a point \f$ \Pose{}{B} \f$ in the body frame to the local frame is:

\f[
\Pose{}{L} = \QuatRot{\Quat{B}{L}} \Pose{}{B} + \Pose{B}{L}
\f]

### Sensor Extrinsics (Sensor to Body)
Sensor extrinsics represent the transformation from a sensor-fixed frame to the body frame. For a generic sensor frame \f$ S \f$ (such as IMU or Camera), the position of a point in the body frame is:

\f[
\Pose{}{B} = \QuatRot{\Quat{S}{B}} \Pose{}{S} + \Pose{S}{B}
\f]

Where:
- \f$ \Pose{S}{B} \f$ is the position of the sensor in the body frame.
- \f$ \Quat{S}{B} \f$ is the rotation from the sensor frame to the body frame.

### Local to Global (GPS Frame Initialization)
GPS measurements are received in geodetic coordinates (Latitude, Longitude, Altitude) which can be converted to ECEF coordinates and back deterministically. The origin of the local frame \f$ L \f$ is aligned with the local East-North-Up (ENU) frame at a global reference point \f$ \Pose{E}{G} \f$, and the z coordinate of the local frame is aligned with the up coordinate of the ENU frame.

Finally, the rotation between the ENU local frame and ENU global frame is parameterized by the local heading offset \f$ \Ang{L}{E} \f$ to account for an unknown initial local heading. The transformation from local ENU coordinates \f$ \Pose{}{L} \f$ to global ECEF coordinates \f$ \Pose{}{E} \f$ is:

\f[
\Pose{}{E} = \QuatRot{\Ang{L}{E}} \Pose{}{L} + \Pose{E}{G}
\f]
