# math → used for angles and sqrt
import math

# deque → sliding window buffer for pose stability
from collections import deque

# Enum → cleaner way to define robot states instead of strings
from enum import Enum, auto

# numpy → arrays and statistics
import numpy as np

# opencv → vision + aruco detection
import cv2

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
import cmath

import geometry_msgs.msg # Uses geometry_msgs.msg.Twist


# from r2_moverrotate
def euler_from_quaternion(x, y, z, w):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw), in radians.
    For us, yaw is the important one because it tells us robot heading.
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


# enum class created and is used to create a set of named, immutable constants for FSM
class RobotState(Enum):
    #auto() automatically assigns an appropriate value.
    #by default, it starts at 1 for the first member of and increments by 1 for each subsequent member of enum class
    
    SEARCHING_FOR_ID = auto()
    CHECKING_INITIAL_POSE_STABILITY = auto()
    MOVING_FORWARD_FOR_STABLE_POSE = auto()

    DECIDE_DOCKING_STRATEGY = auto()

    # coarse alignment = open-loop sideways correction
    COARSE_ALIGN_ROTATE_1 = auto()
    COARSE_ALIGN_SHIFT = auto()
    COARSE_ALIGN_ROTATE_2 = auto()

    # fine stage = closed-loop centering + docking
    FINE_ALIGN_AND_DOCK = auto()

    DONE = auto()

class ArucoPose(Node):
    def __init__(self):
        super().__init__('elec_test_node')

        # fixed settings 
        self.topic = "/image_raw/compressed"
        self.calib_file = "camera_calib_640x480.npz" # to get camMatrix & distCoeffs for pose estimation
        self.marker_length_m = 0.04
        self.target_id = 23

        self.show_debug = True

        # detection / stability tuning 
        self.required_detection_frames = 5      # N correct frames before we say ID is really found
        self.pose_window_size = 5               # use N last pose readings to check for stability
        self.pos_std_thresh = 0.01              # meters of position variation allowed
        self.angle_std_thresh_deg = 5.0         # degrees of angle variation allowed
        
        # final docking target
        # want marker centered (tx ~ 0) and z ~ 0.40 m
        self.target_z_m = 0.20                  # 40 cm - target docking distance 
        self.target_z_tol_m = 0.01             # 1 cm tolerance for docking distance

        self.target_x_m = 0.0
        self.target_x_tol_m = 0.015

        # coarse alignment settings
        # if |tx| too large, do rough side-shift first

        # decide whether coarse alignment is needed
        self.tx_coarse_thresh_m = 0.02   # if |tx| > 5 cm, do coarse side correctrion before fine docking

        # coarse alignment settings
        self.coarse_shift_scale = 0.7    # shift distance = scale * |tx|
        self.coarse_shift_max_m = 0.20   # cap maximum open-loop shift distance
        self.coarse_forward_speed = 0.04  # forward speed during rough shift
        self.coarse_rotate_speed = 0.3    # angular speed during odom-based 90 turn

        # robot-specific timing estimates
        # tune these on your real robot TO DO
        # fine docking P-controller gains
        self.kp_x = 2.5
        self.kp_z = 0.35
        self.max_linear_speed = 0.05
        self.max_angular_speed = 0.30
        self.align_first_x_thresh_m = 0.03  # if tx is still big, rotate first before moving

        # coarse state memory
        # coarse alignment memory
        # coarse_dir_sign:
        #   +1 -> marker on right -> robot should shift right
        #   -1 -> marker on left  -> robot should shift left
        self.coarse_dir_sign = 0         # +1 means marker on right, -1 means marker on left
        self.coarse_shift_distance_m = 0.0
        self.state_start_time_ns = None

        # odometry state
        # yaw will come from /odom
        self.odom_ready = False
        self.roll_odom = 0.0
        self.pitch_odom = 0.0
        self.yaw_odom = 0.0


        # odom turn state
        self.turn_active = False
        self.turn_target_yaw = 0.0
        self.turn_direction = 0.0
        self.turn_tolerance_deg = 3.0
        
        # load camera intrinsics for pose estimation to be used by solvePnP function
        data = np.load(self.calib_file)
        self.camMatrix = data["camMatrix"]
        self.distCoeffs = data["distCoeffs"]

        # Build 3D model of marker corners
        L = self.marker_length_m # we define the real world coordinates of the marker corners
        self.objPoints = np.array([ # this defines coordinates of a square centered at origin; used by solvePnP
            [-L / 2.0,  L / 2.0, 0.0],
            [ L / 2.0,  L / 2.0, 0.0],
            [ L / 2.0, -L / 2.0, 0.0],
            [-L / 2.0, -L / 2.0, 0.0],
        ], dtype=np.float32)

        # OpenCV  version 4.5.4-safe ArUco detection setup
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50) #4x4 markers, 50 possible ids
        self.params = cv2.aruco.DetectorParameters_create() # detect parameters 
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # corner refinement (improves pose estimation)

        # FSM variables
        self.state = RobotState.SEARCHING_FOR_ID # current state
        self.detect_count = 0
        self.pose_history = deque(maxlen=self.pose_window_size) # stores our recent pose frames

        # ROS subscription to camera topic - calls callback each frame
        self.sub = self.create_subscription(CompressedImage, self.topic, self.cb, 10)


        # odom subscriber added so we can rotate actual 90 degrees using yaw
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )
        
        self.publisher_ = self.create_publisher(geometry_msgs.msg.Twist,'cmd_vel',10)
        self.twist = geometry_msgs.msg.Twist()

            ###Node named mover
                #Publisher on /cmd_vel (topic name is 'cmd_vel')
                #message type = Twist
                #queue depth = 10

        
        self.get_logger().info(
            f"Subscribed to {self.topic}\n"
            f"Subscribed to /odom for yaw-based turning\n"
            f"Loaded calib from {self.calib_file}\n"
            f"Target ID={self.target_id}, marker={self.marker_length_m} m"
        )

    def wrap_angle(self, angle_rad):
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


    def odom_callback(self, msg):
        # Reads robot orientation from /odom and stores yaw in radians.
        # This is what lets us rotate by true angles like 90 degrees
        orientation_quat = msg.pose.pose.orientation
        self.roll_odom, self.pitch_odom, self.yaw_odom = euler_from_quaternion(
            orientation_quat.x,
            orientation_quat.y,
            orientation_quat.z,
            orientation_quat.w
    )
        self.odom_ready = True


    ##### PLACEHOLDERS  functions for movement

    #to check
    
    def publish_cmd(self, linear_x, angular_z):
        #Publish one Twist command
        self.twist.linear.x = float(linear_x)
        self.twist.angular.z = float(angular_z)
        self.publisher_.publish(self.twist)
    
    def stop_motion(self):
        # Stop robot
        global twist
        self.publish_cmd(0.0, 0.0)
        self.get_logger().info("[PLACEHOLDER] STOP robot")
        

    def move_forward_placeholder(self):
        # Small forward move used while trying to get stable pose
        self.publish_cmd(0.02, 0.0)
        self.get_logger().info("[PLACEHOLDER] MOVE FORWARD")

    def rotate_search_placeholder(self):
        #Rotate slowly while searching for marker
        self.publish_cmd(0.0, 0.20)
        self.get_logger().info("[PLACEHOLDER] ROTATE ON SPOT TO SEARCH")
        

    #### our utility functions

    # TIME HELPERS
    # only used for coarse FORWARD shift
    # rotation is now odometry-based

  # ODOMETRY-BASED ROTATION

    def start_odom_turn(self, rot_angle_deg):
        
        # Start a non-blocking odom-based turn.
        # Positive angle = left/CCW
        # Negative angle = right/CW
        
        if not self.odom_ready:
            self.get_logger().warn("Odometry not ready yet. Cannot start odom turn.")
            return False

        self.turn_target_yaw = self.wrap_angle(
            self.yaw_odom + math.radians(rot_angle_deg)
        )

        if rot_angle_deg > 0:
                self.turn_direction = +1.0
        else:
                self.turn_direction = -1.0

        self.turn_active = True

        self.get_logger().info(
                f"[ODOM TURN START] current={math.degrees(self.yaw_odom):.2f} deg, "
                f"target={math.degrees(self.turn_target_yaw):.2f} deg, "
                f"dir={self.turn_direction:+.0f}"
            )
        return True
        
    def update_odom_turn(self):
        """
        Continue turning toward target yaw.
        Returns True when turn is complete, else False.
        """
        if not self.turn_active:
            return True

        yaw_error = self.wrap_angle(self.turn_target_yaw - self.yaw_odom)
        yaw_error_deg = math.degrees(yaw_error)

        self.get_logger().info(
            f"[ODOM TURN] current={math.degrees(self.yaw_odom):.2f} deg, "
            f"target={math.degrees(self.turn_target_yaw):.2f} deg, "
            f"error={yaw_error_deg:.2f} deg"
        )

        if abs(yaw_error_deg) <= self.turn_tolerance_deg:
            self.stop_motion()
            self.turn_active = False
            self.get_logger().info("[ODOM TURN] completed")
            return True

        self.publish_cmd(0.0, self.turn_direction * self.coarse_rotate_speed)
        return False

    def start_timed_state(self):
        self.state_start_time_ns = self.get_clock().now().nanoseconds

    def elapsed_in_state_s(self):
        if self.state_start_time_ns is None:
            return 0.0
        return (self.get_clock().now().nanoseconds - self.state_start_time_ns) / 1e9


    #converts incoming jpeg bytes to numpy array for OPen cv
    def decode_compressed(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame # outputs opencv BGR image
    

    #converts rotation vector to roll, pitch, yaw in radians
    def rvec_to_rpy(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        
        return roll, pitch, yaw
        
    # Returns:
    #     found_target (bool)
    #     pose_ok (bool)
    #     rvec (np.ndarray or None)
    #     tvec (np.ndarray or None)
    #     corners_list, ids
    
    def detect_target_and_pose(self, frame): # explained in ros workspace docs
        # Convert color image to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect all ArUco markers in the image
        # corners_list = image corner points for each detected marker
        # ids = marker IDs matching the same order as corners_list
        # _ = rejected candidates, ignored here
        corners_list, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params
        )

        # if no markers were detected at all, return failure
        if ids is None or len(ids) == 0:
            return False, False, None, None, corners_list, ids

        # convert ids from shape like [[23], [7]] to normal Python list [23, 7]
        ids_flat = ids.flatten().tolist()

        # if our desired target marker ID is not among detected markers, return failure
        if self.target_id not in ids_flat:
            return False, False, None, None, corners_list, ids

        # find which detected marker corresponds to target_id
        idx = ids_flat.index(self.target_id)

        # extract that marker's 4 corner image points
        # reshape to (4,2) because solvePnP expects 4 image points with x,y
        corners = corners_list[idx].reshape(4, 2).astype(np.float32)

        # estimate pose of marker using:
        # - known 3D marker corner coordinates (self.objPoints)
        # - detected 2D image corner coordinates (corners)
        # - camera calibration (camMatrix, distCoeffs)
        ok, rvec, tvec = cv2.solvePnP(
            self.objPoints,
            corners,
            self.camMatrix,
            self.distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # if marker was detected but pose estimation failed
        if not ok:
            return True, False, None, None, corners_list, ids

        # success: target found and pose estimated
        return True, True, rvec, tvec, corners_list, ids
        
    def reset_pose_history(self):
        self.pose_history.clear() #clear dequeu

    def add_pose_sample(self, rvec, tvec):
        roll, pitch, yaw = self.rvec_to_rpy(rvec)
        tx, ty, tz = tvec.flatten().tolist()
        self.pose_history.append([
            tx, ty, tz,
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw)
        ])

    # pose samples stored as - [tx, ty, tz, roll, pitch, yaw]

    
    # func for stability check over recent pose history.
    # it uses std dev of translation and angles.
        
    def is_pose_stable(self): # explained in ros workspace docs
    
        if len(self.pose_history) < self.pose_window_size: # checks if we have enough samples at least
            # each deque entry looks like - [tx, ty, tz, roll, pitch, yaw]
            return False

        arr = np.array(self.pose_history, dtype=np.float32) # convert to numpy array so now its a matrix
        # becomes a shape of (5,6) - Meaning: 5 rows → frames and 6 columns → pose values

        pos_std = np.std(arr[:, 0:3], axis=0)      # tx, ty, tz # computes standard deviation column wise
        ang_std = np.std(arr[:, 3:6], axis=0)      # roll, pitch, yaw # computes standard deviation column wise

        pos_stable = np.all(pos_std < self.pos_std_thresh) # checks std against our thresholds
        ang_stable = np.all(ang_std < self.angle_std_thresh_deg) # checks std against our thresholds

        self.get_logger().info(
            f"Pose stability check | "
            f"pos_std={pos_std} m, ang_std={ang_std} deg | "
            f"stable={pos_stable and ang_stable}"
        )

        return pos_stable and ang_stable
    


    # docking helpers

    def z_reached(self, tvec):
        #Check if marker depth is at target docking distance.
        #tz is marker center forward distance in camera frame.
        tz = float(tvec[2, 0]) # cause tvec column vector stored as matrix so row 2 column 0 
        return abs(tz - self.target_z_m) <= self.target_z_tol_m or tz <= self.target_z_m
    
    def x_centered(self, tvec):
        #Check if marker is centered horizontally in camera frame.
        #tx ~= 0 means marker center is on camera centerline.
        
        tx = float(tvec[0, 0])
        return abs(tx - self.target_x_m) <= self.target_x_tol_m
    

    def docking_goal_reached(self, tvec):
        #Final docking condition: centered and at target z
        return self.z_reached(tvec) and self.x_centered(tvec)

    def clip(self, val, low, high):
        return max(low, min(high, val))

    def compute_fine_docking_command(self, tvec):
        
        # Fine closed-loop docking command.

        # angular_z <- from tx
        # linear_x  <- from tz

        # If tx still large, rotate first before moving forward.
        
        tx = float(tvec[0, 0])
        tz = float(tvec[2, 0])

        x_err = tx - self.target_x_m
        z_err = tz - self.target_z_m

        # If marker is on right (tx > 0), robot should usually rotate right.
        # If your robot turns wrong direction, flip this sign.
        angular_z = -self.kp_x * x_err
        angular_z = self.clip(angular_z, -self.max_angular_speed, self.max_angular_speed)

        # If marker still too off-center, rotate first and do not move forward yet
        if abs(x_err) > self.align_first_x_thresh_m:
            linear_x = 0.0
        else:
            if z_err > 0.0:
                linear_x = self.kp_z * z_err
                linear_x = self.clip(linear_x, 0.0, self.max_linear_speed)
            else:
                linear_x = 0.0

        return linear_x, angular_z, tx, tz

    def prepare_coarse_alignment(self, tvec):
        
        # Decide side-shift direction and shift amount from tx.

        # If tx > 0: marker is on right -> shift robot right
        # If tx < 0: marker is on left  -> shift robot left
        
        tx = float(tvec[0, 0])

        if tx > 0.0:
            self.coarse_dir_sign = +1
        else:
            self.coarse_dir_sign = -1

        self.coarse_shift_distance_m = min(
            self.coarse_shift_scale * abs(tx),
            self.coarse_shift_max_m
        )

        self.get_logger().info(
            f"Preparing coarse alignment | "
            f"tx={tx:+.3f} m | "
            f"dir_sign={self.coarse_dir_sign:+d} | "
            f"shift_distance={self.coarse_shift_distance_m:.3f} m"
        )

    def coarse_shift_time_s(self):
        
        # Convert desired coarse shift distance into how long to drive forward.
        # We still do this by time because we only switched rotation to odom-based.
        
        if self.coarse_forward_speed <= 1e-6:
            return 0.0
        return self.coarse_shift_distance_m / self.coarse_forward_speed


    # draw for debugging, detected marker, pose axes, and state text
    def draw_debug(self, frame, corners_list, ids, rvec=None, tvec=None, text=None):
        if not self.show_debug:
            return
        
        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners_list, ids)

        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(
                vis,
                self.camMatrix,
                self.distCoeffs,
                rvec,
                tvec,
                self.marker_length_m * 1.5,
                2
            )

        if text is not None:
            cv2.putText(
                vis,
                text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("aruco_pose", vis)
        cv2.waitKey(1)

        ###### MAIN CALLBACK FUNC #####

    def cb(self, msg: CompressedImage):
        frame = self.decode_compressed(msg)
        if frame is None:
            self.get_logger().warn("Decode failed (frame is None). Skipping.")
            return

        found_target, pose_ok, rvec, tvec, corners_list, ids = self.detect_target_and_pose(frame)

        # ####### STATE: SEARCHING_FOR_ID #######
        if self.state == RobotState.SEARCHING_FOR_ID:
            if found_target:
                self.detect_count += 1
                self.get_logger().info(
                    f"SEARCHING_FOR_ID: target seen "
                    f"({self.detect_count}/{self.required_detection_frames})"
                )
                self.stop_motion() # to check

                if self.detect_count >= self.required_detection_frames:
                    self.get_logger().info("Target ID found consistently. Checking pose stability.")
                    self.state = RobotState.CHECKING_INITIAL_POSE_STABILITY
                    self.reset_pose_history()
            else:
                self.detect_count = 0
                self.rotate_search_placeholder() # to check

            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return
        
        # ####### STATE: CHECKING_INITIAL_POSE_STABILITY #######
         # Stay still and make sure pose estimates are stable
        if self.state == RobotState.CHECKING_INITIAL_POSE_STABILITY:
            self.stop_motion()

            if not found_target or not pose_ok:
                self.get_logger().warn("Lost target or pose during initial stability check. Searching again.")
                self.detect_count = 0
                self.reset_pose_history()
                self.state = RobotState.SEARCHING_FOR_ID
                self.draw_debug(frame, corners_list, ids, text=f"STATE: {self.state.name}")
                return

            self.add_pose_sample(rvec, tvec)

            tx, ty, tz = tvec.flatten().tolist()
            roll, pitch, yaw = self.rvec_to_rpy(rvec)
            self.get_logger().info(
                f"CHECKING_INITIAL_POSE_STABILITY | "
                f"tvec[m]=({tx:+.3f},{ty:+.3f},{tz:+.3f}) "
                f"RPY[deg]=({math.degrees(roll):+.1f},{math.degrees(pitch):+.1f},{math.degrees(yaw):+.1f})"
            )

            if self.is_pose_stable():
                self.get_logger().info("Initial pose is stable. Deciding docking strategy.")
                self.state = RobotState.DECIDE_DOCKING_STRATEGY
            else:
                self.get_logger().info("Pose not stable yet. Move forward and keep trying.")
                self.state = RobotState.MOVING_FORWARD_FOR_STABLE_POSE

            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return
    
    # ###### STATE: MOVING_FORWARD_FOR_STABLE_POSE ######
    #Small forward motion to try to get a cleaner view / stable pose
        if self.state == RobotState.MOVING_FORWARD_FOR_STABLE_POSE:
            if not found_target or not pose_ok:
                self.get_logger().warn("Lost target while moving for stable pose. Stop and rotate to reacquire.")
                self.stop_motion()
                self.detect_count = 0
                self.reset_pose_history()
                self.state = RobotState.SEARCHING_FOR_ID
                self.draw_debug(frame, corners_list, ids, text=f"STATE: {self.state.name}")
                return
            
            self.move_forward_placeholder()
            self.add_pose_sample(rvec, tvec)

            if self.is_pose_stable():
                self.get_logger().info("Stable pose obtained while moving forward. Deciding docking strategy.")
                self.stop_motion()
                self.state = RobotState.DECIDE_DOCKING_STRATEGY

            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return
            
        
        # STATE: DECIDE_DOCKING_STRATEGY
        # If tx small -> fine dock directly
        # If tx large -> do rough side-shift first
        
        if self.state == RobotState.DECIDE_DOCKING_STRATEGY:
            self.stop_motion()

            if not found_target or not pose_ok:
                self.get_logger().warn("Lost target before docking strategy decision. Searching again.")
                self.detect_count = 0
                self.reset_pose_history()
                self.state = RobotState.SEARCHING_FOR_ID
                self.draw_debug(frame, corners_list, ids, text=f"STATE: {self.state.name}")
                return

            tx = float(tvec[0, 0])
            tz = float(tvec[2, 0])

            self.get_logger().info(
                f"DECIDE_DOCKING_STRATEGY | tx={tx:+.3f} m, tz={tz:+.3f} m"
            )

            if abs(tx) <= self.tx_coarse_thresh_m:
                self.get_logger().info(
                    "tx is already small. Skip coarse alignment and go straight to fine docking."
                )
                self.state = RobotState.FINE_ALIGN_AND_DOCK
            else:
                # prepare rough side correction
                self.prepare_coarse_alignment(tvec)
                self.state = RobotState.COARSE_ALIGN_ROTATE_1
                self.get_logger().info("tx is large. Starting coarse side-shift alignment.")

            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return
        

        # STATE: COARSE_ALIGN_ROTATE_1
        # Rotate 90 deg toward side we want to shift
        # Uses odometry yaw, not timing
        # --------------------------------------------------------
        if self.state == RobotState.COARSE_ALIGN_ROTATE_1:
            if not self.odom_ready:
                self.get_logger().warn("Odometry not ready. Cannot do coarse rotate 1 yet.")
                self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
                return

            # Start turn only once
            if not self.turn_active:
                if self.coarse_dir_sign > 0:
                    self.get_logger().info("Coarse rotate 1: marker on right -> rotate right 90 deg.")
                    ok = self.start_odom_turn(-90.0)
                else:
                    self.get_logger().info("Coarse rotate 1: marker on left -> rotate left 90 deg.")
                    ok = self.start_odom_turn(+90.0)

                if not ok:
                    self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
                    return

            # Keep updating turn until done
            if self.update_odom_turn():
                self.start_timed_state()
                self.state = RobotState.COARSE_ALIGN_SHIFT

            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return
        

        # STATE: COARSE_ALIGN_SHIFT
        # Move forward to approximate lateral shift
        # still open-loop and time-based
        # --------------------------------------------------------
        if self.state == RobotState.COARSE_ALIGN_SHIFT:
            elapsed = self.elapsed_in_state_s()
            shift_time = self.coarse_shift_time_s()

            self.publish_cmd(self.coarse_forward_speed, 0.0)

            if elapsed >= shift_time:
                self.stop_motion()
                self.state = RobotState.COARSE_ALIGN_ROTATE_2
                self.get_logger().info("Finished coarse shift. Starting coarse rotate 2.")

            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return
        
        # --------------------------------------------------------
        # STATE: COARSE_ALIGN_ROTATE_2
        # Rotate back so robot faces marker direction again
        # --------------------------------------------------------
        if self.state == RobotState.COARSE_ALIGN_ROTATE_2:
            if not self.odom_ready:
                self.get_logger().warn("Odometry not ready. Cannot do coarse rotate 2 yet.")
                self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
                return

            # Start turn only once
            if not self.turn_active:
                if self.coarse_dir_sign > 0:
                    self.get_logger().info("Coarse rotate 2: rotate left 90 deg to face marker again.")
                    ok = self.start_odom_turn(+90.0)
                else:
                    self.get_logger().info("Coarse rotate 2: rotate right 90 deg to face marker again.")
                    ok = self.start_odom_turn(-90.0)

                if not ok:
                    self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
                    return

            # Keep updating turn until done
            if self.update_odom_turn():
                self.reset_pose_history()
                self.state = RobotState.CHECKING_INITIAL_POSE_STABILITY
                self.get_logger().info("Finished coarse rotate 2. Reacquire stable pose, then fine docking.")

            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return


        # --------------------------------------------------------
        # STATE: FINE_ALIGN_AND_DOCK
        # Fine closed-loop stage:
        #  - center marker (tx -> 0)
        #  - stop at tz = 0.40 m
        # --------------------------------------------------------
        if self.state == RobotState.FINE_ALIGN_AND_DOCK:
            if not found_target or not pose_ok:
                self.get_logger().warn("Lost target during fine docking. Stop and search again.")
                self.stop_motion()
                self.detect_count = 0
                self.reset_pose_history()
                self.state = RobotState.SEARCHING_FOR_ID
                self.draw_debug(frame, corners_list, ids, text=f"STATE: {self.state.name}")
                return

            linear_x, angular_z, tx, tz = self.compute_fine_docking_command(tvec)

            self.get_logger().info(
                f"FINE_ALIGN_AND_DOCK | "
                f"tx={tx:+.3f} m, tz={tz:+.3f} m | "
                f"cmd.linear.x={linear_x:+.3f}, cmd.angular.z={angular_z:+.3f}"
            )

            if self.docking_goal_reached(tvec):
                self.get_logger().info("Docking goal reached: centered and at target z.")
                self.stop_motion()
                self.state = RobotState.DONE
            else:
                self.publish_cmd(linear_x, angular_z)

            self.draw_debug(
                frame,
                corners_list,
                ids,
                rvec,
                tvec,
                text=f"STATE: {self.state.name} | tx={tx:+.3f} tz={tz:+.3f}"
            )
            return


            # ######## STATE: DONE #########
        if self.state == RobotState.DONE:
            self.stop_motion()
            self.get_logger().info("DONE: Marker reached at target z and centered in frame.")
            self.draw_debug(frame, corners_list, ids, rvec, tvec, text=f"STATE: {self.state.name}")
            return


def main():
    rclpy.init()
    node = ArucoPose()
    try:
        rclpy.spin(node)
    finally:
        if node.show_debug:
            cv2.destroyAllWindows()
        #node.destroy_node()
    rclpy.shutdown()

 
if __name__ == "__main__":
    main()

