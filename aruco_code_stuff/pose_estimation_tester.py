#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import math


def rvec_to_euler_xyz(rvec: np.ndarray):
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


class ArucoPose(Node):
    def __init__(self):
        super().__init__('aruco_pose_compressed')

        self.topic = "/image_raw/compressed"
        self.calib_file = "camera_calib_640x480.npz"
        self.marker_length_m = 0.04
        self.target_id = 23
        self.dict_id = cv2.aruco.DICT_4X4_50

        data = np.load(self.calib_file)
        self.camMatrix = data['camMatrix']
        self.distCoeffs = data['distCoeffs']

        L = self.marker_length_m
        self.objPoints = np.array(
            [
                [-L / 2.0,  L / 2.0, 0.0],  # top-left
                [ L / 2.0,  L / 2.0, 0.0],  # top-right
                [ L / 2.0, -L / 2.0, 0.0],  # bottom-right
                [-L / 2.0, -L / 2.0, 0.0],  # bottom-left
            ],
            dtype=np.float32
        )

        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dict_id)
        self.params = cv2.aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.sub = self.create_subscription(
            CompressedImage,
            self.topic,
            self.cb,
            10
        )

        self.get_logger().info(
            f"Pose node listening on: {self.topic}\n"
            f"Target ID: {self.target_id}, marker_length={self.marker_length_m} m\n"
            f"Calibration: {self.calib_file}"
        )

    def decode_compressed(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

    def cb(self, msg: CompressedImage):
        frame = self.decode_compressed(msg)
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_list, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.params
        )

        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners_list, ids)

            ids_flat = ids.flatten().tolist()

            if self.target_id in ids_flat:
                idx = ids_flat.index(self.target_id)
                corners = corners_list[idx].reshape(4, 2).astype(np.float32)

                ok, rvec, tvec = cv2.solvePnP(
                    self.objPoints,
                    corners,
                    self.camMatrix,
                    self.distCoeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if ok:
                    cv2.drawFrameAxes(
                        vis,
                        self.camMatrix,
                        self.distCoeffs,
                        rvec,
                        tvec,
                        self.marker_length_m * 1.5,
                        2
                    )

                    tx, ty, tz = tvec.flatten()

                    roll, pitch, yaw = rvec_to_euler_xyz(rvec)
                    roll_d = math.degrees(roll)
                    pitch_d = math.degrees(pitch)
                    yaw_d = math.degrees(yaw)

                    text1 = f"ID {self.target_id} tvec[m]=({tx:+.3f},{ty:+.3f},{tz:+.3f})"
                    text2 = f"RPY[deg]=({roll_d:+.1f},{pitch_d:+.1f},{yaw_d:+.1f})"

                    cv2.putText(vis, text1, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, text2, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    self.get_logger().info(
                        f"ID {self.target_id} | "
                        f"tx={tx:+.3f} m, ty={ty:+.3f} m, tz={tz:+.3f} m | "
                        f"roll={roll_d:+.1f}, pitch={pitch_d:+.1f}, yaw={yaw_d:+.1f}"
                    )

        cv2.imshow("aruco_pose_out", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ArucoPose()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()


if __name__ == '__main__':
    main()
