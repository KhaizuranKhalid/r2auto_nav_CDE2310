#!/usr/bin/env python3

import math
from collections import deque
from typing import List, Tuple, Optional, Set

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

import tf2_ros
from tf2_ros import Buffer, TransformListener

GridCell = Tuple[int, int]  # (row, col)

# ----------------------------
# Quaternion → Euler conversion
# ----------------------------
def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

# ----------------------------
# Frontier Explorer Node
# ----------------------------
class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, qos_profile_sensor_data
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data
        )

        # Nav2 Action Client
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Map state
        self.map_grid: Optional[np.ndarray] = None
        self.map_info = None
        self.map_received = False

        # Robot pose
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # LiDAR state
        self.scan = np.array([])

        # Frontier params
        self.frontier_min_size = 1
        self.failed_goals: Set[GridCell] = set()

    # ----------------------------
    # Callbacks
    # ----------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info
        raw = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width)
        )
        # 0=free, 1=occupied, -1=unknown
        grid = np.full_like(raw, -1, dtype=np.int8)
        grid[raw == -1] = -1
        grid[(raw >= 0) & (raw <= 50)] = 0
        grid[raw > 50] = 1

        # NO inflation needed; Nav2 handles obstacles
        self.map_grid = grid
        self.map_received = True

    def scan_callback(self, msg: LaserScan):
        self.scan = np.array(msg.ranges, dtype=np.float32)
        self.scan[self.scan == 0.0] = np.nan

    # ----------------------------
    # Grid helpers
    # ----------------------------
    def in_bounds(self, row: int, col: int) -> bool:
        return self.map_grid is not None and 0 <= row < self.map_grid.shape[0] and 0 <= col < self.map_grid.shape[1]

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.map_grid[row, col] == 0

    def is_unknown(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.map_grid[row, col] == -1

    def is_occupied(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.map_grid[row, col] == 1

    def world_to_grid(self, x: float, y: float) -> Optional[GridCell]:
        if self.map_info is None:
            return None
        res = self.map_info.resolution
        ox, oy = self.map_info.origin.position.x, self.map_info.origin.position.y
        col = int((x - ox) / res)
        row = int((y - oy) / res)
        return (row, col) if self.in_bounds(row, col) else None

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        res = self.map_info.resolution
        ox, oy = self.map_info.origin.position.x, self.map_info.origin.position.y
        x = ox + (col + 0.5) * res
        y = oy + (row + 0.5) * res
        return x, y

    def neighbors4(self, row: int, col: int):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    def neighbors8(self, row: int, col: int):
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self.in_bounds(nr, nc):
                    yield nr, nc

    # ----------------------------
    # Frontier detection
    # ----------------------------
    def cell_is_frontier(self, row: int, col: int) -> bool:
        """A frontier cell is free and touches unknown space (8-connectivity)."""
        if not self.is_free(row, col):
            return False
        return any(self.is_unknown(nr, nc) for nr, nc in self.neighbors8(row, col))

    def detect_frontiers(self) -> List[List[GridCell]]:
        """Detect all reachable frontier clusters from robot pose."""
        if self.map_grid is None or self.robot_x is None:
            return []

        start = self.world_to_grid(self.robot_x, self.robot_y)
        if start is None:
            return []

        # BFS reachable free cells
        reachable_free = set([start])
        q = deque([start])
        frontier_cells = set()

        while q:
            r, c = q.popleft()
            if self.cell_is_frontier(r, c):
                frontier_cells.add((r, c))
            for nr, nc in self.neighbors4(r, c):
                if (nr, nc) not in reachable_free and self.is_free(nr, nc):
                    reachable_free.add((nr, nc))
                    q.append((nr, nc))

        # Cluster frontier cells using 8-connectivity
        clusters = []
        unvisited = set(frontier_cells)
        while unvisited:
            seed = unvisited.pop()
            cluster = [seed]
            fq = deque([seed])
            while fq:
                r, c = fq.popleft()
                for nr, nc in self.neighbors8(r, c):
                    if (nr, nc) in unvisited:
                        unvisited.remove((nr, nc))
                        cluster.append((nr, nc))
                        fq.append((nr, nc))
            if len(cluster) >= self.frontier_min_size:
                clusters.append(cluster)

        return clusters

    def choose_frontier_target(self, clusters: List[List[GridCell]]) -> Optional[GridCell]:
        """Choose the frontier cluster closest to the robot, avoiding obstacles."""
        robot_cell = self.world_to_grid(self.robot_x, self.robot_y)
        if robot_cell is None:
            return None
        rr, rc = robot_cell
        best_target = None
        best_dist = float('inf')

        for cluster in clusters:
            # centroid
            r_mean = int(sum(r for r, _ in cluster) / len(cluster))
            c_mean = int(sum(c for _, c in cluster) / len(cluster))

            # skip if centroid touches obstacles
            if any(self.is_occupied(nr, nc) for nr, nc in self.neighbors8(r_mean, c_mean)):
                continue

            # Manhattan distance to robot (fast)
            dist = abs(r_mean - rr) + abs(c_mean - rc)
            if dist < best_dist:
                best_dist = dist
                best_target = (r_mean, c_mean)

        return best_target

    # ----------------------------
    # Robot pose update
    # ----------------------------
    def update_robot_pose(self) -> bool:
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_link', now, timeout=Duration(seconds=0.5)
            )
            self.robot_x = trans.transform.translation.x
            self.robot_y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, self.robot_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return False

    # ----------------------------
    # Nav2 Goal
    # ----------------------------
    def send_goal(self, x: float, y: float, yaw: float = 0.0) -> bool:
        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 server not available!")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        send_goal_future = self.nav2_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected!")
            return False

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result()
        return result.status == GoalStatus.STATUS_SUCCEEDED

    # ----------------------------
    # Main exploration loop
    # ----------------------------
    def explore(self):
        self.get_logger().info("Waiting for map and transform...")

        # wait until we get map and robot pose
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.map_received and self.update_robot_pose():
                break

        self.get_logger().info("Starting frontier exploration.")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.update_robot_pose():
                continue

            clusters = self.detect_frontiers()
            # remove clusters containing failed goals
            clusters = [c for c in clusters if not any(cell in self.failed_goals for cell in c)]
            if not clusters:
                self.get_logger().info("No reachable frontiers left. Exploration finished.")
                break

            target_cell = self.choose_frontier_target(clusters)
            if target_cell is None:
                self.get_logger().warn("No valid frontier target")
                break

            wx, wy = self.grid_to_world(*target_cell)
            self.get_logger().info(f"Navigating to frontier: ({wx:.2f}, {wy:.2f})")

            success = self.send_goal(wx, wy)
            if not success:
                self.get_logger().warn("Nav2 could not reach goal. Adding to failed list.")
                self.failed_goals.add(target_cell)
            else:
                self.get_logger().info("Goal reached successfully.")

# ----------------------------
# Main entry
# ----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        node.explore()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()