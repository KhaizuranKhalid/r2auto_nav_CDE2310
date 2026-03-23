#!/usr/bin/env python3

import math
import heapq
from collections import deque
from typing import List, Tuple, Optional, Set

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


GridCell = Tuple[int, int]   # (row, col)
Path = List[GridCell]


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Publishers / Subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, qos_profile_sensor_data
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data
        )

        # TF for map -> base_link pose
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Map state
        self.map_grid = None              # values: -1 unknown, 0 free, 1 occupied
        self.map_info = None
        self.map_received = False

        # Robot state
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # LiDAR state
        self.scan = np.array([])

        # Tuning parameters
        self.frontier_min_size = 5
        self.goal_tolerance = 0.20          # meters
        self.linear_speed = 0.12            # m/s
        self.angular_speed_limit = 0.8      # rad/s
        self.k_linear = 0.6
        self.k_angular = 1.8
        self.obstacle_stop_distance = 0.25  # meters
        self.base_frame = 'base_link'
        self.map_frame = 'map'

    # ----------------------------
    # Callbacks
    # ----------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info

        raw = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))

        # Normalize:
        # -1 -> unknown
        #  0..50 -> free
        # 51..100 -> occupied
        grid = np.full_like(raw, -1, dtype=np.int8)
        grid[raw == -1] = -1
        grid[(raw >= 0) & (raw <= 50)] = 0
        grid[raw > 50] = 1

        self.map_grid = grid
        self.map_received = True

    def scan_callback(self, msg: LaserScan):
        self.scan = np.array(msg.ranges, dtype=np.float32)
        self.scan[self.scan == 0.0] = np.nan

    # ----------------------------
    # Pose helpers
    # ----------------------------
    def quaternion_to_yaw(self, qx, qy, qz, qw) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def update_robot_pose(self) -> bool:
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False

        self.robot_x = trans.transform.translation.x
        self.robot_y = trans.transform.translation.y

        q = trans.transform.rotation
        self.robot_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        return True

    # ----------------------------
    # Grid helpers
    # ----------------------------
    def in_bounds(self, row: int, col: int) -> bool:
        return (
            self.map_grid is not None and
            0 <= row < self.map_grid.shape[0] and
            0 <= col < self.map_grid.shape[1]
        )

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
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

        col = int((x - ox) / res)
        row = int((y - oy) / res)

        if not self.in_bounds(row, col):
            return None
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

        x = ox + (col + 0.5) * res
        y = oy + (row + 0.5) * res
        return x, y

    def neighbors4(self, row: int, col: int):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    def neighbors8(self, row: int, col: int):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self.in_bounds(nr, nc):
                    yield nr, nc

    # ----------------------------
    # Frontier detection
    # Frontier = free cell adjacent to at least one unknown cell
    # ----------------------------
    def cell_is_frontier(self, row: int, col: int) -> bool:
        if not self.is_free(row, col):
            return False
        for nr, nc in self.neighbors8(row, col):
            if self.is_unknown(nr, nc):
                return True
        return False

    def detect_frontiers(self) -> List[List[GridCell]]:
        """
        Returns a list of frontier clusters.
        Each cluster is a list of free cells that border unknown space.
        """
        if self.map_grid is None:
            return []

        if self.robot_x is None or self.robot_y is None:
            return []

        start = self.world_to_grid(self.robot_x, self.robot_y)
        if start is None:
            return []

        sr, sc = start
        if not self.is_free(sr, sc):
            return []

        # Flood-fill through reachable free space only
        reachable_free = set()
        q = deque([start])
        reachable_free.add(start)

        frontier_cells = set()

        while q:
            r, c = q.popleft()

            if self.cell_is_frontier(r, c):
                frontier_cells.add((r, c))

            for nr, nc in self.neighbors4(r, c):
                if (nr, nc) in reachable_free:
                    continue
                if self.is_free(nr, nc):
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
        """
        Pick the best cluster using a simple score:
        larger clusters are better, but closer ones are also preferred.
        Then choose a representative cell in that cluster.
        """
        if not clusters:
            return None

        robot_cell = self.world_to_grid(self.robot_x, self.robot_y)
        if robot_cell is None:
            return None

        rr, rc = robot_cell

        def cluster_score(cluster: List[GridCell]) -> float:
            centroid_r = sum(p[0] for p in cluster) / len(cluster)
            centroid_c = sum(p[1] for p in cluster) / len(cluster)
            dist = abs(centroid_r - rr) + abs(centroid_c - rc)
            return len(cluster) / (dist + 1.0)

        best_cluster = max(clusters, key=cluster_score)

        # Use the frontier cell in this cluster closest to the robot
        target = min(best_cluster, key=lambda p: abs(p[0] - rr) + abs(p[1] - rc))
        return target

    # ----------------------------
    # A* path planning
    # ----------------------------
    def heuristic(self, a: GridCell, b: GridCell) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start: GridCell, goal: GridCell) -> Path:
        if self.map_grid is None:
            return []
        if not self.is_free(*start):
            return []
        if not self.is_free(*goal):
            return []

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.neighbors4(*current):
                nr, nc = neighbor
                if not self.is_free(nr, nc):
                    continue

                tentative_g = g_score[current] + 1.0
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return []

    def reconstruct_path(self, came_from: dict, current: GridCell) -> Path:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ----------------------------
    # Motion control
    # ----------------------------
    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def front_distance(self) -> float:
        if self.scan.size == 0:
            return float('inf')

        n = len(self.scan)
        mid = n // 2
        span = max(1, int(15 * n / 360))  # roughly +/-15 degrees

        sector = self.scan[max(0, mid - span): min(n, mid + span + 1)]
        sector = sector[np.isfinite(sector)]

        if sector.size == 0:
            return float('inf')
        return float(np.min(sector))

    def angle_wrap(self, ang: float) -> float:
        return (ang + math.pi) % (2.0 * math.pi) - math.pi

    def move_to_waypoint(self, goal_world: Tuple[float, float]) -> bool:
        """
        Move toward one waypoint.
        Returns False if we hit an obstacle and need to replan.
        """
        gx, gy = goal_world

        rate_spin_count = 0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if not self.update_robot_pose():
                continue

            dx = gx - self.robot_x
            dy = gy - self.robot_y
            dist = math.hypot(dx, dy)

            if dist <= self.goal_tolerance:
                self.stop_robot()
                return True

            # Safety stop if something is too close in front
            if self.front_distance() < self.obstacle_stop_distance:
                self.stop_robot()
                return False

            target_yaw = math.atan2(dy, dx)
            yaw_error = self.angle_wrap(target_yaw - self.robot_yaw)

            twist = Twist()

            # Heading control
            twist.angular.z = max(
                -self.angular_speed_limit,
                min(self.angular_speed_limit, self.k_angular * yaw_error)
            )

            # Move forward only if roughly facing the waypoint
            if abs(yaw_error) < 0.5:
                twist.linear.x = min(self.linear_speed, self.k_linear * dist)
            else:
                twist.linear.x = 0.0

            self.cmd_pub.publish(twist)

            rate_spin_count += 1
            if rate_spin_count % 20 == 0:
                self.get_logger().info(
                    f"Moving to waypoint: dist={dist:.2f}, yaw_error={math.degrees(yaw_error):.1f} deg"
                )

        return False

    def follow_path(self, path: Path) -> bool:
        if not path or len(path) < 2:
            return True

        for cell in path[1:]:
            wx, wy = self.grid_to_world(*cell)
            ok = self.move_to_waypoint((wx, wy))
            if not ok:
                return False

        self.stop_robot()
        return True
    
    def scan_environment(self):
        twist = Twist()
        twist.angular.z = 0.6

        start_time = self.get_clock().now()

        while (self.get_clock().now() - start_time).nanoseconds < 6e9:
            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.stop_robot()

    # ----------------------------
    # Main exploration loop
    # ----------------------------
    def explore(self):
        self.get_logger().info("Waiting for map and TF...")

        # Wait until map exists and pose can be read
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.map_received and self.update_robot_pose():
                break

        self.get_logger().info("Starting frontier exploration.")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.update_robot_pose():
                continue

            robot_cell = self.world_to_grid(self.robot_x, self.robot_y)
            if robot_cell is None:
                self.get_logger().warn("Robot is outside the map.")
                self.stop_robot()
                continue

            clusters = self.detect_frontiers()

            if not clusters:
                self.get_logger().info("No frontiers left. Exploration complete.")
                self.stop_robot()
                break

            target_cell = self.choose_frontier_target(clusters)
            if target_cell is None:
                self.get_logger().warn("Could not choose a frontier target.")
                self.stop_robot()
                continue

            path = self.astar(robot_cell, target_cell)

            if not path:
                self.get_logger().warn(f"No A* path to frontier target {target_cell}. Replanning...")
                continue

            self.get_logger().info(
                f"Frontier target: {target_cell}, path length: {len(path)}"
            )

            success = self.follow_path(path)
            
            if not success:
                self.get_logger().warn("Path was interrupted by obstacle. Replanning...")
                continue
            self.scan_environment()

        self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        node.explore()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()