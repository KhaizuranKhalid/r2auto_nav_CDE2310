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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


GridCell = Tuple[int, int]   # (row, col)
Path = List[GridCell]


# code from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
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

    return roll_x, pitch_y, yaw_z # in radians

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
        # Odometry fallback (used if TF is unavailable)
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = None

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
        self.failed_goals = set()           # type: Set[GridCell]
        self.frontier_min_size = 10
        self.goal_tolerance = 0.20          # meters
        self.linear_speed = 0.10            # m/s
        self.angular_speed_limit = 0.6      # rad/s
        self.k_linear = 0.6
        self.k_angular = 1.8
        self.obstacle_stop_distance = 0.25  # meters
        self.base_frame = 'base_link'
        self.map_frame = 'map'

    # ----------------------------
    # Callbacks
    # ----------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info("Map received")
        self.map_info = msg.info

        raw = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width)
        )

        grid = np.full_like(raw, -1, dtype=np.int8)
        grid[raw == -1] = -1
        grid[(raw >= 0) & (raw <= 50)] = 0
        grid[raw > 50] = 1

        # Mild obstacle inflation
        inflation_radius = 4
        inflated = grid.copy()

        height, width = grid.shape

        for r in range(height):
            for c in range(width):
                if grid[r, c] == 1:
                    for dr in range(-inflation_radius, inflation_radius + 1):
                        for dc in range(-inflation_radius, inflation_radius + 1):
                            rr = r + dr
                            cc = c + dc
                            if 0 <= rr < height and 0 <= cc < width:
                                if inflated[rr, cc] == 0:
                                    inflated[rr, cc] = 1

        self.map_grid = inflated
        self.map_received = True

    def scan_callback(self, msg: LaserScan):
        self.scan = np.array(msg.ranges, dtype=np.float32)
        self.scan[self.scan == 0.0] = np.nan

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.odom_x = pos.x
        self.odom_y = pos.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(
            q.x, q.y, q.z, q.w
        )

        self.odom_yaw = yaw

    # ----------------------------
    # Pose helpers
    # ----------------------------
    # def quaternion_to_yaw(self, qx, qy, qz, qw) -> float:
    #     siny_cosp = 2.0 * (qw * qz + qx * qy)
    #     cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    #     return math.atan2(siny_cosp, cosy_cosp)

    def update_robot_pose(self) -> bool:
        # Try TF first (best for SLAM)
        try:
            self.get_logger().info("TF pose received")
            trans = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(seconds=0),
                timeout=Duration(seconds=0.3)
            )

            self.robot_x = trans.transform.translation.x
            self.robot_y = trans.transform.translation.y

            q = trans.transform.rotation
            self.robot_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

            return True

        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

        # Fallback to odom if TF not ready
        self.get_logger().info("Using odom fallback")
        if self.odom_x is not None:
            self.robot_x = self.odom_x
            self.robot_y = self.odom_y
            self.robot_yaw = self.odom_yaw
            return True

        return False

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

        unknown_count = 0
        for nr, nc in self.neighbors4(row, col):
            if self.is_unknown(nr, nc):
                unknown_count += 1

        return unknown_count >= 1

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
        if self.is_occupied(sr, sc):
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
    # Choose the best center cell of the largest frontier cluster, with some distance penalty
    # def choose_frontier_target(self, clusters):
    #     robot_cell = self.world_to_grid(self.robot_x, self.robot_y)
    #     if robot_cell is None:
    #         return None

    #     rr, rc = robot_cell
    #     best_score = -1
    #     best_target = None

    #     for cluster in clusters:
    #         # choose center of cluster
    #         cr = int(sum(p[0] for p in cluster) / len(cluster))
    #         cc = int(sum(p[1] for p in cluster) / len(cluster))

    #         if self.near_obstacle(cr, cc):
    #             continue

    #         dist = math.hypot(cr - rr, cc - rc)

    #         # better scoring
    #         score = len(cluster) - dist * 0.5

    #         if score > best_score:
    #             best_score = score
    #             best_target = (cr, cc)

    #     return best_target

    # Choose the closest cell in any frontier cluster, with obstacle penalty
    def choose_frontier_target(self, clusters):
        robot_cell = self.world_to_grid(self.robot_x, self.robot_y)
        if robot_cell is None:
            return None

        rr, rc = robot_cell
        best_dist = float('inf')
        best_target = None

        for cluster in clusters:
            for cell in cluster:
                r, c = cell

                if self.near_obstacle(r, c):
                    continue

                dist = abs(r - rr) + abs(c - rc)

                if dist < best_dist:
                    best_dist = dist
                    best_target = cell

        return best_target
    
    def obstacle_ahead(self):
        if self.scan.size == 0:
            return False

        n = len(self.scan)
        mid = n // 2

        left = self.scan[mid + 20]
        right = self.scan[mid - 20]

        front = self.front_distance()

        return min(front, left, right) < self.obstacle_stop_distance

    def near_obstacle(self, r, c):
        clearance = 6   # try 6–10

        for dr in range(-clearance, clearance + 1):
            for dc in range(-clearance, clearance + 1):
                rr = r + dr
                cc = c + dc
                if self.in_bounds(rr, cc) and self.is_occupied(rr, cc):
                    return True
        return False
    
    def recover_exploration(self):
        self.get_logger().info("Recovery behavior: scanning for new frontiers")

        twist = Twist()
        twist.angular.z = 0.8

        start_time = self.get_clock().now()

        while (self.get_clock().now() - start_time).nanoseconds < 8e9:
            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.stop_robot()

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
        if self.is_occupied(*goal):
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
            if self.obstacle_ahead() < self.obstacle_stop_distance:
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

        self.get_logger().info("Initial scan to build map")
        self.recover_exploration()

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
                self.recover_exploration()

                clusters = self.detect_frontiers()
                if not clusters:
                    self.get_logger().info("Exploration finished.")
                    break

            clusters = [
                c for c in clusters
                if all(cell not in self.failed_goals for cell in c)
            ]
            target_cell = self.choose_frontier_target(clusters)
            if target_cell is None:
                self.get_logger().warn("Could not choose a frontier target.")
                self.stop_robot()
                continue

            path = self.astar(robot_cell, target_cell)

            if not path:
                # self.failed_goals.add(target_cell)
                self.get_logger().warn(f"No A* path to frontier target {target_cell}. Replanning...")
                continue

            self.get_logger().info(
                f"Frontier target: {target_cell}, path length: {len(path)}"
            )

            success = self.follow_path(path)
            
            if not success:
                self.get_logger().warn("Path was interrupted by obstacle. Replanning...")
                self.failed_goals.add(target_cell)
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