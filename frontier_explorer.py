import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import math
import heapq
import time
from scipy.ndimage import binary_dilation

# TF2
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration

# -----------------------------
# Constants
# -----------------------------
SPEED = 0.12
ROT_SPEED = 0.5
MAP_UNKNOWN = -1
MAP_FREE = 0
OCCUPIED_THRESHOLD = 50
INFLATION_RADIUS = 1  # Increase if the robot still hits walls
SAFE_DISTANCE = 0.20  # Reactive stop distance

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # -----------------------------
        # TF2 Setup
        # -----------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -----------------------------
        # Publishers
        # -----------------------------
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.launch_ball_pub = self.create_publisher(String, 'launch_ball', 10)

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.occ_sub = self.create_subscription(
            OccupancyGrid, 'map', self.occ_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)

        # -----------------------------
        # State variables
        # -----------------------------
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.occdata = None
        self.inflated_occdata = None
        self.latest_scan = None
        self.map_width = self.map_height = 0
        self.map_resolution = 0.0
        self.map_origin = None

        self.get_logger().info("Frontier Explorer Node Initialized and Waiting for Map...")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_callback(self, msg):
        # If you need orientation, convert quaternion to roll, pitch, yaw
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    def occ_callback(self, msg):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        self.occdata = np.array(msg.data, dtype=int).reshape((self.map_height, self.map_width))

        # Inflate obstacles
        occupied_mask = (self.occdata >= OCCUPIED_THRESHOLD)
        struct = np.ones((INFLATION_RADIUS*2, INFLATION_RADIUS*2))
        inflated_mask = binary_dilation(occupied_mask, structure=struct)
        self.inflated_occdata = self.occdata.copy()
        self.inflated_occdata[inflated_mask] = 100

    def scan_callback(self, msg):
        self.latest_scan = np.array(msg.ranges)

    # -----------------------------
    # Pose using TF2
    # -----------------------------
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.5))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            return x, y, yaw
        except Exception as e:
            self.get_logger().warn(f"Pose not available yet: {e}")
            return None, None, None

    # -----------------------------
    # Coordinates & Logic
    # -----------------------------
    def world_to_grid(self, x, y):
        if self.map_origin is None: return 0, 0
        # Use floor division to get the correct cell index
        gx = int((x - self.map_origin.x) / self.map_resolution)
        gy = int((y - self.map_origin.y) / self.map_resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        # Center the coordinate in the middle of the cell
        wx = (gx * self.map_resolution) + self.map_origin.x + (self.map_resolution / 2.0)
        wy = (gy * self.map_resolution) + self.map_origin.y + (self.map_resolution / 2.0)
        return wx, wy

    def is_path_blocked(self):
        if self.latest_scan is None: return False
        # Check front 30 degrees
        n = len(self.latest_scan)
        front = np.concatenate((self.latest_scan[:n//12], self.latest_scan[-n//12:]))
        valid = front[(front > 0.01) & (front < SAFE_DISTANCE)]
        return len(valid) > 0

    def get_frontiers(self):
        if self.occdata is None or self.inflated_occdata is None: return []
        
        # Find all free cells that are NOT inflated (safe to stand)
        safe_free_mask = (self.occdata == MAP_FREE) & (self.inflated_occdata < OCCUPIED_THRESHOLD)
        
        # Find coordinates
        y_coords, x_coords = np.where(safe_free_mask)
        frontiers = []
        
        for x, y in zip(x_coords, y_coords):
            # Check 8-neighbors for UNKNOWN
            # We use a 3x3 slice for speed
            if MAP_UNKNOWN in self.occdata[y-1:y+2, x-1:x+2]:
                frontiers.append((x, y))
                
        return frontiers

    # -----------------------------
    # Navigation
    # -----------------------------
    def astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if math.hypot(current[0]-goal[0], current[1]-goal[1]) < 2:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                neighbor = (current[0]+dx, current[1]+dy)
                if 0 <= neighbor[0] < self.map_width and 0 <= neighbor[1] < self.map_height:
                    # Plan using INFLATED map
                    if self.inflated_occdata[neighbor[1], neighbor[0]] == MAP_FREE:
                        cost = 1.414 if abs(dx)+abs(dy)==2 else 1.0
                        ten_g = g_score[current] + cost
                        if neighbor not in g_score or ten_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = ten_g
                            f = ten_g + math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                            heapq.heappush(open_set, (f, neighbor))
        return None

    def move_to(self, wx, wy):
        self.get_logger().info(f"Target: {wx:.2f}, {wy:.2f}")
        while rclpy.ok():
            curr_x, curr_y, curr_yaw = self.get_robot_pose()
            if curr_x is None: break

            dx, dy = wx - curr_x, wy - curr_y
            dist = math.hypot(dx, dy)
            if dist < 0.18: break

            target_yaw = math.atan2(dy, dx)
            angle_diff = math.atan2(math.sin(target_yaw - curr_yaw), math.cos(target_yaw - curr_yaw))

            twist = Twist()
            blocked = self.is_path_blocked()

            if abs(angle_diff) > 0.5:
                twist.angular.z = ROT_SPEED * np.sign(angle_diff)
            elif blocked:
                self.get_logger().warn("Obstacle! Nudging away.")
                twist.linear.x = 0.0
                twist.angular.z = ROT_SPEED
            else:
                twist.linear.x = SPEED
                twist.angular.z = 0.3 * angle_diff

            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.05)
            if blocked: return # Exit to re-plan if blocked

    def rotate(self):
        t = Twist()
        t.angular.z = ROT_SPEED
        self.cmd_pub.publish(t)
        time.sleep(1.0)
        self.cmd_pub.publish(Twist())

    # -----------------------------
    # Main Loop
    # -----------------------------
    def explore(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.occdata is None: continue

            curr_x, curr_y, _ = self.get_robot_pose()
            if curr_x is None: continue

            frontiers = self.get_frontiers()
            if not frontiers:
                self.get_logger().info("No reachable frontiers. Rotating...")
                self.rotate()
                continue

            # Find closest reachable frontier
            rgx, rgy = self.world_to_grid(curr_x, curr_y)
            dists = [math.hypot(f[0]-rgx, f[1]-rgy) for f in frontiers]
            sorted_indices = np.argsort(dists)

            path = None
            for idx in sorted_indices[:10]: # Try nearest 10
                path = self.astar((rgx, rgy), frontiers[idx])
                if path: break

            if path:
                # Look further ahead if the path is long, or just go to the end
                # For a 4x4m area, 15 cells is about 75cm.
                look_ahead = min(len(path) - 1, 15) 
                target_node = path[look_ahead]
                wx, wy = self.grid_to_world(target_node[0], target_node[1])
                self.move_to(wx, wy)
            else:
                self.get_logger().warn("A* failed to find a valid path to frontiers. Is the robot boxed in by inflation?")
                self.rotate()
def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        node.explore()
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()