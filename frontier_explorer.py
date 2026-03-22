#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
import numpy as np
import heapq

class Explorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.lidar_callback, qos_profile_sensor_data)
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, qos_profile_sensor_data)
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.twist = Twist()
        
        # Grid settings
        self.grid = None
        self.grid_size = 0.05
        self.robot_pos = None
        self.path = []
        self.wp_idx = 0
        self.obstacle_threshold = 0.3  # meters

    # ------------------------
    # Occupancy Grid from /map
    # ------------------------
    def map_callback(self, msg):
        data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.grid = np.zeros_like(data)
        self.grid[data == -1] = -1     # unknown
        self.grid[data > 50] = 1       # occupied
        self.grid[(data >= 0) & (data <= 50)] = 0  # free

        # assume robot starts at map center
        if self.robot_pos is None:
            self.robot_pos = (msg.info.height // 2, msg.info.width // 2)

    # ------------------------
    # LiDAR callback
    # ------------------------
    def lidar_callback(self, msg):
        if self.grid is None:
            return  # wait until map is received

        # simple obstacle avoidance
        laser_range = np.array(msg.ranges)
        laser_range[laser_range == 0] = np.nan
        front = min(np.nanmin(laser_range[0:10]), np.nanmin(laser_range[-10:]))

        if front < self.obstacle_threshold:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5
            self.cmd_pub.publish(self.twist)
            return

        # if path is empty or completed, find next frontier
        if not self.path or self.wp_idx >= len(self.path):
            frontiers = self.find_frontiers()
            goal = self.closest_frontier(frontiers)
            if goal is None:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.cmd_pub.publish(self.twist)
                return
            self.path = self.astar(self.robot_pos, goal)
            self.wp_idx = 0

        # move toward next waypoint
        next_wp = self.path[self.wp_idx]
        dx = next_wp[0] - self.robot_pos[0]
        dy = next_wp[1] - self.robot_pos[1]
        if abs(dx) + abs(dy) < 1:
            self.wp_idx += 1
            self.robot_pos = next_wp
        else:
            self.twist.linear.x = 0.1
            self.twist.angular.z = 0.0

        self.cmd_pub.publish(self.twist)

    # ------------------------
    # Frontier Detection
    # ------------------------
    def find_frontiers(self):
        frontiers = []
        for i in range(1, self.grid.shape[0]-1):
            for j in range(1, self.grid.shape[1]-1):
                if self.grid[i, j] == 0:
                    neighbors = self.grid[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == -1):
                        frontiers.append((i, j))
        return frontiers

    def closest_frontier(self, frontiers):
        if not frontiers:
            return None
        return min(frontiers, key=lambda f: abs(f[0]-self.robot_pos[0])+abs(f[1]-self.robot_pos[1]))

    # ------------------------
    # Simple A* planner
    # ------------------------
    def astar(self, start, goal):
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            close_set.add(current)
            for i,j in neighbors:
                neighbor = current[0]+i, current[1]+j
                tentative_g = gscore[current]+1
                if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1]:
                    if self.grid[neighbor[0], neighbor[1]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor,0):
                    continue
                if tentative_g < gscore.get(neighbor,0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

def main(args=None):
    # self.get_logger().info("Explorer node started!")
    rclpy.init(args=args)
    node = Explorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()