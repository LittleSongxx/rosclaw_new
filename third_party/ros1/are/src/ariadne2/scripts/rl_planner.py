#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter("ignore", UserWarning)

import rospy
import rospkg
import numpy as np
import torch
import os
import time
from std_msgs.msg import Float32, Header, String
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from agent import Agent
from model import PolicyNet
from node_manager import NodeManager
from utils import *
import parameter


class Runner:
    def __init__(self):
        self.map_info = None
        self.device = 'cpu'
        self.step = 0

        # visualization
        self.publish_graph = rospy.get_param('~publish_graph', True)

        # map related
        parameter.CELL_SIZE = rospy.get_param('~map_resolution', parameter.CELL_SIZE)
        parameter.FREE = rospy.get_param('~map_free_value', parameter.FREE)
        parameter.OCCUPIED = rospy.get_param('~map_occupied_value', parameter.OCCUPIED)
        parameter.UNKNOWN = rospy.get_param('~map_unknown_value', parameter.UNKNOWN)

        # utility related
        parameter.SENSOR_RANGE = rospy.get_param('~sensor_range', parameter.SENSOR_RANGE)
        parameter.UTILITY_RANGE = rospy.get_param('~utility_range_factor', 0.5) * parameter.SENSOR_RANGE
        parameter.MIN_UTILITY = rospy.get_param('~min_utility', parameter.MIN_UTILITY)
        parameter.FRONTIER_CELL_SIZE = rospy.get_param('~frontier_downsample_factor', 1) * parameter.CELL_SIZE

        # graph related
        parameter.NODE_RESOLUTION = rospy.get_param('~node_resolution', parameter.NODE_RESOLUTION)
        parameter.CLUSTER_RANGE = rospy.get_param('~frontier_cluster_range', parameter.CLUSTER_RANGE)
        parameter.THR_NEXT_WAYPOINT = rospy.get_param('~next_waypoint_threshold', parameter.THR_NEXT_WAYPOINT)
        parameter.THR_GRAPH_HARD_UPDATE = rospy.get_param('~hard_update_threshold', parameter.THR_GRAPH_HARD_UPDATE)

        # replanning related
        parameter.THR_TO_WAYPOINT = rospy.get_param('~waypoint_threshold', parameter.THR_TO_WAYPOINT)
        parameter.AVOID_OSCILLATION = rospy.get_param('~avoid_waypoint_oscillation', parameter.AVOID_OSCILLATION)
        parameter.ENABLE_SAVE_MODE = rospy.get_param('~enable_save_mode', parameter.ENABLE_SAVE_MODE)
        parameter.ENABLE_DSTARLITE = rospy.get_param('~enable_dstarlite', parameter.ENABLE_DSTARLITE)
        frequency = rospy.get_param('~replanning_frequency', 2.5)

        # ============== 新增性能优化参数 ==============
        # 碰撞检测优化
        parameter.ALLOW_UNKNOWN_PASSTHROUGH = rospy.get_param('~allow_unknown_passthrough', parameter.ALLOW_UNKNOWN_PASSTHROUGH)
        parameter.UNKNOWN_TOLERANCE_CELLS = rospy.get_param('~unknown_tolerance_cells', parameter.UNKNOWN_TOLERANCE_CELLS)
        parameter.SOFT_COLLISION_CHECK = rospy.get_param('~soft_collision_check', parameter.SOFT_COLLISION_CHECK)

        # 邻居节点发现优化
        parameter.EXTENDED_NEIGHBOR_RANGE = rospy.get_param('~extended_neighbor_range', parameter.EXTENDED_NEIGHBOR_RANGE)
        parameter.NEIGHBOR_MATRIX_SIZE = rospy.get_param('~neighbor_matrix_size', parameter.NEIGHBOR_MATRIX_SIZE)

        # 图连通性优化
        parameter.DELAYED_NODE_REMOVAL = rospy.get_param('~delayed_node_removal', parameter.DELAYED_NODE_REMOVAL)
        parameter.NODE_REMOVAL_DELAY_STEPS = rospy.get_param('~node_removal_delay_steps', parameter.NODE_REMOVAL_DELAY_STEPS)

        # 前沿点处理优化
        parameter.DISTANCE_WEIGHTED_UTILITY = rospy.get_param('~distance_weighted_utility', parameter.DISTANCE_WEIGHTED_UTILITY)
        parameter.UTILITY_DISTANCE_DECAY = rospy.get_param('~utility_distance_decay', parameter.UTILITY_DISTANCE_DECAY)

        # D*-Lite优化
        parameter.DSTARLITE_MAX_TIME = rospy.get_param('~dstarlite_max_time', parameter.DSTARLITE_MAX_TIME)
        parameter.DSTARLITE_ADAPTIVE_TIME = rospy.get_param('~dstarlite_adaptive_time', parameter.DSTARLITE_ADAPTIVE_TIME)

        # 狭窄通道检测
        parameter.ENABLE_NARROW_PASSAGE_DETECTION = rospy.get_param('~enable_narrow_passage_detection', parameter.ENABLE_NARROW_PASSAGE_DETECTION)
        parameter.NARROW_PASSAGE_WIDTH_THRESHOLD = rospy.get_param('~narrow_passage_width_threshold', parameter.NARROW_PASSAGE_WIDTH_THRESHOLD)

        # 图连通性检查优化
        parameter.USE_CURRENT_LOCATION_FOR_CONNECTIVITY = rospy.get_param('~use_current_location_for_connectivity', parameter.USE_CURRENT_LOCATION_FOR_CONNECTIVITY)

        # 小波阈值与邻居轻筛选（仅影响图构建与候选动作集）
        parameter.WAVELET_ADAPTIVE_DTH = rospy.get_param('~wavelet_adaptive_dth', parameter.WAVELET_ADAPTIVE_DTH)
        parameter.WAVELET_DTH_ALPHA = rospy.get_param('~wavelet_dth_alpha', parameter.WAVELET_DTH_ALPHA)
        parameter.WAVELET_DTH_MAX_MULT = rospy.get_param('~wavelet_dth_max_mult', parameter.WAVELET_DTH_MAX_MULT)
        parameter.WAVELET_LOCAL_MAP_SIZE = rospy.get_param('~wavelet_local_map_size', parameter.WAVELET_LOCAL_MAP_SIZE)
        parameter.WAVELET_CACHE_CHANGE_RATIO_THRESH = rospy.get_param(
            '~wavelet_cache_change_ratio_thresh',
            parameter.WAVELET_CACHE_CHANGE_RATIO_THRESH,
        )
        parameter.WAVELET_NEIGHBOR_LIGHT_FILTER = rospy.get_param('~wavelet_neighbor_light_filter', parameter.WAVELET_NEIGHBOR_LIGHT_FILTER)
        parameter.WAVELET_NEIGHBOR_MAX_KEEP = rospy.get_param('~wavelet_neighbor_max_keep', parameter.WAVELET_NEIGHBOR_MAX_KEEP)
        parameter.WAVELET_NEIGHBOR_MIN_KEEP = rospy.get_param('~wavelet_neighbor_min_keep', parameter.WAVELET_NEIGHBOR_MIN_KEEP)
        parameter.WAVELET_NEIGHBOR_FILTER_MIN_TRIGGER = rospy.get_param(
            '~wavelet_neighbor_filter_min_trigger',
            parameter.WAVELET_NEIGHBOR_FILTER_MIN_TRIGGER,
        )
        parameter.WAVELET_NEIGHBOR_SCORE_WAVELET_WEIGHT = rospy.get_param(
            '~wavelet_neighbor_score_wavelet_weight',
            parameter.WAVELET_NEIGHBOR_SCORE_WAVELET_WEIGHT,
        )
        parameter.WAVELET_NEIGHBOR_SCORE_UTILITY_WEIGHT = rospy.get_param(
            '~wavelet_neighbor_score_utility_weight',
            parameter.WAVELET_NEIGHBOR_SCORE_UTILITY_WEIGHT,
        )
        parameter.WAVELET_NEIGHBOR_SCORE_DISTANCE_WEIGHT = rospy.get_param(
            '~wavelet_neighbor_score_distance_weight',
            parameter.WAVELET_NEIGHBOR_SCORE_DISTANCE_WEIGHT,
        )

        # RL策略选择
        parameter.GREEDY_ACTION_SELECTION = rospy.get_param('~greedy_action_selection', parameter.GREEDY_ACTION_SELECTION)

        # 打印启用的优化功能
        self.log_optimization_settings()

        # network model file
        self.model_file = "checkpoint.pth"

        # robot coordination wrt map frame
        self.robot_location = None

        # the grid occupied by the robot
        self.robot_cell = None

        # initialize robot planner
        self.robot = None
        self.init_agent()
        self.start = None

        # waypoint
        self.next_waypoint_list = []
        self.history_waypoint_list = []
        self.next_waypoint = None

        # termination status
        self.done = False
        self.exploration_active = rospy.get_param('~start_active', False)
        self.paused = not self.exploration_active
        self.stopped = False

        # save mode
        self.save_mode = False

        # subscribers
        rospy.Subscriber('/projected_map', OccupancyGrid, self.get_map_callback, queue_size=1)
        rospy.Subscriber('/state_estimation', Odometry, self.get_loc_callback, queue_size=1)
        rospy.Subscriber('/rosclaw/exploration_control', String, self.control_callback, queue_size=1)

        # publishers
        self.waypoint_pub = rospy.Publisher('/way_point', PointStamped, queue_size=1)
        self.run_time_pub = rospy.Publisher('/runtime', Float32, queue_size=1)
        self.edge_pub = rospy.Publisher('/edge', Marker, queue_size=1)
        self.node_pub = rospy.Publisher('/node', PointCloud2, queue_size=1)
        self.frontier_pub = rospy.Publisher('/frontier', PointCloud2, queue_size=1)
        self.community_pub = rospy.Publisher('/communities', Marker, queue_size=1)
        
        # get map and robot location
        while self.map_info is None or self.robot_location is None:
            pass

        rate = rospy.Rate(20)
        rospy.Timer(rospy.Duration(1 / frequency), self.run)
        try:
            rate.sleep()
            rospy.spin()
        except KeyboardInterrupt:
            pass

    def get_map_callback(self, msg):
        t1 = time.time()
        delta = msg.info.resolution
        map_origin_x = msg.info.origin.position.x
        map_origin_y = msg.info.origin.position.y
        
        map_width = msg.info.width
        map_height = msg.info.height
        ros_map = np.array(np.array(msg.data).reshape(map_height, map_width).astype(np.int8))

        # padding the map with unknown area to avoid a frontier calculation issue
        pad_size = int(parameter.NODE_RESOLUTION // parameter.CELL_SIZE + 1)
        processed_map = np.pad(ros_map, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=parameter.UNKNOWN)
        map_origin_x -= delta * pad_size
        map_origin_y -= delta * pad_size
        robot_belief_map = processed_map

        self.map_info = MapInfo(robot_belief_map, map_origin_x, map_origin_y, delta)
        t2 = time.time()
        # print("process map using {}".format(t2 - t1))

    def get_loc_callback(self, msg):
        if self.map_info is None:
            return
        self.robot_location = np.around(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]), 1)
        if self.start is None:

            x = np.array([(self.robot_location[0] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION, (self.robot_location[0] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION])
            y = np.array([(self.robot_location[1] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION, (self.robot_location[1] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION])
            t1, t2 = np.meshgrid(x, y)
            candidate_starts = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
            dis_robot = np.linalg.norm(candidate_starts - self.robot_location, axis=1)
            sorted_candidate_starts = candidate_starts[np.argsort(dis_robot)]

            for start in sorted_candidate_starts:
                if is_free(start, self.map_info):
                    self.start = start
                    break

            assert self.start is not None, rospy.logwarn("can not find valid start point")

            self.start = np.around(self.start, 1)
            self.robot.node_manager = NodeManager(self.start)
            print("initialize quad tree at", self.start)
            print("initialize robot location at", self.robot_location)
        self.robot_cell = get_cell_position_from_coords(self.robot_location, self.map_info)

    def waypoint_wrapper(self, loc):
        way_point = PointStamped()
        way_point.header.frame_id = "map"
        way_point.header.stamp = rospy.Time.now()
        way_point.point.x = loc[0]
        way_point.point.y = loc[1]
        return way_point

    def init_agent(self):
        policy_net = PolicyNet(parameter.NODE_INPUT_DIM, parameter.EMBEDDING_DIM).to(self.device)
        model_folder = os.path.join(rospkg.RosPack().get_path('ariadne2'), 'scripts/model')
        model_file = os.path.join(model_folder, self.model_file)
        try:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_file, map_location=self.device)
        except Exception as exc:
            rospy.logwarn(
                "weights_only checkpoint load failed for trusted vendored ARiADNE2 model; "
                "retrying full torch.load: %s",
                exc,
            )
            checkpoint = torch.load(model_file, map_location=self.device)
        policy_net.load_state_dict(checkpoint['policy_model'])

        self.robot = Agent(policy_net, self.device, self.publish_graph)

    def control_callback(self, msg):
        command = msg.data.strip().lower()
        if command == 'start':
            self.exploration_active = True
            self.paused = False
            self.stopped = False
            rospy.loginfo("ROSClaw exploration control: start")
        elif command == 'pause':
            self.paused = True
            self.exploration_active = True
            rospy.loginfo("ROSClaw exploration control: pause")
        elif command == 'resume':
            self.paused = False
            self.exploration_active = True
            self.stopped = False
            rospy.loginfo("ROSClaw exploration control: resume")
        elif command == 'stop':
            self.paused = True
            self.exploration_active = False
            self.stopped = True
            rospy.loginfo("ROSClaw exploration control: stop")
        else:
            rospy.logwarn("ROSClaw exploration control: unknown command '%s'", command)

    def run(self, event=None):
        # no more planning if exploration is completed
        t1 = time.time()
        if self.done or self.stopped or not self.exploration_active or self.paused:
             return

        if self.save_mode:
            if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                return
            else:
                if len(self.next_waypoint_list) > 0:
                    next_waypoint = self.next_waypoint_list.pop(0)
                    while check_collision(self.robot_location, np.array(next_waypoint), self.map_info) is False\
                            and np.linalg.norm(self.robot_location - np.array(next_waypoint)) < (parameter.THR_NEXT_WAYPOINT + parameter.NODE_RESOLUTION)\
                            and len(self.next_waypoint_list) > 0:
                        next_waypoint = self.next_waypoint_list.pop(0)
                    self.next_waypoint = next_waypoint

                    self.history_waypoint_list.append((self.next_waypoint[0], self.next_waypoint[1]))
                    waypoint_msg = self.waypoint_wrapper(self.next_waypoint)
                    self.waypoint_pub.publish(waypoint_msg)
                    run_time = Float32()
                    run_time.data = time.time() - t1

                    # publish
                    self.run_time_pub.publish(run_time)
                    return
                else:
                    self.save_mode = False
                    rospy.logwarn("Switch back to RL")


        # check and solve oscillation between two waypoints
        if parameter.AVOID_OSCILLATION and len(self.history_waypoint_list) > 4:
            if self.history_waypoint_list[-1] == self.history_waypoint_list[-3] and self.history_waypoint_list[-2] == self.history_waypoint_list[-4]:
                self.next_waypoint_list = []
                if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                    return

        # if planned one more step, use it
        if len(self.next_waypoint_list) > 0:
            if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                pass
            else:
                self.robot_location = self.next_waypoint
                self.next_waypoint = self.next_waypoint_list.pop(0)
                waypoint_msg = self.waypoint_wrapper(self.next_waypoint)
                self.waypoint_pub.publish(waypoint_msg)
        self.next_waypoint_list = []
        # print("robot location at", self.robot_location)

        # remove nodes on obstacles if any
        self.robot.node_manager.check_valid_node(self.robot_location, self.map_info)

        # find nearest node to the robot
        robot_node_location = self.robot_location
        if self.robot_location[0] != self.start[0] or self.robot_location[1] != self.start[1]:
            if self.robot.node_manager.nodes_dict.__len__() == 0:
                robot_node_location = self.start
            else:
                nearest_node = self.robot.node_manager.nodes_dict.nearest_neighbors(self.robot_location.tolist(), 1)[0]
                node_coords = nearest_node.data.coords
                robot_node_location = node_coords

        # updating planning graph
        self.robot.update_planning_state(self.map_info, robot_node_location)

        # check the termination status
        if sum(self.robot.key_utility) == 0:
            g = "\033[92m"
            n= "\033[0m"
            rospy.loginfo(f"{g}Exploration Completed{n}")
            self.done = True
            run_time = Float32()
            run_time.data = 0
            self.run_time_pub.publish(run_time)
            return

        # get rl observation
        t2 = time.time()
        observation = self.robot.get_observation(self.robot_location)
        t3 = time.time()

        # network inference to get next waypoint
        next_location, next_node_index = self.robot.select_next_waypoint(observation)

        self.next_waypoint_list.append(next_location)
        if len(self.history_waypoint_list) > 0:
            if (next_location[0], next_location[1]) != self.history_waypoint_list[-1]:
                self.history_waypoint_list.append((next_location[0], next_location[1]))
        else:
            self.history_waypoint_list.append((next_location[0], next_location[1]))

        # planning one more step if next node's utility is zero
        if self.robot.node_manager.nodes_dict.find(next_location.tolist()).data.utility == 0:
            next_observation = self.robot.get_next_observation(next_node_index, observation)
            next_next_location, _ = self.robot.select_next_waypoint(next_observation)

            # if next waypoint is too close, go to the next next waypoint
            if np.linalg.norm(next_location - self.robot_location) < parameter.NODE_RESOLUTION:
                self.next_waypoint_list = []

            self.next_waypoint_list.append(next_next_location)

        t4 = time.time()
        # print("next waypoint at", next_location)
        # print("update planning state using {}".format(t2 - t1))
        # print("prepare tensor input using {}".format(t3 - t2))
        # print("neural network inference using {}".format(t4-t3))

        # if rl gets stuck, go to nearest frontier
        if parameter.ENABLE_SAVE_MODE:
            if self.detect_waypoint_loop():
                self.next_waypoint_list = self.robot.node_manager.path_to_nearest_frontier
                self.save_mode = True
                rospy.logwarn("Switch to save mode")

        # get waypoint message
        self.next_waypoint = self.next_waypoint_list.pop(0)
        waypoint_msg = self.waypoint_wrapper(self.next_waypoint)

        # get planning time message
        run_time = Float32()
        run_time.data = t4 - t1

        # publish
        self.run_time_pub.publish(run_time)
        self.waypoint_pub.publish(waypoint_msg)

        self.step += 1
        if self.publish_graph:
            self.visualize_graph()

    def log_optimization_settings(self):
        """打印启用的优化功能设置"""
        rospy.loginfo("=" * 50)
        rospy.loginfo("ARiADNE2 Optimization Settings:")
        rospy.loginfo("=" * 50)
        
        # 碰撞检测优化
        if parameter.SOFT_COLLISION_CHECK:
            rospy.loginfo("[ENABLED] Soft Collision Check - UNKNOWN areas are passable")
        elif parameter.ALLOW_UNKNOWN_PASSTHROUGH:
            rospy.loginfo(f"[ENABLED] Unknown Passthrough - tolerance: {parameter.UNKNOWN_TOLERANCE_CELLS} cells")
        else:
            rospy.loginfo("[DEFAULT] Standard Collision Check")
        
        # 邻居节点发现优化
        if parameter.EXTENDED_NEIGHBOR_RANGE:
            rospy.loginfo(f"[ENABLED] Extended Neighbor Range - matrix size: {parameter.NEIGHBOR_MATRIX_SIZE}x{parameter.NEIGHBOR_MATRIX_SIZE}")
        
        # 图连通性优化
        if parameter.DELAYED_NODE_REMOVAL:
            rospy.loginfo(f"[ENABLED] Delayed Node Removal - delay steps: {parameter.NODE_REMOVAL_DELAY_STEPS}")
        
        # 前沿点处理优化
        if parameter.DISTANCE_WEIGHTED_UTILITY:
            rospy.loginfo(f"[ENABLED] Distance Weighted Utility - decay: {parameter.UTILITY_DISTANCE_DECAY}")
        
        # D*-Lite优化
        if parameter.ENABLE_DSTARLITE:
            rospy.loginfo(f"[ENABLED] D*-Lite - max time: {parameter.DSTARLITE_MAX_TIME}s")
            if parameter.DSTARLITE_ADAPTIVE_TIME:
                rospy.loginfo("[ENABLED] D*-Lite Adaptive Time")
        
        # 狭窄通道检测
        if parameter.ENABLE_NARROW_PASSAGE_DETECTION:
            rospy.loginfo(f"[ENABLED] Narrow Passage Detection - threshold: {parameter.NARROW_PASSAGE_WIDTH_THRESHOLD}m")
        
        # 图连通性检查优化
        if parameter.USE_CURRENT_LOCATION_FOR_CONNECTIVITY:
            rospy.loginfo("[ENABLED] Use Current Location for Connectivity Check")

        # 小波阈值与邻居轻筛选
        if parameter.WAVELET_ADAPTIVE_DTH:
            rospy.loginfo(
                f"[ENABLED] Wavelet Adaptive DTH - alpha: {parameter.WAVELET_DTH_ALPHA}, "
                f"max_mult: {parameter.WAVELET_DTH_MAX_MULT}, local_map_size: {parameter.WAVELET_LOCAL_MAP_SIZE}m, "
                f"cache_change_ratio_thresh: {parameter.WAVELET_CACHE_CHANGE_RATIO_THRESH}"
            )
        if parameter.WAVELET_NEIGHBOR_LIGHT_FILTER:
            rospy.loginfo(
                f"[ENABLED] Wavelet Neighbor Light Filter - keep: [{parameter.WAVELET_NEIGHBOR_MIN_KEEP}, "
                f"{parameter.WAVELET_NEIGHBOR_MAX_KEEP}], trigger>{parameter.WAVELET_NEIGHBOR_FILTER_MIN_TRIGGER}, "
                f"weights(w,u,d)=("
                f"{parameter.WAVELET_NEIGHBOR_SCORE_WAVELET_WEIGHT}, "
                f"{parameter.WAVELET_NEIGHBOR_SCORE_UTILITY_WEIGHT}, "
                f"{parameter.WAVELET_NEIGHBOR_SCORE_DISTANCE_WEIGHT})"
            )
        
        # RL策略选择
        if parameter.GREEDY_ACTION_SELECTION:
            rospy.loginfo("[ENABLED] Greedy Action Selection - smoother trajectory")
        else:
            rospy.loginfo("[DEFAULT] Stochastic Action Selection - more exploratory")
        
        rospy.loginfo("=" * 50)

    def detect_waypoint_loop(self, max_length=6):
        if len(self.history_waypoint_list) < max_length:
            return False

        waypoint_list_to_check = self.history_waypoint_list[-max_length:]
        loop =[]
        for i, waypoint in enumerate(waypoint_list_to_check[:-1]):
            if waypoint == waypoint_list_to_check[-1]:
                loop = waypoint_list_to_check[i:]

        if loop:
            loop_length = len(loop)
            if len(self.history_waypoint_list) < 2 * loop_length + 1:
                return False
            waypoint_list_to_check2 = self.history_waypoint_list[-max_length-loop_length+1:-loop_length+1]
            # print("length check", waypoint_list_to_check2, loop)
            loop2 = []
            for i, waypoint in enumerate(waypoint_list_to_check2[:-1]):
                if waypoint == waypoint_list_to_check2[-1]:
                    loop2 = waypoint_list_to_check2[i:]
                    break
            if loop2:
                return True
            else:
                return False

    def visualize_graph(self):
        # visualize edges
        edges = Marker()
        edges.header.frame_id = 'map'
        edges.header.stamp = rospy.Time.now()
        edges.type = Marker.LINE_LIST
        edges.scale.x = 0.1
        edges.color.r = 0.0
        edges.color.g = 0.6
        edges.color.b = 0.0
        edges.color.a = 1.0
        edges.pose.orientation.x = 0.0
        edges.pose.orientation.y = 0.0
        edges.pose.orientation.z = 0.0
        edges.pose.orientation.w = 1.0

        for coords in self.robot.key_node_coords:
            node = self.robot.node_manager.key_node_dict[(coords[0], coords[1])]
            for neighbor_coords in node.neighbor_set:
                start = Point()
                start.x = coords[0]
                start.y = coords[1]
                end_coords = (neighbor_coords - coords) / 2 + coords
                end = Point()
                end.x = end_coords[0]
                end.y = end_coords[1]
                edges.points.append(start)
                edges.points.append(end)

        self.edge_pub.publish(edges)

        # visualize nodes
        nodes = []
        for node_coords, utility in zip(self.robot.key_node_coords, self.robot.key_utility):
            nodes.append((node_coords[0], node_coords[1], 0.0, utility))
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        nodes = point_cloud2.create_cloud(header, fields, nodes)
        self.node_pub.publish(nodes)

        # visualize frontiers
        frontiers = []
        for frontier in self.robot.frontier:
            frontiers.append((frontier[0], frontier[1], 0))
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        frontiers = point_cloud2.create_cloud(header, fields, frontiers)
        self.frontier_pub.publish(frontiers)
        
        # visualize communities (cluster centers)
        communities = Marker()
        communities.header.frame_id = 'map'
        communities.header.stamp = rospy.Time.now()
        communities.type = Marker.SPHERE_LIST
        communities.scale.x = 4.0  # 直径4米的球体，更显眼
        communities.scale.y = 4.0
        communities.scale.z = 4.0
        communities.color.r = 1.0
        communities.color.g = 0.6
        communities.color.b = 0.0
        communities.color.a = 0.6  # 半透明
        communities.pose.orientation.x = 0.0
        communities.pose.orientation.y = 0.0
        communities.pose.orientation.z = 0.0
        communities.pose.orientation.w = 1.0
        
        # 添加所有聚类中心点
        for center_coords in self.robot.node_manager.cluster_center_node_dict.keys():
            point = Point()
            point.x = center_coords[0]
            point.y = center_coords[1]
            point.z = 2.0  # 抬高2米以便看清
            communities.points.append(point)
        
        self.community_pub.publish(communities)
        

if __name__ == '__main__':
    rospy.init_node('rl_planner', anonymous=True)
    rl_runner = Runner()
