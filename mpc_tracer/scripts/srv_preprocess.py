#! /usr/bin/python2.7
# -*- coding: utf-8 -*-
from time import time
import numpy as np
# from tf import transformations
import transformations
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
import rospy

import struct
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

class preprocess:
    def __init__(self):
        #调用预处理
        self.pub_map = rospy.Publisher('my_map',OccupancyGrid,queue_size=1)
        self.pub_cloud = rospy.Publisher("point_cloud2", PointCloud2, queue_size=1)

    def process(self,gridmap,current,vel,reference_path):
        self.map_process(gridmap)
        self.current_process(current)
        self.vel_process(vel)
        self.reference_path_process(reference_path)

    #预处理地图
    def map_process(self,gridmap):
        self.saved_map = gridmap
        self.height=gridmap.info.height
        self.width=gridmap.info.width
        self.col_map=np.zeros((self.height,self.width))

        for x in range(self.width):
            for y in range(self.height):
                self.col_map[x][y]=gridmap.data[x+y*self.width]

        self.resolution=gridmap.info.resolution
        self.step_size = (self.height+self.width)/15
        self.end_lim = (self.height+self.width)/15
        #地图坐标系左下角位置
        self.gridmap_position_x=gridmap.info.origin.position.x
        self.gridmap_position_y=gridmap.info.origin.position.y

    #预处理机器人当前位姿
    def current_process(self,current):
        current_x,current_y=[current.pose.position.x, current.pose.position.y]
        _, _, current_yaw = transformations.euler_from_quaternion(
            [current.pose.orientation.x, current.pose.orientation.y, current.pose.orientation.z,
             current.pose.orientation.w])
        self.current_pose = [current_x,current_y, current_yaw]

    #预处理机器人目标位姿
    def vel_process(self,vel):
        self.vel = [vel.linear.x, vel.angular.z]

    def reference_path_process(self, reference_path):
        self.local_goal= [reference_path.poses[-1].pose.position.x, reference_path.poses[-1].pose.position.y]
        self.refer_size = len(reference_path.poses)
        self.refer_x = [0]*100
        self.refer_y = [0]*100
        self.refer_ind = [0]*100
        for i in reference_path.poses:
            self.refer_x[reference_path.poses.index(i)] = i.pose.position.x
            self.refer_y[reference_path.poses.index(i)] = i.pose.position.y
            self.refer_ind[reference_path.poses.index(i)] = reference_path.poses.index(i)+1
        self.refer_x = self.refer_x[:self.refer_size]
        self.refer_y = self.refer_y[:self.refer_size]
        self.refer_ind = self.refer_ind[:self.refer_size]


    def getCost(self, x_world, y_world):
        x_map, y_map = self.world2map(x_world, y_world)
        return self.col_map[int(x_map)][int(y_map)]

    #world坐标系到map坐标系转换
    def world2map(self,w_x,w_y):
        m_x=0
        m_y=0
        if self.resolution!=0:
            m_x=(w_x-self.gridmap_position_x)/self.resolution
            m_y=(w_y-self.gridmap_position_y)/self.resolution
        return m_x,m_y

    #map坐标系到world坐标系转换
    def map2world(self,m_x,m_y):
        w_x=0
        w_y=0
        if self.resolution!=0:
            w_x=self.gridmap_position_x + m_x*self.resolution
            w_y=self.gridmap_position_y + m_y*self.resolution
        return w_x,w_y

    def pubMap(self):
        saved_map = self.saved_map
        saved_map.header.stamp = rospy.Time.now()
        flatten_map = []
        for y in range(self.height):
            for x in range(self.width):
                # flatten_map.append(self.col_map[x][y])
                flatten_map.append(min(100,self.col_map[x][y]))
        # print(flatten_map)
        # flatten_map = tuple(flatten_map)
        saved_map.data = flatten_map

        self.pub_map.publish(saved_map)

        cloud_point_list = []
        for y in range(self.height):
            for x in range(self.width):
                if self.col_map[x][y]!=0:
                    w_x, w_y = self.map2world(x,y)
                    cloud_point_list.append([w_x,w_y,self.col_map[x][y]])
        self.pointCloudPub(cloud_point_list)

    def pointCloudPub(self, list):
        # list = [(x,y,z), ...]
        points = []
        max_z = max([list[i][2] for i in range(len(list))])
        for point in list:
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])/max_z*1.5
            r = int(z*100.0)
            g = 10
            b = 10
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            # print(hex(rgb))
            pt = [x, y, z, rgb]
            points.append(pt)

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                # PointField('rgb', 12, PointField.UINT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1),
                ]

        # print points

        header = Header()
        header.frame_id = "map"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.pub_cloud.publish(pc2)