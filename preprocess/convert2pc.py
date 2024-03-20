#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2

class DepthToPointCloud:
    def __init__(self):
        self.bridge = CvBridge()
        self.pointcloud_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
        self.depth_sub = rospy.Subscriber('/zed2i/zed_node/depth/depth_registered', Image, self.depth_callback)

        self.fx = 952/2 # 相机的焦距 x
        self.fy = 953/2  # 相机的焦距 y
        self.cx = 636/2  # 相机的光心 x
        self.cy = 359/2  # 相机的光心 y

    def depth_callback(self, msg):
        try:
            # 使用cv_bridge将ROS图像消息转换为numpy数组
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(e)
            return

        # 将深度图像中的NaN值替换为0
        depth_image = np.nan_to_num(depth_image)

        # 将深度值从毫米转换为米
        depth_image = depth_image

        # 生成网格矩阵
        x = np.load("/home/kemove/yyz/av-gihub/av-ped/Data/X.npy")
        y = np.load("/home/kemove/yyz/av-gihub/av-ped/Data/Y.npy")
        # u_grid, v_grid = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
        # 将像素坐标转换为相机坐标系下的坐标
        Z = depth_image
        X = x*Z
        Y = y*Z

        # 进行异常值检查
        mask = np.logical_and.reduce((X >= -5, X <= 5, Y >= -1, Y <= 1, Z >= 0, Z <= 10))  # 假设合理的范围是 (-10, 10) 米
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        # 将三维坐标堆叠成点云
        points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

        # 将点云数据转换为PointCloud2消息
        header = msg.header
        header.frame_id = "camera_link"  # 设置坐标系为相机坐标系，你可能需要根据实际情况调整

        # 检查是否有足够的点构成点云
        if len(points) < 1:
            rospy.logwarn("Not enough valid points to create point cloud.")
            return

        pc_msg = pc2.create_cloud_xyz32(header, points)

        # 发布点云消息
        self.pointcloud_pub.publish(pc_msg)

def main():
    rospy.init_node('depth_to_pointcloud', anonymous=True)
    depth_to_pointcloud = DepthToPointCloud()
    rospy.spin()

if __name__ == '__main__':
    main()
