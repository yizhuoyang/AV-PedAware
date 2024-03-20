import os
import open3d
import open3d as o3d
import numpy as np
import numpy as np 
import os,sys
import cv2
# import torch

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """

    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d



def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True,json_path='/home/lx/yyz/Audio_experiment/trans.json',save_path='/home/lx/yyz/Multimodal_exp/Initial_ex_data/vis_result',lidar_path='',vis=1):

    save_name = save_path+'/'+lidar_path[:-4]+'.jpg'
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    # pts.paint_uniform_color([1,1,1])
    vis.add_geometry(pts)
#     if point_colors is None:
#         pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
#     else:
#         pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    # parameter  = o3d.io.read_pinhole_camera_parameters(json_path)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.82)
    ctr.set_lookat(np.array([ -0.69970297813415527, -5.8943835496902466, 0.17353767156600949]))
    ctr.set_up(np.array( [0.62906626293991286, 0.52728664745103815, 0.57117810554209536 ]))
    ctr.set_front((np.array([-0.49192068948081097, -0.29891284659222156, 0.8177194784294054])))
    # ctr.convert_from_pinhole_camera_parameters(parameter)

    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(save_name)

    vis.run()
    vis.destroy_window()


def draw_scenes_no_vis(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True,json_path='/home/lx/yyz/Audio_experiment/trans.json',save_path='/home/lx/yyz/Multimodal_exp/Initial_ex_data/vis_result',lidar_path='',vis=1):

    save_name = save_path+'/'+lidar_path[:-4]+'.jpg'
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
#     if point_colors is None:
#         pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
#     else:
#         pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    # parameter  = o3d.io.read_pinhole_camera_parameters(json_path)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.82)
    ctr.set_lookat(np.array([ -0.69970297813415527, -5.8943835496902466, 0.17353767156600949]))
    ctr.set_up(np.array( [0.62906626293991286, 0.52728664745103815, 0.57117810554209536 ]))
    ctr.set_front((np.array([-0.49192068948081097, -0.29891284659222156, 0.8177194784294054])))
    # ctr.convert_from_pinhole_camera_parameters(parameter)

    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(save_name)

    vis.destroy_window()



def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(color)

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def plot_gt(test_lab,test_lidar,vis=1):
    gt         = test_lab.reshape(1,7)
    pc = test_lidar
    x_range = pc[:,0]
    x_range[x_range<-8.1]=-8.1
    pc[:,0] = x_range

    draw_scenes(pc,gt,gt,'person',1,vis=vis)# 3维点云

def plot_result(test_lab,predict_lab,test_lidar,vis=1):
    gt         = test_lab.reshape(1,7)
    result     = predict_lab.reshape(1,7)
    pc = test_lidar
    x_range = pc[:,0]
    x_range[x_range>4]=4
    x_range[x_range<-4]=-4
    pc[:,0] = x_range
    z_range = pc[:,2]
    z_range[z_range>1.2]=z_range.min()
    pc[:,2] = z_range
    y_range = pc[:,1]
    y_range[y_range<-6.0]=-6.0
    pc[:,1] = y_range


    draw_scenes(pc,gt,result,'person',1,vis=vis)# 3维点云
