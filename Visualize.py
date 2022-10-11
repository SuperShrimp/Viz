import matplotlib.pyplot as plt
import open3d
import numpy as np
from scipy.spatial.transform import Rotation
slope ='clear'
pcl = np.load("slope_data/pcl_%s.npy"%slope)
box = np.load("slope_data/box_%s.npy"%slope)
attr_map = np.load("slope_data/attr_%s.npy"%slope)
#--------------------label读取-------------------#
label =[1, 1, 1, 1, 3, 1, 3, 3, 3, 3, 2, 2, 3, 1, 3]
label_color = [[1,0,0], [0,1,0], [0,1,1]]
viewpoint = "position.json"

def visualize_attr_map(points, box, attr_map, draw_origin=True):
    turbo_cmap = plt.get_cmap('turbo')
    attr_map_scaled = attr_map - attr_map.min()
    attr_map_scaled /= attr_map_scaled.max()
    color = turbo_cmap(attr_map_scaled)[:, :3]
    print("shape of color: ",color.shape)
    np.savetxt("color", color)

    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1440, height=1080)

    vis.get_render_option().point_size = 4.0
    vis.get_render_option().background_color = np.ones(3) * 0.25

    #-------------------加入坐标系-------------------#
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    #-----------------加入检测框-----------------------#
    rot_mat = Rotation.from_rotvec([0, 0, box[0, 6]]).as_matrix()
    detections, data = box.shape
    for i in range(detections):
        box1 = box[i]
        label_bbox= label[i]
        color_bbox = label_color[label_bbox-1]
        bbox = open3d.geometry.OrientedBoundingBox(box1[:3], rot_mat, box1[3:6])
        bbox.color = (color_bbox[:])
        vis.add_geometry((bbox))

    #-------------------------固定视角---------------------#
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.io.write_pinhole_camera_parameters(viewpoint, param)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(viewpoint)
    ctr.convert_from_pinhole_camera_parameters(param)


    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(color)
    vis.add_geometry(pts)


    vis.run()
    vis.destroy_window()


visualize_attr_map(pcl, box, attr_map[0, :])


def graphic_attr(attr_map):
    plt.figure(figsize=(20, 10), dpi = 100)
    picture = plt.plot(attr_map)
    return picture

