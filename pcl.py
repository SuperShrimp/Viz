import numpy as np
import open3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import math
attr_map = np.load("source_data_22_7_21_1140/attri_map_22_7_21_1140.npy")
#pcl = np.load("source_data_22_7_21_1140/pcl_22_7_21_1140.npy")
pcl = np.load("source_data_22_7_21_1127/pcl_22_7_21_1127.npy")
box = np.load("source_data_22_7_21_1140/base_det_boxes_22_7_21_1140.npy")

def dele_attr(attri_map, level):
    '''
    删掉一部分只留Attr中的一部分，并且data里面的索引可以接到pcl上
    '''
    max_attr = np.array(attri_map).max()
    min_attr = np.array(attri_map).min()

    level_attr = min_attr + (max_attr - min_attr)*level
    point_list = np.zeros(len(attri_map))
    data = np.zeros(len(attri_map))
    for i in range(len(attri_map)):
        if(attri_map[i] > level_attr):
           data = np.append(data, attri_map[i])
           point_list = np.append(point_list, i)
    print(point_list)
    return np.array(data), point_list

def new_pointcloud(pcl, pointlist):
    '''
    根据attr的结果构造一个新的点云输出
    '''

    new_pts = []
    for i in range(len(pointlist)):
        index = pointlist[i]
        index = np.int(index)
        new_pts = np.append(new_pts, pcl[index])
    new_pts = new_pts.reshape(len(pointlist), 4)
    print(new_pts.shape)
    return new_pts

def point_crop(pcl, bounding_box):
    '''
    通过bounding_box计算八个边界点，把边界点内的点云剪裁出来
    自己写的，用处并不是很大
    '''
    print(pcl.shape)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcl[:, :3])

    cx = bounding_box[0]
    print(cx)
    cy = bounding_box[1]
    print(cy)
    cz = bounding_box[2]
    print(cz)
    dx = bounding_box[3]
    dy = bounding_box[4]
    dz = bounding_box[5]
    alpha = bounding_box[6]

    #确定边界点坐标

    x1 = [cx + dy * math.sin(alpha) / 2 + dx * math.cos(alpha) / 2,
          cy - dy * math.cos(alpha) / 2 + dx * math.sin(alpha) / 2, cz - dz / 2]
    x2 = [cx + dy * math.sin(alpha) / 2 - dx * math.cos(alpha) / 2,
          cy - dy * math.cos(alpha) / 2 - dx * math.sin(alpha) / 2, cz - dz / 2]
    x3 = [cx - dy * math.sin(alpha) / 2 + dx * math.cos(alpha) / 2,
          cy + dy * math.cos(alpha) / 2 + dx * math.sin(alpha) / 2, cz - dz / 2]
    x4 = [cx - dy * math.sin(alpha) / 2 - dx * math.cos(alpha) / 2,
          cy + dy * math.cos(alpha) / 2 - dx * math.sin(alpha) / 2, cz - dz / 2]
    x5 = [cx + dy * math.sin(alpha) / 2 + dx * math.cos(alpha) / 2,
          cy - dy * math.cos(alpha) / 2 + dx * math.sin(alpha) / 2, cz + dz / 2]
    x6 = [cx + dy * math.sin(alpha) / 2 - dx * math.cos(alpha) / 2,
          cy - dy * math.cos(alpha) / 2 - dx * math.sin(alpha) / 2, cz + dz / 2]
    x7 = [cx - dy * math.sin(alpha) / 2 + dx * math.cos(alpha) / 2,
          cy + dy * math.cos(alpha) / 2 + dx * math.sin(alpha) / 2, cz + dz / 2]
    x8 = [cx - dy * math.sin(alpha) / 2 - dx * math.cos(alpha) / 2,
          cy + dy * math.cos(alpha) / 2 - dx * math.sin(alpha) / 2, cz + dz / 2]

    #bounding_poly = np.array([x1, x2, x3, x4, x5, x6, x7, x8], dtype = np.float32).reshape([-1, 3]).astype("float64")
    bounding_poly = [x1, x2, x3, x4]
    #bbox = np.array(bounding_poly)



    bounding_poly = np.array(bounding_poly)
    bounding_polygon = bounding_poly.astype("float64")

    vol = open3d.visualization.SelectionPolygonVolume()

    vol.orthogonal_axis = "Z"
    vol.axis_max = cz + dz /2
    vol.axis_min = cz- dz /2


    #print(bounding_polygon)

    vol.bounding_polygon = open3d.utility.Vector3dVector(bounding_polygon)
    cropped_pcd = vol.crop_point_cloud(pcd)

    #print(cropped_pcd)
    vis=open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.ones(3) * 0.25
    vis.add_geometry(cropped_pcd)

    rot_mat = Rotation.from_rotvec([0, 0, alpha]).as_matrix()
    bb = open3d.geometry.OrientedBoundingBox(bounding_box[:3], rot_mat, bounding_box[3:6])
    bb.color = (1.0, 0.0, 1.0)
    vis.add_geometry(bb)

    vis.run()



def visualize_attr_map(points, box, attr_map, draw_origin=True):
    turbo_cmap = plt.get_cmap('turbo')
    attr_map_scaled = attr_map - attr_map.min()
    attr_map_scaled /= attr_map_scaled.max()
    color = turbo_cmap(attr_map_scaled)[:, :3]

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 4.0
    vis.get_render_option().background_color = np.ones(3) * 0.25

    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
    bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
    bb.color = (1.0, 0.0, 1.0)
    vis.add_geometry(bb)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(color)
    color1 = np.asarray(pts.colors)
    cropped_pcd = pts.crop(bb)
    crop_points = np.array(cropped_pcd.points)
    print(crop_points.shape)
    #open3d.visualization.draw_geometries([cropped_pcd])

    # vis.run()
    # vis.destroy_window()
#先得到一个新的attr


attr_map1, pt_list = dele_attr(attr_map[0, :], level=0.7)
new_pcl = new_pointcloud(pcl, pt_list)
visualize_attr_map(new_pcl, box[0, :], attr_map1)
