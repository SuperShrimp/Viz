import matplotlib.pyplot as plt
import open3d
import numpy as np
from scipy.spatial.transform import Rotation
pcl = np.load("source_data_22_7_21_1140/pcl_22_7_21_1140.npy")
box = np.load("source_data_22_7_21_1140/base_det_boxes_22_7_21_1140.npy")
attr_map = np.load("source_data_22_7_21_1140/attri_map_22_7_21_1140.npy")
label = np.load("source_data_22_7_21_1140/base_det_labels_22_7_21_1140.npy")
# pcl = np.load("source_data_22_7_21_1127/pcl_22_7_21_1127.npy")
# box = np.load("source_data_22_7_21_1127/base_det_boxes_22_7_21_1127.npy")
# attr_map = np.load("source_data_22_7_21_1127/attri_map_22_7_21_1127.npy")
# label = np.load("source_data_22_7_21_1127/base_det_labels_22_7_21_1127.npy")

print(pcl.shape)
print(attr_map.shape)
print(attr_map[:,1])
# np.savetxt("pcl", pcl)
# np.savetxt("attr_map", attr_map)
print(label)
print("box: ",box)

def visualize_attr_map(points, box, attr_map, draw_origin=True):
    turbo_cmap = plt.get_cmap('turbo')
    attr_map_scaled = attr_map - attr_map.min()
    attr_map_scaled /= attr_map_scaled.max()
    color = turbo_cmap(attr_map_scaled)[:, :3]
    print("shape of color: ",color.shape)
    #np.savetxt("color", color)

    #vis = open3d.visualization.Visualizer()
    #vis.create_window()

    #vis.get_render_option().point_size = 4.0
    #vis.get_render_option().background_color = np.ones(3) * 0.25

    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        #vis.add_geometry(axis_pcd)

    rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
    bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
    bb.color = (1.0, 0.0, 1.0)
    #vis.add_geometry(bb)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(color)
    #color1 = np.asarray(pts.colors)
    #print((color1))
    #vis.add_geometry(pts)
    cropped_pcd = pts.crop(bb)
    #open3d.visualization.draw_geometries(cropped_pcd)
    points2 = cropped_pcd.points
    points2 = np.array(points2)
    pts2 = open3d.geometry.PointCloud()
    pts2.points = open3d.utility.Vector3dVector(points2[:, :3])
    open3d.visualization.draw_geometries([pts2])

    #vis.run()
    #vis.destroy_window()


visualize_attr_map(pcl, box[0, :], attr_map[0, :])
