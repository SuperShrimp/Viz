import matplotlib.pyplot as plt
import open3d
import numpy as np
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D

pcl = np.load("source_data_22_7_21_1127/pcl_22_7_21_1127.npy")
box = np.load("source_data_22_7_21_1127/base_det_boxes_22_7_21_1127.npy")
attr_map = np.load("source_data_22_7_21_1127/attri_map_22_7_21_1127.npy")
label = np.load("source_data_22_7_21_1127/base_det_labels_22_7_21_1127.npy")

print(pcl.shape)
print(attr_map.shape)
print(attr_map[:, 1])
# np.savetxt("pcl", pcl)
# np.savetxt("attr_map", attr_map)
print(label)
print("box: ", box)


def visualize_attr_map(points, box, attr_map, draw_origin=True):
    turbo_cmap = plt.get_cmap('turbo')
    attr_map_scaled = attr_map - attr_map.min()
    attr_map_scaled /= attr_map_scaled.max()
    color = turbo_cmap(attr_map_scaled)[:, :3]
    print("shape of color: ", color.shape)
    # np.savetxt("color", color)

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

    bbox = np.asarray(bb)
    print(bbox)

    bb.color = (1.0, 0.0, 1.0)
    #vis.add_geometry(bb)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(color)
    color1 = np.asarray(pts.colors)
    color_x = color[:, 0]
    color_y = color[:, 1]
    color_z = color[:, 2]
    fig = plt.figure()

    #---------------散点图-----------------
    ax = plt.subplot(111, projection='3d')
    ax.set_title('3d color distribute')
    ax.scatter(color_x, color_y, color_z, c ='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    np.save("color_1127",color)


    #--------折线图------------------#
    #print(len(color1))
    # x = np.zeros(len(color1))
    # y = np.zeros(len(color1))
    # for j in range(0, len(color1)):
    #     y[j] = color1[j,0]**2 + color1[j,1]**2+color1[j, 2]**2
    #     x[j] = j
    # plt.plot(x, y, color='#4169E1', alpha=0.8, linewidth=1)
    # plt.xlabel("points")
    # plt.ylabel("distance of color")
    # plt.show()

visualize_attr_map(pcl, box[0, :], attr_map[0, :])
