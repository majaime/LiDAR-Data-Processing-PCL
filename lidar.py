# This code reads a .bin file, stores the data in a .csv file, and 
# processes (filtering, segmenting, & clustering) LiDAR point
# cloud data using "PCL" library in Python.

""" Two filters for downsampling are represented. I used voxel for segmentation.
    You may choose whichever you want, voxel or statistical outlier"""
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
import pcl.pcl_visualization


# Reading data
inputPath = 'PATH_TO_YOUR_INPUT_BIN_FILE/filename.bin'
outputPath = 'WHERE_YOU_WANT_TO_SAVE_YOUR_OUTPUT/filename.csv'

num = np.fromfile(inputPath, dtype='float32', count=-1, sep='', offset=0)
new = np.asarray(num).reshape(-1, 4)

X = num[0::4]
Y = num[1::4]
Z = num[2::4]
W = num[3::4]

# Storing the encrypted (.bin) file in (.csv) file
for i in range(0, len(new)):
    print(new[i])
    with open(outputPath, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(new[i])

# Creating point cloud
xyz = np.zeros((np.size(X), 3))
xyz[:, 0] = np.reshape(X, -1)
xyz[:, 1] = np.reshape(Y, -1)
xyz[:, 2] = np.reshape(Z, -1)
pc = pcl.PointCloud()
pcd = pc.from_list(xyz)
# Downsampling - Voxel filter
voxel = pc.make_voxel_grid_filter()
voxel.set_leaf_size(0.05, 0.05, 0.05)  # Leaf sizes are in meters
# The bigger leaf sizes are the more samples are combined and the more dot-ish the image becomes
voxel_filter = voxel.filter()
# Downsampling - Statistical outlier filter
outlier = pc.make_statistical_outlier_filter()
outlier.set_mean_k(30)
outlier.set_std_dev_mul_thresh(0.1)
outlier_filter = outlier.filter()
# Segmentation
seg = voxel_filter.make_segmenter()
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_MaxIterations(1000)
seg.set_distance_threshold(1)
inliers, plane_model = seg.segment()
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
inlier_cloud = outlier_filter.extract(inliers, negative=False)
outlier_cloud = outlier_filter.extract(inliers, negative=True)
# Clustering using KdTree and Euclidean distance
tree = voxel_filter.make_kdtree()
ec = voxel_filter.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.05)
ec.set_MinClusterSize(100)
ec.set_MaxClusterSize(25000)
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()
color_cluster_point_list = []
cloud_cluster = pcl.PointCloud()
for j, indices in enumerate(cluster_indices):
    points = np.zeros((len(indices), 3), dtype=np.float32)
    for i, indice in enumerate(indices):
        points[i][0] = pc[indice][0]
        points[i][1] = pc[indice][1]
        points[i][2] = pc[indice][2]

    cloud_cluster.from_array(points)
#   ss = "cloud_cluster_" + str(j) + ".pcd"; # If you want to save yout output as .psd
#   pcl.save(cloud_cluster, ss)

# Visualization
visual = pcl.pcl_visualization.CloudViewing()
visual.ShowMonochromeCloud(cloud_cluster)
