import open3d as o3d
import numpy as np
import time




def draw_registration_result_original_color(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp.to_legacy(), target_temp.to_legacy()],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])
    
def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size*2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size*5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

voxel_size = 0.025

source = o3d.t.io.read_point_cloud("1.ply")
target = o3d.t.io.read_point_cloud("3.ply")


# Initial guess transform between the two point-cloud.
# ICP algortihm requires a good initial allignment to converge efficiently.
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])


#draw_registration_result(source, target, trans_init)
draw_registration_result_original_color(source, target, trans_init)


# Initial alignment or source to target transform.
init_source_to_target = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                    [-0.139, 0.967, -0.215, 0.7],
                                    [0.487, 0.255, 0.835, -1.4],
                                    [0.0, 0.0, 0.0, 1.0]])

# Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()


# Vanilla ICP


# Example callback_after_iteration lambda function:
callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    updated_result_dict["iteration_index"].item(),
    updated_result_dict["fitness"].item(),
    updated_result_dict["inlier_rmse"].item()))

# Search distance for Nearest Neighbour Search [Hybrid-Search is used].
max_correspondence_distance = 0.07

# Convergence-Criteria for Vanilla ICP
criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                       relative_rmse=0.000001,
                                       max_iteration=50)

# Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
save_loss_log = True

s = time.time()


registration_icp = o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance,
                                init_source_to_target, estimation, criteria,
                                voxel_size, callback_after_iteration)



icp_time = time.time() - s
print("Time taken by ICP: ", icp_time)
print("Inlier Fitness: ", registration_icp.fitness)
print("Inlier RMSE: ", registration_icp.inlier_rmse)

draw_registration_result_original_color(source, target, registration_icp.transformation)
