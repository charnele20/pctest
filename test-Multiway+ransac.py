import os
import open3d as o3d
import numpy as np
import time




# Preprocess
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #pcd_down = o3d.geometry.PointCloud()
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals()

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        radius=voxel_size * 5.0,
        max_nn=100)
    return (pcd_down, pcd_fpfh)

# Input
def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    pcd_down_list = []
    pcd_fpfh_list = []
    pcd_path = os.listdir('project/')

    for i in range (len(pcd_path)):    
        pcd = o3d.t.io.read_point_cloud('project/%d.pts'% i)
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        pcd.transform(trans_init)
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
        pcd_down_list.append(pcd_down)
        pcd_fpfh_list.append(pcd_fpfh)
    return pcd, pcd_down_list, pcd_fpfh_list, pcd_path

# Registration
def execute_global_registration(pcd_down, result_down, pcd_fpfh,
                                result_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_down, result_down, pcd_fpfh, result_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return pcd_down, result

def pairwise_registration(pcd, result, ransac_transformation):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        pcd, result, max_correspondence_distance_coarse, ransac_transformation,
        o3d.pipelines.registration.TransformationEstimationPointtoPlane(),save_loss_log)
    icp_fine = o3d.pipelines.registration.registration_icp(
        pcd, result, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),save_loss_log)
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd, result, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcd_path,pcd_down, pcd_fpfh,max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcd_path)

    for pcd_id in range(n_pcds):
        for result_id in range(pcd_id + 1, n_pcds):
            print ("::RANSAC Optimization")
            ransac = execute_global_registration(pcd_down[pcd_id], pcd_down[result_id], pcd_fpfh[pcd_id], pcd_fpfh[result_id],voxel_size)
            rs = time.time()
            ransac_time = time.time() - rs
            print("Time taken by ICP: ", ransac_time)
            transformation_icp, information_icp = pairwise_registration(pcd_down[pcd_id], pcd_down[result_id], ransac.transformation)
            ps = time.time()
            icp_time = time.time() - ps
            print("Time taken by ICP: ", icp_time)
            print("Build o3d.pipelines.registration.PoseGraph")
            if pcd_id == result_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(pcd_id,
                                                             result_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(pcd_id,
                                                             result_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph




# Iteration Multiway

voxel_size = 0.02

pcds, pcd_down, pcd_fpfh, pcd_path = prepare_dataset(voxel_size)
                              
save_loss_log = True




max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
print("Full registration ...")
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcd_path, pcd_down, pcd_fpfh,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

#pcd_down.append(pcd_down[i].transform(pose_graph.nodes[i].pose))
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

pose = pose_graph()
dir_result = os.mkdir ('\\result')
for i in range (len(pose)):
    filename = str(i)+'.pts'
    o3d.t.io.write_point_cloud(os.path.join(dir_result,filename), pose[i], write_ascii=True)
    pcd_down[i].transform(pose_graph.nodes[i].pose)
    if i == (len(pose))+1:
        print("No items left.")
        break



o3d.visualization.draw(pcd_down)