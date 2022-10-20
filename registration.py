'Registration Point Cloud with Python Packages'

#Packages
from ctypes.wintypes import tagRECT
from imp import source_from_cache
import numpy as np
import copy
import open3d as o3d

# Iterative Closest Point (ICP)
# Given two sets of points P and Q, ICP optimizes the rigid transformation to align P with Q
# The ICP starts with an initial alignment and iterates between two steps
#   The correspondence step — find the closest point q_i in Q for each point p_j in P
#   The alignment step — update the transformation by minimizing the L2 distance between the corresponding points

class icp:
    def __init__(self, source, target, transformation):
        self.source = source
        self.target = target
        self.transformation = transformation

    #visualize registration
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        #to change the color of points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    #input
    def input(source,target, threshold):
        source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
        threshold = ()
        trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                [-0.139, 0.967, -0.215, 0.7],
                                [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
        draw_registration_result(source, target, trans_init)
        
        evaluation = o3d.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
        return ("Initial Alignment", evaluation, trans_init)
        
    #registration
    def p2p(source, target, threshold, trans_init):
        reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init, o3d.registration.TransformationEstimationPointToPoint(),ICPConvergenceCriteria(max_iteration = 2000))
        transform = draw_p2p.registration_result (source, target, reg_p2p.transformation)

        return (reg_p2p, 
        "Transformation is:", reg_p2p.transformation, transform)

    def  p2l(source, target, threshold, trans_init):
        reg_p2l = o3d.registration.registration_icp(source, target, threshold, trans_init, o3d.registration.TransformationEstimationPointToPlane())
        draw_registration_result(source, target, reg_p2l.transformation)
        transform = draw_p2p.registration_result (source, target, reg_p2p.transformation)
        return (reg_p2l,
        "Transformation is:", reg_p2l.transformation, transform)

    #xyz to open3d
    def read ():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3Vector (xyz)
        o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    #save and visualize
    def write ():
        pcd_load = o3d.io.read_point_cloud("../../TestData/sync.ply")
        o3d.visualization.draw_geometries([pcd_load])

    #open3d to numpy
    def load(pcd_load):
        xyz_load = np.asarray(pcd_load.points)
        return (xyz_load)

# Global optimal matching (Using Bipartite Graph and KM algorithm)


# Hybrid metrics (Using Euclidean distance and feature distance at the same time)


# Coherent Point Drift (CPD)
# Coherent point drift is a probabilistic approach for the registration problen, which works on both rigid and non-rigid transforms
# 
#class registration 
