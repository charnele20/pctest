from mimetypes import init
import open3d as o3d
import copy
import numpy as np


#import registration


source = o3d.io.read_point_cloud("2.ply")
'''
target = copy.deepcopy(source)
R = source.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
target.rotate(R, center=(0, 0, 0))
#o3d.visualization.draw_geometries([target, target_r],zoom=0.3412,front=[0.4257, -0.2125, -0.8795],lookat=[2.6172, 2.0475, 1.532],up=[-0.0694, -0.9768, 0.2024])

o3d.io.write_point_cloud("3.ply", target)
'''
target = o3d.io.read_point_cloud("3.ply")
threshold = 0.02
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],[-0.139, 0.967, -0.215, 0.7],[0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
#pose = o3d.pipelines.registration.PoseGraphEdge ()
#method = o3d.pipelines.registration.GlobalOptimizationMethod.GlobalOptimizationGaussNewton()
#criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
#option = o3d.pipelines.registration.GlobalOptimizationOption()
#global = o3d.pipelines.registration.global_optimization(pose,method,criteria,option)



def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        #to change the color of points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        return (o3d.visualization.draw([source_temp, target_temp]))
        
#o3d.visualization.draw ([source])
draw_registration_result(source, target, trans_init)


evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                threshold, trans_init)
#print ("Initial Alignment", evaluation, trans_init)


'''
#registration p2p
reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init, o3d.registration.TransformationEstimationPointToPoint(),ICPConvergenceCriteria(max_iteration = 4000))
transform = draw_p2p.registration_result (source, target, reg_p2p.transformation)

print (reg_p2p, 
"Transformation is:", reg_p2p.transformation, transform)


#registration p2l
reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.registration.TransformationEstimationPointToPlane())
draw_p2p = draw_registration_result(source, target, reg_p2l.transformation)
#transform = draw_p2p.pipelines.registration_result (source, target, reg_p2p.transformation)
print (reg_p2l)
print ("Transformation is:", reg_p2l.transformation)
'''

#registration ransac
source, source_feature = o3d.utility.process_point_cloud(point_cloud=source,
                                                    downsample=o3d.utility.DownsampleTypes.VOXEL,
                                                    downsample_factor=0.1,
                                                    compute_feature=True,
                                                    search_param_knn=1000,
                                                    search_param_radius=0.5)

target, target_feature = o3d.utility.process_point_cloud(point_cloud=target,
                                                    downsample=o3d.utility.DownsampleTypes.VOXEL,
                                                    downsample_factor=0.1,
                                                    compute_feature=True,
                                                    search_param_knn=1000,
                                                    search_param_radius=0.5)

reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source,target,source_feature,target_feature)

reg_icp = o3d.pipelines.registration.registration_icp(source,target,init=reg_ransac.transformation,draw=True,overwrite_color=True)


