import os
import open3d as o3d
import numpy as np
import time
from matplotlib import pyplot as plt


#Visualization
def draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(),
         target_temp.to_legacy()],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])

voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])

#Input

source_path = os.listdir('result/')
target_path = os.listdir('project/')

max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])

criteria_list = [
    o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                relative_rmse=0.0001,
                                max_iteration=20),
    o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
    o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 10)]


estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

save_loss_log = True

s = time.time()                                      

#Initial
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                        [-0.139, 0.967, -0.215, 0.7],
                        [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

init_source_to_target = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
#init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

for i in range (len(source_path)):

    source = o3d.t.io.read_point_cloud('result/' + source_path[i])
    target = o3d.t.io.read_point_cloud('project/' + target_path[i])

    source.point["colors"] = source.point["colors"].to(
        o3d.core.Dtype.Float32) / 255.0
    target.point["colors"] = target.point["colors"].to(
        o3d.core.Dtype.Float32) / 255.0


    draw_registration_result(source, target, trans_init)


    #Print
    callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
        updated_result_dict["iteration_index"].item(),
        updated_result_dict["fitness"].item(),
        updated_result_dict["inlier_rmse"].item()))

    #Registration
    registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(source, target, voxel_sizes,
                                            criteria_list,
                                            max_correspondence_distances,
                                            init_source_to_target, estimation, callback_after_iteration)

    icp_time = time.time() - s
    print("Time taken by ICP: ", icp_time)
    print("Inlier Fitness: ", registration_ms_icp.fitness)
    print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)
    if registration_ms_icp.fitness == 0 and registration_ms_icp.inlier_rmse == 0:
        print("ICP Convergence Failed, as no correspondence were found")

    result = draw_registration_result(source, target, registration_ms_icp.transformation)

    result = o3d.t.io.write_point_cloud('project/' + [i+1], result, write_ascii=True, compressed=False)
    #source_path.append(result())
    print(result)
    #source.close()
    #target.close()

else:
    print("No items left.")


'''
#Plot
def plot_rmse(registration_result):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
    axes.set_title("Inlier RMSE vs Iteration")
    axes.plot(registration_result.loss_log["index"].numpy(),
              registration_result.loss_log["inlier_rmse"].numpy())


def plot_scale_wise_rmse(registration_result):
    scales = registration_result.loss_log["scale"].numpy()
    iterations = registration_result.loss_log["iteration"].numpy()

    num_scales = scales[-1][0] + 1

    fig, axes = plt.subplots(nrows=1, ncols=num_scales, figsize=(20, 5))

    masks = {}
    for scale in range(0, num_scales):
        masks[scale] = registration_result.loss_log["scale"] == scale

        rmse = registration_result.loss_log["inlier_rmse"][masks[scale]].numpy()
        iteration = registration_result.loss_log["iteration"][
            masks[scale]].numpy()

        title_prefix = "Scale Index: " + str(scale)
        axes[scale].set_title(title_prefix + " Inlier RMSE vs Iteration")
        axes[scale].plot(iteration, rmse)

plot_rmse(registration_ms_icp)
plot_scale_wise_rmse(registration_ms_icp)'''
