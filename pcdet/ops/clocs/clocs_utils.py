from . import clocs_cuda

def compute_clocs_iou(boxes_3d_projected, 
                      boxes_2d_projected, 
                      scores_3d, 
                      scores_2d, 
                      dis_to_lidar_3d, 
                      overlaps, 
                      tensor_index):
    clocs_cuda.clocs_comput_iou(boxes_3d_projected, 
                                boxes_2d_projected, 
                                scores_3d, 
                                scores_2d, 
                                dis_to_lidar_3d, 
                                overlaps, 
                                tensor_index)
    return overlaps, tensor_index