from . import clocs_cuda

def compute_clocs_iou(boxes, query_boxes, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index, max_num):
    clocs_cuda.clocs_comput_iou(boxes, query_boxes, scores_3d, scores_2d, dis_to_lidar_3d, overlaps, tensor_index, max_num)
    return overlaps, tensor_index,max_num