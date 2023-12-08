from . import clocs_cuda

def compute_clocs_iou_sparse(boxes_3d_projected, 
                            boxes_2d_projected, 
                            scores_3d, 
                            scores_2d, 
                            dis_to_lidar_3d, 
                            overlaps):
    clocs_cuda.clocs_comput_iou_sparse(boxes_3d_projected, 
                                    boxes_2d_projected, 
                                    scores_3d, 
                                    scores_2d, 
                                    dis_to_lidar_3d, 
                                    overlaps)
    return overlaps

def compute_clocs_iou_dense(boxes_3d_projected, 
                            boxes_2d_projected, 
                            scores_3d, 
                            scores_2d, 
                            dis_to_lidar_3d, 
                            max_num,
                            overlaps,
                            tensor_idx,
                            count):
    clocs_cuda.clocs_comput_iou_dense(boxes_3d_projected, 
                                    boxes_2d_projected, 
                                    scores_3d, 
                                    scores_2d, 
                                    dis_to_lidar_3d, 
                                    max_num,
                                    overlaps,
                                    tensor_idx, 
                                    count)

def cos_similarity(lidar_features, 
                    camera_features,
                    cos_sim 
):
    clocs_cuda.cos_similarity_gpu(lidar_features,
                                  camera_features,
                                  cos_sim)