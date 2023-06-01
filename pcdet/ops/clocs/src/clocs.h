#ifndef CLOCS_H
#define CLOCS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int clocs_compute_iou_gpu(at::Tensor boxes3d_projected, 
                            at::Tensor boxes_2d, 
                            at::Tensor scores_3d, 
                            at::Tensor scores_2d,  
                            at::Tensor dis_to_lidar_3d,
                            at::Tensor overlap, 
                            at::Tensor tensor_index);
#endif