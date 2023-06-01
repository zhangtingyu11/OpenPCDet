/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "clocs.h"
#include <stdio.h>

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void clocscomputeiouLauncher(const int num_3d, 
                              const int num_2d, 
                              const float * boxes3d, 
                              const float* boxes_2d, 
                              const float * scores_3d, 
                              const float * scores_2d, 
                              const float* dis_to_lidar_3d,
                              float * overlap, 
                              int * tensor_index);

int clocs_compute_iou_gpu(at::Tensor boxes3d_projected, 
                            at::Tensor boxes_2d, 
                            at::Tensor scores_3d, 
                            at::Tensor scores_2d, 
                            at::Tensor dis_to_lidar_3d,
                            at::Tensor overlap, 
                            at::Tensor tensor_index){

    CHECK_INPUT(boxes3d_projected);
    CHECK_INPUT(boxes_2d);
    CHECK_INPUT(scores_3d);
    CHECK_INPUT(scores_2d);
    CHECK_INPUT(dis_to_lidar_3d);
    CHECK_INPUT(overlap);
    CHECK_INPUT(tensor_index);

    int num_3d = boxes3d_projected.size(0);
    int num_2d = boxes_2d.size(0);


    const float * boxes3d_data = boxes3d_projected.data<float>();
    const float * boxes2d_data = boxes_2d.data<float>();
    float * scores_3d_data = scores_3d.data<float>();
    float * scores_2d_data = scores_2d.data<float>();
    float * dis_to_lidar_3d_data = dis_to_lidar_3d.data<float>();
    float * overlap_data = overlap.data<float>();
    int * tensor_index_data = tensor_index.data<int>();

    clocscomputeiouLauncher(num_3d, 
                            num_2d, 
                            boxes3d_data, 
                            boxes2d_data, 
                            scores_3d_data, 
                            scores_2d_data, 
                            dis_to_lidar_3d_data,
                            overlap_data, 
                            tensor_index_data);
    return 1;
}






;