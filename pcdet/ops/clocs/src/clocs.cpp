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

void clocscomputeiouLauncher(const int num_anchor, const int num_2d, const float * anchor_boxes, const float* boxes_2d, 
                                const float * scores_3d, const float * scores_2d, const float* dis_to_lidar_3d,
                                float * overlaps, int * tensor_idx, int * max_num);



int clocs_compute_iou_gpu(at::Tensor boxes_anchor, at::Tensor boxes_2d, 
                            at::Tensor scores_3d, at::Tensor scores_2d, at::Tensor dis_to_lidar_3d,
                            at::Tensor overlaps, at::Tensor tensor_idx, at::Tensor max_num){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, 1)

    CHECK_INPUT(boxes_anchor);
    CHECK_INPUT(boxes_2d);
    CHECK_INPUT(overlaps);
    CHECK_INPUT(tensor_idx);
    CHECK_INPUT(scores_3d);
    CHECK_INPUT(scores_2d);
    CHECK_INPUT(dis_to_lidar_3d);
    CHECK_INPUT(max_num);

    int num_anchor = boxes_anchor.size(0);
    int num_2d = boxes_2d.size(0);


    const float * boxes_anchor_data = boxes_anchor.data<float>();
    const float * boxes_2d_data = boxes_2d.data<float>();
    float * overlaps_data = overlaps.data<float>();
    int * tensor_idx_data = tensor_idx.data<int>();
    float * scores_3d_data = scores_3d.data<float>();
    float * scores_2d_data = scores_2d.data<float>();
    float * dis_to_lidar_3d_data = dis_to_lidar_3d.data<float>();
    int * max_num_data = max_num.data<int>();

    clocscomputeiouLauncher(num_anchor, num_2d, boxes_anchor_data, boxes_2d_data, 
                            scores_3d_data, scores_2d_data, dis_to_lidar_3d_data,
                            overlaps_data, tensor_idx_data, max_num_data);

    return 1;
}






