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
    // int ind = 0;
    // for (int k=0; k < num_2d;k++){
    //   const float * query_boxes = boxes_2d_data + k*4;
    //   float qbox_area = ((query_boxes[2] - query_boxes[0]) *
    //                 (query_boxes[3] - query_boxes[1]));
    //   for(int n = 0; n < num_anchor;n++){
    //     const float * boxes = boxes_anchor_data + n*4;
    //     float iw = (std::min(boxes[2], query_boxes[2]) -
    //               std::max(boxes[0], query_boxes[0]));
    //     if(iw > 0){
    //       float ih = (std::min(boxes[3], query_boxes[3]) -
    //           std::max(boxes[1], query_boxes[1]));
    //       if(ih > 0){
    //         float ua = (
    //           (boxes[2] - boxes[0]) *
    //           (boxes[3] - boxes[1]) + qbox_area - iw * ih);
    //         float * overlaps = overlaps_data+(ind)*4;

    //         overlaps[0] = iw * ih / ua;
    //         overlaps[1] = scores_3d_data[n];
    //         overlaps[2] = scores_2d_data[k];
    //         overlaps[3] = dis_to_lidar_3d_data[n];
    //         int * tensor_index = tensor_idx_data +(ind)*2;
    //         tensor_index[0] = k;
    //         tensor_index[1] = n;
    //         ind = ind+1;
    //       }
    //       else if (k == num_2d-1){
    //         float * overlaps = overlaps_data+(ind)*4;
    //         overlaps[0] = -10;
    //         overlaps[1] = scores_3d_data[n];
    //         overlaps[2] = -10;
    //         overlaps[3] = dis_to_lidar_3d_data[n];
    //         int * tensor_index = tensor_idx_data +(ind)*2;
    //         tensor_index[0] = k;
    //         tensor_index[1] = n;
    //         ind = ind+1;
    //       }
    //     }
    //     else if(k==num_2d-1){
    //       float * overlaps = overlaps_data+(ind)*4;
    //       overlaps[0] = -10;
    //       overlaps[1] = scores_3d_data[n];
    //       overlaps[2] = -10;
    //       overlaps[3] = dis_to_lidar_3d_data[n];
    //       int * tensor_index = tensor_idx_data +(ind)*2;
    //       tensor_index[0] = k;
    //       tensor_index[1] = n;
    //       ind = ind+1;
    //     }
    //   }
    // }
    // *max_num_data = ind;

    clocscomputeiouLauncher(num_anchor, num_2d, boxes_anchor_data, boxes_2d_data, 
                            scores_3d_data, scores_2d_data, dis_to_lidar_3d_data,
                            overlaps_data, tensor_idx_data, max_num_data);

    return 1;
}






;