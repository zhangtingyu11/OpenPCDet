/*
batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query_gpu.h"

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


int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    
    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx);
    return 1;
}

/**
 * @brief //* dilated ball query
 * 
 * @param b //* batch_size
 * @param n //* 采样范围内的点数
 * @param m //* 需要采样的点数
 * @param radius_in //* 采样的内圈半径
 * @param radius_out //* 采样的外圈半径
 * @param nsample //* 每个group内的最大点数
 * @param new_xyz_tensor  //* 采样的点的坐标
 * @param xyz_tensor  //* 采样范围内的点的坐标
 * @param idx_cnt_tensor //* [batch_size, 需要采样的点个数]的全0张量, 用来存储每个采样点周围半径内的点数
 * @param idx_tensor //*[batch_size, 需要采样的点个数, 每个group中的点的索引], 用来存储每个采样点周围半径内用来group的点的索引
 * @return int 
 */
int ball_query_dilated_wrapper_fast(int b, int n, int m, float radius_in, float radius_out, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_cnt_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx_cnt = idx_cnt_tensor.data<int>();
    int *idx = idx_tensor.data<int>();

    ball_query_dilated_kernel_launcher_fast(b, n, m, radius_in, radius_out, nsample, new_xyz, xyz, idx_cnt, idx);
    return 1;
}
