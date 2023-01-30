/*
batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}


void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<blocks, threads>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
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
 * @param idx_cnt //* [batch_size, 需要采样的点个数]的全0张量, 用来存储每个采样点周围半径内的点数
 * @param idx //*[batch_size, 需要采样的点个数, 每个group中的点的索引], 用来存储每个采样点周围半径内用来group的点的索引
 */
__global__ void ball_query_dilated_kernel_fast(int b, int n, int m, float radius_in, float radius_out, int nsample,
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx_cnt, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx_cnt: (B, M)
    //      idx: (B, M, nsample)
    //* bs_idx是当时处理的batch的索引
    int bs_idx = blockIdx.y;
    //* pt_idx是当前处理的点的索引
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;
    //* 当前处理的这个采样点的起始位置
    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    //* 当前处理的这个sample的点的起始位置
    xyz += bs_idx * n * 3;
    //* 记录当前处理的这个采样点的group内索引的起始位置
    idx += bs_idx * m * nsample + pt_idx * nsample;
    //* 记录当前处理的这个采样点的group内的点数的起始位置
    idx_cnt += bs_idx * m + pt_idx;
    //* 内圈半径的平方
    float radius_in2 =radius_in * radius_in;
    //* 外圈半径的平方
    float radius_out2 = radius_out * radius_out;
    //* 当前处理的采样点的xyz坐标
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    //* 遍历这个sample的所有点
    for (int k = 0; k < n; ++k) {
        //* 当前遍历到的点的xyz坐标
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        //* 计算这个点和当前处理的采样点的距离
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        //* 如果在group范围内，并且还没满，就记录其索引，并将计数器+1
        if (d2 >= radius_in2 && d2 < radius_out2){
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    idx_cnt[0] = cnt;
    //* repeat,直到获得nsample个点
    for (int l = 0; cnt < nsample; ++l, ++cnt) {
        idx[cnt] = idx[l];
    }
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
 * @param idx_cnt //* [batch_size, 需要采样的点个数]的全0张量, 用来存储每个采样点周围半径内的点数
 * @param idx //*[batch_size, 需要采样的点个数, 每个group中的点的索引], 用来存储每个采样点周围半径内用来group的点的索引
 */
void ball_query_dilated_kernel_launcher_fast(int b, int n, int m, float radius_in, float radius_out, int nsample, \
    const float *new_xyz, const float *xyz, int *idx_cnt, int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx_cnt: (B, M)
    //      idx: (B, M, nsample)

    cudaError_t err;
    //* block为[需要采样的点数/THREADS_PER_BLOCK的上界, batch_size]
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    ball_query_dilated_kernel_fast<<<blocks, threads>>>(b, n, m, radius_in, radius_out, nsample, new_xyz, xyz, idx_cnt, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
