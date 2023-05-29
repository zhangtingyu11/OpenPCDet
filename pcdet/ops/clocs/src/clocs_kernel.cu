/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void clocs_compute_iou_kernel(const int num_anchor, const float * anchor_boxes, const int num_2d, const float *boxes_2d, 
                                        const float * scores_3d, const float * scores_2d, const float * dis_to_lidar_3d,
                                        float *overlaps, int* tensor_idx, int * max_num){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    const int b_idx= blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    const int a_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    // if(a_idx==70399){
    //     printf("a_idx:%d\n", a_idx );
    // }
    if (a_idx >= num_anchor || b_idx >= num_2d){
        return;
    }

    const float * query_box = anchor_boxes + a_idx * 4;
    const float * box = boxes_2d + b_idx * 4;
    float qbox_area = (*(query_box+2) - *(query_box+0)) *
                        (*(query_box+3) - *(query_box+1));
    // printf("area is:%f\n", qbox_area);
    float iw = min(*(box+2), *(query_box+2)) - 
                max(*(box+0), *(query_box+0));
    // printf("query_box:%f, %f %f, %f\n", *(query_box+0), *(query_box+1), *(query_box+2), *(query_box+3));
    // printf("box:%f, %f %f, %f\n", *(box+0), *(box+1), *(box+2), *(box+3));

    // printf("iw:%f", iw);
    if(iw > 0){
        float ih = min(*(box+3), *(query_box+3)) - 
                    max(*(box+1), *(query_box+1));
        if(ih > 0){
            int idx = atomicAdd(max_num, 1);
            float ua = ((*(box+2) - *(box+0)) * 
                        (*(box+3) - *(box+1)) + qbox_area - iw * ih);
            // printf("idx:%d, max_num:%d\n", idx, *max_num);

            float * cur_overlap = overlaps+(idx)*4;
            cur_overlap[0] = iw * ih /ua;
            // printf("overlaps0%f\n", cur_overlap[0]);

            cur_overlap[1] = scores_3d[a_idx];
            cur_overlap[2] = scores_2d[b_idx];
            cur_overlap[3] = dis_to_lidar_3d[a_idx];
            // printf("max_num:%d\n", idx);
            int * cur_tensor_idx = tensor_idx+(idx)*2;
            cur_tensor_idx[0] = b_idx;
            cur_tensor_idx[1] = a_idx;
            // printf("max_num:%d\n", idx);
        }
        else if(b_idx == num_2d-1){
            int idx = atomicAdd(max_num, 1);
            float * cur_overlap = overlaps+(idx)*4;
            cur_overlap[0] = -10;
            cur_overlap[1] = scores_3d[a_idx];
            cur_overlap[2] = -10;
            cur_overlap[3] = dis_to_lidar_3d[a_idx];
            int * cur_tensor_idx = tensor_idx+(idx)*2;
            cur_tensor_idx[0] = b_idx;
            cur_tensor_idx[1] = a_idx;
        }
    }
    else if(b_idx == num_2d-1){
        int idx = atomicAdd(max_num, 1);
        float * cur_overlap = overlaps+(idx)*4;
        cur_overlap[0] = -10;
        cur_overlap[1] = scores_3d[a_idx];
        cur_overlap[2] = -10;
        cur_overlap[3] = dis_to_lidar_3d[a_idx];
        int * cur_tensor_idx = tensor_idx+(idx)*2;
        cur_tensor_idx[0] = b_idx;
        cur_tensor_idx[1] = a_idx;
        // printf("in_last:%d\n", idx);
    }
}

void clocscomputeiouLauncher(const int num_anchor, const int num_2d, 
                                const float * anchor_boxes, const float* boxes_2d, 
                                const float * scores_3d, const float * scores_2d, const float * dis_to_lidar_3d,
                                float * overlaps, int * tensor_idx, int * max_num){
    dim3 blocks(DIVUP(num_anchor, THREADS_PER_BLOCK), DIVUP(num_2d, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    clocs_compute_iou_kernel<<<blocks, threads>>>(num_anchor, anchor_boxes, num_2d, boxes_2d, scores_3d, scores_2d, dis_to_lidar_3d, 
                                                    overlaps, tensor_idx, max_num);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}
