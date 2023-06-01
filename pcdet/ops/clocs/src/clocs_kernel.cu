/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void clocs_compute_iou_kernel(const int num_3d, 
                                        const float * boxes_3d, 
                                        const int num_2d, 
                                        const float * boxes_2d, 
                                        const float * scores_3d, 
                                        const float * scores_2d, 
                                        const float * dis_to_lidar_3d,
                                        float * overlap, 
                                        int * tensor_index){
    const int boxes3d_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const int boxes2d_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    if (boxes3d_idx >= num_3d || boxes2d_idx >= num_2d){
        return;
    }

    const float * box_3d = boxes_3d + boxes3d_idx * 4;
    const float * box_2d = boxes_2d + boxes2d_idx * 4;
    float * cur_overlap = overlap + (boxes3d_idx * num_2d + boxes2d_idx) * 4;
    int * cur_tensor_index = tensor_index + (boxes3d_idx * num_2d + boxes2d_idx) * 2;
    
    float qbox_area = (*(box_2d+2) - *(box_2d+0)) *
                        (*(box_2d+3) - *(box_2d+1));
    float iw = min(*(box_3d+2), *(box_2d+2)) - 
                max(*(box_3d+0), *(box_2d+0));

    if(iw > 0){
        float ih = min(*(box_3d+3), *(box_2d+3)) - 
                    max(*(box_3d+1), *(box_2d+1));
        if(ih > 0){
            float ua = ((*(box_3d+2) - *(box_3d+0)) * 
                        (*(box_3d+3) - *(box_3d+1)) + qbox_area - iw * ih);
            cur_overlap[0] = iw * ih /ua;
            cur_overlap[1] = scores_3d[boxes3d_idx];
            cur_overlap[2] = scores_2d[boxes2d_idx];
            cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
            cur_tensor_index[0] = boxes2d_idx;
            cur_tensor_index[1] = boxes3d_idx;
        }
        else if(boxes2d_idx == num_2d-1){
            cur_overlap[0] = -10;
            cur_overlap[1] = scores_3d[boxes3d_idx];
            cur_overlap[2] = -10;
            cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
            cur_tensor_index[0] = boxes2d_idx;
            cur_tensor_index[1] = boxes3d_idx;
        }
    }
    else if(boxes2d_idx == num_2d-1){
        cur_overlap[0] = -10;
        cur_overlap[1] = scores_3d[boxes3d_idx];
        cur_overlap[2] = -10;
        cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
        cur_tensor_index[0] = boxes2d_idx;
        cur_tensor_index[1] = boxes3d_idx;
    }
}

void clocscomputeiouLauncher(const int num_3d, 
                                const int num_2d, 
                                const float * boxes_3d, 
                                const float * boxes_2d, 
                                const float * scores_3d, 
                                const float * scores_2d, 
                                const float * dis_to_lidar_3d,
                                float * overlap, 
                                int * tensor_index){
    dim3 blocks(DIVUP(num_3d, THREADS_PER_BLOCK), DIVUP(num_2d, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    clocs_compute_iou_kernel<<<blocks, threads>>>(num_3d, 
                                                    boxes_3d, 
                                                    num_2d, 
                                                    boxes_2d, 
                                                    scores_3d, 
                                                    scores_2d, 
                                                    dis_to_lidar_3d, 
                                                    overlap, 
                                                    tensor_index);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}
