#ifndef TRAIN_GPU_CU_H
#define TRAIN_GPU_CU_H
#include "train_gpu.h"
#include "loopy_gpu_cu.h"

/* 
Convention:
any variable prepended with device is an area of memory reserved for device memory.
This does not include any structure in gpu_loopy_data, which are all stored in device.
*/



static void gpu_calculate_gradient(gpu_gradient_t *args);

__global__ void gpu_entropy_partial(f32 *mu, i32 *EY, f32 *X, i32 *Y, f32 *V, f32 *V_change, f32* unary_change, i32 *ainc, i32 *binc, f32 alpha, i32 limx, i32 limy, i32 n_factors);
__global__ void gpu_update_params(f32 *V, f32* V_change, f32 lr);

#endif
