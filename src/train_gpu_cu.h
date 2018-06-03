
/*
Copyright 2018 Peter Q. Lee

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


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

void g_RMSprop_swap_vstore(gpu_rmsprop_t *rmsp, i32 index);
  

__global__ void gpu_entropy_partial(f32 *mu, i32 *EY, f32 *X, i32 *Y, f32 *V, f32 *V_change, f32* unary_change, i32 *ainc, i32 *binc, f32 alpha, i32 limx, i32 limy, i32 n_factors);



__global__ void gpu_dice_prob(f32 * unary_c, i32 *EY, f32 *V, i32 *Y, f32 *p, i32 *ainc, i32 *binc, i32 limx, i32 limy, i32 n_factors);

__global__ void gpu_dice_intermediate_summation(f32 *p, i32 *Y, f32 *prod, f32 *sum, i32 limx, i32 limy);

__global__ void gpu_dice_partial(f32 *p, i32 *EY, f32 *prod, f32 *sum, f32 *X, i32 *Y, f32 *V_change, f32 *unary_change, i32 *ainc, i32 *binc, f32 alpha, i32 limx, i32 limy, i32 n_factors);




__global__ void gpu_update_params(f32 *V, f32* V_change, f32 lr, i32 *converged, f32 stop_tol);
__global__ void gpu_RMSprop_update(f32 *v_curr, f32 *v_old, i32 *converged, f32 gamma, f32 alpha, f32 *V, f32 *V_change, i32 n_factors, i32 n_unary, f32 stop_tol);

#endif
