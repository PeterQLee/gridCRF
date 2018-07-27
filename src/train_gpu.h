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

#ifndef TRAIN_GPU_H
#define TRAIN_GPU_H
#include "gridtypes.h"
#include "loopy_gpu.h"
#include "train_cpu.h"

typedef struct {
  f32 *prod, *sum, *prob;
} gpu_dice_error_data_t;


typedef struct {
  gridCRF_t *self;
  f32 *dev_V_change, *dev_unary_change;
  f32 *dev_X;
  i32 *dev_Y;
  i32 *dev_refimg;
  i32 *dev_ainc, *dev_binc;
  npy_intp * dims;
  i32 sample_index;

  gpu_loopy_data *gdata;
  
  i32 num_params, n_factors;
  f32 alpha;

  gpu_loopy_params_t *lpar;
  f32 *dev_L;
  f32 host_L;
  error_func_e error_func;
  void * error_data;
  
  update_type_e update_type;
  void * update_data;
  
  f32 stop_tol;

  i32 *converged;
}gpu_gradient_t;

typedef struct {
  f32 gamma, alpha, **vstore_agg, **vstore, stop_tol, *v_curr, *v_old;
  i32 current_offset, *converged;
}gpu_rmsprop_t;


void GPU_grad_descent(gradient_t *args, i32 epochs, i32 dummy);
/* 
Convention:
any variable prepended with device is an area of memory reserved for device memory.
This does not include any structure in gpu_loopy_data, which are all stored in device.
*/


#endif
