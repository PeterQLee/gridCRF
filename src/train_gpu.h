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
  i32 *dev_ainc, *dev_binc;
  npy_intp * dims;

  gpu_loopy_data *gdata;
  
  i32 num_params, n_factors;
  f32 alpha;
  PyArrayObject *(*loopy_func) (gridCRF_t*, PyArrayObject*, loopy_params_t*,PyArrayObject*); 
  gpu_loopy_params_t *lpar;
  f32 *dev_L;
  f32 host_L;
  error_func_e error_func;
  void * error_data; 

}gpu_gradient_t;

void GPU_grad_descent(gradient_t *args, i32 epochs, i32 dummy);
/* 
Convention:
any variable prepended with device is an area of memory reserved for device memory.
This does not include any structure in gpu_loopy_data, which are all stored in device.
*/


#endif
