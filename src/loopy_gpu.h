#ifndef LOOPY_GPU_H
#define LOOPY_GPU_H
#include "loopy.h"


typedef struct{
  f32  *V_F, *F_V, *mu, *RE, *CE, *unary_c;
  i32 *com, *rom;
  om_pair *co_pairs;
  i32 *EY;
  
  f32 *V_data, *unary_w;

  f32 *X;
}gpu_loopy_data;


typedef struct{
  i32 max_its;
  f32 stop_thresh;
  i32 eval;
  gpu_loopy_data *gdata;
}gpu_loopy_params_t;


typedef struct{
  /* coordinates */

  /* parameter data */
  gpu_loopy_params_t * lpar;
  gridCRF_t *self;
  npy_intp *dims;


  /* flags */
  i32 *dev_converged;
  i32 *host_converged;

  /* gpu data */
  gpu_loopy_data *gdata;
  
}loopygpu_t;

i32 *predict_loopyGPU(gridCRF_t* self, PyArrayObject *X, loopy_params_t *lpar,PyArrayObject *refimg);
i32 *loopyGPU(gridCRF_t* self, PyArrayObject *X, gpu_loopy_params_t *lpar,PyArrayObject *refimg);


#endif
