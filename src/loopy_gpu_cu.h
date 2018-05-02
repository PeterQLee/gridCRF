#ifndef LOOPY_GPU_CU_H
#define LOOPY_GPU_CU_H
#include "loopy.h"

typedef struct{
  /* coordinates */
  i32 *com, *rom;
  om_pair *co_pairs;

  /* parameter data */
  loopy_params_t * lpar;
  gridCRF_t *self;
  PyArrayObject *X_py;
  f32 *X, *refimg;
  f32 *F_V, *V_F;
  f32 *RE, *CE;
  f32 *mu;
  f32 *unary_w, *unary_c;
  /* flags */
  i32 *converged;
  i32 *_converged;

  

}loopygpu_t;

typedef struct{
  i32 max_its;
  f32 stop_thresh;
  f32 *mu;
  i32 eval;
  i32 *EY;
  i32 n_threads;
  f32 *V_data, *V_F, *F_V, *mu, *RE, *CE;
  i32 *com, *rom;
  om_pair *co_pairs;
    
}gpu_loopy_params_t;


void gpu_loopy_F_V(loopygpu_t *targs);
void gpu_loopy_V_F(loopygpu_t *targs);

__global__ void gpu_loopy_F_V__Flow(f32 *F_V, f32 *V_F, f32 *RE, const i32 * refimg,  const i32 * com, const om_pair * co_pairs, i32 n_factors);
__global__ void gpu_loopy_F_V__Fup(f32 *F_V, f32 *V_F,  f32 *CE, const i32 * refimg, const i32 *rom, const om_pair *co_pairs, i32 n_factors);

__global__ void gpu_loopy_V_F__computeunary(f32 * X, f32* unary_w, f32 *unary_c);
__global__ void gpu_loopy_V_F__sumfactors(f32 *F_V, f32 *V_F, f32 *unary_c, const i32 * refimg, i32 n_factors);

__global__ void gpu_loopy_V_F__marginal(f32 *F_V, f32 * unary_c,  f32 * mu, i32 n_factors, f32 stop_thresh, i32 *converged);


__global__ void gpu_loopy_V_F__label(f32 *F_V, f32 * unary_c, i32 *EY, i32 n_factors);

__global__ void gpu_fill_value(f32 *buffer, f32 val);
__global__ void gpu_multiply(f32 *buffer, f32 val);
#endif

