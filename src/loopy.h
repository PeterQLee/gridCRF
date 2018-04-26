#ifndef LOOPY_H
#define LOOPY_H
#include "gridtypes.h"
#include "immintrin.h"

typedef struct{
  i32 *start, *stop; //threading

  /* coordinates */
  i32 *com, *rom;
  om_pair *co_pairs;

  /* parameter data */
  loopy_params_t * lpar;
  gridCRF_t *self;
  PyArrayObject *X, *refimg;
  f32 *F_V, *V_F;
  f32 *RE, *CE;
  f32 *marginals, *mu;
  
  /* flags */
  i32 *converged;
  
} loopycpu_t;

i32* loopyCPU(gridCRF_t* self, PyArrayObject *X,loopy_params_t *lpar,PyArrayObject *refimg);

static void * _loopyCPU__FtoV(loopycpu_t *l_args);
static void * _loopyCPU__VtoF(loopycpu_t *l_args);
#endif
