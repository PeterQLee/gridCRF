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

#ifndef _cplusplus
i32* loopyCPU(gridCRF_t* self, PyArrayObject *X,loopy_params_t *lpar,PyArrayObject *refimg);

static void * _loopyCPU__FtoV(loopycpu_t *l_args);
static void * _loopyCPU__VtoF(loopycpu_t *l_args);
static void * _loopy_label(loopycpu_t *l_args);

static void _compute_unary(f32 *tmp, f32 *base, f32 *unary, i32 n_chan);
#endif

#endif
