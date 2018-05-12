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

#ifndef LOOPY_GPU_CU_H
#define LOOPY_GPU_CU_H
#include "loopy_gpu.h"
#include "common.h"

i32 *loopyGPU(gridCRF_t* self, PyArrayObject *X_py, gpu_loopy_params_t *lpar, PyArrayObject *refimg);
void gpu_loopy_F_V(loopygpu_t *targs);
void gpu_loopy_V_F(loopygpu_t *targs);

__global__ void gpu_loopy_F_V__Flow(f32 *F_V, f32 *V_F, f32 *RE, const i32 * refimg,  const i32 * com, const om_pair * co_pairs, i32 n_factors);
__global__ void gpu_loopy_F_V__Fup(f32 *F_V, f32 *V_F,  f32 *CE, const i32 * refimg, const i32 *rom, const om_pair *co_pairs, i32 n_factors);

__global__ void gpu_loopy_V_F__computeunary(f32 * X, f32* unary_w, f32 *unary_c);
__global__ void gpu_loopy_V_F__sumfactors(f32 *F_V, f32 *V_F, f32 *unary_c, const i32 * refimg, i32 n_factors);

__global__ void gpu_loopy_V_F__marginal(f32 *F_V, f32 * unary_c,  f32 * mu, i32 n_factors, f32 stop_thresh, i32 *converged);


__global__ void gpu_loopy_V_F__label(f32 *mu, i32 *EY, i32 n_factors);

__global__ void gpu_fill_value(f32 *buffer, f32 val, i32 lim);
__global__ void gpu_multiply(f32 *buffer, i32 lim);

__global__ void gpu_swizzle(f32 *dest, f32 *src, i32 n_factors, i32 lim);
#endif

