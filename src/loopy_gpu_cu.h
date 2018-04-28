#ifndef LOOPY_GPU_CU_H
#define LOOPY_GPU_CU_H
#include "loopy.h"

void gpu_loopy_F_V(loopycpu_t *targs);
__global__ void gpu_loopy_F_V__Flow(f32 *F_V, f32 *V_F, f32 *RE, const i32 * refimg,  const i32 * com, const om_pair * co_pairs, i32 n_factors);
__global__ void gpu_loopy_F_V__Fup(f32 *F_V, f32 *V_F,  f32 *CE, const i32 * refimg, const i32 *rom, const om_pair *co_pairs, i32 n_factors);

#endif
