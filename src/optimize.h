#include "types.h"
#include <stdlib.h>
#include "immintrin.h"
#ifndef OPTIMIZE_H
#define OPTIMIZE_H
typedef struct {
  f32 *g, *p, *y, *s, *l_change, *ll_change;
  i32 start, num_params, m, cur;
} lbfgs_t;

lbfgs_t* alloc_lbfgs(i32 m, i32 num_params);
void update_lbfgs();
f32* LBFGS(lbfgs_t *params);
#endif
