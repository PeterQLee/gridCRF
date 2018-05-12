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
