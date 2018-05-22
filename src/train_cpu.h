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

#ifndef TRAIN_CPU_H
#define TRAIN_CPU_H
#include "gridtypes.h"
#include <pthread.h>
#include "immintrin.h"
#include "common.h"
#include "loopy.h"

typedef struct {
  f32 *prod, *sum, *prob;
  pthread_mutex_t *sumlock, *prodlock;
  //pthread_barrier_t *sync_sum;
  void *sync_sum;

}cpu_dice_data_t;

typedef struct {
  gridCRF_t *self;
  f32 *V_change, *unary_change;
  PyObject *X_list, *Y_list;
  i32 *ainc, *binc;
  npy_intp * dims;
  npy_intp * start;
  npy_intp * stop;
  i32 num_params, n_factors, n_unary;
  f32 alpha;
  f32 gamma;
  f32 scale;
  PyArrayObject *(*loopy_func) (gridCRF_t*, PyArrayObject*, loopy_params_t*,PyArrayObject*); 
  loopy_params_t *lpar;
  f32 L;
  i32 *instance_index;
  error_func_e error_func;
  void* error_data;
  update_type_e update_type;
  f32 stop_tol; // min change threshold to continue training
}gradient_t;

typedef struct {
  f32 gamma, alpha, **vstore_agg, **vstore, stop_tol;
  i32 current_offset, *converged;
}rmsprop_t;


static void RMSprop_update(rmsprop_t *rmsp, f32 *V, f32 *V_change, i32 n_factors, i32 n_unary);
void grad_descent(gradient_t *args,i64 epochs,i64 n_threads);
static void* _calculate_gradient(gradient_t *args);

#endif
