#ifndef TRAIN_CPU_H
#define TRAIN_CPU_H
#include "gridtypes.h"
#include <pthread.h>
#include "common.h"
#include "loopy.h"

typedef struct {
  f32 *prod, *sum, *prob;
  pthread_mutex_t *sumlock, *prodlock;
  pthread_barrier_t *sync_sum;

}cpu_dice_data_t;

typedef struct {
  gridCRF_t *self;
  f32 *V_change, *unary_change;
  PyObject *X_list, *Y_list;
  i32 *ainc, *binc;
  npy_intp * dims;
  npy_intp * start;
  npy_intp * stop;
  i32 num_params, n_factors;
  f32 alpha;
  PyArrayObject *(*loopy_func) (gridCRF_t*, PyArrayObject*, loopy_params_t*,PyArrayObject*); 
  loopy_params_t *lpar;
  f32 L;
  i32 *instance_index;
  error_func_e error_func;
  void* error_data;
}gradient_t;

void grad_descent(gradient_t *args,i64 epochs,i64 n_threads);
static void* _calculate_gradient(gradient_t *args);

#endif
