#ifndef TRAIN_CPU_H
#define TRAIN_CPU_H
#include "gridtypes.h"
#include "loopy.h"

static void grad_descent(gradient_t *args,i64 epochs,i64 n_threads);
static void* _calculate_gradient(gradient_t *args);

#endif
