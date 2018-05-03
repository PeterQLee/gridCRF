#ifndef gridtypes_h
#define gridtypes_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "types.h"
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define COORD4(dir,x,y,f,p,X,Y,NF) ((dir)*(X)*(Y)*(NF)*(2) + (x)*(Y)*(NF)*(2) + (y)*(NF)*(2) + (f)*(2) + (p))
#define COORD3(x,y,n, X,Y,NF,O ) ((x)*(Y)*(NF)*(O) + (y)*(NF)*(O) + (n)*(O))
#define COORD2(x,y,X,Y,O) ((x)*(Y)*(O) +(y)*(O))



#define FACT_TYPE NPY_FLOAT32

#define BIG 1234567.0

typedef struct{
  i32 x,y;
}om_pair;


typedef struct{
  PyObject_HEAD
  i64 n_outcomes;
  i64 n_factors;
  i64 depth;
  PyArrayObject *V, *unary_pyarr;
  f32 *V_data;
  f32 *unary;
  i32 gpuflag;
  //i32 *com, *rom;
}gridCRF_t;

typedef struct{
  i32 max_its;
  f32 stop_thresh;
  f32 *mu;
  i32 eval;
  i32 *EY;
  i32 n_threads;
}loopy_params_t;



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
}gradient_t;


typedef struct{
  i32 epochs;
  f32 alpha;
  i32 gpu;
}train_params_t;

#endif

