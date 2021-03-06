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
  i32 n_outcomes;
  i32 n_factors, n_unary, n_inp_channels;
  i32 depth;
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
  i32 reset_flag;
}loopy_params_t;





typedef struct{
  i32 epochs;
  f32 alpha;
  f32 gamma;
  f32 stop_tol;
  i32 error_func; //0 = entropy, 1 = dice
  i32 update_type;
  i32 reset_flag;
}
  train_params_t;

typedef  enum {ENTROPY,
	       DICE
}error_func_e;

typedef enum {SGD,
	      RMSPROP
}update_type_e;

#endif

