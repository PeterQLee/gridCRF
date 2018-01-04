#ifndef SQUARECRF_H
#define SQUARECRF_H

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <types.h>
#include <math.h>
#include "immintrin.h"
#include "avx_mathfun.h"

#define COORD4(dir,x,y,f,p,X,Y,NF) (dir*X*Y*NF*2 + x*Y*NF*2 + y*NF*2 + f*2 + p)
#define COORD3(x,y,n, X,Y,NF,O ) (x*Y*NF*O + y*NF*O + n*O)
#define COORD2(x,y,X,Y,O) (x*Y*O +y*O)
#define FACT_TYPE NPY_FLOAT32

#define BIG 1234567.0

typedef struct{
  PyObject_HEAD
  i64 n_outcomes;
  i64 n_factors;
  i64 depth;
  PyArrayObject *V;
  f32 *V_data;
  i32 *com, *rom;
}gridCRF_t;


static void _train( gridCRF_t * self, PyArrayObject *X, PyArrayObject *Y);
static PyArrayObject* _loopyCPU(gridCRF_t* self, PyArrayObject *X);

static PyObject* fit (gridCRF_t * self, PyObject *args);
static PyObject* predict(gridCRF_t *self, PyObject *args);

static PyMethodDef  gridCRF_methods[]={
  {"fit",(PyCFunction)fit,METH_VARARGS,"Fit model"},
  {"predict",(PyCFunction)predict,METH_VARARGS,"Predict given a trained model"},
  {NULL}
};

static PyMemberDef gridCRF_members[]={
  {"V",T_OBJECT,offsetof(gridCRF_t,V),0,"Energy transfer matrix"},
  {NULL}
};

PyMODINIT_FUNC initmodel(void);

static void gridCRF_dealloc(gridCRF_t *self);
static int gridCRF_init(gridCRF_t *self, PyObject *args, PyObject *kwds);
static PyObject * gridCRF_new (PyTypeObject *type, PyObject *args, PyObject *kwds);



static PyTypeObject gridCRF_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "gridCRF.gridCRF",             /* tp_name */
  sizeof(gridCRF_t), /* tp_basicsize */
  0,                         /* tp_itemsize */
  gridCRF_dealloc,                         /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr */
  0,                         /* tp_setattr */
  0,                         /* tp_as_async */
  0,                         /* tp_repr */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash  */
  0,                         /* tp_call */
  0,                         /* tp_str */
  0,                         /* tp_getattro */
  0,                         /* tp_setattro */
  0,                         /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,        /* tp_flags */
  "grid CRF object",           /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  0,                         /* tp_iter */
  0,                         /* tp_iternext */
  gridCRF_methods,             /* tp_methods */
  gridCRF_members,             /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)gridCRF_init,      /* tp_init */
  0,                         /* tp_alloc */
  gridCRF_new,                 /* tp_new */
};

static PyModuleDef gridCRFmodule = {
  PyModuleDef_HEAD_INIT,
  "gridCRF",
  "TBH",
  -1,
  NULL, NULL, NULL, NULL, NULL
};

#endif
