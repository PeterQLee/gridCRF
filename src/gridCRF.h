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

#ifndef GRIDCRF_H
#define GRIDCRF_H
#define LBFGS_FLOAT 32
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <types.h>
#include <math.h>
#include "immintrin.h"
#include "optimize.h"
#include "gridtypes.h"
#include "loopy.h"
#include "loopy_gpu.h"
#include "train_cpu.h"
#include "train_gpu.h"

static const char mainobj_docstr[] = \
  "Use this class to initialize a gridCRF model\n\n"\
  "Initialization parameters:\n"\
  "n: int\n"\
  "    The width of the grid for the CRF\n\n"\
  "gpuflag=0: int\n"\
  "    Flag for enabling the gpu:\n"\
  "    0 = CPU mode\n"\
  "    1 = GPU mode\n\n"\
  "Example:\n"\
  ">>> import gridCRF\n"\
  ">>> s = gridCRF.gridCRF(2, gpuflag = 1) #initialize a grid with width 2 and gpu\n"\
  ">>> s.fit(datax, datay, epochs=30, alpha=1000, error_type = 1)\n"\
  ">>> predicted_labelse = s.predict(testx)\n\n"\
  " You can also set your custom parameters by modifying gridCRF.V and gridCRF.unary directly\n";

static const char fit_docstr[] =\
  "Fits the model\n"\
  "Parameters:\n"\
  "- X: list\n"\
  "    The list of input training data images. Each element in this list must be a 3D np array with\n"\
  "    dimensions width, height, channel, and must be 32 bit floats. Unary probabilities should be\n"\
  "    log transformed, with all nans and infs converted to 0.0\n\n"\
  "- Y: list\n"\
  "    The list of input training label images. Each element in this list must be a 3D np array with\n"\
  "    dimensions width, height, channel, and must be 32 bit ints.\n\n"
  "- epochs = 100: int\n"\
  "    The number of epochs used to train the model with\n\n"\
  "- alpha = 0.001: float\n"\
  "    The learning rate used train the model. Until a more advanced training algorithm is used, the \n"\
  "    optimal alpha for use will vary depending on the error function. Experimentally, cross entropy\n"\
  "    works well with 0.1, and dice error works well between 100.0 and 1000.0\n\n"\
  "- error_type = 0: int\n"\
  "    The error type used for training. Options currently include:\n"\
  "    0 = cross entropy\n"\
  "    1 = dice error\n\n";

static const char predict_docstr[] =	\
  "Predicts given the trained model\n"
  "Parameters:\n"\
  "- X: list\n"\
  "    The list of input training data images. Each element in this list must be a 3D np array with\n"\
  "    dimensions width, height, channel, and must be 32 bit floats. Unary probabilities should be\n"\
  "    log transformed, with all nans and infs converted to 0.0\n\n"\
  "- stop_thresh = 0.01: float\n"\
  "    The convergence criteria for the crf inference. higher numbers will end quicker, but may be less precise\n\n"\
  "- max_its = 100: int\n"\
  "    Max number of iterations to undergo before stopping\n\n"\
  "- n_threads = 8: int\n"\
  " For cpu inference, not used for gpu inference.\n\n";
  
  

static PyObject* fit (gridCRF_t * self, PyObject *args, PyObject *kws);
static PyObject* predict(gridCRF_t *self, PyObject *args, PyObject *kws);

static PyMethodDef  gridCRF_methods[]={
  {"fit",(PyCFunction)fit,METH_VARARGS|METH_KEYWORDS, fit_docstr},
  {"predict",(PyCFunction)predict,METH_VARARGS|METH_KEYWORDS,predict_docstr},
  {NULL,NULL,0,NULL}
};

static PyMemberDef gridCRF_members[]={
  {"V",T_OBJECT,offsetof(gridCRF_t,V),0,"Energy transfer matrix"},
  {"unary", T_OBJECT, offsetof(gridCRF_t, unary_pyarr),0, "Unary matrix"},
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
  mainobj_docstr,           /* tp_doc */
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
  "grid based conditional random field",
  -1,
  NULL, NULL, NULL, NULL, NULL
};

#endif
