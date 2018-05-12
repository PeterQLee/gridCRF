#include "gridCRF.h"
#include "common.h"
#include <assert.h>
#include <stdio.h>


f32 ALPHA=0.001;

//m, epsilon, past, dleta, max_iterations, linesearch, max_linesearch, min_step, max_step,ftol,wolfe. gtol,xtp;.prtjamtwose+c,orthantwise_start, orthantwise_end

static void gridCRF_dealloc(gridCRF_t *self) {
  printf("Dealloc\n");
  //if (self->V_data != NULL)
  //  free(self->V_data);
  //if (self->V != NULL) 
  //  Py_DECREF(self->V);

  //TODO: free rom and com
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int gridCRF_init(gridCRF_t *self, PyObject *args, PyObject *kwds){
  i64 i;
  i64 n_factors,depth;

  self->V=NULL;
  self->V_data=NULL;
  self->depth=0;
  self->gpuflag=0;
  npy_int depth_=0;
  static char * kwlist []= {"depth", "gpuflag" ,NULL};
  
  if (!PyArg_ParseTupleAndKeywords(args,kwds,"i|i", kwlist, &depth_, &(self->gpuflag))) return 1;
  //depth_=1;
  depth=(i64)depth_;
  self->depth=depth;
  
  n_factors=0;
  for (i=1;i<=depth;i++){
    n_factors= n_factors + i*4;
  }
  self->n_factors= n_factors;
  self->V_data = _mm_malloc((n_factors*2*4) * sizeof(f32),32); //TODO: use aligned malloc
  self->unary = malloc(sizeof(f32)*4);
  self->unary[0]=0.0f;
  self->unary[1]=0.0f;
  self->unary[2]=0.0f;
  self->unary[3]=0.0f;
  for (i=0;i<n_factors*4*2;i++) {
    self->V_data[i]=0.0f; //temporary
  }
  
  npy_intp dims[2]= {n_factors*2, 4};
  self->V=(PyArrayObject *)PyArray_SimpleNewFromData(2,dims,NPY_FLOAT32,self->V_data);
  npy_intp dims1[2]= {2, 2};
  self->unary_pyarr=(PyArrayObject *)PyArray_SimpleNewFromData(2,dims1,NPY_FLOAT32,self->unary);
  Py_INCREF(self->V);
  Py_INCREF(self->unary_pyarr);
  

  return 0;
}
static PyObject * gridCRF_new (PyTypeObject *type, PyObject *args, PyObject *kwds){
  gridCRF_t *self;
  self=(gridCRF_t*)type->tp_alloc(type,0);
  return (PyObject *)self;
}


static void _train( gridCRF_t * self, PyObject *X_list, PyObject *Y_list, train_params_t *tpt){
  #define NUM_UNARY 4
  #define VERBOSE 0 
  i64 depth= self->depth;
  i32 i,j,n,a,b,d,k;
  i32 its, epochs=tpt->epochs;//epochs=1000;
  i32 *l;
  i64 n_factors=self->n_factors;
  f32 *V = self->V_data;
  f32 *V_change = _mm_malloc(sizeof(f32)*(n_factors*4*2+NUM_UNARY),32);

  n=self->depth;
  //f64 *stack[self->n_factors ]; //TODO, preallocate this on heap.
  f32 *tmp;
  f32 *p,*v;
 
  f32 yv[2], change[2];
  f32 max,den;

  /* Coordinate offsets */
  i32 *ainc = malloc(sizeof(i32)*n_factors*2);
  i32 *binc = malloc(sizeof(i32)*n_factors*2);
  
  f32 L=0.0;
  f32 alpha=tpt->alpha;

  __m256 r1,r2;
  
  f32 *unary = self->unary;
  f32 *unary_change  = &V_change[n_factors*4*2];
  i32 num_params=n_factors*4*2 +NUM_UNARY;
  i32 cur,last;
  
  //printf("n_factors %d\n",n_factors);
  //Get the appropriate coordinate offsets
  n=0;
  for (d=1;d<=depth;d++) {
    for (k=0;k<d*4;k++) {
      if (k<d+1) {
	a=-d;
	b=-k;
      }
      else if (k>=d*3) {
	a=d;
	b=-(d-(k-d*3));
      }
      else{
	a=-d+k-d;
	b=-d;
      }
      //printf("n %d %d %d %d\n",n,depth,a,b);
      ainc[n]=a;
      binc[n]=b;

      ainc[n+n_factors]=-a;
      binc[n+n_factors]=-b;
	
      n++;

    }
  }


  loopy_params_t lpar;
  lpar.mu=NULL;
  lpar.max_its=100;//100;
  lpar.stop_thresh=0.001;
  lpar.eval=1;
  lpar.EY=NULL;
  lpar.n_threads=8;

  //calculate initial gradient to get things going
  gradient_t gradargs;


  gradargs.X_list=X_list;
  gradargs.Y_list=Y_list;
  gradargs.V_change=V_change;
  gradargs.self=self;
  gradargs.ainc=ainc;
  gradargs.binc=binc;
  gradargs.unary_change=unary_change;
  gradargs.dims=NULL;
  gradargs.num_params=num_params;
  gradargs.n_factors=n_factors;
  gradargs.alpha=alpha;
  gradargs.error_func = tpt->error_func;
  //gradargs.loopy_func=&_loopyCPU;

  if (self->gpuflag) {
    gradargs.loopy_func=&loopyGPU;
  }
  else{
    gradargs.loopy_func=&loopyCPU;
  }
  
  gradargs.lpar=&lpar;

  #define N_THREADS 1

  if (self->gpuflag){
    GPU_grad_descent(&gradargs,epochs,0);
  }
  else{
    grad_descent(&gradargs,epochs,N_THREADS);
  }

  //Values trained now
  printf("Done train");
  free(ainc);
  free(binc);
  
}


// visible functions

static PyObject* fit (gridCRF_t * self, PyObject *args,PyObject *kwds){

  printf("Start fit\n");
  train_params_t tpt;
  tpt.epochs=100;
  tpt.alpha=0.001f;
  tpt.error_func = 0;
  static char * kwlist[] = {"train","lab","epochs","alpha", "error_type",NULL};
  PyObject *train,*lab,*X,*Y,*f;
  Py_ssize_t n;
  i32 i;
  if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO|ifi",kwlist,&train,&lab,&(tpt.epochs),&(tpt.alpha), &(tpt.error_func) )) return NULL;
  //Needs to be a list of nd arrays
  if (!PyList_Check(train)) {
    PyErr_SetString(PyExc_ValueError, "train must be a list");
    return NULL;
  }
  if (!PyList_Check(lab)) {
    PyErr_SetString(PyExc_ValueError, "lab must be a list");
    return NULL;
  }
  
 //If all these checks have passed the input data is valid.
  //Now we can begin training iterations
  
  n=PyList_Size(train);
  assert(n==PyList_Size(lab));
  
  for (i=0;i<n;i++) {
    //TODO: check to ensure that each element in train and lab match dimensions
    X=PyList_GetItem(train,i);
    Y=PyList_GetItem(lab,i);
    if (PyArray_TYPE(X) != NPY_FLOAT32) {
      PyErr_SetString(PyExc_TypeError, "Training image must be 32-bit floats");
      return NULL;
    }
    if (PyArray_TYPE(Y) != NPY_UINT32) {
      PyErr_SetString(PyExc_TypeError, "Label image must be 32-bit unsigned ints");
      return NULL;
    }
    if (!PyArray_Check(X) || PyArray_NDIM(X) !=3 || PyArray_DIMS(X)[2] !=2) {
      PyErr_SetString(PyExc_ValueError, "train must only contain 3 dimensional arrays, of a binary image");
      return NULL;
    }
    if (!PyArray_Check(Y) || PyArray_NDIM(Y) !=3 || PyArray_DIMS(Y)[2] !=2) {
      PyErr_SetString(PyExc_ValueError, "lab must only contain 3 dimensional arrays, of a binary image");
      return NULL;
    }
  }

  
  _train(self,(PyObject*)train,(PyObject*)lab,&tpt);

  return Py_BuildValue("");  
}

static PyObject *predict(gridCRF_t* self, PyObject *args, PyObject *kwds){//PyArrayObject *X, PyArrayObject *Y){
  PyObject *test, *out, *refimg=NULL;
  loopy_params_t lpar;
  lpar.stop_thresh=0.01f;
  lpar.max_its=100;
  lpar.eval=1;
  lpar.n_threads=8;


  static char * kwlist []= {"X","stop_thresh","max_its","n_threads" ,NULL};
  if (!PyArg_ParseTupleAndKeywords(args,kwds,"O|fii",kwlist,&test,&(lpar.stop_thresh),&(lpar.max_its),&(lpar.n_threads))) return NULL;
  if (!PyArray_Check(test) || PyArray_NDIM(test) !=3 || PyArray_DIMS(test)[2] !=2) {
    PyErr_SetString(PyExc_ValueError, "Input must be a 3 dimensional array");
    return NULL;
  }
  if (PyArray_TYPE(test) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_TypeError, "Data must be 32-bit floats");
    return NULL;
  }
  lpar.mu= (f32* ) _mm_malloc(PyArray_DIMS(test)[0]*PyArray_DIMS(test)[1]*2*sizeof(f32),32);
  
  out=PyArray_SimpleNew(2, PyArray_DIMS(test), NPY_INT32);
  
  lpar.EY=PyArray_DATA(out);

  if (self->gpuflag) {
    //loopyGPU(self,test,&lpar,refimg);
    predict_loopyGPU(self,test,&lpar, refimg);
  }
  else{
    loopyCPU(self,test,&lpar,refimg);
    //_loopyCPU(self,test,&lpar,refimg);
  }
  
  //return Py_BuildValue("");
  return out;
}




PyMODINIT_FUNC PyInit_gridCRF(void) {
  import_array();
  srand(1);
  PyObject *m;
  gridCRF_Type.tp_new=PyType_GenericNew;
  if (PyType_Ready(&gridCRF_Type) < 0)
    return NULL;

  m = PyModule_Create(&gridCRFmodule);
  if (m == NULL)
      return NULL;

  Py_INCREF(&gridCRF_Type);
  PyModule_AddObject(m, "gridCRF", (PyObject *)&gridCRF_Type);
  return m;

}
