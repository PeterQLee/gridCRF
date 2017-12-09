#include "gridCRF.h"
f32 ALPHA=0.001;
static void gridCRF_dealloc(gridCRF_t *self) {
  free(self->float_data);
  Py_DECREF(self->V);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int gridCRF_init(gridCRF_t *self, PyObject *args, PyObject *kwds){
  self->V=NULL;
  i32 depth=0;
  if (!PyArg_ParseTuple(args,"i",&(depth))) return NULL;
  self->n_factors = (1 << depth) -1; //minus something else
  self->float_data = malloc((self->n_factors * 4 ) * sizeof(fact_type));
  npy_intp dims[2]= {self->n_factors, 4};
  self->V=PyArray_SimpleNewFromData(2,dims,FACT_TYPE,self->float_data);
  Py_INCREF(self->V);
			    
  return 0;
}
static PyObject * gridCRF_new (PyTypeObject *type, PyObject *args, PyObject *kwds){
  gridCRF_t *self;
  self=(gridCRF_t*)type->tp_alloc(type,0);
  return (PyObject *)self;
}

//backend functions

static void _train( gridCRF_t * self, PyArrayObject *X, PyArrayObject *Y){
  PyArrayObject *V=self->V;
  npy_intp * dims= PyArray_DIMS(X);
  i32 i,j,n,c,a,b;
  i32 *l;
  i32 n_factors=self->n_factors;
  
  n=self->depth;
  f64 *stack[self->n_factors ]; //TODO, preallocate this on heap.
  f64 *tmp;
  f32 *p;
 
  f32 yv[2], change[2];
  f32 max,den;
  for (i=0;i<dims[0];i++) {
    for (j=0;j<dims[1];j++) {
      *((f64 *) yv)=0.0; // set outome to 0
      if (((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && ((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      //do traversal.
      c=0;
	
      for (a=-n;a<=n;a++) {
	for (b=-n;b<=0;b++){
	  if (b==0 && a==n) break;
	  l=(i32*) PyArray_GETPTR3(Y,i+a,j+b,0);
	  tmp=(f64*)PyArray_GETPTR3(X,i+a,j+b,l);
	  stack[c++] = tmp;

	  //TODO: Improve with SSE
	  yv[0]+= ((f32*)tmp)[0];
	  yv[1]+= ((f32*)tmp)[1];
	  
	}
      }
      //Softmax y

      max=yv[0]>yv[1]? yv[0]:yv[1];

      yv[0]=exp(yv[0]-max);
      yv[1]=exp(yv[1]-max);
      den=1/(yv[0]+yv[1]);
      
      //TODO: speed up with SSE ops
      yv[0]=yv[0]*den;
      yv[1]=yv[1]*den;

      //TODO: speed up with SSE ops
      l=(i32*)PyArray_GETPTR3(Y,i,j,0);
      p=(f32*)PyArray_GETPTR3(X,i,j,0);
      change[0] = ALPHA * ((*l)-yv[0]) * (*p);
      l=(i32*)PyArray_GETPTR3(Y,i,j,1);
      p=(f32*)PyArray_GETPTR3(X,i,j,1);
      change[1] = ALPHA * ((*l)-yv[1]) * (*p);

      //update all of the pointers we visited with change.
      //TODO: speed up with SSE, multithreading, or GPU impl.

      for (a=0;a<n_factors;a++){
	//TODO: speed up (SSE) __m128 _mm_add_ps
	*((f32*)&stack[a]) += change[0];
	*((f32*)(&stack[a] + sizeof(f32))) += change[1];
      }
    }
  }
  //Values trained now
}

static void _loopyCPU(gridCRF_t* self, PyArrayObject *X, PyArrayObject *Y){
  //loopy belief propagation using CPU

  //Need to initialize all messages
  npy_intp * dims= PyArray_DIMS(X);
  i32 n_factors=self->n_factors;
  i32 m,n,d,depth=self->depth,i,x,y;
  f32 * F_V = (f32 *) calloc(self->n_factors* 2* dims[0] * dims[1]  *2, sizeof(f32));
  f32 * V_F = (f32 *) calloc( self->n_factors* 2* dims[0] * dims[1]  *2 , sizeof(f32));
  f32 * marginals= (f32* ) calloc(dims[0]*dims[1]*2,sizeof(f32));
  f32 * outgoing= (f32*) malloc(sizeof(f32) *  self->n_factors * 2);
  i32 rowv_[8]={0,2,4,6,8,10,12,14};
  const i32 roffset =1 , coffset=2;
  i32 colv_[8]={0,1,4,5,8,9,12,13};
  __m256i *rowv=(__m256i *)rowv_;
  __m256i *colv=(__m256i *)colv_;
  __m256 r1,r2,r3;
  
  f32 * ms = malloc(sizeof(f32) * self->n_factors*4);
  // direction, x,y, factor, prob
  i32 com[n_factors]; //TODO: preallocate
  i32 rom[n_factors];
  n=0;
  for (d=1;d<=depth;d++ ) {
    for (i=0;i<d*4;i++) {
      if (i<d) {
	com[n]= -d *dims[1] * self->n_factors*2*2 - i*self->n_factors*2*2;
	rom[n]= +d *dims[1] * self->n_factors*2*2 + i*self->n_factors*2*2;
      }
      else if (i>=d*3) {
	com[n]= +d *dims[1] * self->n_factors*2*2 - (d-(i-d*3))*self->n_factors*2*2;
	rom[n]= -d *dims[1] * self->n_factors*2*2 + (d-(i-d*3))*self->n_factors*2*2;	
      }
      else{
	com[n]= (-d+i)*dims[1] * self->n_factors*2*2 - d*self->n_factors*2*2;
	com[n]= (d-i)*dims[1] * self->n_factors*2*2 + d*self->n_factors*2*2;
      }
      n++;
    }
  }

  //__m256 adj;
  //Need softmaxed versions of V

  //Checkout __m256_max_ps, for factor to variable messages
  RE [2,n_factors * 2, 2];
  for (x=0;x<dims[0];x++) {
    for (y=0;y<dims[1];y++) {
      //factor to variable messages
      //Make MS, energies + variable to factor messages..
      for (n=0;n<n_factors; n+=4) {
	//r1=_mm256_i32gather_ps(ms,co,sizeof(f32));
	//r2=_mm256_i32gather_ps(&(ms[roffset]),co,sizeof(f32));
	//_mm256_max_ps(r1,r2);
	r1=_mm256_load_ps(RE[0,n_factors]);
	r2=_mm256_load_ps(RE[1,n_factors]);
	r3=_mm256_load_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],n_factors,2)]);
	r1=_mm256_add_ps(r1,r3);
	r2=_mm256_add_ps(r2,r3);
	r3=_mm256_max_ps(r1,r2);
	//Need to delegate r3 to appropriate destinations
	
	for (m=n;m<n+4;m++){ //todo, calculate cm
	  ((f64*)&F_V)[com[m]] = ((f64*)&r3)[m-n];
	}
	  
      }

      for (n=n_factors;n<n_factors; n+=4) {
	r1=_mm256_load_ps(CE[0,n_factors]);
	r2=_mm256_load_ps(CE[1,n_factors]);
	r3=_mm256_load_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],n_factors,2)]);
	r1=_mm256_add_ps(r1,r3);
	r2=_mm256_add_ps(r2,r3);
	r3=_mm256_max_ps(r1,r2);
	//Need to delegate r3 to appropriate destinations
	for (m=n;m<n+4;m++){ //todo, calculate rom
	  ((f64*)&F_V)[rom[m-n_factors]] = ((f64*)&r3)[m-n];
	}
      }
      //luckily, # factors is guarunteed to be divisible by 4. So no worry about edge cases!


    }
  }
  for (x=0;x<dims[0];x++) {
    for (y=0;y<dims[1];y++) {
      //variable to factor messages
      f64 base=Fx_Y[x,y];
      r1=_mm256_set1_pd(base); //set all elements in vector this thi
      for (n=0;n<n_factors*2;n+=4) {
	_mm256_store_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],n_factors,2)] ,r1);
      }
      for (i=0;i<n_factors*2;i++) {
	base=*(f64*)(&F_V[x,y,i]);
	r1=_mm256_set1_pd(base);
	for (n=0;n<n_factors*2;n+=4) {
	  r2=_mm256_load_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],n_factors,2)]);
	  r2=_mm256_add_ps(r2,r1);
	  _mm256_store_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],n_factors,2)],r2);
	}
      }
      for (n=0;n<n_factors*2;n+=4) { //correct double counting
	r1=_mm256_load_ps(&F_V[COORD3(x,y,n,dims[0],dims[1],n_factors,2)]);
	r2=_mm256_load_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],n_factors,2)]);
	r2=_mm256_sub_ps(r2,r1);
	_mm256_store_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],n_factors,2)],r2);
      }
  
    }
  }
}

  
// visible functions

static void fit (gridCRF_t * self, PyObject *args){
  PyObject *train,*lab,*f,*g;
  Py_ssize_t n;
  i32 i;
  if (!PyArg_ParseTuple(args,"oo",train,lab)) return NULL;
  //Needs to be a list of nd arrays
  if (!PyList_Check(train)) return NULL;
  if (!PyList_Check(lab)) return NULL;
  f=PyList_GetItem(train,0);
  if (!PyArray_Check(f) || PyArray_NDIM(f) !=3 || PyArray_DIMS(f)[2] !=2) return NULL;
  f=PyList_GetItem(lab,0);
  if (!PyArray_Check(f) || PyArray_NDIM(f) !=3 || PyArray_DIMS(f)[2] !=2) return NULL;

  //TODO: check num elements of f is the same
  //TODO: Also need to ensure that Y is integer and X is f32
  
 //If all these checks have passed the input data is valid.
  //Now we can begin training iterations
  
  n=PyList_Size(train);
  assert n==PyList_Size(lab);
  for (i=0;i<n;i++) {
    //TODO: check to ensure that each element in train and lab match dimensions
    g=PyList_GetItem(train,i);
    h=PyList_GetItem(train,i);
    _train(self,(PyArrayObject*)g,(PyArrayObject*)h);
  }
  
}//PyArrayObject *X, PyArrayObject *Y){

static PyArrayObject *predict(gridCRF_t* self, PyObject *args)[//PyArrayObject *X, PyArrayObject *Y){
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
  PyModule_AddObject(m, "SquareCRF", (PyObject *)&gridCRF_Type);
  return m;

}
