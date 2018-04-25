#include "gridCRF.h"
#include "common.h"
#include <assert.h>
#include <stdio.h>
#include <lbfgs.h>

f32 ALPHA=0.001;

//m, epsilon, past, dleta, max_iterations, linesearch, max_linesearch, min_step, max_step,ftol,wolfe. gtol,xtp;.prtjamtwose+c,orthantwise_start, orthantwise_end
static const lbfgs_parameter_t _defparam = {
  60, 1e-5, 0, 1e-5,
  100, LBFGS_LINESEARCH_DEFAULT, 40,
  1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
  0.0, 0, -1,
};
//-1000 min_step
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
  i64 n, d, n_factors,depth;

  self->V=NULL;
  self->V_data=NULL;
  self->depth=0;
  npy_int depth_=0;

  
  if (!PyArg_ParseTuple(args,"i",&depth_)) return 1;
  //depth_=1;
  depth=(i64)depth_;
  self->depth=depth;
  
  n_factors=0;
  for (i=1;i<=depth;i++){
    n_factors= n_factors + i*4;
  }
  self->n_factors= n_factors;
  self->V_data = _mm_malloc((n_factors*2 * 4 ) * sizeof(f32),32); //TODO: use aligned malloc
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
  Py_INCREF(self->V);

  

  return 0;
}
static PyObject * gridCRF_new (PyTypeObject *type, PyObject *args, PyObject *kwds){
  gridCRF_t *self;
  self=(gridCRF_t*)type->tp_alloc(type,0);
  return (PyObject *)self;
}

static int progress ( void * instance,
		      const lbfgsfloatval_t *x,
		      const lbfgsfloatval_t *g,
		      const lbfgsfloatval_t fx,
		      const lbfgsfloatval_t xnorm,
		      const lbfgsfloatval_t gnorm,
		      const lbfgsfloatval_t step,
		      int n,
		      int k,
		      int ls){
  return 0;
}
static i32 count=0;
//backend functions
static lbfgsfloatval_t _lbfgs_update(void *arg, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step){
  gradient_t *args=(gradient_t *) arg;
  int i;
  gridCRF_t *self=args->self;
  printf("Post self %x %x\n",self,arg);
  f32 *unary=(args->self)->unary;
  f32 *V=(args->self)->V_data;

  i32 num_params= args->num_params;
  i32 n_factors=args->n_factors;
  f32 *V_change=args->V_change;
  const f32 lr=1;
  /*
  for (i=0;i<n_factors*4*2;i++){
    V[i]+=lr*V_change[i];
  }
  unary[0]+=lr*V_change[n_factors*4*2];
  unary[1]+=lr*V_change[n_factors*4*2+1];
  unary[2]+=lr*V_change[n_factors*4*2+2];
  unary[3]+=lr*V_change[n_factors*4*2+3];
  */
  
  for (i=0;i<n_factors*4*2;i++){
    V[i]+=lr*x[i];
  }
  unary[0]+=lr*x[n_factors*4*2];
  unary[1]+=lr*x[n_factors*4*2+1];
  unary[2]+=lr*x[n_factors*4*2+2];
  unary[3]+=lr*x[n_factors*4*2+3];
  
  //12.48040179  -3.73883933
  /*
  unary[0]=12.48;
  unary[1]=-3.739;

  unary[2]=-12.48;
  unary[3]=3.739;
  */
  _calculate_gradient(args);
  printf("g");
  for (i=0;i<num_params;i++) {
    g[i]=V_change[i]; //this works because V_change and unary change are contitigous
    
  }
  f32 L=0;
  printf("%f %f %f %f", g[n_factors*4*2],g[n_factors*4*2+1],g[n_factors*4*2+2],g[n_factors*4*2+3]);  
  printf("\nError %f %d\n",L,count++);
  return L;
}

static void grad_descent(gradient_t *args,i64 epochs,i64 n_threads) {
  i64 h,i,j;
  //PyArrayObject *EY;
  const i32 N_UNARY=4;
  f32 *unary=(args->self)->unary;
  f32 *V=(args->self)->V_data;
  pthread_t *threads= malloc(sizeof(pthread_t)*n_threads);
  npy_intp *dims=args->dims;
  i32 *EY=_mm_malloc(sizeof(i32)*dims[0]*dims[1],2);
  args->lpt->EY=EY;
  
  i32 num_params= args->num_params;
  i64 n_factors=args->n_factors;
  gradient_t *targs=malloc(sizeof(gradient_t) * n_threads);
  npy_intp *start= malloc(sizeof(npy_intp)*n_threads*2);
  npy_intp *stop= malloc(sizeof(npy_intp)*n_threads*2);
  PyArrayObject *X=args->X;
  f32 *V_change;
  npy_intp s0=0;
  for (i=0;i<n_threads;i++ ){
    memcpy(&targs[i],args,sizeof(gradient_t));
    V_change=_mm_malloc(sizeof(f32)*(n_factors*4*2+N_UNARY),32);
    targs[i].V_change=V_change;
    targs[i].unary_change = &V_change[n_factors*4*2];
    start[2*i]=s0/dims[1];
    start[2*i+1]=s0%dims[1];
    s0+=dims[0]*dims[1]/n_threads;
    stop[2*i]=s0/dims[1];
    stop[2*i+1]=s0%dims[1];
    targs[i].start=&start[2*i];
    targs[i].stop=&stop[2*i];
  }
  stop[n_threads*2-2]=dims[0];
  stop[n_threads*2-1]=dims[1];
  args->lpt->EY=EY;
  const f32 lr=0.01;
  f32 totL;
  
  for (i=0;i<epochs;i++) {
    (*args->loopy_func)(args->self,X,args->lpt,NULL);
    
    for (h=0;h<n_threads;h++) {
      pthread_create(&threads[h],NULL,(void*) _calculate_gradient,&targs[h]);
    }
    for (h=0;h<n_threads;h++) {
      pthread_join(threads[h],NULL);
    }
    totL=0.0f;
    /* update params */
    for (h=0;h<n_threads;h++) {
      for (j=0;j<n_factors*4*2;j++){
	V[j]+=lr*targs[h].V_change[j];
      }
      unary[0]+=lr*targs[h].V_change[n_factors*4*2];
      unary[1]+=lr*targs[h].V_change[n_factors*4*2+1];
      unary[2]+=lr*targs[h].V_change[n_factors*4*2+2];
      unary[3]+=lr*targs[h].V_change[n_factors*4*2+3];
      totL+=targs[h].L/n_threads;
    }
    printf("TOTL %f\n",totL);

  }
  for (i=0;i<n_threads;i++ ){
    _mm_free(targs[i].V_change);
  }
  free(targs);
  free(stop);
  free(start);
  free(threads);
  _mm_free(EY);
}
static void* _calculate_gradient(gradient_t *args) {
  i32 i,j;
  PyArrayObject *X=args->X;
  PyArrayObject *Y=args->Y;
  f32 *unary= args->self->unary;
  i32 *ainc=args->ainc, *binc=args->binc;
  f32 * V=args->self->V_data;
  npy_intp *start=args->start, *stop=args->stop;
  f32 *V_change = args->V_change;
  f32 *unary_change = args->unary_change;
  f32 L=0.0f,max=0.0f,den,*p;
  f32 yv [2];
  f32 change[2];
  f32 *v;
  i32 *l;
  npy_intp *dims=args->dims;
  i32 n_factors = args->n_factors;
  
  memset(V_change, 0, sizeof(f32)*n_factors*4*2);
  unary_change[0]=0.0f;
  unary_change[1]=0.0f;
  unary_change[2]=0.0f;
  unary_change[3]=0.0f;

  i32 n=0;
  f32 *tmp;
  f32 alpha=args->alpha;
  printf("dims %d %d\n",dims[0],dims[1]);
  
  //PyArrayObject *EY = args->EY;
  i32 *EY = args->EY;
    
  for (i=start[0];i<dims[0];i++) {
    for (j=start[1];j<dims[1];j++) {
      if (i==stop[0] && j==stop[1]) goto grad_finish;
      *((f64 *)yv)=0.0; // set outome to 0
      
      if (*((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && *((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      
      tmp=(f32*)PyArray_GETPTR3(X,i,j,0);
      yv[0]=(unary[0]*tmp[0]+unary[1]*tmp[1]);
      yv[1]=(unary[2]*tmp[0]+unary[3]*tmp[1]);


      for (n=0;n<n_factors;n++) {
	if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  
	//l=(i32*) PyArray_GETPTR2(EY,i+ainc[n],j+binc[n]);
	l=&EY[COORD2(i+ainc[n],j+binc[n],dims[0],dims[1],1)];
	v=&V[n*4 + ((*l)&1)*2]; // 4x4 transfer matrix for n	
	//We pick the row that corresponds to the outcome
	//TODO: Improve with SSE
	  
	yv[0] += v[0]; //(Dependent on function)
	yv[1] += v[1];
      }
      //do left to right
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	//l=(i32*) PyArray_GETPTR2(EY,i+ainc[n+n_factors],j+binc[n+n_factors]);
	l=&EY[COORD2(i+ainc[n],j+binc[n],dims[0],dims[1],1)];
	v=&V[n_factors*4 + n*4 + 2*((*l)&1)]; 
	yv[0] += v[0];
	yv[1] += v[1];
      }
      max=-yv[0]>-yv[1]? -yv[0]:-yv[1];
    
      yv[0]=exp(-yv[0]-max);
      yv[1]=exp(-yv[1]-max);
      den=1/(yv[0]+yv[1]);
      yv[0]=yv[0]*den;
      yv[1]=yv[1]*den;

      l=((i32*)PyArray_GETPTR3(Y,i,j,0));
      p=(f32*)PyArray_GETPTR3(X,i,j,0);
      //L-= (*l) * log(yv[0]) /dims[0]/dims[1];
      if (*l && yv[0]!=0.0f) {
	if (isinf(yv[0])) {
	  printf("INF yv0\n");
	}
	L-=  log(yv[0]) /dims[0]/dims[1];
      }
      if (*l && yv[0]==0.0f) {
	L+=100;
      }
      change[0] = -alpha * (((*l)&1)-yv[0]) ;
      unary_change[0] += -alpha*(((*l)&1)-yv[0])*tmp[0];
      unary_change[1] += -alpha*(((*l)&1)-yv[0])*tmp[1];
	
      p=(f32*)PyArray_GETPTR3(X,i,j,1);
      l=((i32*)PyArray_GETPTR3(Y,i,j,1));
      
      //L-= (*l)* log(yv[1])/dims[0]/dims[1];
      if (*l && yv[1]!=0.0f) {
	if (isinf(yv[1])) {
	  printf("INF yv1 %f\n",yv[1]);
	}

	L-= log(yv[1])/dims[0]/dims[1];
      }
      if (*l && yv[1]==0.0f) {
	L+=100;
      }

      //printf("yv %f %f\n",yv[0],yv[1]);
      change[1] = -alpha * (((*l)&1)-yv[1]);
      
      unary_change[2] += -alpha*(((*l)&1)-yv[1])*tmp[0];
      unary_change[3] += -alpha*(((*l)&1)-yv[1])*tmp[1];
      //printf("Part L %f %f %f %d %d\n",yv[0],yv[1],L ,  *((i32*)PyArray_GETPTR3(Y,i,j,0)),*l);
      
      if (isinf(L)){
	printf("ISINF\n");
	//exit(1);
      }

      for (n=0;n<n_factors;n++){
	if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	//TODO: speed up (SSE) __m128 _mm_add_ps
	//l=((i32*) PyArray_GETPTR2(EY,i+ainc[n],j+binc[n]));
	l=&EY[COORD2(i+ainc[n],j+binc[n],dims[0],dims[1],1)];
	V_change[n*4 + 2*((*l)&1)] += change[0];
	V_change[n*4 + 2*((*l)&1) + 1] += change[1];
      }
      
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	//l=(i32*) PyArray_GETPTR2(EY,i+ainc[n+n_factors],j+binc[n+n_factors]);

	l=&EY[COORD2(i+ainc[n+n_factors],j+binc[n+n_factors],dims[0],dims[1],1)];
	V_change[n_factors*4 +n*4 + 2*((*l)&1)] += change[0];
	V_change[n_factors*4 +n*4 + 2*((*l)&1) + 1] += change[1];	  
      }
    }

  }
 grad_finish:
  //Py_DECREF(EY);
  printf("L %f %d %d, %d %d: %d %d\n",L, start[0],start[1],stop[0],stop[1], i,j);
  args->L=L;
  //return L;
  return NULL;
}

static f32 _incremental_gradient(gradient_t *args) {
  i64 i,j,i_,j_;
  PyArrayObject *X=args->X;
  PyArrayObject *Y=args->Y;
  printf("Y %x\n",Y);
  f32 *unary= args->self->unary;
  i32 *ainc=args->ainc, *binc=args->binc;
  f32 * V=args->self->V_data;

  f32 L=0.0f,max=0.0f,den,*p;
  f32 yv [2];
  f32 change[2];
  f32 *v;
  i32 *l;
  npy_intp *dims=args->dims;
  i32 n_factors = args->n_factors;
  

  i32 n=0;
  f32 *tmp;
  f32 alpha=args->alpha;
  printf("dims %d %d\n",dims[0],dims[1]);
  i64* i_inds=indlist(dims[0]);
  i64* j_inds= indlist(dims[1]);
  
  PyArrayObject *EY = (*args->loopy_func)(args->self,X,args->lpt,NULL);
    
  for (i_=0;i_<dims[0];i_++) {
    for (j_=0;j_<dims[1];j_++) {
      i=i_inds[i_];
      j=j_inds[j_];
      *((f64 *)yv)=0.0; // set outome to 0
      
      if (*((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && *((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      
      tmp=(f32*)PyArray_GETPTR3(X,i,j,0);
      yv[0]=(unary[0]*tmp[0]+unary[1]*tmp[1]);
      yv[1]=(unary[2]*tmp[0]+unary[3]*tmp[1]);

      //right to left factors
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  
	l=(i32*) PyArray_GETPTR2(EY,i+ainc[n],j+binc[n]);
	v=&V[n*4 + ((*l)&1)*2]; // 4x4 transfer matrix for n	
	//We pick the row that corresponds to the outcome
	//TODO: Improve with SSE
	  
	yv[0] += v[0]; //(Dependent on function)
	yv[1] += v[1];

	/* Mar 3:
	   Two options here. Can either use fuzzy probability to
	   modify these parameters or can use the argmax as the label.
	 */
      }
      //do left to right
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	l=(i32*) PyArray_GETPTR2(EY,i+ainc[n+n_factors],j+binc[n+n_factors]);
	v=&V[n_factors*4 + n*4 + 2*((*l)&1)]; 
	yv[0] += v[0];
	yv[1] += v[1];
      }
      max=-yv[0]>-yv[1]? -yv[0]:-yv[1];
    
      yv[0]=exp(-yv[0]-max);
      yv[1]=exp(-yv[1]-max);
      den=1/(yv[0]+yv[1]);
      yv[0]=yv[0]*den;
      yv[1]=yv[1]*den;

      l=((i32*)PyArray_GETPTR3(Y,i,j,0));
      p=(f32*)PyArray_GETPTR3(X,i,j,0);

      if (*l && yv[0]!=0.0f) {
	if (isinf(yv[0])) {
	  printf("INF yv0\n");
	}
	L-=  log(yv[0]) /dims[0]/dims[1];
      }
      change[0] = -alpha * (((*l)&1)-yv[0]) ;
      unary[0] += -alpha*(((*l)&1)-yv[0])*tmp[0];
      unary[1] += -alpha*(((*l)&1)-yv[0])*tmp[1];
	
      p=(f32*)PyArray_GETPTR3(X,i,j,1);
      l=((i32*)PyArray_GETPTR3(Y,i,j,1));
      
      //L-= (*l)* log(yv[1])/dims[0]/dims[1];
      if (*l && yv[1]!=0.0f) {
	if (isinf(yv[1])) {
	  printf("INF yv1 %f\n",yv[1]);
	}

	L-= log(yv[1])/dims[0]/dims[1];
      }
      //printf("yv %f %f\n",yv[0],yv[1]);
      change[1] = -alpha * (((*l)&1)-yv[1]);
      
      unary[2] += -alpha*(((*l)&1)-yv[1])*tmp[0];
      unary[3] += -alpha*(((*l)&1)-yv[1])*tmp[1];

      if (i==50 && j==50) {
	printf("50 50 change %f %f",-(((*l)&1)-yv[1])*tmp[0],-(((*l)&1)-yv[1])*tmp[1]);
      }

      //printf("Part L %f %f %f %d %d\n",yv[0],yv[1],L ,  *((i32*)PyArray_GETPTR3(Y,i,j,0)),*l);
      
      if (isinf(L)){
	printf("ISINF\n");
	//exit(1);
      }
      
      for (n=0;n<n_factors;n++){
	if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	//TODO: speed up (SSE) __m128 _mm_add_ps
	l=((i32*) PyArray_GETPTR2(EY,i+ainc[n],j+binc[n]));
	V[n*4 + 2*((*l)&1)] += change[0];
	V[n*4 + 2*((*l)&1) + 1] += change[1];
      }
      
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	l=(i32*) PyArray_GETPTR2(EY,i+ainc[n+n_factors],j+binc[n+n_factors]);
	V[n_factors*4 +n*4 + 2*((*l)&1)] += change[0];
	V[n_factors*4 +n*4 + 2*((*l)&1) + 1] += change[1];	  
      }
      
    }
  }
  printf("L %f\n",L);
  Py_DECREF(EY);
  return L;
}


static void _train( gridCRF_t * self, PyArrayObject *X, PyArrayObject *Y, train_params_t tpt){
  #define NUM_UNARY 4
  npy_intp * dims= PyArray_DIMS(X);
  i64 depth= self->depth;
  i32 i,j,n,a,b,d,k;
  i32 its, epochs=tpt.epochs;//epochs=1000;
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
  f32 alpha=tpt.alpha;

  __m256 r1,r2;
  f32 *unary = self->unary;
  //f32 unary_change[2]={0.0f,0.0f};
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

  f32 * mu= (f32* ) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
  loopy_params_t lpt;
  lpt.mu=mu;
  lpt.max_its=100;
  lpt.stop_thresh=0.001;
  lpt.eval=1;
  lpt.EY=NULL;

  //calculate initial gradient to get things going
  gradient_t gradargs;
  printf("PRE Y %x\n",Y);
  gradargs.X=X;
  gradargs.Y=Y;
  gradargs.V_change=V_change;
  gradargs.self=self;
  gradargs.ainc=ainc;
  gradargs.binc=binc;
  gradargs.unary_change=unary_change;
  gradargs.dims=dims;
  gradargs.num_params=num_params;
  gradargs.n_factors=n_factors;
  gradargs.alpha=alpha;
  gradargs.loopy_func=&_loopyCPU;
  gradargs.lpt=&lpt;

  
  #define LBFGS 0
  #define GRAD 1
  #if LBFGS
  _calculate_gradient(&gradargs);
  
  printf("PREDIM %d %d\n",dims[0],dims[1]);
  printf("Pre self %x %x\n",gradargs.self,&gradargs);

  lbfgsfloatval_t *x=lbfgs_malloc(num_params); //TODO: remember to free

  for (i=0;i<n_factors*4*2;i++) {
    x[i]=V_change[i];
  }
  
  x[n_factors*4*2]=unary_change[0];
  x[n_factors*4*2+1]=unary_change[1];
  x[n_factors*4*2+2]=unary_change[2];
  x[n_factors*4*2+3]=unary_change[3];

  //lbfgs_parameter_t lbfgs_param;  
  printf("PPre self %x %x\n",gradargs.self,&gradargs);
  //lbfgs_parameter_init(&lbfgs_param);

  f32 fx;
  i32 ret= lbfgs(num_params,x,&fx, _lbfgs_update, NULL, (void *)&gradargs, &_defparam);
  printf("orth %d %d\n",_defparam.orthantwise_start,epochs); //Invalid orthantwise_start
  printf("ret %d %d\n",ret,LBFGSERR_INVALID_ORTHANTWISE_START); //Invalid
								//orthantwise_start
  lbfgs_free(x);
  #elif GRAD
  grad_descent(&gradargs,epochs,4);
  #else
  for (i=0;i<epochs;i++ ){
    _incremental_gradient(&gradargs);
  }
  #endif

 
  printf("unary %f %f %f %f\n",unary[0],unary[1],unary[2],unary[3]);
  //Values trained now
  printf("Done train");
  free(ainc);
  free(binc);
  _mm_free(mu);
  

  
}

static i32* _loopyCPU(gridCRF_t* self, PyArrayObject *X,loopy_params_t *lpt,PyArrayObject *refimg){ //TODO:
													      //fix type
  i32 WARN_FLAG=1;
  //loopy belief propagation using CPU

  //Need to initialize all messages
  f32 a,b,c,d,denr,denc,maxv;
  npy_intp * dims= PyArray_DIMS(X);
  i64 n_factors=self->n_factors;
  i64 max_it=lpt->max_its,it;
  f32 max_marg_diff=0.0f;
  f32 stop_thresh=lpt->stop_thresh;
  i32 converged=0;
  
  f32 * V_data=self->V_data;
  f32 * unary= self->unary;
  f32 tmp[2];
  
  npy_intp x,y;
  i64 m,n,depth=self->depth,i,j,co;
  i32 l,ll;
  //f32 * F_V = (f32 *) calloc( dims[0] * dims[1] * (self->n_factors*2) *2, sizeof(f32));
  f32 * F_V = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32);
  f32 * V_F = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32);
  
  for (i=0;i<dims[0] * dims[1] * (n_factors*2) *2; i++){
    F_V[i]=0.0f;
    V_F[i]=0.0f;
  }

  f32 * marginals= (f32* ) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
  //f32 * mu= (f32* ) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
  f32 *mu = lpt->mu;
  for (i=0;i<dims[0]*dims[1]*2;i+=1) {
    mu[i]=BIG;
    marginals[i]=0.0f;
  }
  
  __m256 r1,r2,r3,r4;

  i32 *com=(i32*)malloc(sizeof(i32)*n_factors);
  i32 *rom=(i32*)malloc(sizeof(i32)*n_factors);
  om_pair *co_pairs=(om_pair*)malloc(sizeof(om_pair)*n_factors);
  om_pair cop;
  n=0;
  for (j=1;j<=depth;j++ ) {
    for (i=0;i<j*4;i++) {
      if (i<j) {
	com[n]= -j *dims[1] * n_factors*2*2 - i*n_factors*2*2;
	rom[n]= +j *dims[1] * n_factors*2*2 + i*n_factors*2*2;
	co_pairs[n]=(om_pair){-j,-i};
      }
      else if (i>=j*3) {
	com[n]= +j *dims[1] * n_factors*2*2 - (j-(i-j*3))*n_factors*2*2;
	rom[n]= -j *dims[1] * n_factors*2*2 + (j-(i-j*3))*n_factors*2*2;
	co_pairs[n]=(om_pair){j,-(j-(i-j*3))};
      }
      else{
	com[n]= (-2*j+i)*dims[1] * n_factors*2*2 - j*n_factors*2*2;
	rom[n]= (2*j-i)*dims[1] * n_factors*2*2 + j*n_factors*2*2;
	co_pairs[n]=(om_pair){-2*j+i,-j};
      }
      
      n++;
    }
  }
  
  n=0;

  

  //__m256 adj;
  //Need softmaxed versions of V
  //Checkout __m256_max_ps, for factor to variable messages
  i32 origin;
  f32 *RE= _mm_malloc(2 * n_factors * 2 * sizeof(f32),32); //This is the transfer function (i.e. V matrix)
  f32 *CE= _mm_malloc(2 * n_factors * 2 * sizeof(f32),32); //This is the transfer function (i.e. V matrix)
  f64 mvtemp;
  
  //Exponentiate and temporarily store in RE
  for (i=0;i<2*n_factors*2;i+=8) {
    r1=_mm256_load_ps(&V_data[i]);
    r2=_mm256_load_ps(&V_data[i + n_factors*4]);
    //r1=exp256_ps(r1);
    //assert (!(isnan(r1[6]) || isnan(r1[7])));

    /*Swap energies such that remote outcome=1 is seperated from
      remote outcome=0*/
    RE[i/2]=-r1[0];
    RE[i/2+1]=-r1[1];
    RE[n_factors*2+i/2]=-r1[2];
    RE[n_factors*2+i/2+1]=-r1[3];
    RE[i/2+2]=-r1[4];
    RE[i/2+3]=-r1[5];
    RE[n_factors*2+i/2+2]=-r1[6];
    RE[n_factors*2+i/2+3]=-r1[7];
    
    
    CE[i/2]=-r2[0];
    CE[i/2+1]=-r2[1];
    CE[n_factors*2+i/2]=-r2[2];
    CE[n_factors*2+i/2+1]=-r2[3];
    CE[i/2+2]=-r2[4];
    CE[i/2+3]=-r2[5];
    CE[n_factors*2+i/2+2]=-r2[6];
    CE[n_factors*2+i/2+3]=-r2[7];
    
   

    printf("SET %ld %ld %ld %ld %ld %ld %ld %ld\n", i/2, i/2+1, i/2+n_factors*2, i/2+n_factors*2+1 , i/2+2, i/2+3 , i/2+2+n_factors*2,i/2+3+n_factors*2);
    printf("VAL %f %f %f %f %f %f %f %f\n", RE[i/2], RE[i/2+1], RE[i/2+n_factors*2], RE[i/2+n_factors*2+1] , RE[i/2+2], RE[i/2+3] , RE[i/2+2+n_factors*2],RE[i/2+3+n_factors*2]);


  }


  for (it=0;it< max_it && !converged;it++){
    printf("it %d\n",it);
   
    l=1;
    for (x=0;x<dims[0];x++) {
      for (y=0;y<dims[1];y++) {
	//factor to variable messages
	//Make MS, energies + variable to factor messages..
	//do top and left factors
	origin=COORD3(x,y,0,dims[0],dims[1],2*n_factors,2);

	if (refimg)
	  l= *((i32*) PyArray_GETPTR2(refimg,x,y));
	for (n=0;n<n_factors && l; n+=4) {//Here

	  /*
	    r1: c0_00, c0_01
	        c1_00, c1_01,
		c2_00, c2_01,
		c3_00, c2_01
		
	    r2: c0_10, c0_11,
	        c1_10, c1_11
	        c2_10, c2_11
	        c3_10, c3_11

	  */
	  r1=_mm256_load_ps(&RE[n*2]);
	  r2=_mm256_load_ps(&RE[n_factors*2 + n*2]);
	  r3=_mm256_load_ps(&V_F[origin]);
	  r4=_mm256_permute_ps(r3, 0xA0); //Outcomes that start at 0
					  //for factors 0 and 1 and 2 and 3
	  r1=_mm256_add_ps(r1,r4);
	  
	  r4=_mm256_permute_ps(r3, 0xF5); //Outcomes that start at 1
					  //for factors 0 and 1 and 2 and 3
	  r2=_mm256_add_ps(r2,r4);

	  /* Take the max energy based on which state it came from*/
	  r3=_mm256_max_ps(r1,r2);

	  
	  /*Delegate r3 to appropriate destinations*/
	
	  for (m=n;m<n+4;m++){
	    cop=co_pairs[m];
	    if (x+cop.x <0 || x+cop.x >= dims[0] || y+cop.y < 0 || y+cop.y >=dims[1]) continue;
	    if (refimg && *((i32*)PyArray_GETPTR2(refimg,x+cop.x,y+cop.y))==0) continue;
	    co=origin+com[m] + 2*(m-n) + n_factors*2;
	    
	      
	    //((f64*)&F_V)[com[m]] = ((f64*)&r3)[m-n];
	    //((f64*)F_V)[origin+com[m]] = ((__m256d) r3)[m-n];
	    if (!(co< 0 || co >= dims[0] * dims[1] * (n_factors*2) *2)) {
	      F_V[co] = r3[2*(m-n)];
	      //printf("mxy %d %d %d %f %d %d %d %d\n", m,x,y, r3[2*(m-n)],co,x+cop.x,y+cop.y, 2*(m-n));
	      F_V[co+1] = r3[2*(m-n)+1];
	    }
	    
	  }
	}
	//do below and right factors
	origin=COORD3(x,y,0,dims[0],dims[1],2*n_factors,2);
	for (n=n_factors;n<2*n_factors && l; n+=4) {

	  r1=_mm256_load_ps(&CE[(n-n_factors)*2]);
	  r2=_mm256_load_ps(&CE[n_factors*2 + (n-n_factors)*2]);
	  r3=_mm256_load_ps(&V_F[origin]);
	  r4=_mm256_permute_ps(r3, 0xA0);
	  r1=_mm256_add_ps(r1,r4);
	  r4=_mm256_permute_ps(r3, 0xF5);
	  r2=_mm256_add_ps(r2,r4);
	  r3=_mm256_max_ps(r1,r2);
	  //Delegate r3 to appropriate destination
	  for (m=n;m<n+4;m++){ //todo, calculate rom
	    cop=co_pairs[m-n_factors];
	    if (x-cop.x <0 || x-cop.x >= dims[0] || y-cop.y < 0 || y-cop.y >=dims[1]) continue;
	    if (refimg && *((i32*)PyArray_GETPTR2(refimg,x-cop.x,y-cop.y))==0) continue;
	    co=origin+rom[m-n_factors] + 2*(m-n);
	    //((f64*)F_V)[origin+rom[m-n_factors]] = ((__m256d) r3)[m-n];
	    if (!(origin+rom[m-n_factors] + 2*(m-n) < 0 || origin+rom[m-n_factors] + 2*(m-n) >= dims[0] * dims[1] * (n_factors*2) *2)) {
	      F_V[co] = r3[2*(m-n)];
	      //printf("mxy %d %d %d %f %d %d %d %d %d %d\n", m,x,y, r3[2*(m-n)], co, x-cop.x,y-cop.y,2*(m-n),rom[m-n_factors], origin);
	      F_V[co+1] = r3[2*(m-n)+1];
	    }
	  }
	}
	//luckily, # factors is guarunteed to be divisible by 4. So no worry about edge cases!
      }
    }
    max_marg_diff=0;
    for (x=0;x<dims[0];x++) {
      for (y=0;y<dims[1];y++) {
	//variable to factor messages
	//f64 base=Fx_Y[x,y];
	f64 base= *((f64*)PyArray_GETPTR3(X,x,y,0));
	*((f64*)tmp) = base;
	tmp[0]=-(((f32*)&base)[0]*unary[0] + ((f32*)&base)[1]*unary[1]);
	tmp[1]=-(((f32*)&base)[0]*unary[2] + ((f32*)&base)[1]*unary[3]);
	//r1=(__m256)_mm256_set1_pd(base); //set all elements in vector this thi
	r1=(__m256)_mm256_set1_pd(*((f64*)tmp)); //set all elements in vector this thi
	//Warning: possible segfault
	
	for (n=0;n<n_factors*2;n+=4) { //Set baseline, since we know that unary is added to each V_F
	  _mm256_store_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)] ,r1);
	}
      
	for (i=0;i<n_factors*2;i++) {
	  //printf("xyi %ld %ld %ld\n",x,y,i);
	  base=*((f64*)(&F_V[COORD3(x,y,i,dims[0],dims[1],2*n_factors,2)]));
	  r1=(__m256)_mm256_set1_pd(base);
	  for (n=0;n<n_factors*2;n+=4) {
	    r2=_mm256_load_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)]);
	    r2=_mm256_add_ps(r2,r1);
	    _mm256_store_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)],r2);
	  }
	}
	for (n=0;n<n_factors*2;n+=8) { //correct double counting
	  r1=_mm256_load_ps(&F_V[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)]);
	  r2=_mm256_load_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)]);
	  r2=_mm256_sub_ps(r2,r1);
	  _mm256_store_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)],r2);
	}

	/*
	//TODO: normalize
	//This is SSE normalization. Unless more knowledge is gained, it would be slower to use these than individually going over each value.
	*/
	
	//Apply normalization
	for (n=0;n<n_factors*2;n++) {
	  //TODO: optimize
	  a=V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)];
	  b=V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)+1];
	  a=0.5*(a+b);
	  V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)]-=a;
	  V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)+1]-=a;
	}
	
	//TODO:Add to marginals
	origin=COORD2(x,y,dims[0],dims[1],2);

	assert (origin < dims[0]*dims[1]*2);
	//*((f64*)&marginals[origin])=0.0; /OPT
	//marginals[origin]=0.0f;
	//marginals[origin+1]=0.0f;
	marginals[origin]=tmp[0];
	marginals[origin+1]=tmp[1];
	for (i=0;i<n_factors*2;i++) {

	  assert(COORD3(x,y,i,dims[0],dims[1],2*n_factors,2)+ 1 < dims[0] * dims[1] * (n_factors*2) *2 && COORD3(x,y,n,dims[0],dims[1],2*n_factors,2) > 0);
	  marginals[origin]+=F_V[COORD3(x,y,i,dims[0],dims[1],2*n_factors,2)]; // invalid read of 4
	  marginals[origin+1]+=F_V[COORD3(x,y,i,dims[0],dims[1],2*n_factors,2)+1];
	  if (isnan(marginals[origin]) && WARN_FLAG) {
	    printf("MARGINAL WARNING %d %d\n",x,y);
	    for (j=0;j<n_factors*2;j++) {
	      printf("%f %f\n",F_V[COORD3(x,y,j,dims[0],dims[1],2*n_factors,2)],F_V[COORD3(x,y,j,dims[0],dims[1],2*n_factors,2)+1]); // invalid read of 4
	    }
	    WARN_FLAG=0;
	  }
	}
	a=fabs(marginals[origin]-mu[origin]);
	max_marg_diff= a > max_marg_diff ? a : max_marg_diff;
	a=fabs(marginals[origin+1]-mu[origin+1]);
	max_marg_diff= a > max_marg_diff ? a : max_marg_diff;

	mu[origin]=marginals[origin];
	mu[origin+1]=marginals[origin+1];
	//TODO: calculate marginal
      }
    }
    if (max_marg_diff < stop_thresh) {
      converged=1;
      break;
    }
  }
  //PyArrayObject* ret= PyArray_SimpleNew(2,dims,NPY_INT32);
  i32 *ret=lpt->EY;
  if (lpt->eval) {
    printf("MARGINALS\n");
    for (x=0;x<dims[0];x++) {
      for (y=0;y<dims[1];y++) {
	origin=COORD2(x,y,dims[0],dims[1],2); // TODO define coord2
	assert(origin >0 && origin + 1 < dims[0]*dims[1]*2);
	//printf("%f ",marginals[origin+1]-marginals[origin]);
	if (marginals[origin] > marginals[origin+1]) {
	  ret[y*dims[0]+x]=0;
	  //*((npy_int32*)PyArray_GETPTR2(ret,x,y))= 0;
	  //ret_data[x*dims[1] + y]=0;
	}
	else{
	  ret[y*dims[0]+x]=1;
	  //*((npy_int32*)PyArray_GETPTR2(ret,x,y))= 1;
	  //ret_data[x*dims[1] + y]=1;
	}
      }
      //printf("\n");
    }
  }
  //PyArray_ENABLEFLAGS((PyArrayObject*)ret, NPY_ARRAY_OWNDATA); //TODO: check if this actually frees memory
  free(com);
  free(rom);
  free(co_pairs);
  _mm_free(marginals);
  _mm_free(RE);
  _mm_free(CE);
  _mm_free(F_V);
  _mm_free(V_F);
  
  return ret;
}

void _loopyCPU__FtoV(){
  /* Compute factor to variable messages */
  i64 i,j;

  for (i=start[0];i<dims[0];i++) {
    for (j=start[1];j<dims[1];j++ ){
      if (i==stop[0] && j==stop[1]) goto loopyFtoVstop;
      	//factor to variable messages
	//Make MS, energies + variable to factor messages..
	//do top and left factors
	origin=COORD3(x,y,0,dims[0],dims[1],2*n_factors,2);

	if (refimg)
	  l= *((i32*) PyArray_GETPTR2(refimg,x,y));
	for (n=0;n<n_factors && l; n+=4) {//Here

	  /*
	    r1: c0_00, c0_01
	        c1_00, c1_01,
		c2_00, c2_01,
		c3_00, c2_01
		
	    r2: c0_10, c0_11,
	        c1_10, c1_11
	        c2_10, c2_11
	        c3_10, c3_11

	  */
	  r1=_mm256_load_ps(&RE[n*2]);
	  r2=_mm256_load_ps(&RE[n_factors*2 + n*2]);
	  r3=_mm256_load_ps(&V_F[origin]);
	  r4=_mm256_permute_ps(r3, 0xA0); //Outcomes that start at 0
					  //for factors 0 and 1 and 2 and 3
	  r1=_mm256_add_ps(r1,r4);
	  
	  r4=_mm256_permute_ps(r3, 0xF5); //Outcomes that start at 1
					  //for factors 0 and 1 and 2 and 3
	  r2=_mm256_add_ps(r2,r4);

	  /* Take the max energy based on which state it came from*/
	  r3=_mm256_max_ps(r1,r2);
	  
	  /*Delegate r3 to appropriate destinations*/
	
	  for (m=n;m<n+4;m++){
	    cop=co_pairs[m];
	    if (x+cop.x <0 || x+cop.x >= dims[0] || y+cop.y < 0 || y+cop.y >=dims[1]) continue;
	    if (refimg && *((i32*)PyArray_GETPTR2(refimg,x+cop.x,y+cop.y))==0) continue;
	    co=origin+com[m] + 2*(m-n) + n_factors*2;
	    if (!(co< 0 || co >= dims[0] * dims[1] * (n_factors*2) *2)) {
	      F_V[co] = r3[2*(m-n)];

	      F_V[co+1] = r3[2*(m-n)+1];
	    }
	    
	  }
	}
	
	//do below and right factors
	origin=COORD3(x,y,0,dims[0],dims[1],2*n_factors,2);
	for (n=n_factors;n<2*n_factors && l; n+=4) {
	  r1=_mm256_load_ps(&CE[(n-n_factors)*2]);
	  r2=_mm256_load_ps(&CE[n_factors*2 + (n-n_factors)*2]);
	  r3=_mm256_load_ps(&V_F[origin]);
	  r4=_mm256_permute_ps(r3, 0xA0);
	  r1=_mm256_add_ps(r1,r4);
	  r4=_mm256_permute_ps(r3, 0xF5);
	  r2=_mm256_add_ps(r2,r4);
	  r3=_mm256_max_ps(r1,r2);
	  //Delegate r3 to appropriate destination
	  for (m=n;m<n+4;m++){ //todo, calculate rom
	    cop=co_pairs[m-n_factors];
	    if (x-cop.x <0 || x-cop.x >= dims[0] || y-cop.y < 0 || y-cop.y >=dims[1]) continue;
	    if (refimg && *((i32*)PyArray_GETPTR2(refimg,x-cop.x,y-cop.y))==0) continue;
	    co=origin+rom[m-n_factors] + 2*(m-n);
	    if (!(origin+rom[m-n_factors] + 2*(m-n) < 0 || origin+rom[m-n_factors] + 2*(m-n) >= dims[0] * dims[1] * (n_factors*2) *2)) {
	      F_V[co] = r3[2*(m-n)];
	      F_V[co+1] = r3[2*(m-n)+1];
	    }
	  }
	}
	//luckily, # factors is guarunteed to be divisible by 4. So no worry about edge cases!
    }


  }

 loopyFtoVstop:
}

void _loopyCPU__VtoF() {
   /* Compute variable to factor messages */ 
}
  
// visible functions

static PyObject* fit (gridCRF_t * self, PyObject *args,PyObject *kwds){

  printf("Start fit\n");
  train_params_t tpt;
  tpt.epochs=100;
  tpt.alpha=0.001f;
  static char * kwlist[] = {"train","lab","epochs","alpha",NULL};
  PyObject *train,*lab,*X,*Y,*f;
  Py_ssize_t n;
  i32 i;
  if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO|if",kwlist,&train,&lab,&(tpt.epochs),&(tpt.alpha))) return NULL;
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
  }
  
  for (i=0;i<n;i++) {
    X=PyList_GetItem(train,i);
    Y=PyList_GetItem(lab,i);

    _train(self,(PyArrayObject*)X,(PyArrayObject*)Y,tpt);
  }
  return Py_BuildValue("");  
}

static PyObject *predict(gridCRF_t* self, PyObject *args, PyObject *kwds){//PyArrayObject *X, PyArrayObject *Y){
  PyObject *test, *out, *refimg=NULL;
  loopy_params_t lpt;
  lpt.stop_thresh=0.01f;
  lpt.max_its=100;
  lpt.eval=1;


  static char * kwlist []= {"X","stop_thresh","max_its","refimg",NULL};
  if (!PyArg_ParseTupleAndKeywords(args,kwds,"O|fiO",kwlist,&test,&(lpt.stop_thresh),&(lpt.max_its),&refimg)) return NULL;
  if (!PyArray_Check(test) || PyArray_NDIM(test) !=3 || PyArray_DIMS(test)[2] !=2) return NULL;
  if (PyArray_TYPE(test) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_TypeError, "Data must be 32-bit floats");
    return NULL;
  }
  lpt.mu= (f32* ) _mm_malloc(PyArray_DIMS(test)[0]*PyArray_DIMS(test)[1]*2*sizeof(f32),32);
  
  out=PyArray_SimpleNew(2, PyArray_DIMS(test), NPY_INT32);
  
  lpt.EY=PyArray_DATA(out);
  
  _loopyCPU(self,test,&lpt,refimg);
  
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
