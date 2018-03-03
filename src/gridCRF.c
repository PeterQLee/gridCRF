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
  const f32 lr=0.0001;
  
  //for (i=0;i<n_factors*4*2;i++){
  //  V[i]+=0.1*x[i];
  //}
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
  f32 L=_calculate_gradient(args);
  printf("g");
  for (i=0;i<num_params;i++) {
    g[i]=V_change[i]; //this works because V_change and unary change are contitigous

  }
  printf("%f %f %f %f", g[n_factors*4*2],g[n_factors*4*2+1],g[n_factors*4*2+2],g[n_factors*4*2+3]);  
  printf("\nError %f %d\n",L,count++);
  return L;
}

static f32 _calculate_gradient(gradient_t *args) {
  i32 i,j;
  PyArrayObject *X=args->X;
  PyArrayObject *Y=args->Y;
  printf("Y %x\n",Y);
  f32 *unary= args->self->unary;
  i32 *ainc=args->ainc, *binc=args->binc;
  f32 * V=args->self->V_data;
  
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
  for (i=0;i<dims[0];i++) {
    for (j=0;j<dims[1];j++) {
      *((f64 *)yv)=0.0; // set outome to 0
      
      if (*((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && *((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      
      tmp=(f32*)PyArray_GETPTR3(X,i,j,0);
      yv[0]=-(unary[0]*tmp[0]+unary[1]*tmp[1]);
      yv[1]=-(unary[2]*tmp[0]+unary[3]*tmp[1]);


      for (n=0;n<n_factors;n++) {
	if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  
	l=(i32*) PyArray_GETPTR3(Y,i+ainc[n],j+binc[n],0);
	v=&V[n*4 + ((*l)^1)*2]; // 4x4 transfer matrix for n	
	//We pick the row that corresponds to the outcome
	//TODO: Improve with SSE
	  
	yv[0] += v[0]; //(Dependent on function)
	yv[1] += v[1];
      }
      //do left to right
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	l=(i32*) PyArray_GETPTR3(Y,i+ainc[n+n_factors],j+binc[n+n_factors],0);
	v=&V[n_factors*4 + n*4 + 2*((*l)^1)]; 
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
	l=((i32*) PyArray_GETPTR3(Y,i+ainc[n],j+binc[n],0));
	V_change[n*4 + 2*((*l)^1)] += change[0];
	V_change[n*4 + 2*((*l)^1) + 1] += change[1];
      }
      
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	l=(i32*) PyArray_GETPTR3(Y,i+ainc[n+n_factors],j+binc[n+n_factors],0);
	V_change[n_factors*4 +n*4 + 2*((*l)^1)] += change[0];
	V_change[n_factors*4 +n*4 + 2*((*l)^1) + 1] += change[1];	  
      }
    }
  }
  printf("L %f\n",L);
  return L;
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
  for (i_=0;i_<dims[0];i_++) {
    for (j_=0;j_<dims[1];j_++) {
      i=i_inds[i_];
      j=j_inds[j_];
      *((f64 *)yv)=0.0; // set outome to 0
      
      if (*((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && *((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      
      tmp=(f32*)PyArray_GETPTR3(X,i,j,0);
      yv[0]=-(unary[0]*tmp[0]+unary[1]*tmp[1]);
      yv[1]=-(unary[2]*tmp[0]+unary[3]*tmp[1]);


      for (n=0;n<n_factors;n++) {
	if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  
	l=(i32*) PyArray_GETPTR3(Y,i+ainc[n],j+binc[n],0);
	v=&V[n*4 + ((*l)^1)*2]; // 4x4 transfer matrix for n	
	//We pick the row that corresponds to the outcome
	//TODO: Improve with SSE
	  
	yv[0] += v[0]; //(Dependent on function)
	yv[1] += v[1];
      }
      //do left to right
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	l=(i32*) PyArray_GETPTR3(Y,i+ainc[n+n_factors],j+binc[n+n_factors],0);
	v=&V[n_factors*4 + n*4 + 2*((*l)^1)]; 
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
      
      //printf("Part L %f %f %f %d %d\n",yv[0],yv[1],L ,  *((i32*)PyArray_GETPTR3(Y,i,j,0)),*l);
      
      if (isinf(L)){
	printf("ISINF\n");
	//exit(1);
      }
      /*
      for (n=0;n<n_factors;n++){
	if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	//TODO: speed up (SSE) __m128 _mm_add_ps
	l=((i32*) PyArray_GETPTR3(Y,i+ainc[n],j+binc[n],0));
	V[n*4 + 2*((*l)^1)] += change[0];
	V[n*4 + 2*((*l)^1) + 1] += change[1];
      }
      
      for (n=0;n<n_factors;n++) {
	if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	l=(i32*) PyArray_GETPTR3(Y,i+ainc[n+n_factors],j+binc[n+n_factors],0);
	V[n_factors*4 +n*4 + 2*((*l)^1)] += change[0];
	V[n_factors*4 +n*4 + 2*((*l)^1) + 1] += change[1];	  
      }
      */
    }
  }
  printf("L %f\n",L);
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
  #define LBFGS 0
  #if LBFGS
  i32 ret= lbfgs(num_params,x,&fx, _lbfgs_update, NULL, (void *)&gradargs, &_defparam);
  printf("orth %d %d\n",_defparam.orthantwise_start,epochs); //Invalid orthantwise_start
  printf("ret %d %d\n",ret,LBFGSERR_INVALID_ORTHANTWISE_START); //Invalid orthantwise_start
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
  lbfgs_free(x);
  
}

static PyArrayObject* _loopyCPU(gridCRF_t* self, PyArrayObject *X,loopy_params_t lpt,PyArrayObject *refimg){
  i32 WARN_FLAG=1;
  //loopy belief propagation using CPU

  //Need to initialize all messages
  f32 a,b,c,d,denr,denc,maxv;
  npy_intp * dims= PyArray_DIMS(X);
  i64 n_factors=self->n_factors;
  i64 max_it=lpt.max_its,it;
  f32 max_marg_diff=0.0f;
  f32 stop_thresh=lpt.stop_thresh;
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
  //printf ("%lx\n",F_V);
  //*((i64*)x) = 0xfeeddad;
  f32 * marginals= (f32* ) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
  f32 * mu= (f32* ) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
  for (i=0;i<dims[0]*dims[1]*2;i+=1) {
    mu[i]=BIG;
    marginals[i]=0.0f;
  }
  /*for (i=0;i<dims[0]*dims[1]*2;i+=2) {
    *((f64*)&mu[i])=BIG;
    *((f64*)&marginals[i])=0.0;
    }*/

   //f32 * outgoing= (f32*) _mm_malloc(sizeof(f32) *  self->n_factors * 2);
  //i32 rowv_[8]={0,2,4,6,8,10,12,14};
  //const i32 roffset =1 , coffset=2;
  //i32 colv_[8]={0,1,4,5,8,9,12,13};
  //  i32 control_d[8]={0,2,1,3,0,2,1,3};
  //__m256i *rowv=(__m256i *)rowv_;
  //__m256i *colv=(__m256i *)colv_;
  //__m256i control= _mm256_load_si256(&control_d);
  
  __m256 r1,r2,r3,r4;
  
  //f32 * ms = malloc(sizeof(f32) * self->n_factors*4);
  // direction, x,y, factor, prob
  
  //i32 com[n_factors]; //TODO: preallocate
  //i32 rom[n_factors]; //Indices of destinations from factor
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
  /* printf("n_factors %d\n",n_factors); */
  /* printf("com\n"); */
  /* //  for (j=1;j<=depth;j++ ) { */
  /* for (j=0;j<n_factors;j++ ) { */
  /*   //printf("%d %d %d %d\n",com[n],com[n+1],com[n+2],com[n+3]); */
  /*   printf("%d\n",com[j]); */
  /*   //n+=4; */
  /* } */
  /* printf("\nrom\n"); */
  /* for (j=0;j<n_factors;j++ ) { */
  /*   printf("%d\n",rom[j]); */
  /* } */
  /* printf("\ncop\n"); */
  /* for (j=0;j<n_factors;j++ ) { */
  /*   printf("%d %d\n",co_pairs[j].x,co_pairs[j].y); */
  /* } */
  

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
    
    /*
    CE[i/2]=-r1[0];//New
    CE[i/2+1]=-r1[2];
    CE[n_factors*2+i/2]=-r1[1];
    CE[n_factors*2+i/2+1]=-r1[3];
    CE[i/2+2]=-r1[4];
    CE[i/2+3]=-r1[6];
    CE[n_factors*2+i/2+2]=-r1[5];
    CE[n_factors*2+i/2+3]=-r1[7];*/

    printf("SET %ld %ld %ld %ld %ld %ld %ld %ld\n", i/2, i/2+1, i/2+n_factors*2, i/2+n_factors*2+1 , i/2+2, i/2+3 , i/2+2+n_factors*2,i/2+3+n_factors*2);
    printf("VAL %f %f %f %f %f %f %f %f\n", RE[i/2], RE[i/2+1], RE[i/2+n_factors*2], RE[i/2+n_factors*2+1] , RE[i/2+2], RE[i/2+3] , RE[i/2+2+n_factors*2],RE[i/2+3+n_factors*2]);
    /*mvtemp=((__m256d)r1)[0];
    *((f64*)&RE[i/2]) = mvtemp;
    mvtemp=((__m256d)r1)[1];
    *((f64*)&RE[i/2 + n_factors]) = mvtemp;
    mvtemp=((__m256d)r1)[2];
    *((f64*)&RE[i/2 + 2]) = mvtemp;
    mvtemp=((__m256d)r1)[3];
    *((f64*)&RE[i/2 + 2 + n_factors]) = mvtemp;*/
    //printf("SET %d %d %d %d\n", i/2, i/2+n_factors, i/2+2, i/2+2+n_factors);

  }

  //Softmax Rows and columns.
  //Could do SSE on this, but low priority because it is a one time op
  /*
  for (i=0;i<n_factors;i++ ) {
    //printf("I %d %d %d %d\n",i*2,i*2+1, n_factors*2+2*i, n_factors*2+2*i+1);
    a=RE[i*2];
    b=RE[i*2+1];
    c=RE[n_factors*2+2*i];
    d=RE[n_factors*2+2*i+1];
    maxv=a>b?a:b;//row
    if (maxv==0 || a+b==0) printf("ALERT");
    maxv=1/maxv;
    denr=1/((a+b)*maxv);
    RE[i*2]= (a*maxv)*denr;
    RE[i*2+1] = (b*maxv) *denr;
    maxv=c>d?c:d;
    if (maxv==0 || c+d ==0 ) printf("ALERT");
    maxv=1/maxv;
    //maxv=0;
    denr=1/((c+d)*maxv);
    RE[i*2 + n_factors*2]= (c*maxv)*denr;
    RE[i*2+1 + n_factors*2] = (d*maxv)*denr;

    maxv=a>c?a:c;//col
    if (maxv==0 || a+c==0) printf("ALERT");
    maxv=1/maxv;
    denr=1/((a+c)*maxv);
    CE[i*2]= (a*maxv)*denr;
    CE[i*2 + 1] = (c*maxv)*denr;
    maxv=b>d?b:d; //conditional jump
    if (maxv==0 || b+d ==0 ) printf("ALERT");
    maxv=1/maxv;
    denr=1/((b+d)*maxv);
    CE[i*2 + 2*n_factors]= (b*maxv)*denr;
    CE[i*2 + 1 + 2*n_factors] = (d*maxv)*denr;

    }*/
  
  /*
  printf("RE\n");
  for (i=0;i<n_factors;i++ ) {
    //printf("I %d %d %d %d\n",i*2,i*2+1, n_factors*2+2*i, n_factors*2+2*i+1);
    a=RE[i*2];
    b=RE[i*2+1];
    c=RE[n_factors*2+2*i];
    d=RE[n_factors*2+2*i+1];
    printf("%f %f %f %f\n",a,b,c,d);
  }
    printf("CE\n");
  for (i=0;i<n_factors;i++ ) {
    //printf("I %d %d %d %d\n",i*2,i*2+1, n_factors*2+2*i, n_factors*2+2*i+1);
    a=CE[i*2];
    b=CE[i*2+1];
    c=CE[n_factors*2+2*i];
    d=CE[n_factors*2+2*i+1];
    printf("%f %f %f %f\n",a,b,c,d);
    }*/

  for (it=0;it< max_it && !converged;it++){
    printf("it %d\n",it);
    /*
    printf("V_F\n");
    for (n=0;n<dims[0] * dims[1] * (n_factors);n++){
      printf("%f %f %f %f\n",V_F[n*4],V_F[n*4+1],V_F[n*4+2],V_F[n*4+3]);
    }
    printf("\nF_V\n");
    for (n=0;n<dims[0] * dims[1] * (n_factors);n++){
      printf("%f %f %f %f\n",F_V[n*4],F_V[n*4+1],F_V[n*4+2],F_V[n*4+3]);
    }
  
    for (n=0;n<dims[0]*dims[1];n++) {
      printf("%f %f\n",marginals[n*2],marginals[n*2+1]);
    }
    printf("\n");
    for (n=0;n<dims[0]*dims[1];n++) {
      printf("%f %f\n",mu[n*2],mu[n*2+1]);
    }
    printf("\n");
  */
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
	  r1=_mm256_load_ps(&RE[n*2]);
	  r2=_mm256_load_ps(&RE[n_factors*2 + n*2]);
	  r3=_mm256_load_ps(&V_F[origin]);
	  r4=_mm256_permute_ps(r3, 0xA0);
	  r1=_mm256_add_ps(r1,r4);
	  r4=_mm256_permute_ps(r3, 0xF5);
	  r2=_mm256_add_ps(r2,r4);
	  r3=_mm256_max_ps(r1,r2);
	  //Delegate r3 to appropriate destinations
	
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
	for (n=0;n<n_factors*2;n+=8) { //correct double counting
	  r1=_mm256_load_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)]); //TODO: test permuting
	  r2=_mm256_load_ps(&V_F[COORD3(x,y,n+4,dims[0],dims[1],2*n_factors,2)]);
	  r1=_mm256_permutevar_ps(r1,control);
	  r2=_mm256_permutevar_ps(r2,control);
	  r3=_mm256_unpacklo_ps(r1,r2); // 0 column
	  r4=_mm256_unpackhi_ps(r1,r2); // 0 column
	  r3=(__m256) _mm256_permute4x64_pd(r3, 0xD8);  //tested
	  //store r3 into a buffer
	  _mm256_store_ps(&deltas[n], r3);
	}
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
  i32 * ret_data = malloc(dims[0]*dims[1]*sizeof(i32) );
  //PyArray* ret = PyArray_New(subtype,2,dims,NPY_INT32,NULL,ret_data,itemsize,NPY_ARRAY_OWNDATA,obj);

  //Py_INCREF(ret);
  //Give argmax residuals in ret
  PyArrayObject* ret= PyArray_SimpleNew(2,dims,NPY_INT32);
  printf("MARGINALS\n");
  for (x=0;x<dims[0];x++) {
    for (y=0;y<dims[1];y++) {
      origin=COORD2(x,y,dims[0],dims[1],2); // TODO define coord2
      assert(origin >0 && origin + 1 < dims[0]*dims[1]*2);
      //printf("%f ",marginals[origin+1]-marginals[origin]);
      if (marginals[origin] > marginals[origin+1]) {
	*((npy_int32*)PyArray_GETPTR2(ret,x,y))= 0;
	//ret_data[x*dims[1] + y]=0;
      }
      else{
	*((npy_int32*)PyArray_GETPTR2(ret,x,y))= 1;
	//ret_data[x*dims[1] + y]=1;
      }
    }
    //printf("\n");
  }

  //PyArray_ENABLEFLAGS((PyArrayObject*)ret, NPY_ARRAY_OWNDATA); //TODO: check if this actually frees memory

  return ret;
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
  static char * kwlist []= {"X","stop_thresh","max_its","refimg",NULL};
  if (!PyArg_ParseTupleAndKeywords(args,kwds,"O|fiO",kwlist,&test,&(lpt.stop_thresh),&(lpt.max_its),&refimg)) return NULL;
  if (!PyArray_Check(test) || PyArray_NDIM(test) !=3 || PyArray_DIMS(test)[2] !=2) return NULL;
  if (PyArray_TYPE(test) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_TypeError, "Data must be 32-bit floats");
    return NULL;
  }
  out=_loopyCPU(self,test,lpt,refimg);
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
