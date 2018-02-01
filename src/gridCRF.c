#include "gridCRF.h"
#include <assert.h>
#include <stdio.h>
f32 ALPHA=0.001;
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
  self->V_data = _mm_malloc((n_factors * 4 ) * sizeof(f32),32); //TODO: use aligned malloc
  
  for (i=0;i<n_factors*4;i++) {
    self->V_data[i]=1.0f; //temporary
  }
  
  npy_intp dims[2]= {n_factors, 4};
  self->V=(PyArrayObject *)PyArray_SimpleNewFromData(2,dims,NPY_FLOAT32,self->V_data);
  Py_INCREF(self->V);

  

  return 0;
}
static PyObject * gridCRF_new (PyTypeObject *type, PyObject *args, PyObject *kwds){
  gridCRF_t *self;
  self=(gridCRF_t*)type->tp_alloc(type,0);
  return (PyObject *)self;
}

//backend functions

static void _train( gridCRF_t * self, PyArrayObject *X, PyArrayObject *Y, train_params_t tpt){
 
  npy_intp * dims= PyArray_DIMS(X);
  i64 depth= self->depth;
  i32 i,j,n,a,b,d,k;
  i32 its, epochs=tpt.epochs;//epochs=1000;
  i32 *l;
  i64 n_factors=self->n_factors;
  f32 *V = self->V_data;
  n=self->depth;
  //f64 *stack[self->n_factors ]; //TODO, preallocate this on heap.
  f64 *tmp;
  f32 *p,*v;
 
  f32 yv[2], change[2];
  f32 max,den;
  i32 *ainc = malloc(sizeof(i32)*n_factors);
  i32 *binc = malloc(sizeof(i32)*n_factors);

  f32 alpha=tpt.alpha;
  
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
      n++;

    }
  }
  for (its=0;its<epochs;its++) {
    n=0;
    for (i=0;i<dims[0];i++) {
      for (j=0;j<dims[1];j++) {
	*((f64 *)yv)=0.0; // set outome to 0
	if (((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && ((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      //do traversal.
      //printf("ij %d %d\n",i,j);
      
	tmp=(f64*)PyArray_GETPTR3(X,i,j,0);
	f32 vcum_0=0.0, vcum_1=0.0;
	f32 normfact=alpha/1000.0;
	for (n=0;n<n_factors;n++) {
	  if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  //if (b==0 && a==n) break; //TODO: change to proper ordering(like in notes0
	  l=(i32*) PyArray_GETPTR3(Y,i+ainc[n],j+binc[n],0);
	  v=&V[n*4 + ((*l)^1)*2]; // 4x4 transfer matrix for n	
	  //We pick the row that corresponds to the outcome
	  //TODO: Improve with SSE
	  yv[0]+= v[0]*((f32*)tmp)[0];
	  yv[1]+= v[1]*((f32*)tmp)[1];
	  vcum_0+=2*normfact*(v[0]);
	  vcum_0+=2*normfact*(v[1]);
	}
    
	//softmax
	max=yv[0]>yv[1]? yv[0]:yv[1];
    
	yv[0]=exp(yv[0]-max);
	yv[1]=exp(yv[1]-max);
	den=1/(yv[0]+yv[1]);
      
	//TODO: speed up with SSE ops
	yv[0]=yv[0]*den;
	yv[1]=yv[1]*den;
      
	//TODO: speed up with SSE ops
	l=((i32*)PyArray_GETPTR3(Y,i,j,0));
	p=(f32*)PyArray_GETPTR3(X,i,j,0);
	change[0] = alpha * (((*l)&1)-yv[0]) * (*p) ;
	l=((i32*)PyArray_GETPTR3(Y,i,j,1));
	p=(f32*)PyArray_GETPTR3(X,i,j,1);
	change[1] = alpha * (((*l)&1)-yv[1]) * (*p); 
      
	//update all of the pointers we visited with change.
	//TODO: speed up with SSE, multithreading, or GPU impl.

	for (n=0;n<n_factors;n++){
	  if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  //TODO: speed up (SSE) __m128 _mm_add_ps
	  l=((i32*) PyArray_GETPTR3(Y,i+ainc[n],j+binc[n],0));
	  V[n*4 + 2*((*l)^1)] += change[0];
	  V[n*4 + 2*((*l)^1) + 1] += change[1];

	}

      }
    }
  }
    
  //Values trained now
  printf("Done train");
  free(ainc);
  free(binc);

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
    r1=exp256_ps(r1);
    //assert (!(isnan(r1[6]) || isnan(r1[7])));
    RE[i/2]=r1[0];
    RE[i/2+1]=r1[1];
    RE[n_factors*2+i/2]=r1[2];
    RE[n_factors*2+i/2+1]=r1[3];
    RE[i/2+2]=r1[4];
    RE[i/2+3]=r1[5];
    RE[n_factors*2+i/2+2]=r1[6];
    RE[n_factors*2+i/2+3]=r1[7];
    //printf("SET %ld %ld %ld %ld %ld %ld %ld %ld\n", i/2, i/2+1, i/2+n_factors*2, i/2+n_factors*2+1 , i/2+2, i/2+3 , i/2+2+n_factors*2,i/2+3+n_factors*2);

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

  }
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
	r1=(__m256)_mm256_set1_pd(base); //set all elements in vector this thi
	//Warning: possible segfault
	
	for (n=0;n<n_factors*2;n+=4) {
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
	marginals[origin]=0.0f;
	marginals[origin+1]=0.0f;
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
