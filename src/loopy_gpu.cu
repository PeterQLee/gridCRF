extern "C" {
#include "loopy_gpu_cu.h"
}

//#define CPU_TEST

#ifdef CPU_TEST
void __debug_loopy_V_F(loopygpu_t *targs);
void *_loopy_label(loopygpu_t *l_args);
#endif
  
/* 
Optimization strats:
1. Split F_V so that it can take advantage of constant memory for offset coordinates. This can speed up execution possibly.
I.e.
have one block take care of factor index 1 for a portion of variables.

*/

extern "C" i32 *loopyGPU(gridCRF_t* self, PyArrayObject *X_py, gpu_loopy_params_t *lpar, PyArrayObject *refimg){
  npy_intp * dims= PyArray_DIMS(X_py);
  i64 n_factors=self->n_factors;
  i64 max_it=lpar->max_its,it;

  
  f32 *V_data = lpar->V_data;
  i64 n,depth=self->depth,i,j;


  /* transfer matrices */
  loopygpu_t targs;
  // set up threads

  const i32 n_streams = 10;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  /* Allocate message buffers*/
  f32 *F_V = lpar->F_V, *V_F = lpar->V_F;
  n_elem = dims[0] * dims[1] * (n_factors*2) *2;
  dim3 blockGrid(n_elem/128 + 1);
  dim3 threadGrid(128);
  gpu_fill_value<<<blockGrid, threadgrid,0, stream[(curstream++)%n_streams]>>>(F_V, 0.0, n_elem);
  gpu_fill_value<<<blockGrid, threadgrid,0, stream[(curstream++)%n_streams]>>>(V_F, 0.0, n_elem);
  
  /* Allocate coordinate system*/
  i32 *com = lpar->com, *rom = lpar->rom;
  om_pair *co_pairs = lpar->co_pairs;

  f32 *mu = lpar->mu;
  n_elem = dims[0] * dims[1] *2;
  dim3 blockGrid(n_elem/128 + 1);
  dim3 threadGrid(128);
  gpu_fill_value<<<blockGrid, threadgrid, 0, stream[(curstream++)%n_streams]>>>(mu, BIG, n_elem);


  f32 *RE= lpar->RE, *CE= lpar->CE;
  n_elem = n_factors*4;
  dim3 blockGrid(n_elem/128 + 1);
  dim3 threadGrid(128);
  cudaMemcpy(RE, V_data, n_elem * sizeof(f32), cudaDevicetoDevice);
  cudaMemcpy(CE, V_data + n_elem * sizeof(f32), n_elem * sizeof(f32), cudaDevicetoDevice);
  gpu_multiply<<<blockGrid, threadgrid, 0, stream[(curstream++)%n_streams]>>>(RE, n_elem);
  gpu_multiply<<<blockGrid, threadgrid, 0, stream[(curstream++)%n_streams]>>>(CE, n_elem);

  for (i=0;i<n_streams;i++) {
    cudaStreamSynchronize(stream[i]);
    cudaStreamDestroy(stream[i]);
  }
  
  i32 *converged;
  cudaMalloc(&converged,sizeof(i32));
  i32 _converged = 1;
  cudaMemcpy(converged, &_converged, sizeof(i32), cudaHostToDevice);

  targs.com=com;
  targs.rom=rom;
  targs.co_pairs = co_pairs;
  targs.X=X;
  targs.refimg=NULL;
  targs.lpar = lpar;
  targs.self = self;
  targs.F_V = F_V;
  targs.V_F = V_F;
  targs.RE = RE;
  targs.CE = CE;
  targs.mu = mu;
  targs.unary_w=unary_w;
  targs.unary_c=unary_c;
  targs.X_py = X_py;
  targs.converged = converged;
  targs._converged = &_converged;


  for (it = 0; it < max_it; it++){
    if (it%10==0){
      printf("gpu it %d\n", it);
    }
    gpu_loopy_F_V(&targs);
    gpu_loopy_V_F(&targs);
    
    if (_converged) break;
  }
  printf("converged %d %f\n",_converged, lpar->stop_thresh);

  i32 *EY = lpar->EY;

  
  dim3 dimGrid(dims[0],dims[1]);
  dim3 singGrid(2);
  gpu_loopy_V_F__label<<<dimGrid, singGrid, 2*sizeof(f32)>>>(F_V, unary_c, EY, n_factors);

  //cudaMemcpy(lpar->EY, EY, dims[0]*dims[1]*sizeof(i32), cudaMemcpyDeviceToHost);
  
  
  
 cleanup:
  _mm_free(_F_V);
  _mm_free(_V_F);
  free(_com);
  free(_rom);
  free(_co_pairs);
  _mm_free(_RE);
  _mm_free(_CE);

  cudaFree(F_V);
  cudaFree(V_F);
  cudaFree(com);
  cudaFree(rom);
  cudaFree(co_pairs);
  cudaFree(RE);
  cudaFree(CE);
  cudaFree(unary_w);
  cudaFree(unary_c);
  cudaFree(mu);
  cudaFree(X);
  cudaFree(converged);
  cudaFree(EY);

  return lpar->EY;
}


extern "C" void gpu_loopy_F_V(loopygpu_t *targs) { 

  npy_intp * dims= PyArray_DIMS(targs->X_py);
  gridCRF_t *self = targs->self;
  
  i32 n_factors=self->n_factors;
  
  f32 *F_V= targs->F_V;
  f32 *V_F= targs->V_F;

  f32 *RE = targs->RE;
  f32 *CE = targs->CE;

  i32 *com = targs->com;
  i32 *rom = targs->rom;
  om_pair * co_pairs = targs->co_pairs;
  
  i32 i;

  const i32 n_streams = 2;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  

  dim3 dimGrid(dims[0],dims[1]);
  dim3 factorgrid(n_factors);
		  
  gpu_loopy_F_V__Flow<<<dimGrid, factorgrid, 0, stream[0]>>>(F_V, V_F, RE, NULL, rom, co_pairs, n_factors);
  gpu_loopy_F_V__Fup<<<dimGrid, factorgrid, 0, stream[1]>>>(F_V, V_F, CE, NULL, com, co_pairs, n_factors);

  for (i=0;i<2;i++) {
    cudaStreamDestroy(stream[i]);
  }

}


/* Naive method */
__global__ void gpu_loopy_F_V__Flow(f32 *F_V, f32 *V_F, f32 *RE, const i32 * refimg, const i32 * com, const om_pair * co_pairs,  i32 n_factors){
  /* Naive code*/
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 n = threadIdx.x;
  //Note, may need to swap gridDim.x and gridDim.y

  /* Check bounds for upper factor */
  om_pair cop = co_pairs[n];
  if ( ! (x+cop.x <0 || x+cop.x >= gridDim.x || y+cop.y < 0 || y+cop.y >=gridDim.y) ){//&& !(refimg[COORD2(x+cop.x,y+cop.y, gridDim.x, gridDim.y, 1)]==0)) {
    i32 origin=COORD3(x,y,0,gridDim.x,gridDim.y,2*n_factors,2);
    i32 co = origin + com[n] + 2*(n + n_factors);
  
    F_V[co] = RE[n*2] + V_F[origin] > RE[n_factors*2 + n*2] + V_F[origin+1] ? RE[n*2] + V_F[origin] : RE[n_factors*2 + n*2] + V_F[origin+1];
  
    F_V[co+1] = RE[n*2+1] + V_F[origin] > RE[n_factors*2 + n*2 + 1] + V_F[origin+1] ? RE[n*2+1] + V_F[origin] : RE[n_factors*2 + n*2 + 1] + V_F[origin+1];
  }
}

__global__ void gpu_loopy_F_V__Fup(f32 *F_V, f32 *V_F,  f32 *CE, const i32 * refimg, const i32 *rom, const om_pair *co_pairs, i32 n_factors){
  /* Naive code*/
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 n = threadIdx.x;
  

  /* Check bounds for upper factor */
  om_pair cop=co_pairs[n];
  /* Check bounds for lower factor */
  if (!(x-cop.x < 0 || x-cop.x >= gridDim.x || y-cop.y < 0 || y-cop.y >=gridDim.y)){// && !(refimg[COORD2(x-cop.x,y-cop.y, gridDim.x, gridDim.y, 1)]==0)) {
    i32 origin=COORD3(x,y,0,gridDim.x,gridDim.y,2*n_factors,2);
    //i32 co=origin + rom[n]; //check this
    i32 co = origin+rom[n] + 2*n;
    F_V[co] = CE[n*2] + V_F[origin] > CE[n_factors*2 + n*2] + V_F[origin+1] ?
      CE[n*2] + V_F[origin] : CE[n_factors*2 + n*2] + V_F[origin+1];
  
    F_V[co+1] = CE[n*2+1] + V_F[origin] > CE[n_factors*2 + n*2 + 1] + V_F[origin+1] ?
      CE[n*2+1] + V_F[origin] : CE[n_factors*2 + n*2 + 1] + V_F[origin+1];
  }
  
}



extern "C" void gpu_loopy_V_F(loopygpu_t *targs) {


  //TODO:
  // - copy X to gpu

  i32 i;
  gridCRF_t *self = targs->self;
  f32 *X = targs->X;

  loopy_params_t * lpar = targs->lpar;
  
  npy_intp * dims= PyArray_DIMS(targs->X_py);
  i64 n_factors=self->n_factors;
  f32 stop_thresh=lpar->stop_thresh;
  

  f32 * unary_w= targs->unary_w;
  f32 * unary_c = targs->unary_c;

  f32 *F_V = targs->F_V;
  f32 *V_F = targs->V_F;
  f32 *mu = targs->mu;

  /* runtime Flags*/
  i32 *converged = targs->converged;
  
  const i32 n_streams = 1;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  i32 tmp=1;
  cudaMemcpyAsync(converged, &tmp, sizeof(i32),cudaMemcpyHostToDevice,stream[0]);

  for (i=0;i<n_streams;i++) {
    cudaStreamSynchronize(stream[i]);
  }

  
  dim3 dimGrid(dims[0],dims[1]);
  dim3 factorgrid(2*n_factors,2);
  dim3 singGrid(2);
  gpu_loopy_V_F__computeunary<<<dimGrid, singGrid, 0, stream[0]>>>(X, unary_w, unary_c);
  gpu_loopy_V_F__sumfactors<<<dimGrid, factorgrid, sizeof(f32)*n_factors*8, stream[0]>>>(F_V, V_F, unary_c, NULL, n_factors);
  gpu_loopy_V_F__marginal<<<dimGrid, singGrid, 0, stream[0]>>>(F_V, unary_c, mu, n_factors, stop_thresh, converged);

  for (i=0;i<n_streams;i++) {
    cudaStreamDestroy(stream[i]);
  }


  cudaMemcpy(targs->_converged, converged, sizeof(i32), cudaMemcpyDeviceToHost);
 
}

__global__ void gpu_loopy_V_F__computeunary(f32 * X, f32 *unary_w, f32 *unary_c){
    // one possibility. X is not aligned properly.
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 c = threadIdx.x;

  unary_c[COORD2(x,y,gridDim.x, gridDim.y, 2) + c] = -(	    \
    X[COORD2(x,y,gridDim.x, gridDim.y, 2)] * unary_w[c*2] + \
      X[COORD2(x,y,gridDim.x, gridDim.y, 2) + 1] * unary_w[c*2 + 1]);
    
}

__global__ void gpu_loopy_V_F__sumfactors(f32 *F_V, f32 *V_F, f32 *unary_c, const i32 * refimg, i32 n_factors ){
  extern __shared__ char array[];
  f32 *shared_f_v = (f32*) array;
  f32 *shared_v_f = (f32*) (array + sizeof(f32)*n_factors*2*2);

  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 n = threadIdx.x;
  i32 c = threadIdx.y;
  i32 i=0;
  i32 origin = COORD3(x,y,n,gridDim.x, gridDim.y, 2*n_factors, 2) + c;
  // load factor to vvariables into shared memory
  shared_f_v[n*2 + c] = F_V[origin];

  //TODO: make unary a constant?
  f32 sum = unary_c[COORD2(x,y,gridDim.x,gridDim.y,2) + c] - shared_f_v[n*2+c];
  __syncthreads();


  /* Sum up all messages */
  for (i=0;i<2*n_factors;i++) {
//printf("KEK %d\n", i*2+c);
    sum += shared_f_v[i*2+c]; //this is the problem!
  }
  shared_v_f[n*2+c]=sum;
  
  __syncthreads();  
  // Normalize values
  sum = sum - 0.5 * (sum+shared_v_f[n*2+c^1]);
  V_F[origin]=sum;
  
}

/* TODO: finish */
__global__ void gpu_loopy_V_F__marginal(f32 *F_V, f32 * unary_c,  f32 * mu, i32 n_factors, f32 stop_thresh, i32 *converged) {
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 c = threadIdx.x;

  i32 i;
  i32 origin = COORD2(x,y,gridDim.x, gridDim.y, 2) + c;
  f32 sum = unary_c[origin];

  // sum up factors
  for (i=0;i<n_factors*2;i++) {
    sum += F_V[COORD3(x,y,i,gridDim.x, gridDim.y, 2*n_factors, 2) + c];
  }

  if (fabsf(sum - mu[origin]) > stop_thresh) {
    converged[0] = 0;
  }
  mu[origin] = sum;
}

//TODO: change to mu instead
__global__ void gpu_loopy_V_F__label(f32 *F_V, f32 * unary_c, i32 *EY, i32 n_factors) {

  /* Computes the predicted label given the values */
  extern __shared__ char array[];
  f32 *shared_marginal = (f32*) array;
  i32 i;
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 c = threadIdx.x;
  i32 origin = COORD2(x,y,gridDim.x, gridDim.y, 2) + c;

  f32 sum = unary_c[origin];
  // sum up factors
  for (i=0;i<n_factors*2;i++) {
    sum += F_V[COORD3(x,y,i,gridDim.x, gridDim.y, 2*n_factors, 2) + c];
  }
  shared_marginal[c] = sum;
  __syncthreads();
  if (c==0  && shared_marginal[0] > shared_marginal[1]) {
    EY[COORD2(x,y,gridDim.x, gridDim.y, 1)] = 0;
  }
  else if (c==0) {
    EY[COORD2(x,y,gridDim.x, gridDim.y, 1)] = 1;
  }
  
}


__global__ void gpu_fill_value(f32 *buffer, f32 val, i32 lim) {
  i32 index = blockIdx * blockDim.x + threadIdx.x;
  if (index < lim) {buffer[index] = val;}
}

__global__ void gpu_multiply(f32 *buffer, i32 lim) {
   i32 index = blockIdx * blockDim.x + threadIdx.x;
   if (index < lim) {buffer[index] = -buffer[index];}
}


#ifdef CPU_TEST



void __debug_loopy_V_F(loopygpu_t *targs) {

  /* Compute factor to variable messages */
  i32 WARN_FLAG=1;
  i32 i,j;
  gridCRF_t *self = targs->self;
  PyArrayObject *X = targs->X_py;
  loopy_params_t * lpar = targs->lpar;
  
  f32 a,b;
  npy_intp * dims= PyArray_DIMS(X);
  i64 n_factors=self->n_factors;
  f32 stop_thresh=lpar->stop_thresh;
  
  f32 * unary = self->unary;
  
  npy_intp x,y;
  i32 n;


  f32 *F_V = targs->_F_V;
  f32 *V_F = targs->_V_F;

  f32 *gF_V = targs->F_V;
  f32 *gV_F = targs->V_F;

  cudaMemcpy(F_V, gF_V, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyDeviceToHost);
  cudaMemcpy(V_F, gV_F, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyDeviceToHost);
  
  
  f32 *marginals = (f32*) _mm_malloc(sizeof(f32)*dims[0]*dims[1]*2,32);
  f32 *mu = targs->_mu;

  /* coordinates */

  /* runtime Flags*/
  i32 *converged = targs->_converged;
  
  i32 origin;

  f32 tmp[2];

  __m256 r1,r2;
  /* Compute variable to factor messages */
  for (x=0;x<dims[0];x++) {
    for (y=0;y<dims[1];y++) {
      //variable to factor messages
      
      f64 base= *((f64*)PyArray_GETPTR3(X,x,y,0));
      *((f64*)tmp) = base;
      tmp[0]=-(((f32*)&base)[0]*unary[0] + ((f32*)&base)[1]*unary[1]);
      tmp[1]=-(((f32*)&base)[0]*unary[2] + ((f32*)&base)[1]*unary[3]);
      r1=(__m256)_mm256_set1_pd(*((f64*)tmp)); //set all elements in vector this thi
      //Warning: possible segfault
	
      for (n=0;n<n_factors*2;n+=4) { //Set baseline, since we know that unary is added to each V_F
	_mm256_store_ps(&V_F[COORD3(x,y,n,dims[0],dims[1],2*n_factors,2)] ,r1);
      }
      
      for (i=0;i<n_factors*2;i++) {
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
      if (a > stop_thresh) {
	*converged = 0;
      }
      a=fabs(marginals[origin+1]-mu[origin+1]);
      if (a > stop_thresh) {
	*converged = 0;
      }

      mu[origin]=marginals[origin];
      mu[origin+1]=marginals[origin+1];
      //TODO: calculate marginal
    }
  }

  _mm_free(marginals);
  cudaMemcpy(gF_V, F_V, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyHostToDevice);
  cudaMemcpy(gV_F, V_F, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyHostToDevice);
}


void *_loopy_label(loopygpu_t *l_args) {
  loopy_params_t * lpar = l_args->lpar;
  npy_intp * dims= PyArray_DIMS(l_args->X_py);
  i32 *ret=lpar->EY;
  f32 *mu=l_args->_mu;
  i32 x,y;
  i32 origin;
  for (x=0;x<dims[0];x++) {
    for (y=0;y<dims[1];y++) {
   
      origin=COORD2(x,y,dims[0],dims[1],2); 
      assert(origin >= 0 && origin + 1 < dims[0]*dims[1]*2);
      if (mu[origin] > mu[origin+1]) {
	ret[COORD2(x,y,dims[0],dims[1],1)]=0;
      }
      else{
	ret[COORD2(x,y,dims[0],dims[1],1)]=1;

      }
    }

  }

  //cudaMemcpy(ret, gEY, dims[0] * dims[1] * sizeof(i32), cudaMemcpyHostToDevice);

  return NULL;

}
#endif
