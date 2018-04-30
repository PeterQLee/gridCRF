extern "C" {
#include "loopy_gpu_cu.h"
}
#define CPU_TEST
/* 
Optimization strats:
1. Split F_V so that it can take advantage of constant memory for offset coordinates. This can speed up execution possibly.
I.e.
have one block take care of factor index 1 for a portion of variables.

*/

extern "C" i32 *loopyGPU(gridCRF_t* self, PyArrayObject *X_py,loopy_params_t *lpar,PyArrayObject *refimg){
  npy_intp * dims= PyArray_DIMS(X_py);
  i64 n_factors=self->n_factors;
  i64 max_it=lpar->max_its,it;

  f32 * V_data=self->V_data;
  i64 n,depth=self->depth,i,j;

  
  f32 * _F_V = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32);
  f32 * _V_F = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32);
  
  for (i=0;i<dims[0] * dims[1] * (n_factors*2) *2; i++){
    _F_V[i]=0.0f;
    _V_F[i]=0.0f;
  }


  f32 *_mu = lpar->mu;
  f32 * _unary_w = self->unary;
  
  for (i=0;i<dims[0]*dims[1]*2;i+=1) {
    _mu[i]=BIG;

  }
  

  /* Prepare coordinates*/
  i32 *_com=(i32*)malloc(sizeof(i32)*n_factors);
  i32 *_rom=(i32*)malloc(sizeof(i32)*n_factors);
  om_pair *_co_pairs=(om_pair*)malloc(sizeof(om_pair)*n_factors);

  n=0;
  for (j=1;j<=depth;j++ ) {
    for (i=0;i<j*4;i++) {
      if (i<j) {
	_com[n]= -j *dims[1] * n_factors*2*2 - i*n_factors*2*2;
	_rom[n]= +j *dims[1] * n_factors*2*2 + i*n_factors*2*2;
	_co_pairs[n]=(om_pair){-j,-i};
      }
      else if (i>=j*3) {
	_com[n]= +j *dims[1] * n_factors*2*2 - (j-(i-j*3))*n_factors*2*2;
	_rom[n]= -j *dims[1] * n_factors*2*2 + (j-(i-j*3))*n_factors*2*2;
	_co_pairs[n]=(om_pair){j,-(j-(i-j*3))};
      }
      else{
	_com[n]= (-2*j+i)*dims[1] * n_factors*2*2 - j*n_factors*2*2;
	_rom[n]= (2*j-i)*dims[1] * n_factors*2*2 + j*n_factors*2*2;
	_co_pairs[n]=(om_pair){-2*j+i,-j};
      }
      
      n++;
    }
  }
  
  n=0;


  /* transfer matrices */
  f32 *_RE= (f32 *) _mm_malloc(2 * n_factors * 2 * sizeof(f32),32); 
  f32 *_CE= (f32 *) _mm_malloc(2 * n_factors * 2 * sizeof(f32),32); 
  __m256 r1,r2;


  for (i=0;i<2*n_factors*2;i+=8) {
    r1=_mm256_load_ps(&V_data[i]);
    r2=_mm256_load_ps(&V_data[i + n_factors*4]);
    //r1=exp256_ps(r1);
    //assert (!(isnan(r1[6]) || isnan(r1[7])));

    /*Swap energies such that remote outcome=1 is seperated from
      remote outcome=0*/
    _RE[i/2]=-r1[0];
    _RE[i/2+1]=-r1[1];
    _RE[n_factors*2+i/2]=-r1[2];
    _RE[n_factors*2+i/2+1]=-r1[3];
    _RE[i/2+2]=-r1[4];
    _RE[i/2+3]=-r1[5];
    _RE[n_factors*2+i/2+2]=-r1[6];
    _RE[n_factors*2+i/2+3]=-r1[7];
    
    
    _CE[i/2]=-r2[0];
    _CE[i/2+1]=-r2[1];
    _CE[n_factors*2+i/2]=-r2[2];
    _CE[n_factors*2+i/2+1]=-r2[3];
    _CE[i/2+2]=-r2[4];
    _CE[i/2+3]=-r2[5];
    _CE[n_factors*2+i/2+2]=-r2[6];
    _CE[n_factors*2+i/2+3]=-r2[7];
  
  }
 
  loopygpu_t targs;
  // set up threads

  const i32 n_streams = 10;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  /* Allocate message buffers*/
  f32 *F_V, *V_F;
  cudaMalloc(&F_V, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32));
  cudaMalloc(&V_F, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32));

  cudaMemcpyAsync(_F_V, F_V, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(_V_F, V_F, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyHostToDevice, stream[1]);

  /* Allocate coordinate system*/
  i32 *com, *rom;
  om_pair *co_pairs;
  cudaMalloc(&com, sizeof(i32)*n_factors);
  cudaMalloc(&rom, sizeof(i32)*n_factors);
  cudaMalloc(&co_pairs,sizeof(om_pair)*n_factors);

  cudaMemcpyAsync(_com, com, n_factors* sizeof(i32), cudaMemcpyHostToDevice, stream[2]);
  cudaMemcpyAsync(_rom, rom, n_factors* sizeof(i32), cudaMemcpyHostToDevice, stream[3]);
  cudaMemcpyAsync(_co_pairs, co_pairs, n_factors* sizeof(om_pair), cudaMemcpyHostToDevice, stream[4]);

  f32 *RE, *CE;
  cudaMalloc(&RE, 2* n_factors *2* sizeof(f32));
  cudaMalloc(&CE, 2* n_factors *2* sizeof(f32));
  cudaMemcpyAsync(_RE, RE, 2*n_factors*2*sizeof(f32), cudaMemcpyHostToDevice, stream[5]);
  cudaMemcpyAsync(_CE, CE, 2*n_factors*2*sizeof(f32), cudaMemcpyHostToDevice, stream[6]);

  f32 *unary_w, *unary_c;
  cudaMalloc(&unary_w, 4 * sizeof(f32));
  cudaMemcpyAsync(_unary_w, unary_w, 4*sizeof(f32), cudaMemcpyHostToDevice, stream[7]);
    
  cudaMalloc(&unary_c, dims[0]*dims[1]*2*sizeof(f32));

  
  f32 *mu;
  cudaMalloc(&mu, dims[0]*dims[1]*2*sizeof(f32));
  cudaMemcpyAsync(_mu, mu, dims[0]*dims[1]*2*sizeof(f32), cudaMemcpyHostToDevice, stream[8]);

  f32 *X;
  cudaMalloc(&X, dims[0]*dims[1]*2*sizeof(f32));//tmp
  cudaMemcpyAsync(X_py->data, X, dims[0]*dims[1]*2*sizeof(f32), cudaMemcpyHostToDevice, stream[9]);

  for (i=0;i<n_streams;i++) {
    cudaStreamDestroy(stream[i]);
  }
  
  i32 *converged;
  cudaMalloc(&converged,sizeof(i32));
  i32 _converged = 1;

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

  for (it = 0; it < max_it; i++){
    if (it%1==0){
      printf("gpu it %d\n", it);
    }
    gpu_loopy_F_V(&targs);
    gpu_loopy_V_F(&targs);
    if (_converged) break;
  }
  printf("converged %d %f\n",_converged, lpar->stop_thresh);

  i32 *EY;
  cudaMalloc(&EY, dims[0]*dims[1]*sizeof(f32));
  
  dim3 dimGrid(dims[0],dims[1]);
  dim3 singGrid(2);
  gpu_loopy_V_F__label<<<dimGrid, singGrid, 2*sizeof(f32)>>>(F_V, unary_c, EY, n_factors);

  cudaMemcpy(lpar->EY, EY, dims[0]*dims[1]*sizeof(f32), cudaMemcpyDeviceToHost);

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
  cudaMemcpyAsync(&tmp,converged, sizeof(i32),cudaMemcpyHostToDevice,stream[0]);

  for (i=0;i<n_streams;i++) {
    cudaStreamSynchronize(stream[i]);
  }
  #define DEBUG
  #ifdef DEBUG
  f32 *oldmu = (f32*) malloc(sizeof(f32)*dims[0]*dims[1]*2);
  f32 *newmu = (f32*) malloc(sizeof(f32)*dims[0]*dims[1]*2);
  cudaMemcpy(oldmu, mu, sizeof(f32)*dims[0]*dims[1]*2, cudaMemcpyDeviceToHost);
  #endif
  
  dim3 dimGrid(dims[0],dims[1]);
  dim3 factorgrid(2*n_factors,2);
  dim3 singGrid(2);
  gpu_loopy_V_F__computeunary<<<dimGrid, singGrid, 0, stream[0]>>>(X, unary_w, unary_c);
  gpu_loopy_V_F__sumfactors<<<dimGrid, factorgrid, sizeof(f32)*n_factors*8, stream[0]>>>(F_V, V_F, unary_c, NULL, n_factors);
  gpu_loopy_V_F__marginal<<<dimGrid, singGrid, 0, stream[0]>>>(F_V, unary_c, mu, n_factors, stop_thresh, converged);

  for (i=0;i<n_streams;i++) {
    cudaStreamDestroy(stream[i]);
  }

  #ifdef DEBUG
  i32 rip =0;
  cudaMemcpy(newmu, mu, sizeof(f32)*dims[0]*dims[1]*2, cudaMemcpyDeviceToHost);
  for (i=0;i<dims[0]*dims[1]*2 && !rip;i++) {
    if (fabsf(oldmu[i]-newmu[i]) > stop_thresh){
      printf("WTF");
      rip=1;
    }
    if (isnan(oldmu[i]) || isnan(newmu[i])){
      printf("We have a nan\n");
      rip=1;
    }
    //printf("%f ",oldmu[i]-newmu[i]);
  }
  if (rip) {
    printf("\n\n");
  }

  #endif
  cudaMemcpy(targs->_converged, converged, sizeof(i32), cudaMemcpyDeviceToHost);
  

 
}

__global__ void gpu_loopy_V_F__computeunary(f32 * X, f32 *unary_w, f32 *unary_c){
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 c = threadIdx.x;

  unary_c[COORD2(x,y,gridDim.x, gridDim.y, 2) + c] = \
    X[COORD2(x,y,gridDim.x, gridDim.y, 2)] * unary_w[c*2] + \
    X[COORD2(x,y,gridDim.x, gridDim.y, 2) + 1] * unary_w[c*2 + 1];
    
}

__global__ void gpu_loopy_V_F__sumfactors(f32 *F_V, f32 *V_F, f32 *unary_c, const i32 * refimg, i32 n_factors ){
  extern __shared__ char array[];
  f32 *shared_f_v = (f32*) array;
  f32 *shared_v_f = (f32*) array + sizeof(f32)*n_factors*2*2;

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
  for (i=0;i<n_factors;i++) {
    sum += shared_f_v[i*2+c];
  }
  shared_v_f[n*2+c]=sum;
  
  __syncthreads();  
  // Normalize values
  sum = sum - 0.5 * (sum+shared_v_f[n*2+c]);
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
  if (shared_marginal[c] > shared_marginal[c^1]) {
    EY[COORD2(x,y,gridDim.x, gridDim.y, 1)] = 1;
  }
  
}
