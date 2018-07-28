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
extern "C" i32 *predict_loopyGPU(gridCRF_t* self, PyArrayObject *X_py, loopy_params_t *lpar, PyArrayObject *refimg) {
  i32 i,j,h;
  npy_intp *dims=PyArray_DIMS(X_py);
  i32 depth = self->depth;

  gpu_loopy_params_t glpar;
  glpar.max_its = lpar->max_its;
  glpar.stop_thresh = lpar->stop_thresh;
  glpar.eval = lpar->eval;
  glpar.reset_flag = 1;

  i32 n_factors=self->n_factors, nc = self->n_classes;
  i32 n_chan = self->n_inp_channels;  
  i32 n_unary = self->n_unary;

  f32 totL;
  i32 n_samples = 1;
  cudaError_t err = cudaSuccess;

  
  f32 **mu_l = (f32 **) malloc(sizeof(f32*) * n_samples);
  i32 **EY_l = (i32 **) malloc(sizeof(i32*) * n_samples);

  f32 **X_l =  (f32 **) malloc(sizeof(f32*) * n_samples);
  i32 **Y_l =  (i32 **) malloc(sizeof(i32*) * n_samples);

  //cuda streams
  const i32 n_streams = 10;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  i32 curstream = 0;

  /* parameters */
  f32 * V_data;
  cudaMalloc(&V_data, sizeof(f32)*(n_factors*2*nc*nc + n_unary));
  err=cudaMemcpyAsync(V_data, self->V_data, sizeof(f32)*(n_factors*2*nc*nc), cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
  assert(err==cudaSuccess);
  
  f32 *unary_w = V_data + n_factors*2*nc*nc;//sizeof(f32)*n_factors*8;
  err=cudaMemcpyAsync(unary_w, self->unary, n_unary*sizeof(f32), cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
  assert(err==cudaSuccess);
  
  //TODO: copy V to V_data
  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS(X_py);
    //allocate space
    cudaMalloc(&mu_l[j],dims[0]*dims[1]*nc*sizeof(f32));
  
    cudaMalloc(&EY_l[j],sizeof(i32)*dims[0]*dims[1]);
  
    cudaMalloc(&X_l[j], sizeof(f32)*dims[0]*dims[1]*n_chan);
    //Copy images to memory
    err=cudaMemcpyAsync(X_l[j],PyArray_DATA(X_py),sizeof(f32)*dims[0]*dims[1]*n_chan, cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
    assert(err==cudaSuccess);
  }

  i32 **com_l = (i32**) malloc(sizeof(i32*)*n_samples);
  i32 **rom_l = (i32**) malloc(sizeof(i32*)*n_samples);
  om_pair **co_pairs_l = (om_pair**) malloc(sizeof(om_pair*)*n_samples);
  
  /* Prepare coordinates*/
  i32 *_com=(i32*) malloc(sizeof(i32)*n_factors);
  i32 *_rom=(i32*) malloc(sizeof(i32)*n_factors);
  om_pair *_co_pairs=(om_pair*)malloc(sizeof(om_pair)*n_factors);
  for (h=0;h<n_samples;h++) {
    dims=PyArray_DIMS(X_py);


    i32 n=0;
    for (j=1;j<=depth;j++ ) {
      for (i=0;i<j*4;i++) {
	if (i<j) {
	  _com[n]= -j *dims[1] * 2* n_factors*nc - i*2*n_factors*nc;
	  _rom[n]= +j *dims[1] * 2* n_factors*nc + i*2*n_factors*nc;
	  _co_pairs[n]=(om_pair){-j,-i};
	}
	   
	else if (i>=j*3) {
	  _com[n]= +j *dims[1] * 2*n_factors*nc - (j-(i-j*3))*2*n_factors*nc;
	  _rom[n]= -j *dims[1] * 2*n_factors*nc + (j-(i-j*3))*2*n_factors*nc;
	  _co_pairs[n]=(om_pair){j,-(j-(i-j*3))};
	}
	else{
	  _com[n]= (-2*j+i)*dims[1] * 2*n_factors*nc - j*2*n_factors*nc;
	  _rom[n]= (2*j-i)*dims[1] * 2*n_factors*nc + j*2*n_factors*nc;
	  _co_pairs[n]=(om_pair){-2*j+i,-j};
	}
      
	n++;
      }
    }
    
    cudaMalloc(&com_l[h],  sizeof(i32)*n_factors);
    err=cudaMemcpyAsync(com_l[h], _com, sizeof(i32)*n_factors, cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
    assert(err==cudaSuccess);
    cudaMalloc(&rom_l[h],  sizeof(i32)*n_factors);
    err=cudaMemcpyAsync(rom_l[h], _rom, sizeof(i32)*n_factors, cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
    assert(err==cudaSuccess);
    cudaMalloc(&co_pairs_l[h],  sizeof(om_pair)*n_factors);
    err=cudaMemcpyAsync(co_pairs_l[h], _co_pairs, sizeof(om_pair)*n_factors, cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
    assert(err==cudaSuccess);
    
  }
  /* End prepare coordinates*/

  

  f32 ** unary_c_l = (f32**) malloc(sizeof(f32*) * n_samples);
  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS(X_py);
    cudaMalloc(&unary_c_l[j], dims[0]*dims[1]*nc*sizeof(f32));
  }

  f32 *RE, *CE;
  cudaMalloc(&RE, sizeof(f32) * nc* n_factors *nc);
  cudaMalloc(&CE, sizeof(f32) * nc* n_factors *nc);

  f32 **V_F_l = (f32**) malloc(sizeof(f32*) * n_samples);
  f32 **F_V_l = (f32**) malloc(sizeof(f32*) * n_samples);

  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS(X_py);
    cudaMalloc(&V_F_l[j], sizeof(f32)*dims[0]*dims[1]*2*n_factors*nc);
    cudaMalloc(&F_V_l[j], sizeof(f32)*dims[0]*dims[1]*2*n_factors*nc);
  }


  j=0;
  gpu_loopy_data gdata;
  gdata.V_data = V_data;
  gdata.RE = RE;
  gdata.CE = CE;
  gdata.unary_w = unary_w;
  

  gdata.V_F = V_F_l[j];
  gdata.F_V = F_V_l[j];
  gdata.mu = mu_l[j];
  gdata.com = com_l[j];
  gdata.rom = rom_l[j];
  gdata.co_pairs = co_pairs_l[j];
  gdata.unary_c = unary_c_l[j];
  gdata.EY = EY_l[j];
  gdata.X = X_l[j];

  glpar.gdata = &gdata;

  //call loopygpu
  loopyGPU(self, X_py, &glpar, NULL);

  dims= PyArray_DIMS(X_py);
  err=cudaMemcpy(lpar->EY, EY_l[j], sizeof(i32)*dims[0]*dims[1], cudaMemcpyDeviceToHost);
  assert(err==cudaSuccess);
  

  
  cudaDeviceSynchronize();
  cudaFree(V_data);
  cudaFree(RE);
  cudaFree(CE);
  for (j=0;j<n_samples;j++){
    cudaFree(mu_l[j]);
    cudaFree(EY_l[j]);
    cudaFree(com_l[j]);
    cudaFree(rom_l[j]);
    cudaFree(co_pairs_l[j]);
    cudaFree(unary_c_l[j]);
    cudaFree(V_F_l[j]);
    cudaFree(F_V_l[j]);
  }
  free(mu_l);
  free(EY_l);
  free(X_l);
  free(Y_l);
  free(_com);
  free(_rom);
  free(_co_pairs);
  return lpar->EY;
}

extern "C" i32 *loopyGPU(gridCRF_t* self, PyArrayObject *X_py, gpu_loopy_params_t *lpar, PyArrayObject *refimg){
  #define VERBOSE 0
  npy_intp * dims= PyArray_DIMS(X_py);
  i32 n_factors=self->n_factors, nc = self->n_classes;
  i32 max_it=lpar->max_its,it;
  gpu_loopy_data *gdata = lpar->gdata;
  
  f32 *V_data = gdata->V_data;
  i32 i;
  


  /* transfer matrices */
  
  // set up threads

  const i32 n_streams = 10;
  cudaStream_t stream[n_streams];
  i32 curstream=0;
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  /* Allocate message buffers*/
  f32 *F_V = gdata->F_V, *V_F = gdata->V_F;
  i32 n_elem = dims[0] * dims[1] * (n_factors*2) * nc;
  dim3 blockGrid(n_elem/128 + 1);
  dim3 threadGrid(128);
  
  if (lpar->reset_flag) {//reset data if the flag is set.
    gpu_fill_value<<<blockGrid, threadGrid,0, stream[(curstream++)%n_streams]>>>(F_V, 0.0, n_elem);
    gpu_fill_value<<<blockGrid, threadGrid,0, stream[(curstream++)%n_streams]>>>(V_F, 0.0, n_elem);
  }
  /* Allocate coordinate system*/

  f32 *mu = gdata->mu;
  n_elem = dims[0] * dims[1] * nc;
  dim3 blockGrid1(n_elem/128 + 1);
  dim3 threadGrid1(128);

  if (lpar->reset_flag) { //reset data if the flag is set.
    gpu_fill_value<<<blockGrid1, threadGrid1, 0, stream[(curstream++)%n_streams]>>>(mu, BIG, n_elem);
  }

  f32 *RE= gdata->RE, *CE= gdata->CE;
  n_elem = n_factors*nc*nc; //was 4

  i32 blockWidth = (i32) ceil(128.0/nc/nc);
  dim3 blockGrid2(n_elem/blockWidth/nc/nc + 1);
  dim3 threadGrid2(blockWidth,nc,nc); //todo, change in case 32 is too big
 
  gpu_swizzle<<<blockGrid2, threadGrid2,0, stream[(curstream++)%n_streams]>>>(RE, V_data, n_factors, n_elem);
  gpu_swizzle<<<blockGrid2, threadGrid2,0, stream[(curstream++)%n_streams]>>>(CE, V_data + n_elem, n_factors, n_elem);
 
  for (i=0;i<n_streams;i++) {
    cudaStreamDestroy(stream[i]);
  }

    
  i32 *converged;
  assert(cudaMallocManaged(&converged,sizeof(i32))==cudaSuccess);
  //i32 _converged = 1;
  //cudaMemcpy(converged, &_converged, sizeof(i32), cudaMemcpyHostToDevice);
  *converged = 1;

  loopygpu_t targs;
  targs.dims = dims;
  targs.lpar = lpar;
  targs.self = self;
  targs.dev_converged = converged;
  targs.gdata = gdata;

  cudaDeviceSynchronize();
  for (it = 0; it < max_it; it++){
#if VERBOSE
    if (it%10==0){
      printf("gpu it %d\n", it);
    }
    #endif
    gpu_loopy_F_V(&targs);
    gpu_loopy_V_F(&targs);
    
    if (*converged) break;
  }
  #if VERBOSE
  printf("converged %d %f\n",*converged, lpar->stop_thresh);
  #endif
  i32 *EY = gdata->EY;

  
  dim3 dimGrid(dims[0],dims[1]);
  dim3 singGrid(nc);
  gpu_loopy_V_F__label<<<dimGrid, singGrid, nc*sizeof(f32)>>>(mu, EY, n_factors);

  cudaFree(converged);  
  cudaDeviceSynchronize();
  return gdata->EY;
}


extern "C" void gpu_loopy_F_V(loopygpu_t *targs) { 

  npy_intp * dims= targs->dims;
  gridCRF_t *self = targs->self;
  
  i32 n_factors=self->n_factors, nc = self->n_classes;
  
  f32 *F_V= targs->gdata->F_V;
  f32 *V_F= targs->gdata->V_F;

  f32 *RE = targs->gdata->RE;
  f32 *CE = targs->gdata->CE;

  i32 *com = targs->gdata->com;
  i32 *rom = targs->gdata->rom;
  om_pair * co_pairs = targs->gdata->co_pairs;
  
  i32 i;

  const i32 n_streams = 2;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  

  dim3 dimGrid(dims[0],dims[1]);
  dim3 factorgrid(n_factors,nc);
		  
  gpu_loopy_F_V__Flow<<<dimGrid, factorgrid, sizeof(f32)*n_factors*nc, stream[0]>>>(F_V, V_F, RE, NULL, com, co_pairs, n_factors, nc);
  gpu_loopy_F_V__Fup<<<dimGrid, factorgrid, sizeof(f32)*n_factors*nc, stream[1]>>>(F_V, V_F, CE, NULL, rom, co_pairs, n_factors, nc);

  for (i=0;i<2;i++) {
    cudaStreamDestroy(stream[i]);
  }

}

//TODO: this can be optimized by making the blocks be other n_factors, then using shared memory.
// therefore, we can put RE into shared memory. (over cc)
//TODO: optimize by computing each max candidate indpendently
/* Naive method */
__global__ void gpu_loopy_F_V__Flow(f32 *F_V, f32 *V_F, f32 *RE, const i32 * refimg, const i32 * com, const om_pair * co_pairs,  i32 n_factors, i32 nc){
  /* Naive code*/
  extern __shared__ f32 array[];
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 n = threadIdx.x;
  i32 c = threadIdx.y;
  i32 cc = c;
  i32 origin=COORD3(x,y,0,gridDim.x,gridDim.y,2*n_factors,nc);

  array[threadIdx.x*blockDim.y+cc] = V_F[origin + cc];
  __syncthreads();

  /* Check bounds for upper factor */
  om_pair cop = co_pairs[n];
  if ( ! (x+cop.x <0 || x+cop.x >= gridDim.x || y+cop.y < 0 || y+cop.y >=gridDim.y) ){//&& !(refimg[COORD2(x+cop.x,y+cop.y, gridDim.x, gridDim.y, 1)]==0)) {
    

    i32 co = origin + com[n] + nc*(n + n_factors);

    if (!(co< 0 || co >= gridDim.x * gridDim.y * (n_factors*2) *nc)) {
      f32 max = RE[0+n*nc+c] + array[threadIdx.x*blockDim.y];
      for (cc = 1; cc < nc; cc++) {
	//max = max > RE[cc*n_factors*nc + n*nc + c] + V_F[origin + cc] ? max : RE[cc*n_factors*nc + n*nc + c] + V_F[origin + cc];
	max = max > RE[cc*n_factors*nc + n*nc + c] + array[threadIdx.x*blockDim.y + cc] ? max : RE[cc*n_factors*nc + n*nc + c] + array[threadIdx.x*blockDim.y + cc];
      }
      F_V[co+c]=max;
    }
  }
}


//F_V[co] = RE[n*2] + V_F[origin] > RE[n_factors*2 + n*2] + V_F[origin+1] ? RE[n*2] + V_F[origin] : RE[n_factors*2 + n*2] + V_F[origin+1];
//F_V[co+1] = RE[n*2+1] + V_F[origin] > RE[n_factors*2 + n*2 + 1] + V_F[origin+1] ? RE[n*2+1] + V_F[origin] : RE[n_factors*2 + n*2 + 1] + V_F[origin+1];
__global__ void gpu_loopy_F_V__Fup(f32 *F_V, f32 *V_F,  f32 *CE, const i32 * refimg, const i32 *rom, const om_pair *co_pairs, i32 n_factors, i32 nc){
  /* Naive code*/
  extern __shared__ f32 array[];
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 n = threadIdx.x;
  i32 c = threadIdx.y;
  i32 cc = c;
  i32 origin=COORD3(x,y,0,gridDim.x,gridDim.y,2*n_factors,nc);
  array[threadIdx.x*blockDim.y+cc] = V_F[origin + cc]; // shared V_F
  __syncthreads();
  

  /* Check bounds for upper factor */
  om_pair cop=co_pairs[n];
  /* Check bounds for lower factor */
  if (!(x-cop.x < 0 || x-cop.x >= gridDim.x || y-cop.y < 0 || y-cop.y >=gridDim.y)){// && !(refimg[COORD2(x-cop.x,y-cop.y, gridDim.x, gridDim.y, 1)]==0)) {

    //i32 co=origin + rom[n]; //check this
    i32 co = origin + rom[n] + nc*n;

    
    if (!(co< 0 || co >= gridDim.x * gridDim.y * (n_factors*2) *nc)) {


      f32 max = CE[0+n*nc+c] + V_F[origin+0];
      for (cc = 1; cc < nc; cc++) {
	max = max > CE[cc*n_factors*nc + n*nc + c] + array[threadIdx.x*blockDim.y+cc] ? max : CE[cc*n_factors*nc + n*nc + c] + array[threadIdx.x*blockDim.y+cc];
      }
      F_V[co+c]=max;
    }

    
    /*
    if (!(co< 0 || co >= gridDim.x * gridDim.y * (n_factors*2) *2)) {
      F_V[co] = CE[n*2] + V_F[origin] > CE[n_factors*2 + n*2] + V_F[origin+1] ?
	CE[n*2] + V_F[origin] : CE[n_factors*2 + n*2] + V_F[origin+1];
  
      F_V[co+1] = CE[n*2+1] + V_F[origin] > CE[n_factors*2 + n*2 + 1] + V_F[origin+1] ?
	CE[n*2+1] + V_F[origin] : CE[n_factors*2 + n*2 + 1] + V_F[origin+1];
    }
    }*/
  }
}



extern "C" void gpu_loopy_V_F(loopygpu_t *targs) {


  i32 i;
  gridCRF_t *self = targs->self;
  f32 *X = targs->gdata->X;
  i32 n_chan = targs->self->n_inp_channels;

  gpu_loopy_params_t * lpar = targs->lpar;
  
  npy_intp * dims= targs->dims;
  i32 n_factors=self->n_factors, nc = self->n_classes;
  f32 stop_thresh=lpar->stop_thresh;
  

  f32 * unary_w= targs->gdata->unary_w;
  f32 * unary_c = targs->gdata->unary_c;

  f32 *F_V = targs->gdata->F_V;
  f32 *V_F = targs->gdata->V_F;
  f32 *mu = targs->gdata->mu;

  /* runtime Flags*/
  i32 *converged = targs->dev_converged;
  
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
  dim3 factorgrid(2*n_factors,nc);
  dim3 singGrid(nc);
  gpu_loopy_V_F__computeunary<<<dimGrid, singGrid, 0, stream[0]>>>(X, unary_w, unary_c, n_chan);
  gpu_loopy_V_F__sumfactors<<<dimGrid, factorgrid, sizeof(f32)*2*(n_factors*2*nc), stream[0]>>>(F_V, V_F, unary_c, NULL, n_factors);
  gpu_loopy_V_F__marginal<<<dimGrid, singGrid, 0, stream[0]>>>(F_V, unary_c, mu, n_factors, stop_thresh, converged);
  //gpu_loopy_V_F__marginal<<<dimGrid, factorgrid, sizeof(f32)*n_factors*2*nc, stream[0]>>>(F_V, unary_c, mu, n_factors, stop_thresh, converged);
  
  for (i=0;i<n_streams;i++) {
    cudaStreamDestroy(stream[i]);
  }
  

  //cudaMemcpy(targs->host_converged, converged, sizeof(i32), cudaMemcpyDeviceToHost);
 
}

__global__ void gpu_loopy_V_F__computeunary(f32 * X, f32 *unary_w, f32 *unary_c, i32 n_chan){
   
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 c = threadIdx.x;

  f32 sum = 0.0f;
  i32 i;
  for (i=0;i<n_chan;i++) {
    sum -= X[COORD2(x,y,gridDim.x, gridDim.y, n_chan) + i] * unary_w[c*n_chan + i];
  }
  unary_c[COORD2(x,y,gridDim.x, gridDim.y, blockDim.x) + c ] = sum;

  
  /*unary_c[COORD2(x,y,gridDim.x, gridDim.y, 2) + c] = -(   \
    X[COORD2(x,y,gridDim.x, gridDim.y, 2)] * unary_w[c*2] +		\
    X[COORD2(x,y,gridDim.x, gridDim.y, 2) + 1] * unary_w[c*2 + 1]);
  */
  
}

__global__ void gpu_loopy_V_F__sumfactors(f32 *F_V, f32 *V_F, f32 *unary_c, const i32 * refimg, i32 n_factors ){

  extern __shared__ f32 array[];
  f32 *shared_f_v = (f32*) array;
  f32 *shared_v_f = (f32*) &array[n_factors*2*blockDim.y];

  i32 x = blockIdx.x; //dims0
  i32 y = blockIdx.y; //dims1
  i32 n = threadIdx.x; // nfactors*2
  i32 c = threadIdx.y; // 2
  //blockDim.y : nc
  i32 i=0;
  i32 origin = COORD3(x,y,n,gridDim.x, gridDim.y, 2*n_factors, blockDim.y) + c;
  // load factor to vvariables into shared memory
  shared_f_v[n*blockDim.y + c] = F_V[origin];

  //TODO: make unary a constant?
  f32 sum = unary_c[COORD2(x,y,gridDim.x,gridDim.y,blockDim.y) + c] - shared_f_v[n*blockDim.y+c];
  __syncthreads();


  /* Sum up all messages */
  for (i=0;i<2*n_factors;i++) {
    sum += shared_f_v[i*blockDim.y+c];
  }

  shared_v_f[n*blockDim.y+c]=sum;
  __syncthreads();
  // Normalize values
  f32 total = 0;
  for (i=0;i<blockDim.y;i++) {
    total += shared_v_f[n*blockDim.y + i];
  }
  //sum = sum - 0.5 * (sum+shared_v_f[n*blockDim.y+c^1]); //uh oh...
  V_F[origin] = sum - total/blockDim.y;
  
}

/*
Calulate marginals using variable to factor message passing
 */
__global__ void gpu_loopy_V_F__marginal(f32 *F_V, f32 * unary_c,  f32 * mu, i32 n_factors, f32 stop_thresh, i32 *converged) {
  #if 0
  extern __shared__ f32 array[];//
  f32 *shared_f_v = (f32*) array;//
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  //i32 c = threadIdx.x;
  i32 c = threadIdx.y;//
  shared_f_v[threadIdx.x*blockDim.y + c] = F_V[COORD3(x,y,threadIdx.x,gridDim.x, gridDim.y, 2*n_factors, blockDim.y) + c];//
  __syncthreads();//
  if (threadIdx.x!=0) return;//
  i32 i;
  //i32 origin = COORD2(x,y,gridDim.x, gridDim.y, blockDim.x) + c;
  i32 origin = COORD2(x,y,gridDim.x, gridDim.y, blockDim.y) + c;//
  f32 sum = unary_c[origin];

  // sum up factors
  for (i=0;i<n_factors*2;i++) { //TODO: this can be optimized. Fetch coalesced memory using shared
    // then sum up in stridesx
    //sum += F_V[COORD3(x,y,i,gridDim.x, gridDim.y, 2*n_factors, blockDim.x) + c];
    sum += shared_f_v[i*blockDim.y + c];
  }

  if (fabsf(sum - mu[origin]) > stop_thresh) {
    converged[0] = 0;
  }
  mu[origin] = sum;
  #endif
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 c = threadIdx.x;
  i32 i;
  i32 origin = COORD2(x,y,gridDim.x, gridDim.y, blockDim.x) + c;

  f32 sum = unary_c[origin];

  // sum up factors
  for (i=0;i<n_factors*2;i++) { //TODO: this can be optimized. Fetch coalesced memory using shared
    // then sum up in stridesx
    sum += F_V[COORD3(x,y,i,gridDim.x, gridDim.y, 2*n_factors, blockDim.x) + c];
  }

  if (fabsf(sum - mu[origin]) > stop_thresh) {
    converged[0] = 0;
  }
  mu[origin] = sum;
}

//TODO: change to mu instead
__global__ void gpu_loopy_V_F__label(f32 *mu, i32 *EY, i32 n_factors) {

  /* Computes the predicted label given the values */
  extern __shared__ f32 array[];
  f32 *shared_marginal = (f32*) array;
  i32 i;
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 c = threadIdx.x;
  i32 origin = COORD2(x,y,gridDim.x, gridDim.y, blockDim.x) + c;
  
  shared_marginal[c] = mu[origin];
  __syncthreads();
  // find the max and set it as the label.
  for (i=0;i<blockDim.x;i++) {
    if (i==c) continue;
    if (shared_marginal[c] < shared_marginal[i]) return;
  }
  EY[COORD2(x,y,gridDim.x, gridDim.y, 1)] = c;
  
  // if (c==0  && shared_marginal[0] > shared_marginal[1]) {
  //   EY[COORD2(x,y,gridDim.x, gridDim.y, 1)] = 0;
  // }
  // else if (c==0) {
  //   EY[COORD2(x,y,gridDim.x, gridDim.y, 1)] = 1;
  // }
  
}


__global__ void gpu_fill_value(f32 *buffer, f32 val, i32 lim) {
  i32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < lim) {buffer[index] = val;}
}

__global__ void gpu_multiply(f32 *buffer, i32 lim) {
   i32 index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < lim) {buffer[index] = -buffer[index];}
}


__global__ void gpu_swizzle(f32 *dest, f32 *src, i32 n_factors, i32 lim) {


  //TODO: check results if there is an even factor

  i32 srcindex = blockIdx.x*blockDim.x*blockDim.y*blockDim.z + threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z;
  i32 destindex = threadIdx.y * n_factors * 2 + (blockIdx.x*blockDim.x*blockDim.z) + threadIdx.x * blockDim.z + threadIdx.z;
  if ((blockIdx.x*blockDim.x*blockDim.z) + threadIdx.x*blockDim.z + threadIdx.z < n_factors*2) {
    dest[destindex] = -src[srcindex];
  }
}
