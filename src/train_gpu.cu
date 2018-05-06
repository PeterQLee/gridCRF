extern "C" {
#include "train_gpu_cu.h"
  
}
#define N_UNARY 4
//TODO: copy V_data, to allocated V_data
extern "C" void GPU_grad_descent(gradient_t *args,i32 epochs,i32 dummy) {
  #define VERBOSE 0
  i32 h,i,j;

  gridCRF_t *self = args->self;
  npy_intp *dims=args->dims;
  i32 depth = self->depth;

  gpu_loopy_params_t lpar;
  lpar.max_its = args->lpar->max_its;
  lpar.stop_thresh = args->lpar->stop_thresh;
  lpar.eval = args->lpar->eval;

  i32 n_factors=args->n_factors;

  PyObject *X_list=args->X_list;
  PyObject *Y_list=args->Y_list;
  
  
  f32 totL;
  i32 n_samples = PyList_Size(X_list);
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
  cudaMalloc(&V_data, sizeof(f32)*(n_factors*8 + N_UNARY));//check this...
  err=cudaMemcpyAsync(V_data, self->V_data, sizeof(f32)*(n_factors*8), cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
  assert(err==cudaSuccess);
  
  f32 *unary_w = V_data + n_factors*8;//sizeof(f32)*n_factors*8;
  err=cudaMemcpyAsync(unary_w, self->unary, N_UNARY*sizeof(f32), cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
  assert(err==cudaSuccess);
  
  //TODO: copy V to V_data
  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,j));
    //allocate space
    cudaMalloc(&mu_l[j],dims[0]*dims[1]*2*sizeof(f32));
    
    cudaMalloc(&EY_l[j],sizeof(i32)*dims[0]*dims[1]);

    cudaMalloc(&X_l[j], sizeof(f32)*dims[0]*dims[1]*2);
    cudaMalloc(&Y_l[j], sizeof(i32)*dims[0]*dims[1]*2);
    //Copy images to memory
    err=cudaMemcpyAsync(X_l[j],PyArray_DATA(((PyArrayObject*) PyList_GetItem(X_list,j))), \
		    sizeof(f32)*dims[0]*dims[1]*2, cudaMemcpyHostToDevice,
		    stream[(curstream++)%n_streams]);
    assert(err==cudaSuccess);
    err=cudaMemcpyAsync(Y_l[j], PyArray_DATA(((PyArrayObject*)PyList_GetItem(Y_list,j))), \
		    sizeof(i32)*dims[0]*dims[1]*2, cudaMemcpyHostToDevice,\
		    stream[(curstream++)%n_streams]);
    assert(err==cudaSuccess);
  }

  f32 *V_change;
  cudaMalloc(&V_change, sizeof(f32)*(n_factors*4*2+N_UNARY));
  f32 *unary_change = V_change + n_factors*4*2;

  i32 **com_l = (i32**) malloc(sizeof(i32*)*n_samples);
  i32 **rom_l = (i32**) malloc(sizeof(i32*)*n_samples);
  om_pair **co_pairs_l = (om_pair**) malloc(sizeof(om_pair*)*n_samples);
  
  /* Prepare coordinates*/
  i32 *_com=(i32*) malloc(sizeof(i32)*n_factors);
  i32 *_rom=(i32*) malloc(sizeof(i32)*n_factors);
  om_pair *_co_pairs=(om_pair*)malloc(sizeof(om_pair)*n_factors);
  for (h=0;h<n_samples;h++) {
    dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,h));


    i32 n=0;
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
    dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,j));
    cudaMalloc(&unary_c_l[j], dims[0]*dims[1]*2*sizeof(f32));
  }

  f32 *RE, *CE;
  cudaMalloc(&RE, sizeof(f32) * 2* n_factors *2);
  cudaMalloc(&CE, sizeof(f32) * 2* n_factors *2);

  f32 **V_F_l = (f32**) malloc(sizeof(f32*) * n_samples);
  f32 **F_V_l = (f32**) malloc(sizeof(f32*) * n_samples);

  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,j));
    cudaMalloc(&V_F_l[j], sizeof(f32)*dims[0]*dims[1]*n_factors*4);
    cudaMalloc(&F_V_l[j], sizeof(f32)*dims[0]*dims[1]*n_factors*4);
  }


  i32 * ainc;
  i32 * binc;
  cudaMalloc(&ainc, sizeof(i32)*n_factors*2);  
  cudaMalloc(&binc, sizeof(i32)*n_factors*2);
  err=cudaMemcpyAsync(ainc, args->ainc, sizeof(i32)*n_factors*2, cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
  assert(err==cudaSuccess);
  err=cudaMemcpyAsync(binc, args->binc, sizeof(i32)*n_factors*2, cudaMemcpyHostToDevice, stream[(curstream++)%n_streams]);
  assert(err==cudaSuccess);
  
  
  
  gpu_gradient_t g_args;
  g_args.self = self;
  g_args.dev_ainc = ainc;
  g_args.dev_binc = binc;
  g_args.num_params= args->num_params;
  g_args.n_factors = args->n_factors;
  g_args.alpha=args->alpha;
  g_args.lpar = &lpar;
  g_args.host_L = 0.0;
  g_args.dev_V_change = V_change;
  g_args.dev_unary_change = unary_change;
  
  cudaMalloc(&(g_args.dev_L),sizeof(f32));
  
  gpu_loopy_data gdata;
  gdata.V_data = V_data;
  gdata.RE = RE;
  gdata.CE = CE;
  gdata.unary_w = unary_w;
  
  g_args.gdata = &gdata;
  lpar.gdata = &gdata;

  for (i=0;i<n_streams;i++) {
    cudaStreamDestroy(stream[i]);
  }

  //shuffle the training examples
  srand(0);
  i32 *inds = indlist(n_samples);

  
  for (i=0;i < epochs;i++) {
    shuffle_inds(inds, n_samples);
    for (j=0;j < n_samples;j++){
      dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,inds[j]));
      
      gdata.V_F = V_F_l[inds[j]];
      gdata.F_V = F_V_l[inds[j]];
      gdata.mu = mu_l[inds[j]];
      gdata.com = com_l[inds[j]];
      gdata.rom = rom_l[inds[j]];
      gdata.co_pairs = co_pairs_l[inds[j]];
      gdata.unary_c = unary_c_l[inds[j]];
      gdata.EY = EY_l[inds[j]];
      gdata.X = X_l[inds[j]];

      g_args.dev_X = X_l[inds[j]];
      g_args.dev_Y = Y_l[inds[j]];
      g_args.dims= dims;

      loopyGPU(self, (PyArrayObject*)PyList_GetItem(X_list,inds[j]), &lpar, NULL);
      gpu_calculate_gradient(&g_args);
    }
  }
  free(inds);

  
  // copy V_data back to numpy space...
  // also copy unary data back to numpy space3
  cudaMemcpy(self->V_data, V_data,  sizeof(f32)*(n_factors*8), cudaMemcpyDeviceToHost);
  cudaMemcpy(self->unary, unary_w,  sizeof(f32)*(N_UNARY), cudaMemcpyDeviceToHost);

  //Time to clean up everything
  cudaFree(V_data);
  cudaFree(RE);
  cudaFree(CE);

  

  cudaFree(g_args.dev_L);
  for (j=0;j<n_samples;j++){
    cudaFree(mu_l[j]);
    cudaFree(EY_l[j]);
    cudaFree(X_l[j]);
    cudaFree(Y_l[j]);
    cudaFree(com_l[j]);
    cudaFree(rom_l[j]);
    cudaFree(co_pairs_l[j]);
    cudaFree(unary_c_l[j]);
    cudaFree(V_F_l[j]);
    cudaFree(F_V_l[j]);
  }
  cudaFree(V_change);
  cudaFree(ainc);
  cudaFree(binc);
  
  free(mu_l);
  free(EY_l);
  free(X_l);
  free(Y_l);
  free(_com);
  free(_rom);
  free(_co_pairs);
}



static void gpu_calculate_gradient(gpu_gradient_t *args) {
  f32 *X = args->dev_X;
  i32 *Y = args->dev_Y;
  i32 *ainc = args->dev_ainc, *binc = args->dev_binc;
  f32 *V_change = args->dev_V_change;
  f32 *unary_change = args->dev_unary_change;
  f32 *unary_w = args->gdata->unary_w;
  f32 *unary_c = args->gdata->unary_c;
  
  f32 *L = args->dev_L;
  npy_intp *dims = args->dims;
  i32 n_factors = args->self->n_factors;

  i32 * EY = args->gdata->EY;
  f32 * V = args->gdata->V_data;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  i32 n_elem;
  n_elem=(n_factors*4*2+N_UNARY);
  dim3 blockGrid(n_elem/128 + 1);
  dim3 threadGrid(128);
  gpu_fill_value<<<blockGrid, threadGrid,0,stream>>>(V_change,0.0, n_elem);
  //note, this also fills unary_change

  f32 alpha = args->alpha;
  dim3 factorgrid(2*n_factors,2);
  dim3 singGrid(2);
  gpu_loopy_V_F__computeunary<<<factorgrid, singGrid,0 ,stream >>>(X, unary_w, unary_c);

  dim3 blockGrid1(dims[0]/16 + 1, dims[1]/16 + 1);
  dim3 threadGrid1(16,16,2);

  
  gpu_entropy_partial<<<blockGrid1, threadGrid1, sizeof(f32)*16*16*2, stream >>> (unary_c, EY, X, Y, V, V_change, unary_change, ainc, binc, alpha, (i32) dims[0], (i32) dims[1], n_factors);
  
  dim3 blockGrid2(1);
  dim3 threadGrid2(n_factors*8 + N_UNARY);
  gpu_update_params<<<blockGrid2, threadGrid2,0 , stream>>> (V, V_change, 1.0/(dims[0]*dims[1])); //this also includes unary changes

  cudaStreamDestroy(stream);
}


__global__ void gpu_entropy_partial(f32 *unary_c, i32 *EY, f32 *X, i32 *Y, f32 *V, f32 *V_change, f32* unary_change, i32 *ainc, i32 *binc, f32 alpha, i32 limx, i32 limy, i32 n_factors) {

  //last pitch idea. Forget cond until the very end
  
  // TODO: optimize by putting everything V into shared data.
  // Also, possibly EY

  //TODO: check page 83, mentions that V_change and unary need to be properly aligned.
  i32 x = blockIdx.x * 16 + threadIdx.x;
  i32 y = blockIdx.y * 16 + threadIdx.y;
  i32 c= threadIdx.z;
  i32 i;
  i32 l;
  i32 co = ((x)*limy + y);
  i32 cond= (x >= limx || y >= limy) || (Y[co*2+c]==0 && Y[co*2+c^1]==0);
  extern __shared__ char array[];
  //f32 *shared_V = (f32*) array;  // can copy this by using elements in reange

  f32 *shared_sum = (f32*) array ;//+ n_factors*8*sizeof(f32);
  f32 sum, max, s1, change;
  __syncthreads();
  
  if (!cond) {
    sum = -unary_c[2*co+c];
    
    for (i=0;i<n_factors;i++) {
      if (x+ainc[i] < 0 || x+ainc[i]>=limx || y+binc[i] < 0 || y+binc[i] >= limy) continue;
      l= EY[COORD2(x+ainc[i],y+binc[i],limx,limy,1)];
      sum += V[i*4 + (l)*2 + c];
    }
    for (i=0;i<n_factors;i++) {
      if (x+ainc[i+n_factors] < 0 || x+ainc[i+n_factors]>=limx || y+binc[i+n_factors] < 0 || y+binc[i+n_factors] >= limy) continue;
    
      l= EY[COORD2(x+ainc[i+n_factors],y+binc[i+n_factors],limx,limy,1)];
      sum += V[n_factors*4 + i*4 + (l)*2 + c];
    }
    
    //put sum into shared memory
    shared_sum[threadIdx.x*16*2 + threadIdx.y*2 +c] = sum;
  }
  __syncthreads();
  if(!cond) {
    
    if (sum < shared_sum[threadIdx.x*16*2 + threadIdx.y*2+c^1]){
      max = sum;
    }
    else{
      max = shared_sum[threadIdx.x*16*2 + threadIdx.y*2+c^1];
    }
  }
  __syncthreads();
  if (!cond) {  
    s1 = expf(-shared_sum[threadIdx.x*16*2 + threadIdx.y*2+c]-max);

    shared_sum[threadIdx.x*16*2 + threadIdx.y*2+c] = s1;
  }
  __syncthreads();
  // Each thread handles the specific class
  //Softmax
  if (!cond) {
    l = Y[co*2+c];
    s1= shared_sum[threadIdx.x*16*2 + threadIdx.y*2+c] / (shared_sum[threadIdx.x*16*2 + threadIdx.y*2]+shared_sum[threadIdx.x*16*2 + threadIdx.y*2+1]); 

    
    change = -alpha*(l-s1);
    //printf("%d %d %d %f %d %f\n", threadIdx.x, threadIdx.y, c, s1, l, change);


    atomicAdd(&unary_change[c*2], change*X[co*2]);
    atomicAdd(&unary_change[c*2+1], change*X[co*2+1]);
    
    //possible optimization
    for (i=0;i<n_factors;i++) {
      if (x+ainc[i] < 0 || x+ainc[i]>=limx || y+binc[i] < 0 || y+binc[i] >= limy) continue;
      l= EY[COORD2(x+ainc[i],y+binc[i],limx,limy,1)];
    //Atomic add
      atomicAdd(&V_change[i*4 + 2*l +c], change);
    }
    
    for (i=0;i<n_factors;i++) {
      if (x+ainc[n_factors+i] < 0 || x+ainc[n_factors+i]>=limx || y+binc[n_factors+i] < 0 || y+binc[n_factors+i] >= limy) continue;
      
      l= EY[COORD2(x+ainc[n_factors+i],y+binc[n_factors+i],limx,limy,1)];
      //Atomic add
      atomicAdd(&V_change[n_factors*4 + i*4 + 2*l +c], change);
    }
  }
}

__global__ void gpu_update_params(f32 *V, f32* V_change, f32 lr) {
  V[threadIdx.x] += lr*V_change[threadIdx.x];
}
