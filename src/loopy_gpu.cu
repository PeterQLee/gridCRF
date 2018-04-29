extern "C" {
#include "loopy_gpu_cu.h"
}
/* 
Optimization strats:
1. Split F_V so that it can take advantage of constant memory for offset coordinates. This can speed up execution possibly.
I.e.
have one block take care of factor index 1 for a portion of variables.

*/

/* void gpu_loopy_F_V(gridCRF_t* self, PyArrayObject *X,loopy_params_t *lpar,PyArrayObject *refimg) { */

extern "C" void gpu_loopy_F_V(loopycpu_t *targs) { 

  npy_intp * dims= PyArray_DIMS(targs->X);
  gridCRF_t *self = targs->self;
  
  i32 n_factors=self->n_factors;
  

  //TODO: delete this after testing
  f32 *_F_V= targs->F_V;
  f32 *_V_F= targs->V_F;

  f32 *_RE = targs->RE;
  f32 *_CE = targs->CE;

  i32 *_com = targs->com;
  i32 *_rom = targs->rom;
  om_pair * _co_pairs = targs->co_pairs;

  i32 *_refimg = (i32 *) malloc(sizeof(i32) * dims[0] *dims[1]);

  // need to translate refimg
  ///

  
  i32 n, i ,x, y;
  i32 l;
  

  /* Set messages */
  /* f32 * _F_V = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32); */
  /* f32 * _V_F = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32); */

  f32 *F_V, *V_F;
  cudaMalloc(&F_V, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32));
  cudaMalloc(&V_F, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32));

  f32 *RE, *CE;
  cudaMalloc(&RE, 2* n_factors *2* sizeof(f32));
  cudaMalloc(&CE, 2* n_factors *2* sizeof(f32));
  
  i32 *refimg;
  cudaMalloc(&refimg, dims[0] * dims[1] * sizeof(i32));
  /* for (i=0;i<dims[0] * dims[1] * (n_factors*2) *2; i++){ */
  /*   _F_V[i]=0.0f; */
  /*   _V_F[i]=0.0f; */
  /* } */
  const i32 n_streams = 8;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  cudaMemcpyAsync(_F_V, F_V, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(_V_F, V_F, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyHostToDevice, stream[1]);


  /* Allocate coordinate system*/
  i32 *com, *rom;
  om_pair *co_pairs;
  cudaMalloc(&com, sizeof(i32)*n_factors);
  cudaMalloc(&rom, sizeof(i32)*n_factors);
  cudaMalloc(&co_pairs,sizeof(om_pair)*n_factors);
  
  /* i32 *_com=(i32*)malloc(sizeof(i32)*n_factors); */
  /* i32 *_rom=(i32*)malloc(sizeof(i32)*n_factors); */
  /* om_pair *_co_pairs=(om_pair*)malloc(sizeof(om_pair)*n_factors); */
  /* om_pair cop; */
  /* n=0; */
  /* for (j=1;j<=depth;j++ ) { */
  /*   for (i=0;i<j*4;i++) { */
  /*     if (i<j) { */
  /* 	_com[n]= -j *dims[1] * n_factors*2*2 - i*n_factors*2*2; */
  /* 	_rom[n]= +j *dims[1] * n_factors*2*2 + i*n_factors*2*2; */
  /* 	_co_paiurs[n]=(om_pair){-j,-i}; */
  /*     } */
  /*     else if (i>=j*3) { */
  /* 	_com[n]= +j *dims[1] * n_factors*2*2 - (j-(i-j*3))*n_factors*2*2; */
  /* 	_rom[n]= -j *dims[1] * n_factors*2*2 + (j-(i-j*3))*n_factors*2*2; */
  /* 	_co_paiurs[n]=(om_pair){j,-(j-(i-j*3))}; */
  /*     } */
  /*     else{ */
  /* 	_com[n]= (-2*j+i)*dims[1] * n_factors*2*2 - j*n_factors*2*2; */
  /* 	_rom[n]= (2*j-i)*dims[1] * n_factors*2*2 + j*n_factors*2*2; */
  /* 	_co_paiurs[n]=(om_pair){-2*j+i,-j}; */
  /*     } */
      
  /*     n++; */
  /*   } */
  /* } */

  cudaMemcpyAsync(_com, com, n_factors* sizeof(i32), cudaMemcpyHostToDevice, stream[2]);
  cudaMemcpyAsync(_rom, rom, n_factors* sizeof(i32), cudaMemcpyHostToDevice, stream[3]);
  cudaMemcpyAsync(_co_pairs, co_pairs, n_factors* sizeof(om_pair), cudaMemcpyHostToDevice, stream[4]);
  /*
  printf("%d %d %x\n",dims[0],dims[1], targs->refimg);
  for (x=0;x<dims[0];x++){
    for (y=0;y<dims[1];y++) {
      printf("co %d %x\n",COORD2(x,y,dims[0],dims[1],1), PyArray_GETPTR2(targs->refimg,x,y));
      _refimg[COORD2(x,y,dims[0],dims[1],1)] = *((i32*)PyArray_GETPTR2(targs->refimg,x,y));
      
    }
    }*/

  //cudaMemcpyAsync(_refimg, refimg, dims[0] * dims[1] * sizeof(i32), cudaMemcpyHostToDevice, stream[5]);
  cudaMemcpyAsync(_RE, RE, 2*n_factors*2*sizeof(f32), cudaMemcpyHostToDevice, stream[5]);
  cudaMemcpyAsync(_CE, CE, 2*n_factors*2*sizeof(f32), cudaMemcpyHostToDevice, stream[6]);
  for (i=0;i<n_streams;i++) {
    cudaStreamSynchronize(stream[i]);
  }  
  dim3 dimGrid(dims[0],dims[1]);
  dim3 factorgrid(n_factors);
		  
  gpu_loopy_F_V__Flow<<<dimGrid, factorgrid, 0, stream[0]>>>(F_V, V_F, RE, NULL, rom, co_pairs, n_factors);
  gpu_loopy_F_V__Fup<<<dimGrid, factorgrid, 0, stream[1]>>>(F_V, V_F, CE, NULL, com, co_pairs, n_factors);

  for (i=0;i<2;i++) {
    cudaStreamSynchronize(stream[i]);
  }

  #define CPU_TEST
  #ifdef CPU_TEST
  /* Temporary, only for CPU TESTING*/
  cudaMemcpy(targs->F_V, F_V, dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32), cudaMemcpyDeviceToHost);
  #endif
  cudaFree(RE);
  cudaFree(CE);
  cudaFree(F_V);
  cudaFree(V_F);
  cudaFree(refimg);
  cudaFree(com);
  cudaFree(rom);
  cudaFree(co_pairs);
  free(_refimg);
  for (i=0;i<n_streams;i++) {
    cudaStreamDestroy(stream[i]);
  }
}

/* Naive method */
__global__ void gpu_loopy_F_V__Flow(f32 *F_V, f32 *V_F, f32 *RE, const i32 * refimg, const i32 * com, const om_pair * co_pairs,  i32 n_factors){
  /* Naive code*/
  i32 x = blockIdx.x;
  i32 y = blockIdx.y;
  i32 n = threadIdx.x;
  //Note, may need to swap blockDim.x and blockDim.y

  /* Check bounds for upper factor */
  om_pair cop = co_pairs[n];
  if ( ! (x+cop.x <0 || x+cop.x >= blockDim.x || y+cop.y < 0 || y+cop.y >=blockDim.y) ){//&& !(refimg[COORD2(x+cop.x,y+cop.y, blockDim.x, blockDim.y, 1)]==0)) {
    i32 origin=COORD3(x,y,0,blockDim.x,blockDim.y,2*n_factors,2);
    i32 co = origin + com[n] + n_factors * 2;
  
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
  if (!(x-cop.x < 0 || x-cop.x >= blockDim.x || y-cop.y < 0 || y-cop.y >=blockDim.y)){// && !(refimg[COORD2(x-cop.x,y-cop.y, blockDim.x, blockDim.y, 1)]==0)) {
    i32 origin=COORD3(x,y,0,blockDim.x,blockDim.y,2*n_factors,2);
    i32 co=origin + rom[n]; //check this
    F_V[co] = CE[n*2] + V_F[origin] > CE[n_factors*2 + n*2] + V_F[origin+1] ?
      CE[n*2] + V_F[origin] : CE[n_factors*2 + n*2] + V_F[origin+1];
  
    F_V[co+1] = CE[n*2+1] + V_F[origin] > CE[n_factors*2 + n*2 + 1] + V_F[origin+1] ?
      CE[n*2+1] + V_F[origin] : CE[n_factors*2 + n*2 + 1] + V_F[origin+1];
  }
  
}

