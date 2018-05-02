
//TODO: copy V_data, to allocated V_data
extern "C" void GPU_grad_descent(gradient_t *args,i32 epochs) {
  i32 h,i,j;
  //PyArrayObject *EY;
  const i32 N_UNARY=4;
  f32 *unary=(args->self)->unary;
  f32 *V=(args->self)->V_data;
  pthread_t *threads= malloc(sizeof(pthread_t)*n_threads);
  npy_intp *dims=args->dims;
  
  i32 num_params= args->num_params;
  i64 n_factors=args->n_factors;
  gradient_t *targs=malloc(sizeof(gradient_t) * n_threads);
  npy_intp *start= malloc(sizeof(npy_intp)*n_threads*2);
  npy_intp *stop= malloc(sizeof(npy_intp)*n_threads*2);
  PyObject *X_list=args->X_list;
  f32 *V_change;
  
  i32 s0;
  const f32 lr=0.01;
  f32 totL;
  i32 n_samples = PyList_Size(X_list);


  
  f32 **mu_l = (f32 **) malloc(sizeof(f32*) * n_samples);
  f32 **EY_l = (i32 **) malloc(sizeof(i32*) * n_samples);

  f32 **X_l =  (f32 **) malloc(sizeof(f32*) * n_samples);
  f32 **Y_l =  (i32 **) malloc(sizeof(i32*) * n_samples);

  //cuda streams
  const i32 n_streams = 10;
  cudaStream_t stream[n_streams];
  for (i=0;i<n_streams;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  i32 curstream = 0;

  f32 * V_data;
  cudaMalloc(&V_data, sizeof(f32)*n_factors*8);//check this...
  cudaAsyncMemcpy(V_data, self->V_data, sizeof(f32)*n_factors*8, cudaHostToDevice, stream[(curstream++)%n_streams]);
  //TODO: copy V to V_data
  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS(PyList_GetItem(X_list,j));
    //allocate space
    cudaMalloc(&mu_l[j],dims[0]*dims[1]*2*sizeof(f32));
    
    cudaMalloc(&EY_l[j],sizeof(i32)*dims[0]*dims[1]);

    //Copy images to memory
    cudaAsyncMemcpy(X_l[j], PyList_Get(X_list,j)->data,\
		    sizeof(f32)*dims[0]*dims[1]*2, cudaMemcpyHostToDevice,
		    streams[(curstream++)%n_streams]);
    cudaAsyncMemcpy(Y_l[j], PyList_Get(Y_list,j)->data,\
		    sizeof(i32)*dims[0]*dims[1]*2, cudaMemcpyHostToDevice,\
		    streams[(curstream++)%n_streams]);
  }

  f32 *V_change;
  cudaMalloc(&V_change, sizeof(f32)*(n_factors*4*2+N_UNARY));

  i32 **com_l = (i32**) malloc(sizeof(i32*)*n_samples);
  i32 **rom_l = (i32**) malloc(sizeof(i32*)*n_samples);
  om_pair **co_pairs_l = (om_pair**) malloc(sizeof(om_pair*)*n_samples);
  
  /* Prepare coordinates*/
  for (h=0;h<n_samples;h++) {
    dims=PyArray_DIMS(PyList_GetItem(X_list,h));
    i32 *_com=(i32*) malloc(sizeof(i32)*n_factors);
    i32 *_rom=(i32*) malloc(sizeof(i32)*n_factors);
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

    cudaAsyncMemcpy(&com_l[h], _com, sizeof(i32)*n_factors, cudaMemcpyHostToDevice, streams[(curstream++)%n_streams]);
    cudaAsyncMemcpy(&rom_l[h], _rom, sizeof(i32)*n_factors, cudaMemcpyHostToDevice, streams[(curstream++)%n_streams]);
    cudaAsyncMemcpy(&co_pairs_l[h], _co_pairs, sizeof(om_pair)*n_factors, cudaMemcpyHostToDevice, streams[(curstream++)%n_streams]);
    
     n=0;
  }
  /* End prepare coordinates*/

  f32 *unary_w;
  cudaMalloc(&unary_w, 4 * sizeof(f32));
  cudaAsyncMemcpy(unary_w, self->unary, 4*sizeof(f32), cudaHostToDevice, stream[(curstream++)%n_streams]);

  f32 ** unary_c_l = (f32**) malloc(sizeof(f32*) * n_samples);
  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS(PyList_GetItem(X_list,j));
    cudaMalloc(&unary_c_l[j], dims[0]*dims[1]*2*sizeof(f32));
  }

  f32 *RE, *CE;
  cudaMalloc(&RE, 2* n_factors *2* sizeof(f32));
  cudaMalloc(&CE, 2* n_factors *2* sizeof(f32));

  for (i=0;i<epochs;i++) {
    for (j=0;j<n_samples;j++){
      //Assign the correct arguments, call loopyGPU
      //Make training function..
    }
  }


  


  // copy V_change data back to numpy space...
  // also copy unary data back to numpy space
}
