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

#include "train_cpu.h"

//backend functions
/* 
Does straight up gradient descent

Proecedure is the following:
1. Allocate variables
2. organize which threads handle which parts of the array
3. For each epoch; calculate the estimated labels
4. For each epoch; calculate the change based on cross entropy
5. In each epoch, update the parameters.
 */


static void RMSprop_swap_vstore(rmsprop_t *rmsp, i32 index, i32 n_params) {
  /* modify DS to point to sample's past gradient*/
  rmsp->vstore[0] = rmsp->vstore_agg[index*2];
  rmsp->vstore[1] = rmsp->vstore_agg[index*2+1];
}


static void RMSprop_update(rmsprop_t *rmsp, f32 *V, f32 *V_change, i32 n_factors, i32 n_unary) {
  //TODO: make it so vstore is unique for each example
  f32 *v = rmsp->vstore[rmsp->current_offset];
  f32 *v_old = rmsp->vstore[(rmsp->current_offset^1)];
  #ifdef __AVX__
  __m256 r1, gamma_256, invgamma_256, change, r4, alpha_256;
  #elif __SSE__
  __m128 r1, gamma_128, invgamma_128, change, r4, alpha_128;
  #endif
  
  i32 upper = n_factors*4*2+n_unary, i,j;

  #ifdef __AVX__
  gamma_256 = _mm256_set1_ps(rmsp->gamma);
  invgamma_256 = _mm256_set1_ps(1.0f);
  invgamma_256 = _mm256_sub_ps(invgamma_256,gamma_256);
  alpha_256 = _mm256_set1_ps(rmsp->alpha);
  #elif __SSE__
  gamma_128 = _mm_set1_ps(rmsp->gamma);
  invgamma_128 = _mm_set1_ps(1.0f);
  invgamma_128 = _mm_sub_ps(invgamma_128,gamma_128);
  alpha_128 = _mm_set1_ps(rmsp->alpha);
  #endif

  #ifdef __AVX__
  for (i=0;i<8*(upper/8);i+=8) {
    /* calculate change */
    r1 = _mm256_load_ps(&v_old[i]);
    r1 = _mm256_mul_ps(r1,gamma_256);

    change = _mm256_load_ps(&V_change[i]);
    r4 = _mm256_mul_ps(change,change);
    r4 = _mm256_mul_ps(invgamma_256,r4);
    
    r1 = _mm256_add_ps(r1,r4); // this is the new v
    _mm256_store_ps(&v[i], r1); //make this the v change for the next run.

    //Now do the update step
    r1 = _mm256_rsqrt_ps(r1); //Issue here is probably from r1 being 0.
    r1 = _mm256_mul_ps(alpha_256, r1);
    r1 = _mm256_mul_ps(r1, change);
    r4 = _mm256_load_ps(&V[i]);
    r4 = _mm256_add_ps(r4, r1);

    // TODO: find a way to vectorize this
    for (j=0;j<8;j++) {
      if (fabs(r1[i]) > rmsp->stop_tol){
	*(rmsp->converged) = 0;
      }
    }

    // write to memory
    _mm256_store_ps(&V[i], r4);
    assert(!isnan(V[i]));
  }
  for (i=8*(upper/8);i<upper;i++) {
    v[i] = v_old[i]*rmsp->gamma + (1.0-rmsp->gamma)*V_change[i]*V_change[i];
    V[i] += (rmsp->alpha)/sqrt(v[i])*V_change[i];
    if (fabs((rmsp->alpha)/sqrt(v[i])*V_change[i]) > rmsp->stop_tol){
      *(rmsp->converged) = 0;
    }
  }
  #elif __SSE__
  for (i=0;i<4*(upper/4);i+=4) {
    /* calculate change */
    r1 = _mm_load_ps(&v_old[i]);
    r1 = _mm_mul_ps(r1,gamma_128);

    change = _mm_load_ps(&V_change[i]);
    r4 = _mm_mul_ps(change,change);
    r4 = _mm_mul_ps(invgamma_128,r4);
    
    r1 = _mm_add_ps(r1,r4); // this is the new v
    _mm_store_ps(&v[i], r1); //make this the v change for the next run.

    //Now do the update step
    r1 = _mm_rsqrt_ps(r1); //Issue here is probably from r1 being 0.
    r1 = _mm_mul_ps(alpha_128, r1);
    r1 = _mm_mul_ps(r1, change);
    r4 = _mm_load_ps(&V[i]);
    r4 = _mm_add_ps(r4, r1);

    // TODO: find a way to vectorize this
    for (j=0;j<4;j++) {
      if (fabs(r1[i]) > rmsp->stop_tol){
	*(rmsp->converged) = 0;
      }
    }

    // write to memory
    _mm_store_ps(&V[i], r4);
    assert(!isnan(V[i]));
  }
  for (i=4*(upper/4);i<upper;i++) {
    v[i] = v_old[i]*rmsp->gamma + (1.0-rmsp->gamma)*V_change[i]*V_change[i];
    V[i] += (rmsp->alpha)/sqrt(v[i])*V_change[i];
    if (fabs((rmsp->alpha)/sqrt(v[i])*V_change[i]) > rmsp->stop_tol){
      *(rmsp->converged) = 0;
    }
  }
  #endif
}


void grad_descent(gradient_t *args,i64 epochs,i64 n_threads) {
  /*TODO: 
    Find out if we can avoid doing multiple loopy BPs by reusing the energies from the last iteration
*/
  #define L2 0
  #define LAMBDA 0.001
  
  i64 h,i,j,k;
  //PyArrayObject *EY;
  f32 *unary=(args->self)->unary;
  f32 *V=(args->self)->V_data;
  pthread_t *threads= malloc(sizeof(pthread_t)*n_threads);
  npy_intp *dims=args->dims;
  

  i32 n_factors=args->n_factors;
  i32 n_unary = args->n_unary;
  i32 n_params = n_factors*8 + n_unary;
  
  gradient_t *targs=malloc(sizeof(gradient_t) * n_threads);
  npy_intp *start= malloc(sizeof(npy_intp)*n_threads*2);
  npy_intp *stop= malloc(sizeof(npy_intp)*n_threads*2);
  PyObject *X_list=args->X_list;

  
  npy_intp s0;

  /* Delegate which threads handle what*/
  


  //const f32 lr=0.01;
  f32 totL;
  i32 n_samples = PyList_Size(X_list);

  /* Allocate memory for threads */
  f32 **V_change_l = (f32**) malloc(sizeof(f32*) * n_threads);
  f32 **mu_l = (f32 **) malloc(sizeof(f32*) * n_samples);
  i32 **EY_l = (i32 **) malloc(sizeof(i32*) * n_samples);


  for (j=0;j<n_samples;j++){
    dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,j));
    mu_l[j] = (f32*) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
    EY_l[j] = (i32*) _mm_malloc(sizeof(i32)*dims[0]*dims[1],32); //These aren't getting set somewhere...
    

  }
  
  for (h=0;h<n_threads;h++ ){
    V_change_l[h] = _mm_malloc(sizeof(f32)*(n_params),32);
  }

  /* Process error function data */
  
  srand(0);
  i32 *inds = indlist(n_samples);
  void **error_data = (void**) malloc(sizeof(void*)*n_samples);
  f32 *prod, *sum, *tmpprob;
  cpu_dice_data_t *dice_error_tmp;
  
  switch(args->error_func) {
  case DICE:
    prod = (f32*) malloc(sizeof(f32)*4);
    sum = &prod[2];
    pthread_mutex_t *sumlocks = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t)*n_threads);
    pthread_mutex_t *prodlocks = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t)*n_threads);
    #ifndef __APPLE__
    pthread_barrier_t sync_sum;
    pthread_barrier_init(&sync_sum,NULL,1);

    #else
    i32 sync_sum; //TODO: fix this
    #endif
    
    for (h=0;h<n_threads;h++) {
      pthread_mutex_init(&sumlocks[h], NULL);
      pthread_mutex_init(&prodlocks[h], NULL);
    }

    for (j=0;j<n_samples;j++) {
      dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,j));
      dice_error_tmp = (cpu_dice_data_t*) malloc(sizeof(cpu_dice_data_t));
      tmpprob = (f32*) malloc(sizeof(f32)* dims[0]*dims[1]*2);
      dice_error_tmp->prod = prod;
      dice_error_tmp->sum = sum;
      dice_error_tmp->prob = tmpprob;
      dice_error_tmp->sumlock = sumlocks;
      dice_error_tmp->prodlock = prodlocks;
      dice_error_tmp->sync_sum = &sync_sum;
      
      error_data[j] = (void*) dice_error_tmp;
    }
    
    break;
  case ENTROPY:
    break;
  }
  f32 lr, scale, stop_tol = args->stop_tol;
  /* 
     Update method data
   */
  void *update_data;
  i32 converged = 0;
  switch(args->update_type) {
  case RMSPROP:

      update_data = malloc(sizeof(rmsprop_t));
      rmsprop_t *rmstmp = (rmsprop_t*) update_data;
      //rmstmp->vstore = (f32*) _mm_malloc(sizeof(f32)*2*(n_factors*4*2+args->n_unary),32);
      rmstmp->vstore = (f32**) malloc(sizeof(f32*)*2);
      rmstmp->vstore_agg = (f32*) malloc(sizeof(f32*)*n_samples*2);
      rmstmp->gamma = args->gamma;
      rmstmp->alpha = args->alpha;
      rmstmp->current_offset = 0;
      rmstmp->stop_tol = args->stop_tol;
      rmstmp->converged = &converged;
      for (i=0;i<n_samples;i++){
	rmstmp->vstore_agg[i*2] = (f32*) _mm_malloc(sizeof(f32)*(n_factors*4*2+n_unary),64);
	rmstmp->vstore_agg[i*2+1] = (f32*) _mm_malloc(sizeof(f32)*(n_factors*4*2+n_unary),64); 

	for (j=0;j<(n_factors*4*2+n_unary);j++) {
	  rmstmp->vstore_agg[i*2][j]=0.01f;
	  rmstmp->vstore_agg[i*2+1][j]=0.01f;
	}
      }

  break;
  case SGD:
    update_data = NULL;
   
  break;
  }

  i32 perm_resetflag = args->lpar->reset_flag;
  args->lpar->reset_flag = 1;
  for (i=0;i<epochs && !converged;i++) {
    converged = 1;
    shuffle_inds(inds, n_samples);
    for (j=0;j<n_samples;j++){
      dims=PyArray_DIMS((PyArrayObject*)PyList_GetItem(X_list,inds[j]));
      scale = 1.0/(dims[0]*dims[1]);
      args->dims=dims;
      s0=0;
      // get threads ready
      for (h=0;h<n_threads;h++ ){
	memcpy(&targs[h],args,sizeof(gradient_t));// here!
	targs[h].V_change=V_change_l[h];
	targs[h].unary_change = &(V_change_l[h][n_factors*4*2]);
	start[2*h]=s0/dims[1];
	start[2*h+1]=s0%dims[1];
	
	s0+=dims[0]*dims[1]/n_threads;
	
	stop[2*h]=s0/dims[1];
	stop[2*h+1]=s0%dims[1];
	targs[h].start=&start[2*h];
	targs[h].stop=&stop[2*h];
	targs[h].instance_index= &inds[j];
	targs[h].lpar->mu=mu_l[inds[j]];
	targs[h].lpar->EY=EY_l[inds[j]];
	targs[h].error_data = error_data[inds[j]];
	targs[h].scale = scale;
	
      }
      switch (args->error_func) {
      case DICE:
	((cpu_dice_data_t*)error_data[inds[j]])->sum[0] = 0.0f;
	((cpu_dice_data_t*)error_data[inds[j]])->sum[1] = 0.0f;
	((cpu_dice_data_t*)error_data[inds[j]])->prod[0] = 0.0f;
	((cpu_dice_data_t*)error_data[inds[j]])->prod[1] = 0.0f;
	break;
      case ENTROPY:
	break;
      }      
      stop[n_threads*2-2]=dims[0];
      stop[n_threads*2-1]=dims[1];

      args->lpar->mu=mu_l[inds[j]];//redundant
      args->lpar->EY=EY_l[inds[j]];

      
      // Do a loop iteration to get estimated outcomes with this parameterization
      (*args->loopy_func)(args->self, (PyArrayObject *) (PyArrayObject*)PyList_GetItem(X_list,inds[j]), args->lpar, NULL);
      
      for (h=0;h<n_threads;h++) {
	pthread_create(&threads[h], NULL, (void*) _calculate_gradient, &targs[h]);
      }
      for (h=0;h<n_threads;h++) {
	pthread_join(threads[h], NULL);
      }
      
      totL=0.0f;
      /* update params */
      lr = args->alpha;
      switch(args->update_type){
      case RMSPROP:
	//summate gradients
	for (h=1;h<n_threads;h++) {
	  for (k=0;k<n_factors*4*2+n_unary;k++){
	    targs[0].V_change[k]+=lr*targs[h].V_change[k];
	  }
	  totL+=targs[h].L/n_threads;
	}
	RMSprop_swap_vstore((rmsprop_t*) update_data, j, 4*2*n_factors+n_unary);
	RMSprop_update((rmsprop_t*) update_data, V, targs[0].V_change, n_factors, n_unary);
	break;
      case SGD:
	//defaults to gradient descent
#if L2
      for (h=0;h<n_threads;h++) {
	for (k=0;k<n_factors*4*2;k++){
	  V[k]+=lr*targs[h].V_change[k] - lr * LAMBDA * V[k] ;
	}
	for (k=0;k<n_unary;k++){
	  unary[k]+=lr*targs[h].V_change[n_factors*4*2 + k] - lr * LAMBDA * unary[k];
	}
	totL+=targs[h].L/n_threads;
      }
#else
      for (h=0;h<n_threads;h++) {
	for (k=0;k<n_factors*4*2;k++){
	  V[k]+=lr*targs[h].V_change[k];
	  if (fabs(lr*targs[h].V_change[k])>stop_tol){
	    converged = 0;
	  }
	}
	for (k=0;k<n_unary;k++){
	  unary[k]+=lr*targs[h].V_change[n_factors*4*2 + k];
	  if (fabs(lr*targs[h].V_change[n_factors*4*2 + k])>stop_tol){
	    converged = 0;
	  }
	}
	
	totL+=targs[h].L/n_threads;
      }
#endif
      }
      printf("TOTL %f, epoch = %d\n",totL,i);
    

    }
    ((rmsprop_t*)update_data)->current_offset= (((rmsprop_t*)update_data)->current_offset ^ 1);
    args->lpar->reset_flag = perm_resetflag;
    //TODO: check tolerances on each  data.
    // also change rmsprop to save gradients for each data sample
  }
  switch(args->error_func) {
  case DICE:
    free(((cpu_dice_data_t*)error_data[0])->prod);
    free(((cpu_dice_data_t*)error_data[0])->prodlock);
    free(((cpu_dice_data_t*)error_data[0])->sumlock);
    for (j=0;j<n_samples;j++) {
      free(((cpu_dice_data_t*)error_data[j])->prob);
      free((cpu_dice_data_t*)error_data[j]);
    }
    free(error_data);
    break;
  case ENTROPY:
    break;
  }

  switch(args->update_type) {
  case RMSPROP:
    for (i=0;i<n_samples;i++){
      _mm_free(((rmsprop_t*)update_data)->vstore_agg[i*2]);
      _mm_free(((rmsprop_t*)update_data)->vstore_agg[i*2+1]);
    }
    free(((rmsprop_t*)update_data)->vstore_agg);
    free(((rmsprop_t*)update_data)->vstore);
    free(update_data);
    break;
  case SGD:
    break;
  }
  
  for (j=0;j<n_samples;j++) {
    _mm_free(mu_l[j]);
    _mm_free(EY_l[j]);
  }
  free(mu_l);
  free(EY_l);
  for (h=0;h<n_threads;h++ ){
    _mm_free(V_change_l[h]);
  }
  free(V_change_l);
  
  free(targs);
  free(stop);
  free(start);
  free(threads);

}




static void _compute_unary(f32 *yv, f32 *unary, f32 *tmp, i32 n_chan) {
  i32 i;
  for (i=0;i<n_chan; i++) {
    yv[0] += unary[i]*tmp[i];
    yv[1] += unary[n_chan+i]*tmp[i];
  }
}

static void _compute_change_unary(f32 *change, f32 *unary_change, f32 *tmp, i32 n_chan) {
  i32 i;
  for (i=0;i<n_chan; i++) {
    unary_change[i] += change[0]*tmp[i];
    unary_change[n_chan+i] += change[1]*tmp[i];
  }
  
}

static void* _calculate_gradient(gradient_t *args) {
  i32 i,j;
  PyArrayObject *X= (PyArrayObject*)PyList_GetItem(args->X_list,*(args->instance_index));
  PyArrayObject *Y= (PyArrayObject*)PyList_GetItem(args->Y_list,*(args->instance_index));
  f32 *unary= args->self->unary;
  i32 *ainc=args->ainc, *binc=args->binc;
  f32 * V=args->self->V_data;
  npy_intp *start=args->start, *stop=args->stop;
  f32 *V_change = args->V_change;
  f32 *unary_change = args->unary_change;
  f32 L=0.0f,max=0.0f,den;
  f32 yv [2];
  f32 change[2];
  f32 *v;
  i32 *l;
  npy_intp *dims=args->dims;
  
  i32 n_factors = args->n_factors;
  i32 n_unary = args->n_unary;
  i32 n_params = n_factors * 8 + n_unary;
  i32 n_chan = args->self->n_inp_channels;

  /* Clear change var*/
  memset(V_change, 0, sizeof(f32)*n_params);

  i32 n=0;
  f32 *tmp;
  f32 scale=args->scale;
  printf("dims %ld %ld\n",dims[0],dims[1]);
  
  //PyArrayObject *EY = args->EY;
  i32 *EY = args->lpar->EY;
  //TODO: REFACTOR THIS
  switch (args->error_func) {
  case ENTROPY:
    j=start[1];
    for (i=start[0];i<dims[0];i++) {
      for (;j<dims[1];j++) {
	if (i==stop[0] && j==stop[1]) goto grad_finish;
	*((f64 *)yv)=0.0; // set outome to 0
      
	if (*((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && *((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      
	tmp=(f32*)PyArray_GETPTR3(X,i,j,0);

	_compute_unary(yv, unary, tmp, n_chan);
	  //yv[0]=(unary[0]*tmp[0]+unary[1]*tmp[1]);
	  //yv[1]=(unary[2]*tmp[0]+unary[3]*tmp[1]);

	//l here is the estimated label
	for (n=0;n<n_factors;n++) {
	  if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  
	  //l=(i32*) PyArray_GETPTR2(EY,i+ainc[n],j+binc[n]);

	  l=&EY[COORD2(i+ainc[n],j+binc[n],dims[0],dims[1],1)];
	  v=&V[n*4 + ((*l)&1)*2]; // 4x4 transfer matrix for n
	  //we don't negate V because unary gets negated in the form of RE and CE
	  //We pick the row that corresponds to the outcome
	  //TODO: Improve with SSE

	  //uninitialized
	  yv[0] += v[0]; //(Dependent on function)
	  yv[1] += v[1];
	}
	//do left to right
	for (n=0;n<n_factors;n++) {
	  if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	  //l=(i32*) PyArray_GETPTR2(EY,i+ainc[n+n_factors],j+binc[n+n_factors]);
	  l=&EY[COORD2(i+ainc[n+n_factors],j+binc[n+n_factors],dims[0],dims[1],1)];

	  v=&V[n_factors*4 + n*4 + 2*((*l)&1)];

	  yv[0] += v[0];//uninitialized
	  yv[1] += v[1];
	}
	max=-yv[0]>-yv[1]? -yv[0]:-yv[1];
    
	yv[0]=expf(-yv[0]-max);
	yv[1]=expf(-yv[1]-max);
	den=1/(yv[0]+yv[1]);
	yv[0]=yv[0]*den;
	yv[1]=yv[1]*den;

	// l here is the true label
	l=((i32*)PyArray_GETPTR3(Y,i,j,0));
	
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
	change[0] = -scale * (((*l)&1)-yv[0]) ;
	//unary_change[0] += -scale*(((*l)&1)-yv[0])*tmp[0];
	//unary_change[1] += -scale*(((*l)&1)-yv[0])*tmp[1];

	
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
	change[1] = -scale * (((*l)&1)-yv[1]);

	//unary_change[2] += -scale*(((*l)&1)-yv[1])*tmp[0];
	//unary_change[3] += -scale*(((*l)&1)-yv[1])*tmp[1];

	_compute_change_unary(change, unary_change, tmp, n_chan);
	
      
	if (isinf(L)){
	  printf("ISINF\n");
	  //exit(1);
	}

	for (n=0;n<n_factors;n++){
	  if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  //TODO: speed up (SSE) __m128 _mm_add_ps
	  //l=((i32*) PyArray_GETPTR2(EY,i+ainc[n],j+binc[n]));
	  l=&EY[COORD2(i+ainc[n],j+binc[n],dims[0],dims[1],1)];
	  V_change[n*4 + 2*((*l)&1)] += change[0]; //uninitalized
	  V_change[n*4 + 2*((*l)&1) + 1] += change[1]; 

	}
      
	for (n=0;n<n_factors;n++) {
	  if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	  //l=(i32*) PyArray_GETPTR2(EY,i+ainc[n+n_factors],j+binc[n+n_factors]);

	  l=&EY[COORD2(i+ainc[n+n_factors],j+binc[n+n_factors],dims[0],dims[1],1)];
	  V_change[n_factors*4 +n*4 + 2*((*l)&1)] += change[0]; //uninitalized
	  V_change[n_factors*4 +n*4 + 2*((*l)&1) + 1] += change[1]; //uninitliazed

	}
      }
      j=0;

    }
    break;

    /* Error: Dice
       Code for performing dice error minimization
    */
  case DICE:
    j=start[1];
    cpu_dice_data_t *error_data = (cpu_dice_data_t *) args->error_data;
    for (i=start[0];i<dims[0];i++) {
      for (;j<dims[1];j++) {
	if (i==stop[0] && j==stop[1]) goto dice_sync;
	*((f64 *)yv)=0.0; // set outome to 0
      
	if (*((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && *((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
      
	tmp=(f32*)PyArray_GETPTR3(X,i,j,0);
	_compute_unary(yv, unary, tmp, n_chan);
	//yv[0]=(unary[0]*tmp[0]+unary[1]*tmp[1]);
	//yv[1]=(unary[2]*tmp[0]+unary[3]*tmp[1]);

	//l here is the estimated label
	for (n=0;n<n_factors;n++) {
	  if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;
	  
	  //l=(i32*) PyArray_GETPTR2(EY,i+ainc[n],j+binc[n]);

	  l=&EY[COORD2(i+ainc[n],j+binc[n],dims[0],dims[1],1)];
	  v=&V[n*4 + ((*l)&1)*2]; // 4x4 transfer matrix for n
	  //we don't negate V because unary gets negated in the form of RE and CE
	  //We pick the row that corresponds to the outcome
	  //TODO: Improve with SSE

	  //uninitialized
	  yv[0] += v[0]; //(Dependent on function)
	  yv[1] += v[1];
	}
	//do left to right
	for (n=0;n<n_factors;n++) {
	  if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;
	  //l=(i32*) PyArray_GETPTR2(EY,i+ainc[n+n_factors],j+binc[n+n_factors]);
	  l=&EY[COORD2(i+ainc[n+n_factors],j+binc[n+n_factors],dims[0],dims[1],1)];

	  v=&V[n_factors*4 + n*4 + 2*((*l)&1)];

	  yv[0] += v[0];//uninitialized
	  yv[1] += v[1];
	}
	max=-yv[0]>-yv[1]? -yv[0]:-yv[1];
    
	yv[0]=expf(-yv[0]-max);
	yv[1]=expf(-yv[1]-max);
	den=1/(yv[0]+yv[1]);
	yv[0]=yv[0]*den;
	yv[1]=yv[1]*den;

	// l here is the true label
	l=((i32*)PyArray_GETPTR3(Y,i,j,0));
	error_data->prob[COORD2(i,j,dims[0],dims[1],2)] = yv[0];

	//modify values
	//pthread_mutex_lock(&(error_data->sumlock[0]));
	error_data->sum[0] += yv[0]*yv[0]+(*l);
	//pthread_mutex_unlock(&(error_data->sumlock[0]));

	//pthread_mutex_lock(&(error_data->prodlock[0]));
	error_data->prod[0] += yv[0]*(*l);
	//pthread_mutex_unlock(&(error_data->prodlock[0]));
	

	l=((i32*)PyArray_GETPTR3(Y,i,j,1));
	error_data->prob[COORD2(i,j,dims[0],dims[1],2) + 1] = yv[1];
	if (yv[0]) {L=0.0;}
	if (yv[1]) {L=0.0;}
	//pthread_mutex_lock(&(error_data->sumlock[1]));
	error_data->sum[1] += yv[1]*yv[1]+(*(l));
	///pthread_mutex_unlock(&(error_data->sumlock[1]));

	//pthread_mutex_lock(&(error_data->prodlock[1]));
	error_data->prod[1] += yv[1]*(*l);
	//pthread_mutex_unlock(&(error_data->prodlock[1]));


      }
      j=0;
    }
    
  dice_sync:
    //pthread_barrier_wait(&(error_data->sync_sum));

    
    j=start[1];
    f32 dL_dp[2];
    j=start[1];
    for (i=start[0];i<dims[0];i++) {
      for (;j<dims[1];j++) {
	if (i==stop[0] && j==stop[1]) goto grad_finish;
	if (*((i32 *)PyArray_GETPTR3(Y,i,j,0)) == 0  && *((i32 *)PyArray_GETPTR3(Y,i,j,1)) == 0 )continue;
	tmp=(f32*)PyArray_GETPTR3(X,i,j,0);
	yv[0] = error_data->prob[COORD2(i,j,dims[0],dims[1],2)];
	yv[1] = error_data->prob[COORD2(i,j,dims[0],dims[1],2)+1];

	/* Calculate partials w.r.t. p */
	l = (i32*)PyArray_GETPTR3(Y,i,j,0);
	if (yv[0]) {L=0.0;}
	if (yv[1]) {L=0.0;}
	dL_dp[0] = (-2 * (*l) * error_data->sum[0] + 4 * yv[0] * error_data->prod[0]) / (error_data->sum[0] * error_data->sum[0]);

	l = (i32*)PyArray_GETPTR3(Y,i,j,1);

	dL_dp[1] = (-2 * (*l) * error_data->sum[1] + 4 * yv[1] * error_data->prod[1]) / (error_data->sum[1] * error_data->sum[1]);

	
	L += (error_data->prod[0]/error_data->sum[0]) + (error_data->prod[1]/error_data->sum[1]);

	/* Calculate partials w.r.t. V */
	// change for class 0
	//TODO: this may need to be negated
	change[0] = scale * (dL_dp[0] * (yv[0] - yv[0]*yv[0]) + dL_dp[1] * (-yv[0]*yv[1]));
	change[1] = scale * (dL_dp[1] * (yv[1] - yv[1]*yv[1]) + dL_dp[0] * (-yv[1]*yv[0]));
	  
	//TODO: this isn't thread safe
	_compute_change_unary(change, unary_change, tmp, n_chan);
	/* unary_change[0] += change[0] * tmp[0]; */
	/* unary_change[1] += change[0] * tmp[1]; */

	/* unary_change[2] += change[1] * tmp[0]; */
	/* unary_change[3] += change[1] * tmp[1]; */

	for (n=0;n<n_factors;n++){
	  if (i+ainc[n] < 0 || i+ainc[n]>=dims[0] || j+binc[n] < 0 || j+binc[n] >= dims[1]) continue;

	  l=&EY[COORD2(i+ainc[n],j+binc[n],dims[0],dims[1],1)];
	  V_change[n*4 + 2*((*l)&1)] += change[0]; //uninitalized
	  V_change[n*4 + 2*((*l)&1) + 1] += change[1]; 
	}
      
	for (n=0;n<n_factors;n++) {
	  if (i+ainc[n+n_factors] < 0 || i+ainc[n+n_factors]>=dims[0] || j+binc[n+n_factors] < 0 || j+binc[n+n_factors] >= dims[1]) continue;

	  l=&EY[COORD2(i+ainc[n+n_factors],j+binc[n+n_factors],dims[0],dims[1],1)];
	  V_change[n_factors*4 +n*4 + 2*((*l)&1)] += change[0]; 
	  V_change[n_factors*4 +n*4 + 2*((*l)&1) + 1] += change[1]; 
	}
      }
      j=0;
    }
    break;
  }
 grad_finish:
  //Py_DECREF(EY);
  printf("L %f %d %d, %d %d: %d %d\n",L, start[0],start[1],stop[0],stop[1], i,j);
  args->L=L;
  //return L;
  return NULL;
}
