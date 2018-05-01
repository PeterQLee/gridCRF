#include "loopy.h"
#include "loopy_gpu.h"

i32* loopyCPU(gridCRF_t* self, PyArrayObject *X,loopy_params_t *lpar,PyArrayObject *refimg){
  //fix type

  //loopy belief propagation using CPU

  //Need to initialize all messages

  npy_intp * dims= PyArray_DIMS(X);
  i64 n_factors=self->n_factors;
  i64 max_it=lpar->max_its,it;
  f32 max_marg_diff=0.0f;
  i32 n_threads = lpar->n_threads;
  i32 converged=0;
  
  f32 * V_data=self->V_data;

  f32 tmp[2];
  

  i64 n,depth=self->depth,i,j,co,h;
  i32 l;
  //f32 * F_V = (f32 *) calloc( dims[0] * dims[1] * (self->n_factors*2) *2, sizeof(f32));
  f32 * F_V = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32);
  f32 * V_F = (f32 *) _mm_malloc( dims[0] * dims[1] * (n_factors*2) *2* sizeof(f32),32);
  i32 *start = malloc(sizeof(i32)*n_threads*2);
  i32 *stop = malloc(sizeof(i32)*n_threads*2);
  
  for (i=0;i<dims[0] * dims[1] * (n_factors*2) *2; i++){
    F_V[i]=0.0f;
    V_F[i]=0.0f;
  }

  f32 * marginals= (f32* ) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
  //f32 * mu= (f32* ) _mm_malloc(dims[0]*dims[1]*2*sizeof(f32),32);
  f32 *mu = lpar->mu;
  for (i=0;i<dims[0]*dims[1]*2;i+=1) {
    mu[i]=BIG;
    marginals[i]=0.0f;
  }
  

  /* Prepare coordinates*/
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


  i32 origin;
  /* transfer matrices */
  f32 *RE= _mm_malloc(2 * n_factors * 2 * sizeof(f32),32); 
  f32 *CE= _mm_malloc(2 * n_factors * 2 * sizeof(f32),32); 
  __m256 r1,r2,r3,r4;


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
  
  }

  pthread_t *threads = malloc(sizeof(pthread_t)*n_threads);
  loopycpu_t *targs = malloc(sizeof(loopycpu_t)*n_threads);
  // set up threads
  npy_intp s0=0;
  for (i=0;i<n_threads;i++ ){

    start[2*i]=s0/dims[1];
    start[2*i+1]=s0%dims[1];
    s0+=dims[0]*dims[1]/n_threads;
    stop[2*i]=s0/dims[1];
    stop[2*i+1]=s0%dims[1];
    
    targs[i].start=&start[2*i];
    targs[i].stop=&stop[2*i];
    targs[i].com=com;
    targs[i].rom=rom;
    targs[i].co_pairs = co_pairs;
    targs[i].X=X;
    targs[i].refimg=refimg;
    targs[i].lpar = lpar;
    targs[i].self = self;
    targs[i].F_V = F_V;
    targs[i].V_F = V_F;
    targs[i].RE = RE;
    targs[i].CE = CE;
    targs[i].marginals = marginals;
    targs[i].mu = mu;
    targs[i].converged = &converged;
    
  }
  for (it=0;it< max_it;it++){
    if (it%10==0){
      printf("it %d\n",it);
    }
    converged = 1;
    
    /* Calculate factor to variable messages */


    
    for (h=0;h<n_threads;h++) {
      pthread_create(&threads[h], NULL, (void*) _loopyCPU__FtoV, &targs[h]);
    }

    for (h=0;h<n_threads;h++) {
      pthread_join(threads[h],NULL);
    }
    
    /* calculate variable to factor messages */
    
    for (h=0;h<n_threads;h++) {
      pthread_create(&threads[h], NULL, (void*) _loopyCPU__VtoF, &targs[h]);
    }

    for (h=0;h<n_threads;h++) {
      pthread_join(threads[h],NULL);
    }

    if (converged) break;
  }
  for (h=0;h<n_threads;h++) {
    pthread_create(&threads[h], NULL, (void*) _loopy_label, &targs[h]);
  }

  for (h=0;h<n_threads;h++) {
    pthread_join(threads[h],NULL);
  }

  
  _mm_free(F_V);
  _mm_free(V_F);
  free(start);
  free(stop);
  _mm_free(marginals);
  free(com);
  free(rom);
  free(co_pairs);
  _mm_free(RE);
  _mm_free(CE);
  free(threads);
  free(targs);
  
  
  return lpar->EY;
}

static void * _loopyCPU__FtoV(loopycpu_t *l_args){
  /* Compute factor to variable messages */

  i32 *start = l_args->start, *stop = l_args->stop;
  gridCRF_t *self = l_args->self;
  PyArrayObject *X = l_args->X;
  PyArrayObject *refimg = l_args->refimg;
  loopy_params_t * lpar = l_args->lpar;
  
  
  f32 a,b;
  npy_intp * dims= PyArray_DIMS(X);
  i64 n_factors=self->n_factors;
  i64 max_it=lpar->max_its,it;
  f32 max_marg_diff=0.0f;
  f32 stop_thresh=lpar->stop_thresh;
  
  f32 * V_data=self->V_data;
  f32 * unary= self->unary;
  
  npy_intp x,y;
  i64 m,n,depth=self->depth,i,j,co;
  i32 l=1;

  f32 *F_V = l_args->F_V;
  f32 *V_F = l_args->V_F;

  /* coordinates */
  i32 *com=l_args->com;
  i32 *rom=l_args->rom;
  om_pair *co_pairs=l_args->co_pairs;
  om_pair cop;
  
  i32 origin;
  f32 *RE = l_args->RE, *CE = l_args->CE;
  __m256 r1,r2,r3,r4;
  
  for (x=start[0];x<dims[0];x++) {
    for (y=start[1];y<dims[1];y++ ){
      if (x==stop[0] && y==stop[1]) goto loopyFtoVstop;
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
	    co=origin+com[m] + 2*(m) + n_factors*2; //what's the 2*(m-n) for?????
	    //co=origin+com[m] + 2*(m-n) + n_factors*2; //what's the 2*(m-n) for?????
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
	    co=origin+rom[m-n_factors] + 2*(m - n_factors);
	    //co=origin+rom[m-n_factors] + 2*(m - n);
	    if (!(co < 0 || co >= dims[0] * dims[1] * (n_factors*2) *2)) {
	      F_V[co] = r3[2*(m-n)];
	      F_V[co+1] = r3[2*(m-n)+1];
	    }
	  }
	}
	//luckily, # factors is guarunteed to be divisible by 4. So no worry about edge cases!
    }


  }

 loopyFtoVstop:
  return NULL;
}

static void* _loopyCPU__VtoF(loopycpu_t *l_args) {
  /* Compute factor to variable messages */
  i32 WARN_FLAG=1;
  i32 i,j;
  i32 *start = l_args->start, *stop = l_args->stop;
  gridCRF_t *self = l_args->self;
  PyArrayObject *X = l_args->X;
  loopy_params_t * lpar = l_args->lpar;
  
  f32 a,b;
  npy_intp * dims= PyArray_DIMS(X);
  i64 n_factors=self->n_factors;
  f32 max_marg_diff=0.0f;
  f32 stop_thresh=lpar->stop_thresh;
  
  f32 * unary = self->unary;
  
  npy_intp x,y;
  i32 n;


  f32 *F_V = l_args->F_V;
  f32 *V_F = l_args->V_F;
  f32 *marginals = l_args->marginals;
  f32 *mu = l_args->mu;

  /* coordinates */

  /* runtime Flags*/
  i32 *converged = l_args->converged;
  
  i32 origin;

  f32 tmp[2];

  __m256 r1,r2;
  /* Compute variable to factor messages */
  max_marg_diff=0;
  for (x=start[0];x<dims[0];x++) {
    for (y=start[1];y<dims[1];y++) {
      if (x==stop[0] && y==stop[1]) goto loopyVtoFstop;
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
 loopyVtoFstop:
  return NULL;
}


void *_loopy_label(loopycpu_t *l_args) {
  loopy_params_t * lpar = l_args->lpar;
  npy_intp * dims= PyArray_DIMS(l_args->X);
  i32 *ret=lpar->EY;
  f32 *mu=l_args->mu;
  i32 * start = l_args->start;
  i32 * stop = l_args->stop;
  i32 x,y;
  i32 origin;
  for (x=start[0];x<dims[0];x++) {
    for (y=start[1];y<dims[1];y++) {
      if (x==stop[0] && y==stop[1]) goto loopyLabelstop;
      origin=COORD2(x,y,dims[0],dims[1],2); // TODO define coord2
      assert(origin >0 && origin + 1 < dims[0]*dims[1]*2);
      if (mu[origin] > mu[origin+1]) {
	ret[COORD2(x,y,dims[0],dims[1],1)]=0;
      }
      else{
	ret[COORD2(x,y,dims[0],dims[1],1)]=1;

      }
    }
  }
 loopyLabelstop:
  return NULL;

}
