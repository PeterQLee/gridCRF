#include "optimize.h"
lbfgs_t* alloc_lbfgs(i32 m, i32 num_params) {
  lbfgs_t *ret = malloc(sizeof(lbfgs_t));
  ret->start=0;
  ret->num_params = num_params;
  ret-> m = m;
  ret->cur = 0;
  f32 *block=_mm_malloc(sizeof(f32)*(num_params*(m+1)*3+m+1 + num_params*2),32);
  ret->g=block;
  ret->s=&block[num_params*(m+1)];
  ret->y=&block[num_params*(m+1)*2];
  ret->p=&block[num_params*(m+1)*3];
  ret->l_change=&block[num_params*(m+1)*3 + (m+1)];
  ret->ll_change=&block[num_params*(m+1)*3 + (m+1) +num_params];
  return ret;

}
f32* LBFGS(lbfgs_t *params) {


  f32 *y=params->y; // Difference in gradients
  f32 *s = params->s; //past Updates
  f32 *p = params->p; //(scalar)
  i32 num_params = params->num_params , m=params->m;

  f32 *alpha = calloc(num_params,sizeof(f32));

  f32 *z = params->l_change; //link this to the following slot.
  f32 Beta;
  i32 i,j,end,start=params->start;
  f32 den,num;
  if (params->cur == params->m) {
    end=start+m;
  }
  else{
    end=start+params->cur;
  }
  f32 *q=&(params->g[end%(m+1)]); // Gradients
  for (i=0;i<num_params;i++) {
    q[i]=params->g[((end-1)%(m+1))*num_params + i];
  }
	 
  for (i=(end-1)%(m+1);i!=start-1;i=(i-1)%(m+1)) {
    //backwards through time
    //p=1/sum(y[i]*s[i]);
    f32 stot=0;
    for (j=0;j<num_params;j++) {
      stot += s[i*num_params+j] * q[i*num_params+j];
    }
    stot*=p[i];
    //alpha[i]= p[i] * s[i](transpose) * q;
    alpha[i]=stot;//p[i] * stot;
    for (j=0;j<num_params;j++) {
      q[j]=q[j]-alpha[i]*y[i*num_params+j];
    }
  }
  den=0;
  for (j=0;j<num_params;j++){
    den+=y[(start)*num_params+j]*y[(start)*num_params+j];
  }
  num=0;
  for (j=0;j<num_params;j++){
    num+=s[(start)*num_params+j] * q[j];
  }
  for (j=0;j<num_params;j++){
    z[j]=y[(start)*num_params+j]*num/den;
  }
  //H = y[start] * s[start] (transpose)/ y[start](transpose)y[start];
  //z=H*q;
  
  for (i=start;i!=(end)%(m+1);i=(i+1)%(m+1)) {
    Beta=0;
    for (j=0;j<num_params;j++) {
      Beta+= y[i*num_params+j] * z[j];
    }
    Beta*=p[i];
    for (j=0;j<num_params;j++) {
      z[j]= z[j] + s[i*num_params+j]*(alpha[i]-Beta);
    }
  }
  free(alpha);
  return z;
}
