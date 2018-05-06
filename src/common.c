#include "common.h"

i32 *indlist(i32 n) {
  srand(0);
  i32 *ret=(i32*)malloc(sizeof(i32)*n);
  i32 i;
  for (i=0;i<n;ret[i]=i,i++);
  return ret;
}

void shuffle_inds(i32* arr, i32 size) {
  i32 i,j,tmp;
  for (i=0;i<size;i++) {
    j=rand()%size;
    tmp=arr[j];
    arr[j]=arr[i];
    arr[i]=tmp;
  }
}
