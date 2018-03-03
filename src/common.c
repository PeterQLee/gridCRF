#include "common.h"

i64 *indlist(i64 n) {
  srand(0);
  i64 *ret=(i64*)malloc(sizeof(i64)*n);
  i64 i;
  for (i=0;i<n;ret[i]=i,i++);
  return ret;
}

void shuffle_inds(i64* arr, i64 size) {
  i64 i,j,tmp;
  for (i=0;i<size;i++) {
    j=rand()%size;
    tmp=arr[j];
    arr[j]=arr[i];
    arr[i]=tmp;
  }
}
