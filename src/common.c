/*
  common.c Common functions

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
