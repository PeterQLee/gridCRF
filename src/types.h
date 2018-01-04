#ifndef TYPES_H
#define TYPES_H

#define INDEX

typedef int i32;
typedef unsigned int u32;
typedef long long i64;
typedef unsigned long long u64;
typedef float f32;
typedef double f64;

typedef struct {
  f32 * d;
  i32 *dims;
  i32 nd;
}f32_arr;

#endif
