#ifndef LOOPY_GPU_H
#define LOOPY_GPU_H
#include "loopy.h"

i32 *loopyGPU(gridCRF_t* self, PyArrayObject *X,loopy_params_t *lpar,PyArrayObject *refimg);


#endif
