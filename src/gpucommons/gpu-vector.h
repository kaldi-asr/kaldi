#ifndef GPU_VECTOR_H
#define GPU_VECTOR_H

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace kaldi{

template<typename Real>
struct _GPUVector;

template<typename Real>
typedef struct _GPUVector GPUVector;

}

#endif