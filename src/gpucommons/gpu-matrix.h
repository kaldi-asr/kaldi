#ifndef KALDI_GPUCOMMONS_GPU_MATRIX_H
#define KALDI_GPUCOMMONS_GPU_MATRIX_H

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "matrix/kaldi-matrix.h"
#include "base/kaldi-common.h"

namespace kaldi{

template<typename Real>
struct _GPUMatrix;

template<typename Real>
using GPUMatrix = _GPUMatrix<Real>;

}

#endif
