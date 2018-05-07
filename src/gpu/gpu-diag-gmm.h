#ifndef KALDI_GMM_GPU_DIAG_GMM_H_
#define KALDI_GMM_GPU_DIAG_GMM_H_ 1

#include "gmm/diag-gmm.h"
#include "gpucommons/gpu-matrix.h"
#include "gpucommons/gpu-vector.h"
#include "base/kaldi-math.h"

#include <algorithm>

namespace kaldi{

struct _GPUDiagGmm;
typedef struct _GPUDiagGmm GPUDiagGmm;

}

#endif
