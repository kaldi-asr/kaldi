// cudamatrix/cu-device.cc

// Copyright 2009-2012  Karel Vesely
//                2013  Lucas Ondel
//           2013-2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#if HAVE_CUDA == 1

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <algorithm>
#ifndef _MSC_VER
#include <dlfcn.h>
#endif

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrix.h"
#include "base/kaldi-error.h"
#include "base/kaldi-utils.h"
#include "util/common-utils.h"
#include "util/kaldi-io.h"

namespace kaldi {

/**
   This function was added by Dan in July 2015 after upgrading on the CLSP
   cluster to the CUDA 7.0 toolkit; the old mechanism of just calling
   cudaThreadSynchronize() [==cudaDeviceSynchronize()] and having it
   automagically select a GPU (when exclusive mode is on) doesn't seem to work
   any more, in situations where GPU 0 is already being used.  This works.  It's
   not 100% clear if the fact that the old code wasn't working was a bug, or a
   changed feature (the NVidia docs were never super-clear regarding device
   initialization).  But regardless, changing to this new mechanism should be
   harmless even if the problem was specific to the CLSP grid.
*/

static bool GetCudaContext(int32 num_gpus, std::string *debug_str) {

  // Our first attempt to get a device context is: we do cudaFree(0) and see if
  // that returns no error code.  If it succeeds then we have a device
  // context.  Apparently this is the canonical way to get a context.
  if (cudaFree(0) == 0) {
    cudaGetLastError();  // Clear any error status.
    return true;
  }

  // The rest of this code represents how we used to get a device context, but
  // now its purpose is mainly a debugging one.
  std::ostringstream debug_stream;
  debug_stream << "num-gpus=" << num_gpus << ". ";
  for (int32 device = 0; device < num_gpus; device++) {
    cudaSetDevice(device);
    cudaError_t e = cudaFree(0);  // CUDA context gets created here.
    if (e == cudaSuccess) {
      if (debug_str)
        *debug_str = debug_stream.str();
      cudaGetLastError();  // Make sure the error state doesn't get returned in
                           // the next cudaGetLastError().
      return true;
    }
    debug_stream << "Device " << device << ": " << cudaGetErrorString(e) << ".  ";
  }
  if (debug_str)
    *debug_str = debug_stream.str();
  return false;
}

/**
 * SelectGpuId(use_gpu)
 *
 * There are 3 'use_gpu' modes for GPU selection:
 * "yes"      -- Select GPU automatically (or get one by exclusive mode)
 *               and die if this fails.
 * "optional" -- Do as above, but if it fails, back off to CPU.
 * "no"       -- Run on CPU.
 *
 * In case of Compute exclusive mode, the GPU is selected by OS.
 *
 * Otherwise GPU selection is based on largest proportion of free memory.
 * This can eventually lead to multiple processes computing on single GPU,
 * which is slow. More practical is to use "compute exclusive mode".
 *
 * This method is to be called at the very beginning of the program
 * (before first allocation in cudamatrix), or not at all (default to CPU).
 *
 */
void CuDevice::SelectGpuId(std::string use_gpu) {
  // Possible modes
  if (use_gpu != "yes" && use_gpu != "no" && use_gpu != "optional" && use_gpu != "wait") {
    KALDI_ERR << "Please choose : --use-gpu=yes|no|optional|wait, passed '" << use_gpu << "'";
  }

  // Make sure this function is not called twice!
  if (Enabled()) {
    KALDI_ERR << "There is already an active GPU " << active_gpu_id_
              << ", cannot change it on the fly!";
  }
  // Allow the GPU to stay disabled
  if (!Enabled() && use_gpu == "no") {
    KALDI_LOG << "Manually selected to compute on CPU.";
    return;
  }

  // Check that we have a gpu available
  int32 num_gpus = 0;

  cudaError_t e = cudaGetDeviceCount(&num_gpus);

  if (num_gpus == 0) {
    if (use_gpu == "yes" || use_gpu == "wait") {
      KALDI_CUDA_ERR(e, "No CUDA GPU detected!");
    }
    if (use_gpu == "optional") {
      KALDI_WARN << "Running on CPU!!! No CUDA GPU detected...";
      return;
    }
  }

  // Create a CUDA context.
  std::string debug_str;
  bool got_context = GetCudaContext(num_gpus, &debug_str);

  if (use_gpu != "wait") {
    if (!got_context) {
      // So far no we don't have context, sleep a bit and retry.
      int32 sec_sleep = (use_gpu == "yes" ? 20 : 2);
      KALDI_WARN << "Will try again to get a GPU after " << sec_sleep
                 << " seconds.";
      Sleep(sec_sleep);
      if (!GetCudaContext(num_gpus, &debug_str)) {
        if (use_gpu == "yes") {
          {
            Input input;
            input.Open("nvidia-smi 1>&2 |");
          }
          KALDI_LOG << debug_str;
          KALDI_ERR << "Failed to create CUDA context, no more unused GPUs? ";
        }
        if (use_gpu == "optional") {
          KALDI_WARN << "Running on CPU!!! No more unused CUDA GPUs?";
          return;
        }
      }
    }
  } else {
    int32 num_times = 0;
    BaseFloat wait_time = 0.0;
    while (!got_context) {
      int32 sec_sleep = 5;
      if (num_times == 0)
        KALDI_WARN << "Will try again indefinitely every " << sec_sleep
                   << " seconds to get a GPU.";
      num_times++;
      wait_time += sec_sleep;
      Sleep(sec_sleep);
      got_context = GetCudaContext(num_gpus, NULL);
    }

    KALDI_WARN << "Waited " << wait_time
               << " seconds before creating CUDA context";
  }

  // Re-assure we have the context
  KALDI_ASSERT(cudaSuccess == cudaThreadSynchronize());

  // Check if the machine use compute exclusive mode
  if (IsComputeExclusive()) {
    KALDI_LOG << "CUDA setup operating under Compute Exclusive Mode.";
    FinalizeActiveGpu();
    return;
  } else {
    // Suggest to use compute exclusive mode
    KALDI_WARN << "Not in compute-exclusive mode.  Suggestion: use "
        "'nvidia-smi -c 3' to set compute exclusive mode";
    // We want to choose the device more carefully, so release the CUDA context.
    e = cudaThreadExit(); // deprecated, but for legacy reason not cudaDeviceReset
    if (e != cudaSuccess) {
      KALDI_CUDA_ERR(e, "Failed to release CUDA context on a GPU");
    }

    // And select the GPU according to proportion of free memory
    if (SelectGpuIdAuto()) {
      FinalizeActiveGpu();
      return;
    } else {
      // Could not get GPU, after prevously having the CUDA context?
      // Strange but not impossible...
      if (use_gpu == "yes") {
        KALDI_ERR << "Error acquiring GPU.";
      }
      if (use_gpu == "optional") {
        KALDI_WARN << "Running on CPU!!! Error acquiring GPU.";
        return;
      }
    }
  }
}


void CuDevice::FinalizeActiveGpu() {
  // The device at this point should have active GPU, so we can query its name
  // and memory stats and notify user which GPU is finally used.

  // Get the device-id of active device:
  {
    int32 act_gpu_id;
    cudaError_t e = cudaGetDevice(&act_gpu_id);
    if (e != cudaSuccess) {
      KALDI_CUDA_ERR(e, "Failed to get device-id of active device.");
    }
    // Remember the id of active GPU
    active_gpu_id_ = act_gpu_id; // CuDevice::Enabled() is true from now on
    // Initialize CUBLAS.
    CUBLAS_SAFE_CALL(cublasCreate(&handle_));
    // Initialize the cuSPARSE library
    CUSPARSE_SAFE_CALL(cusparseCreate(&cusparse_handle_));

    // Notify user which GPU is finally used
    char name[128];
    DeviceGetName(name,128,act_gpu_id);

    CU_SAFE_CALL(cudaGetDeviceProperties(&properties_, act_gpu_id));

    KALDI_LOG << "The active GPU is [" << act_gpu_id << "]: " << name << "\t"
              << GetFreeMemory(&free_memory_at_startup_, NULL) << " version "
              << properties_.major << "." << properties_.minor;
  }
  return;
}


bool CuDevice::DoublePrecisionSupported() {
  if (!Enabled()) return true;
  return properties_.major > 1 || (properties_.major == 1 && properties_.minor >= 3);
  // Double precision is supported from version 1.3
}


bool CuDevice::IsComputeExclusive() {
  // assume we already have an CUDA context created
  KALDI_ASSERT(cudaSuccess == cudaThreadSynchronize());

  // get the device-id and its device-properties
  int32 gpu_id = -1;
  cudaError_t e = cudaGetDevice(&gpu_id);
  if (e != cudaSuccess) {
    KALDI_CUDA_ERR(e, "Failed to get current device");
  }
  struct cudaDeviceProp gpu_prop;
  e = cudaGetDeviceProperties(&gpu_prop, gpu_id);
  if (e != cudaSuccess) {
    KALDI_CUDA_ERR(e,  "Failed to get device properties");
  }
  // find out whether compute exclusive mode is used
  switch (gpu_prop.computeMode) {
    case cudaComputeModeExclusive :
      return true;
      break;
#if (CUDA_VERSION >= 4000)
    case cudaComputeModeExclusiveProcess :
      return true;
      break;
#endif
    default :
      // in this case we release the GPU context...
      return false;
  }
}

template<typename TA, typename TB>
bool greater_pair(const std::pair<TA, TB> &left, const std::pair<TA, TB>& right) {
  return left.second > right.second;
}

bool CuDevice::SelectGpuIdAuto() {
  // Check that we have at least one gpu
  int32 num_gpus = 0;
  cudaError_t e = cudaGetDeviceCount(&num_gpus);
  if (num_gpus == 0) {
    KALDI_WARN << "No CUDA devices found";
    if (e != cudaSuccess) {
      KALDI_WARN << "cudaGetDeviceCount() returned " << e
                 <<", meaning: \"" << cudaGetErrorString(e)  << "\"";
    }
    return false;
  }

  // The GPU is selected according to maximal free memory ratio
  std::vector< std::pair<int, float> > free_mem_ratio(num_gpus);

  // Get ratios of memory use, if possible
  KALDI_LOG << "Selecting from " << num_gpus << " GPUs";
  for(int32 n = 0; n < num_gpus; n++) {
    int32 ret = cudaSetDevice(n);
    switch(ret) {
      case cudaSuccess : {
        // create the CUDA context for the thread
        cudaThreadSynchronize(); // deprecated, but for legacy not cudaDeviceSynchronize
        // get GPU name
        char name[128];
        DeviceGetName(name,128,n);
        // get GPU memory stats
        int64 free, total;
        std::string mem_stats;
        mem_stats = GetFreeMemory(&free, &total);
        // log
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << name << "\t" << mem_stats;

        // We have seen that in some cases GetFreeMemory returns zero
        // That will produce nan after division, which might confuse
        // the sorting routine. Or maybe not, but let's keep it clean
        if (total <= 0) {
          KALDI_LOG << "Total memory reported for device " << n << " is zero (or less).";
        }
        float mem_ratio = total > 0 ? free/(float)total : 0;
        free_mem_ratio[n] = std::make_pair(n, mem_ratio);

        // destroy the CUDA context for the thread
        cudaThreadExit(); // deprecated, but for legacy reason not cudaDeviceReset
      } break;

#if (CUDA_VERSION > 3020)
      case cudaErrorDeviceAlreadyInUse :
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << "Device cannot be accessed, used EXCLUSIVE-THREAD mode...";
        break;
#endif
      case cudaErrorInvalidDevice :
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << "Device cannot be accessed, not a VALID CUDA device!";
        break;
      default :
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << "returned " << ret << ", "
                  << cudaGetErrorString((cudaError_t)ret);
    }
  }
  // find GPU with max free memory
  int32 max_id=0;
  std::sort(free_mem_ratio.begin(), free_mem_ratio.end(),
            greater_pair<int, float>);
  // the free_mem_ratio should be bigger than zero
  KALDI_ASSERT(free_mem_ratio[max_id].second > 0.0);

  float dev_id;
  float mem_ratio;
  do {
    // try to select the GPU in the best to worst order
    // Note we have to check the return codes manually, as the CU_SAFE_CALL
    // contains call to KALDI_ERR (which will cause the program to abort)

    dev_id = free_mem_ratio[max_id].first;
    mem_ratio = free_mem_ratio[max_id].second;

    KALDI_LOG << "Trying to select device: " << dev_id << " (automatically), mem_ratio: " << mem_ratio;
    e = cudaSetDevice(dev_id);
    if (e != cudaSuccess) {
      KALDI_WARN << "Cannot select this device: return code " << e
                 << ", Error message: \"" << cudaGetErrorString(e) << "\"";
    } else {
      e = cudaThreadSynchronize(); // deprecated, but for legacy not cudaDeviceSynchronize
      if (e != cudaSuccess) {
        KALDI_WARN << "Cannot select this device: return code " << e
                   << ", Error message: \"" << cudaGetErrorString(e) << "\"";
      }
    }
    max_id++;
  } while ((e != cudaSuccess) && (max_id < free_mem_ratio.size()));

  if (e != cudaSuccess) {
    KALDI_WARN << "Failed to (automatically) select any device";
    return false;
  }
  KALDI_LOG << "Success selecting device " << dev_id << " free mem ratio: " << mem_ratio;
  return true;
}


void CuDevice::AccuProfile(const char *function_name,
                           const CuTimer &timer) {
  if (GetVerboseLevel() >= 1) {
    std::string key(function_name);
    cudaDeviceSynchronize();
    double elapsed = timer.Elapsed();

    if (profile_map_.find(key) == profile_map_.end())
      profile_map_[key] = elapsed;
    else
      profile_map_[key] += elapsed;
  }
}

void CuDevice::PrintMemoryUsage() const {
  if (Enabled()) {
    allocator_.PrintMemoryUsage();
    int64 free_memory_now;
    GetFreeMemory(&free_memory_now, NULL);
    KALDI_LOG << "Memory used (according to the device): "
              << (free_memory_at_startup_ - free_memory_now) << " bytes.";
  }
}

void CuDevice::PrintProfile() {
  if (GetVerboseLevel() >= 1) {
    std::ostringstream os;
    os << "-----\n[cudevice profile]\n";
    unordered_map<std::string, double, StringHasher>::iterator it;
    std::vector<std::pair<double, std::string> > pairs;
    double total_time = 0.0;
    for(it = profile_map_.begin(); it != profile_map_.end(); ++it) {
      std::string function_name = it->first;
      double elapsed_time = it->second;
      total_time += elapsed_time;
      pairs.push_back(std::make_pair(elapsed_time, function_name));
    }
    // display from shortest to longest time, so tail will show the longest
    // times at the end.
    std::sort(pairs.begin(), pairs.end());
    size_t max_print = 15, start_pos = (pairs.size() <= max_print ?
                                        0 : pairs.size() - max_print);
    for (size_t i = start_pos; i < pairs.size(); i++)
      os << pairs[i].second << "\t" << pairs[i].first << "s\n";
    os << "Total GPU time:\t" << total_time << "s (may involve some double-counting)\n";
    os << "-----";
    KALDI_LOG << os.str();
    PrintMemoryUsage();
  }
}


std::string CuDevice::GetFreeMemory(int64* free, int64* total) const {
  // WARNING! the CUDA API is inconsistent accross versions!
#ifdef _MSC_VER
  size_t mem_free, mem_total;
  cuMemGetInfo_v2(&mem_free, &mem_total);
#else
#if (CUDA_VERSION >= 3020)
  // define the function signature type
  size_t mem_free, mem_total;
#else
  unsigned int mem_free, mem_total;
#endif
  {
    // we will load cuMemGetInfo_v2 dynamically from libcuda.so
    // pre-fill ``safe'' values that will not cause problems
    mem_free = 1; mem_total = 1;
    // open libcuda.so
    void* libcuda = dlopen("libcuda.so",RTLD_LAZY);
    if (NULL == libcuda) {
      KALDI_WARN << "cannot open libcuda.so";
    } else {
      // define the function signature type
      // and get the symbol
#if (CUDA_VERSION >= 3020)
      typedef CUresult (*cu_fun_ptr)(size_t*, size_t*);
      cu_fun_ptr dl_cuMemGetInfo = (cu_fun_ptr)dlsym(libcuda,"cuMemGetInfo_v2");
#else
      typedef CUresult (*cu_fun_ptr)(int*, int*);
      cu_fun_ptr dl_cuMemGetInfo = (cu_fun_ptr)dlsym(libcuda,"cuMemGetInfo");
#endif
      if (NULL == dl_cuMemGetInfo) {
        KALDI_WARN << "cannot load cuMemGetInfo from libcuda.so";
      } else {
        // call the function
        dl_cuMemGetInfo(&mem_free, &mem_total);
      }
      // close the library
      dlclose(libcuda);
    }
  }
#endif
  // copy the output values outside
  if (NULL != free) *free = mem_free;
  if (NULL != total) *total = mem_total;
  // prepare the text output
  std::ostringstream os;
  os << "free:" << mem_free/(1024*1024) << "M, "
     << "used:" << (mem_total-mem_free)/(1024*1024) << "M, "
     << "total:" << mem_total/(1024*1024) << "M, "
     << "free/total:" << mem_free/(float)mem_total;
  return os.str();
}


void CuDevice::DeviceGetName(char* name, int32 len, int32 dev) {
  // prefill with something reasonable
  strncpy(name,"Unknown GPU",len);
#ifdef _MSC_VER
  cuDeviceGetName(name, len, dev);
#else
  // open libcuda.so
  void* libcuda = dlopen("libcuda.so",RTLD_LAZY);
  if (NULL == libcuda) {
    KALDI_WARN << "cannot open libcuda.so";
  } else {
    // define the function signature type
    typedef CUresult (*cu_fun_ptr)(char*,int,CUdevice);
    // get the symbol
    cu_fun_ptr cuDeviceGetName_ptr = (cu_fun_ptr)dlsym(libcuda,"cuDeviceGetName");
    if (NULL == cuDeviceGetName_ptr) {
      KALDI_WARN << "cannot load cuDeviceGetName from libcuda.so";
    } else {
      // call the function
      cuDeviceGetName_ptr(name, len, dev);
    }
    // close the library
    dlclose(libcuda);
  }
#endif
}


void CuDevice::CheckGpuHealth() {
  if (!Enabled()) return;
  CuTimer t;
  // prepare small matrices for a quick test
  Matrix<BaseFloat> a(50, 100);
  Matrix<BaseFloat> b(100 ,50);
  a.SetRandn();
  b.SetRandUniform();
  // multiply 2 small matrices in CPU:
  Matrix<BaseFloat> c(50, 50);
  c.AddMatMat(1.0, a, kNoTrans, b, kNoTrans, 0.0);
  // multiply same matrices in GPU:
  CuMatrix<BaseFloat> c1(50, 50);
  c1.AddMatMat(1.0, CuMatrix<BaseFloat>(a), kNoTrans, CuMatrix<BaseFloat>(b), kNoTrans, 0.0);
  // check that relative differnence is <1%
  AssertEqual(c, Matrix<BaseFloat>(c1), 0.01);
  // measure time spent in this check
  AccuProfile(__func__, t);
}

CuDevice::CuDevice() :
    active_gpu_id_(-1), debug_stride_mode_(false),
    num_debug_stride_allocations_(0), allocator_(CuAllocatorOptions()),
    multi_threaded_(false) { }


// The instance of the static singleton
CuDevice CuDevice::global_device_;
}


#endif // HAVE_CUDA
