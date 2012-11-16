// cudamatrix/cu-devide.cc

// Copyright 2009-2012  Karel Vesely

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



#if HAVE_CUDA==1

#include <cublas.h>
#include <cuda.h>

#include <vector>

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "base/kaldi-error.h"


namespace kaldi {

CuDevice::CuDevice()
 : active_gpu_id_(-1), verbose_(true) {
  //get number of devices
  int N_GPU = 0;
  cudaGetDeviceCount(&N_GPU);
  //look which GPUs are available, get free memory stats
  if(N_GPU > 0) {
#if (CUDA_VERSION > 3020)
    // First check if operating under Compute Exclusive Mode:
    int32 gpu_id = -1;
    cudaGetDevice(&gpu_id);
    cudaDeviceProp gpu_prop;
    cudaGetDeviceProperties(&gpu_prop, gpu_id);
    if (gpu_prop.computeMode == cudaComputeModeExclusive
        || gpu_prop.computeMode == cudaComputeModeExclusiveProcess) {
      cudaDeviceSynchronize();
      char gpu_name[128];
      cuDeviceGetName(gpu_name, 128, gpu_id);
      std::string mem_stats = GetFreeMemory(NULL, NULL);
      KALDI_LOG << "CUDA setup operating under Compute Exclusive Mode.\n"
                << "  Using device " << gpu_id << ": " << gpu_name << "\t" << mem_stats;
      active_gpu_id_ = gpu_id;
      cuSafeCall(cublasInit());
      return;
    }
#endif
    // If not operating under Compute Exclusive Mode, or using a version of CUDA
    // where such a check cannot be performed, select the GPU with most free memory.
    std::vector<float> free_mem_ratio(N_GPU+1, 0.0);
    //get ratios of memory use, if possible
    KALDI_LOG << "Selecting from " << N_GPU << " GPUs";
    for(int32 n=0; n<N_GPU; n++) {
      int32 ret = cudaSetDevice(n);
      switch(ret) {
        case cudaSuccess : {
          //create the CUDA context for the thread
          cudaThreadSynchronize(); //deprecated, but for legacy reason...
          //get GPU name
          char name[128];
          cuDeviceGetName(name,128,n);
          //get GPU memory stats
          int64 free, total;
          std::string mem_stats;
          mem_stats = GetFreeMemory(&free, &total);
          //log
          KALDI_LOG << "cudaSetDevice(" << n << "): "
                    << name << "\t" << mem_stats;
          //store the free/total ratio
          free_mem_ratio[n] = free/(float)total;
          //destroy the CUDA context for the thread
          cudaThreadExit(); //deprecated, but for legacy reason...
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
                    << "returned " << ret << ", " << cudaGetErrorString((cudaError_t)ret);
      }
//      //reset the error state to cudaSuccess
//      cudaGetLastError();
    }
    //find GPU with max free memory
    int32 max_id=0;
    for(int32 n=1; n<free_mem_ratio.size(); n++) {
      if(free_mem_ratio[n] > free_mem_ratio[max_id]) max_id=n;
    }
    //finally select the GPU
    if(free_mem_ratio[max_id] > 0.0) {
      KALDI_LOG << "Selected device: " << max_id << " (automatically)";
      cuSafeCall(cudaSetDevice(max_id));
      active_gpu_id_ = max_id;
      //initialize the CUBLAS
      cuSafeCall(cublasInit());
    } else {
      KALDI_WARN << "CUDA will NOT be used!!! None of the " << N_GPU << " devices could be selected...";
    }
  } else {
    KALDI_WARN << "CUDA will NOT be used!!! No CUDA capable GPU detected...";
  }
}



CuDevice::~CuDevice() {
  if (Enabled()) {
    cuSafeCall(cublasShutdown());
  } else {
    KALDI_WARN << "CUDA was NOT used!";
  }
}



void CuDevice::SelectGpuId(int32 gpu_id) {
  //release the CUBLAS and CUDA context, if any
  if(Enabled()) {
    cuSafeCall(cublasShutdown());
    cudaThreadExit(); //deprecated, but for legacy reason...
    active_gpu_id_ = -1;
  }
  //allow manual GPU disable
  if(gpu_id == -1) {
    KALDI_LOG << "Selected device: " << gpu_id << " (manually disabling GPU)";
    return;
  }
  //try to select the desired GPU
  int32 ret = cudaSetDevice(gpu_id);
  //handle the possible errors (no recovery!!!)
  switch(ret) {
    case cudaSuccess :
      //remember the id of active GPU 
      active_gpu_id_ = gpu_id;
      //initialize the CUBLAS
      cuSafeCall(cublasInit());
      KALDI_LOG << "Selected device: " << gpu_id << " (manual override...)";
      return; //we are done!
#if (CUDA_VERSION > 3020)
    case cudaErrorDeviceAlreadyInUse :
      KALDI_ERR << "cudaSetDevice(" << gpu_id << "): "
                << "Device cannot be accessed, used EXCLUSIVE-THREAD mode...";
      break;
#endif
    case cudaErrorInvalidDevice :  
      KALDI_ERR << "cudaSetDevice(" << gpu_id << "): "
                << "Device cannot be accessed, not a VALID CUDA device!";
      break;
    default :  
      KALDI_ERR << "cudaSetDevice(" << gpu_id << "): "
                << "returned " << ret << ", " << cudaGetErrorString((cudaError_t)ret);
  }
//  //reset the error state to cudaSuccess
//  cudaGetLastError();
}



void CuDevice::AccuProfile(const std::string &key, double time) { 
  if (profile_map_.find(key) == profile_map_.end()) {
    profile_map_[key] = 0.0;
  }
  profile_map_[key] += time;
}



void CuDevice::PrintProfile() {
  if (verbose_ && Enabled()) { 
    std::ostringstream os;
    os << "-----\n[cudevice profile]\n";
    std::map<std::string, double>::iterator it;
    for(it = profile_map_.begin(); it != profile_map_.end(); ++it) {
      os << it->first << "\t" << it->second << "s\n";
    }
    os << "-----";
    KALDI_LOG << os.str();
  }
}


std::string CuDevice::GetFreeMemory(int64* free, int64* total) {
// WARNING! the CUDA API is inconsistent accross versions!
#if (CUDA_VERSION >= 3020)
  size_t mem_free, mem_total;
#else
  unsigned int mem_free, mem_total;
#endif
  // get the free memory stats
  cuMemGetInfo(&mem_free, &mem_total);
  // post them outside
  if(NULL != free) *free = mem_free;
  if(NULL != total) *total = mem_total;
  // prepare the text output
  std::ostringstream os;
  os << "free:" << mem_free/(1024*1024) << "M, "
     << "used:" << (mem_total-mem_free)/(1024*1024) << "M, "
     << "total:" << mem_total/(1024*1024) << "M, " 
     << "free/total:" << mem_free/(float)mem_total;
  return os.str();
}



////////////////////////////////////////////////
// The instance of the static singleton 
//
CuDevice CuDevice::msDevice;
//
////////////////////////////////////////////////



}


#endif // HAVE_CUDA
