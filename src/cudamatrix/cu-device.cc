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

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "base/kaldi-error.h"


namespace kaldi {

CuDevice::CuDevice()
 : enabled_(false), verbose_(true) {
  //get number of devices
  int32 N_GPU = 0;
  cudaGetDeviceCount(&N_GPU);
  //select device if more than one
  if(N_GPU > 1) {
    char name[128];
#if (CUDA_VERSION >= 4000)
    size_t free, total;
#else
    unsigned int free, total;
#endif
    std::vector<float> free_mem_ratio;
    //get ratios of memory use
    KALDI_LOG << "Selecting from " << N_GPU << " GPUs";
    for(int32 n=0; n<N_GPU; n++) {
      cuSafeCall(cudaSetDevice(n));//context created by cuSafeCall(...)
      cuDeviceGetName(name,128,n);
      cuSafeCall(cuMemGetInfo(&free,&total));
      KALDI_LOG << "cudaSetDevice(" << n << "): " <<
                << name << "\t";
                << "free: " << free/1024/1024 << "M, "
                << "total: "<< total/1024/1024 << "M, "
                << "ratio: "<< free/(float)total;
      free_mem_ratio.push_back(free/(float)total);
      cudaThreadExit();//destroy context
    }
    //find GPU with max free memory
    int32 max_id=0;
    for(int32 n=1; n<free_mem_ratio.size(); n++) {
      if(free_mem_ratio[n] > free_mem_ratio[max_id]) max_id=n;
    }
    KALDI_LOG << "Selected device: " << max_id << " (automatically)";
    cuSafeCall(cudaSetDevice(max_id));
  }
    
  if(N_GPU > 0) {
    //initialize the CUBLAS
    cuSafeCall(cublasInit());
    enabled_ = true;
  } else {
    KALDI_WARN << "CUDA will NOT be used!!! The cublasInit() returns: " << ret;
  }
}


CuDevice::~CuDevice() {
  if (enabled_) {
    cuSafeCall(cublasShutdown());
  } else {
    KALDI_WARN << "CUDA was NOT used!";
  }
}


void CuDevice::AccuProfile(const std::string &key, double time) { 
  if (profile_map_.find(key) == profile_map_.end()) {
    profile_map_[key] = 0.0;
  }
  profile_map_[key] += time;
}


void CuDevice::PrintProfile() {
  if (verbose_ && enabled_) { 
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


std::string CuDevice::GetFreeMemory() {
//fix the 64-bit compilation issue,
//the CUDA API is inconsistent!
#if (CUDA_VERSION >= 4000)
  size_t mem_free, mem_total;
#else
  unsigned int mem_free, mem_total;
#endif
  cuMemGetInfo(&mem_free, &mem_total);
  std::ostringstream os;
  os << "Free:" << mem_free/(1024*1024) << "MB "
     << "Used:" << (mem_total-mem_free)/(1024*1024) << "MB "
     << "Total:" << mem_total/(1024*1024) << "MB";
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
