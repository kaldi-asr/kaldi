// cudamatrix/cu-device.cc

// Copyright 2009-2012  Karel Vesely
//                2013  Lucas Ondel
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <algorithm>
#include <dlfcn.h>

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "base/kaldi-error.h"


namespace kaldi {

CuDevice::CuDevice()
 : active_gpu_id_(-3), verbose_(true) 
{ }



CuDevice::~CuDevice() {
  if (Enabled()) {
    CU_SAFE_CALL(cublasShutdown());
  } else if (active_gpu_id_ == -2) {
    KALDI_WARN << "CUDA was NOT used! No CUDA GPU detected!";
  }
}



/** 
 * SelectGpuId(gpu_id) 
 *
 * The argument 'gpu_id' meaning: 0..N selects a GPU, 
 * -1 disables CUDA, -2 performs GPU auto-detection.
 *
 * If there is no GPU in the system, and we have GPU auto-detection,
 * or GPU is manually disabled the computation will run on CPU. 
 * In other cases it is an error (manual selection).
 *
 * In case of Compute exclusive mode, the GPU is selected by OS, 
 * this has priority over manual/auto selection of GPU.
 *
 * Since the autoselection of GPU is not perfect, it may still 
 * happen that two processes compute on single GPU, which is slow. 
 * The users are advised to use manual selection or exclusive mode.
 *
 * This method must be called at the very beginning of the program
 * (before the cudamatrix objects allocate memory for the data), 
 * or not at all (when we intentionally want to run on the CPU). 
 *
 */
void CuDevice::SelectGpuId(int32 gpu_id) {
  // Make sure this function is not called twice!
  if(Enabled()) {
    KALDI_ERR << "There is already an active GPU " << active_gpu_id_ 
              << ", cannot change it on the fly!";
  }
  // Allow the GPU to stay disabled
  if(!Enabled() && gpu_id == -1) { 
    KALDI_LOG << "Selected device: " << gpu_id 
              << ", we don't even try to get a GPU. We run on CPU.";
    active_gpu_id_ = -1;
    return;
  }
  // Check that we have a gpu available
  int32 n_gpu = 0;
  cudaGetDeviceCount(&n_gpu);
  if(n_gpu == 0 && gpu_id == -2) {
    // If we do automatic selection and no GPU is found, we run on a CPU
    KALDI_WARN << "CUDA will NOT be used!!! No CUDA capable GPU detected...";
    active_gpu_id_ = -2;
    return;
  }
  // In other cases it is an error, no GPU is an error
  if(n_gpu == 0) {
    KALDI_ERR << "No CUDA capable GPU detected, while explicitly asked for gpu-id '"
              << gpu_id << "'.";
  }


  //Now we know that there is a GPU in the system, 
  //and we don't want to have it disabled. 
  //
  //For the GPU selection there are 3 possibilities, 
  //with priorities according to the order:
  //
  //1.) We have compute exclusive mode on (GPU is selected by OS)
  //2.) User did not specify the GPU-id (default value -2), 
  //    we will do automatic selection.
  //3.) User specified the GPU to run on, so we select it.
  if(IsComputeExclusive()) {
    //we have the GPU context now...
    ;
  } else if(gpu_id == -2) {
    SelectGpuIdAuto();
  } else {
    //try to select the desired GPU
    int32 ret = cudaSetDevice(gpu_id);
    //handle the possible errors (no recovery!!!)
    switch(ret) {
      case cudaSuccess : {
        //create the GPU context
        cudaError_t e;
        e = cudaThreadSynchronize(); //deprecated, but for legacy not cudaDeviceSynchronize
        if(e != cudaSuccess) {
          KALDI_ERR << "Failed to create CUDA context on a GPU.";
        }
        //this was okay, so we are done!
        KALDI_LOG << "Selected device: " << gpu_id << " (manually)";
        break;
      }
      case cudaErrorInvalidDevice : { 
        int32 n_gpu = 0;
        cudaGetDeviceCount(&n_gpu);
        KALDI_ERR << "cudaSetDevice(" << gpu_id << "):"
                  << " '" << gpu_id << "' is not a VALID CUDA device! "
                  << " (system has " << n_gpu << " GPUs,"
                  << " valid IDs 0.." << n_gpu-1 << ")";
        break;
      }
      default :
        KALDI_ERR << "cudaSetDevice(" << gpu_id << "): "
                  << "returned " << ret << ", " 
                  << cudaGetErrorString((cudaError_t)ret);
    }
  }


  // Now the we should have active GPU, 
  // so we can query its name and memory stats
  // and notify user which GPU is finally used.
  //
  // Get the device-id of active device:
  {
    int32 act_gpu_id;
    cudaError_t e;
    e = cudaGetDevice(&act_gpu_id);
    if(e != cudaSuccess) {
      KALDI_ERR << "Failed to get device-id of active device.";
    }
    // Remember the id of active GPU 
    active_gpu_id_ = act_gpu_id; //CuDevice::Enabled() is true from now on
    // Initialize the CUBLAS
    CU_SAFE_CALL(cublasInit());

    // Notify user which GPU is finally used
    char name[128];
    DeviceGetName(name,128,act_gpu_id);

    CU_SAFE_CALL(cudaGetDeviceProperties(&properties_, act_gpu_id));

    KALDI_LOG << "The active GPU is [" << act_gpu_id << "]: "
              << name << "\t" << GetFreeMemory(NULL, NULL) << " version "
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
  // check that we have a gpu
  int32 n_gpu = 0;
  cudaGetDeviceCount(&n_gpu);
  if(n_gpu == 0) {
    KALDI_LOG << "No CUDA devices found";
    return false;
  }
  
  // Create a GPU context
  // This will be kept if we detect compute exclusive mode
  // or released in the other case.
  //
  // It does not harm if the function gets called twice,
  // and the context is already created.
  cudaError_t e;
  e = cudaThreadSynchronize(); //deprecated, but for legacy not cudaDeviceSynchronize
  if(e != cudaSuccess) {
    KALDI_ERR << "Failed to create CUDA context on a GPU. No more unused GPUs in compute exclusive mode?";
  }
  
  // get the device-id and its device-properties
  int32 gpu_id = -1;
  e = cudaGetDevice(&gpu_id);
  if(e != cudaSuccess) {
    KALDI_ERR << "Failed to get current device";
  }
  struct cudaDeviceProp gpu_prop;
  e = cudaGetDeviceProperties(&gpu_prop, gpu_id);
  if(e != cudaSuccess) {
    KALDI_ERR << "Failed to get device properties";
  }
  // find out whether compute exclusive mode is used
  switch (gpu_prop.computeMode) {
    case cudaComputeModeExclusive :
      KALDI_LOG << "CUDA setup operating under Compute Exclusive Mode.";
      return true;
      break;
    #if (CUDA_VERSION >= 4000)
    case cudaComputeModeExclusiveProcess :
      KALDI_LOG << "CUDA setup operating under Compute Exclusive Process Mode.";
      return true;
      break;
    #endif
    default :
      // The computation mode is not compute-exclusive,
      // in this case we release the GPU context...
      e = cudaThreadExit(); //deprecated, but for legacy reason not cudaDeviceReset
      if(e != cudaSuccess) {
        KALDI_ERR << "Failed to release CUDA context on a GPU";
      }
      return false;
  }
}



void CuDevice::SelectGpuIdAuto() {
  // check that we have at least one gpu
  int32 n_gpu = 0;
  cudaGetDeviceCount(&n_gpu);
  if(n_gpu == 0) {
    KALDI_ERR << "No CUDA devices found";
    return;
  }

  // The GPU is selected according to maximal free memory ratio
  std::vector<float> free_mem_ratio(n_gpu+1, 0.0);
  //get ratios of memory use, if possible
  KALDI_LOG << "Selecting from " << n_gpu << " GPUs";
  for(int32 n = 0; n < n_gpu; n++) {
    int32 ret = cudaSetDevice(n);
    switch(ret) {
      case cudaSuccess : {
        //create the CUDA context for the thread
        cudaThreadSynchronize(); //deprecated, but for legacy not cudaDeviceSynchronize
        //get GPU name
        char name[128];
        DeviceGetName(name,128,n);
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
        cudaThreadExit(); //deprecated, but for legacy reason not cudaDeviceReset
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
  //find GPU with max free memory
  int32 max_id=0;
  for(int32 n=1; n<free_mem_ratio.size(); n++) {
    if(free_mem_ratio[n] > free_mem_ratio[max_id]) max_id=n;
  }
  //the free_mem_ratio should be bigger than zero
  if(!free_mem_ratio[max_id] > 0.0) {
    KALDI_ERR << "No device could be selected (this should never happen)";
  }

  //finally select the GPU
  KALDI_LOG << "Selected device: " << max_id << " (automatically)";
  CU_SAFE_CALL(cudaSetDevice(max_id));
  //create the context
  cudaError_t e;
  e = cudaThreadSynchronize(); //deprecated, but for legacy not cudaDeviceSynchronize
  if(e != cudaSuccess) {
    KALDI_ERR << "Failed to create CUDA context on a GPU.";
  }
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
    std::vector<std::pair<double, std::string> > pairs;
    for(it = profile_map_.begin(); it != profile_map_.end(); ++it)
      pairs.push_back(std::make_pair(it->second, it->first));
    std::sort(pairs.begin(), pairs.end());
    for (size_t i = 0; i < pairs.size(); i++) 
      os << pairs[i].second << "\t" << pairs[i].first << "s\n";
    os << "-----";
    KALDI_LOG << os.str();
  }
}


std::string CuDevice::GetFreeMemory(int64* free, int64* total) {
// WARNING! the CUDA API is inconsistent accross versions!
#if (CUDA_VERSION >= 3020)
  //define the function signature type
  size_t mem_free, mem_total;
#else
  unsigned int mem_free, mem_total;
#endif
  { 
    //we will load the cuMemGetInfo dynamically from libcuda.so
    //cuMemGetInfo(&mem_free, &mem_total);
    //pre-fill ``safe'' values that will not cause problems
    mem_free = 1; mem_total = 1;
    //open libcuda.so
    void* libcuda = dlopen("libcuda.so",RTLD_LAZY);
    if(NULL == libcuda) { 
      KALDI_WARN << "cannot open libcuda.so"; 
    } else {
      //define the function signature type
      //and get the symbol
#if (CUDA_VERSION >= 3020)
      typedef CUresult (*cu_fun_ptr)(size_t*, size_t*);
      cu_fun_ptr dl_cuMemGetInfo = (cu_fun_ptr)dlsym(libcuda,"cuMemGetInfo_v2"); 
#else
      typedef CUresult (*cu_fun_ptr)(int*, int*);
      cu_fun_ptr dl_cuMemGetInfo = (cu_fun_ptr)dlsym(libcuda,"cuMemGetInfo"); 
#endif
      if(NULL == dl_cuMemGetInfo) {
        KALDI_WARN << "cannot load cuMemGetInfo from libcuda.so";
      } else {
        //call the function
        dl_cuMemGetInfo(&mem_free, &mem_total);
      }
      //close the library
      dlclose(libcuda);
    }
  }
  // copy the output values outside
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


void CuDevice::DeviceGetName(char* name, int32 len, int32 dev) {
  //prefill with something reasonable
  strncpy(name,"Unknown GPU",len);
  //open libcuda.so
  void* libcuda = dlopen("libcuda.so",RTLD_LAZY);
  if(NULL == libcuda) {
    KALDI_WARN << "cannot open libcuda.so"; 
  } else {
    //define the function signature type
    typedef CUresult (*cu_fun_ptr)(char*,int,CUdevice);
    //get the symbol
    cu_fun_ptr cuDeviceGetName_ptr = (cu_fun_ptr)dlsym(libcuda,"cuDeviceGetName"); 
    if(NULL == cuDeviceGetName_ptr) {
      KALDI_WARN << "cannot load cuDeviceGetName from libcuda.so"; 
    } else {
      //call the function
      cuDeviceGetName_ptr(name, len, dev);
    }
    //close the library
    dlclose(libcuda);
  }
}


// The instance of the static singleton 
CuDevice CuDevice::global_device_;


}


#endif // HAVE_CUDA
