// cudamatrix/cu-device.cc

// Copyright 2009-2012  Karel Vesely
//                2013  Lucas Ondel
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <algorithm>
#include <dlfcn.h>
#include <unistd.h> // for sleep

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrix.h"
#include "base/kaldi-error.h"
#include "util/common-utils.h"

namespace kaldi {


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
  if (use_gpu != "yes" && use_gpu != "no" && use_gpu != "optional") {
    KALDI_ERR << "Please choose : --use-gpu=yes|no|optional, passed '" << use_gpu << "'";
  }
 
  // Make sure this function is not called twice!
  if (Enabled()) {
    KALDI_ERR << "There is already an active GPU " << active_gpu_id_ 
              << ", cannot change it on the fly!";
  }
  // Allow the GPU to stay disabled
  if(!Enabled() && use_gpu == "no") { 
    KALDI_LOG << "Manually selected to compute on CPU.";
    return;
  }

  // Check that we have a gpu available
  int32 n_gpu = 0;
  cudaGetDeviceCount(&n_gpu);
  if(n_gpu == 0) {
    if (use_gpu == "yes") {
      KALDI_ERR << "No CUDA GPU detected!";
    }
    if (use_gpu == "optional") {
      KALDI_WARN << "Running on CPU!!! No CUDA GPU detected...";
      return;
    }
  }

  //
  // Create a CUDA context : in case of compute-exclusive mode OS selects gpu_id,
  // or default gpu_id=0. In the case with no free GPUs a context cannot be created
  // (compute-exclusive mode).
  //
  cudaError_t e;
  e = cudaThreadSynchronize(); //<< CUDA context gets created here.
  if (e != cudaSuccess) {
    // So far no we don't have context, sleep a bit and retry.
    int32 sec_sleep = (use_gpu == "yes" ? 20 : 2);
    KALDI_WARN << "Will try again to get a GPU after " << sec_sleep 
               << " seconds.";
    sleep(sec_sleep);
    cudaGetLastError(); // reset the error state    
    e = cudaThreadSynchronize(); //<< 2nd trial to get CUDA context.
    if (e != cudaSuccess) {
      if (use_gpu == "yes") {
        KALDI_ERR << "Failed to create CUDA context, no more unused GPUs?";
      }
      if (use_gpu == "optional") {
        KALDI_WARN << "Running on CPU!!! No more unused CUDA GPUs?";
        return;
      }
    }
  }

  // Re-assure we have the context
  KALDI_ASSERT(cudaSuccess == cudaThreadSynchronize());

  // Check if the machine use compute exclusive mode 
  if (IsComputeExclusive()) {
    FinalizeActiveGpu();
    return;
  } else {
    // Or suggest to use compute exclusive mode
    if(n_gpu > 1) { 
      KALDI_WARN << "Suggestion: use 'nvidia-smi -c 1' to set compute exclusive mode";
    }
    // And select the GPU according to proportion of free memory
    if(SelectGpuIdAuto()) {
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
    
    KALDI_LOG << "The active GPU is [" << act_gpu_id << "]: " << name << "\t"
              << GetFreeMemory(&free_memory_at_startup_, NULL) << " version "
              << properties_.major << "." << properties_.minor;

    if (verbose_) PrintMemoryUsage();
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


bool CuDevice::SelectGpuIdAuto() {
  // Check that we have at least one gpu
  int32 n_gpu = 0;
  cudaGetDeviceCount(&n_gpu);
  if(n_gpu == 0) {
    KALDI_WARN << "No CUDA devices found";
    return false;
  }
  
  // The GPU is selected according to maximal free memory ratio
  std::vector<float> free_mem_ratio(n_gpu+1, 0.0);
  // Get ratios of memory use, if possible
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
  KALDI_ASSERT(free_mem_ratio[max_id] > 0.0);

  //finally select the GPU
  KALDI_LOG << "Selected device: " << max_id << " (automatically)";
  CU_SAFE_CALL(cudaSetDevice(max_id));
  //create the context
  cudaError_t e;
  e = cudaThreadSynchronize(); //deprecated, but for legacy not cudaDeviceSynchronize
  if(e != cudaSuccess) {
    KALDI_WARN << "Failed to create CUDA context on a GPU.";
    return false;
  }
  return true;
}


void CuDevice::AccuProfile(const std::string &key, double time) { 
  if (profile_map_.find(key) == profile_map_.end()) {
    profile_map_[key] = 0.0;
  }
  profile_map_[key] += time;
}

void CuDevice::PrintMemoryUsage() const {
  if (Enabled()) {
    int64 free_memory_now;
    GetFreeMemory(&free_memory_now, NULL);
    KALDI_LOG << "Memory used: " << (free_memory_at_startup_ - free_memory_now) << " bytes.";
  }
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
    size_t max_print = 15, start_pos = (pairs.size() <= max_print ?
                                        0 : pairs.size() - max_print);
    for (size_t i = start_pos; i < pairs.size(); i++) 
      os << pairs[i].second << "\t" << pairs[i].first << "s\n";
    os << "-----";
    KALDI_LOG << os.str();
    PrintMemoryUsage();
  }
}


std::string CuDevice::GetFreeMemory(int64* free, int64* total) const {
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


void CuDevice::CheckGpuHealth() {
  if(!Enabled()) return;
  Timer t;
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
  AccuProfile(__func__, t.Elapsed());
}


struct CuAllocatorOptions {
  bool cache_memory; // Enable GPU memory caching, (false = disable).
  int32 count; // Number of times we free and delete a particular size before we
               // start to cache it.
  int32 cleanup_interval_bytes;
  CuAllocatorOptions()
   : cache_memory(true), count(1), cleanup_interval_bytes(1000000) { }
};


/// We define class CuAllocator inside the .cc file, because we don't want to
/// expose it in the header.  Its purpose is to hang on to memory that we have
/// freed, so that we don't waste time in cudaMalloc and cudaMallocPitch().
/// For some reason, they are sometimes very slow.
class CuAllocator {
 public:
  CuAllocator(const CuAllocatorOptions &opts, CuDevice *device):
      device_(device), opts_(opts),
      cleanup_countdown_bytes_(opts.cleanup_interval_bytes) { }
  
  inline void *Malloc(size_t size);
  
  inline void *MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch);
  
  inline void Free(void *ptr);

  inline void DisableCaching();

  ~CuAllocator();
 private:
  inline void *MallocInternal(size_t row_bytes, size_t num_rows, size_t *pitch);
  
  // struct MemInfoForSize stores information associated with a particular size
  // of allocated memory.  The row_bytes and num_rows refer to the arguments of
  // a cudaMallocPitch call; for regular, non-pitch allocations with cudaMalloc,
  // we make "row_bytes" zero and the size in bytes is "num_rows"... there is a
  // reason why we do it this way round (make num_rows contain the size in
  // bytes); it relates to the ordering of the map, and the behavior when
  // we didn't find the exact size and want to find larger match.

  
  struct MemInfoForSize {
    size_t row_bytes; // or zero, if a regular CudaMalloc, not
                      // CudaMallocPitch.
    size_t num_rows; // or the number of rows, if it's a regular CudaMalloc
                     // call, not CudaMallocPitch.
    size_t pitch; // If CudaMallocPitch, the pitch returned by CudaMallocPitch;
                  // this code assumes (and checks) that it's a deterministic
                  // function of row_bytes and num_rows.
    size_t countdown; // number that have been freed and not cached.
    size_t currently_used; // number that are "in the wild".. kept for
                           // diagnostics and error detection.
    std::vector<void*> freed; // freed and cached...
      
    MemInfoForSize(size_t row_bytes,
                   size_t num_rows,
                   int32 count):
        row_bytes(row_bytes),
        num_rows(num_rows),
        pitch(0),
        countdown(count),
        currently_used(0) { }
  };


  // FindMemInfo returns the MemInfoForSize object for this (row_bytes,
  // num_rows) combination if it exists; otherwise...
  // if there is a MemInfoForSize object with the same row_bytes and larger (but
  // not more than twice larger) num_rows that has freed memory waiting, it
  // returns that; otherwise, it returns a new MemInfoForSize object for the
  // requested size).
  
  inline MemInfoForSize *FindMemInfo(size_t row_bytes,
                                     size_t num_rows) {
    if (row_bytes >= size_to_list_.size())
      size_to_list_.resize(row_bytes + 1, NULL);
    
    // note: we set row_bytes to 0 for regular, linear allocation.
    KALDI_ASSERT(num_rows != 0);

    if (size_to_list_[row_bytes] == NULL)
      size_to_list_[row_bytes] = new std::map<size_t, MemInfoForSize*>;


    std::map<size_t, MemInfoForSize*> &size_to_list = *(size_to_list_[row_bytes]);

    typedef std::map<size_t, MemInfoForSize* >::iterator IterType;

    // get an iterator to the requested object or the next-larger one.
    // Here, upper_bound(num_rows - 1) returns an object strictly greater
    // than num_rows - 1, which could be num_rows itself.  We need to
    // treat num_rows == 0 as a special case because of size_t being
    // unsigned.
    IterType iter = (num_rows == 0 ? size_to_list.begin() :
                     size_to_list.upper_bound(num_rows - 1));
    
    if (iter != size_to_list.end() && iter->first == num_rows) {
      // Found a MemInfoForSize object
      // with the requested size -> return it.
      KALDI_ASSERT(iter->second->row_bytes == row_bytes &&
                   iter->second->num_rows == num_rows);
      return iter->second;
    } else if (iter != size_to_list.end() &&
               iter->second->num_rows <= 2 * num_rows &&
               !iter->second->freed.empty()) {
      // Return the non-matching one with freed memory, which is larger than
      // this one but not more than twice larger.
      KALDI_ASSERT(iter->second->row_bytes == row_bytes &&
                   iter->second->num_rows > num_rows); // confirm expectations.
      return iter->second;
    } else {
      // There was no such object, and the next-larger object either did not
      // exist, had more than twice the num-rows requested, or had no free
      // memory -> create an object with the requested size.
      return (size_to_list[num_rows] =  new MemInfoForSize(row_bytes, num_rows,
                                                           opts_.count));
    }
  }
                 
  void PossiblyCleanup(size_t num_bytes);

  // A periodic housekeeping task..
  void Cleanup();

  // Frees all memory in the "freed" vectors; memory that the
  // user freed but we held on to.  If destroy == true, also
  // clean up all memory held in the size_to_list_ object (i.e.
  // allocated maps and MemInfoForSize objects).
  void ReleaseAllCachedMemory(bool destroy = false);

  CuDevice *device_; // device this is attached to...
  CuAllocatorOptions opts_;


  unordered_map<void*, MemInfoForSize*> addr_to_list_;

  // size_to_list_ is indexed first by row_bytes (which is zero for linear
  // mallocs) and then by num_rows (which for linear mallocs, is the actual size
  // in bytes).
  std::vector<std::map<size_t, MemInfoForSize*>* > size_to_list_;
  
  int32 cleanup_countdown_bytes_; // countdown in bytes, until the next time we check
                                  // whether we should do cleanup
};


void* CuAllocator::Malloc(size_t size) {
  KALDI_ASSERT(size > 0);
  return MallocInternal(0, size, NULL);
}

void* CuAllocator::MallocPitch(size_t num_rows, size_t row_bytes,
                               size_t *pitch) {
  KALDI_ASSERT(num_rows > 0 && row_bytes > 0 && pitch != NULL);
  return MallocInternal(num_rows, row_bytes, pitch);
}

void* CuAllocator::MallocInternal(size_t row_bytes,
                                  size_t num_rows,
                                  size_t *pitch_out) {
  // we share the code for standard cudaMalloc and cudaMallocPitch
  // because most of it is the same.  for cudaMalloc, we'll have
  // row_bytes == 0, and num_rows is just the size to be allocated.
  KALDI_ASSERT(num_rows != 0 && (row_bytes != 0) == (pitch_out != NULL));
  
  MemInfoForSize *info = FindMemInfo(row_bytes, num_rows);
  if (!info->freed.empty()) { // We can satisfy the request with cached,
                              // previously-allocated memory.
    void *ans = info->freed.back();
    info->freed.pop_back();
    info->currently_used++;
    addr_to_list_[ans] = info;
    if (pitch_out) *pitch_out = info->pitch;
    return ans;
  } else {
    PossiblyCleanup(row_bytes == 0 ? num_rows : row_bytes * num_rows);
    void *ans;
    if (row_bytes == 0) { // Simple malloc request, not "MallocPitch".
      size_t size = num_rows;
      int32 ret = cudaMalloc(&ans, size);
      if (ret != 0) {
        KALDI_WARN << "Allocation of memory block of " << size << " bytes "
                   << "failed, releasing cached memory and retrying.";
        cudaGetLastError(); // reset the error state
        ReleaseAllCachedMemory();
        ret = cudaMalloc(&ans, size);
        if (ret != 0) {
          KALDI_WARN << "Allocation failed for the second time.    Printing "
                    << "device memory usage and exiting";
          device_->PrintMemoryUsage();
          KALDI_ERR << "Memory allocation failure";
        }
      }
    } else {
      size_t pitch;
      int32 ret = cudaMallocPitch(&ans, &pitch, row_bytes, num_rows);
      if (ret != 0) { // allocation failed...
        KALDI_WARN << "Allocation of " << num_rows << " rows, each of size "
                   << row_bytes << " bytes failed,  releasing cached "
                   << "memory and retrying.";
        cudaGetLastError(); // reset the error state
        ReleaseAllCachedMemory();
        ret = cudaMallocPitch(&ans, &pitch, row_bytes, num_rows);
        if (ret != 0) {
          KALDI_WARN << "Allocation failed for the second time.    Printing "
                    << "device memory usage and exiting";
          device_->PrintMemoryUsage();
          KALDI_ERR << "Memory allocation failure";
        }
      }
      KALDI_ASSERT(pitch > 0);
      if (info->pitch == 0) { // First allocation; have not set info->pitch yet.
        info->pitch = pitch;
      } else if (pitch != info->pitch) {
        KALDI_ERR << "Pitch differs between multiple calls with the same "
                  << "parameters: " << pitch << " vs. " << info->pitch;
      }
      *pitch_out = info->pitch;
    }
    addr_to_list_[ans] = info;
    info->currently_used++;
    return ans;
  }
}

void CuAllocator::Free(void *addr) {
  unordered_map<void*, MemInfoForSize*>::iterator iter
      = addr_to_list_.find(addr);
  if (iter == addr_to_list_.end()) {
    KALDI_ERR << "Attempt to free address " << addr << " that was not allocated "
              << "by CuDevice::Malloc() (or was previously freed);";
  }
  MemInfoForSize *info = iter->second;
  addr_to_list_.erase(addr); // Erase this element in the addr_to_list_ map.
  info->currently_used--;
  if (info->countdown == 0 && opts_.cache_memory) { 
                              // We have freed [i.e. actually freed with
                              // CudaFree()] enough of these that we think
                              // we're wasting too much time this way and
                              // need to start caching them.
    info->freed.push_back(addr);
  } else { // Actually free the address, and decrease "countdown".
    info->countdown--;
    CU_SAFE_CALL(cudaFree(addr)); // This is how we free, even if allocated with
                                  // cudaMallocPitch().
  }
}


inline void CuAllocator::DisableCaching() {
  KALDI_LOG << "Disabling caching of GPU memory.";
  KALDI_ASSERT(size_to_list_.empty()); // No memory allocated yet!
  opts_.cache_memory = false;
}

void CuAllocator::ReleaseAllCachedMemory(bool destroy) {
  KALDI_VLOG(2) << "Releasing all cached memory.";
  for (size_t i = 0; i < size_to_list_.size(); i++) {
    if (size_to_list_[i] == NULL)
      continue;
    typedef std::map<size_t, MemInfoForSize*>::iterator  IterType;
    for (IterType iter = size_to_list_[i]->begin();
         iter != size_to_list_[i]->end(); ++iter) {
      MemInfoForSize *info = iter->second;
      if (destroy && !info->freed.empty()) {
        // When called from the destructor at program end, if verbose level is
        // high, say the sizes we had.
        if (info->row_bytes == 0) {
          KALDI_VLOG(3) << "Releasing " << info->freed.size() << " blocks of "
                        << info->num_rows << " bytes.";
        } else {
          KALDI_VLOG(3) << "Releasing " << info->freed.size()
                        << " 2-d blocks of " << info->num_rows << " rows of "
                        << info->row_bytes << " bytes each.";
        }
      }
      if (!destroy) {
        // We only do this freeing part when we're *not* called from the
        // destuctor (destroy = false).  This leads to a crash when called from
        // the destructor, with cudaFree returning "unload of CUDA runtime
        // failed".  Presumably this has to do with the destruction order of
        // C++, which we can't really control.
        while (!info->freed.empty()) {
          CU_SAFE_CALL(cudaFree(info->freed.back()));
          info->freed.pop_back();
        }
      }
      if (destroy)
        delete info;
    }
    if (destroy) {
      delete size_to_list_[i];
      size_to_list_[i] = NULL;
    }
  }
}

void CuAllocator::Cleanup() {
  // TODO: implement this or remove it (and also PossiblyCleanup).
  // Actually we may never implement this, as just calling
  // ReleaseAllCachedMemory whenever an allocation fails is probably
  // sufficient.
}
void CuAllocator::PossiblyCleanup(size_t num_bytes) {
  if (static_cast<size_t>(cleanup_countdown_bytes_) <= num_bytes) {
    Cleanup();
    cleanup_countdown_bytes_ = opts_.cleanup_interval_bytes;
  } else {
    cleanup_countdown_bytes_ -= static_cast<int32>(num_bytes);
  }
}

CuAllocator::~CuAllocator() {
  // Check that nothing was allocated by the user and not freed.
  std::set<MemInfoForSize*> unfreed_set;
  typedef unordered_map<void*, MemInfoForSize *>::iterator IterType;
  for (IterType iter = addr_to_list_.begin(); iter != addr_to_list_.end();
       ++iter)
    unfreed_set.insert(iter->second);
  for (std::set<MemInfoForSize*>::iterator iter = unfreed_set.begin();
       iter != unfreed_set.end(); ++iter) {
    MemInfoForSize *info = *iter;
    KALDI_ASSERT(info->currently_used > 0); // Or should not be in this set
                                            // (code error or memory corruption)
    if (info->num_rows == 0) {
      KALDI_WARN << info->currently_used << " memory chunks of size "
                 << info->row_bytes << " were allocated and not freed.";
    } else {
      KALDI_WARN << info->currently_used << " memory chunks of size "
                 << info->row_bytes << " per row, and " << info->num_rows
                 << " rows, were allocated and not freed.";
    }
  }
  
  bool destroy = true;
  ReleaseAllCachedMemory(destroy);
}

void CuDevice::Free(void *ptr) { allocator_->Free(ptr); }

void* CuDevice::MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch) {
  return allocator_->MallocPitch(row_bytes, num_rows, pitch);
}

void* CuDevice::Malloc(size_t size) {
  return allocator_->Malloc(size);
}

void CuDevice::DisableCaching() {
  allocator_->DisableCaching();
}

CuDevice::CuDevice(): active_gpu_id_(-1), verbose_(true),
                      allocator_(new CuAllocator(CuAllocatorOptions(), this))
  { }


CuDevice::~CuDevice() {
  if (allocator_ != NULL)
    delete allocator_;
  if (Enabled())
    CU_SAFE_CALL(cublasShutdown());
}
  
// The instance of the static singleton 
CuDevice CuDevice::global_device_;


}


#endif // HAVE_CUDA
