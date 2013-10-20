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

#include <vector>
#include <algorithm>
#include <dlfcn.h>

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "base/kaldi-error.h"
#include "util/common-utils.h"

namespace kaldi {



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
  if (Enabled()) {
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
    size_t max_print = 15, start_pos = (pairs.size() > max_print ?
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


struct CuAllocatorOptions {
  int32 count; // Number of times we free and delete a particular size before we
               // start to cache it.
  int32 cleanup_interval_bytes;
  double count_increment; // Each time we allocate a new size, we increment
                          // count by this much; it's a heuristic to say that if
                          // we are allocating many different size, we raise the
                          // count-threshold before caching any particular size.
  CuAllocatorOptions(): count(10), cleanup_interval_bytes(1000000),
                        count_increment(0.5) { }
};


/// We define class CuAllocator inside the .cc file, because we don't want to
/// expose it in the header.  Its purpose is to hang on to memory that we have
/// freed, so that we don't waste time in cudaMalloc and cudaMallocPitch().
/// For some reason, they are sometimes very slow.
class CuAllocator {
 public:
  CuAllocator(const CuAllocatorOptions &opts, CuDevice *device):
      device_(device), opts_(opts), count_(opts.count),
      cleanup_countdown_bytes_(opts.cleanup_interval_bytes) { }
  
  inline void *Malloc(size_t size);
  
  inline void *MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch);
  
  inline void Free(void *ptr);

  ~CuAllocator();
 private:
  inline void *MallocInternal(size_t row_bytes, size_t num_rows, size_t *pitch);
  
  // struct MemInfoForSize stores information associated with a particular size
  // of allocated memory.  The row_bytes and num_rows refer to the arguments of
  // a cudaMallocPitch call; for regular, non-pitch allocations with cudaMalloc,
  // we make "num_rows" zero.
  struct MemInfoForSize {
    size_t row_bytes; // or the size, if a regular CudaMalloc, not
                      // CudaMallocPitch.
    size_t num_rows; // or zero if it's a regular CudaMalloc call, not
                     // CudaMallocPitch.
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

  inline MemInfoForSize *FindMemInfo(size_t row_bytes,
                                     size_t num_rows) {
    std::pair<size_t, size_t> this_pair(row_bytes, num_rows);
    // set num_rows to 0 for regular, linear allocation.
    KALDI_ASSERT(row_bytes != 0);
    unordered_map<std::pair<size_t, size_t>, MemInfoForSize*>::iterator iter =
        size_to_list_.find(this_pair);
    if (iter == size_to_list_.end()) {
      int32 count = count_;
      count_ += opts_.count_increment; // This is a kind of heuristic, that if
                                       // we're allocating a lot of different
                                       // sizes, we increase the number of times
                                       // we free a particular size before we
                                       // start caching it.
      return (size_to_list_[this_pair] =  new MemInfoForSize(row_bytes, num_rows,
                                                             count));
    } else {
      return iter->second;
    }
  }

  void PossiblyCleanup(size_t num_bytes);

  // A periodic housekeeping task..
  void Cleanup();

  void ReleaseAllCachedMemory();

  CuDevice *device_; // device this is attached to...
  CuAllocatorOptions opts_;
  
  unordered_map<void*, MemInfoForSize*> addr_to_list_;
  typedef unordered_map<std::pair<size_t, size_t>, MemInfoForSize*,
                        PairHasher<size_t> > SizeHash;
  typedef SizeHash::iterator SizeHashIterator;
  SizeHash size_to_list_;
  
  double count_; // We initialize countdown for each size to this value each time
                 // we encounter a new size.  We increment this by
                 // opts_.count_increment each time; this is a heuristic that if
                 // the program is allocating many different sizes, we put a
                 // higher threshold for any given size.
  int32 cleanup_countdown_bytes_; // countdown in bytes, until the next time we check
                                  // whether we should do cleanup
};


void* CuAllocator::Malloc(size_t size) {
  KALDI_ASSERT(size > 0);
  return MallocInternal(size, 0, NULL);
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
  // num_rows == 0, and row_bytes is just the size to be allocated.
  KALDI_ASSERT(row_bytes != 0 && (num_rows != 0) == (pitch_out != NULL));
  
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
    PossiblyCleanup(num_rows == 0 ? row_bytes : row_bytes * num_rows);
    void *ans;
    if (num_rows == 0) { // Simple malloc request, not "MallocPitch".
      size_t size = row_bytes;
      int32 ret = cudaMalloc(&ans, size);
      if (ret != 0) {
        KALDI_WARN << "Allocation of memory block fo " << size << " bytes "
                   << "failed, releasing cached memory and retrying.";
        ReleaseAllCachedMemory();
        ret = cudaMalloc(&ans, size);
        if (ret != 0)
          KALDI_WARN << "Allocation failed for the second time.    Printing "
                    << "device memory usage and exiting";
          device_->PrintMemoryUsage();
          KALDI_ERR << "Memory allocation failure";
      }
    } else {
      size_t pitch;
      int32 ret = cudaMallocPitch(&ans, &pitch, row_bytes, num_rows);
      if (ret != 0) { // allocation failed...
        KALDI_WARN << "Allocation of " << num_rows << " rows, each of size "
                   << row_bytes << " bytes failed,  releasing cached "
                   << "memory and retrying.";
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
  if (info->countdown == 0) { // We have freed [i.e. actually freed with
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

void CuAllocator::ReleaseAllCachedMemory() {
  typedef unordered_map<std::pair<size_t, size_t>, MemInfoForSize*> SetType;
  typedef SetType::const_iterator IterType;
  for (IterType iter = size_to_list_.begin(); iter != size_to_list_.end();
       ++iter) {
    MemInfoForSize *info = iter->second;
    while (!info->freed.empty()) {
      CU_SAFE_CALL(cudaFree(info->freed.back()));
      info->freed.pop_back();
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
  // Check that nothing was allocated by thge user and not freed.
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
}

void CuDevice::Free(void *ptr) { allocator_->Free(ptr); }

void* CuDevice::MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch) {
  return allocator_->MallocPitch(row_bytes, num_rows, pitch);
}

void* CuDevice::Malloc(size_t size) {
  return allocator_->Malloc(size);
}

CuDevice::CuDevice(): active_gpu_id_(-3), verbose_(true),
                      allocator_(new CuAllocator(CuAllocatorOptions(), this))
  { }


CuDevice::~CuDevice() {
  if (Enabled()) {
    CU_SAFE_CALL(cublasShutdown());
  } else if (active_gpu_id_ == -2) {
    KALDI_WARN << "CUDA was NOT used! No CUDA GPU detected!";
  }
  if (allocator_ != NULL)
    delete allocator_;
}
  
// The instance of the static singleton 
CuDevice CuDevice::global_device_;


}


#endif // HAVE_CUDA
