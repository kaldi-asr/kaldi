// cudamatrix/cu-device.h

// Copyright 2009-2012  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_DEVICE_H_
#define KALDI_CUDAMATRIX_CU_DEVICE_H_

#if HAVE_CUDA == 1
#include <cublas_v2.h>
#include <cusparse.h>
#include <map>
#include <string>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "base/kaldi-common.h"
#include "base/timer.h"
#include "cudamatrix/cu-allocator.h"

namespace kaldi {

class CuTimer;

/**
   This class contains code for selecting the CUDA device, initializing the
   cuBLAS and cuSparse handles, and providing an interface for memory allocation
   (which supports caching, to avoid the slowness of the CUDA memory allocator).

   There is a separate instance of the CuDevice object for each thread of the
   program, but many of its variables are static (hence, shared between all
   instances).

   We only (currently) support using a single GPU device; however, we support
   multiple CUDA streams.  The expected programming model here is that you will
   have multiple CPU threads, and each CPU thread automatically gets its own
   CUDA stream because we compile with -DCUDA_API_PER_THREAD_DEFAULT_STREAM.

   In terms of synchronizing the activities of multiple threads: The CuDevice
   object (with help from the underlying CuAllocator object) ensures that the
   memory caching code won't itself be a cause of synchronization problems,
   i.e. you don't have to worry that when you allocate with CuDevice::Malloc(),
   the memory will still be in use by another thread on the GPU.  However, it
   may sometimes still be necessary to synchronize the activities of multiple
   streams by calling the function SynchronizeGpu()-- probably right before a
   thread increments a semaphore, right after it waits on a semaphore, or
   right after it acquires a mutex, or something like that.

 */
class CuDevice {
 public:

  // You obtain the CuDevice for the current thread by calling
  //   CuDevice::Instantiate()
  // At the beginning of the program, if you want to use a GPU, you
  // should call CuDevice::Instantiate().SelectGpuId(..).
  static inline CuDevice& Instantiate() {
    CuDevice &ans = this_thread_device_;
    if (!ans.initialized_)
      ans.Initialize();
    return ans;
  }

  inline cublasHandle_t GetCublasHandle() { return cublas_handle_; }
  inline cusparseHandle_t GetCusparseHandle() { return cusparse_handle_; }

  // We provide functions Malloc(), MallocPitch() and Free() which replace
  // cudaMalloc(), cudaMallocPitch() and cudaFree().  Their function is to cache
  // the results of previous allocations to avoid the very large overhead that
  // CUDA's allocation seems to give for some setups.
  inline void* Malloc(size_t size) {
    return multi_threaded_ ? g_cuda_allocator.MallocLocking(size) :
        g_cuda_allocator.Malloc(size);
  }

  inline void* MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch) {
    if (multi_threaded_) {
      return g_cuda_allocator.MallocPitchLocking(row_bytes, num_rows, pitch);
    } else if (debug_stride_mode_) {
      // The pitch bucket size is hardware dependent.
      // It is 512 on K40c with CUDA 7.5
      // "% 8" ensures that any 8 adjacent allocations have different pitches
      // if their original pitches are same in the normal mode.
      return g_cuda_allocator.MallocPitch(
          row_bytes + 512 * RandInt(0, 4), num_rows,
          pitch);
    } else {
      return g_cuda_allocator.MallocPitch(row_bytes, num_rows, pitch);
    }
  }

  inline void Free(void *ptr) {
    if (multi_threaded_) g_cuda_allocator.FreeLocking(ptr);
    else g_cuda_allocator.Free(ptr);
  }

  /// Select a GPU for computation.  You are supposed to call this function just
  /// once, at the beginning of the program (from the main thread), or not at
  /// all.
  /// The 'use_gpu' modes are:
  ///  "yes" -- Select GPU automatically and die if this fails.  If you have set
  ///           the GPUs to exclusive mode it will select one
  ///           pseudo-randomly; otherwise it will choose whichever one has
  ///           the most free memory (but we recommend to set GPUs to
  ///           exclusive mode, or controlling which GPU to use by setting
  ///           the variable CUDA_VISIBLE_DEVICES to the id of the GPU you
  ///           want the program to use.
  ///  "optional" -- Do as above, but if it fails, back off to CPU.
  ///  "no"       -- Run on CPU.
  void SelectGpuId(std::string use_gpu);

  /// Check if the CUDA GPU is selected for use
  bool Enabled() const {
    return (device_id_ > -1);
  }

  /// Returns true if either we have no GPU, or we have a GPU
  /// and it supports double precision.
  bool DoublePrecisionSupported();

  /// This function accumulates stats on timing that
  /// are printed out when you call PrintProfile().  However,
  /// it only does something if VerboseLevel() >= 1.
  void AccuProfile(const char *function_name, const CuTimer &timer);

  /// Print some profiling information using KALDI_LOG.
  void PrintProfile();

  /// Print some memory-usage information using KALDI_LOG.
  void PrintMemoryUsage() const;

  /// The user should call this if the program plans to access the GPU (e.g. via
  /// using class CuMatrix) from more than one thread.  If you fail to call this
  /// for a multi-threaded program, it may occasionally segfault (and also
  /// the code will detect that you failed to call it, and will print a warning).
  inline void AllowMultithreading() { multi_threaded_ = true; }

  /// Get the name of the GPU
  void DeviceGetName(char* name, int32 len, int32 dev);

  /// Check if GPU is in good condition by multiplying small matrices on GPU+CPU.
  /// Overheated GPUs may give inaccurate results, which we want to detect.
  void CheckGpuHealth();

  /// If Enabled(), returns the number n of bytes such that the matrix stride
  /// will always be a multiple of n (from properties_.textureAlignment).
  /// Otherwise, return 16, which is the stride used for CPU matrices.
  int32 GetMatrixAlignment() const;

  /// Call SetDebugStrideMode(true) to activate a mode where calls
  /// to MallocPitch will purposely allocate arrays with different pitch
  /// (inconsistent between calls).  This is only useful for testing code.
  /// This function returns the previous mode, where true means inconsistent
  /// pitch.  Note that you cannot ever rely on the strides from MallocPitch()
  /// being consistent for the same request, but in practice they tend to be
  /// consistent unless you are close to running out of memory.
  bool SetDebugStrideMode(bool mode) {
    bool old_mode = debug_stride_mode_;
    debug_stride_mode_ = mode;
    return old_mode;
  }

  /// Check if the GPU is set to compute exclusive mode (you can set this mode,
  /// if you are root, by doing: `nvidia-smi -c 3`).  Returns true if we have a
  /// GPU and it is running in compute exclusive mode.  Returns false otherwise.
  /// WILL CRASH if we are not using a GPU at all.  If calling this as a user
  /// (i.e. from outside the class), call this only if Enabled() returns true.
  bool IsComputeExclusive();

  ~CuDevice();
 private:
  // Default constructor used to initialize this_thread_device_
  CuDevice();
  CuDevice(CuDevice&); // Disallow.
  CuDevice &operator=(CuDevice&);  // Disallow.


  /// The Initialize() function exists to do the following, in threads other
  /// than the main thread, and only if we are using a GPU: call
  /// cudaSetDevice(), and set up cublas_handle_ and cusparse_handle_.  It does
  /// get called in the main thread (see documentation by its definition), but
  /// does nothing interesting there.
  void Initialize();

  /// Automatically select GPU and get CUDA context (this is only called, from
  /// SelectGpuId(), if the GPUs are in non-exclusive mode).  Returns true on
  /// success.
  bool SelectGpuIdAuto();

  /// This function, called from SelectGpuId(), is to be called when a
  /// GPU context corresponding to the GPU we want to use exists; it
  /// works out the device-id, creates the cuBLAS and cuSparse handles,
  /// and prints out some information that's useful for debugging.
  /// It also sets initialized_ to true, to suppress Initialize() from
  /// being called on this, the main thread, in future, since
  /// that would try to create the handles again.
  void FinalizeActiveGpu();

  /// Should only be called if Enabled() == true.
  int32 MajorDeviceVersion();

  /// Should only be called if Enabled() == true.
  int32 MinorDeviceVersion();


  // Each thread has its own CuDevice object, which contains the cublas and
  // cusparse handles.  These are unique to the thread (which is what is
  // recommended by NVidia).
  static thread_local CuDevice this_thread_device_;

  // The GPU device-id that we are using.  This will be initialized to -1, and will
  // be set when the user calls
  //  CuDevice::Instantiate::SelectGpuId(...)
  // from the main thread.  Background threads will, when spawned and when
  // CuDevice::Instantiate() is called from them the first time, will
  // call cudaSetDevice(device_id))
  static int32 device_id_;

  // This will automatically be set to true if the application has multiple
  // threads that access the GPU device.  It is used to know whether to
  // use locks when accessing the allocator and the profiling-related code.
  static bool multi_threaded_;

  // The variable profile_map_ will only be used if the verbose level is >= 1;
  // it will accumulate some function-level timing information that is printed
  // out at program end.  This makes things a bit slower as we have to call
  // cudaDeviceSynchronize() to make the timing information meaningful.
  static unordered_map<std::string, double, StringHasher> profile_map_;
  // profile_mutex_ guards profile_map_ in case multi_threaded_ is true.
  static std::mutex profile_mutex_;

  // free_memory_at_startup_ is just used in printing the memory used according
  // to the device.
  static int64 free_memory_at_startup_;
  static cudaDeviceProp properties_;

  // If set to true by SetDebugStrideMode(), code will be activated to use
  // pseudo-random stride values when allocating data (to detect errors which
  // otherwise would be rare).
  static bool debug_stride_mode_;


  // The following member variable is initialized to false; if the user calls
  // Instantiate() in a thread where it is still false, Initialize() will be
  // called, in order to -- if a GPU is being used-- call cudaSetDevice() and
  // set up the cublas and cusparse handles.
  bool initialized_;

  // This variable is just a copy of the static variable device_id_.  It's used
  // to detect when this code is called in the wrong way.
  int32 device_id_copy_;

  cublasHandle_t cublas_handle_;

  cusparseHandle_t cusparse_handle_;

}; // class CuDevice


// Class CuTimer is a convenience wrapper for class Timer which only
// sets the time if the verbose level is >= 1.  This helps avoid
// an unnecessary system call if the verbose level is 0 and you
// won't be accumulating the timing stats.
class CuTimer: public Timer {
 public:
  CuTimer(): Timer(GetVerboseLevel() >= 1) { }
};

// This function is declared as a more convenient way to get the CUDA device handle for use
// in the CUBLAS v2 API, since we so frequently need to access it.
inline cublasHandle_t GetCublasHandle() { return CuDevice::Instantiate().GetCublasHandle(); }
// A more convenient way to get the handle to use cuSPARSE APIs.
inline cusparseHandle_t GetCusparseHandle() { return CuDevice::Instantiate().GetCusparseHandle(); }


}  // namespace kaldi

#endif // HAVE_CUDA


namespace kaldi {

/**
   The function SynchronizeGpu(), which for convenience is defined whether or
   not we have compiled for CUDA, is intended to be called in places where threads
   need to be synchronized.

   It just launches a no-op kernel into the legacy default stream.  This will
   have the effect that it will run after any kernels previously launched from
   any stream(*), and before kernels that will later be launched from any stream(*).
   (*) does not apply to non-blocking streams.

   Note: at the time of writing we never call SynchronizeGpu() from binary-level
   code because it hasn't become necessary yet; the only program that might have
   multiple threads actually using the GPU is rnnlm-train (if the user were to
   invoke it with the ,bg option for loading training examples); but the only
   CUDA invocation the RnnlmExample::Read() function uses (via
   CuMatrix::Read()), is cudaMemcpy, which is synchronous already.

*/
void SynchronizeGpu();

}   // namespace kaldi

#endif // KALDI_CUDAMATRIX_CU_DEVICE_H_
