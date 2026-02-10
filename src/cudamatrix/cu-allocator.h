// cudamatrix/cu-allocator.h

// Copyright 2015   Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_ALLOCATOR_H_
#define KALDI_CUDAMATRIX_CU_ALLOCATOR_H_

#if HAVE_CUDA == 1
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <map>
#include <set>
#include <mutex>
#include <list>
#include <queue>
#include <thread>
#include <iostream>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"
#include "itf/options-itf.h"

namespace kaldi {


// For now we don't give the user a way to modify these from the command line.
// or the code, it just documents what the default options are.  To change
// the options, you have to do it in the code.
struct CuAllocatorOptions {
  // True if we are going to actually cache memory allocations on this device.
  // You'd normally set it to false only if you wanted to debug a possible
  // memory problem using cuda-memcheck or cuda-gdb.  It will be slower, but
  // using CUDA's native allocator allows those tools to detect out-of-region
  // memory accesses.
  bool cache_memory;

  // The proportion of the device's memory that the CuAllocator allocates to
  // start with; by default this is 0.5, although if you want to share the
  // device (not recommended!) you should set this lower.
  BaseFloat memory_proportion;

  // The target number of subregions of the entire CUDA device memory (we'll
  // start with a smaller number of memory_proportion is << 1).  Kind of
  // a tuning knob.. more regions will make it more aggressively consolidate
  // memory low addresses.
  int32 num_subregions;

  CuAllocatorOptions():
      cache_memory(true), memory_proportion(0.5), num_subregions(20) { }

  void Register(OptionsItf *po) {
    po->Register("cuda-cache-memory", &cache_memory, "True if you want "
                 "to use the caching allocator.  Set this to false only if you "
                 "want to use cuda-memcheck or cuda-gdb; it will be slower.");
    po->Register("cuda-memory-proportion", &memory_proportion,
                 "Proportion of the GPU device memory that the allocator "
                 "should allocate at the start");
  }

  void Check() {
    // don't let it get too close to 1;
    KALDI_ASSERT(memory_proportion >= 0.05 && memory_proportion < 0.99);
  }
};

extern CuAllocatorOptions g_allocator_options;

inline void RegisterCuAllocatorOptions(OptionsItf *po) {
  g_allocator_options.Register(po);
}


} // namespace kaldi


#if HAVE_CUDA == 1
namespace kaldi {

/**
   This class allocates large regions of memory from the GPU and allocates
   sub-blocks of it for the user.  This is needed because the CUDA malloc and
   free routines are very slow.

   The user doesn't access this class directly, it is accessed via the CuDevice
   object.  The CuDevice class allocates memory using this class's Malloc() and
   MallocPitch() functions, and frees them with its Free() function, and this
   class caches the memory blocks to avoid calling the CUDA library's
   malloc/free functions too often.  If the application is using multiple
   threads, it's necessary to lock this class before using it, and in that case
   the CuDevice class calls the MallocLocking() and MallocPitchLocking()
   versions of the allocation functions (but the user should call
   CuDevice::AllowMultithreading() if the application plans to use GPU
   functionality from multiple CPU threads).

   NOTE ON SYNCHRONIZATION: if multiple CUDA streams are used there is a
   potential problem with any caching allocator which shares its pool across
   CUDA streams.  That is: if a memory block is freed by stream 1 and allocated to
   stream 2, an operation might start in stream 2 before stream 1 has finished
   working with that memory location.  We solve this here using a rather low-tech
   solution, relying on calling SynchronizeGpu() which submits a no-op kernel
   into the legacy default stream.  Each
   time CuMemoryAllocator()::Free() is called and we cache the memory block
   in this class, we record the thread-id of the CPU thread from which it was
   freed, as well as a timestamp (the t_ member of CuMemoryAllocator, which
   we increment every time the class is used).  When we allocate memory
   that was cached, we try to allocate it from a block that was relased by the
   same CPU thread; and if that is not possible and we haven't called
   SynchronizeGpu() since the block was freed, then we call
   SynchronizeGpu().  The hope is that this will happen quite rarely.
   Note that this is based on the assumption that the user is using the
   per-thread default stream (indeed this is how we compile).  If the
   user were to make explicit use of CUDA streams, this mechanism would
   not necessarily be sufficient to prevent data-race conditions and the
   user might have to take further precautions.

   NOTE ON FRAGMENTATION: Memory fragmentation is one of the main problems that
   you'll run into with allocators like this.  This allocator will allocate a
   small number of large regions of memory, and allocate smaller pieces of
   memory that it splits off from the regions as needed.  It will always merge
   adjacent blocks as much as it can when the user frees memory.  The main
   heuristic to avoid memory fragmenting too much is that it always allocates,
   where possible, from memory that's as close as possible to the start of a
   memory region.  This will tend to keep all the small allocations together at
   the beginning of the memory region, and hopefully keep large blocks availale
   at the end.  The mechanism to always allocate from as close as possible to
   the start of the memory region, is that we split up the memory regions into
   a small number of sub-regions and, when handling a request for allocation,
   allocate it from the lowest-numbered sub-region that can meet a request for
   that size.  (Note: we can allocate blocks that span sub-regions, so this
   approach does not limit the block size we can allocate).

*/

class CuMemoryAllocator {
 public:
  /// Allocates memory on the CUDA device, of size 'size'.  size == 0 is not
  /// allowed and is an error.
  void* Malloc(size_t size);

  /// Allocation function for matrix-like things.
  void* MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch);

  /// Free device memory allocated by Malloc() or MallocPitch().
  void Free(void *ptr);

  /// Mutex-guarded version of Malloc(), for use in multi-threaded programs.
  inline void* MallocLocking(size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
    return Malloc(size);
  }
  /// Mutex-guarded version of Malloc(), for use in multi-threaded programs.
  inline void* MallocPitchLocking(size_t row_bytes, size_t num_rows, size_t *pitch) {
    std::unique_lock<std::mutex> lock(mutex_);
    return MallocPitch(row_bytes, num_rows, pitch);
  }
  /// Mutex-guarded version of Free(), for use in multi-threaded programs.
  void FreeLocking(void *ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    Free(ptr);
  }

  void PrintMemoryUsage() const;

  // returns the current memory allocated within the cache
  size_t GetAllocatedMemory() { return allocated_memory_; }

  //  returns the maximum memory used within the cache during current execution
  size_t GetMaxAllocatedMemory() { return max_allocated_memory_; }

  CuMemoryAllocator();

  // Allows you to set options: must be called before any Malloc function is
  // called on this class.  It's done this way so the options can be changed
  // by the user (c.f. RegisterCuAllocatorOptions()) before the options are read.
  void SetOptions(const CuAllocatorOptions &opts) { opts_ = opts; }

  ~CuMemoryAllocator();

 private:

  struct SubRegion;

  struct MemoryBlock {
    char *begin;  // The beginning of the block (in CUDA memory)
    char *end;  // the end of the block (in CUDA memory)
    SubRegion *subregion;  // Pointer to the SubRegion to which this memory
                            // block belongs.
    bool allocated;  // True if this MemoryBlock has currently been given to the
                     // user; false if not.

    size_t t;        // Zero if this memory block was never given to the user;
                     // otherwise, the time value (t_ in the CuAllocator class)
                     // when it was most recently either allocated to the user
                     // or freed by the user.

    std::thread::id thread_id;  // If allocated == false and t > 0 (i.e. this
                                // memory block was released by the user), the
                                // thread-id of the user thread that freed this
                                // block, or the invalid thread-id as created by
                                // the constructor of std::thread::id if this
                                // block was created by merging blocks from
                                // different threads.  Required for
                                // synchronization; and note that we assume
                                // there is one CUDA stream per CPU thread.

    MemoryBlock *next;  // The next MemoryBlock within this MemoryRegion (or
                        // NULL if this is the last one); its 'begin' would be
                        // the same as the 'end' of this block.
    MemoryBlock *prev;  // The previous MemoryBlock within this MemoryRegion (or
                        // NULL if this is the first one); its 'end' would be the
                        // same as the 'begin' of this block.

  };

  // a MemoryRegion is a large piece of memory that we allocated via CudaMalloc.
  // there normally won't be more than about 3 or 4 of these.
  // We'll identify MemoryRegions by a size_t (e.g 0, 1, 2, 3... ) which is an
  // index into the memory_regions_ vector.
  struct MemoryRegion {
    char *begin;  // 'begin' is the start of the memory region.
    char *end;  // 'end' is the end of the memory region.
    SubRegion *subregion_begin;  // The first SubRegion that belongs to this
                                 // MemoryRegion.
    MemoryBlock *block_begin;  // The first MemoryBlock that belongs to this
                               // MemoryRegion.
  };

  // a SubRegion is a smaller zone of memory within a MemoryRegion.  For
  // example, we divide the first MemoryRegion we allocate into 10 blocks, and
  // if we allocate blocks of memory later on, we'll sub-divide them into blocks
  // of about the same size.  A SubRegion is just a largish bin into which we
  // put any blocks of memory that happen to start within that SubRegion;
  // actually, memory blocks may cross over the boundaries of SubRegions.  The
  // motivation for dividing up MemoryRegions into SubRegions is that it allos
  // us an efficient mechanism to segregate smaller memory blocks into higher
  // memory and larger ones into lower memory: for each allocation, we allocate
  // it from the highest-numbered SubRegion that is able to allocate something of
  // that size.  Over time, this will lead to smaller memory blocks being
  // concentrated in higher-numbered SubRegions.
  struct SubRegion {
    size_t memory_region;  // This is an index into the memory_regions_ vector
                           // which identifies which MemoryRegion this SubRegion
                           // is a part of.
    size_t subregion_index;  // The index of this SubRegion within the
                             // subregions_ vector; this can change when we
                             // allocate more MemoryRegions.
    char *begin;  // 'begin' is the start of the memory in this SubRegion.
    char *end;    // 'end' is the end of the memory in this SubRegion.

    // Contains the free MemoryBlocks starting within this SubRegion.
    std::set<std::pair<size_t, MemoryBlock*> > free_blocks;

    // Pointer to the next SubRegion within this MemoryRegion (i.e. the SubRegion
    // whose begin equals this one's end), or NULL if this is the last one.
    SubRegion *next;
  };

  // Tries to allocate CUDA memory of the given size; will crash if it was not
  // able to.
  inline void* MallocInternal(size_t size);

  // Allocates from a given SubRegion, after we have determined that it
  // can satisfy this request.  Broken out of MallocInternal for clarity.
  inline void* MallocFromSubregion(SubRegion *subregion, size_t size);


  // Splits the given MemoryBlock so that one piece is of size 'size', and
  // returns the piece which is of size 'size'.  The caller guarantees that
  // 'size' is less than the current size of the memory block, that 'block' is
  // not currently allocated (i.e. block->allocated == false).  This function
  // assumes that, at entry, 'block' is not present in its subregion's
  // 'free_blocks' (because the caller has removed it), and it takes
  // responsibility for entering the 'unused' part (the part we're not
  // returning) into its subregion's 'free_blocks' by calling AddToFreeBlocks().
  inline MemoryBlock *SplitBlock(MemoryBlock *block, size_t size);

  // Removes this block from the 'free_blocks' set of the SubRegion to which
  // it belongs.  This is called when allocating a block, and from other places.
  void RemoveFromFreeBlocks(MemoryBlock *block);

  // Adds this block to the 'free_blocks' set of the SubRegion to which it
  // belongs.  This is called when freeing a block, and from other places.
  void AddToFreeBlocks(MemoryBlock *block);

  // This function is called when an allocation failed and we need to try to
  // allocate more memory from the evice.  The 'size' is the size of the
  // requested memory block whose allocation failed-- it's provided so that
  // we can be sure to allocate a new region of at least this size.
  void AllocateNewRegion(size_t size);

  // Called from AllocateNewRegion(), this ensures that the subregions are
  // sorted as we want (which is a kind of heuristic that will be discussed in
  // the code), and it also recomputes the largest_free_block_ array.
  void SortSubregions();



  CuAllocatorOptions opts_;

  std::vector<MemoryRegion> memory_regions_;

  std::vector<SubRegion*> subregions_;

  // For each SubRegion in sub_regions_, this vector gives us the size of the
  // largest free block present in that SubRegion, which is equal to
  // sub_regions_[i]->free_blocks.begin()->first.  It allows us to fairly
  // efficiently find the lowest-numbered SubRegion which can handle a
  // particular request for memory.
  std::vector<size_t> largest_free_block_;

  size_t t_;  // time counter, incremented with each call.
  size_t synchronize_gpu_t_;     // value of t_ at the last time we called
                                 // SynchronizeGpu().
  size_t num_synchronizations_;  // number of times we called SynchronizeGpu()
  double tot_time_taken_;  // Total time taken in calls to this object.
  double malloc_time_taken_;  // Total time we spent calling cudaMalloc().

  // This is a map from memory locations currently owned by the user, to the
  // MemoryBlock which stores the information about that location.
  std::unordered_map<void*, MemoryBlock*> allocated_block_map_;

  // this is only locked by the '*Locking' versions of the functions (necessary only
  // in multi-threaded applications).
  std::mutex mutex_;

  // Keep track of the memory usage from the cache to track the maximum memory used by
  //   the application
  size_t max_allocated_memory_;
  size_t allocated_memory_;
};


// This function returns some printable information about the memory used
// as a string: an example showing the format is:
//  "free: 10M, used: 490M, total: 500M: free/total: 0.02"
// In addition, if the pointers 'free' and 'total' are non-NULL, it will
// output to them the free memory and the total memory of the device.
std::string GetFreeGpuMemory(int64* free, int64* total);

extern CuMemoryAllocator g_cuda_allocator;

}  // namespace kaldi

#endif // HAVE_CUDA


#endif
