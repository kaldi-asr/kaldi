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
#include <map>
#include <mutex>
#include <list>
#include <queue>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"

namespace kaldi {


// For now we don't give the user a way to modify these from the command line.
struct CuAllocatorOptions {
  // memory_factor is the total amount of (allocated + cached) memory that we
  // allow to be held, relative to the max amount of memory the program has ever
  // allocated.  It will increase the amount of memory the program will
  // potentially consume, by this factor.
  BaseFloat memory_factor;

  // This is the minimum amount of memory that we will delete when we are forced
  // to delete stuff, relative to the max amount of memory the program has ever
  // allocated.  This should be less than memory_factor - 1.0 and > 0.  It
  // shouldn't be too critical.  The reason it exists is to avoid calling the
  // cleanup code and only releasing very small amounts of memory, because there
  // is a constant overhead proportional to the number of buckets.
  BaseFloat delete_factor;

  CuAllocatorOptions(): memory_factor(1.3),
                        delete_factor(0.001) { }

  void Check() {
    KALDI_ASSERT(delete_factor < memory_factor - 1.0 && delete_factor > 0.0);
  }
};




// Class that caches memory for us (the CUDA
// malloc and free routines are very slow).
// This is a member of the CuDevice class.
class CuMemoryAllocator {
 public:
  /// Allocates memory on the CUDA device, of size 'size'.
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


  // the maximum amount of memory that was ever allocated in the lifetime of the
  // program, in bytes.
  size_t MaxMemoryAllocated() const { return max_bytes_allocated_; }

  // memory held in the cache currently, in bytes.
  size_t MemoryCached() const { return cur_bytes_allocated_ - cur_bytes_used_; }

  // memory that's cached plus memory that's allocated, in bytes.
  size_t MemoryAllocated() const { return cur_bytes_allocated_; }

  void PrintMemoryUsage() const;

  CuMemoryAllocator(CuAllocatorOptions opts);
 private:

  void FreeSomeCachedMemory(size_t bytes_to_free);

  // This calls CudaMallocPitch, checks for errors (dies if it has to), and
  // returns the result.  It's up to the caller to do all the bookkeeping though.
  inline void* MallocPitchInternal(size_t row_bytes, size_t num_rows, size_t *pitch);

  typedef std::pair<size_t, size_t> MemoryRequest;  // (row_bytes, num_rows).
  struct CachedMemoryElement {
    void *pointer;  // the CUDA memory location that we own
    size_t t;       // time value when we put this in the cache.
    size_t pitch;   // pitch of this memory region (c.f. cudaMallocPitch()).
    CachedMemoryElement() { }
    CachedMemoryElement(void *pointer, size_t t, size_t pitch):
        pointer(pointer), t(t), pitch(pitch) { }
  };

  // This class caches a map from MemoryRequest to a list of CachedMemoryElements,
  // and gives us access to the least-recently-used element for efficient.
  // removal.
  // We will have an instance of this class for each power-of-2 of size in
  // bytes.  This makes it easier to, when we need to delete something, find
  // the item for which the (time-since-used * size-in-bytes) is approximately
  // greatest.
  class MruCache {
   public:
    size_t LeastRecentTime() const;  // t value of least recent CachedMemoryElement (0
                                     // if empty).

    size_t RemoveLeastRecentlyUsed();  // Remove least-recently-used element
                                       // from cache.  Return size in bytes of
                                       // that removed memory region.  Crash if
                                       // this was empty.

    // Attempts lookup of the most recently cached element corresponding to
    // 'request'.  If available, removes it from the cache and puts it to
    // 'output', and returns true.  Otherwise returns false.
    bool Lookup(const MemoryRequest &request,
                CachedMemoryElement *output);

    // Inserts this CachedMemoryElement to the list of CachedMemoryElements for this
    // MemoryRequest.  The time in the CachedMemoryElement is expected to be greater
    // than times in previously supplied CachedMemoryElements.
    void Insert(const MemoryRequest &request,
                const CachedMemoryElement &element);

    struct MemoryRequestHasher {
      // input is interpreted as (row_bytes, num_rows).  row_bytes will always
      // be a multiple of 4, and num_rows will frequently be a multiple of
      // powers of 2 also.  We need to shift right and add so that there will be
      // some action in the lower-order bits.
      size_t operator () (const std::pair<size_t,size_t> &p) const noexcept {
        size_t temp = p.first + 1867 * p.second;
        return temp + (temp >> 2) + (temp >> 8);
      }
    };

    MruCache() { }
    // Define these to make inclusion in std::vector possible, but make them
    // fail if called on anything but empty cache objects-- we never resize
    // the vector of caches after initializing it.
    MruCache &operator = (const MruCache &other);
    MruCache(const MruCache &other);
   private:
    typedef std::list<MemoryRequest> ListType;
    typedef std::list<MemoryRequest>::iterator ListIterType;
    typedef std::deque<std::pair<CachedMemoryElement, ListIterType> > MapValueType;
    typedef unordered_map<MemoryRequest, MapValueType,
                          MemoryRequestHasher> MapType;
    // 'list_' contains MemoryRequests with the most recent on the back (where they are added),
    // and least recent on the front (where they are removed by RemoveLeastRecentlyUsed, although
    // they are also removed from random parts of the list by Lookup().
    // There will in general be duplicates of MemoryRequests in the list, as
    // many as there are entries in the MapValueType.
    ListType list_;
    // 'map_' maps from a MemoryRequest to a queue of (memory-element,
    // iterator), with the most-recently-added things at the back; we remove
    // things from the front of these queues (oldest) inside
    // RemoveLeastRecentlyUsed(), and from the back (newest) in Lookup.
    MapType map_;
  };


  inline MruCache &GetCacheForSize(size_t num_bytes);

  CuAllocatorOptions opts_;

  // indexed by log_2 (amount of memory requested), the caches.
  std::vector<MruCache> caches_;

  size_t cur_bytes_allocated_;  // number of bytes currently owned by callers or
                                // cached.
  size_t max_bytes_allocated_;  // the max over all time, of cur_bytes_allocated_.
  size_t cur_bytes_used_;  // number of bytes currently owned by callers.
  size_t max_bytes_used_;  // the max over all time, of cur_bytes_used_.
  size_t t_;  // time counter, incremented with each call.
  size_t num_user_allocations_;  // number of times user calls Malloc*
  size_t num_system_allocations_;  // number of times we call cudaMalloc*.
  double tot_time_taken_in_cuda_malloc_;  // time in cudaMalloc
  double tot_time_taken_in_cuda_malloc_pitch_;  // time in cudaMallocPitch
  double tot_time_taken_in_cuda_free_;  // time in cudaFree
  double tot_time_taken_in_malloc_pitch_;  // time in this->MallocPitch()


  // a memory element is 'used' when it is currently possessed by the caller
  // (and is not in our cache).
  struct UsedMemoryElement {
    size_t row_bytes;
    size_t num_rows;
    size_t pitch;
    UsedMemoryElement() { }
    UsedMemoryElement(size_t row_bytes, size_t num_rows, size_t pitch):
        row_bytes(row_bytes), num_rows(num_rows), pitch(pitch)  { }
  };

  struct PointerHasher {
    size_t operator() (const void *arg) const noexcept {
      // the last few bits tend to be very predictable, for alignment reasons (CUDA
      // allocation may align on 256 byte or 512 byte boundaries or something similar).
      size_t temp = reinterpret_cast<size_t>(arg);
      return (temp >> 4) + (temp >> 9);
    }
  };

  // This is a map from memory locations owned by the user, so we can recover
  // the information when people call Free() and we add it back into the cache.
  unordered_map<void*, UsedMemoryElement, PointerHasher> used_map_;

  // this is only locked by the '*Locking' versions of the functions.
  std::mutex mutex_;

};


}  // namespace

#endif // HAVE_CUDA


#endif
