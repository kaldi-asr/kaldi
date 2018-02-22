// cudamatrix/cu-allocator.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrix.h"
#include "base/kaldi-error.h"
#include "base/kaldi-utils.h"
#include "util/common-utils.h"

namespace kaldi {


void* CuMemoryAllocator::Malloc(size_t size) {
  // For now just call MallocPitch and throw away the pitch, to avoid
  // duplicating code here.  Apparently the time difference is quite small.
  size_t pitch;
  return MallocPitch(size, 1, &pitch);
}

// Returns max(0, floor(log_2(i))).   Not tested independently.
static inline size_t IntegerLog2(size_t i) {
  size_t ans = 0;
  while (i > 256) {
    i >>= 8;
    ans += 8;
  }
  while (i > 16) {
    i >>= 4;
    ans += 4;
  }
  while (i > 1) {
    i >>= 1;
    ans++;
  }
  return ans;
}

//inline
CuMemoryAllocator::MruCache& CuMemoryAllocator::GetCacheForSize(
    size_t num_bytes) {
  size_t bucket_index = IntegerLog2(num_bytes);
  KALDI_ASSERT(num_bytes > 0 && bucket_index < caches_.size());
  return caches_[bucket_index];
}

//inline
void* CuMemoryAllocator::MallocPitchInternal(size_t row_bytes,
                                             size_t num_rows,
                                            size_t *pitch) {
  num_system_allocations_++;
  void *ans;
  cudaError_t e;
  for (int32 i = 0; i <= 2; i++) {
    if (num_rows != 1) {
      CuTimer tim;
      e = cudaMallocPitch(&ans, pitch, row_bytes, num_rows);
      tot_time_taken_in_cuda_malloc_pitch_ += tim.Elapsed();
    } else {
      CuTimer tim;
      // we might save a little time this way.
      e = cudaMalloc(&ans, row_bytes);
      tot_time_taken_in_cuda_malloc_ += tim.Elapsed();
      *pitch = row_bytes;
    }
    if (e != cudaSuccess) {
      PrintMemoryUsage();
      // On the first 2 out of the 3 iters, try freeing memory.
      if (i <= 1) {
        KALDI_WARN << "Allocation of " << row_bytes << " x "
                   << num_rows << " region failed: freeing some memory and "
                   << "trying again. ";
        BaseFloat new_memory_factor = 1.1;
        if (opts_.memory_factor > new_memory_factor) {
          KALDI_LOG << "To avoid future problems like this, changing "
                    << "memory_factor from " << opts_.memory_factor << " to "
                    << new_memory_factor;
          opts_.memory_factor = new_memory_factor;
        }
        size_t memory_cached = MemoryCached(),
            memory_requested = row_bytes * num_rows,
            memory_to_free = std::max<size_t>(memory_cached / 2,
                                              std::min<size_t>(memory_cached,
                                                               memory_requested));
        FreeSomeCachedMemory(memory_to_free);
      } else {
        KALDI_ERR << "Cannot allocate the requested memory ("
                  << row_bytes << " x " << num_rows << " = "
                  << row_bytes * num_rows << " bytes)";
      }
      cudaGetLastError();  // Clear the error state.
    } else {
      break;
    }
  }
  return ans;
}

void CuMemoryAllocator::PrintMemoryUsage() const {
  KALDI_LOG << "Memory usage: " << cur_bytes_allocated_
            << " bytes currently allocated (max: "
            << max_bytes_allocated_ << "); " << cur_bytes_used_
            << " currently in use by user (max: " << max_bytes_used_ << ")"
            << "; " << num_system_allocations_ << '/'
            << num_user_allocations_ << " calls to Malloc* resulted in "
            << "CUDA calls.";
  if (GetVerboseLevel() >= 1) {
    // CuTimer only accumulates stats at verbose level 1 or above.
    KALDI_LOG << "Time taken in cudaMallocPitch=" << tot_time_taken_in_cuda_malloc_pitch_
              << ", in cudaMalloc=" << tot_time_taken_in_cuda_malloc_
              << ", in cudaFree=" << tot_time_taken_in_cuda_free_
              << ", in this->MallocPitch()=" << tot_time_taken_in_malloc_pitch_;
  }
}

CuMemoryAllocator::CuMemoryAllocator(CuAllocatorOptions opts):
    opts_(opts),
    caches_(40),
    cur_bytes_allocated_(0),
    max_bytes_allocated_(0),
    cur_bytes_used_(0),
    max_bytes_used_(0),
    t_(1),
    num_user_allocations_(0),
    num_system_allocations_(0),
    tot_time_taken_in_cuda_malloc_(0.0),
    tot_time_taken_in_cuda_malloc_pitch_(0.0),
    tot_time_taken_in_cuda_free_(0.0),
    tot_time_taken_in_malloc_pitch_(0.0) { }

void* CuMemoryAllocator::MallocPitch(size_t row_bytes,
                                     size_t num_rows,
                                     size_t *pitch) {
  CuTimer tim;
  t_++;
  num_user_allocations_++;
  size_t requested_bytes = row_bytes * num_rows;
  if (cur_bytes_used_ + requested_bytes > max_bytes_used_)
    max_bytes_used_ = cur_bytes_used_ + requested_bytes;
  MruCache &cache = GetCacheForSize(requested_bytes);
  MemoryRequest request(row_bytes, num_rows);
  CachedMemoryElement output;
  if (cache.Lookup(request, &output)) {
    // we have cached memory with this value.
    void *ans = output.pointer;
    *pitch = output.pitch;
    used_map_[ans] = UsedMemoryElement(row_bytes, num_rows, output.pitch);
    cur_bytes_used_ += requested_bytes;
    tot_time_taken_in_malloc_pitch_ += tim.Elapsed();
    return ans;
  } else {
    // note: it's important that we already updated max_bytes_used_.
    size_t next_bytes_allocated = cur_bytes_allocated_ + requested_bytes,
        max_bytes_to_allocate =
        static_cast<size_t>(opts_.memory_factor * max_bytes_used_);
    ssize_t bytes_overflow = next_bytes_allocated - max_bytes_to_allocate;
    if (bytes_overflow > 0) {
      // The amount we would have allocated, after fulfilling this request,
      // would exceed our limits (we don't allow ourselves to allocate more than
      // memory_factor times the maximum amount of memory the user ever owns
      // during the lifetime of the program).  So free some memory.
      KALDI_ASSERT(bytes_overflow <= MemoryCached());  // sanity check.
      FreeSomeCachedMemory(static_cast<size_t>(bytes_overflow));
      KALDI_ASSERT(cur_bytes_allocated_ + requested_bytes <=
                   max_bytes_to_allocate);
    }
    void *ans = MallocPitchInternal(row_bytes, num_rows, pitch);
    cur_bytes_allocated_ += requested_bytes;
    if (cur_bytes_allocated_ > max_bytes_allocated_)
      max_bytes_allocated_ = cur_bytes_allocated_;
    used_map_[ans] = UsedMemoryElement(row_bytes, num_rows, *pitch);
    cur_bytes_used_ += requested_bytes;
    tot_time_taken_in_malloc_pitch_ += tim.Elapsed();
    return ans;
  }
}

void CuMemoryAllocator::FreeSomeCachedMemory(size_t bytes_to_free_in) {
  CuTimer tim;
  // the next few lines are responsible for increasing the amount of memory we
  // are going to free, in case the user requested an amount that's very tiny
  // compared with the total amount of memory ever used.  This helps us
  // to amortize the cost of visiting all of the buckets inside this code.
  // (there are only 40 buckets so it's not so big, but we're being careful.
  size_t bytes_cached = cur_bytes_allocated_ - cur_bytes_used_,
      min_to_free = static_cast<size_t>(max_bytes_used_ * opts_.delete_factor);
  size_t bytes_to_free = std::min(bytes_cached,
                                  std::max(bytes_to_free_in, min_to_free)),
      bytes_freed = 0;

  size_t num_caches = caches_.size(),
      t = t_;
  // size_factor contains the approximate (power-of-two) size of the pointers
  // that each cache's pointers contain.  The 'cost' of keeping any given pointer,
  // we declare to be the time since we last used it multiplied by the size
  // of the memory in the pointer.
  std::vector<BaseFloat> size_factor(num_caches);
  for (size_t i = 0, j=1; i < num_caches; i++, j *= 2)
    size_factor[i] = j;

  std::priority_queue<std::pair<BaseFloat,int32> > queue;
  // Set up the queue.
  for (int32 i = 0; i < num_caches; i++) {
    const MruCache &cache = caches_[i];
    size_t cache_t = cache.LeastRecentTime();
    if (cache_t > 0) {  // t == 0 means the cache is empty.
      size_t interval = t - cache_t;
      BaseFloat cost = size_factor[i] * interval;
      KALDI_ASSERT(interval > 0);
      queue.push(std::pair<BaseFloat,int32>(cost, i));
    }
  }
  while (bytes_freed < bytes_to_free) {
    // If the following fails it means I made some kind of bookkeeping error,
    // and most likely we are trying to free more memory than we really have
    // cached.
    KALDI_ASSERT(!queue.empty() && "Code error.");
    std::pair<BaseFloat, int32> p = queue.top();
    int32 cache_index = p.second;
    MruCache &cache = caches_[cache_index];
    queue.pop();
    if (queue.empty()) {
      while (bytes_freed < bytes_to_free) {
        bytes_freed += cache.RemoveLeastRecentlyUsed();
      }
    } else {
      BaseFloat next_worst_cost = queue.top().first;
      while (1)  {
        bytes_freed += cache.RemoveLeastRecentlyUsed();
        if (bytes_freed >= bytes_to_free)
          break;
        size_t least_recent_time = cache.LeastRecentTime();
        if (least_recent_time == 0)  // this cache is now empty
          break;
        size_t interval = t - least_recent_time;
        KALDI_ASSERT(interval > 0);
        BaseFloat cost = size_factor[cache_index] * interval;
        if (cost < next_worst_cost) {
          // There is another bucket that has worse cost than this,
          // so stop processing this bucket-- but first put it
          // back in the queue.
          queue.push(std::pair<BaseFloat, int32>(cost, cache_index));
          break;
        }
      }
    }
  }
  KALDI_ASSERT(bytes_freed <= cur_bytes_allocated_);
  cur_bytes_allocated_ -= bytes_freed;
  tot_time_taken_in_cuda_free_ += tim.Elapsed();
}

void CuMemoryAllocator::Free(void *ptr) {
  t_++;
  unordered_map<void*, UsedMemoryElement, PointerHasher>::iterator iter =
      used_map_.find(ptr);
  if (iter == used_map_.end()) {
    KALDI_ERR << "Attempt to free CUDA memory pointer that was not allocated: "
              << ptr;
  }
  const UsedMemoryElement &elem = iter->second;
  size_t num_bytes = elem.row_bytes * elem.num_rows;

  cur_bytes_used_ -= num_bytes;
  MruCache &cache = GetCacheForSize(num_bytes);

  cache.Insert(MemoryRequest(elem.row_bytes, elem.num_rows),
               CachedMemoryElement(ptr, t_, elem.pitch));
  used_map_.erase(iter);
}

size_t CuMemoryAllocator::MruCache::LeastRecentTime() const {
  if (list_.empty()) {
    KALDI_PARANOID_ASSERT(map_.empty());
    return 0;
  } else {
    const MemoryRequest &mr = list_.front();
    MapType::const_iterator iter = map_.find(mr);
    KALDI_ASSERT(iter != map_.end());
    const MapValueType &queue = iter->second;
    KALDI_ASSERT(!queue.empty());
    return queue.front().first.t;
  }
}

bool CuMemoryAllocator::MruCache::Lookup(const MemoryRequest &request,
                                         CachedMemoryElement *output) {
  MapType::iterator iter = map_.find(request);
  if (iter == map_.end())
    return false;
  MapValueType &q = iter->second;
  KALDI_ASSERT(!q.empty());
  // use q.back() as we want to return the most recently used one if there
  // is a choice.  We believe this will give better caching behavior.
  *output = q.back().first;
  list_.erase(q.back().second);
  q.pop_back();
  if (q.empty())
    map_.erase(request);
  return true;
}

void CuMemoryAllocator::MruCache::Insert(const MemoryRequest &request,
                                         const CachedMemoryElement &element) {
  list_.push_back(request);
  map_[request].push_back(std::pair<CachedMemoryElement, ListIterType>(
      element,
      --list_.end()));
}

size_t CuMemoryAllocator::MruCache::RemoveLeastRecentlyUsed() {
  // Remove least-recently-used element from cache.
  KALDI_ASSERT(!list_.empty());
  MemoryRequest request = list_.front();
  MapType::iterator iter = map_.find(request);
  KALDI_ASSERT(iter != map_.end());
  MapValueType &queue = iter->second;
  KALDI_ASSERT(!queue.empty());
  // least recently used elements are at the front of the queue.
  std::pair<CachedMemoryElement, ListIterType> &p = queue.front();
  KALDI_ASSERT(p.second == list_.begin());
  CU_SAFE_CALL(cudaFree(p.first.pointer));
  queue.pop_front();
  if (queue.empty())
    map_.erase(request);
  list_.pop_front();
  return request.first * request.second;
}

CuMemoryAllocator::MruCache& CuMemoryAllocator::MruCache::operator = (
    const CuMemoryAllocator::MruCache &other) {
  KALDI_ASSERT(other.list_.empty());
  return *this;
}
CuMemoryAllocator::MruCache::MruCache(
    const CuMemoryAllocator::MruCache &other) {
  KALDI_ASSERT(other.list_.empty());
}




}


#endif // HAVE_CUDA
