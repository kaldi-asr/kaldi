// cudamatrix/cu-allocator.cc

// Copyright      2015-2018  Johns Hopkins University (author: Daniel Povey)

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



#include "cudamatrix/cu-allocator.h"

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

namespace kaldi {


void* CuMemoryAllocator::Malloc(size_t size) {
  Timer tim;
  if (!opts_.cache_memory) {
    void *ans;
    CU_SAFE_CALL(cudaMalloc(&ans, size));
    double elapsed = tim.Elapsed();
    tot_time_taken_ += elapsed;
    malloc_time_taken_ += elapsed;
    t_++;
    return ans;
  }

  // We could perhaps change this to KALDI_PARANOID_ASSERT to save time.
  KALDI_ASSERT(size != 0);

  // Round up 'size' to a multiple of 256; this ensures the right kind of
  // memory alignment.
  size = (size + 255) & ~((size_t)255);
  void *ans = MallocInternal(size);
  tot_time_taken_ += tim.Elapsed();
  return ans;
}


CuMemoryAllocator::MemoryBlock *CuMemoryAllocator::SplitBlock(
    MemoryBlock *block, size_t size) {
  SubRegion *subregion = block->subregion;
  // new_block will become the right-most part of 'block', and 'block' will
  // be the left-most part.
  MemoryBlock *new_block = new MemoryBlock;
  bool return_new_block;
  char *new_begin;

  // We now decide whether to make the left part of 'block' be of size ('size')
  // and return it (the 'if' branch of the if-else block below), or the right
  // part (the 'else' branch).  We decide this based on heuristics.  Basically,
  // we want to allocate the sub-block that's either next to the edge of the
  // MemoryRegion, or next to something that was allocated long ago (and which,
  // we assume won't be deallocated for a relatively long time).  That is: we
  // want to leave the un-allocated memory next to a memory block that was
  // recently allocated (and thus is likely to be freed sooner), so that when
  // that block is freed we can merge it with the still-unallocated piece into a
  // larger block; this will reduce fragmentation.  But if this block spans
  // multiple sub-regions we don't want to do that, as that would be against our
  // heuristic of, where possible, allocating memory from lower-numbered
  // sub-regions.
  //
  // Bear in mind that we can assume block->next and block->prev, if they are
  // non-NULL, are both currently allocated, since 'block' is un-allocated and
  // we would have merged any adjacent un-allocated sub-regions.
  if (block->next != NULL && block->prev != NULL &&
      block->prev->t < block->next->t &&
      block->next->subregion == subregion) {
    // We'll allocate the right part of the block, since the left side is next
    // to a relatively recently-allocated block.
    return_new_block = true;
    new_begin = block->end - size;
  } else {
    // We'll allocate the left part of the block.
    return_new_block = false;
    new_begin = block->begin + size;
  }

  // The following code makes sure the SubRegion for 'new_block' is correct,
  // i.e. its 'begin' is >= the 'begin' of the subregion and < the 'end' of the
  // subregion.  If the following loop segfaults, it indicates a bug somewhere
  // else.
  while (new_begin >= subregion->end)
    subregion = subregion->next;
  MemoryBlock *next_block = block->next;
  new_block->begin = new_begin;
  new_block->end = block->end;
  new_block->subregion = subregion;
  new_block->allocated = false;
  new_block->thread_id = block->thread_id;
  new_block->t = block->t;
  new_block->next = next_block;
  new_block->prev = block;
  if (next_block)
    next_block->prev = new_block;
  block->next = new_block;
  block->end = new_begin;

  // Add the split-up piece that we won't be allocating, to the
  // 'free_blocks' member of its subregion.
  if (return_new_block) {
    AddToFreeBlocks(block);
    return new_block;
  } else {
    AddToFreeBlocks(new_block);
    return block;
  }
}


void CuMemoryAllocator::RemoveFromFreeBlocks(MemoryBlock *block) {
  SubRegion *subregion = block->subregion;
  size_t block_size = block->end - block->begin;
  std::pair<size_t, MemoryBlock*> p(block_size, block);
  size_t num_removed = subregion->free_blocks.erase(p);
  KALDI_ASSERT(num_removed != 0);
  // Update largest_free_block_, if needed.
  size_t subregion_index = subregion->subregion_index;
  if (block_size == largest_free_block_[subregion_index]) {
    if (subregion->free_blocks.empty())
      largest_free_block_[subregion_index] = 0;
    else
      largest_free_block_[subregion_index] =
          subregion->free_blocks.rbegin()->first;
  }
}

void CuMemoryAllocator::AddToFreeBlocks(MemoryBlock *block) {
  SubRegion *subregion = block->subregion;
  KALDI_PARANOID_ASSERT(block->begin >= subregion->begin &&
                        block->begin < subregion->end);
  size_t block_size = block->end - block->begin,
       subregion_index = subregion->subregion_index;
  // Update largest_free_block_, if needed.
  if (block_size > largest_free_block_[subregion_index]) {
    largest_free_block_[subregion_index] = block_size;
  }
  subregion->free_blocks.insert(std::pair<size_t, MemoryBlock*>(block_size, block));
}


void* CuMemoryAllocator::MallocFromSubregion(SubRegion *subregion,
                                             size_t size) {
  // NULL is implementation defined and doesn't have to be zero so we can't
  // guarantee that NULL will be <= a valid pointer-- so we cast to a pointer
  // from zero instead of using NULL.
  std::pair<size_t, MemoryBlock*> p(size, (MemoryBlock*)0);

  std::set<std::pair<size_t, MemoryBlock*> >::iterator iter =
      subregion->free_blocks.lower_bound(p);
  // so now 'iter' is the first member of free_blocks whose size_t value is >=
  // size.  If 'iter' was equal to the end() of that multi_map, it would be a
  // bug because the calling code checked that the largest free block in this
  // region was sufficiently large.  We don't check this; if it segfaults, we'll
  // debug.

  // search for a block that we don't have to synchronize on
  int max_iters = 20;
  auto search_iter = iter;
  for (int32 i = 0;
       search_iter != subregion->free_blocks.end() && i < max_iters;
       ++i, ++search_iter) {
    if (search_iter->second->thread_id == std::this_thread::get_id() ||
        search_iter->second->t <= synchronize_gpu_t_) {
      iter = search_iter;
      break;
    }
  }

  MemoryBlock *block = iter->second;
  // Erase 'block' from its subregion's free blocks list... the next lines are
  // similar to RemoveFromFreeBlocks(), but we code it directly as we have the
  // iterator here, and it would be wasteful to do another lookup.
  subregion->free_blocks.erase(iter);
  // Update largest_free_block_, if needed.  The following few lines of code also appear
  // in RemoveFromFreeBlocks().
  size_t block_size = block->end - block->begin,
      subregion_index = subregion->subregion_index;
  if (block_size == largest_free_block_[subregion_index]) {
    if (subregion->free_blocks.empty())
      largest_free_block_[subregion_index] = 0;
    else
      largest_free_block_[subregion_index] =
          subregion->free_blocks.rbegin()->first;
  }

  KALDI_PARANOID_ASSERT(block_size >= size && block->allocated == false);

  // the most memory we allow to be 'wasted' by failing to split a block, is the
  // smaller of: 1/16 of the size we're allocating, or half a megabyte.
  size_t allowed_extra_size = std::min<size_t>(size >> 4, 524288);
  if (block_size > size + allowed_extra_size) {
    // If the requested block is substantially larger than what was requested,
    // split it so we don't waste memory.
    block = SplitBlock(block, size);
  }

  if (std::this_thread::get_id() != block->thread_id &&
      block->t > synchronize_gpu_t_) {
    // see NOTE ON SYNCHRONIZATION in the header.
    SynchronizeGpu();
    synchronize_gpu_t_ = t_;
    num_synchronizations_++;
  }
  block->allocated = true;
  block->t = t_;
  allocated_block_map_[block->begin] = block;
  allocated_memory_ += (block->end - block->begin);
  if (allocated_memory_ > max_allocated_memory_) 
    max_allocated_memory_ = allocated_memory_;
  return block->begin;
}

// By the time MallocInternal is called, we will have ensured that 'size' is
// a nonzero multiple of 256 (for memory aligment reasons).
// inline
void* CuMemoryAllocator::MallocInternal(size_t size) {
start:
  std::vector<size_t>::const_iterator iter = largest_free_block_.begin(),
      end = largest_free_block_.end();
  size_t subregion_index = 0;
  for (; iter != end; ++iter, ++subregion_index) {
    if (*iter > size) {
      return MallocFromSubregion(subregions_[subregion_index], size);
    }
  }
  // We dropped off the loop without finding a subregion with enough memory
  // to satisfy the request -> allocate a new region.
  AllocateNewRegion(size);
  // An infinite loop shouldn't be possible because after calling
  // AllocateNewRegion(size), there should always be a SubRegion
  // with that size available.
  goto start;
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

std::string GetFreeGpuMemory(int64* free, int64* total) {
#ifdef _MSC_VER
  size_t mem_free, mem_total;
  cuMemGetInfo_v2(&mem_free, &mem_total);
#else
  // define the function signature type
  size_t mem_free, mem_total;
  {
    // we will load cuMemGetInfo_v2 dynamically from libcuda.so
    // pre-fill ``safe'' values that will not cause problems
    mem_free = 1; mem_total = 1;
    // open libcuda.so
    void* libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (NULL == libcuda) {
      KALDI_WARN << "cannot open libcuda.so";
    } else {
      // define the function signature type
      // and get the symbol
      typedef CUresult (*cu_fun_ptr)(size_t*, size_t*);
      cu_fun_ptr dl_cuMemGetInfo = (cu_fun_ptr)dlsym(libcuda,"cuMemGetInfo_v2");
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

void CuMemoryAllocator::PrintMemoryUsage() const {
  if (!opts_.cache_memory) {
    KALDI_LOG << "Not caching allocations; time taken in "
              << "malloc/free is " << malloc_time_taken_
              << "/" << (tot_time_taken_ - malloc_time_taken_)
              << ", num operations is " << t_
              << "; device memory info: "
              << GetFreeGpuMemory(NULL, NULL);
    return;
  }

  size_t num_blocks_allocated = 0, num_blocks_free = 0,
      memory_allocated = 0, memory_held = 0,
      largest_free_block = 0, largest_allocated_block = 0;

  for (size_t i = 0; i < memory_regions_.size(); i++) {
    MemoryBlock *m = memory_regions_[i].block_begin;
    KALDI_ASSERT(m->begin == memory_regions_[i].begin);
    for (; m != NULL; m = m->next) {
      size_t size = m->end - m->begin;
      if (m->allocated) {
        num_blocks_allocated++;
        memory_allocated += size;
        if (size > largest_allocated_block)
          largest_allocated_block = size;
      } else {
        num_blocks_free++;
        if (size > largest_free_block)
          largest_free_block = size;
      }
      memory_held += size;
      // The following is just some sanity checks; this code is rarely called so
      // it's a reasonable place to put them.
      if (m->next) {
        KALDI_ASSERT(m->next->prev == m && m->end == m->next->begin);
      } else {
        KALDI_ASSERT(m->end == memory_regions_[m->subregion->memory_region].end);
      }
    }
  }
  KALDI_LOG << "Memory usage: " << memory_allocated << "/"
            << memory_held << " bytes currently allocated/total-held; "
            << num_blocks_allocated << "/" << num_blocks_free
            << " blocks currently allocated/free; largest "
            << "free/allocated block sizes are "
            << largest_allocated_block << "/" << largest_free_block
            << "; time taken total/cudaMalloc is "
            << tot_time_taken_ << "/" << malloc_time_taken_
            << ", synchronized the GPU " << num_synchronizations_
            << " times out of " << (t_/2) << " frees; "
            << "device memory info: " << GetFreeGpuMemory(NULL, NULL)
            << "maximum allocated: " << max_allocated_memory_  
            << "current allocated: " << allocated_memory_; 
}

// Note: we just initialize with the default options, but we can change it later
// (as long as it's before we first use the class) by calling SetOptions().
CuMemoryAllocator::CuMemoryAllocator():
    opts_(CuAllocatorOptions()),
    t_(0),
    synchronize_gpu_t_(0),
    num_synchronizations_(0),
    tot_time_taken_(0.0),
    malloc_time_taken_(0.0),
    max_allocated_memory_(0),
    allocated_memory_(0) {
  // Note: we don't allocate any memory regions at the start; we wait for the user
  // to call Malloc() or MallocPitch(), and then allocate one when needed.
}


void* CuMemoryAllocator::MallocPitch(size_t row_bytes,
                                     size_t num_rows,
                                     size_t *pitch) {
  Timer tim;
  if (!opts_.cache_memory) {
    void *ans;
    CU_SAFE_CALL(cudaMallocPitch(&ans, pitch, row_bytes, num_rows));
    double elapsed = tim.Elapsed();
    tot_time_taken_ += elapsed;
    malloc_time_taken_ += elapsed;
    return ans;
  }

  // Round up row_bytes to a multiple of 256.
  row_bytes = (row_bytes + 255) & ~((size_t)255);
  *pitch = row_bytes;
  void *ans = MallocInternal(row_bytes * num_rows);
  tot_time_taken_ += tim.Elapsed();
  return ans;
}

void CuMemoryAllocator::Free(void *ptr) {
  Timer tim;
  if (!opts_.cache_memory) {
    CU_SAFE_CALL(cudaFree(ptr));
    tot_time_taken_ += tim.Elapsed();
    t_++;
    return;
  }
  t_++;
  unordered_map<void*, MemoryBlock*>::iterator iter =
      allocated_block_map_.find(ptr);
  if (iter == allocated_block_map_.end()) {
    KALDI_ERR << "Attempt to free CUDA memory pointer that was not allocated: "
              << ptr;
  }
  MemoryBlock *block = iter->second;
  allocated_memory_ -= (block->end - block->begin);
  allocated_block_map_.erase(iter);
  block->t = t_;
  block->thread_id = std::this_thread::get_id();
  block->allocated = false;

  // If this is not the first block of the memory region and the previous block
  // is not allocated, merge this block into the previous block.
  MemoryBlock *prev_block = block->prev;
  if (prev_block != NULL && !prev_block->allocated) {
    RemoveFromFreeBlocks(prev_block);
    prev_block->end = block->end;
    if (prev_block->thread_id != block->thread_id) {
      // the two blocks we're merging were freed by different threads, so we
      // give the 'nonexistent thread' as their thread, which means that
      // whichever thread requests that block, we force synchronization.  We can
      // assume that prev_block was previously allocated (prev_block->t > 0)
      // because we always start from the left when allocating blocks, and we
      // know that this block was previously allocated.
      prev_block->thread_id = std::thread::id();
    }
    prev_block->t = t_;
    prev_block->next = block->next;
    if (block->next)
      block->next->prev = prev_block;
    delete block;
    block = prev_block;
  }

  // If this is not the last block of the memory region and the next block is
  // not allocated, merge the next block into this block.
  MemoryBlock *next_block = block->next;
  if (next_block != NULL && !next_block->allocated) {
    // merge next_block into 'block', deleting 'next_block'.  Note: at this
    // point, if we merged with the previous block, the variable 'block' may now
    // be pointing to that previous block, so it would be a 3-way merge.
    RemoveFromFreeBlocks(next_block);
    block->end = next_block->end;
    if (next_block->thread_id != block->thread_id && next_block->t > 0) {
      // the two blocks we're merging were freed by different threads, so we
      // give the 'nonexistent thread' as their thread, which means that
      // whichever thread requests that block, we force synchronization.  there
      // is no need to do this if next_block->t == 0, which would mean it had
      // never been allocated.
      block->thread_id = std::thread::id();
    }
    // We don't need to inspect the 't' value of next_block; it can't be
    // larger than t_ because t_ is now.
    block->next = next_block->next;
    if (block->next)
      block->next->prev = block;
    delete next_block;
  }
  AddToFreeBlocks(block);
  tot_time_taken_ += tim.Elapsed();
}

void CuMemoryAllocator::AllocateNewRegion(size_t size) {
  int64 free_memory, total_memory;
  std::string mem_info = GetFreeGpuMemory(&free_memory, &total_memory);
  opts_.Check();
  size_t region_size = static_cast<size_t>(free_memory * opts_.memory_proportion);
  if (region_size < size)
    region_size = size;
  // Round up region_size to an exact multiple of 1M (note: we expect it will
  // be much larger than that).  1048575 is 2^20 - 1.
  region_size = (region_size + 1048575) & ~((size_t)1048575);

  if (!memory_regions_.empty()) {
    // If this is not the first region allocated, print some information.
    KALDI_LOG << "About to allocate new memory region of " << region_size
              << " bytes; current memory info is: " << mem_info;
  }
  void *memory_region;
  cudaError_t e;
  {
    Timer tim;
    e = cudaMalloc(&memory_region, region_size);
    malloc_time_taken_ += tim.Elapsed();
  }
  if (e != cudaSuccess) {
    PrintMemoryUsage();
    if (!CuDevice::Instantiate().IsComputeExclusive()) {
      KALDI_ERR << "Failed to allocate a memory region of " << region_size
                << " bytes.  Possibly this is due to sharing the GPU.  Try "
                << "switching the GPUs to exclusive mode (nvidia-smi -c 3) and using "
                << "the option --use-gpu=wait to scripts like "
                << "steps/nnet3/chain/train.py.  Memory info: "
                << mem_info
                << " CUDA error: '" << cudaGetErrorString(e) << "'";
    } else {
      KALDI_ERR << "Failed to allocate a memory region of " << region_size
                << " bytes.  Possibly smaller minibatch size would help.  "
                << "Memory info: " << mem_info
                << " CUDA error: '" << cudaGetErrorString(e) << "'";
    }
  }
  // this_num_subregions would be approximately 'opts_.num_subregions' if
  // 'region_size' was all the device's memory.  (We add one to round up).
  // We're aiming to get a number of sub-regions approximately equal to
  // opts_.num_subregions by the time we allocate all the device's memory.
  size_t this_num_subregions = 1 +
      (region_size * opts_.num_subregions) / total_memory;

  size_t memory_region_index = memory_regions_.size();
  memory_regions_.resize(memory_region_index + 1);
  MemoryRegion &this_region = memory_regions_.back();

  this_region.begin = static_cast<char*>(memory_region);
  this_region.end = this_region.begin + region_size;
  // subregion_size will be hundreds of megabytes.
  size_t subregion_size = region_size / this_num_subregions;

  std::vector<SubRegion*> new_subregions;
  char* subregion_begin = static_cast<char*>(memory_region);
  for (size_t i = 0; i < this_num_subregions; i++) {
    SubRegion *subregion = new SubRegion();
    subregion->memory_region = memory_region_index;
    subregion->begin = subregion_begin;
    if (i + 1 == this_num_subregions) {
      subregion->end = this_region.end;
      KALDI_ASSERT(subregion->end > subregion->begin);
    } else {
      subregion->end = subregion_begin + subregion_size;
      subregion_begin = subregion->end;
    }
    subregion->next = NULL;
    if (i > 0) {
      new_subregions.back()->next = subregion;
    }
    new_subregions.push_back(subregion);
  }
  // Initially the memory is in a single block, owned by
  // the first subregion.  It will be split up gradually.
  MemoryBlock *block = new MemoryBlock();
  block->begin = this_region.begin;
  block->end = this_region.end;
  block->subregion = new_subregions.front();
  block->allocated = false;
  block->t = 0; // was never allocated.
  block->next = NULL;
  block->prev = NULL;
  for (size_t i = 0; i < this_num_subregions; i++)
    subregions_.push_back(new_subregions[i]);
  SortSubregions();
  this_region.block_begin = block;

  AddToFreeBlocks(block);
}

// We sort the sub-regions according to the distance between the start of the
// MemoryRegion of which they are a part, and the start of the SubRegion.  This
// will generally mean that the highest-numbered SubRegion-- the one we keep
// free at all costs-- will be the end of the first block which we allocated
// (which under most situations will be the largest block).
void CuMemoryAllocator::SortSubregions() {
  largest_free_block_.resize(subregions_.size());

  std::vector<std::pair<size_t, SubRegion*> > pairs;
  for (size_t i = 0; i < subregions_.size(); i++) {
    SubRegion *subregion = subregions_[i];
    MemoryRegion &memory_region = memory_regions_[subregion->memory_region];
    size_t distance = subregion->begin - memory_region.begin;
    pairs.push_back(std::pair<size_t, SubRegion*>(distance, subregion));
  }
  std::sort(pairs.begin(), pairs.end());
  for (size_t i = 0; i < subregions_.size(); i++) {
    subregions_[i] = pairs[i].second;
    subregions_[i]->subregion_index = i;
    if (subregions_[i]->free_blocks.empty())
      largest_free_block_[i] = 0;
    else
      largest_free_block_[i] = subregions_[i]->free_blocks.rbegin()->first;
  }
}

CuMemoryAllocator::~CuMemoryAllocator() {
  // We mainly free these blocks of memory so that cuda-memcheck doesn't report
  // spurious errors.
  for (size_t i = 0; i < memory_regions_.size(); i++) {
    // No need to check the return status here-- the program is exiting anyway.
    cudaFree(memory_regions_[i].begin);
  }
  for (size_t i = 0; i < subregions_.size(); i++) {
    SubRegion *subregion = subregions_[i];
    for (auto iter = subregion->free_blocks.begin();
         iter != subregion->free_blocks.end(); ++iter)
      delete iter->second;
    delete subregion;
  }
}


CuMemoryAllocator g_cuda_allocator;


}  // namespace kaldi


#endif // HAVE_CUDA


namespace kaldi {

// Define/initialize this global variable.  It was declared in cu-allocator.h.
// This has to be done outside of the ifdef, because we register the options
// whether or not CUDA is compiled in (so that the binaries accept the same
// options).
CuAllocatorOptions g_allocator_options;

}
