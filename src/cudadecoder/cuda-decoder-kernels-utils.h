// cudadecoder/cuda-decoder-kernels-utils.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CUDA_DECODER_CUDA_DECODER_KERNELS_UTILS_H_
#define KALDI_CUDA_DECODER_CUDA_DECODER_KERNELS_UTILS_H_

// NO_KEY == -1 is ok, because all keys will be >= 0 (FST states)
#define KALDI_CUDA_DECODER_HASHMAP_NO_KEY -1
#define KALDI_CUDA_DECODER_HASHMAP_NO_VAL                 \
  {                                                       \
    KALDI_CUDA_DECODER_HASHMAP_NO_KEY, 0, ULLONG_MAX \
  }

#include "util/stl-utils.h"

namespace kaldi {
namespace cuda_decoder {

// MinPlus and PlusPlus
// int2 operators used in Scan or Reduce operations
struct MinPlus {
  __device__ int2 operator()(const int2 &a, const int2 &b) const {
    int2 c;
    c.x = min(a.x, b.x);
    c.y = a.y + b.y;
    return c;
  }
};
struct PlusPlus {
  __device__ int2 operator()(const int2 &a, const int2 &b) const {
    int2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
  }
};

struct PlusPlusPlusPlus {
  __device__ int4 operator()(const int4 &a, const int4 &b) const {
    int4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    return c;
  }
};

// 1:1 Conversion float <---> sortable int
// We convert floats to sortable ints in order
// to use native atomics operation, which are
// way faster than looping over atomicCAS
__device__ __forceinline__ int32 floatToOrderedInt(float floatVal) {
  int32 intVal = __float_as_int(floatVal);
  return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat(int32 intVal) {
  return __int_as_float((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

// binsearch_maxle (device)
// With L=[all indexes low<=i<=high such as vec[i]<= val]
// binsearch_maxle returns max(L)
// the array vec must be sorted
// Finds that value using a binary search
__device__ __forceinline__ int32 binsearch_maxle(const int32 *vec,
                                                 const int32 val, int32 low,
                                                 int32 high) {
  while (true) {
    if (low == high) return low;  // we know it exists
    if ((low + 1) == high) return (vec[high] <= val) ? high : low;

    int32 mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}

// Atomic operations on int2 (device)
// atomicAddI2, atomicMinI2, atomicSubI2
//
// union used
union UInt64UnionInt2 {
  int2 i2;
  unsigned long long int ull;
};

#if __CUDA_ARCH__ < 350
__device__ __inline__ void atomicMinULL(unsigned long long *ptr,
                                        unsigned long long val) {
  unsigned long long old = *ptr, assumed;
  do {
    assumed = old;
    old = atomicCAS(ptr, assumed, val);
  } while (old > val && assumed != old);
}
#else
__device__ __forceinline__ void atomicMinULL(unsigned long long *ptr,
                                             unsigned long long val) {
  atomicMin(ptr, val);
}
#endif

__device__ __inline__ int2 atomicAddI2(int2 *ptr, int2 val) {
  unsigned long long int *ptr64 =
      reinterpret_cast<unsigned long long int *>(ptr);
  UInt64UnionInt2 uval, uold;
  uval.i2 = val;
  uold.ull = atomicAdd(ptr64, uval.ull);
  return uold.i2;
}

// We should switch to native atom64 on atomicMinI2 and atomicSubI2
__device__ __inline__ void atomicMinI2(int2 *ptr, int2 val) {
  unsigned long long int *ptr64 =
      reinterpret_cast<unsigned long long int *>(ptr);
  UInt64UnionInt2 old, assumed, value;
  old.ull = *ptr64;
  value.i2 = val;
  if (old.i2.x <= val.x) return;
  do {
    assumed = old;
    old.ull = atomicCAS(ptr64, assumed.ull, value.ull);
  } while (old.ull != assumed.ull && old.i2.x > value.i2.x);
}

__device__ void atomicSubI2(int2 *ptr, int2 sub) {
  unsigned long long int *ptr64 =
      reinterpret_cast<unsigned long long int *>(ptr);
  UInt64UnionInt2 old, assumed, value;
  old.ull = *ptr64;
  do {
    assumed = old;
    value.i2.x = assumed.i2.x - sub.x;
    value.i2.y = assumed.i2.y - sub.y;
    old.ull = atomicCAS(ptr64, assumed.ull, value.ull);
  } while (old.ull != assumed.ull);
}

// Hash function used in the hashmap.
// Using identity for now. They keys are the FST states, some randomness already
// exists
__device__ __forceinline__ int hash_func(int key) {
  return key;  // using identity for now
}

// Packing and unpacking a minimum + its argument into a single uint64
// (min is first, used for sorting)
// Not using an union because documentation is not clear regarding reordering in structs
// (for instance, in int2, y is stored before x)

__device__ __inline__ void PackArgminInUInt64(const uint32_t min, const uint32_t arg, unsigned long long *argmin) {
	unsigned long long p = min;
	p <<= 32;
	p |= arg;
	*argmin = p;
}

__device__ __inline__ void GetMinFromPackedArgminUInt64(const unsigned long long argmin, uint32_t *min) {
	*min = (uint32_t)((argmin & 0xFFFFFFFF00000000LL) >> 32);
}

__device__ __inline__ void GetArgFromPackedArgminUInt64(const unsigned long long argmin, uint32_t *arg) {
	*arg = (uint32_t)(argmin & 0xFFFFFFFFLL);
}

// hashmap_insert_or_aggregate
// Inserting a new value into the hashmap. If the key already exists in the
// hashmap,
// we'll aggregate the existing value with the new one, and set the result as
// value for that key.
// The new value inserted at key is (1, (int_cost, arg_int_cost)
// With values being [count (int32), [min_cost, argmin_cost] (int2)]
// If a value already exists for a key, we will aggregate the two values:
// hashmap[key] = old_value +_ new_value
// with +_ being (integer +, argmin)
// It returns the hash_idx, i.e. where the key was inserted in the hashmap
// The owner will then use that to access the data, and clear it for future use
// It also returns local_idx, which informs how many values of that same key
// were inserted before that call.
// e.g. if thread 23 inserts the key 3, then thread 9 inserts the key 3,
// thread 23 will have local_idx=0, thread 9 will have local_idx=1
//
// We use hashmap_insert in the context of a ReduceByKey. The same thread will
// always
// access the same key. Which is why we do not need a hashmap_find, and can
// simply remember the hash_idx
// from our last insert.
//
// Restriction: that function can only be used if we know that we will have
// enough space in the hashmap
// ie hashmap_capacity > total number of keys
//
// keys must be >= 0 (to avoid collisions with
// KALDI_CUDA_DECODER_HASHMAP_NO_KEY)
__device__ __inline__ void hashmap_insert_or_aggregate(
    HashmapValueT *d_map_values, int key, int int_cost, int arg_int_cost,
    int capacity, int *local_idx, int *out_hash_idx) {
  int hash_idx = hash_func(key) % capacity;
  int c = 0;
  HashmapValueT *d_val = NULL;
  do {
    d_val = &d_map_values[hash_idx];
    // Looking for a spot in the hashmap
    int old = atomicCAS(&d_val->key, KALDI_CUDA_DECODER_HASHMAP_NO_KEY, key);
    if (old == KALDI_CUDA_DECODER_HASHMAP_NO_KEY || old == key)
      break;  // found a spot
    hash_idx = (hash_idx + 1) % capacity;
    ++c;
  } while (c < capacity);
  // The condition in which we use the hashmap always ensure that we have space
  // asserting that we found a spot
  assert(d_val);

  // Updating values
  *local_idx = atomicAdd(&d_val->count, 1);
  *out_hash_idx = hash_idx;
  unsigned long long argmin_u64;
  PackArgminInUInt64(int_cost, arg_int_cost, &argmin_u64);
  atomicMinULL(&d_val->min_and_argmin_int_cost_u64, argmin_u64);
}

// In FSTStateHashIndex, we store both the hash_idx and a boolean
// is_representative
// which tells if the current thread is responsible for the state stored at
// index hash_idx
// We use the bit sign for that
// Setter and getter
__device__ __inline__ void SetFSTStateHashIndex(int32 raw_hash_idx,
                                                bool is_representative,
                                                FSTStateHashIndex *hash_idx) {
  *hash_idx = is_representative ? (-raw_hash_idx - 1)  // -1 to force it < 0
                                : raw_hash_idx;
}

__device__ __inline__ void GetFSTStateHashIndex(FSTStateHashIndex &hash_idx,
                                                int32 *raw_hash_idx,
                                                bool *is_representative) {
  *is_representative = (hash_idx < 0);
  *raw_hash_idx = *is_representative ? (-(hash_idx + 1)) : hash_idx;
}

}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // KALDI_CUDA_DECODER_CUDA_DECODER_KERNELS_UTILS_H_
