// cudamatrix/cu-array-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-array.h"

using namespace kaldi;


namespace kaldi {


template<class T>
void AssertEqual(const std::vector<T> &vec1,
                 const std::vector<T> &vec2) {
  // use this instead of "vec1 == vec2" because
  // for doubles and floats, this can fail if we
  // have NaNs, even if identical.
  KALDI_ASSERT(vec1.size() == vec2.size());
  if (vec1.size() > 0) {
    const char *p1 = reinterpret_cast<const char*>(&(vec1[0]));
    const char *p2 = reinterpret_cast<const char*>(&(vec2[0]));
    size_t size = sizeof(T) * vec1.size();
    for (size_t i = 0; i < size; i++)
      KALDI_ASSERT(p1[i] == p2[i]);
  }
}


template<class T>
static void UnitTestCuArray() {
  for (int32 i = 0; i < 30; i++) {
    int32 size = rand() % 5;
    size = size * size * size; // Have a good distribution of sizes, including >256.
    int32 size2 = rand() % 4;
    std::vector<T> vec(size);
    std::vector<T> garbage_vec(size2); // We just use garbage_vec to make sure
                                       // we sometimes resize from empty,
                                       // sometimes not.
    
    int32 byte_size = size * sizeof(T);
    std::vector<char> rand_c(byte_size);
    for (size_t i = 0; i < byte_size; i++)
      rand_c[i] = rand() % 256;
    if (!vec.empty()) {
      std::memcpy((void*)&(vec[0]), (void*)&(rand_c[0]),
                  byte_size);
    }

    { // test constructor from vector and CopyToVec.
      CuArray<T> cu_vec(vec);
      std::vector<T> vec2;
      cu_vec.CopyToVec(&vec2);
    }

    { // test assignment operator from CuArray.
      CuArray<T> cu_vec(vec);
      CuArray<T> cu_vec2(garbage_vec);
      cu_vec2 = cu_vec;
      std::vector<T> vec2;
      cu_vec2.CopyToVec(&vec2);
      AssertEqual(vec, vec2);
      KALDI_ASSERT(cu_vec2.Dim() == int32(vec2.size())); // test Dim()
    }
      
    { // test resize with resize_type = kSetZero.
      CuArray<T> cu_vec(vec);
      cu_vec.Resize(size, kSetZero);
      std::vector<T> vec2(vec);

      if (!vec2.empty())
        std::memset(&(vec2[0]), 0, vec2.size() * sizeof(T));
      std::vector<T> vec3;
      cu_vec.CopyToVec(&vec3);
      AssertEqual(vec2, vec3); // testing equality of zero arrays.
    }

    if (sizeof(T) == sizeof(int32) && size > 0) { // test Set for type int32, or same size.
      CuArray<T> cu_vec(vec);
      cu_vec.Set(vec[0]);
      for (size_t i = 1; i < vec.size(); i++) vec[i] = vec[0];
      std::vector<T> vec2;
      cu_vec.CopyToVec(&vec2);
      AssertEqual(vec, vec2);
    }
  }
}


} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    //kaldi::UnitTestCuArray<float>();
    kaldi::UnitTestCuArray<double>();
    kaldi::UnitTestCuArray<int32>();
    kaldi::UnitTestCuArray<std::pair<int32, int32> >();

    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.\n";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.\n";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}
