// tensor/array-ref.h

//  Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#include <base/kaldi-error.h>
#include <tensor/tensor-common.h>


/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {


// Similar to llvm/PyTorch's ArrayRef, this is a lightweight way to store an
// array (zero or more elements of type T).  The array is not owned here; it
// will generally be unsafe to use an ArrayRef as other than a local variable.
//
// ArrayRef has only two members and it will probably make sense to pass it by
// value most of the time.
template <typename T>
struct ArrayRef final {
  const T *data;
  size_t size;

  inline T& operator [] (uint64_t i) const {
    KALDI_ASSERT(i < size);
    return data[i];
  }

  constexpr ArrayRef() : size(0), data(nullptr) { }

  // Construct from one element.
  // Caution: this constructor allows you to evade 'const'.
  constexpr ArrayRef(const T &element) : size(1), data(&element) { }

  // Construct from data and size
  constexpr ArrayRef(const T* data, size_t size): data(data), size(size) { }

  /// Construct from a range.  Caution: this constructor allows
  /// you to evade 'const'.
  constexpr ArrayRef(const T* begin, const T* end): data(begin), size(end - begin) { }

  /// Construct from a std::vector.
  ArrayRef(const std::vector<T> &vec): data(vec.data()), size(vec.size()) { }

  /// Construct from a C array.
  template <size_t N>
      constexpr ArrayRef(const T (&data)[N]): data(data), size(N) { }

  /// Construct from a std::initializer_list
  constexpr ArrayRef(const std::initializer_list<T> &vec):
      data(vec.data()), size(vec.size()) { }

  // We will add iterators later if they are needed.
};


}  // namespace tensor
}  // namespace kaldi
