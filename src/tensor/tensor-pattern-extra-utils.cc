// tensor/tensor-pattern-extra-utils.cc

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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


#include "tensor/tensor-pattern-extra-utils.h"

namespace kaldi {
namespace tensor {


class IntersectionComputer {
 public:
  IntersectionComputer(const TensorPattern &pattern1,
                       const TensorPattern &pattern2):
      pattern1_(pattern1), pattern2_(pattern2) {
    CanonicalizePattern(&pattern1_);
    CanonicalizePattern(&pattern2_);
  }

 private:

  // Attempts to find a common list of strides which can be used for the
  // combined patterns.  Returns false if this cannot be done.  This is done by
  // taking the union of {1} and the strides in pattern1_ and pattern2_, sorting
  // them, and then checking that each stride in the sequence divides the next
  // (it returns true if this is the case, false otherwise).
  bool FindCommonStrides(std::vector<int32> *axes);


  // This function converts a pattern in canonical form to a list of Patterns
  // that are equivalent (after taking their union, viewed as memory-index-sets)
  // to the original Pattern, where the strides of the output patterns are equal
  // to the provided 'common_strides' vector.  This function requires that the
  // actual strides in 'pattern' all be present in the list 'common_strides',
  // that the elements of 'common_strides' be positive and that each element in
  // 'common_strides' divide the next element exactly.
  // The codes of 'patterns' are not set.
  static void ConvertToCommonStrides(const TensorPattern &pattern,
                                     const std::vector<int32> &common_strides,
                                     std::vector<TensorPattern*> patterns);

  /**
     Computes the intersection between pattern1 and pattern2, which must
     have identical axes and strides, and must be valid *except* that
     it's not required that axes with dim=1 must have stride=0, and the
     code does not have to be set.

     If the intersection was nonempty, it returns true and outputs it to
     'pattern_out'; otherwise it returns false.  If this function
     returns true, 'pattern_out' will be valid and will have trivial
     axes removed (but will not have its code set).
  */
  static bool ComputeIntersection(const TensorPattern &pattern1,
                                  const TensorPattern &pattern2,
                                  TensorPattern *pattern_out);

  // This function converts a pattern in canonical form to a possibly-invalid
  // Pattern whose strides are equal to the provided 'common_strides' vector.
  // It is required that the actual strides in 'pattern' all be present
  // in the list 'common_strides', that the elements of 'common_strides' be
  // positive and that each element in 'common_strides' divide the next element
  // exactly.
  //
  // By 'possibly-invalid' what we mean is that the output pattern might
  // not satisfy the property that, for a TensorPattern with axes sorted
  // by increasing stride,
  //  for 0 <= raxis < num_axes - 1,
  //   stride[raxis+1] >= stride[raxis] * dim[raxis].
  //
  // That is because we ensure that pattern_out has the strides from
  // 'common_strides' by simply inserting trivial dims with any missing strides,
  // while keeping the strides in increasing order.
  // The code of pattern_out is not set.
  static void ConvertLazilyToCommonStrides(const TensorPattern &pattern_in,
                                           const std::vector<int32> &common_strides,
                                           TensorPattern* pattern_out);

  /**
     This function makes sure that the pattern 'pattern' has the property that

       `pattern->strides[raxis+1] >= pattern->strides[raxis] * pattern->dims[raxis]`

     If it does not already have this property, this function ensures that it
     does have it by modifying its dims for raxis and raxis+1, and if necessary,
     'forking' the pattern to 'extra_pattern'.  This will be necessary if the
     value of `pattern->dims[raxis]` at entry is not a multiple of
     `pattern->strides[raxis+1] / pattern->strides[raxis]`.

         @param [in]      raxis    The axis on which we are doing the check
         @param [in,out]  pattern  The input pattern (possibly invalid, as
                                explained in the docs for ConvertLazilyToCommonStrides()).
                                Its strides must be in increasing order and each must
                                divide the next.
         @param [out]     extra_pattern   This function writes to 'extra_pattern' if
                                and only if it returns true.  See below.
         @return  Returns true if it wrote to extra_pattern.  If it returns true,
                  then it guarantees that the union of the memory-index-sets of
                  'pattern' and 'extra_pattern' at exit are equal to the memory-index-set
                  of 'pattern' at entry.  If it returns false, then it guarantees
                  that the memory-index-set of 'pattern' has been unchanged.
                  In either case it guarantees that the property mentioned above
                  holds for pattern and, if it returns true, to 'extra_pattern'
                  as well.
                  The codes of pattern and extra_patter are not set.
  */
  static bool EnsurePropertyHolds(int32 raxis,
                                  TensorPattern *pattern,
                                  TensorPattern *extra_pattern);



  // the same as pattern1 passed to the constructor, but reduced to
  // canonical form
  TensorPattern pattern1_;
  // the same as pattern1 passed to the constructor, but reduced to
  // canonical form
  TensorPattern pattern2_;

  // patterns1_ is the list of patterns we get when we convert pattern1_
  // to have the shared list of strides.  Will have at least one element.
  std::vector<TensorPattern> patterns1_;
  // patterns2_ is the list of patterns we get when we convert pattern2_
  // to have the shared list of strides.  Will have at least one element.
  std::vector<TensorPattern> patterns2_;

  std::vector<TensorPattern> *intersection_;
};



bool IntersectionComputer::EnsurePropertyHolds(
    int32 raxis, TensorPattern *pattern,
    TensorPattern *extra_pattern) {
  KALDI_PARANOID_ASSERT(raxis + 1 < pattern->num_axes);
  if (pattern->strides[raxis + 1] >=
      pattern->strides[raxis] * pattern->dims[raxis]) {
    // Property already holds -> nothing to do.  Return false
    // because 'extra_pattern' is not needed.
    return false;
  }

  // It would not make sense if pattern->dims[raxis + 1] were > 1; that would
  // imply we started with some kind of self-overlapping pattern, whicg would
  // not be valid.
  KALDI_PARANOID_ASSERT(pattern->strides[raxis + 1] %
                        pattern->strides[raxis] == 0 &&
                        pattern->dims[raxis + 1] == 1);

  int32 ratio = pattern->strides[raxis + 1] / pattern->strides[raxis];
  int32 orig_dim = pattern->dims[raxis];
  pattern->dims[raxis] = ratio;
  int32 next_dim = orig_dim / ratio;
  pattern->dims[raxis + 1] = orig_dim;

  int32 remainder = orig_dim % ratio;
  if (remainder == 0) {
    // We didn't need to make use of 'extra_pattern', so return false.
    return false;
  } else {
    *extra_pattern = pattern;
    extra_pattern->dims[raxis] = remainder;
    extra_pattern->dims[raxis + 1] = 1;
    extra_pattern->offset += next_dim * pattern->strides[raxis];
    // we used extra_pattern, so return true.
    return true;
  }
}


void IntersectionComputer::ConvertLazilyToCommonStrides(
    const TensorPattern &pattern_in,
    const std::vector<int32> &common_strides,
    TensorPattern* pattern_out) {
  int32 num_axes_in = pattern_in.num_axes,
      num_axes_out = common_strides.size();
  pattern_out->num_axes = num_axes_out;
  int32 raxis_in = 0;
  pattern_out->offset = pattern_in->offset;
  for (int32 raxis_out = 0; raxis_out < num_axes_out; raxis_out++) {
    int32 stride = common_strides[raxis_out];
    pattern_out->strides[raxis_out] = stride;
    if (pattern_in.strides[raxis_in] == stride) {
      pattern_out->dims[raxis_out] = pattern_in.dims[raxis_in];
      pattern_in++;
    } else {
      pattern_out->dims[raxis_out] = 1;
    }
  }
  if (raxis_in != num_axes_in) {
    KALDI_ERR << "Something went wrong converting strides (likely code error)";
  }
}


void IntersectionComputer::ConvertToCommonStrides(
    const TensorPattern &pattern,
    const std::vector<int32> &common_strides,
    std::vector<TensorPattern*> *patterns) {

  patterns->resize(1);
  ConvertLazilyToCommonStrides(pattern, &((*patterns)[0]));
  int32 num_axes = common_strides.size();
  for (int32 raxis = 0; raxis + 1 < num_axes; raxis++) {
    TensorPattern extra_pattern;
    int32 num_patterns = patterns->size();
    for (int32 p = 0; p < num_patterns; p++) {
      if (EnsurePropertyHolds(raxis, &((*patterns)[p]), &extra_pattern))
        patterns->push_back(extra_pattern);
    }
  }
}


// intersection between patterns with identical strides.
bool IntersectionComputer::ComputeIntersection(
    const TensorPattern &pattern1,
    const TensorPattern &pattern2,
    TensorPattern *pattern_out) {
  // First ensure that pattern1.offset <= pattern2.offset.
  if (pattern1.offset > pattern2.offset)
    return ComputeIntersection(pattern2, pattern1, pattern_out);

  int64 extra_offset = pattern2.offset - pattern1.offset;
  int32 dim_offset[KALDI_MAX_TENSOR_DIM];
  // What we are doing conceptually here is shifting pattern1 to have the same
  // offset as pattern2 by saying that on each axis, instead of starting the
  // index from zero to dim - 1, we start that index from some number less than
  // zero i.e. we shift those indexes to the left.  The index of the
  // intersection will still start from zero though, because pattern2's index
  // still starts from zero.
  // We are going to express 'extra_offset' as a sum


  // pattern1 and pattern2 are required to have the same stride and num_axes.
  int32 num_axes = pattern1.num_axes;
  for (int32 raxis = num_axes - 1; raxis >= 0; raxis--) {
    int32 this_stride = pattern1.strides[raxis],
        this_offset = extra_offset / this_stride;

  }
}

bool TensorPatternRebaser::Convert(TensorPattern *pattern) {
  if (!needs_conversion_)
    return;  // An optimization to make the common case fast.

  pattern->offset = ConvertMemoryIndex(pattern->offset);

  if (num-axes_ == 0)
    return;  // Another optimization to make a fairly common case fast.
  int32 num_axes = pattern->num_axes;
  for (int32 raxis = 0; raxis < num_axes; raxis++) {
    int32 stride = pattern->strides[raxis],
        dim = pattern->dims[raxis];
    if (stride == 0)
      continue;
    int32 pstride = std::abs(stride),
        product = pstride * dim;
    // We will convert 'pstride' using


  }
  return true;  // Success

}


int64 TensorPatternRebaser::ConvertMemoryIndex(int64 m) {
  int32 num_axes = num_axes_;
  int64 ans = dest_offset_;
  m -= src_offset_;
  if (num_axes == 0)
    return m;
  // We visit the compressed axes in order from greatest to least src_stride.
  // What this loop does is to reverse engineer the indexes into (the compressed
  // version of) src_pattern that we'd need to get memory-offset m.  The 'i'
  // values in the loop are those indexes.
  for (int32 raxis = num_axes - 1; raxis >= 0; raxis--) {
    int32 stride = src_strides_[raxis];
    int64 i = m / stride;
    m -= i * stride;
    ans += i * dest_strides_[raxis]
  }
  if (m != 0) {
    // This should not happen; likely it means the memory-index m was not covered
    // by the src_pattern passed to the constructor, so someone was trying
    // to rebase a pattern which was not covered by src_pattern.
    KALDI_ERR << "Could not convert this memory-index (likely code error)";
  }
  return ans;
}




}  // namespace kaldi
}  // namespace tensor
