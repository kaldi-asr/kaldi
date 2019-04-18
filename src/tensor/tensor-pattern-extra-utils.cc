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
      pattern1_(pattern1), pattern2_(pattern2);

  /**
     Computes the intersection between the pattern1 and pattern2 given to the
     constructor (must be called only once); if it could be computed, the
     intersection is represented as the union between all the (disjoint)
     patterns in patterns_out.

        @param [in] patterns_out  A list of patterns (in arbitrary order)
                         is written to here.  The union of this list of patterns
                         will, if this function returns true,
                         represent the intersection between the pattern1 and
                         pattern2 passed to the constructor.  These patterns
                         will be valid without a code, but won't be
                         in canonical form (the user can do that themselves;
                         we don't it here because in most cases the caller will
                         only care whether the union is empty or not).
  */
  bool ComputeIntersection(std::vector<TensorPattern> *patterns_out) {
    CanonicalizePattern(&pattern1_);
    CanonicalizePattern(&pattern2_);
    std::vector<int32> axes;
    if (!FindCommonStrides(&axes))
      return false;
    std::vector<TensorPattern> patterns1, patterns2;
    patterns1.reserve(8);
    patterns2.reserve(8);
    ConvertToCommonStrides(pattern1_, &patterns1);
    ConvertToCommonStrides(pattern2_, &patterns2);
    patterns_out->clear();
    ComputeIntersection(patterns1, patterns2, patterns_out);
    return true;
  }

 private:

  // Attempts to find a common list of strides which can be used for the
  // combined patterns.  Returns false if this cannot be done.  This is done by
  // taking the union of the strides in pattern1_ and pattern2_, sorting them,
  // and then checking that each stride in the sequence divides the next (it
  // returns true if this is the case, false otherwise).
  // These strides must all be positive because pattern1_ and pattern2_ have
  // both been canonicalized.
  bool FindCommonStrides(std::vector<int32> *axes);


  /**
    This function converts a pattern 'pattern' in canonical form to a list of Patterns
    whose union (viewed as memory-index-sets) is equivalent to 'pattern'
    where the strides of the output patterns are equal to the provided 'common_strides'
    vector.

    This function requires that the actual strides in 'pattern' all be present in
    the list 'common_strides'; that the elements of 'common_strides' be positive
    and sorted from smallest to greatest; and that each element in
    'common_strides' divide the next element exactly.


       @param [in] pattern  Input pattern in canonical form, valid except for
                         code.
       @param [in] common_strides   A sorted list of integers >0, with the
                         property that each element must divide the next
                         element exactly, and also that each stride in
                         'pattern' must be present in 'common_strides'.
       @param [out] patterns   This will be set to a nonempty list of patterns
                         whose union (viewed as a memory-index-set) equals
                         'pattern', and whose strides are equal to
                         'common_strides'.  The patterns in `*patterns` at
                         output will be valid except for the code and for
                         property (iv) (search Valid Pattern in
                         tensor-pattern.h): that is, it will have nonzero
                         strides for axes with dim != 1.
  */
  static void ConvertToCommonStrides(const TensorPattern &pattern,
                                     const std::vector<int32> &common_strides,
                                     std::vector<TensorPattern> *patterns);

  /**
     Computes the intersection between pattern1 and pattern2, which must have
     identical axes and strides, and must be valid *except* for property (iv),
     i.e.  it's not required that axes with dim=1 must have stride=0, and the
     code does not have to be set.


        @param [in] pattern1   The first input pattern.  Must be valid
                               except for property (iv), and must have positive
                               strides.
        @param [in] pattern2   The second input pattern.  Must be valid
                               except for property (iv), and must have
                               the same strides (in the same order)
                               as pattern1.
        @param [out] patterns_out  The output patterns; this function will write
                               to this location a vector of disjoint patterns
                               whose union (viewed as a memory-index-set) is
                               identical to the intersection of pattern1
                               and pattern2.  The patterns in this vector will
                               be valid except for property (iv) [i.e. they
                               won't have zero strides for axes with dim=1], and
                               they will not have their code set.
  */
  static ComputeIntersection(const TensorPattern &pattern1,
                             const TensorPattern &pattern2,
                             std::vector<TensorPattern> *patterns_out) {
    patterns_out->clear();
    ComputeIntersection(pattern1, pattern2, pattern1.num_axes,
                        patterns_out);
  }

  /**
     In this recursive implementation of ComputeIntersection() [see version
     above for more information on pattern1, pattern2 and patterns_out], the
     user guarantees that for all axes with raxis-index `raxis >=
     identical_raxis`, pattern1 and pattern2 have the same dimension, and
     it may be assumed that we are only interested in the part of the
     intersection where the indexes are the same for pattern1 and pattern2,
     for all raxis >= identical_raxis.

     In this recursion, when we get to 'identical_raxis == 0', it means pattern1
     and pattern2 have identical dims and strides; and if they also have the
     same offset, all we need to do is append one of them to 'patterns_out'
     (otherwise this part of the intersection is empty; but note that this
     function may in general fork into two branches each time it recurses).
     This is all part of a process of trying to make the 'offset' identical
     between the two patterns by discarding some leading dimensions on one of
     the two patterns.  On raxis-indexes that we have processed, we also make
     the 'dim' the same by lopping off trailing dimensions.
  */
  static bool ComputeIntersection(const TensorPattern &pattern1,
                                  const TensorPattern &pattern2,
                                  int32 identical_raxis,
                                  std::vector<TensorPattern> *patterns_out);


  /**
     This function, called by ConvertToCommonStrides() converts a pattern in
     canonical form to a Pattern whose strides are equal to the
     provided 'common_strides' vector, and which is valid *except for*
     the axis-sorting (property (vi) of a valid Pattern) and
     for property (iv), that strides must be nonzero for axes
     with dim != 1.

         @param [in] pattern_in  The input pattern; must be valid and
                                 in canonical form.
         @param [in] common_strides  The list of strides.  Must be sorted,
                                 have the property that each element
                                 divides the next element, and all
                                 strides in pattern_in must be present
                                 in this list.
         @param [out] pattern_out   The output pattern.  Will be equivalent
                                 to pattern_in in terms of memory-index-set,
                                 its strides will be equal to 'common_strides'
                                 (including the order), and it will be valid
                                 except for properties (iv) and (vi), as
                                 mentioned above.
  */
  static void ConvertLazilyToCommonStrides(const TensorPattern &pattern_in,
                                           const std::vector<int32> &common_strides,
                                           TensorPattern* pattern_out);

  /**
     This function makes sure that the axis-sorting property in 'pattern'
     holds for the axis numbered 'raxis' (in the private numbering, of
     course).  I.e. it ensures that:

       `pattern->strides[raxis+1] >= pattern->strides[raxis] * pattern->dims[raxis]`

     If it does not already have this property, this function ensures that it
     does have it by modifying its dims for raxis and raxis + 1, and if necessary,
     moving part of the pattern to 'extra_pattern'.  This will be necessary if the
     value of `pattern->dims[raxis]` at entry is not a multiple of
     `pattern->strides[raxis+1] / pattern->strides[raxis]`.

         @param [in]      raxis    The axis on which we are doing the check
         @param [in,out]  pattern  The input pattern, valid except for properties
                                (iv) and (vi).  Its strides must be in
                                increasing order (in the private numbering) and
                                each must divide the next.
         @param [out]     extra_pattern   This function writes to 'extra_pattern' if
                                and only if it returns true.  See documentation of
                                return status.
         @return  Returns true if it wrote to extra_pattern.  If it returns true,
                  then it guarantees that the union of the memory-index-sets of
                  'pattern' and 'extra_pattern' at exit are equal to the memory-index-set
                  of 'pattern' at entry.  If it returns false, then it guarantees
                  that the memory-index-set of 'pattern' has been unchanged.
                  In either case it guarantees that property (vi), the axis-sorting
                  property, holds for axis 'raxis', in 'pattern' and (if applicable)
                  in `extra_pattern`.
                  The codes of pattern and extra_pattern are not set.
  */
  static bool EnsureAxisSortingPropertyHolds(int32 raxis,
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


bool IntersectionComputer::FindCommonStrides(std::vector<int32> *axes) {
  axes->clear();
  axes->reserve(pattern1_.num_axes + pattern2_.num_axes);
  for (int32 raxis = 0; raxis < pattern1_.num_axes; raxis++)
    axes->push_back(pattern1_.strides[raxis]);
  for (int32 raxis = 0; raxis < pattern2_.num_axes; raxis++)
    axes->push_back(pattern2_.strides[raxis]);
  SortAndUniq(axes);  // sort from least to greatest, remove duplicates.
  int32 prev_stride = (*axes)[0];
  size_t num_axes = axes->size();
  for (size_t i = 1; i < num_axes; i++) {
    int32 cur_stride = (*axes)[i];
    if (cur_stride % prev_stride != 0)
      return false;  // prev_stride does not divide cur_stride; our algorithm
                     // for detecting overlap cannot be used.  This shouldn't
                     // really happen in "reasonable" uses of Tensors.
    prev_stride = cur_stride;
  }
  return true;
}

bool IntersectionComputer::EnsureAxisSortingPropertyHolds(
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
      if (EnsureAxisSortingPropertyHolds(raxis, &((*patterns)[p]),
                                         &extra_pattern))
        patterns->push_back(extra_pattern);
    }
  }
}


// see declaration for documentation.
void IntersectionComputer::ComputeIntersection(
    const TensorPattern &pattern1,
    const TensorPattern &pattern2,
    int32 identical_raxis,
    std::vector<TensorPattern> *patterns_out) {
  if (identical_raxis == 0) {
    if (pattern1.offset == pattern2.offset) {
      patterns_out->push_back(pattern1);
      RemoveTrivialAxes(&(patterns_out->back()));
    }
    return;
  }
  // we'll be modifying the dims and strides on axis 'raxis'.
  int32 raxis = identical_raxis - 1,
      stride = pattern1.strides[raxis]; // will be the same in pattern2, and positive.

  // By the '?..:' statements below we possibly switch pattern2 and
  // pattern1, thereby ensuring that pattern2_mod.offset >= pattern1_mod.offset
  TensorPattern pattern1_mod(pattern2.offset >= pattern1.offset ? pattern1 : pattern2),
      pattern2_mod(pattern2.offset >= pattern1.offset ? pattern2 : pattern1);


  // pattern2_mod's offset is larger (or the same), so we may need to discard
  // some leading indexes of pattern1_mod (on axis 'raxis'), increasing the
  // offset and reducing the dim, to get the offsets closer to being the same,
  // and then take the min of the dims on that axis.

  // 'dim_discarded' below will be rounded down in the division, and we will
  // also need to also consider the value that's one larger than that.  We don't
  // need to consider any other values of 'dim_discarded' other than these two,
  // because it's possible to prove that if we recurse with the remaining offset
  // being greater than 'stride', we would never be able to get to offset=0
  // without discarding all dims of at least one axis numbered less than raxis.
  // The proof requires the axis-sorting property.
  int32 offset_diff = pattern2_mod.offset - pattern1_mod.offset,
      min_dim1_discarded = offset_diff / stride,
      max_dim1_discarded = ((offset_diff == min_dim1_discarded * stride) ?
                            min_dim1_discarded : min_dim1_discarded + 1);

  // Make a copy of the relevant dims, and pattern1's offset, because the
  // versions in the patterns may get modified in the loop.
  int32 pattern1_dim = pattern1_mod.dims[raxis],
      pattern2_dim = pattern2_mod.dims[raxis],
      pattern1_offset = pattern1.offset;
  for (int32 dim1_discarded = min_dim1_discarded;
       dim1_discarded <= max_dim1_discarded; dim1_discarded++) {
    pattern1_mod.offset = pattern1_offset + dim1_discarded * stride;
    int32 new_pattern1_dim = pattern1_dim - dim1_discarded;
    if (new_pattern1_dim <= 0)
      continue;
    pattern1_mod.dims[raxis] = new_pattern1_dim;
    // set both dims of pattern1_mod and pattern2_mod to the minimum
    // of the two dims.
    if (pattern2_dim > new_pattern1_dim) {
      pattern1_mod.dims[raxis] = new_pattern1_dim;
      pattern2_mod.dims[raxis] = new_pattern1_dim;
    } else {
      pattern1_mod.dims[raxis] = pattern2_dim;
      pattern2_mod.dims[raxis] = pattern2_dim;
    }
    // Recurse.  We would have continued above if we discarded all dims on this
    // axis.
    ComputeIntersection(pattern1, pattern2, raxis, patterns_out);
  }
}



bool IntersectionComputer::ComputeIntersection(
    const TensorPattern &pattern1,
    const TensorPattern &pattern2,
    std::vector<TensorPattern> *patterns_out) {
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
