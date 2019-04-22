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




/**
   This function, not declared in the header, creates a sorted list of all the
   stride values which are present in either 'pattern1' or 'pattern2'.  These
   will all be positive, since pattern1 and pattern2 are required to be in
   canonical form.

     @param [in] pattern1   First input pattern, must be in canonical form.
     @param [in] pattern2   Second input pattern, must be in canonical form.
     @param [out] strides   A sorted list of all stride values that are present
                            in either pattern1 or pattern2 will be written
                            to here.  There will be no repeats.
*/
static void FindAllStrides(
    const TensorPattern &pattern1,
    const TensorPattern &pattern2,
    std::vector<int32> *strides) {
  KALDI_PARANOID_ASSERT(IsCanonical(pattern1) && IsCanonical(pattern2));
  strides->clear();
  strides->reserve(pattern1_.num_axes + pattern2_.num_axes);
  for (int32 raxis = 0; raxis < pattern1.num_axes; raxis++)
    strides->push_back(pattern1.strides[raxis]);
  for (int32 raxis = 0; raxis < pattern2.num_axes; raxis++)
    strides->push_back(pattern2_.strides[raxis]);
  SortAndUniq(strides);  // sort from least to greatest; remove duplicates.
}



// See declaration in header.
bool IsRegular(const TensorPattern &pattern) {
  int32 num_axes = pattern.num_axes;

  for (int32 i = 0; i + 1 < num_axes; i++) {
    int32 this_stride = pattern.strides[i],
        this_dim = pattern.dims[i],
        this_prod = this_stride * this_dim;
    for (int32 j = i + 1; j < num_axes; j++) {
      if (pattern.strides[j] >= this_prod) {
        // in this case, 'j' would be the 'k' value used in the proof.  If we
        // fall off this loop, it would correspond to k == num_axes, which is
        // also OK.
        break;
      } else if (pattern.dims[j] != 1 ||
                 pattern.strides[j] % this_stride != 0) {
        return false;
      }
    }
  }
  return true;
}


/**
   This function, called by ConvertPatternStrides(), is not declared in the
   header.  It converts a pattern in canonical form to a Pattern whose strides
   are equal to the provided 'strides' vector, which is valid-2,
   satisfies the uniqueness property, and has normalized (i.e.
   positive and increasing) strides.

       @param [in] pattern_in  The input pattern; must be valid and
                               in canonical form.
       @param [in] strides     The list of strides which we want
                               'pattern_out' to have.  Must be a list of
                               positive integers sorted from least to
                               greatest with size <= KALDI_TENSOR_MAX_AXES,
                               and all strides in pattern_in must
                               be present in this list.
       @param [out] pattern_out  The output pattern (must not point to
                               pattern_in).  On exit its memory-index-set will
                               equal that of pattern_in; its strides will be
                               equal to 'strides' (including the order, when
                               numbered in the private numbering); it will
                               be valid-2 and satisfy the uniqueness property;
                               and it will be linear in pattern_in.
*/
static void ConvertPatternStridesLazily(
    const TensorPattern &pattern_in,
    const std::vector<int32> &strides,
    TensorPattern* pattern_out) {
  KALDI_PARANOID_ASSERT(IsCanonical(pattern_in));
  int32 num_axes_in = pattern_in.num_axes,
      num_axes_out = strides.size();
  pattern_out->num_axes = num_axes_out;
  pattern_out->code = -1;
  int32 raxis_in = 0;
  pattern_out->offset = pattern_in->offset;
  // The following code relies on pattern_in being in canonical form
  // (so its strides are in sorted order), and all of its strides being
  // present in the list 'strides'.
  for (int32 raxis_out = 0; raxis_out < num_axes_out; raxis_out++) {
    int32 stride = strides[raxis_out];
    pattern_out->strides[raxis_out] = stride;
    if (pattern_in.strides[raxis_in] == stride) {
      pattern_out->dims[raxis_out] = pattern_in.dims[raxis_in];
      pattern_in++;
    } else {
      pattern_out->dims[raxis_out] = 1;
    }
  }
  if (raxis_in != num_axes_in) {
    KALDI_ERR << "Something went wrong converting strides; trying to "
        "convert pattern with strides = " << StridesAsString(pattern_in)
              << " to strides " << ArrayAsString(strides);
  }
}



/**
   This function, not declared in the header, attempts to ensure that the axis-sorting
   property in a provided Pattern holds for the axis-index 'raxis' (in the private
   numbering, of course).  I.e. it ensures (for the pattern we are to modify) that:

      `pattern->strides[raxis+1] >= pattern->strides[raxis] * pattern->dims[raxis]`.

   This function expects that the pattern will also satisfy that property for
   all axis-indexes `0 <= i < raxis`, and will be valid--.  This function will
   always succeed if the pattern is regular (see IsRegular(), and "Regularity
   property" in the glossary).

   Ensuring this property exists may sometimes require splitting this Pattern up
   (i.e. adding extra Patterns); the union of their memory-index-sets together
   with that of the modified pattern will equal the memory-index-set of the
   original pattern at input (these sets being unioned will be disjoint).  Any
   newly created Patterns will be appended to the vector 'patterns'.

    @param [in]      raxis    The axis for which we are ensuring that the
                             axis-sorting property holds.
    @param [in]      pattern_index  The index in the vector 'patterns'
                             of the pattern for which we are ensuring that
                             the axis-sorting property holds.
    @param [in,out]  patterns  The vector of patterns in which to look for the
                             pattern to operate on; we may also append
                             Patterns to this vector if needed, as mentioned
                             above.  Note: the newly added patterns may not satisfy
                             the axis-sorting property for 'raxis', but they will
                             still satisfy it for all axes numbered less than
                             'raxis', assuming the pattern at 'pattern_index'
                             did at entry.

    @return                  Returns true on success, false on failure.
                             Will always return true if `(*patterns)[pattern_index]`,
                             satisfied the 'regularity property' at entry;
                             see IsRegular().
 */
static bool EnsureAxisSortingPropertyHolds(
    int32 raxis,
    int32 pattern_index,
    std::vector<TensorPattern> *patterns) {
  TensorPattern *pattern = (*patterns)[pattern_index];
  // We use 'i' as the internal name for 'raxis', because we want to mirror the
  // notation used for the regularity property in the glossary, and in the
  // function IsRegular() that checks for it.  There is an index k with `i < k
  // <= num_axes`, that appears in the definition of the regularity property.
  // The algorithm used here iteratively decreases the value of k until it
  // equals i + 1, adding new patterns as needed, at which point the
  // axis-sorting property will hold for index i.
  int32 i = raxis, num_axes = pattern->num_axes;
  int32 this_stride = pattern->strides[i],
      this_dim = pattern->dims[i],
      this_prod = this_stride * this_dim;
  if (this_dim == 1)  // This is a small optimization for a common case.
    return true;
  KALDI_PARANOID_ASSERT(raxis + 1 < num_axes && this_stride > 0 &&
                        ValidMM(*pattern));
  int32 j, k = num_axes;
  for (j = i + 1; j < num_axes; j++) {
    if (pattern->strides[j] >= this_prod) {
      k = j;
      break;  // regularity property is OK as far as this 'i' is concerned.
    } else if (pattern->dims[k] != 1 ||
               pattern->strides[k] % this_stride != 0) {
      return false;  // Pattern was not regular.
    }
  }
  for (; j = k - 1; j > i; j--) {
    int32 j_stride = pattern->strides[j],
        stride_ratio = j_stride / this_stride;  // will divide exactly; we
                                                     // checked above.
    KALDI_PARANOID_ASSERT(j_stride % this_stride == 0);

    // We can prove that j_dim will always be at least 1; if this is the
    // first time round the loop this is easy to show (else k would be smaller);
    // otherwise we can use the fact that the strides for axes i, i+1 .. k-1 are
    // strictly increasing and all multiples of this_stride (hence stride_ratio
    // strictly increases from one j to the next).
    int32 j_dim = this_dim / stride_ratio,
        remainder = this_dim % stride_ratio;

    if (remainder != 0) {
      patterns->resize(patterns->size() + 1);
      pattern = (*patterns)[i];  // in case it was reallocated.
      TensorPattern *remainder_pattern = &(patterns->back());
      *remainder_pattern = *pattern;
      remainder_pattern->dims[i] = remainder;
      remainder_pattern->offset += j_stride * j_dim;
    }

    pattern->dims[j] = j_dim;
    pattern->dims[i] = stride_ratio;
    this_prod = j_stride;
  }
  return true;
}


// see declaration in header for documentation.
bool ConvertPatternStrides(
    const TensorPattern &pattern,
    const ArrayRef<int32> &strides,
    std::vector<TensorPattern*> *patterns) {
  patterns->resize(1);
  ConvertPatternStridesLazily(pattern, &((*patterns)[0]));
  int32 num_axes = strides.size();
  for (int32 raxis = 0; raxis + 1 < num_axes; raxis++) {
    for (int32 p = 0; p < static_cast<int32>(patterns->size()); p++) {
      if (!EnsureAxisSortingPropertyHolds(raxis, p, patterns)){
        patterns->clear();
        return false;  // Couldn't be converted, because 'pattern' was not
                       // regular.
      }
    }
  }
#ifdef KALDI_PARANOID
  {
    int64 num_elements = NumElements(pattern),
        num_elements_check = 0;
    for (int32 p = 0; p < static_cast<int32>(patterns->size()); p++) {
      KALDI_ASSERT(IsValidM(*patterns)[p]);
      num_elements_check += NumElements((*patterns)[p]);
    }
    KALDI_ASSERT(num_elements == num_elements_check);
  }
#endif
  return true;
}

/**
   This recursive function is used to compute the intersection between
   pattern1 and pattern2, which must have identical num_axes and strides,
   must have normalized strides, and must be valid-1.  The user would call
   this with identical_raxis == pattern1.num_axes, and the recursion on
   identical_raxis takes care of the actual implementation.


        @param [in] pattern1   The first input pattern.  Must be valid-1 and
                               have normalized strides.
        @param [in] pattern2   The second input pattern.  Must be valid-1 and
                               have the same num_axes and strides as pattern1.
        @param [in] identical_raxis  Let num_axes be the num_axes of pattern1 or
                               pattern2 (it's the same).  By passing in
                               a particular value of identical_raxis, the caller
                               asserts that for all raxis with
                               identical_raxis <= raxis < num_axes,
                               `pattern1.dim[raxis] == pattern2.dim[raxis]`;
                               and furthermore that the caller is only
                               interested in the part of the overlap for which
                               pattern1 and pattern2 have the same index for all
                               raxis >= identical_raxis (and if there was
                               another part, it has been handled separately).
        @param [out] patterns_out  The output patterns; this function will
                               append to this location a number (possibly zero)
                               of disjoint valid patterns, each of which is
                               linear in pattern1 and pattern2, the union of whose
                               memory-index-sets is identical to the intersection
                               of pattern1 and pattern2's memory-index-sets.
  */
void ComputeIntersectionRecursive(const TensorPattern &pattern1,
                                  const TensorPattern &pattern2,
                                  int32 identical_raxis,
                                  bool keep_all_patterns,
                                  std::vector<TensorPattern> *patterns_out) {
  if (identical_raxis == 0) {
    /*
      The base-case of the recursion; if we reach here, it means pattern1 and
      pattern2 have identical dims and strides; and if they also have the same
      offset, all we need to do is append one of them to 'patterns_out'
      (otherwise this part of the intersection is empty).  This is all part of a
      process of trying to make the 'offset' identical between the two patterns
      by discarding some leading indexes on one of the two patterns, and
      discarding any trailing indexes as needed to make the dim the same.  (See
      "Index:" in glossary for clarity on its meaning here).
    */

    if (pattern1.offset == pattern2.offset) {
      size_t cur_size = patterns_out->size();
      patterns_out->resize(cur_size + 1);
      push_back(pattern1);
      RemoveTrivialAxes(pattern1, &(patterns_out[cur_size]));
    }
    return;
  }
  // we'll be modifying the dims and strides on axis 'raxis'.
  int32 raxis = identical_raxis - 1,
      stride = pattern1.strides[raxis]; // will be the same in pattern2, and positive.

  // By the '?..:' statements below we possibly switch pattern2 and
  // pattern1, thereby ensuring that pattern2_mod.offset >= pattern1_mod.offset;
  // this simplifies the later code.
  TensorPattern pattern1_mod(pattern2.offset >= pattern1.offset ? pattern1 : pattern2),
      pattern2_mod(pattern2.offset >= pattern1.offset ? pattern2 : pattern1);


  // pattern2_mod's offset is larger (or the same), so we may need to discard
  // some leading indexes of pattern1_mod (on axis 'raxis'), increasing
  // pattern1_mod's offset and reducing its dim on this raxis, to get the
  // offsets closer to being the same.

  // 'min_dim1_discarded' below will be rounded down in the division, and we will
  // also need to also consider the value that's one larger than that.  We don't
  // need to consider any other values of 'dim1_discarded' other than these two,
  // because it's possible to prove that if we recurse with the remaining offset
  // being greater than 'stride', we would never be able to get to offset=0
  // without discarding all dims of at least one axis numbered less than raxis.
  // The proof requires the axis-dominance property (together with normalized
  // strides).
  int32 offset_diff = pattern2_mod.offset - pattern1_mod.offset,
      min_dim1_discarded = offset_diff / stride,
      max_dim1_discarded = ((offset_diff == min_dim1_discarded * stride) ?
                            min_dim1_discarded : min_dim1_discarded + 1);

  // Make a copy of the relevant dims, and pattern1's offset, because the
  // versions in the patterns may get modified in the loop below.
  int32 pattern1_dim = pattern1_mod.dims[raxis],
      pattern2_dim = pattern2_mod.dims[raxis],
      pattern1_offset = pattern1.offset;
  for (int32 dim1_discarded = min_dim1_discarded;
       dim1_discarded <= max_dim1_discarded; dim1_discarded++) {
    pattern1_mod.offset = pattern1_offset + dim1_discarded * stride;
    int32 new_pattern1_dim = pattern1_dim - dim1_discarded;
    if (new_pattern1_dim <= 0)
      continue;  // There's no overlap here.
    pattern1_mod.dims[raxis] = new_pattern1_dim;
    // set both dims of pattern1_mod and pattern2_mod to the minimum
    // of the two dims.
    if (pattern2_dim > new_pattern1_dim) {
      pattern2_mod.dims[raxis] = new_pattern1_dim;
    } else {
      pattern1_mod.dims[raxis] = pattern2_dim;
      pattern2_mod.dims[raxis] = pattern2_dim;
    }
    // Recurse.
    ComputeIntersectionRecursive(pattern1, pattern2, raxis,
                                 keep_all_patterns, patterns_out);
  }
}


// See documentation in header.
bool ComputeIntersection(const TensorPattern &pattern1_in,
                         const TensorPattern &pattern2_in,
                         std::vector<TensorPattern> *intersection,
                         bool keep_all_patterns) {
  TensorPattern pattern1(pattern1_in),
      pattern2(pattern2_in);
  CanonicalizePattern(&pattern1);
  CanonicalizePattern(&pattern2);
  std::vector<int32> strides;
  FindAllStrides(pattern1, pattern2, &strides);
  int32 num_axes = strides.size();
  if (num_axes == 0) {
    // Some of the code below with num_axes - 1 would crash
    // in this case, so handle it separately.
    if (pattern1.offset == pattern2.offset) {
      intersection->resize(1);
      (*intersection)[0] = pattern1;
    } else {
      intersection->clear();
    }
    return true;
  }
  std::vector<TensorPattern> patterns1, patterns2;
  patterns1.reserve(8);
  patterns2.reserve(8);
  intersection->clear();
  if (!ConvertPatternStrides(pattern1, strides, &patterns1) ||
      !ConvertPatternStrides(pattern2, strides, &patterns2))
    return false;

  auto iter1 = patterns1.begin(), end1 = patterns1.end();
  for (; iter1 != end1; ++iter1) {
    Pattern &sub_pattern1 = *iter1;
    auto iter2 = patterns2.begin(), end2 = patterns2.end();

    // Below, 'max_mindex1' is not the actual largest mindex in `sub_pattern1`,
    // but an upper bound on it (in fact, it is strictly greater than it); to
    // prove this we require the axis-dominance property and the fact that the
    // strides are normalized (positive and increasing).  This is part of an
    // optimization to more quickly skip over pairs of patterns that will have
    // empty intersection.
    int64 min_mindex1 = sub_pattern1.mindex,
        max_mindex1 = min_mindex1 +
        sub_pattern1.strides[num_axes - 1] * sub_pattern1.dims[num_axes - 1];

    for (; iter2 != end2; ++iter2) {
      Pattern &sub_pattern2 = *iter2;
      int64 min_mindex2 = sub_pattern2.mindex,
          max_mindex2 = min_mindex2 +
          sub_pattern2.strides[num_axes - 1] * sub_pattern2.dims[num_axes - 1];
      if (min_mindex2 >= max_mindex1 || min_mindex1 >= max_mindex2)
        continue;  //  This is an optimization for efficiency when it's easy to
                   // see that two Patterns won't overlap.

      // Here, sub_pattern1 and sub_pattern2 are the sub-pieces of pattern1 and
      // pattern2 that have been converted to share the same list of strides
      // (That conversion process may end up splitting patterns into several
      // pieces, even if it was possible, which is not always; hopefuly there is
      // just one piece in each case, but there may be more).  The following
      // call may add elements to 'intersection'.
      ComputeIntersectionRecursive(sub_pattern1, sub_pattern2,
                                   num_axes,
                                   keep_all_patterns,
                                   intersection);
      if (!keep_all_patterns && !intersection.empty())
        return true;
    }
  }
  return true;
}

bool PatternContains(const TensorPattern &pattern_in,
                     int64 mindex) {
  TensorPattern pattern_mod;
  const Pattern *pattern;
  if (!IsCanonical(pattern_in)) {
    CanonicalizePattern(pattern_in, &pattern_mod);
    pattern = &pattern_mod;
  } else {
    pattern = &pattern_in;
  }
  mindex -= pattern->offset;
  int32 num_axes = pattern->num_axes;
  for (int32 raxis = num_axes - 1; raxis >= 0; raxis--) {
    int32 index = mindex / p->strides[raxis];
    // The following expression returns true if index is outside
    //  range [ 0, p->dims[raxis] - 1 ].
    if (static_cast<uint32>(index) >= static_cast<uint32>(p->dims[raxis]))
      return false;
    mindex -= p->strides[raxis] * index;
  }
  return (mindex == 0);
}



bool ToMemoryIndexSet(const TensorPattern &pattern_in,
                      std::vector<char> *s) {
  KALDI_PARANOID_ASSERT(pattern.IsValid());
  s->clear();
  TensorPattern pattern_mod;
  const Pattern *pattern;
  if (!IsCanonical(pattern_in)) {
    CanonicalizePattern(pattern_in, &pattern_mod);
    pattern = &pattern_mod;
  } else {
    pattern = &pattern_in;
  }
  int32 num_axes = pattern->num_axes;
  if (num_axes == 0)
    num_axes = 1;  // this does the right thing, as there will be dim=1,
                   // stride=0 physically present in the pattern.

  // 'max_mindex' is actually a strict upper bound on the maximum possible
  // memory-index, i.e. it is more than the largest possible memory-index.  We
  // rely on the axis-dominance property and also, thanks to the canonical form,
  // the fact that the strides are normalized (sorted and positive).
  int64 max_mindex = pattern->strides[num_axes - 1] *
      pattern->dims[num_axes - 1];
  s->clear();
  s->resize(max_mindex, static_cast<char>(0));

  auto recursively_set_elements = [pattern] (int32 raxis, int64 mindex) {
    int32 this_stride = pattern->strides[raxis],
         this_dim = pattern->dims[raxis];
    if (raxis == 0) {
      // Base case
      char *c = &((*s)[mindex]);
      for (int32 d = 0; d < this_dim; d++)
        c[d * static_cast<int64>(this_stride)] = static_cast<char>(1);
    } else {
      for (int32 d = 0; d < this_dim; d++)
        recursively_set_elements(raxis - 1, mindex + d * this_stride);
    }
  }
  recursively_set_elements(num_axes - 1, pattern->offset);
}

int64 RandomMemoryIndex(const TensorPattern &pattern) {
  int32 num_axes = pattern.num_axes;
  int64 mindex = pattern.offset;
  for (int32 raxis = 0; raxis < num_axes; raxis++) {
    mindex += RandInt(0, pattern.dims[raxis] - 1) * pattern.strides[raxis];
  }
  return mindex;
}


bool PatternsIntersectExhaustive(const TensorPattern &pattern1,
                                 const TensorPattern &pattern2) {
}


bool PatternsIntersect(const TensorPattern &pattern1,
                       const TensorPattern &pattern2) {
  KALDI_PARANOID_ASSERT(pattern1.IsValid() && pattern2.IsValid());
  int64 min_mindex1, max_mindex1,
      min_mindex2, max_mindex2;
  ComputeMinAndMaxMindex(pattern1, &min_mindex1, &max_mindex1);
  ComputeMinAndMaxMindex(pattern2, &min_mindex2, &max_mindex2);
  if (min_mindex2 > max_mindex1 ||
      min_mindex1 > max_mindex2)
    return false;

  // The next line is a check to see if one or other of the patterns includes
  // the first element of the other; this much faster than the algorithm for
  // computing pattern intersection.
  if (min_mindex2 >= min_mindex1) {
    if (PatternContains(pattern1, min_mindex2))
      return true;
  } else {
    if (PatternContains(pattern2, min_mindex1))
      return true;
  }

  bool keep_all_patterns = false;  // Settin keep_all_patterns to false sets
                                   // "fast mode", used where we just want to
                                   // see whether the intersection is empty.

  std::vector<TensorPattern> intersection;
  if (ComputeIntersection(pattern1, pattern2, &intersection,
                          keep_all_patterns)) {
    return (!intersection.empty());
  }

  // OK, if we reached here it was not possible to convert both patterns to the
  // same set of strides.  This is not expected to happen in practice for any
  // reasonable program.  Warn.
  static int32 num_warned = 0;
  int32 warn_limit = 10;
  if (num_warned < warn_limit) {
    num_warned++;
    KALDI_WARN << "Testing intersection of patterns that cannot be brought "
        "to common strides.  This will be extremely slow!";
  }

  // Randomly select 10 memory-indexes from the smaller pattern and see if it is
  // in the later pattern; this is faster than the next thing we'll try.
  const int32 num_draws = 10;
  if (NumElements(pattern1) < NumElements(pattern2)) {
    for (int32 i = 0; i < num_draws; i++)
      if (PatternContains(pattern2, RandomMemoryIndex(pattern1)))
        return true;
  } else {
    for (int32 i = 0; i < num_draws; i++)
      if (PatternContains(pattern1, RandomMemoryIndex(pattern2)))
        return true;
  }
  // OK, just try an exhaustive search.  If speed becomes an issue we may find a
  // way to disable the next check, which could be extremely slow for large
  // patterns.
  return PatternsIntersectSlow(pattern1, pattern2);
}

bool PatternsIntersectSlow(const TensorPattern &pattern1_in,
                           const TensorPattern &pattern2_in) {
  TensorPattern pattern1(pattern1_in),
      pattern2(pattern2_in);
  Canonicalize(&pattern1);
  Canonicalize(&pattern2);
  // Note: the offsets are the minimum elements, now that the
  // patterns are canonical.
  int64 min_offset = std::min(pattern1.offset, pattern2.offset);
  pattern1.offset -= min_offset;
  pattern2.offset -= min_offset;
  int64 max_offset = std::max(pattern1.offset, pattern2.offset);
  // Explicitly get the memory-index-set of pattern1 and pattern2
  // as possibly-huge arrays, and see if they intersect.  Obviously
  // this will be extremely slow.
  std::vector<char> pattern1_mindexes, pattern2_mindexes;
  ToMemoryIndexSet(pattern1, &pattern1_mindexes);
  ToMemoryIndexSet(pattern2, &pattern2_mindexes);
  auto iter1 = pattern1_mindexes.begin() + max_offset,
      iter2 = pattern2_mindexes.begin() + max_offset;
  for (; iter1 != pattern1_mindexes.begin() &&
           iter2 != pattern2_mindexes.end();
       ++iter1, ++iter2) {
    if (*iter1 && *iter2)
      return true;
  }
  return false;
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
