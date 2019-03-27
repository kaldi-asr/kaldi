#include "tensor/tensor-pattern-utils.h"

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {

int32 ComputePatternCode(const TensorPattern &pattern) {
  int32 ans = 0;

  int32 n = 0;
  // n is going to be:
  // n = 0 if no axis had stride=1, otherwise:
  // n = num_axes - (the axis that had stride=1).

  bool found_negative_dim = false;

  // caution: this axis index is a shifted-to-the-right index,
  // not the one that the public interface of Tensor exposes.
  for (int32 axis = KALDI_TENSOR_MAX_DIM - pattern.num_axes;
       axis < KALDI_TENSOR_MAX_DIM; axis++) {
    int32 dim = pattern.dims[axis],
        stride = pattern.strides[axis];
    if (dim != 1) {
      ans |= 1;  // set least significant bit of 'ans' to 1.
      if (dim < 0)
        found_negative_dim = true;
      if (stride == 1)
        n = KALDI_TENSOR_MAX_DIM - axis;
    }
    ans <<= 1;  // shift left by one.
  }

  // add in the value 'n' shifted 8 bits to the left,
  // and set the 11th bit if we found a negative dim.
  ans |= (n << 8) |  (found_negative_dim ? 1 << 11 : 0);
}


/**
   This function returns true if any of the tensor patterns in 'patterns'
   contains a negative stride; it works this out from their 'code'.
 */
static inline bool NegativeStrideExists(ArrayRef<TensorPattern*> patterns) {
  bool ans = false;
  for (size_t p = 0; p < patterns.size; p++)
    if ((patterns[0]->code | kPatternContainsNegativeStride) != 0)
      return true;
  return false;
}


/**
   This utility function used in CompressPatterns() normalizes the signs of the
   strides in all the dimensions, prior to any merging of axes, and sets the
   'data_offsets' variables.

   Consider an axis-index i (i.e. an index into the patterns' dims or strides
   vector).  We say that the strides for axis i
   are normalized if either all patterns have zero stride for that axis
   or the lowest-numbered pattern which has nonzero stride for that axis
   has positive stride for that axis.

   This type of normalization is done to increase the chance that we can combine
   axes, because the rule we use for combining axes only applies if any nonzero
   strides present have the same sign between the two axes.  In terms of being
   able to combine the maximum number of axes this rule is optimal, because any
   two axes where the pattern-index of the first pattern with a nonzero stride
   for those axes is different, would *not* be combinable.  So for any pair of
   axes that are potentially combinable according to that criterion and which
   have any nonzero strides, our normalization rule ensures that at least one
   pair of nonzero strides has the same sign.  If there were another pattern for
   which the sign was opposite after applying our rule, those two axes would not
   be combinable whatever the sign normalization.

     @param [in,out] patterns  The patterns to have their strides normalized
     @param [in]    max_num_axes  The maximum of any of the patterns'
                          num_axes (provided so we don't have to work it
                          out from 'patterns').
     @param [in,out] data_offsets  Data offsets, an array of dimension
                          patterns.size, which will be *added to* as needed by
                          this function, by the amount required to ensure that
                          the memory locations visited by the set of possible
                          indexes into these patterns is the same before and
                          after any change of sign.
     @return   Returns true if it made a change, else false.

   CAUTION!  Does not update the pattern code (the code for that is commented).
   If this were moved to a header we would have to make it update the pattern
   code.
 */
static inline bool NormalizeSigns(ArrayRef<TensorPattern*> patterns,
                                  int32 max_num_axes,
                                  int64 *data_offsets) {
  bool changed = false;
  size_t num_patterns = patterns.size;

  for (int32 a = KALDI_TENSOR_MAX_DIM - max_num_axes;
       a < KALDI_TENSOR_MAX_DIM; a++) {
    for (size_t p = 0; p < size; p++) {
      if (patterns[p]->strides[a] != 0) {
        // We have identified the first pattern-index with nonzero
        // stride for this axis
        if (patterns[p]->strides[a] < 0) {
          changed = true;
          // The stride is negative, so we have to flip it for this axis.
          // (Note: we flip it for all patterns, but we can ignore
          // pattern-indexes q < p because we know all those strides are zero.
          for (size_t q = p; q < size; q++) {
            // cast to int64 before muiltiplication to avoid potential
            // overflow
            if (patterns[q]->strides[a] != 0) {
              int64 this_offset =
                  static_cast<int64>(patterns[q]->dims[a] - 1) *
                  static_cast<int64>(patterns[q]->strides[a]);
              data_offsets[q] += this_offset;
              patterns[q]->strides[a] *= -1;
              // patterns[q]->code = -1;  // A signal to recompute the code.
            }
          }
        }
        // break from loop over patterns; we identified the first pattern-index
        // with nonzero stride for this axis, which is the only thing that
        // determines whether we change the sign of this axis.
        break;
      }
    }
  }
  //if(changed)
  //  for (size_t p = 0; p < size; p++)
  //    if (patterns[p]->code == -1)
  //      patterns[p]->code == GetDimsCode(*(patterns[p]));
  return changed;
}


/**
   This is a note on the semantics of combining dimensions in CompressPatterns.
   It is not a commutative property: Combinable(pattern, i, j) might not
   equal Combinable(pattern, j, i).

   We can only ever combine pairs of axes that were combinable for *all* patterns
   passed to CompressPatterns().

   When we combine axes we'll set dims[j] := dims[i] * dims[j], and make axis i
   a no-op by setting dims[i] = 1, strides[i] = 0.
 */
static inline bool Combinable(const TensorPattern &pattern,
                              int32 axis1, int32 axis2) {
  return pattern.strides[axis1] == pattern.strides[axis2] * pattern.dims[axis2];
}


// Returns true iff the axis 'axis' has zero stride (and hence dim=1)
// for all the supplied patterns.  An axis like this can be removed without
// affecting the result.
static inline bool AxisIsTrivial(ArrayRef<TensorPattern> patterns,
                                 int32 axis) {
  for (size_t p = 0; p < patterns.size; p++)
    if (patterns[p].strides[axis] != 0)
      return false;
  return true;
}

// Combine the two axes axis1 and axis2 in all the patterns (which the user
// asserts is possible); at exit, the lower numbered of the two axes is
// guaranteed to have dim=1, stride=0 in all patterns.  (we will later get rid
// of that trivial axis).  axis2 is the one with the smaller stride, and is the
// one whose stride we keep in the combined axis; that is the asymmetry
// between axis1 and axis2.
static inline void CombineAxes(ArrayRef<TensorPattern*> patterns,
                               int32 axis1, int32 axis2) {
  size_t num_patterns = patterns.size;
#ifdef KALDI_PARANOID
  for (size_t p = 0; p < num_patterns; p++) {
    KALDI_ASSERT(Combinable(*(patterns[p]), axis1, axis2));
  }
#endif
  if (axis1 < axis2) {
    // keep axis2.
    // We want the 'trivial' axis (the one with dim=1, stride=0 for all
    // patterns) to be the lower-numbered axis; this is more convenient for our
    // algorithm because we might later want to do further combination on the
    // nontrivial axis.
    for (size_t p = 0; p < num_patterns; p++) {
      TensorPattern *pattern = patterns[p];
      pattern->dims[axis2] *= pattern->dims[axis1];
      pattern->dims[axis1] = 1;
      pattern->strides[axis1] = 0;
    }
  } else {
    // keep axis1.
    for (size_t p = 0; p < num_patterns; p++) {
      TensorPattern *pattern = patterns[p];
      pattern->dims[axis1] *= pattern->dims[axis1];
      pattern->strides[axis1] = pattern->strides[axis2];
      pattern->dims[axis2] = 1;
      pattern->strides[axis2] = 0;
    }
  }
}

/**
   Removes trivial axes, defined as axes for which, for all patterns, dim=1 and
   stride=0.  Assumes the user has already found out which axes are trivial and
   is passing in this information as the array 'trivial_axis'.
   'max_num_axes' is required to be the maximum of the 'num_axes' of all the
   patterns passed in.  (or should at least be >= that value).

   This function removes those axes, shifts the dims and strides arrays to
   the right as needed, and decreases the 'num_axes' of the patterns
   appropriately (note: this is not as simple as just subtracting the number
   of axes removed, because removing an axis that was out of the range
   of valid axes for a given pattern needs to be a no-op).

   @param [in]  trivial_axis    An array which identifies the axes to
                       be removed.  At least one element must be true.
                       Note: since everything is shifted to the right,
                       a pattern with num_axes = 1 would have that
                       one axis at KALDI_TENSOR_MAX_DIM - 1.
   @param [in]  max_num_axes  The maximum of the num_axes of any
                       pattern in 'patterns'.
   @param [in,out]  patterns    The patterns to be modified.

   CAUTION: this function does not update the codes of 'patterns'.
 */
static void RemoveTrivialAxes(bool trivial_axis[KALDI_TENSOR_MAX_DIM],
                              int32 max_num_axes,
                              ArrayRef<TensorPattern*> patterns) {
  int32 last_trivial_axis = -1;
  for (int32 axis = KALDI_TENSOR_MAX_DIM - 1; axis >= 0; axis--) {
    if (trivial_axis[axis]) {
      last_trivial_axis = axis;
      break;
    }
  }
  KALDI_PARANOID_ASSERT(last_trivial_axis >= 0);

  for (size_t p = 0; p < patterns.size; p++) {
    TensorPattern *pattern = patterns[p];
    // Keep the axes right-justified.  We work from the right to the left.

    // We do the loop over axes inside the loop over p for memory locality.
    // We keep the axes shifted to the right so the loop goes backwards.
    int32 axis_out = last_trivial_axis,
        num_axes = pattern->num_axes;
    for (int32 axis_in = axis_out;
         axis_in >= KALDI_TENSOR_MAX_DIM - num_axes; axis_in--) {
      if (trivial_axis[axis_in]) {
        KALDI_PARANOID_ASSERT(pattern->dims[axis_in] == 1);
      } else {
        if (axis_out != axis_in) {
          pattern->dims[axis_out] = pattern->dims[axis_in];
          pattern->strides[axis_out] = pattern->strides[axis_in];
        }
        axis_out--;
      }
    }
    pattern->num_axes = KALDI_TENSOR_MAX_DIM - 1 - axis_out;
    // Make sure the axes we removed are set to dim=1, stride=0.
    for (; axis_out >= KALDI_TENSOR_MAX_DIM - num_axes; axis_out--) {
      pattern->dims[axis_out] = 1;
      pattern->strides[axis_out] = 0;
    }
  }
}

void CompressPatterns(ArrayRef<TensorPattern*> patterns,
                      int64_t *data_offsets) {
  size_t num_patterns = patterns.size;
  for (size_t p = 0; p < num_patterns; p++)
    data_offsets[p] = 0;
#ifdef KALDI_PARANOID
  KALDI_ASSERT(num_patterns > 0);
  for (size_t p = 0; p < num_patterns; p++) {
    KALDI_ASSERT(patterns[p]->code == ComputePatternCode(*(patterns[p])));
    for (size_t q = p + 1; q < num_patterns; q++) {
      KALDI_ASSERT(Broadcastable(*(patterns[p]), *(patterns[q])));
    }
  }
#endif

  int32 max_num_axes = patterns[0]->num_axes,
      combined_code = patterns[0]->code;
  for (size_t p = 1; p < num_patterns; p++) {
    max_num_axes = std::max<int32>(max_num_axes, patterns[p]->num-axes);
    combined_code |= patterns[p]->code;
  }
  bool changed = false;
  if (ContainsNegativeStride(combined_code))
    changed = NormalizeSigns(patterns, data_offsets);

  // note: the codes won't be fully up to date at this point.

  bool exists_trivial_axis = false;
  // The = {} ensures (I believe) that they are all set
  // to 0, meaning false.
  bool is_trivial_axis[KALDI_TENSOR_MAX_DIM] = {};

  for (int32 axis = KALDI_TENSOR_MAX_DIM - max_num_axes;
       axis < KALDI_TENSOR_MAX_DIM; axis++) {
    if (AxisIsTrivial(combined_code, axis)) {
      // note: we could optimize the AxisIsTrivial() thing slightly by shifting a
      // numer by 1 each time round the loop.
      is_trivial_axis[axis] = true;
      exists_trivial_axis = true;
      continue;
    }
    // see if axis i can be combined (in either direction with any later-numbered axis.
    for (int32 j = i + 1; j < KALDI_TENSOR_MAX_DIM; j++) {
      bool combinable_ij = true;
      for (size_t p = 0; p < num_patterns; p++) {
        if (!Combinable(patterns[p], i, j)) {
          combinable_ij = false;
          break;
        }
      }
      if (combinable_ij) {
        CombineAxes(patterns, i, j);
        is_trivial_axis[i] = true;
        exists_trivial_axis = true;
        changed = true;
        // Break from the loop on j and continue over the loop on i, meaning
        // we are done combining with the i'th axis.  At this point all the
        // (strides,dims) for axis i are just
        break;
      }
      bool combinable_ji = true;
      for (size_t p = 0; p < num_patterns; p++) {
        if (!Combinable(patterns[p], j, i)) {
          combinable_ji = false;
          break;
        }
      }
      if (combinable_ji) {
        CombineAxes(patterns, j, i);
        is_trivial_axis[i] = true;   // not a typo.  Lower-numbered axis gets
                                     // dim=1,stride=0.
        exists_trivial_axis = true;
        changed = true;
        break;
      }
    }
  }
  if (exists_trivial_axis) {
    RemoveTrivialAxes(num_axes, trivial_axis, patterns);
    changed = true;
  }
  if (changed)
    for (size_t p = 0; p < num_patterns; p++)
      patterns[p]->code = ComputePatternCode(*(patterns[p]));
  return changed;
}


void CompressOnePattern(TensorPattern *pattern,
                        int64 *data_offset) {
  // We may at some point implement this specially; doing this would be more efficient.
  CompressPatterns({pattern}, data_offset);
}



int32 GetDimsCode(const TensorPattern &pattern) {
  // we may not need this after all.
}


}  // namespace kaldi
}  // namespace tensor
x
