#include "tensor/tensor-pattern-utils.h"

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {

/**
   This function returns true if any of the tensor patterns in 'patterns'
   contains a negative stride.  All patterns are assumed to have the same
   num-axes.
 */
static inline bool NegativeStrideExists(ArrayRef<TensorPattern> patterns) {
  bool ans = false;
  int32 num_axes = patterns[0]->num_axes;  // required
  for (size_t p = 0; p < patterns.size; p++) {
    const TensorPattern &pattern = patterns[p];
    for (int32 i = 0; i < num_axes; i++) {
      if (pattern->strides[i] < 0)
        return true;
    }
  }
  return false;
}


/**
   This utility function used in CompressPatterns() normalizes the signs of the
   strides in all the dimensions, prior to any merging of axes, and sets the
   'data_offsets' variables.

   Consider an axis-index 0 <= i < num_axes.  We say that the strides for axis i
   are normalized if the the lowest-numbered pattern which has nonzero stride on
   axis i (if such a pattern exists) is positive.  If, on the other hand, all
   the strides are zero, we also say that it is normalized (since flipping the
   sign would make no difference).

   This type of normalization is done to increase the chance that we can combine
   axes, because the rule we use for combining axes only applies if any nonzero
   strides present have the same sign between the two axes.  In terms of being
   able to combine axes this rule is optimal, because any two axes where the
   pattern-index of the first pattern with a nonzero stride for those axes is
   different, would *not* be combinable.  So for any pair of axes that are
   potentially combinable according to that criterion and which have any nonzero
   strides, our normalization rule ensures that at least one pair of nonzero
   strides has the same sign.  If there were another pattern for which the sign
   was opposite after applying our rule, those two axes would not be combinable
   whatever the sign normalization.

     @param [in,out] patterns  The patterns to have their strides normalized
     @param [in,out] data_offsets  Data offsets, an array of dimension
                          patterns.size, which will be *added to* by this
                          function, by the amount required to ensure that
                          the memory locations visited by the set of possible
                          indexes into these patterns is the same before
                          and after any change of sign.
 */
static inline void NormalizeSigns(ArrayRef<TensorPattern> patterns,
                                  int64 *data_offsets) {
  size_t num_patterns = patterns.size;
  int32 num_axes = patterns[0].num_axes;
  for (int32 a = 0; a < num_axes; a++) {
    for (size_t p = 0; p < size; p++) {
      if (patterns[p].strides[a] != 0) {
        // We have identified the first pattern-index with nonzero
        // stride for this axis
        if (patterns[p].strides[a] < 0) {
          // The stride is negative, so we have to flip it
          // for this axis.  (Note: we flip it for all patterns,
          // for this dim, but we can ignore q < p because
          // we know all those strides are zero.
          for (size_t q = p; q < size; q++) {
            // cast to int64 before muiltiplication to avoid potential
            // overflow
            int64 this_offset =
                static_cast<int64>(patterns[q].dims[a] - 1) *
                static_cast<int64>(patterns[q].strides[a]);
            data_offsets[q] += this_offset;
            patterns[q].strides[a] *= -1;
          }
        }
        // break from loop over patterns; we identified the first pattern-index
        // with nonzero stride for this axis, which is the only thing that
        // determines whether we change the sign of this axis.
        break;
      }
    }
  }
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

// Combine the two axes axis1 and axis2 in all the patterns (which
// the user asserts is possible); at exit, the lower numbered of the
// two axes is guaranteed to have dim=1, stride=0 in all patterns.
// (we will later get rid of that trivial axis).
// axis2 is the one with the smaller stride (for patterns where the
// stride is nonzero), and is the one whose stride we keep in the
// combined axis; that is the asymmetry.
static inline void CombineAxes(ArrayRef<TensorPattern> patterns,
                               int32 axis1, int32 axis2) {
  size_t num_patterns = patterns.size;
#ifdef KALDI_PARANOID
  for (size_t p = 0; p < num_patterns; p++) {
    KALDI_PARANOID_ASSERT(Combinable(patterns[p], axis1, axis2));
  }
#endif
  if (axis1 < axis2) {
    // the if-statement is because we want the 'trivial' axis (the one with
    // dim=1, stride=0 for all patterns) to be the lower-numbered axis; this is
    // more convenient for our algorithm because we might later want to do
    // further combination on the nontrivial axis (if the lower-numbered one
    // were changed, we might repeat the search for an axis to combine with it.
    for (size_t p = 0; p < num_patterns; p++) {
      TensorPattern &pattern = patterns[p];
      pattern.dims[axis2] *= pattern.dims[axis1];
      pattern.dims[axis1] = 1;
      pattern.strides[axis1] = 0;
    }
  } else {
    for (size_t p = 0; p < num_patterns; p++) {
      TensorPattern &pattern = patterns[p];
      pattern.dims[axis2] *= pattern.dims[axis1];
      pattern.strides[axis2] = pattern.strides[axis1];
      pattern.dims[axis1] = 1;
      pattern.strides[axis1] = 0;
    }
  }
}

/**
   Removes trivial axes, defined as axes for which, for all patterns,
   dim=1 and stride=0.  Assumes the user has already which axes
   are trivial and passes in as the array 'trivial_axis'.
 */
inline static bool RemoveTrivialAxes(int32 num_axes,
                                     bool trivial_axis[],
                                     ArrayRef<TensorPattern> patterns) {
  for (size_t p = 0; p < patterns.size; p++) {
    const TensorPattern &pattern = patterns[p];
    // we do the loop over axes inside the loop over p for memory locality.
    int32 axis_out = 0;
    for (int32 axis_in = 0; axis_in < num_axes; axis_in++) {
      if (axis_out != axis_in && !trivial_axis[axis_in]) {
        pattern.dims[axis_out] = pattern.dims[axis_in];
        pattern.dims[axis_out] = pattern.dims[axis_in];
      }
      if (!trivial_axis[axis_in])
        axis_out++;
    }
    pattern.num_axes = axis_out;  // will be the same for all p.
  }
}

void CompressPatterns(ArrayRef<TensorPattern> patterns,
                      int64_t *data_offsets) {
  size_t num_patterns = patterns.size;
  for (size_t p = 0; p < num_patterns; p++)
    data_offsets[p] = 0;
#ifdef KALDI_PARANOID
  // check the input
  KALDI_ASSERT(num_patterns > 0 && num_patterns < 6);
  for (size_t p = 0; p < num_patterns; p++) {
    for (size_t q = p + 1; q < num_patterns; q++) {
      KALDI_ASSERT(Broadcastable(patterns[p], patterns[q]));
    }
  }
#endif
  if (NegativeStrideExists(patterns))
    NormalizeSigns(patterns, data_offsets);
  bool is_trivial_axis[6] = { false, false, false, false, false, false }
  bool exists_trivial_axis = false;
  int32 num_axes = patterns[0].num_axes;
  for (int32 i = 0; i < num_axes; i++) {
    if (AxisIsTrivial(patterns, i)) {
      is_trivial_axis[i] = true;
      exists_trivial_axis = true;
      continue;
    }
    // see if axis i can be combined (in either direction with any later-numbered axis.
    for (int32 j = i + 1; j < num_axes; j++) {
      bool combinable_ij = true, combinable_ji = true;
      for (size_t p = 0; p < num_patterns; p++) {
        if (!Combinable(patterns[p], i, j))
          combinable_ij = false;
        if (!Combinable(patterns[p], j, i))
          combinable_ji = false;
      }
      if (combinable_ij) {
        CombineAxes(patterns, i, j);
        is_trivial_axis[i] = true;
        exists_trivial_axis = true;
        // Break from the loop on j and continue over the loop on i, meaning
        // we are done combining with the i'th axis.  At this point all the
        // (strides,dims) for axis i are just
        break;
      } else if (combinable_ji) {
        CombineAxes(patterns, j, i);
        is_trivial_axis[i] = true;   // not a typo.  Lower-numbered axis gets
        // dim=1,stride=0.
        exists_trivial_axis = true;
        break;
      }
    }
  }
  if (exists_trivial_axis)
    RemoveTrivialAxes(num_axes, trivial_axis, patterns);
}


void CompressOnePattern(TensorPattern *pattern,
                        int64 *data_offset) {
}


  int32 GetDimsCode(const TensorPattern &pattern) {
  }


}  // namespace kaldi
}  // namespace tensor
x
