// tensor/tensor-pattern-extra-utils.h

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

#ifndef KALDI_TENSOR_TENSOR_PATTERN_EXTRA_UTILS_H_
#define KALDI_TENSOR_TENSOR_PATTERN_EXTRA_UTILS_H_ 1

#include "tensor/tensor-common.h"
#include "tensor/tensor-pattern.h"
#include "tensor/array-ref.h"


// This header includes various functions operating on Patterns,
// particularly ones relating to set-theoretic views of Patterns
// and obscure, less-user-facing ones.


namespace kaldi {
namespace tensor {


/**
   Returns true if there is overlap between pattern1 and pattern2,
   meaning that pattern1's memory-index-set and pattern2's
   memory-index-set have nonempty intersection.

         @param [in] First pattern.  Must be valid.
         @param [in] Second pattern.  Must be valid.
         @return  Return if the two patterns' memory-index-sets'
                  intersection is nonempty.
 */
bool PatternsIntersect(const TensorPattern &pattern1,
                       const TensorPattern &pattern2);


/**
   This is a slow but simple version of PatternsIntersect(), with the same
   interface.  it should not be called by users as it is slow.  It is exposed
   here for testing purposes.
*/
bool PatternsIntersectSlow(const TensorPattern &pattern1,
                           const TensorPattern &pattern2);


/**
   Returns information about whether pattern2's memory-index-set is a subset of
   pattern1's memory-index-set.  See glossary in tensor-pattern.h for
   explanation of memory-index-set.
        @param [in] pattern1  First input pattern; must be valid.
        @param [in] pattern2  First input pattern; must be valid.
        @return   Returns:
            0 if we determined that pattern1 does not include pattern2
            1 if we determined that pattern1 includes pattern2
           -1 if we could not compute the intersection (so our
              algorithm could not determine whether one included the other).
 */
int32 PatternIncludes(const TensorPattern &pattern1,
                      const TensorPattern &pattern2);


/**
   Returns true if the two patterns are equivalent in the sense that their
   memory-index-sets are the same.  See glossary in tensor-pattern.h for
   explanation.

   This function works by reducing both patterns to canonical form
   and testing whether their canonical forms are equal.

       @param [in] pattern1  First input pattern
       @param [in] pattern2  Second input pattern
       @return  Returns true if the patterns are equivalent, otherwise
                false.
 */
bool PatternsEquivalent(const TensorPattern &pattern1,
                        const TensorPattern &pattern2);


/**
   This function tries to compute the set-wise intersection between two patterns
   (i.e. the intersection between their memory-index-sets).  On success it
   outputs a vector of patterns rather than a single pattern, because this
   intersection may be empty or may not be expressible as a single pattern but
   only as a union of patterns (i.e. a union of the patterns this function
   outputs).  This function may fail to compute the intersection (see
   documentation of return status).

      @param [in] pattern1  The first of the two patterns of which
                        we want the intersection; must be valid.
      @param [in] pattern2  The first of the two patterns of which
                        we want the intersection; must be valid.
      @param [out] intersection  On success, this function outputs
                       a possibly-empty vector of patterns (in arbitrary
                       order), the union of whose memory-index-sets (which
                       will all be disjoint) equals the intersection fo the
                       memory-index-sets of `pattern1` and `pattern2`.
                       (However, see `keep_all_patterns`).
      @param [in]  keep_all_patterns   If this parameter is set to false,
                       the algorithm will stop as soon as the
                       `intersection` vector has one element.  This
                       is used for a fast test whether an intersection
                       is empty or ont.

      @return  Returns true if the intersection could be computed, and
               false otherwise.  This function will always return true if,
               when the strides of pattern1 and pattern2 are sorted and
               duplicates removed and listed in increasing order, each
               stride divides the next one in the list exactly; but this is
               not a necessary condition.   (The necessary condition
               is that both patterns, when compressed and converted
               to common strides, are "Regular" (c.f. "Regularity
               property" in glossary).
*/
bool ComputeIntersection(const TensorPattern &pattern1,
                         const TensorPattern &pattern2,
                         std::vector<TensorPattern> *intersection,
                         bool keep_all_patterns = true);


/**
   This function returns true if the memory-index-sets of pattern1 and pattern2
   have nonempty intersection, and false otherwise.  Requires that
   pattern1 and pattern2 be valid.

      @param [in] pattern1  First pattern to compare; must be valid.
      @param [in] pattern2  Second pattern to compare; must be valid.
      @return               Returns true if the memory-index-set of
                            pattern1 and pattern2 have nonempty intersection.
 */
bool PatternsIntersect(const TensorPattern &pattern1,
                       const TensorPattern &pattern2);

/**
      @param [in] pattern   The pattern about whose memory-index-set
                            we are asking.  Must be valid-1, or
                            return status is undefined.
      @param [in] mindex    The memory-index we are asking about
      @return               Return true if the memory-index-set of `pattern`
                            contains `mindex` (i.e. if there is an
                            index-tuple i such that `pattern[i] == mindex`;
                            see "Indexing a pattern" in the glossary.
*/
bool PatternContains(const TensorPattern &pattern,
                     int64 mindex);

/**
   Compute the minimum and maximum memory-indexs present in
   a pattern's memory-index-set (i.e. the minimum and maximum
   indexes into the underlying array).

   This is inlined for speed; see tensor-pattern-extra-utils-inl.h.

      @param [in] pattern  The pattern whose minimum and maximum
                           memory-index we are computing
      @param [out] min_mindex  The minimum memory-index in the
                           memory-index-set of the pattern.  Will
                           be zero in Patterns with non-negative
                           strides (e.g. Patterns in canonical form,
                           or other Patterns with normalized
                           strides).  Should always be >= 0 in
                           Patterns created by a valid program.
      @param [out] max_mindex  The maximum memory-index in the
                           memory-index-set of the pattern.
                           Will always be >= min_mindex.
*/
inline void ComputeMinAndMaxMindex(const TensorPattern &pattern,
                                   int64 *min_mindex,
                                   int64 *max_mindex);


/**
   Outputs the memory-index-set corresponding to the pattern 'pattern' to 's'.
   See glossary in tensor-pattern.h for definitions.

   This is strictly to be used in debugging code, as it is extremely
   inefficient.

      @param [in] pattern  The input pattern; must be valid
      @param [out] s   The memory-index-set, represented as a vector
                       of bool, actually stored as char.  This will be set to a
                       vector at least as large as the maximum memory-index in
                       `pattern`, containing 1 for memory-indexse in the set and 0 for
                       those out of the set.
 */
bool ToMemoryIndexSet(const TensorPattern &pattern,
                      std::vector<char> *s);

/**
   This function returns a memory-index randomly chosen
   from the memory-index-set of `pattern`.
     @param [in] pattern   Pattern; must be valid-1.
     @return  Returns randomly chosen memory-index.
 */
int64 RandomMemoryIndex(const TensorPattern &pattern);



/**
   Outputs the memory-index-tuple-set corresponding to the pattern 'pattern' to
   's' (see tensor-pattern.h for definition).  For storage in 's', each tuple is
   converted into a single integer by a hashing function that should keep
   distinct tuples separate as long as the memory-indexes were not huge.  (We
   may output the actual tuples at some point in the future if they are ever
   needed).

   This function is strictly to be used in debugging code, as it is
   extremely inefficient.

      @param [in] pattern  The input pattern
      @param [out] s   The memory-index-set
 */
bool ToMemoryIndexTupleSet(const ArrayRef<TensorPattern*>  patterns,
                           std::unordered_set<int64> *s);


/**
   Returns true if the two pattern-tuples are equivalent in the sense
   that their memory-index-tuple-sets are the same.  See glossary
   in tensor-pattern.h for explanation.
 */
bool PatternTuplesEquivalent(const ArrayRef<const TensorPattern*> patterns1,
                             const ArrayRef<const TensorPattern*> patterns2);

/**
   This function returns true if a Pattern is regular (see Regularity property
   in the glossary in tensor-pattern.h) and false otherwise.  'pattern' must
   have all positive strides, the strides must be in increasing order (in the
   private numbering), and it must be valid-2 (see glossary).
 */
bool IsRegular(const TensorPattern &pattern);


/**
   This function returns true if a Pattern is valid-1 (see definition in
   glossary); see also TensorPattern::Valid() and IsValidMM().
 */
bool IsValidM(const TensorPattern &pattern);

/**
   This function returns true if a Pattern is valid-2 (see definition in
   glossary); see also TensorPattern::Valid() and IsValidM().
 */
bool IsValidMM(const TensorPattern &pattern);


/**
   This function attempts to convert a pattern 'pattern' in canonical form
   (c.f. "Canonical form" in glossary, and CanonicalizePattern()) to a list of
   Patterns (see documentation of `patterns` below for note on their possible
   non-validity), whose strides (in the private numbering) are equal to the
   provided 'strides' vector, the union of whose memory-index-sets (which will
   all be disjoint) is equal to the memory-index-set of the input Pattern, and
   which are all linear in `pattern` (c.f. documentation of "Linear Property).

   This function is not guaranteed to always succeed (return true) but it will
   always succeed when people are doing "reasonable" things with Tensors.  It
   will always succeed if each element in 'strides' divides the next element
   exactly, although this is not a necessary condition for success.

       @param [in] pattern  A valid Pattern in canonical form
       @param [in] strides   A list of positive integers, sorted from
                        smallest to greatest; it must contain all strides in
                        `pattern`.
       @param [out] patterns  On success (see documentation of return status)
                        'patterns' will be set to a nonempty list of patterns,
                        the union of whose memory-index-sets equals the
                        memory-index-set of `pattern`; all of whose strides are
                        equal to `strides`; and each of which is valid-1 and
                        linear in `pattern` (see "Linear property").

                        except for property (iv) (search for "Valid
                        Pattern" in tensor-pattern.h): that is, they may have
                        nonzero strides for axes with dim == 1.  Each elements
                        of 'strides' dividing the next is a sufficient but not
                        necessary condition for this function to always return
                        true.
                          On failure, `patterns->empty()` will be empty.

        @return         Returns true if pattern strides could be converted using
                        our algorithm, false if not.  This algorithm will work
                        for any 'reasonable' request, but it doesn't attempt to
                        cover the types of cases where, to solve them, we would
                        have to output a number of patterns that couldn't be
                        bounded given the number of axes.
  */
bool ConvertPatternStrides(const TensorPattern &pattern,
                           const ArrayRef<int32> strides,
                           std::vector<TensorPattern> *patterns);


/**
   Class TensorPatternRebaser is an object that converts TensorPattern
   when memory layouts change.  The main use-case is when a base Variable
   (c.f. variable.h for definition) has a TensorPattern that is not
   contiguous (see tensor-pattern.h for definition of 'contiguous'), and
   its gradient Tensor is allocated contiguously.  This class is
   needed to convert patterns for Variables into patterns for their
   corresponding gradients.

   We make it an object rather than a function in order to avoid repetition when
   multiple patterns need to be rebased.
 */
class TensorPatternRebaser {

  /*
    Constructor.
       @param [in] src_pattern  The pattern that we are converting *from*,
                              e.g. the pattern of a Variable whose gradient
                              has a different layout from itself.
       @param [in] dest_pattern  The pattern that we are converting *to*.
                              Must have the same num_axes and the same dims
                              as 'src_pattern'.

    Let t be a valid index-tuple for src_pattern/dest_pattern, determined
    by their 'dims' and 'num_axes'.  Using t to index src_pattern and
    dest_pattern gives memory-indexes:
       m_src = src_pattern[t]
       m_dest = dest_pattern[t]
    View this object as a function from memory-indexes to memory-indexes
    (m_src -> m_dest), whose domain is the memory-index-set of src_pattern
    and whose range is the memory-index-set of dest_pattern.

    The purpose of this object is to modify patterns in a way that maps
    their memory-indexes with the same function.
  */
  TensorPatternRebaser(const TensorPattern &src_pattern,
                       const TensorPattern &dest_pattern);


  /**
     This function attempts to modify pattern->offset and pattern->strides in a
     way that does the mapping of memory-indexes m_src -> m_dest that is implied
     by the src_pattern and dest_pattern passed to the constructor.  That is,
     for any index-tuple t valid for 'pattern', the memory-index `pattern[t]`
     evaluated before and after calling this function gets mapped according
     to the function (m_src -> m_dest) mentioned in our documentation for
     the constructor.

     @param [in,out]  pattern  The pattern to be rebased.  Must, at entry,
                          satisfy `PatternIncludes(src_pattern, *pattern)`,
                          where `src_pattern` was the pattern passed to the
                          constructor.  On success (i.e. if this function
                          returns true), the condition
                          `PatternIncludes(dest_pattern, *pattern)` will
                          be satisfied.  On failure, the contents of
                          'pattern' is undefined.

     @return  Returns true if the conversion was possible.
   */
  bool Rebase(TensorPattern *pattern);

  private:

  // TODO: remove src_pattern_ and dest_pattern_ once everything
  // is debugged.  They are copies of the src_pattern and dest_pattern
  // passed to the constructor.
  TensorPattern src_pattern_;
  TensorPattern dest_pattern_;

  // If needs_conversion_ is false, it means the patterns don't need any conversion
  // at all (this is an optimization).
  bool needs_conversion_;

  // The 'offset' value of src_pattern_compressed (i.e. the src_pattern passed
  // to the constructor, which has been jointly compressed and normalized with
  // dest_pattern (to make all src_strides positive).
  int64 src_offset_;
  // The 'offset' value of dest_pattern_compressed
  int64 dest_offset_;

  // num_axes_ is the number of axes, not in the original src_pattern /
  // dest_pattern but after the two patterns have been jointly compressed and
  // then sorted from smallest to greatest stride in src_pattern.
  // src_strides_ are the resulting strides from src_pattern_compressed, and
  // dest_strides_ are the resulting strides from dest_pattern_compressed.

  // dest_pattern_ are the strides of the thus-modified src_pattern and
  // dest_pattern.  As an optimization, if src_strides and dest_strides end up
  // being the same, we set num_axes to zero and skip modifying the strides when
  // CompressPattern() is called.

  // Note: all of src_strides_[0] .. src_strides_[num_axes_ - 1] will be greater
  // than zero.  We can guarantee this because src_pattern and dest_pattern as
  // passed to the constructor had the same dims, so any axes with dim=1 would
  // have had dim=1 for both src and dest, hence they would have been removed by
  // CompressPatterns(), hence no strides would be zero after
  // CompressPatterns(); and CompressPatterns() normalizes the signs of the
  // strides so the first one (i.e. src_pattern) has positive strides.
  int32 num_axes_;
  int32 src_strides_[KALDI_TENSOR_MAX_DIM];
  int32 dest_strides_[KALDI_TENSOR_MAX_DIM];

  // The basic algorithm in Convert() is:
  //  First, add offset_ to its offset.
  //   Then:
  //     For each nontrivial axis of 'pattern', we are going to modify
  //     its stride as needed.
  //     Let that stride be `stride`, and the corresponding dim `dim`.
  //     Let `pstride = abs(stride)` be the absolute value of the stride
  //     (we'll modify that, and then restore the sign.
  //     positive.
  //



  // Converts a memory-index from the src to dest pattern.  This is applying,
  // to a single arbitrary memory-index m_src, the mapping (m_src -> m_dest);
  // see the comments above for explanation of this notation.
  // It is required that m >= 0 (otherwise it would not have been inside
  // the source pattern).
  int64 ConvertMemoryIndex(int64 m);

};


}  // namespace tensor
}  // namespace kaldi

// Include implementation of inline functions.
#include "tensor/tensor-pattern-extra-utils-inl.h"

#endif  // KALDI_TENSOR_TENSOR_PATTERN_EXTRA_UTILS_H_
