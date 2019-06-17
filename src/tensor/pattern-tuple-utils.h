// tensor/pattern-tuple-utils.h

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

#ifndef KALDI_TENSOR_TENSOR_PATTERN_TUPLE_UTILS_H_
#define KALDI_TENSOR_TENSOR_PATTERN_TUPLE_UTILS_H_ 1

#include "tensor/tensor-common.h"
#include "tensor/pattern.h"
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
bool PatternsIntersect(const Pattern &pattern1,
                       const Pattern &pattern2);


/**
   This is a slow but simple version of PatternsIntersect(), with the same
   interface.  it should not be called by users as it is slow.  It is exposed
   here for testing purposes.
*/
bool PatternsIntersectSlow(const Pattern &pattern1,
                           const Pattern &pattern2);


/**
   If i is an index-tuple (in the private numbering) valid for `pattern`,
   returns the memory-index
     `m = pattern[i] = pattern.offset + \sum_r i[r] * pattern.strides[r]`.
   If `check_valid == true` this will crash for i not in the index-tuple-set
   of `pattern`; if false it will just return the above expression computedn
   for i and not check.
 */
int64 IndexPattern(const Pattern &pattern,
                   const std::vector<int32> &i,
                   bool check_valid = true);


/**
   FindOffsets() is a utility function used in computing pattern intersections
   and set differences.  We will be using the notation described "Indexing a
   Pattern" in pattern.h.  Let pattern1 and pattern2 be patterns satisfying
   SameStrides(pattern1, pattern2).  Let n be the num-axes of the patterns.
   Let Offsets(pattern1, pattern2) be the set of n-tuples o such that there
   exists an i with pattern1[i + o] = pattern2[i], with of course i + o in the
   index-tuple-set of pattern1 and i in the index-tuple-set of pattern2.
   This function outputs the set of such offsets o.  The algorithm is a little
   complicated so we describe it with the implementation.

       @param [in] pattern1   First input pattern.  Must be valid-1 and
                        normalized+ (i.e. HasNormalizedPositiveStrides(pattern1)).
       @param [in] pattern2   Second input pattern.  Must be valid-1 and
                        satisfy SameStrides(pattern1, pattern2).
       @param [in] find_all_offsets  True if the user wants all of the
                        offsets.  If false, this function may save computation
                        by stopping after one or more offsets.  (Useful in
                        testing whether patterns intersect).
       @param [out] offsets   The offsets will be written to here in
                         arbitrary order.  Each offset will be a vector with
                         size() equal to the num_axes of the patterns; the
                         elements may be positive or negative.

   See also (not all of these are declared in headers), OffsetToPattern(),
   OffsetToHyperrectangle()).
 */
bool FindOffsets(const Pattern &pattern1,
                 const Pattern &pattern2,
                 bool find_all_offsets,
                 std::vector<std::vector<int32> > *offsets);



/**
   Returns information about whether pattern2's memory-index-set is a subset of
   pattern1's memory-index-set.  See glossary in pattern.h for
   explanation of memory-index-set.
        @param [in] pattern1  First input pattern; must be valid.
        @param [in] pattern2  First input pattern; must be valid.
        @return   Returns:
            0 if we determined that pattern1 does not include pattern2
            1 if we determined that pattern1 includes pattern2
           -1 if we could not compute the intersection (so our
              algorithm could not determine whether one included the other).
 */
int32 PatternIncludes(const Pattern &pattern1,
                      const Pattern &pattern2);

/**
   Inline function that sets dim=1, stride=0 for all axes with
   num_axes <= raxis < KALDI_TENSOR_MAX_DIM.  Often useful.
 */
inline void SetUnusedDimsAndStrides(int32 num_axes,
                                    Pattern *dest) {
#pragma unroll(2)
  for (int32 raxis = num_axes; raxis < KALDI_TENSOR_MAX_DIM; raxis++) {
    dest->dims[raxis] = 1;
    dest->strides[raxis] = 0;
  }
}

/**
   Inline function that sets dest->code = -1 and dest->properties = 0;
   often saves coding in functions that create or modify patterns.
 */
inline void SetDefaultCodeAndProperties(Pattern *dest) {
  dest->code = -1;
  dest->properties = 0;
}


/**
   Returns true if the two patterns are equivalent in the sense that their
   memory-index-sets are the same.  See glossary in pattern.h for
   explanation.

   This function works by reducing both patterns to canonical form
   and testing whether their canonical forms are equal.

       @param [in] pattern1  First input pattern
       @param [in] pattern2  Second input pattern
       @return  Returns true if the patterns are equivalent, otherwise
                false.
 */
bool PatternsEquivalent(const Pattern &pattern1,
                        const Pattern &pattern2);


/**
   This function tries to compute the set-wise intersection between two patterns
   (i.e. the intersection between their memory-index-sets).  On success it
   outputs a vector of patterns rather than a single pattern, because this
   intersection may be empty or may not be expressible as a single pattern but
   only as a union of patterns (i.e. a union of the patterns this function
   outputs).  This function may fail to compute the intersection in certain
   very pathological cases (see documentation of return status).

      @param [in] pattern1  The first of the two patterns of which
                        we want the intersection; must be valid.
      @param [in] pattern2  The first of the two patterns of which
                        we want the intersection; must be valid.
      @param [in]  keep_all_patterns   If this parameter is false,
                       the algorithm will stop as soon as the
                       `intersection` vector has one element.  This
                       is used for a fast test whether an intersection
                       is empty or not.
      @param [out] intersection  On success, this function outputs
                       a possibly-empty vector of patterns (in arbitrary
                       order), the union of whose memory-index-sets (which
                       will all be disjoint) equals the intersection of the
                       memory-index-sets of `pattern1` and `pattern2`.
                       (However, see `keep_all_patterns`).
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
bool ComputeIntersection(const Pattern &pattern1,
                         const Pattern &pattern2,
                         bool keep_all_patterns,
                         std::vector<Pattern> *intersection);



/**
   This function tries to compute the set-wise difference pattern1 - pattern2.
   Viewed as memory-index-sets, it is trying to compute the set of
   memory-indexes in pattern1 but not in pattern2.  This is computed as a list
   of Patterns.  This function may fail to compute the set difference in
   certain very pathological cases (see documentation of return status).

      @param [in] pattern1  The pattern we are subtracting from;
                       if it does not intersect with pattern2, the
                       result will be identical to pattern1.
                       Must be valid.
      @param [in] pattern2  The pattern we are subtracting; must be valid.
      @param [out] difference  On success, this function outputs
                     a possibly-empty vector of patterns (in arbitrary
                     order), the union of whose memory-index-sets (which
                     will all be disjoint) equals the set-wise difference
                     M(pattern1) - M(pattern2) of the memory-index-sets of
                     `pattern1` and `pattern2`.
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
bool ComputeDifference(const Pattern &pattern1,
                       const Pattern &pattern2,
                       std::vector<Pattern> *difference);




/**
   This function returns true if the memory-index-sets of pattern1 and pattern2
   have nonempty intersection, and false otherwise.  Requires that
   pattern1 and pattern2 be valid.

      @param [in] pattern1  First pattern to compare; must be valid.
      @param [in] pattern2  Second pattern to compare; must be valid.
      @return               Returns true if the memory-index-set of
                            pattern1 and pattern2 have nonempty intersection.
 */
bool PatternsIntersect(const Pattern &pattern1,
                       const Pattern &pattern2);

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
bool PatternContains(const Pattern &pattern,
                     int64 mindex);


/**
   Returns true if the memory-index-set of pattern p is a subset of the
   memory-index-set of pattern q.  Note: the algorithm is not super trivial or
   fast (although the tiem taken doesn't grow with the dims or strides, only
   with the number of axes).

      @param [in] p   First pattern; must be valid.
      @param [in] q   Second pattern; must be valid.
      @return   Returns true if memory-index-set of p is a subset of
                the memory-index-set of q (see pattern.h for definition;
                of memory-index-set).
 */
bool PatternIsSubsetOf(const Pattern &p,
                       const Pattern &q);


/**
   Compute the minimum and maximum memory-indexes present in
   a pattern's memory-index-set (i.e. the minimum and maximum
   indexes into the underlying array).

      @param [in] pattern  The pattern whose minimum and maximum
                           memory-index we are computing
      @param [out] min_mindex  The minimum memory-index in the
                           memory-index-set of the pattern.  Will be zero in
                           Patterns with non-negative strides.  Will always be
                           >= 0 in Patterns created by a program that's
                           doing something that makes sense.
      @param [out] max_mindex  The maximum memory-index in the
                           memory-index-set of the pattern.  Will always be >=
                           min_mindex.
*/
void ComputeMinAndMaxMindex(const Pattern &pattern,
                            int64 *min_mindex,
                            int64 *max_mindex);



/**
   Outputs the memory-index-set corresponding to the pattern 'pattern' to 's'.
   See glossary in pattern.h for definitions.

   This is strictly to be used in debugging code, as it is extremely
   inefficient.

      @param [in] pattern  The input pattern; must be valid
      @param [out] s   The memory-index-set, represented as a vector
                       of bool, actually stored as char.  This will be set to a
                       vector at least as large as the maximum memory-index in
                       `pattern`, containing 1 for memory-indexse in the set and 0 for
                       those out of the set.
 */
bool ToMemoryIndexSet(const Pattern &pattern,
                      std::vector<char> *s);

/**
   This function returns a memory-index randomly chosen
   from the memory-index-set of `pattern`.
     @param [in] pattern   Pattern; must be valid-1.
     @return  Returns randomly chosen memory-index.
 */
int64 RandomMemoryIndex(const Pattern &pattern);



/**
   Outputs the memory-index-tuple-set corresponding to the pattern 'pattern' to
   's' (see pattern.h for definition).

   This function is strictly to be used in debugging code, as it is
   extremely inefficient.

      @param [in] pattern  The input pattern
      @param [out] s   The memory-index-tuple-set
 */
bool ToMemoryIndexTupleSet(const ArrayRef<Pattern*>  patterns,
                           std::unordered_set<std::vector<int32>, VectorHasher> *s);


/**
   Returns true if the two pattern-tuples are equivalent in the sense
   that their memory-index-tuple-sets are the same.  See glossary
   in pattern.h for explanation.
 */
bool PatternTuplesEquivalent(const ArrayRef<const Pattern*> patterns1,
                             const ArrayRef<const Pattern*> patterns2);

/**
   Returns true if Pattern p is linear in Pattern q.  (Note:
   this is a rather technical property, see pattern.h for definition).

      @param [in] p  The first pattern.  Must be valid
      @param [in] q  The second pattern.  Must be valid and must satisfy
                     `PatternIsSubsetOf(p, q);`
 */
bool IsLinearIn(const Pattern &p,
                const Pattern &q);

/**
   This function returns true if a Pattern is regular (see Regularity property
   in the glossary in pattern.h) and false otherwise.  'pattern' must
   have all positive strides, the strides must be in increasing order (in the
   private numbering), and it must be valid-2 (see glossary).
 */
bool IsRegular(const Pattern &pattern);


/**
   This function returns true if a Pattern is valid-1 (see definition in
   glossary); see also Pattern::Valid() and IsValid2().
 */
bool IsValid1(const Pattern &pattern);

/**
   This function returns true if a Pattern is valid-2 (see definition in
   glossary); see also Pattern::Valid() and IsValid1().
 */
bool IsValid2(const Pattern &pattern);


/**
   This function attempts to convert a pattern 'pattern' in canonical form
   (c.f. "Canonical form" in glossary, and CanonicalizePattern()) to a list of
   valid-1 Patterns whose strides (in the private numbering) are equal to the
   provided 'strides' vector, the union of whose memory-index-sets (which will
   all be disjoint) is equal to the memory-index-set of the input Pattern, and
   which are all linear in `pattern` (c.f. documentation of "Linear Property).

   This function is not guaranteed to always succeed (return true), but it will
   always succeed when people are doing "reasonable" things with Tensors.  It
   will always succeed if each element in 'strides' divides the next element
   exactly, although this is not a necessary condition for success.

       @param [in] pattern  A valid Pattern in canonical form
       @param [in] strides   A list of positive integers, sorted from
                        smallest to greatest; it must contain all strides in
                        `pattern`.
       @param [out] patterns  On success (see documentation of return status)
                        'patterns' will be set to a nonempty list of valid-1
                        patterns, the union of whose memory-index-sets equals
                        the memory-index-set of `pattern`; all of whose strides
                        are equal to `strides`; and each of which is linear in
                        `pattern` (see "Linear property").
                           On failure, 'patterns' will be empty.
       @return          Returns true if pattern strides could be converted using
                        our algorithm, false if not.  This algorithm will work
                        for any 'reasonable' request, but it doesn't attempt to
                        cover the types of cases where, to solve them, we would
                        have to output a number of patterns that couldn't be
                        bounded given only the number of axes.
  */
bool ConvertPatternStrides(const Pattern &pattern,
                           const ArrayRef<int32> strides,
                           std::vector<Pattern> *patterns);

/**
   This function fills in any 'gaps' in the memory-indexes in 'src' and
   shifts so the lowest memory-index is 0, copying the resulting pattern
   to 'dest'.  It is used when constructing gradient Tensors for
   base Variables whose data Tensor is not contiguous and justified.

   The more mathematical description is as follows:
   Let m be the memory-index-set of `src`, and let f
   be the function that maps m to the set  \f$ [0, |m|-1] \f$ while
   preserving the ordering of the elements.  Then the relationship
   between 'src' and 'dest' is that 'dest' has the same num_axes and
   dims and 'src', and the strides are such as to satisfy
   \f$  dest[i] = f(src[i]) \f$,
   where i is a valid Index-tuple for `src`.  See "Indexing a Pattern"
   in the glossary in pattern.h for explanation of this notation.

         @param [in] src  The source pattern.  Must be valid.
         @param [out] dest  The destination pattern.  Will be identical
                        to `src` if `CompactAndJustified(src)`, else
                        will have the relationship explained above.
                        Will satisfy `CompactAndJustified(*dest)`,
                        and also `IsValid(*dest)`, assuming `IsValid(src)`.
 */
void MakeCompactAndJustified(const Pattern &src,
                             Pattern *dest);


/**
   This function possibly modifies the offset of the pattern `p`
   so that it will be justified (meaning: lowest-numbered
   memory-index equals zero).

     @param [in,out] p    A Pattern, must be valid at entry
                         (`p->IsValid()`).  At exit, will be
                         valid and also justified (`IsJustified(p)`).
 */
void MakeJustified(Pattern *p);


/**
   This function copies the Pattern 'src' from 'dest', preserving the
   num_axes and dims while possibly modifying the strides and offset.  The
   strides of 'dest' will be normalized (i.e. nonnegative with positive strides
   strictly increasing in the private axis-numbering), the pattern will be
   compact (no gaps) and the offset will be set to zero (making the pattern
   justified, since strides are nonnegative).

       @param [in] src  The source pattern.  Must be valid.
       @param [out] dest  The destination pattern.  Will share
                      num_axes and dims with src, but the strides
                      will be normalized, the pattern will be compact
                      (no gaps between memory-indexes) and offset will be 0.
 */
void MakeCompactNormalizedAndJustified(const Pattern &src,
                                       Pattern *dest);


/**
   This function copies the Pattern 'src' from 'dest', preserving the
   num_axes and dims while possibly modifying the strides and offset.  The
   strides of 'dest' will be nonnegative but the ordering from least to greatest
   of the nonzero strides will be the same as the ordering of the absolute
   values of the strides in 'src'.  The output pattern will be compact (no gaps)
   and justified (meaning offset == 0, since the strides will be nonnegative).

       @param [in] src  The source pattern.  Must be valid.
       @param [out] dest  The destination pattern.  Will share
                  num_axes and dims with src, but the strides and
                  offset may be different.
*/
void MakeCompactNonnegativeAndJustified(const Pattern &src,
                                        Pattern *dest);




/**
   Class PatternRebaser is an object that converts Pattern
   when memory layouts change.  The main use-case is when a base Variable
   (c.f. variable.h for definition) has a Pattern that is not
   contiguous (see pattern.h for definition of 'contiguous'), and
   its gradient Tensor is allocated contiguously.  This class is
   needed to convert patterns for Variables into patterns for their
   corresponding gradients.

   We make it an object rather than a function in order to avoid repetition when
   multiple patterns need to be rebased.
 */
class PatternRebaser {

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
  PatternRebaser(const Pattern &src_pattern,
                       const Pattern &dest_pattern);


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
  bool Rebase(Pattern *pattern);

  private:

  // TODO: remove src_pattern_ and dest_pattern_ once everything
  // is debugged.  They are copies of the src_pattern and dest_pattern
  // passed to the constructor.
  Pattern src_pattern_;
  Pattern dest_pattern_;

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

/**
   This object is to be instantiated when you want to know what permutation
   you'd get if you were to change the ordering of axes so that the abs(stride)
   were strictly increasing.  (Note: this is not a total order if there are >1
   axes with stride=0, so the ordering may be somewhat arbitrary).

   See the documentation for its GetIndex() function.
 */
class OutOfPlaceAxisSorter {
 public:
  // Constructor.
  inline OutOfPlaceAxisSorter(const Pattern &src) {
    int32 num_axes = src.num_axes;
    for (int32 raxis = 0; raxis < num_axes; raxis++)
      orig_raxis_[raxis] = raxis;
    std::sort(orig_raxis_, orig_raxis_ + num_axes,
              // a comparator (less-than) operator implemented as a lambda is
              // below.  Sort from least to greatest abs(stride), disambiguating
              // based on dim.
              [src] (int32 raxis1, int32 raxis2) {
                int32 abs_stride1 = std::abs(src.strides[raxis1]),
                    abs_stride1 =  std::abs(src.strides[raxis2]);
                if (abs_stride1 < abs_stride2) return true;
                else if (abs_stride1 > abs_stride2) return false;
                else return (src.dims[raxis1] < src.dims[raxis2]);
              });
  }
  // Returns the 'source' raxis-index for a particular destination
  // raxis-index, e.g..:  `src_raxis = GetIndex(dest_raxis)`.
  // Copying as e.g. `dest.strides[dest_raxis] = src.strides[src_raxis]`,
  // and the same for the dims, would give you a `dest` with axes
  // sorted from smallest to greatest absolute value.
  inline int32 GetIndex(int32 raxis) { return orig_raxis_[raxis]; }

 private:
  int32 orig_raxis_[KALDI_TENSOR_MAX_DIM];
};



/**
   This function sorts the axes in 'patterns' (which must be a valid
   pattern-tuple, see pattern.h for explanation) using TupleAxisComparator.  See
   its documentation in pattern-tuple-utils.cc for description of what this
   order is.

     @param [in,out]  The patterns whose axes are to be sorted.  The same
                     permutation will be applied to all the patterns.
 */
void SortTupleAxes(ArrayRef<Pattern*> patterns);

/**
   Compresses a Pattern-tuple by removing or combining as many axes as possible.
   See the documentation for CompressOnePattern() in pattern-utils.h basic
   concept of compressing a single Pattern to a pattern with possibly fewer axes
   (and maybe with negative strides converted to positive), which covers the
   same set of memory locations as the original Tensor.

   The difference with just calling CompressOnePattern() several times is
   that CompressPatterns() preserves the relationships between the tensors.
   In the language developed in pattern.h, this means the memory-index-tuple-set
   is preserved.

   Note: while the first Pattern will have no negative strides at output,
   the others may.

     @param [in,out] patterns   An nonempty array of the patterns
                         to be jointly compressed.

      @return  Returns true if it made any change to the patterns,
               false if they were unchanged.

 Examples are below, where we write a Pattern as
 `{{dim1,dim2,..}, {stride1,stride2,..}}`.

\verbatim
    src1                src2              dest1,offset1       dest2,offset2
  {{10},{1}}           {{10},{1}}        {{10},{1}},0        {{10},{1}},0  # no-op
  {{8},{1}}            {{1},{0}}         {{8},{1}},0         {{1},{0}},0   # no-op
  {{7},{-1}}           {{7},{1}}         {{7},{1}},-6         {{7},{-1}},6 # flip sign
 {{3,4},{4,1}}        {{3,4},{4,1}}      {{12},{1}},0         {{12},{1}},0 # combine dims
 {{3,4},{4,1}}        {{3,1},{4,0}}      {{3,4},{4,1}}        {{3,1},{4,0}} # can't combine, would be incompatible
 {{3,4},{4,1}}        {{1,1},{0,0}}      {{12},{1}}           {{1},{0}}    # combine
\endverbatim

   See also SortTupleAxes() and NormalizePatternTuple().
 */
bool CompressPatternTuple(ArrayRef<Pattern*> patterns);


/**
   Reduces a pattern-tuple to a normalized form.  (Caution: this may not be 100%
   deterministic, i.e. there may be two pattern-tuples in normalized form,
   i.e. the form produced by this function, which share the same
   memory-index-tuple-set but are not equal).

   This just calls CompressPatternTuple() and then SortPatternTupleAxes().

     @param [in,out] patterns.

*/
inline bool NormalizePatternTuple(ArrayRef<Pattern*> patterns) {
  CompressPatternTuple(patterns);
  NormalizePatternTupleAxes(patterns);
}




}  // namespace tensor
}  // namespace kaldi

// Include implementation of inline functions.
#include "tensor/pattern-tuple-utils-inl.h"

#endif  // KALDI_TENSOR_TENSOR_PATTERN_TUPLE_UTILS_H_
