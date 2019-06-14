// tensor/memory-checker.h

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

#ifndef KALDI_TENSOR_MEMORY_CHECKER_H_
#define KALDI_TENSOR_MEMORY_CHECKER_H_ 1

#include <functional>
#include "tensor/tensor-common.h"
#include "tensor/pattern.h"


namespace kaldi {
namespace tensor {



/**
   class ChangeTracker is something we only use in 'debug mode'.  Its purpose is
   to keep track of when data was last changed, to make sure people don't mutate
   data via in-place operations in a way that will invalidate the backprop.
   This is a replacement for the 'version numbering' of Variables used in
   PyTorch, i.e. it's a different way of solving the same problem.  The
   mechanism is (I think) more exact than version numbering, and less hassle for
   the calling code; but since it's slower, we will only activate it
   occasionally.  c.f. SetDebugMode(), GetDebugMode().

   During the forward pass, when an Op records, as members, certain Tensors that
   will be needed during the backprop pass, it also records the time in
   ticks (c.f. GetTick()) at which the forward pass happened.  Then in
   the backward pass, during debug mode we want to check that the memory
   underlying those Tensors has not beeen changed since that recorded tick.
   (If people use in-place operations in an unsupported way, this might
   have happened).

   This class provides a mechanism to do that.  It's actually quite an
   interesting mathematical problem as it involves detecting overlap between
   patterns in memory and we want to do it efficiently, not using a huge array.
   Note: in debug mode, any time a memory region underlying a tracked Variable's
   data is written to (whether or not that write actually went through a tracked
   Variable or even a regular Tensor), we record the change (see member function
   RecordChange()).
*/
class ChangeTracker {
 public:

  /** Constructor.  Note: a Storage object is created for each allocated block
      of memory, and each Storage object has at most one ChangeTracker object.

      @param [in] num_bytes  The number of bytes allocated in this block.
                           Only needed for checking, to make sure that
                           the patterns do not overstep this bound.
   */
  ChangeTracker(size_t num_bytes);


  /**
     Record a write to this storage region at the current time (obtained by
     GetTick()).  Just appends it to the vector of writes after canonicalizing
     the pattern.  Inlined since it's only called from Storage::WrittenSince().

     @param [in] element_size  The size in bytes of the data type being stored
                             here: for example, 4 for float.
     @param [in] pattern    The pattern being changed.  It will be reduced
                            to canonical form (c.f. CanonicalizePattern())
                            before being stored.
   */
  inline void RecordWrite(int32 element_size,
                          const Pattern &pattern);


  /**
     Returns true if any element covered by this pattern has been
     changed since the time given by 'tick'.  Inlined since it's only
     called from Storage::WrittenSince().

      @param [in] element_size  The size in bytes of the data type being stored
                       here: for example, 4 for float.
      @param [in] pattern  The pattern that we are checking
      @param [in] tick  The time (obtained by GetTick()) since when
                       we want to know about changes

   */
  inline bool WrittenSince(int32 element_size,
                           const Pattern &pattern,
                           int64 tick);

 private:

  // number of bytes in this storage region (or possibly just a very big number,
  // if the size of the region was not known).
  int64 num_bytes_;

  // The size of elements in this storage region (e.g. 4 for float).  If for
  // some region the same region was accessed with multiple different element
  // sizes, this will be their lowest common denominator and all patterns
  // will have their strides and offsets scaled appropriately.
  // (We don't just store patterns in terms of bytes because we don't want
  // to increase the risk of overflowing int32 storage).
  int32 element_size_;


  struct WriteRecord {
    Pattern pattern;  // The pattern (offset, dims, strides) that was
                            // changed within this storage region.  This pattern
                            // will have been reduced to canonical form.  View
                            // it as a memory-index-set (c.f. glossary in
                            // pattern.h).

    int64 tick;             // The time, in ticks (c.f. NextTick()) at which
                            // this set of memory-indexes was changed.

    // Next in a singly linked list of WriteRecord.
    std::unique_ptr<WriteRecord> tail;
  };


  // Head of a singly linked list of changes.  When RecordChange() is called, we
  // will add to the head of this (and then de-dupe; see doc for change_map)).
  // When WrittenSince() is called, we will traverse it element by element until
  // we get to the tick passed to WrittenSince, and if there is any overlap with
  // the passed-in pattern, we'll return true.
  std::unique_ptr<WriteRecord> changes_;


  // This is a map from a pointer to the Pattern in WriteRecord::pattern
  // (hashing the pattern itself, not the pointer value), to the WriteRecord
  // that holds it.  We actually map to the address of the std::unique_ptr
  // pointing to that WriteRecord, which might be the address of this->changes_
  // or WriteRecord::tail, because we need to be able to write to that to
  // remove a WriteRecord from the singly linked list.  This map is used
  // in de-duping the list of changes, so that if someone provides the
  // exact same pattern twice, we only keep the most recent tick; this
  // keeps memory usage under control.
  std::unordered_map<Pattern*, std::unique_ptr<WriteRecord>*,
                     PatternPtrHasher, PatternPtrEqual> change_map_;
};



// This class is a common base-class for UninitializedDataChecker and
// InvalidatedDataChecker.
class DataCheckerBase {
 protected:
  DataCheckerBase(int64 num_bytes);

  /**
     This function records an event (i.e. that this memory area is being written to,
     or is now no longer valid, depending on the child class).
     It may insert something into map_, if an event with this pattern hasn't
     been recorded before.

       @param [in] element_size  The size, in bytes, of the element that this
                          array contains (e.g. 4 for float, 8 for double)
                          Currently expected to be the same for all invocations
                          (we can later extend this code to handle changes).
       @param [in] pattern   The pattern which we are recording as an event
                          (e.g. saying that its memory-index-set has been
                          written to, or has been invalidated.  Its memory-index-set
                          must be within [0, k-1] where k = num_bytes_ / element_size.
   */
a  void RecordEvent(int32 element_size,
                   const Pattern &pattern);

  /**
     This function is intended to return true if the memory-index-set of
     the provided Pattern is fully covered by the Patterns passed to
     previous invocations of RecordEvent.

     Because it sometimes (for efficiency) uses a randomized algorithm,
     it may not always detect less-than-complete coverage.  That is, there
     may be situations where `pattern` is not fully covered and it returns
     true anyway; but if it returns false, then `pattern` is definitely
     not covered by all the patterns passed to RecordEvent().

     The algorithm is:

       - If we can find a pattern identical to `pattern` in
         `map_`, return true (this is a common special case).
       - If `map_` contains exactly one pattern:
         See whether the the memory-index-set of `pattern` is
         a subset of the memory-index-set of that one pattern,
         and return true if so; else false.
       - Otherwise: choose a number of random memory-indexes from
         `pattern`, and for each one, see whether they are covered
         by any of the stored patterns.  If any such memory-index
         is not so covered, return false; else return true.  (Note:
         this last `true` may be inaccurate, meaning we fail to
         detect a problem we should have detected.)

      @param [in] element_size  The size, in bytes, of the element that this
                      array contains (e.g. 4 for float, 8 for double).
                      Currently required to be the same as the element_size
                      provided to any invocations of RecordEvent(); we may
                      relax that assumption in future.
      @param [in] pattern   The pattern we are checking. Its memory-index-set
                     must be within [0, k-1] where k = num_bytes_ / element_size.
      @return   True if `pattern` was fully covered by patterns recorded in
                RecordEvent() or if our randomized algorithm failed to detect
                the less-than-complete coverage.  False otherwise.
   */
  bool FullyCovered(int32 element_size,
                    const Pattern &pattern);

  /**
     This function is intended to return true if the memory-index-set of
     `pattern` has nonempty intersection with the memory-index-set of at least
     ones of the Patterns provided to RecordEvent().

     Because it is a randomized algorithm, it may sometimes return false
     when an exact version would have returned true, but not vice versa.

     The algorithm is:

       - If we can find a pattern identical to `pattern` in `map_`, return true
         (this is a common special case).
       - Otherwise:
          - For some or all of the Patterns provided to `RecordEvent()`:
            - If `pattern` has nonempty intersection with that pattern:
               return true
          - return false
   */
  bool PartlyCovered(int32 element_size,
                     const Pattern &pattern);

 private:

  // number of bytes in this storage region (or possibly just a very big number,
  // if the size of the region was not known).
  int64 num_bytes_;

  // The size of elements in this storage region (e.g. 4 for float).  If for
  // some region the same region was accessed with multiple different element
  // sizes, this will be their lowest common denominator and all patterns
  // will have their strides and offsets scaled appropriately.
  // (We don't just store patterns in terms of bytes because we don't want
  // to increase the risk of overflowing int32 storage).
  int32 element_size_;


  // `map` can actually be thought of as a set of Patterns, but it's
  // actually stored as a map from Pattern* to the std::unique_ptr holding
  // that same Pattern.  This may seem an odd thing to do; it's just
  // a convenient way to manage the memory.  Thanks to PatternPtrHasher,
  // we can avoid storing duplicate records for the same Pattern.
  std::unordered_map<Pattern*, std::unique_ptr<Pattern*>,
                     PatternPtrHasher, PatternPtrEqual> map_;


  // This is another way of storing the Patterns that have been recorded,
  // ordered by NumElements(); this enables us to check the larger patterns
  // first, which may be more efficient.
  std::multimap<int64, Pattern*> by_size_;
};

/**
   The purpose of this class is to check for use of uninitialized data.  It will
   only be used when debug mode is enabled.

   There are situations when initializing the memory of a Tensor/Variable (say,
   to zero) would be wasteful because we know that we're going to eventually
   write to all of it.  But doing this is risky because we might end up using
   values in uninitialized memory if we're not careful.  This class detects that
   situation, but only we are in debug mode; see SetDebugMode(), GetDebugMode().
 */
class UninitializedDataChecker: public DataCheckerBase {
 public:

  /** Constructor.  Note: a Storage object is created for each allocated block
      of memory, and each Storage object has at most one
      UninitializedDataChecker object.

      @param [in] num_bytes  The number of bytes allocated in this block.
                          Only needed for checking, to make sure that
                          the patterns do not overstep this bound.
   */
  UninitializedDataChecker(size_t num_bytes):
      DataCheckerBase(num_bytes),
      disabled_(false) { }


  /**
     This function records that this memory area is being written to.

        @param [in] element_size  The size of the element stored in the
                  Tensor, e.g. 4 for float, 8 for double.
        @param [in] pattern  The pattern which is being written; this
                  function records the write.
   */
  inline void RecordWrite(int32 element_size,
                          const Pattern &pattern) {
    RecordEvent(element_size, pattern);
  }

  /**
     This function checks that this memory area is currently uninitialized;
     if any part of it was previously initialized, it will crash.

        @param [in] element_size  The size of the element stored in the
                  Tensor, e.g. 4 for float, 8 for double.
        @param [in] pattern  The pattern which we are checking
   */
  inline void CheckUninitialized(int32 element_size,
                                 const Pattern &pattern);


  /**
     This function is called when this memory area is being read from.
     It will (usually) crash if an element of this memory area has not been
     written to.  The algorithm is randomized so a problem won't be
     detected in all cases.
        @param [in] element_size  The size of the element stored in the
                  Tensor, e.g. 4 for float, 8 for double.
        @param [in] pattern  The pattern which is being read.  If it
                  is not fully covered by the Patterns passed to
                  RecordWrite, this call will (usually) crash.
   */
  void RecordRead(int32 element_size,
                  const Pattern &pattern);
};


/**
   The purpose of this class is to check for use of invalidated data.  It will
   only be used when debug mode is enabled.

   This is a checking mechanism that helps us to fairly safely avoid certain
   unnecessary operations on parts of Variables in the backprop phase.  (If the
   check fails, user-level code will have to be changed).  It's best illustrated
   with an example.  Let A and B both be Variables representing 2x2 matrices
   that have been freshly created with uninitialized data.  Suppose we do:

      (1) Initialize A's data to something requiring derivative tracking
      (2) Copy A to B
      (3) Copy A to B again
      (4) Do something that depends on the value of B

   In the backprop, when doing the backprop of operation (3), after propagating
   the derivative back to A we'd need to zero out the first row of B's
   derivative matrix, to reflect the fact that its value before operation (3)
   doesn't affect the outcome; otherwise after the backprop of (2) we would have
   twice the value we should really have for the derivative w.r.t. A.  So
   naively, any time we do the backprop for an operation that writes to a
   variable that was already tracked at the time we did that operation, we would
   have to zero out that part of the derivative matrix afterwards.  But much of
   the time we wouldn't have previously written to that part of memory, so such
   zeroing would be wasteful.  (Note: we can't just rely on checking whether or
   not this base Variable has previously had an operation done on it; the hard
   case is where there are multiple Variables that are sub-parts of the same
   base Variable).

   The way we handle this is: we assume by default that any time we do an
   operation that sets a Variable but does not depend on its previously existing
   value, the memory underlying it has not been previously written to in an
   operation that required derivative-tracking.  That is, the framework assumes
   by default that you DO NOT REUSE MEMORY, except for in-place operations.  If
   you do want to re-use memory (specifically:a if you do something that does
   require overwriting previously-written data that required derivative
   tracking, like the above), you can inform the framework that you plan to do
   this as follows:
     DoSomethingWith(a, b, &c.Overwrite());
   instead of
     DoSomethingWith(a, b, &c);
   (here a, b and c are Variables; and let's suppose this operation
   DoSomethingWith() ignores the previous value of `c`).

   This purpose of class InvalidatedDataChecker is to detect cases where someone
   should have invoked Overwrite() because tracked data was overwritten, but
   failed to do so.

   See also the comment for the overwrite_ member of class VariableImpl, and
   the Untouched() member of Variable.
 */
class InvalidatedDataChecker: public DataCheckerBase {
 public:

  /** Constructor.  Note: a Storage object is created for each allocated block
      of memory, and each Storage object has at most one InvalidatedDataChecker
      object.

      @param [in] num_bytes  The number of bytes allocated in this block.
                         Only needed for checking, to make sure that
                         the patterns do not overstep this bound.
   */
  InvalidatedDataChecker(size_t num_bytes):
      DataCheckerBase(num_bytes) { }


  /**
     This function records that this memory area is being invalidated Normally
     this object will be attached to the Tensor for a derivative, and will be
     called when we do the backprop for an Op that should ideally have zeroed
     out this part of the matrix, but we didn't do that because we believe this
     memory region won't be read from in future.
   */
  inline void RecordInvalidation(int32 element_size,
                                 const Pattern &pattern) {
    RecordEvent(element_size, pattern);
  }


  /**
     This function is called when this memory area is being read from.  It will
     (usually, since the algorithm is randomized) crash if `pattern` has
     nonempty overlap with a pattern passed to RecordInvalidation().

        @param [in] element_size  The size of the element stored in the
                  Tensor, e.g. 4 for float, 8 for double.
        @param [in] pattern  The pattern which is being read.  If it
                  overlaps with an invalidated Pattern, this will
                  (usually) crash.
  */
  void RecordRead(int32 element_size,
                  const Pattern &pattern);
};


class MemoryChecker {
 public:

  /**
     Constructor: constructs a MemoryChecker object for a storage region

        @param [in] num_bytes   Number of bytes in the storage region
        @param [in] new_region  True if this object is being allocated at
                     the same time as we are allocating this region.
                     (may be false if debug mode was not active when
                     the region was first allocated).
  */
  MemoryChecker(int64 num_bytes,
                bool new_region): num_bytes_(num_bytes) {
    Initialize(new_region);
  }

  /**
     This is called by functions that implement low-level functions on tensors,
     before or after actually accessing the memory.  The options are:
         kRead
         kReadWrite
         kWrite
         kCheckUninitialized
         kReadAndInvalidate
         kInvalidate
     From a user's perspective the only thing this function might do is crash--
     which it is designed to do if it detects various "disallowed" things.
  */
  void RecordUse(int32 element_size,
                 const Pattern &pattern,
                 TensorUseEnum use_type) {
    KALDI_PARANOID_ASSERT(DebugMode());
    if (debug_tick_ != DebugTick())
        Initialise(false);  // false means: not a new region.

    if (use_type == kInitialize || use_type == kCheckUninitialized) {
      if (uninitialized_checker_)
        uninitialized_checker_->CheckUninitialized(element_size, pattern);
    }
    if (use_type == kRead || use_type == kReadWrite ||
        use_type == kReadInvalidate) {
      invalidated_checker_->RecordRead(element_size, pattern);
      if (uninitialized_checker_)
        uninitialized_checker_->RecordRead(element_size, pattern);
    }
    if (use_type == kWrite || use_type == kReadWrite ||
        use_type == kInitialize) {
      // Important that this happens after checking the reads above.
      // uninitialized_checker_ would never find an error in RecordRead() if it
      // was done after the RecordWrite().
      if (uninitialized_checker_)
        uninitialized_checker_->RecordWrite(element_size, pattern);
      change_tracker_->RecordWrite(element_size,  pattern);
    }
    if (use_type == kInvalidate || use_type == kReadInvalidate) {
      RecordInvalidation(element_size, pattern);
    }
  }

  /**
     Record the invalidation of data.  This occurs in certain backprop
     operations as a way to avoid unnecessary zeroing operations.  See
     the documentation for class InvalidatedDataChecker for a longer
     explanation.
   */
  void RecordInvalidation(int32 element_size,
                          const Pattern &pattern) {
    if (!invalidated_checker_)
      invalidated_checker_ = new InvalidatedDataChecker(num_bytes_);
    invalidated_checker_->RecordInvalidation(element_size, pattern);
  }

  /**
     Record that the entire storage region is being zeroed.  (This avoids the
     need to use uninitialized_checker_, so we delete it if it was set).
   */
  inline void RecordZeroing() { uninitialized_checker_ = NULL; }


  /**
     This function is called by the backprop code in Ops when it wants to
     make sure that certain data stored from the forward pass has not
     been written to since the specified tick.
   */
  void CheckUnchangedSince(
      int32 element_size,
      const Pattern &pattern,
      int64 tick) {
    if (change_tracker_ &&
        change_tracker_->WrittenSince(element_size, pattern, tick)) {
      KALDI_ERR << "Quantity needed during backprop has changed since "
          "the value used in the forward pass.  You have likely used "
          "an in-place or overwriting operation in a way that's not "
          "allowed.  Solution: don't overwrite data if you want "
          "to do backprop.";
    }
  }

 private:
  /**
     Initialize all members of this object except for num_bytes_ (which is set
     in the constructor).  This is called from the constructor, but also whenever
     we detect that debug mode has been turned off and then on again.
   */
  void Initialize(bool new_region);

  // the number of bytes in the region, set only in the constructor.
  int64 num_bytes_;

  // debug_tick_ is the value of DebugTick() at the time when Initialize() was
  // most recently called.  I.e. it's the start of the current debug cycle.
  // It's used to detect when debug mode has been turned off and then on, which
  // requires us to re-initialize this object.
  int64 debug_tick_;

  // Checker object for uninitialized data.  This is only non-NULL if
  // the following two conditions hold:
  //   (a) `new_region` as passed to Initialize() was true (because if we
  //      started debugging after this region was already created, we
  //      wouldn't know whether any data in it was uninitialized, so
  //      this check is meaningless.
  //   (b) No-one has called RecordZeroing() since Initialize() was
  //      last called.  (This records that the entire region was
  //      zeroed, which means there would be no uninitialized data.
  std::unique_ptr<UninitializedDataChecker> uninitialized_checker_;

  // Checker object for invalidated data.  Will only be allocated if
  // RecordInvalidation() has been called since Initialize().  See docs for
  // InvalidatedDataChecker for explanation of what this means.
  std::unique_ptr<InvalidatedDataChecker> invalidated_checker_;

  // Checker object that checks that we don't overwrite quantities
  // that will be needed in the backward pass.
  std::unique_ptr<ChangeTracker> change_tracker_;


};



}  // namespace tensor
}  // namespace kaldi

#endif  // KALDI_TENSOR_MEMORY_CHECKER_H_
