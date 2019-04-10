// tensor/change-tracker.h

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

#ifndef KALDI_TENSOR_CHANGE_TRACKER_H_
#define KALDI_TENSOR_CHANGE_TRACKER_H_ 1

#include <functional>
#include "tensor/tensor-common.h"
#include "tensor/tensor-pattern.h"


namespace kaldi {
namespace tensor {



/**
   class ChangeTracker is something we only use in 'debug mode'.  Its purpose is
   to keep track of when data was last changed

   to make sure people don't mutate data via in-place operations in a way that
   will invalidate the backprop.  This is a replacement for the 'version
   numbering' of Variables used in PyTorch, i.e. it's a different way of solving
   the same problem.  The mechanism is (I think) more exact than version
   numbering, and less hassle for the calling code; but since it's slower, we
   will only activate it occasionally.  c.f. SetDebugMode(), GetDebugMode().

   When a computation requiring derivatives creates a graph that will (when
   Backprop()'d) require a certain Tensor's data to remain unchanged until
   the backprop is done, we put a lock on the relevant memory region.
   This is done by LockPattern().  Conceptually the locking is done at the
   byte level, but without explicitly creating a byte-level map; it's
   done by detecting overlap of Patterns and will be reasonably efficient
   unless the user is creating a large number of different views of the same
   memory region.

   The same piece of memory may be locked multiple times.  This is not a write
   lock, it is a lock that prevents modification of that memory location.
   Attempts to mutate that memory (assuming the code calls Mutate()) will cause a
   crash.  The solution would be to remove the offending in-place operation from
   your code.

 */
class ChangeTracker {
 public:

  // Increments the global counter and returns that value.
  static uint64 GetTick();


  /** Constructor.  A Storage object is created for each allocated block of
      memory, and each Storage object has at most one ChangeTracker object.

      @param [in] num_bytes  The number of bytes allocated in this block.
                           Only needed for checking, to make sure that
                           the patterns do not overstep this bound.
   */
  ChangeTracker(size_t num_bytes);


  /**
     Record a change to this storage region at the current time (obtained by
     GetTick()).  Just appends it to the vector of changes after canonicalizing
     the pattern.

     @param [in] element_size  The size in bytes of the data type being stored
                             here: for example, 4 for float.
     @param [in] pattern    The pattern being changed.  It will be reduced
                            to canonical form (c.f. CanonicalizePattern())
                            before being stored.
   */

  void RecordChange(int32 element_size,
                    const TensorPattern &pattern);


  /**
     Returns true if any element covered by this pattern has been
     changed since the time given by 'tick'.

      @param [in] tick  The time (obtained by GetTick()) since when
                     we want to know about changes
      @param [in] pattern  The pattern that we are checking
   */
  bool ChangedSince(int64 tick,
                    const TensorPattern &pattern);

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
  int64 element_size_;

  struct ChangeRecord {
    TensorPattern pattern;  // The pattern (offset, dims, strides) that was
                            // changed within this storage region.  This pattern
                            // will have been reduced to canonical form.
    int64 tick;             // The time, in ticks (c.f. tick_counter_) at which
                            // this set of elements was changed.
  };


  // changes_ is a sequences of changes.
  std::vector<ChangeRecord> changes_;

  static int64 tick_counter_{0};

};



}  // namespace tensor
}  // namespace kaldi

#endif  // KALDI_TENSOR_CHANGE_TRACKER_H_
