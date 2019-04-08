// util/hash-list.h

// Copyright 2009-2011   Microsoft Corporation
//                2013   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_UTIL_HASH_LIST_H_
#define KALDI_UTIL_HASH_LIST_H_

#include <vector>
#include <set>
#include <algorithm>
#include <limits>
#include <cassert>
#include "util/stl-utils.h"


/* This header provides utilities for a structure that's used in a decoder (but
   is quite generic in nature so we implement and test it separately).
   Basically it's a singly-linked list, but implemented in such a way that we
   can quickly search for elements in the list.  We give it a slightly richer
   interface than just a hash and a list.  The idea is that we want to separate
   the hash part and the list part: basically, in the decoder, we want to have a
   single hash for the current frame and the next frame, because by the time we
   need to access the hash for the next frame we no longer need the hash for the
   previous frame.  So we have an operation that clears the hash but leaves the
   list structure intact.  We also control memory management inside this object,
   to avoid repeated new's/deletes.

   See hash-list-test.cc for an example of how to use this object.
*/
namespace kaldi {

template<class I, class T>
class HashList {
 public:
  struct Elem {
    I key;
    T val;
    Elem *tail;
  };

  /// Constructor takes no arguments.
  /// Call SetSize to inform it of the likely size.
  HashList() : list_head_(nullptr), bucket_list_tail_(boundary_tag_),
               hash_size_(0), freed_head_(nullptr) {}

  /// Clears the hash and gives the head of the current list to the user;
  /// ownership is transferred to the user (the user must call Delete()
  /// for each element in the list, at his/her leisure).
  Elem *Clear() noexcept {
    // Clears the hashtable and gives ownership of the currently contained list
    // to the user.
    for (size_t cur_bucket = bucket_list_tail_; cur_bucket != boundary_tag_;
         cur_bucket = buckets_[cur_bucket].prev_bucket) {
      buckets_[cur_bucket].last_elem = nullptr;  // this is how we indicate "empty".
    }

    bucket_list_tail_ = boundary_tag_;
    Elem *ans = list_head_;
    list_head_ = nullptr;
    return ans;
  }

  /// Gives the head of the current list to the user.  Ownership retained in the
  /// class.  Caution: in December 2013 the return type was changed to const
  /// Elem* and this function was made const.  You may need to change some types
  /// of local Elem* variables to const if this produces compilation errors.
  const Elem *GetList() const noexcept { return list_head_; }

  /// Think of this like delete().  It is to be called for each Elem in turn
  /// after you "obtained ownership" by doing Clear().  This is not the opposite
  /// of. Insert, it is the opposite of New.  It's really a memory operation.
  inline void Delete(Elem *e) noexcept {
    e->tail = freed_head_;
    freed_head_ = e;
  }

  /// This should probably not be needed to be called directly by the user.
  /// Think of it as opposite to Delete();
  inline Elem *New() noexcept {
    if (!freed_head_) {
      Elem *tmp = new Elem[allocate_block_size_];

      for (size_t i = 0; i + 1 < allocate_block_size_; ++i) {
        tmp[i].tail = tmp + i + 1;
      }

      tmp[allocate_block_size_ - 1].tail = nullptr;
      freed_head_ = tmp;
      allocated_.push_back(tmp);
    }

    Elem *ans = freed_head_;
    freed_head_ = freed_head_->tail;
    return ans;
  }

  /// Find tries to find this element in the current list using the hashtable.
  /// It returns NULL if not present.  The Elem it returns is not owned by the
  /// user, it is part of the internal list owned by this object, but the user
  /// is free to modify the "val" element.
  inline Elem *Find(I key) noexcept {
    const HashBucket &bucket = buckets_[static_cast<size_t>(key) % hash_size_];

    if (bucket.last_elem == nullptr) {
      return nullptr;  // empty bucket.
    } else {
      Elem *head = bucket.prev_bucket == boundary_tag_ ?
                   list_head_ : buckets_[bucket.prev_bucket].last_elem->tail;

      for (Elem *e = head; e != bucket.last_elem->tail; e = e->tail) {
        if (e->key == key) {
          return e;
        }
      }

      return nullptr;  // Not found.
    }
  }

  /// Insert inserts a new element into the hashtable/stored list.  By calling
  /// this,
  /// the user asserts that it is not already present (e.g. Find was called and
  /// returned NULL).  With current code, calling this if an element already
  ///  exists will result in duplicate elements in the structure, and Find()
  ///  will find the first one that was added.
  /// [but we don't guarantee this behavior].
  inline void Insert(I key, T val) noexcept {
    const size_t index = static_cast<size_t>(key) % hash_size_;
    HashBucket &bucket = buckets_[index];
    Elem *elem = New();
    elem->key = key;
    elem->val = val;

    if (bucket.last_elem == nullptr) {
      // Unoccupied bucket.  Insert at head of bucket list (which is tail of
      // regular list, they go in opposite directions).
      if (bucket_list_tail_ == boundary_tag_) {
        // list was empty so this is the first elem.
        KALDI_ASSERT(list_head_ == nullptr);
        list_head_ = elem;
      } else {
        // link in to the chain of Elems
        buckets_[bucket_list_tail_].last_elem->tail = elem;
      }

      elem->tail = nullptr;
      bucket.last_elem = elem;
      bucket.prev_bucket = bucket_list_tail_;
      bucket_list_tail_ = index;
    } else {
      // Already-occupied bucket.  Insert at tail of list of elements within
      // the bucket.
      elem->tail = bucket.last_elem->tail;
      bucket.last_elem->tail = elem;
      bucket.last_elem = elem;
    }
  }

  /// SetSize tells the object how many hash buckets to allocate (should
  /// typically be at least twice the number of objects we expect to go in the
  /// structure, for fastest performance).  It must be called while the hash
  /// is empty (e.g. after Clear() or after initializing the object, but before
  /// adding anything to the hash.
  void SetSize(size_t size) noexcept {
    hash_size_ = size;

    // make sure empty.
    KALDI_ASSERT(list_head_ == nullptr && bucket_list_tail_ == boundary_tag_);

    if (size > buckets_.size()) {
      buckets_.resize(size, HashBucket(0, nullptr));
    }
  }

  /// Returns current number of hash buckets.
  size_t Size() const noexcept { return hash_size_; }

  ~HashList() noexcept {
    // First test whether we had any memory leak within the
    // HashList, i.e. things for which the user did not call Delete().
    size_t num_in_list = 0, num_allocated = 0;

    for (Elem *e = freed_head_; e != nullptr; e = e->tail) {
      ++num_in_list;
    }

    for (size_t i = 0; i < allocated_.size(); i++) {
      num_allocated += allocate_block_size_;
      delete[] allocated_[i];
    }

    if (num_in_list != num_allocated) {
      KALDI_WARN << "Possible memory leak: " << num_in_list << " != "
                 << num_allocated << ": you might have forgotten to call Delete"
                 << " on some Elems";
    }
  }

 private:
  struct HashBucket {
    size_t prev_bucket;  // index to next bucket (-1 if list tail).  Note:
    // list of buckets goes in opposite direction to list of Elems.
    Elem *last_elem;  // pointer to last element in this bucket (NULL if empty)
    inline HashBucket(size_t i, Elem *e) : prev_bucket(i), last_elem(e) {}
  };

  Elem *list_head_;  // head of currently stored list.
  size_t bucket_list_tail_;  // tail of list of active hash buckets.

  size_t hash_size_;  // number of hash buckets.

  std::vector<HashBucket> buckets_;

  // head of list of currently freed elements. [ready for allocation]
  Elem *freed_head_;

  std::vector<Elem *> allocated_;  // list of allocated blocks.

  // Number of Elements to allocate in one block.  Must be largish so storing
  // allocated_ doesn't become a problem.
  static const size_t allocate_block_size_ = 1024;

  static const size_t boundary_tag_ = static_cast<size_t>(-1);
};
}  // end namespace kaldi
#endif  // KALDI_UTIL_HASH_LIST_H_
