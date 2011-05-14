// partition.h

// Copyright 2010  Microsoft Corporation

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
//
// This is a modified file from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: johans@google.com (Johan Schalkwyk)
//
// \file Functions and classes to create a partition of states
//

#ifndef FST_LIB_PARTITION_H__
#define FST_LIB_PARTITION_H__

#include <vector>
using std::vector;
#include <algorithm>

#include <fst/queue.h>


namespace fst {

template <typename T> class PartitionIterator;

// \class Partition
// \brief Defines a partitioning of states. Typically used to represent
//        equivalence classes for Fst operations like minimization.
//
template <typename T>
class Partition {
  friend class PartitionIterator<T>;

  struct Element {
    Element() : value(0), next(0), prev(0) {}
    Element(T v) : value(v), next(0), prev(0) {}

   T        value;
   Element* next;
   Element* prev;
  };

 public:
  Partition(bool allow_repeated_split):
      allow_repeated_split_(allow_repeated_split) {}

  Partition(bool allow_repeated_split, T num_states):
      allow_repeated_split_(allow_repeated_split) {
    Initialize(num_states);
  }

  ~Partition() {
    for (size_t i = 0; i < elements_.size(); ++i)
      delete elements_[i];
  }

  // Create an empty partition for num_states. At initialization time
  // all elements are not assigned to a class (i.e class_index = -1).
  // Initialize just creates num_states of elements. All element
  // operations are then done by simply disconnecting the element from
  // it current class and placing it at the head of the next class.
  void Initialize(size_t num_states) {
    for (size_t i = 0; i < elements_.size(); ++i)
      delete elements_[i];
    elements_.clear();
    classes_.clear();
    class_index_.clear();

    elements_.resize(num_states);
    class_index_.resize(num_states, -1);
    class_size_.reserve(num_states);
    for (size_t i = 0; i < num_states; ++i)
      elements_[i] = new Element(i);
    num_states_ = num_states;
  }

  // Add a class, resize classes_ and class_size_ resource by 1.
  size_t AddClass() {
    size_t num_classes = classes_.size();
    classes_.resize(num_classes + 1, 0);
    class_size_.resize(num_classes + 1, 0);
    class_split_.resize(num_classes + 1, 0);
    split_size_.resize(num_classes + 1, 0);
    return num_classes;
  }

  void AllocateClasses(T num_classes) {
    size_t n = classes_.size() + num_classes;
    classes_.resize(n, 0);
    class_size_.resize(n, 0);
    class_split_.resize(n, 0);
    split_size_.resize(n, 0);
  }

  // Add element_id to class_id. The Add method is used to initialize
  // partition. Once elements have been added to a class, you need to
  // use the Move() method move an element from once class to another.
  void Add(T element_id, T class_id) {
    Element* element = elements_[element_id];

    if (classes_[class_id])
      classes_[class_id]->prev = element;
    element->next = classes_[class_id];
    element->prev = 0;
    classes_[class_id] = element;

    class_index_[element_id] = class_id;
    class_size_[class_id]++;
  }

  // Move and element_id to class_id. Disconnects (removes) element
  // from it current class and
  void Move(T element_id, T class_id) {
    T old_class_id = class_index_[element_id];

    Element* element = elements_[element_id];
    if (element->next) element->next->prev = element->prev;
    if (element->prev) element->prev->next = element->next;
    else               classes_[old_class_id] = element->next;

    Add(element_id, class_id);
    class_size_[old_class_id]--;
  }

  // split class on the element_id
  void SplitOn(T element_id) {
    T class_id = class_index_[element_id];
    if (class_size_[class_id] == 1) return;

    // first time class is split
    if (split_size_[class_id] == 0) { 
      visited_classes_.push_back(class_id);
      class_split_[class_id] = classes_[class_id];
    }
    // increment size of split (set of element at head of chain)
    split_size_[class_id]++;
    
    // update split point
    if (class_split_[class_id] != 0
        && class_split_[class_id] == elements_[element_id])
      class_split_[class_id] = elements_[element_id]->next;

    // move to head of chain in same class
    Move(element_id, class_id);
  }

  // Finalize class_id, split if required, and update class_splits,
  // class indices of the newly created class. Returns the new_class id
  // or -1 if no new class was created.
  T SplitRefine(T class_id) {

    Element* split_el = class_split_[class_id];
    // only split if necessary
    //if (class_size_[class_id] == split_size_[class_id]) {
    if(split_el == NULL) { // we split on everything...
      split_size_[class_id] = 0;
      return -1;
    } else {
      T new_class = AddClass();

      if(allow_repeated_split_) { // split_size_ is possibly
        // inaccurate, so work it out exactly.
        size_t split_count;  Element *e;
        for(split_count=0,e=classes_[class_id];
            e != split_el; split_count++, e=e->next);
        split_size_[class_id] = split_count;
      }
      size_t remainder = class_size_[class_id] - split_size_[class_id];
      if (remainder < split_size_[class_id]) {  // add smaller
        classes_[new_class] = split_el;
        split_el->prev->next = 0;
        split_el->prev = 0;
        class_size_[class_id] = split_size_[class_id];
        class_size_[new_class] = remainder;
      } else {
        classes_[new_class] = classes_[class_id];
        class_size_[class_id] = remainder;
        class_size_[new_class] = split_size_[class_id];
        split_el->prev->next = 0;
        split_el->prev = 0;
        classes_[class_id] = split_el;
      }

      // update class index for element in new class
      for (Element* el = classes_[new_class]; el; el = el->next)
        class_index_[el->value] = new_class;

      class_split_[class_id] = 0;
      split_size_[class_id] = 0;

      return new_class;
    }
  }

  // Once all states have been processed for a particular class C, we
  // can finalize the split. FinalizeSplit() will update each block in the
  // partition, create new once and update the queue of active classes
  // that require further refinement.
  template <class Queue>
  void FinalizeSplit(Queue* L) {
    for (size_t i = 0; i < visited_classes_.size(); ++i) {
      T new_class = SplitRefine(visited_classes_[i]);
      if (new_class != -1 && L)
        L->Enqueue(new_class);
    }
    visited_classes_.clear();
  }


  const T class_id(T element_id) const {
    return class_index_[element_id];
  }

  const vector<T>& class_sizes() const {
    return class_size_;
  }

  const size_t class_size(T class_id)  const {
    return class_size_[class_id];
  }

  const T num_classes() const {
    return classes_.size();
  }


 private:
  int num_states_;

  // container of all elements (owner of ptrs)
  vector<Element*> elements_;

  // linked list of elements belonging to class
  vector<Element*> classes_;

  // pointer to split point for each class
  vector<Element*> class_split_;

  // class index of element
  vector<T> class_index_;

  // class sizes
  vector<T> class_size_;

  // size of split for each class
  // in the nondeterministic case, split_size_ is actually an upper
  // bound on the size of split for each class.
  vector<T> split_size_;

  // set of visited classes to be used in split refine
  vector<T> visited_classes_;

  // true if input fst was deterministic: we can make
  // certain assumptions in this case that speed up the algorithm.
  bool allow_repeated_split_;
};


// iterate over members of a class in a partition
template <typename T>
class PartitionIterator {
  typedef typename Partition<T>::Element Element;
 public:
  PartitionIterator(const Partition<T>& partition, T class_id)
      : p_(partition),
        element_(p_.classes_[class_id]),
        class_id_(class_id) {}

  bool Done() {
    return (element_ == 0);
  }

  const T Value() {
    return (element_->value);
  }

  void Next() {
    element_ = element_->next;
  }

  void Reset() {
    element_ = p_.classes_[class_id_];
  }

 private:
  const Partition<T>& p_;

  const Element* element_;

  T class_id_;
};
}

#endif  // FST_LIB_PARTITION_H__
