/*
 * IntervalHeap.h --
 *    Heap implementation with both min and max retrieval/removal functions
 *
 * Contributed by Dustin Hillard (hillard@ssli.ee.washington.edu)
 *
 * Copyright (c) 2010 SRI International.  All Rights Reserved.

 * @(#)$Header: /home/srilm/CVS/srilm/dstruct/src/IntervalHeap.h,v 1.5 2010/06/02 04:52:43 stolcke Exp $
 *
 *    Implementation started with the code from http://www.mhhe.com/engcs/compsci/sahni/enrich/c9/interval.pdf
 *      An online portion of the textbook "Data Structures, Algorithms, and Applications in C++" by Sartaj Sahni
 *
 *    Description from that text:
 *     An interval heap is an elegant extension of a min heap and a max heap that permits
 *     us to insert and delete elements in O(logn) time, where n is the number of
 *     elements in the double-ended priority queue.
 *
 *
 *    Changes made to example code:
 *       (1) add empty() test function
 *       (2) change function names -- Ex: Insert() -> push(), DeleteMin() -> pop_min()
 *       (3) remove 'throw' function that depended on another header, replace with assert()
 *       (4) change pop functions to not save the value popped (it should be looked at and saved with the top functions)
 *       (5) change push and pop to not return '*this'
 *       (6) add arguments to template to allow for custom sorting (less than, greater than, and equal to functions), 
 *            replace '<' in functions with 'less', and '>' with 'greater' ( || with 'equal' for '<=' and '>=' )
 *       (7) fix bug in example code function pop_max() to have correct behavior when filling last node at end of resort
 *            (the fix is to not reinsert the bottom node max, because that is also the top node max -- which is supposed to be removed)
 *       (8) change to use Array template, rather than just doing a 'new[maxSize]'
 *       (9) change pop_min() to have the same end behavior as pop_max()
 */   


#ifndef _IntervalHeap_h_
#define _IntervalHeap_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <assert.h>

#include "Boolean.h"
#include "Array.cc"

template <class _Tp>
struct lessThan
{
  Boolean operator()(const _Tp& __x, const _Tp& __y) const { return __x < __y; }
};

template <class _Tp>
struct greaterThan
{
  Boolean operator()(const _Tp& __x, const _Tp& __y) const { return __x > __y; }
};

template <class _Tp>
struct equalTo
{
  Boolean operator()(const _Tp& __x, const _Tp& __y) const { return __x == __y; }
};

template<class T, class Less, class Greater, class Equal> class IntervalHeap;
							// forward declaration

template <class T>
class TwoElement {
  friend class IntervalHeap<T, lessThan<T>, greaterThan<T>, equalTo<T> >;
 public:
  T left;  // left element
  T right; // right element
};


template<class T, class Less, class Greater, class Equal>
class IntervalHeap {
 public:
  IntervalHeap(int IntervalHeapSize = 10);
  ~IntervalHeap() {}
  unsigned size() const {return CurrentSize;}
  int empty() const {return (CurrentSize == 0);}
  T top_min() {assert(CurrentSize != 0);
           //if (CurrentSize == 0)
           //   throw OutOfBounds();
           return heap[1].left;
           }
  T top_max() {assert(CurrentSize != 0);
           //if (CurrentSize == 0)
           //   throw OutOfBounds();
           return heap[1].right;
           }
  void push(const T& x);
  void pop_min();
  void pop_max();
 protected:
  Less    less;
  Greater greater;
  Equal   equal;
 private:
  int CurrentSize;             // number of elements in heap
  int MaxSize;                 // max elements permitted
  Array< TwoElement<T> > heap; // element array
};


template<class T>
inline void IntHeapSwap(T& a, T& b)
{// Swap a and b.
   T temp = a; a = b; b = temp;
}

#endif /* _IntervalHeap_h_ */
