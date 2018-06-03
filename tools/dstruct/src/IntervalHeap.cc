/*
 * IntervalHeap.cc --
 *	Heap implementation with both min and max retrieval/removal functions
 *
 * Contributed by Dustin Hillard (hillard@ssli.ee.washington.edu)
 *
 */

#ifndef _IntervalHeap_cc_
#define _IntervalHeap_cc_

#ifndef lint
static char IntervalHeap_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/IntervalHeap.cc,v 1.7 2012/05/17 06:46:36 stolcke Exp $";
#endif

#include "IntervalHeap.h"

template<class T, class Less, class Greater, class Equal>
IntervalHeap<T, Less, Greater, Equal>::IntervalHeap(int IntervalHeapSize)
  : CurrentSize(0),
    heap(0, IntervalHeapSize / 2 + IntervalHeapSize % 2)
{
}

template<class T, class Less, class Greater, class Equal>
void IntervalHeap<T, Less, Greater, Equal>::push(const T& x)
{ // Insert x into the interval heap.
  // handle CurrentSize < 2 as a special case
  
  if (CurrentSize < 2) {
    if (CurrentSize) // CurrentSize is 1
      if ( less(x, heap[1].left) )
	heap[1].left = x;
      else heap[1].right = x;
    else {// CurrentSize is 0
      heap[1].left = x;
      heap[1].right = x;
    }
    CurrentSize++;
    return;
  }
  // CurrentSize >= 2
  int LastNode = CurrentSize / 2 + CurrentSize % 2;
  Boolean minHeap; // true iff x is to be inserted in the min heap part of the interval heap
  if (CurrentSize % 2)
    // odd number of elements
    if ( less(x, heap[LastNode].left) )
      // x will be an interval left end
      minHeap = true;
    else minHeap = false;
  else {// even number of elements
    LastNode++;
    if ( less(x, heap[LastNode / 2].left) || equal(x, heap[LastNode / 2].left) )
      minHeap = true;
    else minHeap = false;
  }

  // make sure memory for lastNode gets allocated
  // if not, heap[LastNode].xxx = heap[LastNode/2].xxx may cause problem if LastNode triggers memory allocation
  T &dummy = heap[LastNode].left; 

  if (minHeap) {// fix min heap of interval heap 
    // find place for x 
    // i starts at LastNode and moves up tree
    int i = LastNode;
    
    while (i != 1 && (less(x, heap[i / 2].left) || equal(x, heap[i / 2].left)) ){
      // cannot put x in heap[i]
      // move left element down
      heap[i].left = heap[i / 2].left;
      i /= 2; // move to parent
    }
    heap[i].left = x;
    CurrentSize++;
    if (CurrentSize % 2)
      // new size is odd, put dummy in LastNode
      heap[LastNode].right = heap[LastNode].left;
  }
  else {// fix max heap of interval heap
    // find place for x
    // i starts at LastNode and moves up tree
    int i = LastNode;
    while (i != 1 && (greater(x, heap[i / 2].right) || equal(x, heap[i / 2].right)) ) {
      // cannot put x in heap[i]
      // move right element down
      heap[i].right = heap[i / 2].right;
      i /= 2; // move to parent
    }
    heap[i].right = x;
    CurrentSize++;
    if (CurrentSize % 2)
      // new size is odd, put dummy in LastNode
      heap[LastNode].left = heap[LastNode].right;
  }
  return;
}

template<class T, class Less, class Greater, class Equal>
void IntervalHeap<T, Less, Greater, Equal>::pop_min()
{ 
  // min element from interval heap.
  // check if interval heap is empty
  if (CurrentSize == 0)
    //throw OutOfBounds(); // empty
    assert(0);

  // restructure min heap part
  int LastNode = CurrentSize / 2 + CurrentSize % 2;
  T y; // element removed from last node
  if (CurrentSize % 2) {// size is odd
    y = heap[LastNode].left;
    LastNode--;
  }
  else {// size is even // change to y = .left in attempt to speed up luck in insertion
    y = heap[LastNode].left;
    heap[LastNode].left = heap[LastNode].right;
  }
  CurrentSize--;
  // find place for y starting at root
  int i = 1, // current node of heap
    ci = 2; // child of i
  while (ci <= LastNode) {// find place to put y
    // heap[ci].left should be smaller child of i
    if (ci < LastNode &&
	greater(heap[ci].left, heap[ci+1].left) ) ci++;
    // can we put y in heap[i]?
    if ( less(y, heap[ci].left) || equal(y, heap[ci].left) ) break; // yes

    // no
    heap[i].left = heap[ci].left; // move child up
    if (greater(y, heap[ci].right) )
      IntHeapSwap(y, heap[ci].right);
    i = ci; // move down a level
    ci *= 2;
  }
  // when CurrentSize is 1, we don't want to put y back on the heap, it was the element removed (and is equal to x) -- instead we leave heap[1] alone (which means that .left was copied to .right in the else statement above
  if(CurrentSize > 1)
    heap[i].left = y;
  /*if (i == LastNode && CurrentSize % 2)
    heap[LastNode].left = heap[LastNode].right;
    else heap[i].left = y;*/
  return;
}

template<class T, class Less, class Greater, class Equal>
void IntervalHeap<T, Less, Greater, Equal>::pop_max()
{
  // max element from interval heap.
  if (CurrentSize == 0)
    //throw OutOfBounds(); // empty
    assert(0);

  // restructure max heap part
  int LastNode = CurrentSize / 2 + CurrentSize % 2;
  T y; // element removed from last node
  if (CurrentSize % 2) {// size is odd
    y = heap[LastNode].left;
    LastNode--;
  }
  else {// size is even
    y = heap[LastNode].right;
    heap[LastNode].right = heap[LastNode].left;
  }
  CurrentSize--;
  // find place for y starting at root
  int i = 1, // current node of heap
    ci = 2; // child of i
  while (ci <= LastNode) {// find place to put y
    // heap[ci].right should be larger child of i
    if (ci < LastNode &&
	less(heap[ci].right, heap[ci+1].right) ) ci++;
    // can we put y in heap[i]?
    if ( greater(y, heap[ci].right) || equal(y, heap[ci].right) ) break; // yes
    // no
    heap[i].right = heap[ci].right; // move child up
    if ( less(y, heap[ci].left) )
      IntHeapSwap(y, heap[ci].left);
    i = ci; // move down a level
    ci *= 2;
  }
  // when CurrentSize is 1, we don't want to put y back on the heap, it was the element removed (and is equal to x) -- instead we leave heap[1] alone (which means that .left was copied to .right in the else statement above
  if(CurrentSize > 1)
    heap[i].right = y;
  return;
}

#endif /* _IntervalHeap_cc_ */

