/*
 * LatticeReduce.cc --
 *	Lattice compaction algorithms
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1997-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeReduce.cc,v 1.6 2014-08-29 21:35:47 frandsen Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "Lattice.h"

#include "SArray.cc"
#include "LHash.cc"
#include "Array.cc"

#define DebugPrintFatalMessages         0 

#define DebugPrintFunctionality         1 
// for large functionality listed in options of the program
#define DebugPrintOutLoop               2
// for out loop of the large functions or small functions
#define DebugPrintInnerLoop             3
// for inner loop of the large functions or outloop of small functions


/*
 * Compare two transition lists for equality
 */
Boolean
compareTransitions(const TRANS_T<NodeIndex,LatticeTransition> &transList1,
		   const TRANS_T<NodeIndex,LatticeTransition> &transList2)
{
    if (transList1.numEntries() != transList2.numEntries()) {
	return false;
    }

#ifdef USE_SARRAY
    // SArray already sorts indices internally
    TRANSITER_T<NodeIndex,LatticeTransition> transIter1(transList1);
    TRANSITER_T<NodeIndex,LatticeTransition> transIter2(transList2);

    NodeIndex node1, node2;
    while (transIter1.next(node1)) {
	if (!transIter2.next(node2) || node1 != node2) {
	    return false;
	}
    }
#else
    // assume random access is efficient
    TRANSITER_T<NodeIndex,LatticeTransition> transIter1(transList1);
    NodeIndex node1;
    while (transIter1.next(node1)) {
	if (!transList2.find(node1)) {
	    return false;
	}
    }
#endif

    return true;
}

/*
 * Merge two nodes 
 */
void
Lattice::mergeNodes(NodeIndex nodeIndex1, NodeIndex nodeIndex2,
			LatticeNode *node1, LatticeNode *node2, Boolean maxAdd)
{
    if (nodeIndex1 == nodeIndex2) {
	return;
    }

    if (node1 == 0) node1 = findNode(nodeIndex1);
    if (node2 == 0) node2 = findNode(nodeIndex2);

    assert(node1 != 0 && node2 != 0);

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::mergeNodes: "
	     << "      i.e., (" << getWord(node1->word)
	     << ", " << getWord(node2->word) << ")\n";
    }

    // add the incoming trans of nodeIndex2 to nodeIndex1
    TRANSITER_T<NodeIndex,LatticeTransition> 
      inTransIter2(node2->inTransitions);
    NodeIndex fromNodeIndex;
    while (LatticeTransition *trans = inTransIter2.next(fromNodeIndex)) {
      insertTrans(fromNodeIndex, nodeIndex1, *trans, maxAdd);
    }

    // add the outgoing trans of nodeIndex2 to nodeIndex1
    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIter2(node2->outTransitions);
    NodeIndex toNodeIndex;
    while (LatticeTransition *trans = outTransIter2.next(toNodeIndex)) {
      insertTrans(nodeIndex1, toNodeIndex, *trans, maxAdd);
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::mergeNodes: "
	     << "(" << nodeIndex2 << ") has been removed\n";
    }
    // delete this redudant node.
    removeNode(nodeIndex2); 
}

/*
 * Try merging successors of nodeIndex
 */
void
Lattice::packNodeF(NodeIndex nodeIndex, Boolean maxAdd)
{
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::packNodeF: "
	     << "processing (" << nodeIndex << ") ***********\n"; 
    }

    LatticeNode *node = findNode(nodeIndex); 
    // skip nodes that have already beend deleted
    if (!node) {
      return;
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::packNodeF: "
	     << "      i.e., (" << getWord(node->word) << ")\n";
    }

    unsigned numTransitions = node->outTransitions.numEntries();
    makeArray(NodeIndex, nodeList, numTransitions); 
    makeArray(VocabIndex, wordList, numTransitions);
    
    // going in a forward direction
    // collect all the out-nodes of nodeIndex, because we need to 
    // be able to delete transitions while iterating over them later
    // Also, store the words on each node to allow quick equality checks
    // without findNode() later.  Note we cannot save the findNode() result
    // itself since node objects may more around as a result of deletions.
    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIter(node->outTransitions);
    unsigned position = 0; 
    NodeIndex toNodeIndex;
    while (outTransIter.next(toNodeIndex)) {
	wordList[position] = findNode(toNodeIndex)->word;
	nodeList[position] = toNodeIndex; 
	position ++;
    }
    // For static code analysis, ensure any remainder of arrays
    // are initialized
    if (position < numTransitions) {
	memset(wordList + position, 0, (numTransitions - position) * sizeof(NodeIndex));
	memset(nodeList + position, 0, (numTransitions - position) * sizeof(VocabIndex));
    }

    // do a pair-wise comparison for all the successor nodes.
    for (unsigned i = 0; i < numTransitions; i ++) {
	// check if node has been merged
	if (Map_noKeyP(nodeList[i])) continue;

	for (unsigned j = i + 1; j < numTransitions; j ++) {
	    LatticeNode *nodeI, *nodeJ;

	    if (!Map_noKeyP(nodeList[j]) &&
		wordList[i] == wordList[j] &&
		(nodeI = findNode(nodeList[i]),
		 nodeJ = findNode(nodeList[j]),
		 compareTransitions(nodeI->inTransitions,
				    nodeJ->inTransitions)))
	    {
		mergeNodes(nodeList[i], nodeList[j], nodeI, nodeJ, maxAdd);

		// mark node j as merged for check above
		Map_noKey(nodeList[j]);
	    }
	}
    }
}

/*
 * Try merging predecessors of nodeIndex
 */
void
Lattice::packNodeB(NodeIndex nodeIndex, Boolean maxAdd)
{
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::packNodeB: "
	     << "processing (" << nodeIndex << ") ***********\n"; 
    }

    LatticeNode *node = findNode(nodeIndex); 
    // skip nodes that have already beend deleted
    if (!node) {
      return;
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::packNodeB: "
	     << "      i.e., (" << getWord(node->word) << ")\n";
    }

    unsigned numTransitions = node->inTransitions.numEntries();
    makeArray(NodeIndex, nodeList, numTransitions); 
    makeArray(VocabIndex, wordList, numTransitions);
    
    // going in a reverse direction
    // collect all the in-nodes of nodeIndex, because we need to 
    // be able to delete transitions while iterating over them
    // Also, store the words on each node to allow quick equality checks
    // without findNode() later.  Note we cannot save the findNode() result
    // itself since node objects may more around as a result of deletions.
    TRANSITER_T<NodeIndex,LatticeTransition> 
      inTransIter(node->inTransitions);
    unsigned position = 0; 
    NodeIndex fromNodeIndex;
    while (inTransIter.next(fromNodeIndex)) {
	wordList[position] = findNode(fromNodeIndex)->word;
	nodeList[position] = fromNodeIndex; 
	position ++;
    }
    // For static code analysis, ensure any remainder of arrays
    // are initialized
    if (position < numTransitions) {
	memset(wordList + position, 0, (numTransitions - position) * sizeof(NodeIndex));
	memset(nodeList + position, 0, (numTransitions - position) * sizeof(VocabIndex));
    }

    // do a pair-wise comparison for all the predecessor nodes.
    for (unsigned i = 0; i < numTransitions; i ++) {
	// check if node has been merged
	if (Map_noKeyP(nodeList[i])) continue;

	for (unsigned j = i + 1; j < numTransitions; j ++) {
	    LatticeNode *nodeI, *nodeJ;

	    if (!Map_noKeyP(nodeList[j]) &&
	 	wordList[i] == wordList[j] &&
		(nodeI = findNode(nodeList[i]),
		 nodeJ = findNode(nodeList[j]),
		 compareTransitions(nodeI->outTransitions,
				    nodeJ->outTransitions)))
	    {
		mergeNodes(nodeList[i], nodeList[j], nodeI, nodeJ, maxAdd);

		// mark node j as merged for check above
		Map_noKey(nodeList[j]);
	    }
	}
    }
}

/* ******************************************************** 

   A straight forward implementation for packing bigram lattices.
   combine nodes when their incoming or outgoing node sets are the
   same.

   ******************************************************** */
Boolean 
Lattice::simplePackBigramLattice(unsigned iters, Boolean maxAdd)
{
    // keep track of number of node to know when to stop iteration
    unsigned numNodes = getNumNodes();

    Boolean onlyOne = false;

    if (iters == 0) {
	iters = 1;
	onlyOne = true; // do only one backward pass
    }
	
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::simplePackBigramLattice: "
	     << "starting packing...."
	     << " (" << numNodes  << " nodes)\n";
    }

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);

    while (iters-- > 0) { 
      unsigned numReachable = sortNodes(sortedNodes, true);

      if (numReachable != numNodes) {
	dout() << "Lattice::simplePackBigramLattice: "
	       << "warning: there are " << (numNodes - numReachable)
	       << " unreachable nodes\n";
      }

      for (unsigned i = 0; i < numReachable; i ++) {
	packNodeB(sortedNodes[i], maxAdd);
      }

      unsigned newNumNodes = getNumNodes();
      
      // finish the Backward reduction
      if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::simplePackBigramLattice: "
	       << "done with B Reduction"
	       << " (" << newNumNodes  << " nodes)\n";
      }

      if (onlyOne) {
	break;
      }

      // now sort into forward topological order
      numReachable = sortNodes(sortedNodes, false);

      if (numReachable != newNumNodes) {
	dout() << "Lattice::simplePackBigramLattice: "
	       << "warning: there are " << (newNumNodes - numReachable)
	       << " unreachable nodes\n";
      }

      for (unsigned i = 0; i < numReachable; i ++) {
	packNodeF(sortedNodes[i], maxAdd);
      }

      newNumNodes = getNumNodes();

      if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::simplePackBigramLattice: "
	       << "done with F Reduction"
	       << " (" << newNumNodes  << " nodes)\n";
      }
      // finish one F-B iteration

      // check that lattices got smaller -- if not, stop
      if (newNumNodes == numNodes) {
	break;
      }
      numNodes = newNumNodes;
    }

    delete [] sortedNodes;
    return true; 
}

/* ********************************************************
   this procedure tries to compare two nodes to see whether their
   outgoing node sets overlap more than x.
   ******************************************************** */

Boolean 
Lattice::approxMatchInTrans(NodeIndex nodeIndexI, NodeIndex nodeIndexJ,
			     unsigned overlap)
{
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxMatchInTrans: "
	     << "try to match (" << nodeIndexI
	     << ", " << nodeIndexJ << ") with overlap "
	     << overlap << "\n"; 
    }

    LatticeNode * nodeI = findNode(nodeIndexI); 
    LatticeNode * nodeJ = findNode(nodeIndexJ); 

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxMatchInTrans: "
	     << "      i.e., (" << getWord(nodeI->word)
	     << ", " << getWord(nodeJ->word) << ")\n";
    }

    unsigned numInTransI = nodeI->inTransitions.numEntries();
    unsigned numInTransJ = nodeJ->inTransitions.numEntries();
    unsigned minIJ = ( numInTransI > numInTransJ ) ? 
      numInTransJ : numInTransI;

    if (overlap > minIJ) return false;

    int nonOverlap = minIJ-overlap; 

    NodeIndex toNodeIndex, fromNodeIndex;

    if (debug(DebugPrintInnerLoop)) {
      dout() << "Lattice::approxMatchInTrans: "
	     << "number of transitions (" 
	     << numInTransI << ", " << numInTransJ << ")\n";
    }

    // **********************************************************************
    // preventing self loop generation
    // **********************************************************************

    if (nodeI->inTransitions.find(nodeIndexJ)) {
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::approxMatchInTrans: "
	       << " preventing potential selfloop (" << nodeIndexJ
	       << ", " << nodeIndexI << ")\n";
      }
      return false;
    }

    if (nodeI->outTransitions.find(nodeIndexJ)) {
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::approxMatchInTrans: "
	       << " preventing potential selfloop (" << nodeIndexI
	       << ", " << nodeIndexJ << ")\n";
      }
      return false;
    }


    // **********************************************************************
    // compare the sink nodes of the incoming edges of the two given
    // nodes.
    // **********************************************************************

    // loop over J
    TRANSITER_T<NodeIndex,LatticeTransition> 
      inTransIterJ(nodeJ->inTransitions);
    while (inTransIterJ.next(fromNodeIndex)) {
      
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::approxMatchInTrans: "
	       << "loop (" << fromNodeIndex << ")\n";
      }

      LatticeTransition * transI = nodeI->inTransitions.find(fromNodeIndex);
      if (!transI) {
	// one mismatch occurs;
	if (--nonOverlap < 0) { 
	  return false; }
      } else {
	if (--overlap == 0) { 
	  // already exceed minimum overlap required.
	  break;
	}
      }
    }

    // **********************************************************************
    // I and J are qualified for merging.
    // **********************************************************************

    // merging incoming node sets.
    inTransIterJ.init(); 
    while (LatticeTransition *trans = inTransIterJ.next(fromNodeIndex)) {
      insertTrans(fromNodeIndex, nodeIndexI, *trans);
    }

    // merging outgoing nodes
    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIterJ(nodeJ->outTransitions);
    while (LatticeTransition *trans = outTransIterJ.next(toNodeIndex)) {
      insertTrans(nodeIndexI, toNodeIndex, *trans);
    }
    
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxMatchInTrans: "
	     << "(" << nodeIndexJ << ") has been merged with "
	     << "(" << nodeIndexI << ") and has been removed\n";
    }

    // delete this redudant node.
    removeNode(nodeIndexJ); 
    
    return true;
}

/* ********************************************************
   this procedure is to pack two nodes if their OutNode sets 
   overlap beyong certain threshold
   
   input: 
   1) nodeIndex: the node to be processed;
   2) nodeQueue: the current queue.
   3) base: overlap base
   4) ratio: overlap ratio

   function:
       going through all the in-nodes of nodeIndex, and tries to merge
       as many as possible.
   ******************************************************** */
Boolean 
Lattice::approxRedNodeF(NodeIndex nodeIndex, NodeQueue &nodeQueue, 
			int base, double ratio)
{

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeF: "
	     << "processing (" << nodeIndex << ") ***********\n"; 
    }

    NodeIndex fromNodeIndex, toNodeIndex; 
    LatticeNode * node = findNode(nodeIndex); 
    if (!node) { // skip through this node, being merged.
      return true; 
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeF: "
	     << "      i.e., (" << getWord(node->word) << ")\n";
    }

    // going in a forward direction
    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIter(node->outTransitions);
    unsigned numTransitions = node->outTransitions.numEntries();
    makeArray(NodeIndex, list, numTransitions+1); 
    unsigned position = 0; 
    
    // collect all the out-nodes of nodeIndex
    while (outTransIter.next(toNodeIndex)) {
      list[position++] = toNodeIndex; 
    }

    // do a pair-wise comparison for all the out-nodes.
    unsigned i, j; 
    for (i = 0; i< position; i++) {
        j = i+1; 

	if (j >= position) { 
	  break; }

	LatticeNode * nodeI = findNode(list[i]);

	// ***********************************
	// compare nodeI with nodeJ:
	//    Notice that numInTransI changes because
	//    merger occurs in the loop. this is different
	//    from exact match case.   
	// ***********************************
	unsigned merged = 0; 
	while (j < position) {
	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::approxRedNodeF: "
		   << "comparing (" << list[i] << ", "
		   << list[j] << ")\n"; 
	  }

          LatticeNode * nodeI = findNode(list[i]);
	  unsigned numInTransI = nodeI->inTransitions.numEntries();

	  LatticeNode * nodeJ = findNode(list[j]);
	  unsigned numInTransJ = nodeJ->inTransitions.numEntries();
	  
	  if (nodeI->word == nodeJ->word) {
	    unsigned overlap; 
	    if ((!base && (numInTransI < numInTransJ)) ||
		(base && (numInTransI > numInTransJ))) {
	      overlap = (unsigned ) floor(numInTransI*ratio + 0.5);
	    } else {
	      overlap = (unsigned ) floor(numInTransJ*ratio + 0.5);
	    }

	    overlap = (overlap > 0) ? overlap : 1;
	    if (approxMatchInTrans(list[i], list[j], overlap)) {
	      merged = 1; 
	      list[j] = list[--position];
	      continue;
	    } 
	  } 
	  j++; 
	}

	// clear marks on the inNodes, if nodeI matches some other nodes.
	if (merged) {
	  nodeI = findNode(list[i]);
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    inTransIter(nodeI->inTransitions);
	  while (inTransIter.next(fromNodeIndex)) {
	    LatticeNode * fromNode = findNode(fromNodeIndex); 
	    fromNode->unmarkNode(markedFlag); 
	  }
	}
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeF: "
	     << "adding nodes ("; 
    }

    // **************************************************
    // preparing next level nodes for merging processing.
    // **************************************************
    node = findNode(nodeIndex); 
    TRANSITER_T<NodeIndex,LatticeTransition> 
      transIter(node->outTransitions);
    while (transIter.next(toNodeIndex)) {
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::approxRedNodeF: "
	       << " " << toNodeIndex; 
      }
	  
      nodeQueue.addToNodeQueueNoSamePrev(toNodeIndex); 
      // nodeQueue.addToNodeQueue(toNodeIndex); 
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeF: "
	     << ") to the queue\n"; 
    }

    // mark that the current node has been processed.
    node->markNode(markedFlag);
    
    return true;
}


/* ********************************************************
   this procedure tries to compare two nodes to see whether their
   outgoing node sets overlap more than x.
   ******************************************************** */

Boolean 
Lattice::approxMatchOutTrans(NodeIndex nodeIndexI, NodeIndex nodeIndexJ,
			     unsigned overlap)
{
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxMatchOutTrans: "
	     << "try to match (" << nodeIndexI
	     << ", " << nodeIndexJ << ") with overlap "
	     << overlap << "\n"; 
    }

    LatticeNode * nodeI = findNode(nodeIndexI); 
    LatticeNode * nodeJ = findNode(nodeIndexJ); 

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxMatchOutTrans: "
	     << "      i.e., (" << getWord(nodeI->word)
	     << ", " << getWord(nodeJ->word) << ")\n";
    }

    unsigned numOutTransI = nodeI->outTransitions.numEntries();
    unsigned numOutTransJ = nodeJ->outTransitions.numEntries();
    unsigned minIJ = ( numOutTransI > numOutTransJ ) ? 
      numOutTransJ : numOutTransI;

    if (overlap > minIJ) return false;

    int nonOverlap = minIJ-overlap; 

    NodeIndex toNodeIndex, fromNodeIndex;

    if (debug(DebugPrintInnerLoop)) {
      dout() << "Lattice::approxMatchOutTrans: "
	     << "number of transitions (" 
	     << numOutTransI << ", " << numOutTransJ << ")\n";
    }

    // **********************************************************************
    // preventing self loop generation
    // **********************************************************************

    if (nodeI->inTransitions.find(nodeIndexJ)) {
      if (debug(DebugPrintInnerLoop)) {
        dout() << "Lattice::approxMatchInTrans: "
               << " preventing potential selfloop (" << nodeIndexJ
               << ", " << nodeIndexI << ")\n";
      }
      return false;
    }

    if (nodeI->outTransitions.find(nodeIndexJ)) {
      if (debug(DebugPrintInnerLoop)) {
        dout() << "Lattice::approxMatchInTrans: "
               << " preventing potential selfloop (" << nodeIndexI
               << ", " << nodeIndexJ << ")\n";
      }
      return false;
    }

    // **********************************************************************
    // compare the sink nodes of the outgoing edges of the two given
    // nodes.
    // **********************************************************************

    // loop over J
    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIterJ(nodeJ->outTransitions);
    while (outTransIterJ.next(toNodeIndex)) {
      
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::approxMatchOutTrans: "
	       << "loop (" << toNodeIndex << ")\n";
      }

      LatticeTransition * transI = nodeI->outTransitions.find(toNodeIndex);
      if (!transI) {
	// one mismatch occurs;
	if (--nonOverlap < 0) { 
	  return false; }
      } else {
	if (--overlap == 0) { 
	  // already exceed minimum overlap required.
	  break;
	}
      }
    }

    // **********************************************************************
    // I and J are qualified for merging.
    // **********************************************************************

    // merging outgoing node sets.
    outTransIterJ.init(); 
    while (LatticeTransition *trans = outTransIterJ.next(toNodeIndex)) {
      insertTrans(nodeIndexI, toNodeIndex, *trans);
    }

    // merging incoming nodes
    TRANSITER_T<NodeIndex,LatticeTransition> 
      inTransIterJ(nodeJ->inTransitions);
    while (LatticeTransition *trans = inTransIterJ.next(fromNodeIndex)) {
      insertTrans(fromNodeIndex, nodeIndexI, *trans);
    }
    
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxMatchOutTrans: "
	     << "(" << nodeIndexJ << ") has been merged with "
	     << "(" << nodeIndexI << ") and has been removed\n";
    }

    // delete this redudant node.
    removeNode(nodeIndexJ); 
    
    return true;
}

/* ********************************************************
   this procedure is to pack two nodes if their OutNode sets 
   overlap beyong certain threshold
   
   input: 
   1) nodeIndex: the node to be processed;
   2) nodeQueue: the current queue.
   3) base: overlap base
   4) ratio: overlap ratio

   function:
       going through all the in-nodes of nodeIndex, and tries to merge
       as many as possible.
   ******************************************************** */
Boolean 
Lattice::approxRedNodeB(NodeIndex nodeIndex, NodeQueue &nodeQueue, 
			int base, double ratio)
{

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeB: "
	     << "processing (" << nodeIndex << ") ***********\n"; 
    }

    NodeIndex fromNodeIndex, toNodeIndex; 
    LatticeNode * node = findNode(nodeIndex); 
    if (!node) { // skip through this node, being merged.
      return true; 
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeB: "
	     << "      i.e., (" << getWord(node->word) << ")\n";
    }

    // going in a backward direction
    TRANSITER_T<NodeIndex,LatticeTransition> 
      inTransIter(node->inTransitions);
    unsigned numTransitions = node->inTransitions.numEntries();
    makeArray(NodeIndex, list, numTransitions+1); 
    unsigned position = 0; 
    
    // collect all the in-nodes of nodeIndex
    while (inTransIter.next(fromNodeIndex)) {
      list[position++] = fromNodeIndex; 
    }

    // do a pair-wise comparison for all the in-nodes.
    unsigned i, j; 
    for (i = 0; i< position; i++) {
        j = i+1; 

	if (j >= position) { 
	  break; }

	LatticeNode * nodeI = findNode(list[i]);
	// ***********************************
	// compare nodeI with nodeJ:
	//    Notice that numInTransI changes because
	//    merger occurs in the loop. this is different
	//    from exact match case.   
	// ***********************************
	unsigned merged = 0; 
	while (j < position) {
	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::approxRedNodeB: "
		   << "comparing (" << list[i] << ", "
		   << list[j] << ")\n"; 
	  }

          LatticeNode * nodeI = findNode(list[i]);
	  unsigned numOutTransI = nodeI->outTransitions.numEntries();

	  LatticeNode * nodeJ = findNode(list[j]);
	  unsigned numOutTransJ = nodeJ->outTransitions.numEntries();
	  if (nodeI->word == nodeJ->word) {
	    unsigned overlap; 
	    if ((!base && (numOutTransI < numOutTransJ)) ||
		(base && (numOutTransI > numOutTransJ))) {
	      overlap = (unsigned ) floor(numOutTransI*ratio + 0.5);
	    } else {
	      overlap = (unsigned ) floor(numOutTransJ*ratio + 0.5);
	    }

	    overlap = (overlap > 0) ? overlap : 1;
	    if (approxMatchOutTrans(list[i], list[j], overlap)) {
	      merged = 1; 
	      list[j] = list[--position];
	      continue;
	    } 
	  } 
	  j++; 
	}

	// clear marks on the outNodes, if nodeI matches some other nodes.
	if (merged) {
	  nodeI = findNode(list[i]);
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    outTransIter(nodeI->outTransitions);
	  while (outTransIter.next(toNodeIndex)) {
	    LatticeNode * toNode = findNode(toNodeIndex); 
	    toNode->unmarkNode(markedFlag); 
	  }
	}
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeB: "
	     << "adding nodes ("; 
    }

    // **************************************************
    // preparing next level nodes for merging processing.
    // **************************************************
    node = findNode(nodeIndex); 

    if (!node) { 
      if (debug(DebugPrintFatalMessages)) {
	dout() << "warning: (Lattice::approxRedNodeB) "
	       <<  " this node " << nodeIndex << " get deleted!\n";
      }
      return false;
    }
	
    TRANSITER_T<NodeIndex,LatticeTransition> 
      transIter(node->inTransitions);
    while (transIter.next(fromNodeIndex)) {
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::approxRedNodeB: "
	       << " " << fromNodeIndex; 
      }
	  
      nodeQueue.addToNodeQueueNoSamePrev(fromNodeIndex); 
      // nodeQueue.addToNodeQueue(fromNodeIndex); 
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::approxRedNodeB: "
	     << ") to the queue\n"; 
    }

    // mark that the current node has been processed.
    node->markNode(markedFlag);
    
    return true;
}



/* ******************************************************** 

   An approximating algorithm for reducing bigram lattices.
   combine nodes when their incoming or outgoing node sets overlap
   a significant amount (decided by base and ratio).

   ******************************************************** */
Boolean 
Lattice::approxRedBigramLattice(unsigned iters, int base, double ratio)
{
    // keep track of number of node to know when to stop iteration
    unsigned numNodes = getNumNodes();

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::approxRedBigramLattice: "
	     << "starting reducing...."
	     << " (" << numNodes  << " nodes)\n";
    }
    // if it is a fast mode (default), it returns.
    if (iters == 0) {
      clearMarkOnAllNodes(markedFlag);

      NodeQueue nodeQueue; 
      nodeQueue.addToNodeQueue(final); 

      // use width first approach to go through the whole lattice.
      // mark the first level nodes and put them in the queue.
      // going through the queue to process all the nodes in the lattice
      // this is in a backward direction.
      while (nodeQueue.is_empty() == false) {
	NodeIndex nodeIndex = nodeQueue.popNodeQueue();

	if (nodeIndex == initial) {
	  continue;
	}
	LatticeNode * node = findNode(nodeIndex);
	if (!node) { 
	  continue; 
	}
	if (node->getFlag(markedFlag)) {
	  continue;
	}
	approxRedNodeB(nodeIndex, nodeQueue, base, ratio); 
      }

      numNodes = getNumNodes();

      if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::approxRedBigramLattice: "
	       << "done with only one B Reduction"
	       << " (" << numNodes  << " nodes)\n";
      }
      return true;
    }

    while (iters-- > 0) { 
      clearMarkOnAllNodes(markedFlag);

      // the queue must be empty. Going Backward
      NodeQueue nodeQueue; 
      nodeQueue.addToNodeQueue(final); 

      // use width first approach to go through the whole lattice.
      // mark the first level nodes and put them in the queue.
      // going through the queue to process all the nodes in the lattice
      // this is in a backward direction.
      while (nodeQueue.is_empty() == false) {
	NodeIndex nodeIndex = nodeQueue.popNodeQueue();
	
	if (nodeIndex == initial) {
	  continue;
	}
	LatticeNode * node = findNode(nodeIndex);
	if (!node) { 
	  continue; 
	}
	if (node->getFlag(markedFlag)) {
	  continue;
	}

	approxRedNodeB(nodeIndex, nodeQueue, base, ratio); 
      }
      // finish the Backward reduction
      if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::approxRedBigramLattice: "
	       << "done with B Reduction\n"; 
      }
      clearMarkOnAllNodes(markedFlag);

      // the queue must be empty. Going Forward
      nodeQueue.addToNodeQueue(initial); 

      // use width first approach to go through the whole lattice.
      // mark the first level nodes and put them in the queue.
      // going through the queue to process all the nodes in the lattice
      // this is in a forward direction.
      while (nodeQueue.is_empty() == false) {
	NodeIndex nodeIndex = nodeQueue.popNodeQueue();

	if (nodeIndex == final) {
	  continue;
	}
	LatticeNode * node = findNode(nodeIndex);
	if (!node) { 
	  continue; 
	}
	if (node->getFlag(markedFlag)) {
	  continue;
	}
	
	approxRedNodeF(nodeIndex, nodeQueue, base, ratio); 
      }

      unsigned newNumNodes = getNumNodes();

      if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::approxRedBigramLattice: "
	       << "done with F Reduction"
	       << " (" << newNumNodes  << " nodes)\n";
      }
      // finish one F-B iteration

      // check that lattices got smaller -- if not, stop
      if (newNumNodes == numNodes) {
	break;
      }
      numNodes = newNumNodes;
    }
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::approxRedBigramLattice: "
	     << "done with multiple iteration(s)\n"; 
    }
    return true; 
}

/*
 * Reduce lattice by collapsing all nodes with the same word
 *	(except those in the exceptions sub-vocabulary)
 */
Boolean
Lattice::collapseSameWordNodes(SubVocab &exceptions)
{
    LHash<VocabIndex, NodeIndex> wordToNodeMap;

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::collapseSameWordNodes: "
	     << "starting with " << getNumNodes() << " nodes\n";
    }

    NodeIndex nodeIndex;
    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {

	if (node->word != Vocab_None &&
	    !ignoreWord(node->word) &&
	    !exceptions.getWord(node->word))
	{
	    Boolean foundP;

	    NodeIndex *oldNode = wordToNodeMap.insert(node->word, foundP);

	    if (!foundP) {
		// word is new, save its node index
		*oldNode = nodeIndex;
	    } else {
		// word has been found before -- merge nodes
		mergeNodes(*oldNode, nodeIndex);
	    }
	}
    }

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::collapseSameWordNodes: "
	     << "finished with " << getNumNodes() << " nodes\n";
    }

    return true;
}

