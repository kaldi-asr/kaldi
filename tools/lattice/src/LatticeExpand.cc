/*
 * LatticeExpand.cc --
 *	Lattice expansion and LM rescoring algorithms
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1997-2012 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeExpand.cc,v 1.11 2012/10/18 20:55:21 mcintyre Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "Lattice.h"

#include "LHash.cc"
#include "Map2.cc"
#include "Array.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_MAP2(NodeIndex, VocabContext, NodeIndex);
INSTANTIATE_LHASH(VocabIndex,PackedNode);
#endif

/*
 * If the intlog weights of two transitions differ by no more than this
 * they are considered identical in PackedNodeList::packNodes().
 */
#define PACK_TOLERANCE			0

#define DebugPrintFatalMessages         0 

#define DebugPrintFunctionality         1 
// for large functionality listed in options of the program
#define DebugPrintOutLoop               2
// for out loop of the large functions or small functions
#define DebugPrintInnerLoop             3
// for inner loop of the large functions or outloop of small functions


#ifndef USE_SARRAY_MAP2
/*
 * Word ngram sorting function
 * (used to iterate over contexts in node expansion maps in same order
 * regardless of underlying datastructure)
 */
static int
ngramCompare(const VocabIndex *n1, const VocabIndex *n2)
{
    return SArray_compareKey(n1, n2);
}
#endif /* USE_SARRAY_MAP2 */

/* this code is to replace weights on the links of a given lattice with
 * the LM weights.
 */
Boolean 
Lattice::replaceWeights(LM &lm)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::replaceWeights: "
	     << "replacing weights with new LM\n";
    }

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {

      NodeIndex toNodeIndex;
      VocabIndex wordIndex;
      if (nodeIndex == initial) {
	wordIndex = vocab.ssIndex();
      } else {
	wordIndex = node->word; 
      }
      // need to check to see whether the word is in the vocab


      TRANSITER_T<NodeIndex,LatticeTransition> transIter(node->outTransitions);
      while (transIter.next(toNodeIndex)) {
	LatticeNode * toNode = nodes.find(toNodeIndex);

	VocabIndex toWordIndex;
        LogP weight;

	if (toNodeIndex == final) {
	  toWordIndex = vocab.seIndex(); }
	else {
	  toWordIndex = toNode->word; }

	if (toWordIndex == Vocab_None || toWordIndex == lm.vocab.pauseIndex()) {
	  /*
	   * NULL and pause nodes don't receive an language model weight
	   */
	  weight = LogP_One;
	} else {
	  VocabIndex context[2];
	  context[0] = wordIndex; 
	  context[1] = Vocab_None; 

	  weight = lm.wordProb(toWordIndex, context); 
	}

	setWeightTrans(nodeIndex, toNodeIndex, weight);
      }
    }

    return true; 
}

/*
 * Compute outgoing transition prob on demand.  This saves LM computation
 * for transitions that are cached.
 */
static Boolean
computeOutWeight(PackInput &packInput)
{
  if (packInput.lm != 0) {
    VocabIndex context[3];
    context[0] = packInput.wordName;
    context[1] = packInput.fromWordName;
    context[2] = Vocab_None;

    packInput.outWeight = packInput.lm->wordProb(packInput.toWordName, context);
    packInput.lm = 0;
  }

  return true;
}

/* this function tries to pack together nodes in lattice
 * 1) for non-self loop case: only when trigram prob exists,
 *    the from nodes with the same wordName will be packed;
 * 2) for self loop case: 
 *    the from nodes with the same wordName will be packed, 
 *    regardless whether the trigram prob exists.
 *    But, the bigram and trigram will have separate nodes,
 *    which is reflected in two different out transitions from
 *    the mid node to the two different toNodes (bigram and trigram)
 */
Boolean 
PackedNodeList::packNodes(Lattice &lat, PackInput &packInput)
{
  PackedNode *packedNode = packedNodesByFromNode.find(packInput.fromWordName);

  if (!packedNode && lastPackedNode != 0 &&
      (packInput.toNodeIndex == lastPackedNode->toNode &&
       computeOutWeight(packInput) &&
       abs(LogPtoIntlog(packInput.outWeight) -
	   LogPtoIntlog(lastPackedNode->outWeight)) <= PACK_TOLERANCE))
  {
    packedNode = lastPackedNode;
    NodeIndex midNode = packedNode->midNodeIndex;

    // the fromNode could be different this time around, so we need to
    // re-cache the mid-node 
    packedNode = packedNodesByFromNode.insert(packInput.fromWordName); 

    packedNode->midNodeIndex = midNode; 
    packedNode->toNode = packInput.toNodeIndex; 
    packedNode->outWeight = packInput.outWeight; 

    if (packInput.toNodeId == 2) { 
      packedNode->toNodeId = 2; 
    } else if (packInput.toNodeId == 3) { 
      packedNode->toNodeId = 3; 
    } else {
      packedNode->toNodeId = 0; 
    }

    lastPackedNode = packedNode;
  }

  if (packedNode) {
    // only one transition is needed;
    LatticeTransition t(packInput.inWeight, packInput.inFlag);
    lat.insertTrans(packInput.fromNodeIndex, packedNode->midNodeIndex, t);

    if (!packInput.toNodeId) { 
      // this is for non-self-loop node, no additional outgoing trans
      // need to be added.

      LatticeNode *midNode = lat.findNode(packedNode->midNodeIndex);
      LatticeTransition * trans = 
        midNode->outTransitions.find(packInput.toNodeIndex);

      // if it is another toNode, we need to create a link to it.
      if (!trans) {
        // it indicates that there is another ngram node needed.
        computeOutWeight(packInput);
	LatticeTransition t(packInput.outWeight, packInput.outFlag);
        lat.insertTrans(packedNode->midNodeIndex, packInput.toNodeIndex, t);

	if (debug(DebugPrintInnerLoop)) {
	  dout() << "PackedNodeList::packNodes: \n"
		 << "insert (" << packInput.fromNodeIndex
		 << ", " << packedNode->midNodeIndex << ", " 
		 << packInput.toNodeIndex << ")\n";
        }
      }

      return true;
    } else {
	if (debug(DebugPrintInnerLoop)) {
	  dout() << "PackedNodeList::packNodes: \n"
		 << "reusing (" << packInput.fromNodeIndex
		 << ", " << packedNode->midNodeIndex << ", " 
		 << packInput.toNodeIndex << ")\n";
        }
    }

    // the following part is for selfLoop case
    // the toNode is for p(a | a, x) doesn't exist.
    if (packInput.toNodeId == 2) {
      if (!packedNode->toNode) {
        computeOutWeight(packInput);
	LatticeTransition t(packInput.outWeight, packInput.outFlag); 
	lat.insertTrans(packedNode->midNodeIndex, packInput.toNodeIndex, t);
	packedNode->toNode = packInput.toNodeIndex;
      }
      return true;
    }

    // the toNode is for p(a | a, x) exists.
    if (packInput.toNodeId == 3) {
      if (!packedNode->toNode) {
        computeOutWeight(packInput);
	LatticeTransition t(packInput.outWeight, packInput.outFlag);
	lat.insertTrans(packedNode->midNodeIndex, packInput.toNodeIndex, t);
	packedNode->toNode = packInput.toNodeIndex;
      }
      return true;
    }
  } else {
    // this is the first time to create triple.
    NodeIndex newNodeIndex = lat.dupNode(packInput.wordName, markedFlag);
    
    LatticeTransition t1(packInput.inWeight, packInput.inFlag); 
    lat.insertTrans(packInput.fromNodeIndex, newNodeIndex, t1);

    computeOutWeight(packInput);
    LatticeTransition t2(packInput.outWeight, packInput.outFlag);
    lat.insertTrans(newNodeIndex, packInput.toNodeIndex, t2);

    if (debug(DebugPrintInnerLoop)) {
      dout() << "PackedNodeList::packNodes: \n"
	     << "insert (" << packInput.fromNodeIndex
	     << ", " << newNodeIndex << ", " 
	     << packInput.toNodeIndex << ")\n";
    }

    packedNode = packedNodesByFromNode.insert(packInput.fromWordName); 

    packedNode->midNodeIndex = newNodeIndex; 
    packedNode->toNode = packInput.toNodeIndex; 
    packedNode->outWeight = packInput.outWeight; 

    if (packInput.toNodeId == 2) { 
      packedNode->toNodeId = 2; 
    } else if (packInput.toNodeId == 3) { 
      packedNode->toNodeId = 3; 
    } else {
      packedNode->toNodeId = 0; 
    }

    lastPackedNode = packedNode;
  }

  return true;
}

// *************************************************
// compact expansion to trigram
// *************************************************
/*  Basic Algorithm: 
 *  Try to expand self loop to accomodate trigram
 *  the basic idea has two steps:
 *  1) ignore the loop edge and process other edge combinations
 *     just like in other cases, this is done in the main expandNodeToTrigram 
 *     program
 *  2) IN THIS PROGRAM: 
 *     a) duplicate the loop node (called postNode); 
 *     b) add an additional node (called preNode) between fromNode and the 
 *        loop node (postNode);
 *     c) create links between fromNode, preNode, postNode and toNode; and 
 *        create the loop edge on the loop node (postNode).
 */
void 
Lattice::initASelfLoopDB(SelfLoopDB &selfLoopDB, LM &lm, 
			NodeIndex nodeIndex, LatticeNode *node, 
			LatticeTransition *trans)
{
    selfLoopDB.preNodeIndex = selfLoopDB.postNodeIndex2 = 
      selfLoopDB.postNodeIndex3 = 0; 
    selfLoopDB.nodeIndex = nodeIndex; 

    selfLoopDB.selfTransFlags = trans->flags;

    selfLoopDB.wordName = node->word; 

    VocabIndex context[3];
    context[0] = selfLoopDB.wordName; 
    context[1] = selfLoopDB.wordName; 
    context[2] = Vocab_None; 

    selfLoopDB.loopProb = lm.wordProb(selfLoopDB.wordName, context); 
}

void 
Lattice::initBSelfLoopDB(SelfLoopDB &selfLoopDB, LM &lm, 
			NodeIndex fromNodeIndex, LatticeNode * fromNode, 
			LatticeTransition *fromTrans)
{
    // reinitialize the preNode
    selfLoopDB.preNodeIndex = 0; 

    // 
    selfLoopDB.fromNodeIndex = fromNodeIndex; 
    selfLoopDB.fromWordName = fromNode->word; 

    // 
    selfLoopDB.fromSelfTransFlags = fromTrans->flags; 

    // compute prob for the link between preNode and postNode
    VocabIndex context[3];
    context[0] = selfLoopDB.wordName; 
    context[1] = selfLoopDB.fromWordName; 
    context[2] = Vocab_None; 
    
    selfLoopDB.prePostProb =
			lm.wordProb(selfLoopDB.wordName, context); 

    // compute prob for fromPreProb; 
    context[0] = selfLoopDB.fromWordName; 
    context[1] = Vocab_None;

    selfLoopDB.fromPreProb = 
    			lm.wordProb(selfLoopDB.wordName, context); 
}

void 
Lattice::initCSelfLoopDB(SelfLoopDB &selfLoopDB, NodeIndex toNodeIndex, 
			LatticeTransition *toTrans)
{
    selfLoopDB.toNodeIndex = toNodeIndex;
    selfLoopDB.selfToTransFlags = toTrans->flags;
}

/* 
 * creating an expansion network for a self loop node.
 * the order in which the network is created is reverse:
 *  1) build the part of the network starting from postNode to toNode
 *  2) use PackedNodeList class function to build the part of 
 *     the network starting from fromNode to PostNode. 
 *
 */
Boolean
Lattice::expandSelfLoop(LM &lm, SelfLoopDB &selfLoopDB, 
			       PackedNodeList &packedSelfLoopNodeList)
{
    unsigned id = 0;
    NodeIndex postNodeIndex, toNodeIndex = selfLoopDB.toNodeIndex;
    LogP fromPreProb = selfLoopDB.fromPreProb; 
    LogP prePostProb = selfLoopDB.prePostProb; 

    VocabIndex wordName = selfLoopDB.wordName; 

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::expandSelfLoop: "
	     << "nodeIndex (" << selfLoopDB.nodeIndex << ")\n";
    }

    // create the part of the network from postNode to toNode 
    //   if it doesn't exist.
    // first compute the probs of the links in that part.
    VocabIndex context[3];
    context[0] = wordName; 
    context[1] = wordName; 
    context[2] = Vocab_None; 

    LatticeNode *toNode = findNode(toNodeIndex); 
    VocabIndex toWordName = toNode->word; 

    LogP triProb = lm.wordProb(toWordName, context);

    unsigned usedContextLength;
    lm.contextID(context, usedContextLength);

    context[1] = Vocab_None;
    LogP biProb = lm.wordProb(toWordName, context);

    LogP postToProb; 

    if (usedContextLength > 1) {

      // get trigram prob for (post, to) edge: p(c|a, a)
      postToProb = triProb; 

      // create post node and loop if it doesn't exist;
      if (!selfLoopDB.postNodeIndex3) {
	selfLoopDB.postNodeIndex3 = 
	  postNodeIndex = dupNode(wordName, markedFlag);
	
	// create the loop, put trigram prob p(a|a,a) on the loop
	LatticeTransition t(selfLoopDB.loopProb, selfLoopDB.selfTransFlags);
	insertTrans(postNodeIndex, postNodeIndex, t); 
	// end of creating of loop
      }

      postNodeIndex = selfLoopDB.postNodeIndex3; 
      id = 3; 

    } else {
      
      // get an adjusted weight for the link between preNode to postNode
      LogP wordBOW = triProb - biProb;

      prePostProb = combWeights(prePostProb, wordBOW); 

      // get existing weight of (node, toNode) as the weight for (post, to).
      LatticeNode *node = findNode(selfLoopDB.nodeIndex); 
      if (!node) {
	if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal Error in Lattice::expandSelfLoop: "
		 << "can't find node " << selfLoopDB.nodeIndex << "\n"; 
	}
	exit(-1);
      }

      // compute postToProb
      postToProb = biProb; 

      // create post node and loop if it doesn't exist;
      if (!selfLoopDB.postNodeIndex2) {
	selfLoopDB.postNodeIndex2 = 
	  postNodeIndex = dupNode(wordName, markedFlag);

	// create the loop, put trigram prob p(a|a,a) on the loop
	LatticeTransition t(selfLoopDB.loopProb, selfLoopDB.selfTransFlags);
	insertTrans(postNodeIndex, postNodeIndex, t); 
	// end of creating loop
      }
      postNodeIndex = selfLoopDB.postNodeIndex2; 
      id = 2; 
    }

    // create link from postNode to toNode if (postNode, toNode) doesn't exist;
    toNode = findNode(toNodeIndex); 
    LatticeTransition *postToTrans = toNode->inTransitions.find(postNodeIndex);
    if (!postToTrans) {
      // create link from postNode to toNode;
      LatticeTransition t(postToProb, selfLoopDB.selfToTransFlags); 
      insertTrans(postNodeIndex, toNodeIndex, t); 
    }
    // done with first part of the network. 

    // create the part of the network from fromNode to postNode.
    // create preNode and (from, pre) edge.
    NodeIndex preNodeIndex = selfLoopDB.preNodeIndex; 

    PackInput packSelfLoop;
    packSelfLoop.wordName = wordName; 
    packSelfLoop.fromWordName = selfLoopDB.fromWordName;
    packSelfLoop.toWordName = toNode->word; 
    packSelfLoop.fromNodeIndex = selfLoopDB.fromNodeIndex; 
    packSelfLoop.toNodeIndex = postNodeIndex; 
    packSelfLoop.inWeight = selfLoopDB.fromPreProb;
    packSelfLoop.inFlag = selfLoopDB.fromSelfTransFlags; 
    packSelfLoop.outWeight = prePostProb; 
    packSelfLoop.toNodeId = id; 
    packSelfLoop.lm = 0; 

    packedSelfLoopNodeList.packNodes(*this, packSelfLoop); 

    return true;
}

Boolean 
Lattice::expandNodeToTrigram(NodeIndex nodeIndex, LM &lm, unsigned maxNodes)
{
    SelfLoopDB selfLoopDB; 

    PackedNodeList packedNodeList, 
      packedSelfLoopNodeList; 

    LatticeTransition *outTrans;
    NodeIndex fromNodeIndex;
    NodeIndex toNodeIndex;
    LatticeTransition *inTrans;
    LatticeNode *fromNode; 
    VocabIndex context[3];
    LatticeNode *node = findNode(nodeIndex); 
    if (!node) {
	if (debug(DebugPrintFatalMessages)) {
            dout() << "Lattice::expandNodeToTrigram: "
		   << "Fatal Error: current node doesn't exist!\n";
	}
	exit(-1); 
    }

    LatticeTransition * selfLoop = node->inTransitions.find(nodeIndex); 
    Boolean selfLoopFlag; 
    if (selfLoop) { 
      selfLoopFlag = true; 
      initASelfLoopDB(selfLoopDB, lm, nodeIndex, node, selfLoop); 
    } else {
      selfLoopFlag = false; 
    }

    TRANSITER_T<NodeIndex,LatticeTransition> inTransIter(node->inTransitions);
    TRANSITER_T<NodeIndex,LatticeTransition> outTransIter(node->outTransitions);

    VocabIndex wordName = node->word; 

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::expandNodeToTrigram: "
	     << "processing word name: " << getWord(wordName) << ", Index: " 
	     << nodeIndex << "\n";
    }

    // going through all its incoming edges
    while ((inTrans = inTransIter.next(fromNodeIndex))) {

      if (nodeIndex == fromNodeIndex) {

	if (debug(DebugPrintOutLoop)) {
	  dout() << "Lattice::expandNodeToTrigram: jump over self loop: " 
	         << fromNodeIndex << "\n"; 
	}

	continue; 
      }

      fromNode = findNode(fromNodeIndex); 
      if (!fromNode) {
	if (debug(DebugPrintFatalMessages)) {
	    dout() << "Lattice::expandNodeToTrigram: "
		   << "Fatal Error: fromNode " 
	           << fromNodeIndex << " doesn't exist!\n";
	}
	exit(-1); 
      }
      VocabIndex fromWordName = fromNode->word; 

      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::expandNodeToTrigram: processing incoming edge: (" 
	       << fromNodeIndex << ", " << nodeIndex << ")\n" 
	       << "      (" << getWord(fromWordName)
	       << ", " << getWord(wordName) << ")\n"; 
      }

      // compute in bigram prob
      LogP inWeight; 
      if (fromNodeIndex == getInitial()) {
	context[0] = fromWordName; 
	context[1] = Vocab_None; 
	inWeight = lm.wordProb(wordName, context);
      } else { 
	inWeight = inTrans->weight;
      }

      context[0] = wordName; 
      context[1] = fromWordName; 
      context[2] = Vocab_None; 

      unsigned inFlag = inTrans->flags; 

      // initialize it for self loop processing.
      if (selfLoopFlag) {
	initBSelfLoopDB(selfLoopDB, lm, fromNodeIndex, fromNode, inTrans);
      }

      // going through all the outgoing edges
      //       node = findNode(nodeIndex); 

      outTransIter.init(); 
      while (LatticeTransition * outTrans = outTransIter.next(toNodeIndex)) {
	
	if (nodeIndex == toNodeIndex) {
	  dout() << " In expandNodeToTrigram: self loop: " 
	         << toNodeIndex << "\n"; 
	  
	  continue; 
	}

	LatticeNode * toNode = findNode(toNodeIndex); 
	if (!toNode) {
	    if (debug(DebugPrintFatalMessages)) {
	        dout() << "Lattice::expandNodeToTrigram: "
		       << "Fatal Error: toNode " 
	               << toNode << " doesn't exist!\n";
	    }
	    exit(-1); 
	}

	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::expandNodeToTrigram: the toNodeIndex (" 
	         << toNodeIndex << " has name "
		 << getWord(toNode->word) << ")\n"; 
	}

	// initialize selfLoopDB;
	if (selfLoopFlag) { 
	  initCSelfLoopDB(selfLoopDB, toNodeIndex, outTrans); 
	}

	// duplicate a node if the trigram exists.

	// computed on demand in packNodes(), saving work for cached transitions
	// LogP logProb = lm.wordProb(toNode->word, context);
	  
	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::expandNodeToTrigram: tripleIndex (" 
	         << toNodeIndex << " | " << nodeIndex << ", "
	         << fromNodeIndex << ")\n"
	         << "      trigram prob: (" 
	         << getWord(toNode->word) << " | "
		 << context << ") found!!!!!!!!\n"; 
	}

	// create one node and two edges to place trigram prob
	// I need to do packing nodes here.

	PackInput packInput;
	packInput.fromWordName = fromWordName;
	packInput.wordName = wordName;
	packInput.toWordName = toNode->word;
	packInput.fromNodeIndex = fromNodeIndex;
	packInput.toNodeIndex = toNodeIndex; 
	packInput.inWeight = inWeight; 
	// computed on demand in packNodes()
	//packInput.outWeight = logProb; 
	packInput.lm = &lm;
	packInput.inFlag = inFlag; 
	packInput.outFlag = outTrans->flags; 
	packInput.nodeIndex = nodeIndex; 
	packInput.toNodeId = 0;
	
	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::expandNodeToTrigram: "
	         << "outgoing edge for first incoming edge: (" 
	         << nodeIndex << ", " << toNodeIndex << ") is reused\n"; 
	}

	packedNodeList.packNodes(*this, packInput); 

        if (maxNodes > 0 && getNumNodes() > maxNodes) {
	  dout() << "Lattice::expandNodeToTrigram: "
	         << "aborting with number of nodes exceeding "
	         << maxNodes << endl;
	  return false;
	}
	
	// processing selfLoop
	if (selfLoopFlag) { 
	  expandSelfLoop(lm, selfLoopDB, packedSelfLoopNodeList); 
	}
      }	  // end of inter-loop
    } // end of out-loop

    // processing selfLoop case
    if (selfLoopFlag) { 
          node = findNode(nodeIndex);
	  node->inTransitions.remove(nodeIndex);
	  node = findNode(nodeIndex);
	  
	  if (!node->outTransitions.remove(nodeIndex)) {
	    dout() << "Lattice::expandNodeToTrigram: "
	           << "nonFatal Error: non symetric setting\n";
	    exit(-1); 
	  }
    }

    // remove bigram transitions along with the old node
    removeNode(nodeIndex); 
    return true; 
}

Boolean 
Lattice::expandToTrigram(LM &lm, unsigned maxNodes)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::expandToTrigram: "
	     << "starting expansion to conventional trigram lattice ...\n";
    }

    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::expandToTrigram: warning: called with unreachable nodes\n";
      }
    }

    for (unsigned i = 0; i < numReachable; i++) {
      NodeIndex nodeIndex = sortedNodes[i];

      if (nodeIndex == initial || nodeIndex == final) {
	continue;
      }
      if (!expandNodeToTrigram(nodeIndex, lm, maxNodes)) {
        delete [] sortedNodes;
	return false;
      }
    }

    delete [] sortedNodes;
    return true; 
}

/*
 * Expand bigram lattice to trigram, with bigram packing 
 *   (just like in nodearray.)
 *
 *   BASIC ALGORITHM: 
 *      1) foreach node u connecting with the initial NULL node, 
 *              let W be the set of nodes that have edge go into
 *              node u.
 *              a) get the set of outgoing edges e(u) of node u whose 
 *                      the other ends of nodes are not marked as processed.
 *
 *              b) for each edge e = (u, v) in e(u): 
 *                      for each node w in W do:
 *                        i)  if p(v | u, w) exists, 
 *                              duplicate u to get u' ( word name ), 
 *                              and edge (w, u') and (u', v)
 *                              with all the attributes.
 *                              place p(v | u, w) on edge (u', v)
 *
 *                       ii)  if p(v | u, w) does not exist,
 *                              add p(v | u) on edge (u, v)
 *                              multiply bo(w,u) to p(u | w) on (w, u)
 * 
 *  reservedFlag: to indicate that not all the outGoing nodes from the 
 *      current node have trigram probs and this bigram edge needs to be 
 *      preserved for bigram prob.
 */
static /*const*/ LogP zerobow = 0.0; 
Boolean 
Lattice::expandNodeToCompactTrigram(NodeIndex nodeIndex, Ngram &ngram,
							unsigned maxNodes)
{
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::expandNodeToCompactTrigram: \n";
    }

    SelfLoopDB selfLoopDB; 
    PackedNodeList packedNodeList, 
      packedSelfLoopNodeList; 
    LatticeTransition *outTrans;
    NodeIndex fromNodeIndex, backoffNodeIndex;
    NodeIndex toNodeIndex;
    LatticeNode *fromNode; 
    VocabIndex * bowContext = 0; 
    int inBOW = 0; 
    VocabIndex context[3];
    LatticeNode *node = findNode(nodeIndex); 
    if (!node) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal Error in Lattice::expandNodeToCompactTrigram: "
		 << "current node has lost!\n";
	}
	exit(-1); 
    }

    LatticeTransition * selfLoop = node->inTransitions.find(nodeIndex); 
    Boolean selfLoopFlag; 
    if (selfLoop) { 
      selfLoopFlag = true; 
      initASelfLoopDB(selfLoopDB, ngram, nodeIndex, node, selfLoop); 
    } else {
      selfLoopFlag = false; 
    }

    TRANSITER_T<NodeIndex,LatticeTransition> inTransIter(node->inTransitions);
    TRANSITER_T<NodeIndex,LatticeTransition> outTransIter(node->outTransitions);

    VocabIndex wordName = node->word; 

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::expandNodeToCompactTrigram: "
	     << " processing word name: " << getWord(wordName) << ", Index: " 
	     << nodeIndex << "\n";
    }

    // going through all its incoming edges
    unsigned numInTrans = node->inTransitions.numEntries(); 

    while (LatticeTransition *inTrans = inTransIter.next(fromNodeIndex)) {

      if (nodeIndex == fromNodeIndex) {
	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::expandNodeToCompactTrigram: "
		 << "jump over self loop: " 
		 << fromNodeIndex << "\n"; 
	}
	continue; 
      }

      fromNode = findNode(fromNodeIndex); 
      if (!fromNode) {
	if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal Error in Lattice::expandNodeToCompactTrigram: "
		 << "fromNode " 
		 << fromNodeIndex << " doesn't exist!\n";
	}
	exit(-1); 
      }
      VocabIndex fromWordName = fromNode->word; 

      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::expandNodeToCompactTrigram: "
	       << "processing incoming edge: (" 
	       << fromNodeIndex 
	       << ", " << nodeIndex << ")\n"
	       << "      (" << getWord(fromWordName) << ", "
	       << getWord(wordName) << ")\n"; 
      }

      // compute in-coming bigram prob
      LogP inWeight; 
      if (fromNodeIndex == getInitial()) {
	context[0] = fromWordName; 
	context[1] = Vocab_None; 
	// this transition can have never been processed.
	inWeight = ngram.wordProb(wordName, context);
	inTrans->weight = inWeight; 

	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::expandNodeToCompactTrigram: "
		 << "processing incoming edge: (" 
		 << fromNodeIndex 
		 << ", " << nodeIndex << ")\n"
		 << "      (" << getWord(fromWordName)
		 << ", " << getWord(wordName) << ") = "
		 << "  " << inWeight << ";\n"; 
	}
      } else { 
	// the in-coming trans has been processed and 
	// we should preserve this value.
	inWeight = inTrans->weight;
      }

      context[0] = wordName; 
      context[1] = fromWordName; 
      context[2] = Vocab_None; 

      // LogP inWeight = ngram.wordProb(wordName, context);
      unsigned inFlag = inTrans->flags; 

      // initialize it for self loop processing.
      if (selfLoopFlag) {
	initBSelfLoopDB(selfLoopDB, ngram, fromNodeIndex, fromNode, inTrans); }

      // going through all the outgoing edges
      outTransIter.init(); 
      while (LatticeTransition *outTrans = outTransIter.next(toNodeIndex)) {
	
	if (nodeIndex == toNodeIndex) {
	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::expandNodeToCompactTrigram: "
		   << "self loop: " 
		   << toNodeIndex << "\n"; 
	  }
	  continue; 
	}

	LatticeNode * toNode = findNode(toNodeIndex); 
	if (!toNode) {
	  if (debug(DebugPrintFatalMessages)) {
	    dout() << "Fatal Error in Lattice::expandNodeToCompactTrigram: "
		   << "toNode " 
		   << toNode << " doesn't exist!\n";
	  }
	  exit(-1); 
	}

	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::expandNodeToCompactTrigram: "
		 << "the toNodeIndex (" 
		 << toNodeIndex << " has name "
		 << getWord(toNode->word) << ")\n"; 
	}

	// initialize selfLoopDB;
	if (selfLoopFlag) { 
	  initCSelfLoopDB(selfLoopDB, toNodeIndex, outTrans); 
	}

	// duplicate a node if the trigram exists.
	// see class Ngram in file /home/srilm/devel/lm/src/Ngram.h
	
	LogP * triProb; 

	if ((triProb = ngram.findProb(toNode->word, context))) {
	  LogP logProb = *triProb; 
	  
	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::expandNodeToCompactTrigram: "
		   << "tripleIndex (" 
		   << toNodeIndex << " | " << nodeIndex << ", "
		   << fromNodeIndex << ")\n"
		   << "      trigram prob: (" 
		   << getWord(toNode->word) << " | " << context << ") found!\n"; 
	  }

	  // create one node and two edges to place trigram prob
	  PackInput packInput;
	  packInput.fromWordName = fromWordName;
	  packInput.wordName = wordName;
	  packInput.toWordName = toNode->word;
	  packInput.fromNodeIndex = fromNodeIndex;
	  packInput.toNodeIndex = toNodeIndex; 
	  packInput.inWeight = inWeight; 
	  packInput.inFlag = inFlag; 
	  packInput.outWeight = logProb; 
	  packInput.outFlag = outTrans->flags; 
	  packInput.nodeIndex = nodeIndex; 
	  packInput.toNodeId = 0;
	  packInput.lm = 0;
	  
	  packedNodeList.packNodes(*this, packInput); 

	  // to remove the outGoing edge if all the outgoing nodes have
	  // trigram probs.
	  if (numInTrans == 1 && 
	      !(outTrans->getFlag(reservedTFlag))) {

	    if (debug(DebugPrintInnerLoop)) {
	      dout() << "Lattice::expandNodeToCompactTrigram: "
		     << "outgoing edge: (" 
		     << nodeIndex << ", " << toNodeIndex << ") is removed\n"; 
	    }
	    removeTrans(nodeIndex, toNodeIndex); 
	  }

	  if (maxNodes > 0 && getNumNodes() > maxNodes) {
	    dout() << "Lattice::expandNodeToCompactTrigram: "
		   << "aborting with number of nodes exceeding "
		   << maxNodes << endl;
	    return false;
	  }
	} else {
	  // there is no trigram prob for this context

	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::expandNodeToCompactTrigram: "
		   << "no trigram context (" 
		   << context << ") has been found -- keep " 
		   << fromNodeIndex << "\n"; 
	  }

	  // note down backoff context and in-coming node for 
	  // preservation, in case explicit trigram does not exist.
	  bowContext = context; 
	  outTrans->markTrans(reservedTFlag); 
	  backoffNodeIndex = fromNodeIndex;
	}
	
	// processing selfLoop
	if (selfLoopFlag) { 
	  expandSelfLoop(ngram, selfLoopDB, packedSelfLoopNodeList); 
	}
      }	  // end of inter-loop

      // processing incoming bigram cases.
      if (!bowContext) {
  	  // for this context, all the toNodes have trigram probs

	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::expandNodeToCompactTrigram: "
		   << "incoming edge ("
		   << fromNodeIndex << ", " << nodeIndex
		   << ") is removed\n"; 
	  }

  	  removeTrans(fromNodeIndex, nodeIndex); 
      } else {
	  if (debug(DebugPrintInnerLoop)) {
	      dout() << "Lattice::expandNodeToCompactTrigram: "
		     << "updating trigram backoffs on edge("
		     << fromNodeIndex << ", " << nodeIndex << ")\n"; 
	  }

 	  LogP * wordBOW = ngram.findBOW(bowContext);
	  if (!(wordBOW)) {
  	      if (debug(DebugPrintOutLoop)) {
		dout() << "nonFatal Error in Lattice::expandNodeToCompactTrigram: "
		       << "language model - BOW (" 
		       << bowContext << ") missing!\n";
	      }

	      wordBOW = &zerobow; 
	  }

	  LogP logProbW = *wordBOW; 
	  LogP weight = combWeights(inWeight, logProbW); 

	  setWeightTrans(backoffNodeIndex, nodeIndex, weight); 

	  bowContext = 0;
	  inBOW = 1; 
      }

      numInTrans--;
    } // end of out-loop

    // if trigram prob exist for all the tri-node paths
    if (!inBOW) {

        if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::expandNodeToCompactTrigram: "
		 << "node "
		 << getWord(wordName) << " (" << nodeIndex
		 << ") has trigram probs for all its contexts\n"
		 << " and its bigram lattice node is removed\n"; 
	}

        removeNode(nodeIndex); 
    } else {
        node = findNode(nodeIndex);
        if (selfLoopFlag) { 
	  node->inTransitions.remove(nodeIndex);
	  node = findNode(nodeIndex);
	  
	  if (!node->outTransitions.remove(nodeIndex)) {
 	      if (debug(DebugPrintFatalMessages)) {
		dout() << "nonFatal Error in Lattice::expandNodeToCompactTrigram: "
		       << "non symetric setting \n";
	      }
	      exit(-1); 
	  }
	} 

	// process backoff to bigram weights. 
	TRANSITER_T<NodeIndex,LatticeTransition> 
	  outTransIter(node->outTransitions);
	while (LatticeTransition *outTrans = outTransIter.next(toNodeIndex)) {
	  
	  LatticeNode * toNode = findNode(toNodeIndex);
	  context[0] = wordName; 
	  context[1] = Vocab_None; 
	  LogP weight = ngram.wordProb(toNode->word, context); 

	  setWeightTrans(nodeIndex, toNodeIndex, weight); 
	}
    }

    return true; 
}

Boolean 
Lattice::expandToCompactTrigram(Ngram &ngram, unsigned maxNodes)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::expandToCompactTrigram: "
	     << "starting expansion to compact trigram lattice ...\n";
    }

    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::expandToCompactTrigram: warning: called with unreachable nodes\n";
      }
    }

    for (unsigned i = 0; i < numReachable; i++) {
      NodeIndex nodeIndex = sortedNodes[i];

      if (nodeIndex == initial || nodeIndex == final) {
	continue;
      }
      if (!expandNodeToCompactTrigram(nodeIndex, ngram, maxNodes)) {
        delete [] sortedNodes;
	return false;
      }
    }

    delete [] sortedNodes;
    return true; 
}

/*
 * Expand lattice to implement general LMs
 * Algorithm: replace each node in lattice with copies that are 
 * associated with specific LM contexts. The mapping 
 *	(original node, context) -> new node
 * is constructed incrementally as the lattice is traversed in topological
 * order.
 *
 *	expandMap[startNode, <s>] := newStartNode;
 *	expandMap[endNode, </s>] := newEndNode;
 * 
 *	for oldNode in topological order
 *	    for expandMap[oldNode, c] = newNode
 *		for oldNode2 in successors(oldNode)
 *		    c2 = lmcontext(c + word(oldNode2));
 *		    find or create expandMap[oldNode2, c2] = newNode2;
 *		    word(newNode2) := word(oldNodes2);
 *		    prob(newNode->newNode2) := P(word(newNode2) | c);
 *	    delete oldNode;
 *	    delete expandMap[oldNode]; # to save space
 *
 * As an optimization, we let
 *
 *	lmcontext(c + word(oldNode2)) be the longest context used by the LM
 *		for predicting words following oldNode2 in the lattice, and
 *	BOW(c2) be the backoff weight associated with backing off from the 
 *		full LM context (c + word(oldNode2)) to c2
 *
 * Nodes with NULL or pause are handled by ignoring them in context
 * construction, but otherwise handling (i.e., duplicating) them as above.
 */
Boolean
Lattice::expandNodeToLM(VocabIndex oldIndex, LM &lm, unsigned maxNodes,
			Map2<NodeIndex, VocabContext, NodeIndex> &expandMap)
{
    unsigned insufficientLookaheadNodes = 0;

    Map2Iter2<NodeIndex, VocabContext, NodeIndex>
#ifdef USE_SARRAY_MAP2
				expandIter(expandMap, oldIndex);
#else
				expandIter(expandMap, oldIndex, ngramCompare);
#endif
    NodeIndex *newIndex;
    VocabContext context;

    while ((newIndex = expandIter.next(context))) {

	// node structure might have been moved as a result of insertions
	LatticeNode *oldNode = findNode(oldIndex);
	assert(oldNode != 0);

	unsigned contextLength = Vocab::length(context);

	makeArray(VocabIndex, newContext, contextLength + 2);
	Vocab::copy(&newContext[1], context);

	TRANSITER_T<NodeIndex,LatticeTransition> 
			      transIter(oldNode->outTransitions);
	NodeIndex oldIndex2;
	while (LatticeTransition *oldTrans = transIter.next(oldIndex2)) {
	    LatticeNode *oldNode2 = findNode(oldIndex2);
	    assert(oldNode2 != 0);

	    VocabIndex word = oldNode2->word;

	    // determine context used by LM
	    unsigned usedLength = 0;
	    VocabIndex *usedContext;
	    LogP wordProb;

	    // The old context is extended by word on this node, unless
	    // it is null or pause, which are both ignored by the LM.
	    // Non-events are added to the context but aren't evaluated.
	    if (ignoreWord(word)) {
		usedContext = &newContext[1];
		wordProb = LogP_One;
	    } else if (vocab.isNonEvent(word)) {
		newContext[0] = word;
		usedContext = newContext;
		wordProb = LogP_One;
	    } else {
		newContext[0] = word;
		usedContext = newContext;
	        wordProb = lm.wordProb(word, context);
	    }

	    // find all possible following words and determine maximal context
	    // needed for wordProb computation
	    // skip pause and null nodes up to some depth
	    LatticeFollowIter followIter(*this, *oldNode2);

	    NodeIndex oldIndex3;
	    LatticeNode *oldNode3;
	    LogP weight;
	    while ((oldNode3 = followIter.next(oldIndex3, weight))) {

		// if one of the following nodes has null or pause as word then
		// we don't attempt any further look-ahead and use the maximal
		// context from the LM
		VocabIndex nextWord = oldNode3->word;
		if (ignoreWord(nextWord)) {
		    insufficientLookaheadNodes += 1;
		    nextWord = Vocab_None;
		}
		
		unsigned usedLength2;
		lm.contextID(nextWord, usedContext, usedLength2);

		if (usedLength2 > usedLength) {
		    usedLength = usedLength2;
		}
	    }

	    if (!expandAddTransition(usedContext, usedLength,
					word, wordProb, lm,
					oldIndex2, *newIndex, oldTrans,
					maxNodes, expandMap))
		return false;
	} 
    }

    if (insufficientLookaheadNodes > 0) {
	dout() << "Lattice::expandNodeToLM: insufficient lookahead on "
	       << insufficientLookaheadNodes << " nodes\n";
    }

    // old node (and transitions) is fully replaced by expanded nodes 
    // (and transitions)
    removeNode(oldIndex);

    // save space in expandMap by deleting entries that are no longer used
    expandMap.remove(oldIndex);

    return true;
}

/* 
 * The "compact expansion" version makes two changes
 *
 * - add outgoing transitions to node duplicates with shorter contexts,
 *   not just the maximal context.
 * - only transition out of an expanded node if the LM uses the full context
 *   associated with that node (*** A *** below).
 *   This is possible because of the first change, and it's 
 *   where the savings compared to the general expansion are realized.
 *
 * The resulting algorithm is a generalization of expandToCompactTrigram().
 */
Boolean
Lattice::expandNodeToCompactLM(VocabIndex oldIndex, LM &lm, unsigned maxNodes, 
		        Map2<NodeIndex, VocabContext, NodeIndex> &expandMap)
{
    unsigned insufficientLookaheadNodes = 0;

    Map2Iter2<NodeIndex, VocabContext, NodeIndex>
#ifdef USE_SARRAY_MAP2
				expandIter(expandMap, oldIndex);
#else
				expandIter(expandMap, oldIndex, ngramCompare);
#endif
    NodeIndex *newIndex;
    VocabContext context;

    while ((newIndex = expandIter.next(context))) {

	// node structure might have been moved as a result of insertions
	LatticeNode *oldNode = findNode(oldIndex);
	assert(oldNode != 0);

	Boolean ignoreOldNodeWord = ignoreWord(oldNode->word);

	unsigned contextLength = Vocab::length(context);

	makeArray(VocabIndex, newContext, contextLength + 2);
	Vocab::copy(&newContext[1], context);

	TRANSITER_T<NodeIndex,LatticeTransition> 
			      transIter(oldNode->outTransitions);
	NodeIndex oldIndex2;
	while (LatticeTransition *oldTrans = transIter.next(oldIndex2)) {
	    LatticeNode *oldNode2 = findNode(oldIndex2);
	    assert(oldNode2 != 0);

	    VocabIndex word = oldNode2->word;

	    // Find the context length used for LM transition oldNode->oldNode2.
	    // Because each LM context gets its own node duplicate
	    // in this expansion algorithm, we only need to process
	    // transitions where the LM context fully matches the context 
	    // associated with the specific oldNode duplicate we're working on.
	    unsigned usedLength0;
	    lm.contextID(word, context, usedLength0);

	    // if the node we're coming from has a real word then
	    // there is no point in using a null context
	    // (so we can collapse null and unigram contexts)
	    if (usedLength0 == 0 && !ignoreOldNodeWord) {
		usedLength0 = 1;
	    }

	    // *** A ***
	    if (!ignoreWord(word) &&
		context[0] != vocab.ssIndex() &&
		usedLength0 != contextLength)
	    {
		continue;
	    }

	    // determine context used by LM
	    VocabIndex *usedContext;
	    LogP wordProb;

	    // The old context is extended by word on this node, unless
	    // it is null or pause, which are both ignored by the LM.
	    // Non-events are added to the context by aren't evaluated.
	    if (ignoreWord(word)) {
		usedContext = &newContext[1];
		wordProb = LogP_One;
	    } else if (vocab.isNonEvent(word)) {
		newContext[0] = word;
		usedContext = newContext;
		wordProb = LogP_One;
	    } else {
		newContext[0] = word;
		usedContext = newContext;
	        wordProb = lm.wordProb(word, context);
	    }

	    // find all possible following words and compute LM context
	    // used for their prediction.
	    // then create duplicate nodes for each context length
	    // needed for wordProb computation
	    LatticeFollowIter followIter(*this, *oldNode2);

	    NodeIndex oldIndex3;
	    LatticeNode *oldNode3;
	    LogP weight;
	    unsigned lastUsedLength = (unsigned)-1;
	    while ((oldNode3 = followIter.next(oldIndex3, weight))) {

		unsigned usedLength;

		// if one of the following nodes has null or pause as word then
		// we have to back off to null context to make sure 
		// further expansion can connect above at *** A ***
		VocabIndex nextWord = oldNode3->word;
		if (ignoreWord(nextWord)) {
		    insufficientLookaheadNodes += 1;
		    usedLength = 0;
		} else {
		    lm.contextID(nextWord, usedContext, usedLength);
		}

		// if the node we're going to has a real word then
		// there is no point in using a null context (same as above)
		if (usedLength == 0 && !ignoreWord(word)) {
		    usedLength = 1;
		}

		// optimization: no need to re-add transition if usedLength
		// is same as on previous pass through
		if (usedLength == lastUsedLength) {
		    continue;
		} else {
		    lastUsedLength = usedLength;
		}

		if (!expandAddTransition(usedContext, usedLength,
					    word, wordProb, lm,
					    oldIndex2, *newIndex, oldTrans,
					    maxNodes, expandMap))
		    return false;
	    }

	    // transitions to the final node have to be handled specially 
	    // because the above loop won't apply to it, since the final node
	    // has no outgoing transitions!
	    if (oldIndex2 == final) {
		// by convention, the final node always uses context of length 0
		if (!expandAddTransition(usedContext, 0,
					    word, wordProb, lm,
					    oldIndex2, *newIndex, oldTrans,
					    maxNodes, expandMap))
		    return false;
	    }
	} 
    }

    if (insufficientLookaheadNodes > 0) {
	dout() << "Lattice::expandNodeToCompactLM: insufficient lookahead on "
	       << insufficientLookaheadNodes << " nodes\n";
    }

    // old node (and transitions) is fully replaced by expanded nodes 
    // (and transitions)
    removeNode(oldIndex);

    // save space in expandMap by deleting entries that are no longer used
    expandMap.remove(oldIndex);

    return true;
}

/*
 * Helper for expandNodeToLM() and expandNodeToCompactLM()
 */
Boolean
Lattice::expandAddTransition(VocabIndex *usedContext, unsigned usedLength,
		    VocabIndex word, LogP wordProb, LM &lm,
		    NodeIndex oldIndex2, NodeIndex newIndex,
		    LatticeTransition *oldTrans, unsigned maxNodes,
		    Map2<NodeIndex, VocabContext, NodeIndex> &expandMap)
{
    LogP transProb;
    if (ignoreWord(word)) {
	transProb = LogP_One;
    } else {
	transProb = wordProb;
    }

   
    if (!noBackoffWeights && usedContext[0] != vocab.seIndex()) {
	// back-off weight to truncate the context
	// (needs to be called before context truncation below)
	// Note: we check above that this is not a backoff weight after </s>,
	// which some LMs contain but should be ignored for our purposes
	// since lattices all end implictly in </s>.
	transProb += lm.contextBOW(usedContext, usedLength);
    }

    // truncate context to what LM uses
    VocabIndex saved = usedContext[usedLength];
    usedContext[usedLength] = Vocab_None;

    Boolean found;
    NodeIndex *newIndex2 =
	expandMap.insert(oldIndex2, (VocabContext)usedContext, found);

    if (!found) {
	LatticeNode *oldNode2 = findNode(oldIndex2);
	assert(oldNode2 != 0);

	// create new node copy and store it in map
	*newIndex2 = dupNode(word, oldNode2->flags, oldNode2->htkinfo);

	if (maxNodes > 0 && getNumNodes() > maxNodes) {
	    dout() << "Lattice::expandAddTransition: "
		   << "aborting with number of nodes exceeding "
		   << maxNodes << endl;
	    return false;
	}
    }

    LatticeTransition newTrans(transProb, oldTrans->flags);
    insertTrans(newIndex, *newIndex2, newTrans, 0);

    // restore full context
    usedContext[usedLength] = saved;

    return true;
}

Boolean 
Lattice::expandToLM(LM &lm, unsigned maxNodes, Boolean compact)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::expandToLM: "
 	     << "starting " << (compact ? "compact " : "")
	     << "expansion to general LM (maxNodes = " << maxNodes
 	     << ") ...\n";
    }

    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::expandToLM: warning: called with unreachable nodes\n";
      }
    }

    Map2<NodeIndex, VocabContext, NodeIndex> expandMap;

    // prime expansion map with initial/final nodes
    LatticeNode *startNode = findNode(initial);
    assert(startNode != 0);
    VocabIndex newStartIndex = dupNode(startNode->word, startNode->flags,
							startNode->htkinfo);

    VocabIndex context[2];
    context[1] = Vocab_None;
    context[0] = vocab.ssIndex();
    *expandMap.insert(initial, context) = newStartIndex;

    LatticeNode *endNode = findNode(final);
    assert(endNode != 0);
    VocabIndex newEndIndex = dupNode(endNode->word, endNode->flags,
							endNode->htkinfo);

    context[0] = Vocab_None;
    *expandMap.insert(final, context) = newEndIndex;

    for (unsigned i = 0; i < numReachable; i++) {
      NodeIndex nodeIndex = sortedNodes[i];

      if (nodeIndex == final) {
	removeNode(final);
      } else if (compact) {
        if (!expandNodeToCompactLM(nodeIndex, lm, maxNodes, expandMap)) {
	  delete [] sortedNodes;
	  return false;
	}
      } else {
        if (!expandNodeToLM(nodeIndex, lm, maxNodes, expandMap)) {
	  delete [] sortedNodes;
	  return false;
        }
      }
    }

    initial = newStartIndex;
    final = newEndIndex;

    delete [] sortedNodes;
    return true; 
}

