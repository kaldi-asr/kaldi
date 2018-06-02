/*
 * Lattice.cc --
 *	Word lattices
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1997-2015 SRI International, 2011-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/Lattice.cc,v 1.146 2016/06/17 00:11:06 victor Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <math.h>

#include "Lattice.h"

#include "SArray.cc"
#include "LHash.cc"
#include "Array.cc"
#include "WordAlign.h"                  /* for *_COST constants */
#include "NBest.h"			/* for phoneSeparator defn */

const char *LATTICE_OR		= "or";
const char *LATTICE_CONCATE	= "concatenate";
const char *LATTICE_NONAME	= "***NONAME***";

const char *NullNodeName	= "NULL";

const LogP LogP_PseudoZero	= -100;

#define DebugPrintFatalMessages         0 

#define DebugPrintFunctionality         1 
// for large functionality listed in options of the program
#define DebugPrintOutLoop               2
// for out loop of the large functions or small functions
#define DebugPrintInnerLoop             3
// for inner loop of the large functions or outloop of small functions

#ifdef INSTANTIATE_TEMPLATES
#ifdef USE_SARRAY
INSTANTIATE_SARRAY(NodeIndex,LatticeTransition);
#else
INSTANTIATE_LHASH(NodeIndex,LatticeTransition);
#endif
INSTANTIATE_LHASH(NodeIndex,LatticeNode);
INSTANTIATE_ARRAY(HTKWordInfo *);
#endif

#if defined(_MSC_VER) && _MSC_VER < 1800
# define nextafter(x,y)    _nextafter(x,y)
#endif


/* **************************************************
   the queue related methods
   ************************************************** */

NodeQueue::~NodeQueue()
{
    while (is_empty() != true) {
	(void) popNodeQueue();
    }
}

NodeIndex NodeQueue::popNodeQueue()
{
    if (is_empty() == true) {
        dout() << "popNodeQueue() on an empty queue\n";
        exit(-1); 
    }

    QueueItem *pt = queueHead; 
    queueHead = queueHead->next;
    if (queueHead == 0) { 
	queueTail = 0;
    }
    NodeIndex retval = pt->item;
    delete pt;
    return retval;
} 

QueueItem * NodeQueue::popQueueItem()
{
    if (is_empty() == true) {
        dout() << "popQueueItem() on an empty queue\n";
        exit(-1); 
    }

    QueueItem *pt = queueHead; 
    queueHead = queueHead->next;

    return pt;
} 
    
Boolean NodeQueue::addToNodeQueue(NodeIndex nodeIndex, 
				  unsigned level, LogP weight)
{
    QueueItem *pt = new QueueItem(nodeIndex, level, weight);
    assert(pt != 0); 

    if (is_empty() == true) {
        queueHead = queueTail = pt;
    } else {
        queueTail->next = pt;
	queueTail = pt;
    }

    return true;
}

// add to NodeQueue if the previous element is not the same
// as the element to be added.
Boolean 
NodeQueue::addToNodeQueueNoSamePrev(NodeIndex nodeIndex, 
				    unsigned level, LogP weight)
{
    if (is_empty() == false && nodeIndex == queueTail->item) {
        if (debug(DebugPrintOutLoop)) {
	  dout() << "In addToNodeQueueNoSamePrev: skip the current nodeIndex ("
		 << nodeIndex << ")" << endl;
	}
        return true;
    }

    if (debug(DebugPrintOutLoop)) {
      dout() << "In addToNodeQueueNoSamePrev: add the current nodeIndex ("
	     << nodeIndex << ")" << endl;
    }
    QueueItem *pt = new QueueItem(nodeIndex, level, weight);
    assert(pt != 0); 

    if (is_empty() == true) {
        queueHead = queueTail = pt;
    } else {
        queueTail->next = pt;
	queueTail = pt;
    }

    return true;
}

Boolean NodeQueue::inNodeQueue(NodeIndex nodeIndex)
{
  
    QueueItem *pt = queueHead;

    while (pt != 0) {
	if (pt->item == nodeIndex) {
	    return true; 
	}
	pt = pt->next;
    }

    return false;
}

QueueItem::QueueItem(NodeIndex nodeIndex, unsigned clevel, LogP cweight)
{
    item = nodeIndex;
    level = clevel;
    weight = cweight; 
    next = 0; 
}


/* ************************************************************
   function definitions for class LatticeFollowIter
   ************************************************************ */

LatticeFollowIter::LatticeFollowIter(Lattice &lat, LatticeNode &node,
				     LHash<NodeIndex, LogP> *useVisitedNodes,
				     LogP startWeight)
    : lat(lat), transIter(node.outTransitions), subFollowIter(0),
      startWeight(startWeight), visitedNodes(useVisitedNodes)
{
    /*
     * Use shared visitedNodes table if passed from our recursive invoker,
     * otherwise create our own.
     */
    if (useVisitedNodes == 0) {
	visitedNodes = new LHash<NodeIndex, LogP>;
	assert(visitedNodes != 0);

	freeVisitedNodes = true;
    } else {
	freeVisitedNodes = false;
    }
}

LatticeFollowIter::~LatticeFollowIter()
{
    delete subFollowIter;

    if (freeVisitedNodes) {
	delete visitedNodes;
    }
}

void
LatticeFollowIter::init()
{
    /* 
     * terminate recursive iteration, if any
     */
    delete subFollowIter;
    subFollowIter = 0;

    /*
     * restart top-level iteration
     */
    transIter.init();

    /*
     * forget visited nodes 
     */
    if (freeVisitedNodes) {
	delete visitedNodes;
	visitedNodes = new LHash<NodeIndex, LogP>;
	assert(visitedNodes != 0);
    }
}

LatticeNode *
LatticeFollowIter::next(NodeIndex &followIndex, LogP &weight)
{
    LatticeNode *followNode;

    /*
     * if a recursive iteration is pending, continue it
     */
    if (subFollowIter != 0) {
	followNode = subFollowIter->next(followIndex, weight);

	if (followNode) {
	    return followNode;
	} else {
	    delete subFollowIter;
	    subFollowIter = 0;
	}
    }
	    
    /*
     * otherwise proceed with the next transition at this node
     */
    LatticeTransition *trans;
    Boolean visitedBefore;

    do {
	/*
	 * find next follow node that hasn't already been visited
	 */
	do {
	    trans = transIter.next(followIndex);

	    if (trans == 0) break;

	    /* record that node has been visited */
	    LogP *oldWeight = visitedNodes->insert(followIndex, visitedBefore);

	    /* if previous visit had lower weight, visit again */
	    if (visitedBefore && *oldWeight < startWeight + trans->weight) {
		visitedBefore = false;
	    }
		    
	    if (!visitedBefore) {
		/* record weight for this visit for future reference */
		*oldWeight = startWeight + trans->weight;
	    }
	} while (visitedBefore);

	if (trans == 0) {
	    /* no more transitions to follow, so stop */
	    break;
	} else {
	    LatticeNode *followNode = lat.findNode(followIndex);
	    assert(followNode != 0);

	    if (!lat.ignoreWord(followNode->word)) {
		weight = startWeight + trans->weight;
		return followNode;
	    } else {
		/*
		 * recursively iterate over following nodes, sharing
		 * visitedNodes table to avoid double visits and infinite loops.
		 */
		delete subFollowIter;
		subFollowIter =
		    new LatticeFollowIter(lat, *followNode, visitedNodes,
					  startWeight + trans->weight);
		assert(subFollowIter != 0);

		LatticeNode *result = subFollowIter->next(followIndex, weight);
		if (result) {
		    return result;
		}
	    }
	}
    } while (1);

    /* end of iteration reached */
    return 0;
}


/* ************************************************************
   function definitions for class LatticeNode
   ************************************************************ */
LatticeNode::LatticeNode()
    : flags(0), word(Vocab_None), posterior(LogP_Zero), htkinfo(0)
{
}

/* ************************************************************
   function definitions for class LatticeTransition
   ************************************************************ */

/* ************************************************************
   function definitions for class Lattice
   ************************************************************ */

Lattice::Lattice(Vocab &vocab, const char *name)
    : noBackoffWeights(false), useUnk(false), keepUnk(false),
      limitIntlogs(false), printSentTags(false),
      vocab(vocab), ignoreVocab(vocab), maxIndex(0),
      name(strdup(name != 0 ? name : LATTICE_NONAME)),
      duration(0.0), initial(NoNode), final(NoNode), top(true)
{
    /*
     * Default is to ignore pause nodes (only)
     */
    if (vocab.pauseIndex() != Vocab_None) {
	ignoreVocab.addWord(vocab.pauseIndex());
    }
}

Lattice::Lattice(Vocab &vocab, const char *name, SubVocab &ignore)
    : noBackoffWeights(false), useUnk(false), keepUnk(false),
      limitIntlogs(false), printSentTags(false),
      vocab(vocab), ignoreVocab(vocab), maxIndex(0),
      name(strdup(name != 0 ? name : LATTICE_NONAME)),
      duration(0.0), initial(NoNode), final(NoNode), top(true)
{
    /*
     * Copy words-to-be-ignored locally
     */
    VocabIter ignoreIter(ignore);
    VocabIndex ignoreWord;
    while (ignoreIter.next(ignoreWord)) {
	ignoreVocab.addWord(ignoreWord);
    }
}

Lattice::~Lattice()
{
    free((void *)name);
    name = 0;

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	node->~LatticeNode();
    }

    for (unsigned i = 0; i < htkinfos.size(); i ++) {
	delete htkinfos[i];
    }
}

const char *
Lattice::setName(const char *newname)
{
    if (name) {
	free((void *)name);
    }
    name = strdup(newname);
    assert(name != 0);
    return name;
}

VocabString
Lattice::getWord(VocabIndex word)
{
    if (word == Vocab_None) {
	return NullNodeName;
    } else {
	return vocab.getWord(word);
    }
}

/* To insert a node with name *word in the NodeLattice */
Boolean 
Lattice::insertNode(const char *word, NodeIndex nodeIndex)
{
    VocabIndex windex;

    if (strcmp(word, NullNodeName) == 0) {
	windex = Vocab_None;
    } else if (useUnk || keepUnk) {
	windex = vocab.getIndex(word, vocab.unkIndex());
    } else {
	windex = vocab.addWord(word);
    }

    LatticeNode *node = nodes.insert(nodeIndex);

    node->word = windex;

    if (maxIndex <= nodeIndex) {
	maxIndex = nodeIndex + 1;
    }

    return true; 
}

/* duplicate a node with a same word name without making 
   any commitment on edges
   */
NodeIndex
Lattice::dupNode(VocabIndex windex, unsigned markedFlag, HTKWordInfo *htkinfo) 
{
    LatticeNode *node = nodes.insert(maxIndex);

    node->word = windex;
    node->flags = markedFlag; 
    node->htkinfo = htkinfo;

    return maxIndex++; 
}

/* remove the node and all of its edges (incoming and outgoing)
   and check to see whether it makes the graph disconnected
*/
Boolean 
Lattice::removeNode(NodeIndex nodeIndex) 
{
    LatticeNode *node = findNode(nodeIndex);
    if (!node) {
      if (debug(DebugPrintOutLoop)) {
	dout() << " In Lattice::removeNode: undefined node in graph " 
	       << nodeIndex << endl;
      }
      return false; 
    }

    // delete incoming transitions -- need do only the fromNode->outTransitions
    // node->inTransition will be freed shortly
    TRANSITER_T<NodeIndex,LatticeTransition> 
      inTransIter(node->inTransitions);
    NodeIndex fromNodeIndex;
    while (inTransIter.next(fromNodeIndex)) {
	LatticeNode *fromNode = findNode(fromNodeIndex);
	assert(fromNode != 0);
	fromNode->outTransitions.remove(nodeIndex);
    } 

    // delete outgoing transitions -- need do only the toNode->inTransitions
    // node->outTransition will be freed shortly
    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIter(node->outTransitions);
    NodeIndex toNodeIndex;
    while (outTransIter.next(toNodeIndex)) {
	LatticeNode *toNode = findNode(toNodeIndex);
	assert(toNode != 0);
	toNode->inTransitions.remove(nodeIndex);
    } 

    // remove this node
    LatticeNode removedNode;
    nodes.remove(nodeIndex, &removedNode);
    if (debug(DebugPrintOutLoop)) {
      dout() << "In Lattice::removeNode: remove node " << nodeIndex << endl; 
    }

    if (nodeIndex == initial) {
	initial = NoNode;
    } 
    if (nodeIndex == final) {
	final = NoNode;
    } 
    
    return true;
}

void
Lattice::removeAll()
{
    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;

    // remove all nodes
    // (this also removes all transitions)
    while (nodeIter.next(nodeIndex)) {
	removeNode(nodeIndex);
    }

    maxIndex = 0;
}

// try to find a transition between the two given nodes.
// return 0, if there is no transition.
// return 1, if there is a transition.
LatticeTransition *
Lattice::findTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex)
{
    LatticeNode *toNode = nodes.find(toNodeIndex); 
    if (!toNode) {
      // the end node does not exist.
      return 0;
    }

    LatticeNode *fromNode = nodes.find(fromNodeIndex);
    if (!fromNode) {
      // the begin node does not exist.
      return 0;
    }

    LatticeTransition *trans = toNode->inTransitions.find(fromNodeIndex); 

    if (!trans) {
      // the transition does not exist.
      return 0;
    }

#ifdef DEBUG
    LatticeTransition *outTrans = fromNode->outTransitions.find(toNodeIndex);
    if (!outTrans) {
      // asymmetric transition.
      if (debug(DebugPrintFatalMessages)) {
	dout() << "nonFatal error in Lattice::findTrans: asymmetric transition."
	       << endl;
      }
      return 0;
    }
#endif

    return trans;
}


// to insert a transition between two nodes. If the transition exists already
// union their weights. maxAdd == 0, take the max of the existing and the new 
// weights; maxAdd == 1, take the added weights of the two (notice that the 
// weights are in log scale, so log(x+y) = logx + log (y/x + 1)
Boolean
Lattice::insertTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex, 
			     const LatticeTransition &t, Boolean maxAdd)  
{

    LatticeNode *toNode = nodes.find(toNodeIndex); 
    if (!toNode) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal error in Lattice::insertTrans: can't find toNode"
		 << toNodeIndex << endl;
	}
	exit(-1); 
    }

    LatticeNode *fromNode = nodes.find(fromNodeIndex);
    if (!fromNode) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal error in Lattice::insertTrans: can't find fromNode"
		 << fromNodeIndex << endl;
	}
	exit(-1); 
    }

    Boolean found;
    LatticeTransition *trans =
			toNode->inTransitions.insert(fromNodeIndex, found); 

    if (!found) {
      // it's a new edge; 
      *trans = t;
    } else {
      // there is already an edge 
      if (!maxAdd) {
        trans->weight = unionWeights(trans->weight, t.weight); 
      } else { 
        trans->weight = AddLogP(trans->weight, t.weight); 
      }
      trans->flags = trans->flags | t.flags |
			(!trans->getFlag(pauseTFlag) ? directTFlag : 0);
    }

    // duplicate intransition as outtransition
    *fromNode->outTransitions.insert(toNodeIndex) = *trans; 
    
    return true;
}

Boolean
Lattice::setWeightTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex, 
			LogP weight)  
{

    LatticeNode *toNode = nodes.find(toNodeIndex); 
    if (!toNode) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal error in Lattice::setWeightTrans: can't find toNode" 
		 << toNodeIndex << endl;
	}
	exit(-1); 
    }

    LatticeTransition * trans = toNode->inTransitions.find(fromNodeIndex); 
    if (!trans) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal error in Lattice::setWeightTrans: can't find inTrans(" 
		 << fromNodeIndex << "," << toNodeIndex << ")\n";
	}
	exit(-1); 
    }

    trans->weight = weight;

    LatticeNode *fromNode = nodes.find(fromNodeIndex);
    if (!fromNode) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal error in Lattice::setWeightTrans: can't find fromNode" 
		 << fromNodeIndex << endl;
	}
	exit(-1); 
    }

    trans = fromNode->outTransitions.find(toNodeIndex); 
    if (!trans) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal error in Lattice::setWeightTrans: can't find outTrans (" 
		 << fromNodeIndex << "," << toNodeIndex << ")\n";
	}
	exit(-1); 
    }

    trans->weight = weight;

    return true;

}

Boolean 
Lattice::removeTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex)
{
    LatticeNode *fromNode = nodes.find(fromNodeIndex);
    
    if (!fromNode) {
        if (debug(DebugPrintOutLoop)) {
	  dout() << "nonFatal error in Lattice::removeTrans:\n" 
		 << "   undefined source node in transition " 
		 << "(" << fromNodeIndex << ", " << toNodeIndex << ")\n";
	}
	return false;
    }

    if (!fromNode->outTransitions.remove(toNodeIndex)) {
        if (debug(DebugPrintOutLoop)) {
	  dout() << "nonFatal error in Lattice::removeTrans:\n" 
		 << "   no outTrans (" << fromNodeIndex << ", " 
		 << toNodeIndex << ")\n";
	}
	return false;
    }

    LatticeNode *toNode = nodes.find(toNodeIndex);
    
    if (!toNode) {
        if (debug(DebugPrintOutLoop)) {
	  dout() << "nonFatal error in Lattice::removeTrans:\n" 
		 << "undefined sink node " << toNodeIndex << endl;
	}
	return false;
    }

    if (!toNode->inTransitions.remove(fromNodeIndex)) {
        if (debug(DebugPrintOutLoop)) {
	  dout() << "nonFatal error in Lattice::removeTrans:\n" 
		 << "   no inTrans (" << fromNodeIndex << ", " 
		 << toNodeIndex << ")\n";
	}
	return false;
    }

    return true;    
}   


void 
Lattice::markTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex, unsigned flag)
{
    LatticeNode *fromNode = nodes.find(fromNodeIndex); 
    if (!fromNode) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal Error in Lattice::markTrans: can't find source node" 
		 << fromNodeIndex << endl;
	}
	exit(-1); 
    }

    LatticeTransition * trans = fromNode->outTransitions.find(toNodeIndex); 
    if (!trans) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal Error in Lattice::markTrans: can't find outTrans ("
		 << fromNodeIndex << ", " << toNodeIndex << ")\n"; 
	}
	exit(-1); 
    }
    trans->flags |= flag; 

     LatticeNode *toNode = nodes.find(toNodeIndex); 
    if (!toNode) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal Error in Lattice::markTrans: can't find sink node" 
		 << toNodeIndex << endl;
	}
	exit(-1); 
    }

    trans = toNode->inTransitions.find(fromNodeIndex); 
    if (!trans) {
        if (debug(DebugPrintFatalMessages)) {
	  dout() << "Fatal Error in Lattice::markTrans: can't find inTrans ("
		 << toNodeIndex << ", " << toNodeIndex << ")\n"; 
	}
	exit(-1); 
    }
    trans->flags |= flag; 
}

void
Lattice::clearMarkOnAllNodes(unsigned flag) {
    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
      node->flags &= ~flag;
    }
}

void
Lattice::dumpFlags() {
    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
      dout() << "node " << nodeIndex << " flags = " << node->flags << endl;

      TRANSITER_T<NodeIndex,LatticeTransition> outIter(node->outTransitions);
      NodeIndex outIndex; 
      while (LatticeTransition *outTrans = outIter.next(outIndex)) {
        dout() << "  trans -> " << outIndex 
	       << " flags = " << outTrans->flags << endl;
      }

      TRANSITER_T<NodeIndex,LatticeTransition> inIter(node->inTransitions);
      NodeIndex inIndex; 
      while (LatticeTransition *inTrans = inIter.next(inIndex)) {
        dout() << "  trans <- " << inIndex 
	       << " flags = " << inTrans->flags << endl;
      }
    }
}

void
Lattice::clearMarkOnAllTrans(unsigned flag) {

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
      TRANSITER_T<NodeIndex,LatticeTransition> outIter(node->outTransitions);
      NodeIndex outIndex; 
      while (LatticeTransition *outTrans = outIter.next(outIndex)) {
	outTrans->flags &= ~flag;
      }

      TRANSITER_T<NodeIndex,LatticeTransition> inIter(node->inTransitions);
      NodeIndex inIndex; 
      while (LatticeTransition *inTrans = inIter.next(inIndex)) {
	inTrans->flags &= ~flag;
      }
    }
}

// compute 'or' operation on two input lattices
// using implantLattice, 
// assume that the initial state of this lattice is empty.

Boolean
Lattice::latticeOr(Lattice &lat1, Lattice &lat2)
{
    initial = 1;
    final = 0;

    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::latticeOr: doing OR on "
	       << lat1.getName() << " and " << lat2.getName() << "\n"; 
    }

    // if name is default, inherit if from the input lattice
    if (strcmp(name, LATTICE_NONAME) == 0) {
	free((void *)name);
	name = strdup(lat1.name);
	assert(name != 0);
    }
    // inherit HTK header info from first lattice
    htkheader = lat1.htkheader;

    LatticeNode *newNode = nodes.insert(1);
    newNode->word = Vocab_None;
    newNode->flags = 0;

    // get initial time as the minimum of the start times of the two lattices
    LatticeNode *initialNode1 = lat1.findNode(lat1.initial);
    LatticeNode *initialNode2 = lat2.findNode(lat2.initial);
    if ((initialNode1->htkinfo != 0 &&
	 initialNode1->htkinfo->time != HTK_undef_float) ||
        (initialNode2->htkinfo != 0 &&
	 initialNode2->htkinfo->time != HTK_undef_float))
    {
	HTKWordInfo *linkinfo = new HTKWordInfo;
	assert(linkinfo != 0);
	newNode->htkinfo = htkinfos[htkinfos.size()] = linkinfo;

	if (initialNode1->htkinfo == 0 ||
	    initialNode1->htkinfo->time == HTK_undef_float ||
	    (initialNode2->htkinfo != 0 &&
	     initialNode2->htkinfo->time != HTK_undef_float &&
	     initialNode1->htkinfo->time > initialNode2->htkinfo->time))
	{
	    linkinfo->time = initialNode2->htkinfo->time;
	} else {
	    linkinfo->time = initialNode1->htkinfo->time;
	} 
    }

    newNode = nodes.insert(0);
    newNode->word = Vocab_None;
    newNode->flags = 0;

    // get final time as the maximum of the end times of the two lattices
    LatticeNode *finalNode1 = lat1.findNode(lat1.final);
    LatticeNode *finalNode2 = lat2.findNode(lat2.final);
    if ((finalNode1->htkinfo != 0 &&
	 finalNode1->htkinfo->time != HTK_undef_float) ||
        (finalNode2->htkinfo != 0 &&
	 finalNode2->htkinfo->time != HTK_undef_float))
    {
	HTKWordInfo *linkinfo = new HTKWordInfo;
	assert(linkinfo != 0);
	newNode->htkinfo = htkinfos[htkinfos.size()] = linkinfo;

	if (finalNode1->htkinfo == 0 ||
	    finalNode1->htkinfo->time == HTK_undef_float ||
	    (finalNode2->htkinfo != 0 &&
	     finalNode2->htkinfo->time != HTK_undef_float &&
	     finalNode1->htkinfo->time < finalNode2->htkinfo->time))
	{
	    linkinfo->time = finalNode2->htkinfo->time;
	} else {
	    linkinfo->time = finalNode1->htkinfo->time;
	} 
    }

    newNode = nodes.insert(2);
    newNode->word = Vocab_None;
    newNode->flags = 0;

    newNode = nodes.insert(3);
    newNode->word = Vocab_None;
    newNode->flags = 0;

    maxIndex = 4;

    LatticeTransition t(0, 0);
    insertTrans(1, 2, t);
    insertTrans(1, 3, t);
    insertTrans(2, 0, t);
    insertTrans(3, 0, t);
    
    if (!(implantLattice(2, lat1))) return false;
    if (!(implantLattice(3, lat2))) return false;

    limitIntlogs = lat1.limitIntlogs || lat2.limitIntlogs;

    return true;
}

Boolean
Lattice::latticeCat(Lattice &lat1, Lattice &lat2, float interSegmentTime)
{
    initial = 1;
    final = 0;

    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::latticeCat: doing CONCATENATE on "
	       << lat1.getName() << " and " << lat2.getName() << "\n"; 
    } 

    // if name is default, inherit if from the input lattice
    if (strcmp(name, LATTICE_NONAME) == 0) {
	free((void *)name);
	name = strdup(lat1.name);
	assert(name != 0);
    }
    // inherit HTK header info from first lattice
    htkheader = lat1.htkheader;

    LatticeNode *newNode = nodes.insert(1);
    newNode->word = Vocab_None;
    newNode->flags = 0;

    // get initial time from first lattice
    LatticeNode *initialNode1 = lat1.findNode(lat1.initial);
    if (initialNode1->htkinfo != 0 &&
	initialNode1->htkinfo->time != HTK_undef_float)
    {
	HTKWordInfo *linkinfo = new HTKWordInfo;
	assert(linkinfo != 0);
	newNode->htkinfo = htkinfos[htkinfos.size()] = linkinfo;

	linkinfo->time = initialNode1->htkinfo->time;
    }

    // get final time from first lattice
    float finalTime1 = 0.0;
    LatticeNode *finalNode1 = lat1.findNode(lat1.final);
    if (finalNode1->htkinfo != 0 &&
	finalNode1->htkinfo->time != HTK_undef_float)
    {
	finalTime1 = finalNode1->htkinfo->time;
    }

    newNode = nodes.insert(0);
    newNode->word = Vocab_None;
    newNode->flags = 0;

    // get final time from second lattice
    LatticeNode *finalNode2 = lat2.findNode(lat2.final);
    if (finalNode2->htkinfo != 0 &&
	finalNode2->htkinfo->time != HTK_undef_float)
    {
	HTKWordInfo *linkinfo = new HTKWordInfo;
	assert(linkinfo != 0);
	newNode->htkinfo = htkinfos[htkinfos.size()] = linkinfo;

	// add interSegmentTime and final time, which are zero if no times were found
	linkinfo->time = finalNode2->htkinfo->time + interSegmentTime + finalTime1;
    }

    if (interSegmentTime != 0.0) {
	newNode = nodes.insert(2);
	newNode->word = Vocab_None;
	newNode->flags = 0;
	
	newNode = nodes.insert(3);
	newNode->word = vocab.pauseIndex();
	HTKWordInfo *linkinfo = new HTKWordInfo;
	assert(linkinfo != 0);
	newNode->htkinfo = htkinfos[htkinfos.size()] = linkinfo;
	linkinfo->time = finalTime1;
	{
	    makeArray(char, pause, strlen(phoneSeparator)*2 + 12);
	    sprintf(pause,"%s-,%.2f%s",
		      phoneSeparator, interSegmentTime, phoneSeparator);
	    linkinfo->div = strdup(pause);
	}
	newNode->flags = 0;

	newNode = nodes.insert(4);
	newNode->word = Vocab_None;
	newNode->flags = 0;
	
	maxIndex = 5;

	LatticeTransition t(0, 0);;
	insertTrans(1, 2, t);
	insertTrans(2, 3, t);
	insertTrans(3, 4, t);
	insertTrans(4, 0, t);
	
	if (!(implantLattice(2, lat1))) return false;
	
	if (!(implantLattice(4, lat2, finalTime1+interSegmentTime)))
	    return false;
    } else {
	newNode = nodes.insert(2);
	newNode->word = Vocab_None;
	newNode->flags = 0;
	
	newNode = nodes.insert(3);
	newNode->word = Vocab_None;
	newNode->flags = 0;
	
	maxIndex = 4;

	LatticeTransition t(0, 0);;
	insertTrans(1, 2, t);
	insertTrans(2, 3, t);
	insertTrans(3, 0, t);
	
	if (!(implantLattice(2, lat1, finalTime1))) return false;
	if (!(implantLattice(3, lat2, finalTime1))) return false;
    }

    limitIntlogs = lat1.limitIntlogs || lat2.limitIntlogs;

    return true;
}

// add a string of words to the lattice
// (similar to MultiAlign::addWords())
void
Lattice::addWords(const VocabIndex *words, Prob prob, Boolean addPauses)
{
    NodeIndex lastNode = initial;
    LatticeTransition trans(ProbToLogP(prob), 0);

    for (unsigned i = 0; words[i] != Vocab_None; i ++) {
	NodeIndex wordNode = dupNode(words[i], 0);

	insertTrans(lastNode, wordNode, trans);

	if (addPauses) {
	    NodeIndex pauseNode = dupNode(vocab.pauseIndex(), 0);

	    insertTrans(lastNode, pauseNode, trans);

	    trans.weight = LogP_One;
	    insertTrans(pauseNode, wordNode, trans);
	}

	trans.weight = LogP_One;
	lastNode = wordNode;
    }

    insertTrans(lastNode, final, trans);

    if (addPauses) {
	NodeIndex pauseNode = dupNode(vocab.pauseIndex(), 0);

	insertTrans(lastNode, pauseNode, trans);

	trans.weight = LogP_One;
	insertTrans(pauseNode, final, trans);
    }
}

// reading in recursively defined PFSGs
// 

Boolean
Lattice::readRecPFSGs(File &file)
{
    if (debug(DebugPrintFunctionality)) {
      dout()  << "Lattice::readRecPFSGs: "
	      << "reading in dag PFSG......\n"; 
    }

    NodeQueue nodeQueue; 

    if (!readPFSG(file)) {
      return false;
    }

    while (file.fgetc() != EOF) {
      file.fseek(-1, SEEK_CUR); 

      Lattice *lat = new Lattice(vocab); 
      if (!lat->readPFSG(file)) {
	if (debug(DebugPrintFunctionality)) {
	  dout()  << "Lattice::readRecPFSGs: "
		  << "failed in reading in dag PFSGs.\n";
	}
	dout() << "Test: 0\n"; 
	delete lat;
	return false; 
      }
      
      VocabIndex word = vocab.addWord(lat->getName());
      Lattice **pt = subPFSGs.insert(word);
      *pt = lat; 

      if (debug(DebugPrintOutLoop)) {
	dout()  << "Lattice::readRecPFSGs: got "
		<< "sub-PFSG (" << (*pt)->getName() << ").\n";
      }      
    }

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
      // for non-terminal node, i.e., sub-PFSGs, get its non
      // recursive equivalent PFSG.
      dout() << "Processing node (" << getWord(node->word) << ")\n";
      Lattice ** pt = subPFSGs.find(node->word);
      if (!pt) {
	// terminal node
	dout() << "It's terminal node!\n";
	continue;
      }
      Lattice * lat = *pt; 
      dout() << "It's NON-terminal node!\n";
      nodeQueue.addToNodeQueue(nodeIndex); 
    }

    if (debug(DebugPrintOutLoop)) {
      dout()  << "Lattice::readRecPFSGs: done with "
	      << "preparing vtmNodes for implants\n"; 
    }

    while (nodeQueue.is_empty() == false) {
      nodeIndex = nodeQueue.popNodeQueue();
      // duplicate lat within this current lattice.
      LatticeNode * node = nodes.find(nodeIndex);
      
      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::readRecPFSGs: processing subPFSG (" 
	       << nodeIndex << ", " << getWord(node->word) << ")\n";
      }

      Lattice *lat = getNonRecPFSG(node->word);
      if (!lat) {
	continue; 
      }
      implantLattice(nodeIndex, *lat);

      dout() << "Lattice::readRecPFSGs: maxIndex (" 
	     << maxIndex << ")\n";
    }


    // release all the memory after all the subPFSGs have implanted.
    LHashIter<VocabIndex, Lattice *> subPFSGsIter(subPFSGs);
    VocabIndex word;
    while (Lattice **pt = subPFSGsIter.next(word)) {
      delete (Lattice *) (*pt); 
    }

    return true;
}

// return a flattened PFSG with nodeVocab as its PFSG name.
Lattice *
Lattice::getNonRecPFSG(VocabIndex nodeVocab)
{

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::getNonRecPFSG: processing subPFSG (" 
	     << getWord(nodeVocab) << ")\n";
    }

    NodeQueue nodeQueue; 
    Lattice **pt = subPFSGs.find(nodeVocab); 
    if (!pt) {
      // this is a terminal node
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::getNonRecPFSG: terminal node (" 
	       << nodeVocab << ")\n";
      }
      return 0;
    }
    
    Lattice * subPFSG = *pt; 
    LHashIter<NodeIndex, LatticeNode> nodeIter(subPFSG->nodes, nodeSort);
    NodeIndex nodeIndex;
    while (LatticeNode *node = nodeIter.next(nodeIndex)) { 
      Lattice **pt = subPFSGs.find(node->word);
      if (!(pt)) {
	// terminal node
	continue;
      }
      nodeQueue.addToNodeQueue(nodeIndex); 
    }

    while (nodeQueue.is_empty() == false) {
      nodeIndex = nodeQueue.popNodeQueue();
      // duplicate lat within this current lattice.
      LatticeNode * node = nodes.find(nodeIndex);

      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::getNonRecPFSG: processing subPFSG (" 
	       << nodeIndex << ", " << getWord(node->word) << ")\n";
      }

      Lattice *lat = getNonRecPFSG(node->word);
      if (!lat) {
	continue;
      }
      subPFSG->implantLattice(nodeIndex, *lat);
    }

    return subPFSG; 
}

// substitute the current occurence of lat.word in subPFSG with
// lat
Boolean
Lattice::implantLattice(NodeIndex vtmNodeIndex, Lattice &lat, float addTime)
{

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::implantLattice: processing subPFSG (" 
	     << lat.getName() << ", " << vtmNodeIndex
	     << ", times offset by " << addTime <<" seconds)\n";
    }

    // going through lat to duplicate all its nodes in the current lattice
    LHashIter<NodeIndex, LatticeNode> nodeIter(lat.nodes, nodeSort);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) { 
        // make one copy of the subPFSG' nodes in this current PFSG
	LatticeNode *newNode = nodes.insert(maxIndex+nodeIndex);
	newNode->flags = node->flags; 
	if (node->word == Vocab_None) {
	    newNode->word = Vocab_None;
	} else {
	    // this is necessary because the two lattices might use
	    // different vocabularies
	    VocabString wn = lat.vocab.getWord(node->word);
	    newNode->word = vocab.addWord(wn);
	}

	// copy HTKWordInfo information
	if (node->htkinfo) {
	    HTKWordInfo *linkinfo = new HTKWordInfo;
	    assert(linkinfo != 0);
	    newNode->htkinfo = htkinfos[htkinfos.size()] = linkinfo;

	    *linkinfo = *node->htkinfo;
	    linkinfo->word = newNode->word;	// see above
	    linkinfo->time = linkinfo->time + addTime;
	}

	// clone transition
	{ // for incoming nodes
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->inTransitions);
	  NodeIndex fromNodeIndex;
	  while (LatticeTransition *trans = transIter.next(fromNodeIndex)) {
	    *(newNode->inTransitions.insert(maxIndex+fromNodeIndex)) = *trans;
	  }
	}
	{ // for outgoing nodes
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->outTransitions);
	  NodeIndex toNodeIndex;
	  while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
	    *(newNode->outTransitions.insert(maxIndex+toNodeIndex)) = *trans;
	  }
	}
    }

    NodeIndex subInitial = lat.getInitial();
    LatticeNode * initialNode;
    if (!(initialNode = 
          nodes.find(maxIndex+subInitial))) {
        // check initial node
        if (debug(DebugPrintFatalMessages)) {
          dout()  << "Fatal Error in Lattice::implantLattice: (" 
		  << lat.getName() 
                  << ") undefined initial node Index ("
		  << subInitial << ")\n";
        }
        exit(-1); 
    } else {
      initialNode->word = Vocab_None;
    }

    NodeIndex subFinal = lat.getFinal();
    LatticeNode *finalNode; 
    if (!(finalNode = 
          nodes.find(maxIndex+subFinal))) {
        // only check the last final node
        if (debug(DebugPrintFatalMessages)) {
          dout()  << "Fatal Error in Lattice::implantLattice: (" 
		  << lat.getName() 
                  << ") undefined initial node Index ("
		  << subFinal << ")\n";
        } 
        exit(-1); 
    } else {
      finalNode->word = Vocab_None;
    }
    // done with checking the nodes

    // connecting initial and final subPFSG nodes with the existing PFSG
    {

	// processing incoming and outgoing nodes of node vtmNodeIndex;
	LatticeNode * node = nodes.find(vtmNodeIndex);

	NodeIndex subInitial = maxIndex+lat.getInitial();
	NodeIndex subFinal = maxIndex+lat.getFinal();
	{
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->inTransitions);
	  NodeIndex fromNodeIndex;
	  while (LatticeTransition *trans = transIter.next(fromNodeIndex)) {
	    insertTrans(fromNodeIndex, subInitial, *trans);
          }
	  // end of processing incoming nodes of node
	}
	{
	  // processing outgoing nodes of node
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->outTransitions);
	  NodeIndex toNodeIndex;
	  while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
	    insertTrans(subFinal, toNodeIndex, *trans);
          }
	  // end of processing outgoing nodes of node
	}
    }

    removeNode(vtmNodeIndex);
    maxIndex += lat.getMaxIndex();

    return true; 
}

// substitute all the occurences of lat.word in subPFSG with
// lat
Boolean
Lattice::implantLatticeXCopies(Lattice &lat)
{

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::implantLatticeXCopies: processing subPFSG (" 
	     << lat.getName() << ")\n";
    }

    unsigned numCopies = 0;
    {
      LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
      NodeIndex nodeIndex;
      VocabIndex subPFSGWord = vocab.getIndex(lat.getName());
      while (LatticeNode *node = nodeIter.next(nodeIndex)) { 
	if (node->word == subPFSGWord) {
	  numCopies++;
	}
      }  
    }

    // need to check whether numNodes is the same as lat.maxIndex
    unsigned numNodes = lat.getNumNodes(); 
    LHashIter<NodeIndex, LatticeNode> nodeIter(lat.nodes, nodeSort);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) { 
      // need to make numCopies
      // make "copies" of the subPFSG' nodes in this current PFSG
      for (unsigned k = 0; k < numCopies; k++) {
	LatticeNode *newNode = nodes.insert(maxIndex+k*numNodes+nodeIndex);
	newNode->word = node->word;
	newNode->flags = node->flags; 

	// clone transition
	{ // for incoming nodes
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->inTransitions);
	  NodeIndex fromNodeIndex;
	  while (transIter.next(fromNodeIndex)) {
	    newNode->inTransitions.insert(maxIndex+k*numNodes+fromNodeIndex);
	  }
	}
	{ // for outgoing nodes
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->outTransitions);
	  NodeIndex toNodeIndex;
	  while (transIter.next(toNodeIndex)) {
	    newNode->outTransitions.insert(maxIndex+k*numNodes+toNodeIndex);
	  }
	}
      }
    }

    NodeIndex subInitial = lat.getInitial();
    LatticeNode * initialNode;
    if (!(initialNode = 
          nodes.find(subInitial+maxIndex+(numCopies-1)*numNodes))) {
        // only check the last initial node
        if (debug(DebugPrintFatalMessages)) {
          dout()  << "Fatal Error in Lattice::implantLatticeXCopies: (" 
		  << lat.getName() 
                  << ") undefined initial node Index\n";
        }
        exit(-1); 
    }

    NodeIndex subFinal = lat.getFinal();
    LatticeNode *finalNode; 
    if (!(finalNode = 
          nodes.find(subFinal+maxIndex+(numCopies-1)*numNodes))) {
        // only check the last final node
        if (debug(DebugPrintFatalMessages)) {
          dout()  << "Fatal Error in Lattice::implantLatticeXCopies: "
		  << lat.getName() 
                  << "undefined final node Index\n";
        }
        exit(-1); 
    }
    // done with checking the nodes

    // connecting initial and final subPFSG nodes with the existing PFSG
    {
      unsigned k = 0;
      LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
      NodeIndex nodeIndex, fromNodeIndex, toNodeIndex;
      VocabIndex subPFSGWord = vocab.getIndex(lat.getName());
      for (nodeIndex = 0; nodeIndex< maxIndex; nodeIndex++) {
	
	LatticeNode * node = nodes.find(nodeIndex); 
	if (node->word == subPFSGWord) {
	  // processing incoming and outgoing nodes of node

	  NodeIndex subInitial = 
	    lat.getInitial()+maxIndex+(numCopies-1)*numNodes;
	  NodeIndex subFinal = 
	    lat.getFinal()+maxIndex+(numCopies-1)*numNodes;
	  {
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->inTransitions);
	  while (LatticeTransition *trans = transIter.next(fromNodeIndex)) {
	    insertTrans(fromNodeIndex, subInitial, *trans);
            removeTrans(fromNodeIndex, nodeIndex);
          }
	  // end of processing incoming nodes of node
	  }
	  {
	  // processing outgoing nodes of node
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(node->outTransitions);
	  while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
	    insertTrans(subFinal, toNodeIndex, *trans);
            removeTrans(nodeIndex, toNodeIndex);
          }
	  // end of processing outgoing nodes of node
	  }
	  k++;
	  if (k > numCopies) {
	    break;
	  }
	} // end of processing k-th copy of implanted node
      } // end of going through the existing lattices
    }

    maxIndex += numCopies * lat.getMaxIndex();

    return true; 
}

Boolean
Lattice::readPFSGs(File &file)
{
  if (debug(DebugPrintFunctionality)) {
    dout()  << "Lattice::readPFSGs: reading in nested PFSGs...\n"; 
  }

  removeAll();

  char buffer[1024];
  char *subName = 0;

  // usually, it is the main PFSG
  if (!readPFSG(file)) {
    return false;
  }

  char *line;
  while ((line = file.getline()) &&
         sscanf(line, " name %1024s", buffer) == 1)
  {

    subName = strdup(buffer);
    assert(subName != 0);

    // get the number of copies needed for each subPFSG.
    unsigned copies = 0; 
    {
      VocabIndex windex = vocab.addWord(subName);
      LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
      NodeIndex nodeIndex;
      while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	if (node->word == windex) {
	    copies++;
	}
      }
    }
    
    unsigned numNodes;

    line = file.getline();
    if (!line || sscanf(line, " nodes %u", &numNodes) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "missing nodes in sub-PFSG of the PFSG file\n";
	}
	free(subName);
	return false;
    }
    
    // Parse node names and create nodes
    makeArray(VocabString, fields, numNodes + 3);
    unsigned numFields = Vocab::parseWords(line, fields, numNodes + 3);

    if (numFields != numNodes + 2) {
	if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "incorrect number of node labels\n";
	}
	free(subName);
	return false;
    }

    for (NodeIndex n = 0; n < numNodes; n ++) {
	/*
	 * Map word string to VocabIndex
	 */
	VocabIndex windex;
	if (strcmp(fields[n + 2], NullNodeName) == 0) {
	    windex = Vocab_None;
	} else if (useUnk || keepUnk) {
	    windex = vocab.getIndex(fields[n + 2], vocab.unkIndex());
	} else {
	    windex = vocab.addWord(fields[n + 2]);
	}

	// make "copies" of the subPFSGs' nodes
	for (unsigned k = 0; k < copies; k++) {
	  LatticeNode *node = nodes.insert(maxIndex+k*numNodes+n);
	  node->word = windex;
	  node->flags = 0; 
	}
    }

    unsigned initialNodeIndex;
    line = file.getline();
    if (!line || sscanf(line, " initial %u", &initialNodeIndex) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "missing initial node Index in PFSG file\n";
	}
	free(subName);
	return false;
    }

    LatticeNode * initialNode;
    if (!(initialNode = 
	  nodes.find(initialNodeIndex+maxIndex+(copies-1)*numNodes))) {
        // only check the last initial node
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "undefined initial node Index\n";
	}
	free(subName);
	return false;
    }

    unsigned finalNodeIndex;
    line = file.getline();
    if (!line || sscanf(line, " final %u", &finalNodeIndex) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "missing final node Index in PFSG file\n";
	}
	free(subName);
	return false;
    }

    LatticeNode *finalNode; 
    if (!(finalNode = 
	  nodes.find(finalNodeIndex+maxIndex+(copies-1)*numNodes))) {
        // only check the last final node
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "undefined final node Index\n";
	}
	free(subName);
	return false;
    }

    // reading in all the transitions for the current subPFSG
    unsigned numTransitions;
    line = file.getline();
    if (!line || sscanf(line, " transitions %u", &numTransitions) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "missing transitions in PFSG file\n";
	}
	free(subName);
	return false;
    }
    
    for (unsigned i = 0; i < numTransitions; i ++) {
	unsigned from, to;
	double prob;

	line = file.getline();
	if (!line || sscanf(line, " %u %u %lf", &from, &to, &prob) != 3) {
  	    if (debug(DebugPrintFatalMessages)) {
	      dout()  << "Fatal Error in Lattice::readPFSGs: "
		      << "missing transition " << i << " in PFSG file\n";
	    }
	    free(subName);
	    return false;
	}

	for (unsigned k = 0; k < copies; k++) {
	  LatticeTransition t(IntlogToLogP(prob), 0);
	  insertTrans(maxIndex+k*numNodes+from, maxIndex+k*numNodes+to, t);
	}
	
    }
    if (debug(DebugPrintOutLoop)) {
      dout()  << "Lattice::readPFSGs: done with reading "
	      << copies << " copies of sub pfsg (" << subName << ")\n";
    }

    // going through all the nodes to flatten the current PFSG
    VocabIndex windex = vocab.addWord(subName);
    unsigned k = 0;            // number of copies consumed
    initialNodeIndex += maxIndex;
    finalNodeIndex += maxIndex;
    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
    NodeIndex nodeIndex;
    LatticeNode *node;
    while ((node = nodeIter.next(nodeIndex)) && (k < copies)) {
    
      // do a substitution
      if (node->word == windex) {

	LatticeNode * initSubPFSGNode = nodes.find(initialNodeIndex);
	NodeIndex fromNodeIndex, toNodeIndex;

	// connecting incoming nodes to initialNodeIndex
	{
	  TRANSITER_T<NodeIndex,LatticeTransition>
	    transIter(node->inTransitions); 
	  while (LatticeTransition *trans = transIter.next(fromNodeIndex)) {
	    *(initSubPFSGNode->inTransitions.insert(fromNodeIndex)) = *trans;
	  }
	}
	if (debug(DebugPrintInnerLoop)) {
	  dout()  << "Lattice::readPFSGs: done with removing orig connection\n";
	}
	{
	  // add initSubPFSG to the incoming nodes
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(initSubPFSGNode->inTransitions); 
	  while (LatticeTransition *trans = transIter.next(fromNodeIndex)) {
	    LatticeNode * fromNode = nodes.find(fromNodeIndex);
	    *(fromNode->outTransitions.insert(initialNodeIndex)) = *trans;
	  }
	}
	if (debug(DebugPrintInnerLoop)) {
	  dout()  << "Lattice::readPFSGs: done with adding new connection\n"; 
	}

	// connecting subPfsg->final to outgoing nodes
	LatticeNode * finalSubPFSGNode = nodes.find(finalNodeIndex);
	{
	  TRANSITER_T<NodeIndex,LatticeTransition>
	    transIter(node->outTransitions); 
	  while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
	    *(finalSubPFSGNode->outTransitions.insert(toNodeIndex)) = *trans;
	  }
	}
	if (debug(DebugPrintInnerLoop)) {
	  dout()  << "Lattice::readPFSGs: done with "
		  << "removing its old transitions for outgoing\n"; 
	}

	{
	  // add finalSubPFSG to the outgoing nodes
	  TRANSITER_T<NodeIndex,LatticeTransition> 
	    transIter(finalSubPFSGNode->outTransitions); 
	  while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
	    LatticeNode * toNode = nodes.find(toNodeIndex);
	    *(toNode->inTransitions.insert(finalNodeIndex)) = *trans;
	  }
	}
	if (debug(DebugPrintInnerLoop)) {
	  dout()  << "Lattice::readPFSGs: done with "
		  << "adding new connection for outgoing\n"; 
	}

	// removing this pseudo word, finally
	removeNode(nodeIndex); 

	if (debug(DebugPrintInnerLoop)) {
	  dout()  << "Lattice::readPFSGs: done with "
		  << "removing node (" << nodeIndex << ")\n"; 
	}

	k++;
	initialNodeIndex += numNodes;
	finalNodeIndex += numNodes;

	if (debug(DebugPrintInnerLoop)) {
	  dout()  << "Lattice::readPFSGs: done with "
		  << "single flattening (" << subName << ")\n"; 
	}

      } // end of single flatening
    } // end of flatening loop

    maxIndex += copies*numNodes;

    if (subName) {
      free(subName);
    }
  }

  return true;
}


Boolean
Lattice::readPFSG(File &file)
{
    removeAll();

    if (debug(DebugPrintOutLoop)) {
        dout() << "Lattice::readPFSG: reading in PFSG....\n"; 
    }

    char buffer[1024];

    char *line = file.getline();
    if (!line || sscanf(line, " name %1024s", buffer) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSG: "
		  << "missing name in PFSG file\n";
	}
	return false;
    }

    if (name) {
	free((void *)name);
    }

    /*
     * try to parse duration from name string,
     * and strip it from name if present
     */
    {
	char *pos = strchr(buffer, '(');
	double value;

	if (pos != 0 && sscanf(pos, "(duration=%lf)", &value) == 1) {
	    duration = value;
	    *pos = '\0';
	} else {
	    duration = 0.0;
	}
    }

    name = strdup(buffer);
    assert(name != 0);

    unsigned numNodes;

    line = file.getline();
    if (!line || sscanf(line, " nodes %u", &numNodes) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSG: "
		  << "missing nodes in PFSG file\n";
	}
	return false;
    }

    maxIndex = numNodes; 

    /*
     * Parse node names and create nodes
     */
    makeArray(VocabString, fields, numNodes + 3);
    unsigned numFields = Vocab::parseWords(line, fields, numNodes + 3);

    if (numFields != numNodes + 2) {
	if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSGs: "
		  << "incorrect number of node labels\n";
	}
	return false;
    }

    for (NodeIndex n = 0; n < numNodes; n ++) {
	/*
	 * Map word string to VocabIndex
	 */
	VocabIndex windex;
	if (strcmp(fields[n + 2], NullNodeName) == 0) {
	    windex = Vocab_None;
	} else if (useUnk || keepUnk) {
	    windex = vocab.getIndex(fields[n + 2], vocab.unkIndex());
	} else {
	    windex = vocab.addWord(fields[n + 2]);
	}

	LatticeNode *node = nodes.insert(n);

	node->word = windex;
	node->flags = 0; 
    }

    unsigned initialNodeIndex;
    line = file.getline();
    if (!line || sscanf(line, " initial %u", &initialNodeIndex) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSG: "
		  << "missing initial node Index in PFSG file\n";
	}
	return false;
    }

    LatticeNode * initialNode;
    if (! (initialNode = nodes.find(initialNodeIndex))) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSG: "
		  << "undefined initial node Index\n";
	}
	return false;
    }

    initial = initialNodeIndex;

    unsigned finalNodeIndex;
    line = file.getline();
    if (!line || sscanf(line, " final %u", &finalNodeIndex) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSG: "
		  << "missing final node Index in PFSG file\n";
	}
	return false;
    }

    LatticeNode *finalNode; 
    if (! (finalNode = nodes.find(finalNodeIndex))) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSG: "
		  << "undefined final node Index\n";
	}
	return false;
    }

    final = finalNodeIndex;

    if (top) {
      initialNode->word = vocab.ssIndex(); 
      finalNode->word = vocab.seIndex(); 
      top = false;
    }

    unsigned numTransitions;
    line = file.getline();
    if (!line || sscanf(line, " transitions %u", &numTransitions) != 1) {
        if (debug(DebugPrintFatalMessages)) {
	  dout()  << "Fatal Error in Lattice::readPFSG: "
		  << "missing transitions in PFSG file\n";
	}
	return false;
    }
    
    for (unsigned i = 0; i < numTransitions; i ++) {
	unsigned from, to;
	double prob;

	line = file.getline();
	if (!line || sscanf(line, " %u %u %lf", &from, &to, &prob) != 3) {
  	    if (debug(DebugPrintFatalMessages)) {
	      dout()  << "Fatal Error in Lattice::readPFSG: "
		      << "missing transition " << i << " in PFSG file\n";
	    }
	    return false;
	}

	LatticeNode *fromNode = nodes.find(from);
	if (!fromNode) {
  	    if (debug(DebugPrintFatalMessages)) {
	      dout()  << "Fatal Error in Lattice::readPFSG: "
		      << "undefined source node in transition " << i << "\n";
	    }
	    return false;
	}

	LatticeNode *toNode = nodes.find(to);
	if (!toNode) {
  	    if (debug(DebugPrintFatalMessages)) {
	      dout()  << "Fatal Error in Lattice::readPFSG: "
		      << "undefined target node in transition " << i << "\n";
	    }
	    return false;
	}

	LogP weight = IntlogToLogP(prob);

	LatticeTransition *trans = fromNode->outTransitions.insert(to);
	trans->weight = weight;
	trans->flags = 0;

	trans = toNode->inTransitions.insert(from);
	trans->weight = weight;
	trans->flags = 0;
    }

    return true;
}

Boolean
Lattice::readMesh(File &file)
{
    WordMesh inputMesh(vocab);

    Boolean status = inputMesh.read(file);

    if (!status) {
	return status;
    }

    return createFromMesh(inputMesh);
}


/* inputMesh must use the same vocab as lattice */
Boolean
Lattice::createFromMesh(WordMesh &inputMesh)
{
    if (&inputMesh.vocab != &vocab)
	return false;
  
    // set the weight of the score we insert to 1
    htkheader.x1scale = 1.0;

    removeAll();

    setInitial(dupNode(Vocab_None));
    NodeIndex fromNodeIndex = getInitial();

    LatticeNode *fromNode = nodes.find(fromNodeIndex);
    assert(fromNode != 0);
    fromNode->posterior = ProbToLogP(inputMesh.totalPosterior);

    Boolean haveSentStart = false;
    Boolean haveSentEnd = false;

    for (unsigned i = 0; i < inputMesh.length(); i ++) {
	// insert NULL node as destination for the following words
	// in this alignment column
	NodeIndex toNodeIndex = getMaxIndex();
	insertNode(NullNodeName, toNodeIndex);

	LHash<VocabIndex,Prob> *words = inputMesh.wordColumn(i);
	LHash<VocabIndex,NBestWordInfo> *winfos =
					inputMesh.wordinfoColumn(i);
	LHashIter<VocabIndex,Prob> wordsIter(*words);

	Prob *prob;
	VocabIndex word;

	while ((prob = wordsIter.next(word))) {
	    // insert a HTK lattice link (which is a node between two NULL
	    // nodes) for each word in this alignment column,
	    // also put the log of the posterior prob in xscore1
	    NodeIndex newNode = getMaxIndex();

	    // replace the *DELETE* word with NULL node
	    if (word == inputMesh.deleteIndex) {
		insertNode(NullNodeName, newNode);
	    } else {
		insertNode(vocab.getWord(word), newNode);
	    }

	    if (word == vocab.ssIndex()) {
		haveSentStart = true;
	    }
	    if (word == vocab.seIndex()) {
		haveSentEnd = true;
	    }

	    LatticeNode *node = findNode(newNode);
	    assert(node != 0);

	    HTKWordInfo *linkinfo = new HTKWordInfo;
	    assert(linkinfo != 0);
	    node->htkinfo = htkinfos[htkinfos.size()] = linkinfo;

	    /*
	     * By default use the posterior as a transition score
	     * (in HTK format, this is output as the a= score)
	     */
	    LogP transitionScore = ProbToLogP(*prob);

	    /*
	     * Transfer the confidence (whether set or not) back to
	     * caller as an x2 score
	     */
	    LogP confidenceScore = LogP_PseudoZero;
	    LogP confidenceScore2 = LogP_PseudoZero;
	    LogP confidenceScore3 = LogP_PseudoZero;

	    /*
	     * Transfer acoustic info from mesh, if available
	     */
	    NBestWordInfo *winfo = winfos ? winfos->find(word) : 0;
	    if (winfo && winfo->valid()) {
		node->htkinfo->time = winfo->start;
		node->htkinfo->acoustic = winfo->acousticScore;
		node->htkinfo->language = winfo->languageScore;
		if (winfo->phones) {
		   node->htkinfo->div = strdup(winfo->phones);
		}
		transitionScore = winfo->acousticScore;
		confidenceScore = winfo->confidenceScore;
		confidenceScore2 = winfo->confidenceScore2;
		confidenceScore3 = winfo->confidenceScore3;
	    }

	    node->posterior = ProbToLogP(*prob);

	    /* for backward compatibility, also save the posterior
		as an x1 score */
	    node->htkinfo->xscore1 = node->posterior;
	    node->htkinfo->xscore2 = confidenceScore;
	    node->htkinfo->xscore3 = confidenceScore2;
	    node->htkinfo->xscore4 = confidenceScore3;

	    LatticeTransition inTrans(transitionScore, 0);
	    LatticeTransition outTrans(LogP_One, 0);
	    insertTrans(fromNodeIndex, newNode, inTrans);
	    insertTrans(newNode, toNodeIndex, outTrans);
	}

	// the next column of words will follow this column
	fromNodeIndex = toNodeIndex;

	// assign the total posterior to the common nodes
        LatticeNode *fromNode = nodes.find(fromNodeIndex);
        assert(fromNode != 0);
        fromNode->posterior = ProbToLogP(inputMesh.totalPosterior);
    }

    // make last node the final one
    setFinal(fromNodeIndex);

    // provide sentence start/end tags if not found in lattice
    if (!haveSentStart) {
	LatticeNode *node = findNode(initial);
	assert(node != 0);
	node->word = vocab.ssIndex();
    }
    if (!haveSentEnd) {
	LatticeNode *node = findNode(final);
	assert(node != 0);
	node->word = vocab.seIndex();
    }

    return true;
}

Boolean
Lattice::writeCompactPFSG(File &file)
{
    if (debug(DebugPrintFunctionality)) {
      dout()  << "Lattice::writeCompactPFSG: writing ";
    }

    if (duration != 0.0) {
	file.fprintf("name %s(duration=%lg)\n", name, duration);
    } else {
	file.fprintf("name %s\n", name);
    }
	
    /*
     * We remap the internal node indices to consecutive unsigned integers
     * to allow a compact output representation.
     * We iterate over all nodes, renumbering them, and also counting the
     * number of transitions overall.
     */

    // map nodeIndex to unsigned
    LHash<NodeIndex,unsigned> nodeMap;
    unsigned numNodes = 0;
    unsigned numTransitions = 0;

    file.fprintf("nodes %d", getNumNodes());

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	*nodeMap.insert(nodeIndex) = numNodes ++;
	numTransitions += node->outTransitions.numEntries();

	file.fprintf(" %s", ((!printSentTags && nodeIndex == initial) ||
			      (!printSentTags && nodeIndex == final) ||
			      node->word == Vocab_None) ?
				NullNodeName : getWord(node->word));
    }

    file.fprintf("\n");

    if (initial != NoNode) {
	file.fprintf("initial %u\n", *nodeMap.find(initial));
    }
    if (final != NoNode) {
	file.fprintf("final %u\n", *nodeMap.find(final));
    }

    file.fprintf("transitions %u\n", numTransitions);

    if (debug(DebugPrintFunctionality)) {
      dout()  << numNodes << " nodes, "
	      << numTransitions << " transitions\n";
    }

    nodeIter.init(); 
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
        unsigned *fromNodeId = nodeMap.find(nodeIndex);

 	NodeIndex toNode;

	TRANSITER_T<NodeIndex,LatticeTransition>
	  transIter(node->outTransitions);
	while (LatticeTransition *trans = transIter.next(toNode)) {
	    unsigned int *toNodeId = nodeMap.find(toNode); 
	    assert(toNodeId != 0);

	    int logToPrint = LogPtoIntlog(trans->weight);

	    if (limitIntlogs && logToPrint < minIntlog) {
		logToPrint = minIntlog;
	    }

	    file.fprintf("%u %u %d\n", *fromNodeId, *toNodeId, logToPrint);
	}
    }

    file.fprintf("\n");

    return true;
}

Boolean
Lattice::writePFSG(File &file)
{
    if (debug(DebugPrintFunctionality)) {
      dout()  << "Lattice::writePFSG: writing ";
    }

    if (duration != 0.0) {
	file.fprintf("name %s(duration=%lg)\n", name, duration);
    } else {
	file.fprintf("name %s\n", name);
    }
	
    NodeIndex nodeIndex;

    unsigned numTransitions = 0;

    file.fprintf("nodes %d", maxIndex);

    for (nodeIndex = 0; nodeIndex < maxIndex; nodeIndex ++) {
        LatticeNode *node = nodes.find(nodeIndex);

	if (node) {
	   numTransitions += node->outTransitions.numEntries();
	}

	file.fprintf(" %s",
		((!printSentTags && nodeIndex == initial) ||
		 (!printSentTags && nodeIndex == final) ||
		 node == 0 || node->word == Vocab_None) ?
				NullNodeName : getWord(node->word));
    }

    file.fprintf("\n");

    if (initial != NoNode) {
	file.fprintf("initial %u\n", initial);
    }
    if (final != NoNode) {
	file.fprintf("final %u\n", final);
    }

    file.fprintf("transitions %u\n", numTransitions);

    if (debug(DebugPrintFunctionality)) {
      dout()  << maxIndex << " nodes, " << numTransitions << " transitions\n";
    }

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
 	NodeIndex toNode;

	TRANSITER_T<NodeIndex,LatticeTransition>
	  transIter(node->outTransitions);
	while (LatticeTransition *trans = transIter.next(toNode)) {
	    int logToPrint = LogPtoIntlog(trans->weight);

	    if (limitIntlogs && logToPrint < minIntlog) {
		logToPrint = minIntlog;
	    }

	    file.fprintf("%u %u %d\n",
			nodeIndex,
			toNode, 
			logToPrint);
	}
    }

    file.fprintf("\n");

    return true;
}

unsigned
Lattice::getNumTransitions()
{
    unsigned numTransitions = 0;

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	numTransitions += node->outTransitions.numEntries();
    }

    return numTransitions;
}

// this is for debugging purpose
Boolean
Lattice::printNodeIndexNamePair(File &file)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::printNodeIndexNamePair: "
	     << "printing Index-Name pairs!\n";
    }

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
    NodeIndex nodeIndex;
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	file.fprintf("%d %s (%d)\n", nodeIndex,
					getWord(node->word), node->word);
    }

    return true;
}

Boolean
Lattice::readPFSGFile(File &file)
{
    Boolean val = false; 

    while (file.fgetc() != EOF) {

      file.fseek(-1, SEEK_CUR); 

      val = readPFSG(file); 
      
      while (file.fgetc() == '\n' || file.fgetc() == ' ') {} 

      file.fseek(-1, SEEK_CUR); 
    }

    return val;
}

Boolean
Lattice::writePFSGFile(File &file)
{
    return true;
}

/* **************************************************
   some more complex functions of Lattice class
   ************************************************** */

// *****************************************************
// *******************algorithm*************************
// going through all the Null nodes, 
//   if nodeIndex is the initial or final node, skip, 
//   if nodeIndex is not a Null node, skip
//   
//   if nodeIndex is a Null node, 
//     going through all the inTransitions,
//       collect weight for the inTransition,
//       collect the source node s,
//       remove the inTransition, 
//       going through all the outTransitions,
//         collect the weight for the outTransition,
//         combine it with the inTransition weight,
//         insert an outTransition to s,
//         remove the outTransition

LogP 
Lattice::detectSelfLoop(NodeIndex nodeIndex)
{
    LogP base = 10;
    LogP weight = unit();

    LatticeNode *node = nodes.find(nodeIndex);
    if (!node) {
      if (debug(DebugPrintFatalMessages)) {
	dout() << "Fatal Error in Lattice::detectSelfLoop: "
	       << nodeIndex << "\n";
      }
      exit(-1); 
    }

    LatticeTransition *trans;

    trans = node->outTransitions.find(nodeIndex);

    if (!trans) {
      return weight;
    } else {
      weight = combWeights(trans->weight, weight);
    }

    if (!weight) {
      return weight; }
    else {
      return (-log(1-exp(weight*log(base)))/log(base)); 
    }
}

// it removes all the nodes that have given word
Boolean
Lattice::removeAllXNodes(VocabIndex xWord)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::removeAllXNodes: "
	     << "removing all " << getWord(xWord) << endl;
    }

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {

      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::removeAllXNodes: processing nodeIndex " 
	       << nodeIndex << "\n";
      }

      if (nodeIndex == final || nodeIndex == initial) {
	  continue; 
      }

      if (node->word == xWord) {
	// this node is a Null node
	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::removeAllXNodes: " 
		 << "remove node " << nodeIndex << "\n";
	}

	LogP loopweight = detectSelfLoop(nodeIndex); 

	// remove the current node, all the incoming and outgoing edges
	//    and create new edges
	// Notice that all the edges are recorded in two places: 
	//    inTransitions and outTransitions
	TRANSITER_T<NodeIndex,LatticeTransition>
	  inTransIter(node->inTransitions);
	NodeIndex fromNodeIndex;
	while (LatticeTransition *inTrans = inTransIter.next(fromNodeIndex)) {

	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::removeAllXNodes: " 
		   << "  fromNodeIndex " << fromNodeIndex << "\n";
	  }

	  LogP inWeight = inTrans->weight;

	  if (fromNodeIndex == nodeIndex) {
	    continue; 
	  }

	  TRANSITER_T<NodeIndex,LatticeTransition>
	    outTransIter(node->outTransitions);
	  NodeIndex toNodeIndex;
	  while (LatticeTransition *trans = outTransIter.next(toNodeIndex)) {

	    if (debug(DebugPrintInnerLoop)) {
	      dout() << "Lattice::removeAllXNodes: " 
		     << "     toNodeIndex " << toNodeIndex << "\n";
	    }

	    if (toNodeIndex == nodeIndex) {
	      continue; 
	    }

	    // loopweight is 1 in the prob domain and 
	    // loopweight is 0 in the log domain, if no loop 
	    //     for the current node
	    LogP weight = combWeights(inWeight, trans->weight); 
	    weight = combWeights(weight, loopweight); 

	    unsigned flag = 0;
	    // record where pause nodes were eliminated
	    if (xWord != Vocab_None && xWord == vocab.pauseIndex()) {
		flag = pauseTFlag;
	    }

	    LatticeTransition t(weight, flag);
	    // new transition inherits properties from both parents
	    t.flags |= inTrans->flags | trans->flags; 
	    // ... except for "direct (non-pause) connection"
	    t.flags &= ~directTFlag;
	    // a non-pause connection is carried over if we are removing
	    // a null-node and each of the joined transitions was direct
	    if (xWord == Vocab_None &&
		inTrans->flags&directTFlag && trans->flags&directTFlag)
	    {
		t.flags |= directTFlag;
	    }

	    insertTrans(fromNodeIndex, toNodeIndex, t);
	  }
	} // end of inserting new edges

	// deleting xWord node
	removeNode(nodeIndex);

      } // end of processing xWord node
    }

    return true; 
}

Boolean
Lattice::recoverPause(NodeIndex nodeIndex, Boolean loop, Boolean all)
{
    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::recoverPause: " 
	     << "processing nodeIndex " << nodeIndex << "\n";
    }

    // this array is created to avoid inserting new elements into 
    //   temporary index, while iterating over it.
    TRANS_T<NodeIndex,LatticeTransition> newTransitions; 

    // going throught all the successive nodes of the current node (nodeIndex)
    LatticeNode *node = findNode(nodeIndex); 

    // see if we want to insert a pause after this word unconditionally
    Boolean alwaysInsertPause = all && !ignoreWord(node->word);

    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIter(node->outTransitions);
    NodeIndex toNodeIndex; 
    while (LatticeTransition *trans = outTransIter.next(toNodeIndex)) {
      // processing nodes at the next level 
      LatticeNode *toNode = findNode(toNodeIndex); 
      LogP weight = trans->weight; 
      Boolean direct = trans->getFlag(directTFlag); 

      // if we're inserting pauses everywhere OR
      // if the current edge is a pause edge. insert a pause node
      // and its two edges.
      if ((alwaysInsertPause && toNode->word != vocab.pauseIndex()) ||
	  trans->getFlag(pauseTFlag)) {
	NodeIndex newNodeIndex = dupNode(vocab.pauseIndex(), 0); 
	LatticeNode *newNode = findNode(newNodeIndex); 
	LatticeTransition *newTrans = newTransitions.insert(newNodeIndex); 
	newTrans->flags = 0; 
	newTrans->weight = weight; 

	LatticeTransition t(unit(), 0);
	insertTrans(newNodeIndex, toNodeIndex, t);
	// add self-loop
	if (loop) { 
	  insertTrans(newNodeIndex, newNodeIndex, t);
	}
	
	if (!alwaysInsertPause && !direct) {
	  removeTrans(nodeIndex, toNodeIndex); 
	}
      }
    } // end of outGoing edge loop

    TRANSITER_T<NodeIndex,LatticeTransition> 
      newTransIter(newTransitions);
    NodeIndex newNodeIndex;
    while (LatticeTransition *newTrans = newTransIter.next(newNodeIndex)) {
      LatticeTransition t(newTrans->weight, 0);
      insertTrans(nodeIndex, newNodeIndex, t);
    }

    return true;
}


/* ********************************************************
   recover the pauses that have been removed.
   ******************************************************** */
Boolean
Lattice::recoverPauses(Boolean loop, Boolean all)
{
    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::recoverPauses: "
	       << (all ? "inserting" : "recovering") << " pauses\n";
    }

    unsigned numNodes = getNumNodes();
    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
	dout() << "Lattice::recoverPauses: "
	       << "warning: there are " << (numNodes - numReachable)
	       << " unreachable nodes\n";
    }

    for (unsigned i = 0; i < numReachable; i ++) {
	recoverPause(sortedNodes[i], loop, all);
    }

    delete [] sortedNodes;
    return true; 
}

Boolean
Lattice::recoverCompactPause(NodeIndex nodeIndex, Boolean loop, Boolean all)
{
    unsigned firstPauTrans = 1;

    if (debug(DebugPrintOutLoop)) {
      dout() << "Lattice::recoverCompactPause: "
	     << "processing node: ("
	     << nodeIndex << ")\n"; 
    }

    NodeIndex newNodeIndex;

    // going through all the successive nodes of the current node (nodeIndex)
    LatticeNode *node = findNode(nodeIndex); 

    // see if we want to insert a pause after this word unconditionally
    Boolean alwaysInsertPause = all && !ignoreWord(node->word);

    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIter(node->outTransitions);
    NodeIndex toNodeIndex; 
    while (LatticeTransition *trans = outTransIter.next(toNodeIndex)) {
      // processing nodes at the next level 
      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::recoverCompactPause: "
	       << "  the toNode index " << toNodeIndex << "\n";
      }

      LatticeNode *toNode = findNode(toNodeIndex);
      LogP weight = trans->weight; 
      Boolean direct = trans->getFlag(directTFlag); 

      // if we're inserting pauses everywhere, OR
      // if the current edge is a pause edge. insert a pause node
      // and its two edges.
      if ((alwaysInsertPause && toNode->word != vocab.pauseIndex()) ||
	  trans->getFlag(pauseTFlag))
      {
	if (debug(DebugPrintInnerLoop)) {
	  dout() << "Lattice::recoverCompactPause: "
		 << "inserting pause node between ("
		 << nodeIndex << ", " << toNodeIndex << ")\n"; 
	}

	if (firstPauTrans) { 
	  firstPauTrans = 0; 
	  newNodeIndex = dupNode(vocab.pauseIndex(), 0); 
	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "Lattice::recoverCompactPause: "
		   << "new node index ("
		   << newNodeIndex << ")\n"; 
	  }
	}

	LatticeTransition t(weight, 0);
	insertTrans(newNodeIndex, toNodeIndex, t);
	
	if (!alwaysInsertPause && !direct) {
	  removeTrans(nodeIndex, toNodeIndex); 
	}
      }
    } // end of outGoing edge loop

    // it means that there are outgoing edges with pauses.
    if (!firstPauTrans) { 
      LatticeTransition t(unit(), 0);
      insertTrans(nodeIndex, newNodeIndex, t);
      if (loop) {
	insertTrans(newNodeIndex, newNodeIndex, t);
      }
    }

    return true;
}

/* this method will create separate pauses for each trans that have 
   pause mark on.
   it works well.
   */ 

/* this is to recover compact pauses in lattice
 */
Boolean
Lattice::recoverCompactPauses(Boolean loop, Boolean all)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::recoverCompactPauses: "
	     << (all ? "inserting" : "recovering") << " compact pauses\n";
    }

    unsigned numNodes = getNumNodes();
    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
	dout() << "Lattice::recoverCompactPauses: "
	       << "warning: there are " << (numNodes - numReachable)
	       << " unreachable nodes\n";
    }

    for (unsigned i = 0; i < numReachable; i ++) {
	recoverCompactPause(sortedNodes[i], loop, all);
    }

    delete [] sortedNodes;
    return true; 
}

/*
 * NOTE: MAX_COST is small enough that we can add any of the *_COST constants
 * without overflow.
 */
const unsigned MAX_COST = (unsigned)(-10);      // an impossible path

/*
 * error type used in tracing back word/lattice alignmennts
 */
typedef enum {
      ERR_NONE, ERR_SUB, ERR_INS, ERR_DEL
} ErrorType;


template <class T>
void reverseArray(T *array, unsigned length)
{
    int i, j;	/* j can get negative ! */

    for (i = 0, j = length - 1; i < j; i++, j--) {
	T x = array[i];
	array[i] = array[j];
	array[j] = x;
    }
}

/*
 * sortNodes --
 *	Sort node indices topologically
 *
 * Result:
 *	The number of reachable nodes.
 *
 * Side effects:
 *	sortedNodes is filled with the sorted node indices.
 */
void
Lattice::sortNodesRecursive(NodeIndex nodeIndex, unsigned &numVisited,
			NodeIndex *sortedNodes, Boolean *visitedNodes)
{
    if (visitedNodes[nodeIndex]) {
	return;
    }
    visitedNodes[nodeIndex] = true;

    LatticeNode *node = findNode(nodeIndex); 
    if (!node) {
      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::sortNodesRecursive: "
	       << "can't find an "
	       << "existing node (" << nodeIndex << ")\n";
      }
      return; 
    }

    TRANSITER_T<NodeIndex,LatticeTransition> 
      outTransIter(node->outTransitions);
    NodeIndex nextNodeIndex;
    while (outTransIter.next(nextNodeIndex)) {
        sortNodesRecursive(nextNodeIndex, numVisited, 
			   sortedNodes, visitedNodes);
    }

    sortedNodes[numVisited++] = nodeIndex; 
}

unsigned
Lattice::sortNodes(NodeIndex *sortedNodes, Boolean reversed)
{
    Boolean *visitedNodes = new Boolean[maxIndex];
    assert(visitedNodes != 0);

    for (NodeIndex j = 0; j < maxIndex; j ++) {
	visitedNodes[j] = false;
    }

    unsigned numVisited = 0;

    sortNodesRecursive(initial, numVisited, sortedNodes, visitedNodes);
    
    if (!reversed) {
	// reverse the node order from the way we generated it
	reverseArray(sortedNodes, numVisited);
    }

    delete [] visitedNodes;

    return numVisited;
}

/*
 * latticeWER --
 *	compute minimal word error of path through lattice
 * using two dimensional chart:
 * node axis which indicates the current state in the lattice
 * word axis which indicates the current word in the ref anything before 
 *     which have been considered.
 */
unsigned
Lattice::latticeWER(const VocabIndex *words,
		   unsigned &sub, unsigned &ins, unsigned &del,
		   SubVocab &ignoreWords)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::latticeWER: "
	     << "processing (" << (vocab.use(), words) << ")\n";
    }

    unsigned numWords = Vocab::length(words);
    unsigned numNodes = getNumNodes(); 

    /*
     * The states indexing the DP chart correspond to lattice nodes.
     */
    const unsigned NO_PRED = (unsigned)(-1);	// default for pred link

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
	dout() << "Lattice::latticeWER warning: called with unreachable nodes\n";
    }

    /*
     * Allocate the DP chart.
     * chartEntries are indexed by [word_position][lattice_node],
     * where word_position = 0 is the  left string margin,
     * word_position = numWords + 1 is the right string margin.
     */
    typedef struct {
	unsigned cost;		// minimal path cost to this state
	unsigned ins, del, sub; // error counts by type
	NodeIndex predNode;	// predecessor state used in getting there
	ErrorType errType;	// error type
    } ChartEntry;

    ChartEntry **chart = new ChartEntry *[numWords + 2];
    assert(chart != 0);

    unsigned i;
    for (i = 0; i <= numWords + 1; i ++) {
	chart[i] = new ChartEntry[maxIndex];
	assert(chart[i] != 0);
	for (unsigned j = 0; j < maxIndex; j ++) {
	    chart[i][j].cost = MAX_COST;
	    chart[i][j].sub = chart[i][j].ins = chart[i][j].del = 0;
	    chart[i][j].predNode = NO_PRED;
	    chart[i][j].errType = ERR_NONE;
	}
    }

    /*
     * Prime the chart by anchoring the alignment at the left edge
     */
    chart[0][initial].cost = 0;
    chart[0][initial].ins = chart[0][initial].del = chart[0][initial].sub = 0;
    chart[0][initial].predNode = initial;
    chart[0][initial].errType = ERR_NONE;

    /*
     * Check that lattice starts with <s>
     */
    LatticeNode *initialNode = findNode(initial);
    assert(initialNode != 0);
    if (initialNode->word != vocab.ssIndex() &&
	initialNode->word != Vocab_None)
    {
	dout() << "warning: initial node has non-null word; will be ignored\n";
    }

    // initialize the chart along the node axis
    /*
     * Insertions before the first word
     * NOTE: since we process nodes in topological order this
     * will allow chains of multiple insertions.
     */
    for (unsigned j = 0; j < numReachable; j ++) {
	NodeIndex curr = sortedNodes[j];
	LatticeNode *node = findNode(curr); 
	assert(node != 0);

	unsigned cost = chart[0][curr].cost;
	if (cost >= MAX_COST) continue;

	TRANSITER_T<NodeIndex,LatticeTransition> 
	  outTransIter(node->outTransitions);
	NodeIndex next;
	while (outTransIter.next(next)) {
	    LatticeNode *nextNode = findNode(next); 
	    assert(nextNode != 0);

	    unsigned haveIns = (nextNode->word != Vocab_None &&
	    			    !ignoreWord(nextNode->word) &&
				    !ignoreWords.getWord(nextNode->word));
	    unsigned insCost = cost + haveIns * INS_COST;

	    if (insCost < chart[0][next].cost) {
		chart[0][next].cost = insCost;
		chart[0][next].ins = chart[0][curr].ins + haveIns;
		chart[0][next].del = chart[0][curr].del;
		chart[0][next].sub = chart[0][curr].sub;
		chart[0][next].predNode = curr;
		chart[0][next].errType = ERR_INS;
	    }
	}
    }

    /*
     * For all word positions, compute minimal cost alignment for each
     * state.
     */
    for (i = 1; i <= numWords + 1; i ++) {
	/*
	 * Compute partial alignment cost for all lattice nodes
	 */
	unsigned j;
	for (j = 0; j < numReachable; j ++) {
 	    NodeIndex curr = sortedNodes[j];
	    LatticeNode *node = findNode(curr);
	    assert(node != 0);

	    unsigned cost = chart[i - 1][curr].cost;

	    if (cost >= MAX_COST) continue;

	    /*
	     * Deletion error: current word not matched by lattice
	     */
	    {
		unsigned haveDel = !ignoreWord(words[i - 1]) &&
					!ignoreWords.getWord(words[i - 1]);
		unsigned delCost = cost + haveDel * DEL_COST;

		if (delCost < chart[i][curr].cost) {
		    chart[i][curr].cost = delCost;
		    chart[i][curr].del = chart[i - 1][curr].del + haveDel;
		    chart[i][curr].ins = chart[i - 1][curr].ins;
		    chart[i][curr].sub = chart[i - 1][curr].sub;
		    chart[i][curr].predNode = curr;
		    chart[i][curr].errType = ERR_DEL;
		}
	    }

	    /*
	     * Substitution errors
	     */
	    
	    TRANSITER_T<NodeIndex,LatticeTransition> 
	      outTransIter(node->outTransitions);
	    NodeIndex next;
	    while (outTransIter.next(next)) {
		LatticeNode *nextNode = findNode(next); 
		assert(nextNode != 0);

		VocabIndex word = nextNode->word;

		/*
		 * </s> (on final node) matches Vocab_None in word string
		 */
		if (word == vocab.seIndex()) {
		    word = Vocab_None;
		}

		unsigned haveSub = (word != words[i - 1]);
		unsigned subCost = cost + haveSub * SUB_COST;

		if (subCost < chart[i][next].cost) {
		    chart[i][next].cost = subCost;
		    chart[i][next].sub = chart[i - 1][curr].sub + haveSub;
		    chart[i][next].ins = chart[i - 1][curr].ins;
		    chart[i][next].del = chart[i - 1][curr].del;
		    chart[i][next].predNode = curr;
		    chart[i][next].errType = haveSub ? ERR_SUB : ERR_NONE;
		}
	    }
	}

	for (j = 0; j < numReachable; j ++) {
 	    NodeIndex curr = sortedNodes[j]; 
	    LatticeNode *node = findNode(curr);
	    assert(node != 0);

	    unsigned cost = chart[i][curr].cost;
	    if (cost >= MAX_COST) continue;

	    /*
	     * Insertion errors: lattice node not matched by word
	     * NOTE: since we process nodes in topological order this
	     * will allow chains of multiple insertions.
	     */

	    TRANSITER_T<NodeIndex,LatticeTransition> 
	      outTransIter(node->outTransitions);
	    NodeIndex next;
	    while (outTransIter.next(next)) {
		LatticeNode *nextNode = findNode(next); 
		assert(nextNode != 0);

		unsigned haveIns = (nextNode->word != Vocab_None &&
					!ignoreWord(nextNode->word) &&
					!ignoreWords.getWord(nextNode->word));
	        unsigned insCost = cost + haveIns * INS_COST;

		if (insCost < chart[i][next].cost) {
		    chart[i][next].cost = insCost;
		    chart[i][next].ins = chart[i][curr].ins + haveIns;
		    chart[i][next].del = chart[i][curr].del;
		    chart[i][next].sub = chart[i][curr].sub;
		    chart[i][next].predNode = curr;
		    chart[i][next].errType = ERR_INS;
		}
	    }
	}
    }

    if (chart[numWords+1][final].predNode == NO_PRED) {
	dout() << "Lattice::latticeWER warning: called with unreachable final node\n";
        sub = ins = 0;
	del = numWords;
    } else {
	sub = chart[numWords+1][final].sub;
	ins = chart[numWords+1][final].ins;
	del = chart[numWords+1][final].del;
    }

    if (debug(DebugPrintInnerLoop)) {
      dout() << "Lattice::latticeWER: printing chart:\n";
    }
    for (i = 0; i <= numWords + 1; i ++) {
        unsigned j; 
        for (j = 0; j < numReachable; j++) {
	  if (debug(DebugPrintInnerLoop)) {
	    dout() << "chart[" << i << ", " << j << "] "
		   << chart[i][j].cost << " (" 
		   << chart[i][j].sub << ","
		   << chart[i][j].ins << ","
		   << chart[i][j].del << ")  ";
	  }
	}
	
	if (debug(DebugPrintInnerLoop)) {
	  dout() << "\n"; 
	}
	delete [] chart[i];
    }
    delete [] chart;

    delete [] sortedNodes;
    
    return sub + ins + del;
}

/*
 * Compute node forward and backward probabilities based on transition scores
 * 	Returns the max-min-posterior: the maximum over all lattice paths,
 *	of the minimum posterior of all nodes along the path.
 */
LogP2
Lattice::computeForwardBackward(LogP2 forwardProbs[], LogP2 backwardProbs[],
							double posteriorScale)
{
    /*
     * Algorithm: 
     * 0) sort nodes in topological order
     * 1) forward pass: compute forward probabilities
     * 2) backward pass: compute backward probabilities and posteriors
     * Note: avoid allocating large arrays on stack to avoid problems with
     * resource limits.
     */

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::computeForwardBackward: "
	     << "processing (posterior scale = " << posteriorScale << ")\n";
    }

    /*
     * topological sort
     */
    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
	dout() << "Lattice::computeForwardBackward: warning: called with unreachable nodes\n";
    }
    if (sortedNodes[0] != initial) {
	dout() << "Lattice::computeForwardBackward: initial node is not first\n";
	delete [] sortedNodes;
        return LogP_Zero;
    }
    unsigned finalPosition = 0;
    for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
	if (sortedNodes[finalPosition] == final) break;
    }
    if (finalPosition == numReachable) {
	dout() << "Lattice::computeForwardBackward: final node is not reachable\n";
	delete [] sortedNodes;
        return LogP_Zero;
    }

    /*
     * compute forward probabilities
     */
    for (unsigned i = 0; i < maxIndex; i ++) {
	forwardProbs[i] = LogP_Zero;
    }
    forwardProbs[initial] = LogP_One;

    for (unsigned position = 1; position < numReachable; position ++) {
	LatticeNode *node = nodes.find(sortedNodes[position]);
        assert(node != 0);

	LogP2 prob = LogP_Zero;

        TRANSITER_T<NodeIndex,LatticeTransition>
					inTransIter(node->inTransitions);
	LatticeTransition *inTrans;
	NodeIndex fromNodeIndex; 
	while ((inTrans = inTransIter.next(fromNodeIndex))) {
	    LogP transProb = inTrans->weight;

	    if (transProb == LogP_Zero) {
		transProb = LogP_PseudoZero;
	    }

	    prob = AddLogP(prob,
			   forwardProbs[fromNodeIndex] +
				transProb/posteriorScale);
	}
	forwardProbs[sortedNodes[position]] = prob;
    }

    /*
     * compute backward probabilities
     */
    for (unsigned i = 0; i < maxIndex; i ++) {
	backwardProbs[i] = LogP_Zero;
    }
    backwardProbs[final] = LogP_One;

    for (unsigned position = finalPosition - 1; (int)position >= 0; position --) {
	LatticeNode *node = nodes.find(sortedNodes[position]);
        assert(node != 0);

	LogP2 prob = LogP_Zero;

        TRANSITER_T<NodeIndex,LatticeTransition>
					outTransIter(node->outTransitions);
	LatticeTransition *outTrans;
	NodeIndex toNodeIndex; 
	while ((outTrans = outTransIter.next(toNodeIndex))) {
	    LogP transProb = outTrans->weight;

	    if (transProb == LogP_Zero) {
		transProb = LogP_PseudoZero;
	    }

	    prob = AddLogP(prob,
			   backwardProbs[toNodeIndex] +
				transProb/posteriorScale);
	}
	backwardProbs[sortedNodes[position]] = prob;
    }

    /*
     * set posteriors of unreachable nodes to zero
     */
    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	node->posterior = LogP_Zero;
    }

    /*
     * array of partial min-max-posteriors
     * Note: need to initialize all maxMinPosteriors to 0 since not all
     * might be visited in topological order if nodes have become unreachable.
     */
    LogP2 *maxMinPosteriors = new LogP2[maxIndex];
    assert(maxMinPosteriors != 0);

    for (unsigned i = 0; i < maxIndex; i ++) {
	maxMinPosteriors[i] = LogP_Zero;
    }

    /*
     * compute unnormalized log posteriors, as well as min-max-posteriors
     */
    for (unsigned position = 0; position < numReachable; position ++) {
        NodeIndex nodeIndex = sortedNodes[position];
	LatticeNode *node = nodes.find(nodeIndex);
        assert(node != 0);

        node->posterior = forwardProbs[nodeIndex] + backwardProbs[nodeIndex];

	/*
	 * compute max-min-posteriors
	 */
	if (position == 0) {
	    maxMinPosteriors[nodeIndex] = node->posterior;
	} else {
	    LogP2 maxInPosterior = LogP_Zero;

	    TRANSITER_T<NodeIndex,LatticeTransition>
					    inTransIter(node->inTransitions);
	    NodeIndex fromNodeIndex; 
	    while (inTransIter.next(fromNodeIndex)) {
		 if (maxMinPosteriors[fromNodeIndex] > maxInPosterior) {
		    maxInPosterior = maxMinPosteriors[fromNodeIndex];
		 }
	    }

	    maxMinPosteriors[nodeIndex] =
			(node->posterior > maxInPosterior) ? maxInPosterior
							   : node->posterior;
	}
    }

    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::computeForwardBackward: "
	       << "unnormalized posterior = " << forwardProbs[final] 
	       << " max-min path posterior = " << maxMinPosteriors[final]
	       << endl;
    }

    LogP2 result = maxMinPosteriors[final];

    delete [] sortedNodes;
    delete [] maxMinPosteriors;

    return result;
}

/*
 * Compute node forward/backward viterbi probabilities based on transition
 * scores
 * 	Returns the probability of the best path. 
 */
LogP
Lattice::computeForwardBackwardViterbi(LogP forwardProbs[],
				       LogP backwardProbs[])
{
    /*
     * Algorithm: 
     * 0) sort nodes in topological order
     * 1) forward pass: compute forward probabilities
     * 2) backward pass: compute backward probabilities
     * Note: avoid allocating large arrays on stack to avoid problems with
     * resource limits.
     */

    /*
     * topological sort
     */
    unsigned numNodes = getNumNodes(); 
    
    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
	dout() << "Lattice::computeForwardBackwardViterbi: warning: called with unreachable nodes\n";
    }
    if (sortedNodes[0] != initial) {
	dout() << "Lattice::computeForwardBackwardViterbi: initial node is not first\n";
	delete [] sortedNodes;
        return LogP_Zero;
    }
    unsigned finalPosition = 0;
    for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
	if (sortedNodes[finalPosition] == final) break;
    }
    if (finalPosition == numReachable) {
	dout() << "Lattice::computeForwardBackwardViterbi: final node is not reachable\n";
	delete [] sortedNodes;
        return LogP_Zero;
    }

    /*
     * compute forward probabilities
     */
    for (unsigned i = 0; i < maxIndex; i ++) {
	forwardProbs[i] = LogP_Zero;
    }
    forwardProbs[initial] = LogP_One;

    for (unsigned position = 1; position < numReachable; position ++) {
	LatticeNode *node = nodes.find(sortedNodes[position]);
        assert(node != 0);

	LogP prob = LogP_Zero;

        TRANSITER_T<NodeIndex,LatticeTransition>
					inTransIter(node->inTransitions);
	LatticeTransition *inTrans;
	NodeIndex fromNodeIndex; 
	while ((inTrans = inTransIter.next(fromNodeIndex))) {
	    LogP transProb = inTrans->weight;

	    if (transProb == LogP_Zero) {
		transProb = LogP_PseudoZero;
	    }

            LogP pr = forwardProbs[fromNodeIndex] + transProb;
            if (prob < pr) prob = pr;
	}
	forwardProbs[sortedNodes[position]] = prob;
    }

    /*
     * compute backward probabilities
     */
    for (unsigned i = 0; i < maxIndex; i ++) {
	backwardProbs[i] = LogP_Zero;
    }
    backwardProbs[final] = LogP_One;

    for (unsigned position = finalPosition - 1; (int)position >= 0; position --) {
	LatticeNode *node = nodes.find(sortedNodes[position]);
        assert(node != 0);

	LogP prob = LogP_Zero;

        TRANSITER_T<NodeIndex,LatticeTransition>
					outTransIter(node->outTransitions);
	LatticeTransition *outTrans;
	NodeIndex toNodeIndex; 
	while ((outTrans = outTransIter.next(toNodeIndex))) {
	    LogP transProb = outTrans->weight;

	    if (transProb == LogP_Zero) {
		transProb = LogP_PseudoZero;
	    }

            LogP pr = backwardProbs[toNodeIndex] + transProb;
            if (prob < pr) prob = pr;
	}
	backwardProbs[sortedNodes[position]] = prob;
    }

    LogP bestProb = forwardProbs[final];

    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::computeForwardBackwardViterbi: "
	       << "best prob = " << bestProb << endl;
    }

    delete [] sortedNodes;
    return bestProb;
}

/*
 * Compute node posterior probabilities based on transition scores
 * 	returns max-min path posterior
 */
LogP2
Lattice::computePosteriors(double posteriorScale, Boolean normalize)
{
    LogP2 *forwardProbs = new LogP2[maxIndex];
    assert(forwardProbs != 0);

    LogP2 *backwardProbs = new LogP2[maxIndex];
    assert(backwardProbs != 0);

    LogP2 result =
	    computeForwardBackward(forwardProbs, backwardProbs, posteriorScale);

    if (normalize) {
	LogP2 totalPosterior = forwardProbs[final];

	// normalize the node-level posteriors
	LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
	NodeIndex nodeIndex;

	while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	    node->posterior -= totalPosterior;
	}
    }

    delete [] forwardProbs;
    delete [] backwardProbs;

    return result;
}

/*
 * Write node and transition posteriors in SRILM format:
 *
 *     Word lattices:
 *     version 2
 *     name s
 *     initial i
 *     final f
 *     node n w a p n1 p1 n2 p2 ...
 *     ...
 *
 *     (see wlat-format(5) man page)
 *
 */
Boolean
Lattice::writePosteriors(File &file, double posteriorScale)
{
    LogP2 *forwardProbs = new LogP2[maxIndex];
    assert(forwardProbs != 0);

    LogP2 *backwardProbs = new LogP2[maxIndex];
    assert(backwardProbs != 0);

    LogP2 maxMinPosterior =
	computeForwardBackward(forwardProbs, backwardProbs, posteriorScale);
    LogP2 totalPosterior = maxMinPosterior != LogP_Zero ? 
				nodes.find(initial)->posterior : LogP_Zero;

    file.fprintf("version 2\n");
    file.fprintf("name %s\n", name);
    file.fprintf("initial %u\n", initial);
    file.fprintf("final %u\n", final);

    
    for (unsigned i = 0; i < maxIndex; i ++) {
	LatticeNode *node = nodes.find(i);

        if (node) {
	    file.fprintf("node %u %s -1 %.*lg", i, getWord(node->word),
			  Prob_Precision,
			  (double)LogPtoProb(node->posterior - totalPosterior));

	    TRANSITER_T<NodeIndex,LatticeTransition>
					outTransIter(node->outTransitions);
	    LatticeTransition *outTrans;
	    NodeIndex toNodeIndex; 
	    while ((outTrans = outTransIter.next(toNodeIndex))) {
		file.fprintf(" %u %.*lg", toNodeIndex,
			Prob_Precision, 
			(double)LogPtoProb(forwardProbs[i] +
					    outTrans->weight/posteriorScale +
					    backwardProbs[toNodeIndex] -
					    totalPosterior));
	    }
	    file.fprintf("\n");
        }
    }

    delete [] forwardProbs;
    delete [] backwardProbs;

    return true;
}

/*
 * Removed useless nodes, i.e., those that do not lie on a path from 
 * 	start to end nodes
 */
unsigned
Lattice::removeUselessNodes()
{
    /*
     * Algorithm: 
     * 0) sort nodes in topological order
     * 1) forward pass: compute forward reachability
     * 2) backward pass: compute backward reachability
     * 3) remove all nodes that are not both forward and backward reachable
     */

    Boolean *forwardReachable = new Boolean[maxIndex];
    assert(forwardReachable != 0);

    Boolean *backwardReachable = new Boolean[maxIndex];
    assert(backwardReachable != 0);

    for (unsigned i = 0; i < maxIndex; i ++) {
	forwardReachable[i] = false;
	backwardReachable[i] = false;
    }

    /*
     * topological sort
     */
    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (sortedNodes[0] != initial) {
	dout() << "Lattice::removeUselessNodes: initial node is not first\n";
	delete [] sortedNodes;
	delete [] backwardReachable;
	delete [] forwardReachable;
        return 0;
    }
    unsigned finalPosition = 0;
    for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
	if (sortedNodes[finalPosition] == final) break;
    }
    if (finalPosition == numReachable) {
	dout() << "Lattice::removeUselessNodes: final node is not reachable\n";
	delete [] sortedNodes;
	delete [] backwardReachable;
	delete [] forwardReachable;
        return 0;
    }

    /*
     * compute forward reachability
     */
    forwardReachable[initial] = true;

    for (unsigned position = 1; position < numReachable; position ++) {
	LatticeNode *node = nodes.find(sortedNodes[position]);
        assert(node != 0);

        TRANSITER_T<NodeIndex,LatticeTransition>
					inTransIter(node->inTransitions);
	NodeIndex fromNodeIndex; 
	while (inTransIter.next(fromNodeIndex)) {
	    if (forwardReachable[fromNodeIndex]) {
		forwardReachable[sortedNodes[position]] = true;
		break;
	    }
	}
    }

    /*
     * compute backward reachability
     */
    backwardReachable[final] = true;

    for (unsigned position = finalPosition - 1; (int)position >= 0; position --) {
	LatticeNode *node = nodes.find(sortedNodes[position]);
        assert(node != 0);

        TRANSITER_T<NodeIndex,LatticeTransition>
					outTransIter(node->outTransitions);
	NodeIndex toNodeIndex; 
	while (outTransIter.next(toNodeIndex)) {
	    if (backwardReachable[toNodeIndex]) {
		backwardReachable[sortedNodes[position]] = true;
	    }
	}
    }

    /*
     * prune unreachable nodes
     */
    unsigned numRemoved = 0;

    for (unsigned i = 0; i < maxIndex; i ++) {
	if (!(forwardReachable[i] && backwardReachable[i])) {
	    if (removeNode(i)) {
		numRemoved += 1;
	    }
	}
    }

    delete [] forwardReachable;
    delete [] backwardReachable;
    delete [] sortedNodes;

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::removeUselessNodes: removed "
	     << numRemoved << " nodes\n";
    }

    return numRemoved;
}

/*
 * Prune lattice nodes based on posterior probabilities
 *	All nodes (and attached transitions) with posteriors less than
 *	theshold (if > 0) of the total posterior are removed.
 *	If maxDensity is specified (non-zero), then threshold is iteratively
 *	lowered until the lattice density falls below the given value
 *	The "fast" option avoid recomputing posteriors after each pruning
 *	step, and just removes useless nodes instead.
 */
Boolean
Lattice::prunePosteriors(Prob threshold, double posteriorScale,
			    double maxDensity, unsigned maxNodes, Boolean fast)
{
    // keep track of number of node to know when to stop iteration
    unsigned numNodes = getNumNodes();
    
    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::prunePosteriors: "
	       << "starting with " << numNodes << " nodes\n";
    }

    LogP2 maxMinPosterior;

    if (fast) {
    	maxMinPosterior = computePosteriors(posteriorScale);
    }

    while (numNodes > 0) {
	if (!fast) {
	    maxMinPosterior = computePosteriors(posteriorScale);
	}

	if (maxMinPosterior == LogP_Zero) {
	    dout() << "Lattice::prunePosteriors: "
		   << "no paths left in lattice\n";
	    return false;
	}

	/*
	 * If a posterior threshold > 0 was specified, prune accordingly.
	 * Otherwise find the maximum threshold that wouldn't prune anything,
	 * so that density pruning can proceed.
	 */
	LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
	NodeIndex nodeIndex;

	Prob minimumPosterior = 1.0;

	while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	    Prob normalizedPosterior =
			LogPtoProb(node->posterior - maxMinPosterior);
	    if (threshold > 0.0) {
		if (normalizedPosterior < threshold) {
		    removeNode(nodeIndex);
		}
	    } else {
		if (normalizedPosterior < minimumPosterior && normalizedPosterior > 0.0) {
		    minimumPosterior = normalizedPosterior;
		}
	    }
	}

	if (threshold <= 0.0) {
	    threshold = minimumPosterior;

	    if (debug(DebugPrintFunctionality)) {
		dout() << "Lattice::prunePosteriors: "
		       << "initial posterior theshold set to " << threshold
		       << endl;
	    }
	}

	if (fast) {
	    removeUselessNodes();
	}

	unsigned newNumNodes = getNumNodes();

	if (newNumNodes == numNodes) {
	    if (maxDensity > 0.0 || maxNodes > 0) {
		unsigned numRealNodes;
		double d = computeDensity(numRealNodes);
		if (maxDensity > 0.0 && d == HUGE_VAL) {
		    dout() << "Lattice::prunePosteriors: "
			   << "cannot compute lattice density\n";
		    break;
		} else if ((maxDensity > 0.0 && d > maxDensity) ||
			   (maxNodes > 0 && numRealNodes > maxNodes)) 
		{
		    threshold = max(threshold * 1.25893,
				    nextafter(threshold, Prob(1.0)));  // * 10^(1/10) or minimum increment 

		    // avoid disconnecting lattice, keep threshold below 1
		    if (threshold > 1.0) {
			break;
		    }

		    if (debug(DebugPrintFunctionality)) {
			dout() << "Lattice::prunePosteriors: "
		               << "density = " << d << ", "
			       << numRealNodes << " real nodes, "
			       << "increasing threshold to "
			       << threshold << "\n";
		    }
		} else {
		    if (debug(DebugPrintFunctionality)) {
			dout() << "Lattice::prunePosteriors: "
		               << "density = " << d << ", "
			       << numRealNodes << " real nodes, "
			       << "stopping\n";
		    }
		    break;
		}
	    } else {
		break;
	    }
	}
	numNodes = newNumNodes;

	if (debug(DebugPrintFunctionality)) {
	    dout() << "Lattice::prunePosteriors: "
		   <<  "left with " << numNodes << " nodes\n";
	}
    }

    return numNodes > 0;
}

/*
 * Determine duration of lattice
 *	If the lattice has HTK information, use the start/end node times
 *	If not, look for (duration=D) string in lattice name.
 *	Otherwise give up and return 0.
 */
double
Lattice::getDuration()
{
    // try to determine lattice duration from HTK info
    LatticeNode *initialNode = findNode(initial);
    assert(initialNode != 0);
    LatticeNode *finalNode = findNode(final);
    assert(finalNode != 0);

    if (initialNode->htkinfo != 0 &&
	initialNode->htkinfo->time != HTK_undef_float &&
	finalNode->htkinfo != 0 &&
	finalNode->htkinfo->time != HTK_undef_float)
    {
	return finalNode->htkinfo->time - initialNode->htkinfo->time;
    } else {
	return duration;
    }
}

/*
 * Compute lattice density (number of non-null nodes/unit time).
 * Also returns number of non-null nodes.
 */
double
Lattice::computeDensity(unsigned &numNodes, double dur)
{
    if (dur == 0.0) {
	dur = getDuration();
    }

    if (dur == 0.0) {
	return HUGE_VAL;
    }

    // count non-null nodes
    unsigned numNonNulls = 0;

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    NodeIndex nodeIndex;

    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	if (node->word != Vocab_None &&
	    node->word != vocab.ssIndex() &&
	    node->word != vocab.seIndex())
	{
	    numNonNulls ++;
	}
    }

    numNodes = numNonNulls;

    return numNonNulls / dur;
}

/*
 * Compute forward and backward Viterbi pointers
 */
LogP
Lattice::computeViterbi(NodeIndex forwardPreds[], NodeIndex *backwardPreds)
{
    /*
     * topological sort
     */
    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
	dout() << "Lattice::computeViterbi: warning: called with unreachable nodes\n";
    }
    if (sortedNodes[0] != initial) {
	dout() << "Lattice::computeViterbi: initial node is not first\n";
	delete [] sortedNodes;
        return LogP_Inf;
    }

    unsigned finalPosition = 0;
    for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
	if (sortedNodes[finalPosition] == final) break;
    }
    if (finalPosition == numReachable) {
	dout() << "Lattice::computeViterbi: final node is not reachable\n";
	delete [] sortedNodes;
        return LogP_Inf;
    }

    LogP *viterbiProbs = new LogP[maxIndex];		// maximum prob to node
    assert(viterbiProbs != 0);

    /*
     * compute forward Viterbi probabilities and best predecessors
     */
    for (unsigned i = 0; i < maxIndex; i ++) {
	    viterbiProbs[i] = LogP_Zero;
	    forwardPreds[i] = NoNode;
    }
    viterbiProbs[initial] = LogP_One;

    for (unsigned position = 1; position < numReachable; position ++) {
	NodeIndex nodeIndex = sortedNodes[position];
	LatticeNode *node = nodes.find(nodeIndex);
	assert(node != 0);

	LogP max = LogP_Zero;
	NodeIndex best = NoNode;

	TRANSITER_T<NodeIndex,LatticeTransition>
					inTransIter(node->inTransitions);
	LatticeTransition *inTrans;
	NodeIndex fromNodeIndex; 
	while ((inTrans = inTransIter.next(fromNodeIndex))) {
	    LogP prob = viterbiProbs[fromNodeIndex] + inTrans->weight;

	    if (prob > max) {
		max = prob;
		best = fromNodeIndex;
	    }
	}
	viterbiProbs[nodeIndex] = max;
	forwardPreds[nodeIndex] = best;
    }

    LogP result = viterbiProbs[final];

    if (backwardPreds != 0) {
	/*
	 * compute backward Viterbi probabilities and best predecessors
	 */
	for (unsigned i = 0; i < maxIndex; i ++) {
		viterbiProbs[i] = LogP_Zero;
		backwardPreds[i] = NoNode;
	}
	viterbiProbs[final] = LogP_One;

	for (int position = numReachable - 2; position >= 0; position --) {
	    NodeIndex nodeIndex = sortedNodes[position];
	    LatticeNode *node = nodes.find(nodeIndex);
	    assert(node != 0);

	    LogP max = LogP_Zero;
	    NodeIndex best = NoNode;

	    TRANSITER_T<NodeIndex,LatticeTransition>
					    outTransIter(node->outTransitions);
	    LatticeTransition *outTrans;
	    NodeIndex toNodeIndex; 
	    while ((outTrans = outTransIter.next(toNodeIndex))) {
		LogP prob = viterbiProbs[toNodeIndex] + outTrans->weight;

		if (prob > max) {
		    max = prob;
		    best = toNodeIndex;
		}
	    }
	    viterbiProbs[nodeIndex] = max;
	    backwardPreds[nodeIndex] = best;
	}
    }

    delete [] sortedNodes;
    delete [] viterbiProbs;

    return result;
}

/*
 * Compute word sequence with highest probability path through latttice
 */
LogP
Lattice::bestWords(VocabIndex *words, unsigned maxWords, SubVocab &ignoreWords)
{
    NBestWordInfo *winfo = new NBestWordInfo[maxWords];
    assert(winfo != 0);

    LogP result = bestWords(winfo, maxWords, ignoreWords);

    for (unsigned i = 0; i < maxWords; i ++) {
	words[i] = winfo[i].word;

	if (words[i] == Vocab_None) break;
    }

    delete [] winfo;

    return result;
}

/* 
 * Generalized version returning acoustic and timing information
 */
LogP
Lattice::bestWords(NBestWordInfo *winfo, unsigned maxWords,
							SubVocab &ignoreWords)
{
    /*
     * Algorithm: 
     * 0) sort nodes in topological order
     * 1) forward pass: compute viterbi probabilities
     * 2) backward pass: backtrace highest probability path
     * Note: we allocate large arrays on the heap rather than the stack
     * to avoid running into resource limits.
     */
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::bestWords: processing\n";
    }

    NodeIndex *viterbiPreds = new NodeIndex[maxIndex];
    assert(viterbiPreds != 0);

    LogP result = computeViterbi(viterbiPreds);

    if (result == LogP_Inf) {
	delete [] viterbiPreds;
	winfo[0].word = Vocab_None;
	return LogP_Zero;
    }

    NodeIndex *nodesOnPath = new NodeIndex[maxIndex + 1];
    assert(nodesOnPath != 0);

    unsigned numWords = 0;
    unsigned numNodes = 0;
    NodeIndex current = final;
    do {
	LatticeNode *node = nodes.find(current);

	if (node->word != Vocab_None) {
	    nodesOnPath[numNodes ++] = current;
		
	    if (!ignoreWord(node->word) && !ignoreWords.getWord(node->word)) {
	        numWords ++;
	    }
	}
	current = viterbiPreds[current];
    } while (current != NoNode);

    if (numWords > maxWords) {
	// not enough room for words, return zero length string and log(0)
        dout() << "Lattice::bestWords: "
	       << "word string longer than " << maxWords << endl;

	winfo[0].word = Vocab_None;
	result = LogP_Zero;
    } else {
	// reverse and copy node information into result buffer
	unsigned i = numWords;		// word index

	if (i < maxWords) {
	    winfo[i].word = Vocab_None;
	}

	for (unsigned j = 0; j < numNodes; j ++) {
	    LatticeNode *node = nodes.find(nodesOnPath[j]);

	    if (!ignoreWord(node->word) && !ignoreWords.getWord(node->word)) {
		winfo[--i].word = node->word;

		if (node->htkinfo == 0) {
		    winfo[i].invalidate();
		} else {
		    // find the time of the predecessor node
		    float startTime = 0.0;
		    LatticeNode *predNode;

		    if (j < numNodes - 1 &&
			(predNode = nodes.find(nodesOnPath[j+1])) &&
			predNode->htkinfo)
		    {
			startTime = predNode->htkinfo->time;
		    }

		    winfo[i].start = startTime;
		    winfo[i].duration = node->htkinfo->time - startTime;
		    winfo[i].acousticScore = node->htkinfo->acoustic;

		    if (node->htkinfo->language != HTK_undef_float) {
			winfo[i].languageScore = node->htkinfo->language;
		    } else {
			winfo[i].languageScore = node->htkinfo->ngram;
		    }
		}

		winfo[i].wordPosterior = winfo[i].transPosterior = 1.0;
	    }
	}
    }

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::bestWords: "
	     << "best path prob = " << result << endl;
    }

    delete [] viterbiPreds;
    delete [] nodesOnPath;

    return result;
}

/* check connectivity of this lattice given two nodeIndices.
 * it can go either forward or backward directions.
 */
Boolean
Lattice::areConnected(NodeIndex fromNodeIndex, NodeIndex toNodeIndex, 
		      unsigned direction)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::areConnected: "
	     << "processing (" << fromNodeIndex << "," << toNodeIndex
	     << ")\n";
    }

    if (fromNodeIndex == toNodeIndex) {
      if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::areConnected: "
	       << "(" << fromNodeIndex << "," << toNodeIndex << ") are connected\n";
      }
      return true;
    }
     
    clearMarkOnAllNodes(markedFlag);

    int i = 0; 
    NodeQueue nodeQueue; 
    nodeQueue.addToNodeQueue(fromNodeIndex); 

    // use width first approach to go through the whole lattice.
    // mark the first level nodes and put them in the queue.
    // going through the queue to process all the nodes in the lattice
    while (nodeQueue.is_empty() == false) {
      NodeIndex nodeIndex = nodeQueue.popNodeQueue();

      // extend one more level in lattice and put the nodes 
      // there to the queue.
      LatticeNode * node = findNode(nodeIndex);
      if (!node) {
	if (debug(DebugPrintOutLoop)) {
	  dout() << "NonFatal Error in Lattice::areConnected: "
		 << "can't find an existing node (" << nodeIndex << ")\n";
	}
	continue;
      }

      // TRANSITER_T<NodeIndex,LatticeTransition> 
      // outTransIter(node->outTransitions);
      TRANSITER_T<NodeIndex,LatticeTransition> 
	outTransIter(!direction ? node->outTransitions : node->inTransitions);
      NodeIndex nextNodeIndex;
      while (outTransIter.next(nextNodeIndex)) {
	// processing nodes at the next level 

	if (nextNodeIndex == toNodeIndex) {

	  if (debug(DebugPrintFunctionality)) {
	    dout() << "Lattice::areConnected: "
		   << "(" << fromNodeIndex << "," << toNodeIndex << ") are connected\n";
	  }
	  return true; 
	}

	LatticeNode *nextNode = findNode(nextNodeIndex);
	if (nextNode->getFlag(markedFlag)) {
	  continue;
	}

	nextNode->markNode(markedFlag); 
        nodeQueue.addToNodeQueue(nextNodeIndex); 
      }
    }

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::areConnected: "
	     << "(" << fromNodeIndex << "," << toNodeIndex << ") are NOT connected\n";
    }
    return false; 
}

Boolean
Lattice::computeNodeEntropy()
{

    NodeIndex nodeIndex;

    double fanInEntropy = 0.0, fanOutEntropy = 0.0;
    int total = 0;

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {

      if (debug(DebugPrintInnerLoop)) {
	dout() << "Lattice::computeNodeEntropy: processing nodeIndex " 
	       << nodeIndex << "\n";
      }

      int fanOut = node->outTransitions.numEntries();
      int fanIn = node->inTransitions.numEntries();

      total += fanIn;

      if (nodeIndex == final) { 
	fanOut = 0; 
      }

      if (nodeIndex == initial) {
	fanIn = 0;
      }

      if (fanOut) {
	fanOutEntropy += (double) fanOut*log10((double)fanOut); 
      }

      if (fanIn) {
	fanInEntropy += (double) fanIn*log10((double)fanIn); 
      }
    }

    fanOutEntropy = log10((double)total) - fanOutEntropy/total; 
    fanInEntropy = log10((double)total) - fanInEntropy/total; 

    double uniform = log10((double)getNumNodes()); 

    printf("The fan-in/out/uniform entropies are ( %f %f %f )\n", 
	   fanInEntropy, fanOutEntropy, uniform); 

    return true;

}

void
Lattice::splitMultiwordNodes(MultiwordVocab &vocab, LM &lm)
{
    VocabIndex emptyContext[1];
    emptyContext[0] = Vocab_None;

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::splitMultiwordNodes: splitting multiword nodes\n";
    }

    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    for (unsigned i = 0; i < numReachable; i++) {
      NodeIndex nodeIndex = sortedNodes[i];
      LatticeNode *node = findNode(nodeIndex); 

      VocabIndex oneWord[2];
      oneWord[0] = node->word;
      oneWord[1] = Vocab_None;
      VocabIndex expanded[maxWordsPerLine + 1];

      unsigned expandedLength =
		vocab.expandMultiwords(oneWord, expanded, maxWordsPerLine);

      /*
       * We don't split multiwords that are in the LM
       */
      if (expandedLength < 2 || ignoreWord(node->word)) {
	;;	// nothing to do
      } else if (lm.wordProb(node->word, emptyContext) > LogP_Zero) {
	VocabString w = vocab.getWord(node->word);
	dout() << "warning: not splitting multiword "
	       << (w != 0 ? w : "NULL") << " because LM prob > 0\n";
      } else {
	// change orignal node to emit first component word
	node->word = expanded[0];

	// update the HTKWordInfo information attached to this node
	HTKWordInfo *htkinfo = node->htkinfo;

	if (htkinfo != 0) {
	    htkinfo->word = expanded[0];
	}

	NodeIndex prevNodeIndex = nodeIndex;
	NodeIndex firstNewIndex;
	
	// create new nodes for all subsequent word components, and
	// string them together with zero weight transitions
	for (unsigned i = 1; i < expandedLength; i ++) {

	    HTKWordInfo *newinfo = 0;

	    /* 
	     * if original node hat HTKWordInfo attached, create dummy
	     * HTKWordInfo for new node
	     */
	    if (htkinfo != 0) {
		newinfo = new HTKWordInfo;
		assert(newinfo != 0);

		htkinfos[htkinfos.size()] = newinfo;
		newinfo->word = expanded[i];

		if (htkinfo->time != HTK_undef_float) {
		    // make all the multiword components have the same time
		    newinfo->time = htkinfo->time;
		}

		// scores of all subsequent components are 0 since the first
		// component carries the full scores
		if (htkinfo->acoustic != HTK_undef_float) {
		    newinfo->acoustic = LogP_One;
		}
		if (htkinfo->language != HTK_undef_float) {
		    newinfo->language = LogP_One;
		}
		if (htkinfo->ngram != HTK_undef_float) {
		    newinfo->ngram = LogP_One;
		}
		if (htkinfo->pron != HTK_undef_float) {
		    newinfo->pron = LogP_One;
		}
	    }

	    NodeIndex newNodeIndex = dupNode(expanded[i], 0, newinfo);

	    // delay inserting the first new transition to not interfere
	    // with removal of old links below
	    if (prevNodeIndex == nodeIndex) {
		firstNewIndex = newNodeIndex;
	    } else {
	        LatticeTransition trans;
	        insertTrans(prevNodeIndex, newNodeIndex, trans);
	    }
	    prevNodeIndex = newNodeIndex;
	}

	// node may have moved since others were added!!!
        node = findNode(nodeIndex); 

	// copy original outgoing transitions onto final new node
        TRANSITER_T<NodeIndex,LatticeTransition>
				transIter(node->outTransitions);
	NodeIndex toNodeIndex;
        while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
	    // prevNodeIndex still has the last of the newly created nodes
	    insertTrans(prevNodeIndex, toNodeIndex, *trans);
	    removeTrans(nodeIndex, toNodeIndex);
	}

	// now insert new transition out of original node
	{
	    LatticeTransition trans;
	    insertTrans(nodeIndex, firstNewIndex, trans);
	}
      }
    }

    delete [] sortedNodes;
}

