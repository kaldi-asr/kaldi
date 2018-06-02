/*
 * LatticeAlign.cc --
 *	Multiple alignment of lattice paths
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2003-2010 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeAlign.cc,v 1.17 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

#include <time.h>

#include "Lattice.h"
#include "WordMesh.h"

#include "LHash.cc"

#ifdef INSTANTIATE_TEMPLATES
#ifdef USE_SHORT_VOCAB
INSTANTIATE_LHASH(NodeIndex,LogP2);
#endif
#endif

#define DebugPrintFunctionality         1

#define DebugProgressModulus		100	// how often to print progress messages

#ifdef DEBUG_LATTICE_ALIGNMENT
static void
printPath(NodeIndex *path, unsigned len)
{
    for (unsigned i = 0; i < len; i ++) {
	if (path[i] == NoNode) {
	    cerr << "NoNode" << " ";
	} else {
	    cerr << path[i] << " ";
	}
    }
}
#endif


/*
 * This ugly thing is the body of an iteration to add one node (and its word)
 * to the path to be aligned.
 */
static inline void
addNodeToPath(Lattice &lat, SubVocab &ignoreWords, NodeIndex thisNode,
	      unsigned &numWords, 
	      NodeIndex &lastNode, NodeIndex &lastWordNode,
	      LogP &probToLastWord,
	      LogP2 forwardProbs[], LogP2 backwardProbs[],
	      LogP2 totalPosterior, double posteriorScale,
	      NodeIndex pathNodes[], NBestWordInfo pathWords[],
	      Boolean acousticInfo)
{
    if (thisNode == NoNode) {
	pathNodes[numWords] = NoNode;
	pathWords[numWords].invalidate();
	pathWords[numWords].word = lat.vocab.unkIndex();
	pathWords[numWords].wordPosterior = 0.0;
	pathWords[numWords].transPosterior = 0.0;

	numWords ++;
	return;
    }

    if (lastNode != NoNode) {
	LatticeTransition *trans = lat.findTrans(lastNode, thisNode);
	assert(trans != 0);

	probToLastWord += trans->weight;
    }

    LatticeNode *node = lat.findNode(thisNode);
    if (node->word != Vocab_None &&
        !lat.ignoreWord(node->word) &&
	!ignoreWords.getWord(node->word))
    {
	LogP2 transPosterior = LogP_Zero;

	if (lastWordNode != NoNode) {
	    transPosterior = forwardProbs[lastWordNode] + 
				probToLastWord/posteriorScale +
				backwardProbs[thisNode]
			     - totalPosterior;
	    probToLastWord = LogP_One;
	}

	pathNodes[numWords] = thisNode;

    	pathWords[numWords].invalidate();
	pathWords[numWords].word = node->word;
	pathWords[numWords].wordPosterior =
	    pathWords[numWords].transPosterior = LogPtoProb(node->posterior);
	if (acousticInfo && node->htkinfo) {
	    LatticeNode *node0;
	    if (lastNode != NoNode &&
		(node0 = lat.findNode(lastNode)) && node0->htkinfo)
	    {
		pathWords[numWords].start = node0->htkinfo->time;
		pathWords[numWords].duration = node->htkinfo->time - 
						    node0->htkinfo->time;
	    }
	    pathWords[numWords].acousticScore = node->htkinfo->acoustic;
	    pathWords[numWords].languageScore = node->htkinfo->language;
	    if (node->htkinfo->div) {
		pathWords[numWords].phones = strdup(node->htkinfo->div);
	    }
	    /*
	     * A hack requested by Kemal for WS'04:
	     * Since the phone durations are already included in
	     * the phone string field, use the phoneDuration field to 
	     * encode the duration *model* scores, if defined.
	     */
	    if (node->htkinfo->duration != HTK_undef_float)  {
		char durationScore[100];
		sprintf(durationScore, "%.*lg",
			LogP_Precision, (double)node->htkinfo->duration);
		pathWords[numWords].phoneDurs = strdup(durationScore);
	    }
	}

	if (numWords > 0) {
	    pathWords[numWords-1].transPosterior = LogPtoProb(transPosterior);
#ifdef DEBUG_LATTICE_ALIGNMENT
	    cerr << pathWords[numWords-1].transPosterior << " ";
#endif
	}

#ifdef DEBUG_LATTICE_ALIGNMENT
	cerr << lat.vocab.getWord(node->word) << " ";
#endif
	numWords ++;

	lastWordNode = thisNode;
    }

    lastNode = thisNode;
}

/*
 * Align all word nodes in a lattice into a sausage (which may already 
 * contain a partial alignment).
 * Words contained in SubVocab ignoreWords are not included in the alignment.
 * The algorithm iteratively aligns all paths in the lattices so as to
 * minimize expected word error, using WordMesh::alignWords().
 *
 * Let N := set of all lattice nodes 
 *
 * while N is not empty
 *	n = node with highest posterior in N
 *	p = path through n with highest probability 
 *	p' = portion of p containing nodes in N
 *	align p' to sausage (*)
 *	N = N - nodes in p'
 *
 * Step (*) incrementally builds a mapping of lattice nodes to alignment
 * positions (sausage bins).  This mapping allows us to find the position
 * between which the partial paths p' must be aligned.  This way, we
 * only align each node once, which is essential to correctly transfer
 * the node posterior probabilities to the sausage, but also to minimize
 * the total amount of work.
 */
void
Lattice::alignLattice(
    WordMesh &sausage, 
    SubVocab &ignoreWords,
    double posteriorScale, 
    Boolean acousticInfo) 
{
    LHash<NodeIndex,unsigned> latticeNodeToUnsortedMeshMap;
    LHash<NodeIndex,VocabIndex> latticeNodeToVocabMap;

    alignLattice(sausage, ignoreWords, latticeNodeToUnsortedMeshMap, latticeNodeToVocabMap, 
        posteriorScale, acousticInfo);
}

void
Lattice::alignLattice(
    WordMesh &sausage, 
    SubVocab &ignoreWords,
    LHash<NodeIndex,unsigned>& nodeToAlignMap,
    LHash<NodeIndex,VocabIndex>& latticeNodeToVocabMap,
    double posteriorScale, 
    Boolean acousticInfo) 
{
    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::alignLattice: aligning "
	       << getName() << " with " << getNumNodes() << " nodes\n"; 
    }

    LHash<NodeIndex,LogP2> nodesNotAligned;


    /*
     * Compute forward/backward and viterbi information
     */
    LogP2 *forwardProbs = new LogP2[maxIndex];
    LogP2 *backwardProbs = new LogP2[maxIndex];
    assert(forwardProbs != 0 && backwardProbs != 0);

    LogP maxMinPosterior = computeForwardBackward(forwardProbs, backwardProbs,
								posteriorScale);

    if (maxMinPosterior == LogP_Zero) {
	delete [] backwardProbs;
	delete [] forwardProbs;
	return;
    }

    NodeIndex *forwardPreds = new NodeIndex[maxIndex];
    NodeIndex *backwardPreds = new NodeIndex[maxIndex];
    assert(forwardPreds != 0 && backwardPreds != 0);

    LogP maxProb = computeViterbi(forwardPreds, backwardPreds);

    if (maxProb == LogP_Inf) {
	delete [] backwardPreds;
	delete [] forwardPreds;
	delete [] backwardProbs;
	delete [] forwardProbs;
	return;
    }

    /*
     * Normalize node posteriors and initialize nodesNotAligned map
     */
    LogP2 totalPosterior = forwardProbs[final];

    {
	LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
	NodeIndex nodeIndex;

	while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	    node->posterior -= totalPosterior;

	    // only align non-null nodes
	    if (node->word != Vocab_None &&
	    	!ignoreWord(node->word) &&
	        !ignoreWords.getWord(node->word))
	    {
		*nodesNotAligned.insert(nodeIndex) = node->posterior;
	    }
	}
    }

    /*
     * Main loop
     */

    unsigned numToAlign = nodesNotAligned.numEntries();
    unsigned numUnaligned;
    unsigned numIter = 0;

    NodeIndex *pathToStart = new NodeIndex[maxIndex + 1];
    NodeIndex *pathToEnd = new NodeIndex[maxIndex + 1];
    assert(pathToStart != 0 && pathToEnd != 0);

    NBestWordInfo *pathWords = new NBestWordInfo[maxIndex + 3];
    NodeIndex *pathNodes = new NodeIndex[maxIndex + 3];
    unsigned *pathPositions = new unsigned[maxIndex + 3];
    assert(pathWords != 0 && pathNodes != 0 && pathPositions != 0);

    while ((numUnaligned = nodesNotAligned.numEntries()) > 0) {
        numIter += 1;

	if (debug(DebugPrintFunctionality) && numIter % DebugProgressModulus == 0) {
	    dout() << "Lattice::alignLattice: " << (numToAlign-numUnaligned) << " aligned"
	           << " (" << (100.0 * (numToAlign - numUnaligned) / numToAlign) << "%)\r";
	}

	LogP2 maxPosterior;
	NodeIndex nextNodeToAlign =
			findMaxPosteriorNode(nodesNotAligned, maxPosterior);

#ifdef DEBUG_LATTICE_ALIGNMENT
	cerr << "nextNodeToAlign = " << nextNodeToAlign << endl;
#endif

	unsigned lengthToStart =
		findFirstAligned(nextNodeToAlign, forwardPreds,
				 nodeToAlignMap, pathToStart);
	unsigned lengthToEnd = 
		findFirstAligned(nextNodeToAlign, backwardPreds,
				 nodeToAlignMap, pathToEnd);
#ifdef DEBUG_LATTICE_ALIGNMENT
	cerr << "toPath = " ; printPath(pathToStart, lengthToStart);
	cerr << endl;
	cerr << "fromPath = " ; printPath(pathToEnd, lengthToEnd);
	cerr << endl;
#endif

	/*
	 * Assemble the word string to align for the complete path.
	 * Note: we skip NULL nodes since they are not handled by WordMesh
	 */
	unsigned numWords = 0;
	NodeIndex lastNode = NoNode;
	NodeIndex lastWordNode = NoNode;
	LogP probToLastWord = LogP_One;

#ifdef DEBUG_LATTICE_ALIGNMENT
	cerr << "words to align = ";
#endif
	for (int i = lengthToStart - 1; i >= 0; i --) {
	    addNodeToPath(*this, ignoreWords, pathToStart[i],
			    numWords, lastNode, lastWordNode, probToLastWord,
			    forwardProbs, backwardProbs,
			    totalPosterior, posteriorScale,
			    pathNodes, pathWords, acousticInfo);
	}
	
	    addNodeToPath(*this, ignoreWords, nextNodeToAlign,
			    numWords, lastNode, lastWordNode, probToLastWord,
			    forwardProbs, backwardProbs,
			    totalPosterior, posteriorScale,
			    pathNodes, pathWords, acousticInfo);

	for (unsigned i = 0; i < lengthToEnd; i ++) {
	    addNodeToPath(*this, ignoreWords, pathToEnd[i],
			    numWords, lastNode, lastWordNode, probToLastWord,
			    forwardProbs, backwardProbs,
			    totalPosterior, posteriorScale,
			    pathNodes, pathWords, acousticInfo);
	}
#ifdef DEBUG_LATTICE_ALIGNMENT
	cerr << endl;
#endif
	// Eliminate the last Word, which we included only as a dummy
	// to obtain the final transition posterior.
	// The final entry will instead be used to encode the transition
	// posterior INTO the FIRST word.
	numWords --;
	pathWords[numWords].word = Vocab_None;
	pathWords[numWords].invalidate();
	pathWords[numWords].transPosterior = pathWords[0].transPosterior;

	/*
	 * Find start/end position within the existing alignment
	 */
	unsigned startPos = sausage.length();
	if (pathToStart[lengthToStart-1] != NoNode) {
	    unsigned *mappedPos =
			nodeToAlignMap.find(pathToStart[lengthToStart-1]);
	    assert(mappedPos != 0);

	    startPos = *mappedPos;
	}

	unsigned endPos = sausage.length();
	if (pathToEnd[lengthToEnd-1] != NoNode) {
	    unsigned *mappedPos =
			nodeToAlignMap.find(pathToEnd[lengthToEnd-1]);
	    assert(mappedPos != 0);

	    endPos = *mappedPos;
	}
#ifdef DEBUG_LATTICE_ALIGNMENT
	cerr << "startpos = " << startPos << " endpos = " << endPos << endl;
#endif

	/* 
	 * Align me, baby!
	 */
	if (!sausage.alignWords(&pathWords[1], LogPtoProb(maxPosterior),
			        0, 0, startPos, endPos, &pathPositions[1])) {
	    /*
	     * alignment failed, probably because lattice topology was
	     * violated.  Remove nextNodeToAlign from the nodesNotAligned set,
	     * so we don't try it again, but don't add anything to the
	     * nodeToAlignMap.  In the worst case the node remains unaligned.
	     */
	    cerr << "Lattice::alignLattice: warning: failed to align "
		 << (numWords - 1) << " word(s), "
		 << "max posterior = " << LogPtoProb(maxPosterior) << endl;
	    nodesNotAligned.remove(nextNodeToAlign);
	} else {
#ifdef DEBUG_LATTICE_ALIGNMENT
	    cerr << "NEW WORD MESH\n";
	    {  File f(stderr); sausage.write(f); }
	    cerr << "END WORD MESH\n";
	    cerr << "alignment = ";
#endif
	    /*
	     * Remove aligned nodes from nodesNotAligned
	     * and extend the nodeToAlignMap
	     */
	    for (unsigned k = 1; k < numWords; k ++) {
#ifdef DEBUG_LATTICE_ALIGNMENT
		cerr << " " << vocab.getWord(pathWords[k].word)
		     << "/" << pathNodes[k]
		     << "/" << pathPositions[k];
#endif
		nodesNotAligned.remove(pathNodes[k]);
		*nodeToAlignMap.insert(pathNodes[k]) = pathPositions[k];
                *latticeNodeToVocabMap.insert(pathNodes[k]) = pathWords[k].word;
	    }
#ifdef DEBUG_LATTICE_ALIGNMENT
	    cerr << endl;
#endif

	}
    }

    if (debug(DebugPrintFunctionality) && numIter >= DebugProgressModulus) {
	dout() << "Lattice::alignLattice: " << numToAlign << " aligned"
	       << " (100%)          \n";
    }

    // fix up the null posteriors (due to not aligning all the transitions)
    sausage.normalizeDeletes();

    delete [] pathToStart;
    delete [] pathToEnd;

    delete [] pathWords;
    delete [] pathNodes;
    delete [] pathPositions;

    delete [] forwardProbs;
    delete [] backwardProbs;

    delete [] forwardPreds;
    delete [] backwardPreds;
}


/*
 * Helper functions
 */

NodeIndex
Lattice::findMaxPosteriorNode(LHash<NodeIndex, LogP2> &nodeSet, LogP2 &max)
{
    NodeIndex maxNode = NoNode;
    LogP2 maxPosterior = LogP_Zero;

    LHashIter<NodeIndex, LogP2> iter(nodeSet);
    NodeIndex node;
    LogP2 *nodePosterior;

    while ((nodePosterior = iter.next(node))) {
	if (maxNode == NoNode || *nodePosterior > maxPosterior) {
	    maxPosterior = *nodePosterior;
	    maxNode = node;
	}
    }

    max = maxPosterior;
    return maxNode;
}

/*
 * Backtrace through the predecessors array starting at node 'from',
 * up to the first node in nodeSet.
 * Return the sequence of nodes traversed in 'path'.
 */
unsigned
Lattice::findFirstAligned(NodeIndex from, NodeIndex predecessors[],
			    LHash<NodeIndex, unsigned> &nodeSet,
			    NodeIndex pathNodes[])
{
    unsigned pathLength = 0;
    NodeIndex current = from;

    do {
	current = predecessors[current];
	pathNodes[pathLength ++] = current;
    } while (current != NoNode && !nodeSet.find(current));

    return pathLength;
}


