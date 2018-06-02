/*
 * LatticeNgrams.cc --
 *	Expected N-gram count computation on lattices
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2005-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeNgrams.cc,v 1.11 2010/06/02 05:54:08 stolcke Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "Lattice.h"

#include "Map2.cc"
#include "Array.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_MAP2(NodeIndex, const NBestWordInfo *, LogP2);
#endif

#define DebugPrintFatalMessages         0 

#define DebugPrintFunctionality         1 
// for large functionality listed in options of the program
#define DebugPrintOutLoop               2
// for out loop of the large functions or small functions
#define DebugPrintInnerLoop             3
// for inner loop of the large functions or outloop of small functions


/*
 * Compute N-gram count expectations over a lattice.
 * Returns expected unigram count sum.
 *
 * Algorithm: 
 *	1. compute backward probabilities
 *	2. compute forward probabilities while dynamically expanding
 *	   lattice to reflect (N-1)-gram context nodes
 *	   (the forward probabilities for each expanded node are stored,
 *	   but not the expanded lattice itself).
 *
 * The main data structure is a Map2 of the form
 *	(original node, Ngram-context) -> forward probability
 * The forward probs are combined with transition and backward probs to
 * compute expected counts, which are accumulated in a global N-gram trie.
 *
 * Nodes with NULL or pause are handled by ignoring them when in the final
 * position of the N-gram, and skipping them when computing N-gram contexts.
 */
Prob
Lattice::countNgramsAtNode(VocabIndex oldIndex, unsigned order,
			   LogP2 backwardProbs[], double posteriorScale,
			   Map2<NodeIndex, const NBestWordInfo *, LogP2> &forwardProbMap,
			   Lattice::NgramAccumulatorFunction *accumulator,
			   void *clientData, Boolean acousticInfo)
{
    LatticeNode *oldNode = findNode(oldIndex);
    assert(oldNode != 0);

    Map2Iter2<NodeIndex, const NBestWordInfo *, LogP2>
					expandIter(forwardProbMap, oldIndex);
    Prob countSum = 0.0;

    LogP2 *forwardProb;
    const NBestWordInfo *context;

    while ((forwardProb = expandIter.next(context))) {
	unsigned contextLength = NBestWordInfo::length(context);

	makeArray(NBestWordInfo, extendedContext, contextLength + 2);
	NBestWordInfo::copy(extendedContext, context);

	TRANSITER_T<NodeIndex,LatticeTransition> 
			      transIter(oldNode->outTransitions);
	NodeIndex nextIndex;
	while (LatticeTransition *oldTrans = transIter.next(nextIndex)) {
	    LatticeNode *nextNode = findNode(nextIndex);
	    assert(nextNode != 0);

	    VocabIndex word = nextNode->word;

	    LogP transProb = oldTrans->weight/posteriorScale;
	    LogP2 backwardProb = backwardProbs[nextIndex];

	    LogP posterior = *forwardProb + transProb + backwardProb;

	    // normalize by posterior of whole lattice
	    posterior -= backwardProbs[initial];

	    // fix some obvious numerical problems
	    if (posterior > LogP_One) posterior = LogP_One;

	    // determine ngram context for sucessor nodes
	    const NBestWordInfo *newContext;
	    if (ignoreWord(word)) {
		// context is unchanged
		newContext = context;
	    } else {
		// construct context extended by one word
		extendedContext[contextLength].word = word;
		// fill in acoustic information
		if (acousticInfo && nextNode->htkinfo) {
		    if (oldNode->htkinfo &&
		        oldNode->htkinfo->time != HTK_undef_float &&
			nextNode->htkinfo->time != HTK_undef_float)
		    {
			extendedContext[contextLength].start =
						oldNode->htkinfo->time;
			extendedContext[contextLength].duration =
						nextNode->htkinfo->time -
						    oldNode->htkinfo->time;
		    }
		    extendedContext[contextLength].acousticScore =
						nextNode->htkinfo->acoustic;
		    extendedContext[contextLength].languageScore =
						nextNode->htkinfo->language;
		    if (nextNode->htkinfo->div) {
		    	if (extendedContext[contextLength].phones) {
			    free(extendedContext[contextLength].phones);
			}
			extendedContext[contextLength].phones =
						strdup(nextNode->htkinfo->div);
		    }
		}
		extendedContext[contextLength+1].word = Vocab_None;

		// decide if extended context needs to be shortened
		if (contextLength < order - 1) {
		    newContext = extendedContext;
		} else {
		    newContext = extendedContext + 1;
		}
	    }

	    // accumulate forward probability for successor context
	    Boolean found;
	    LogP2 *nextProb =
			forwardProbMap.insert(nextIndex, newContext, found);
	    if (!found) {
		*nextProb = *forwardProb + transProb;
	    } else {
		*nextProb = AddLogP(*nextProb, *forwardProb + transProb);
	    }

	    // accumulate ngram counts, but only if the successor word
	    // is not one to be ignored.
	    // (note we can reuse the extended context buffer from above)
	    if (!ignoreWord(word)) {
		Prob count = LogPtoProb(posterior);

		for (NBestWordInfo *ngram = extendedContext;
		     ngram->word != Vocab_None;
		     ngram ++)
		{
		    (*accumulator)(this, ngram, count, clientData);
		}

		countSum += count;
	    }
	}
    }

    // save space by deleting entries that are no longer used
    forwardProbMap.remove(oldIndex);

    return countSum;
}

//
// Callback function for Lattice::countNgramsAtNode():
//	sum ngram counts globally
//
static void
ngramCountAccumulator(Lattice *lat, const NBestWordInfo *ngram, Prob count,
						NgramCounts<FloatCount> *counts)
{
    makeArray(VocabIndex, words, NBestWordInfo::length(ngram)+1);

    NBestWordInfo::copy(words, ngram);

    *counts->insertCount(words) += count;
}

FloatCount
Lattice::countNgrams(unsigned order, NgramCounts<FloatCount> &counts,
						    double posteriorScale)
{
    LogP2 *forwardProbs = new LogP2[maxIndex];
    LogP2 *backwardProbs = new LogP2[maxIndex];
    assert(forwardProbs != 0 && backwardProbs != 0);
    
    computeForwardBackward(forwardProbs, backwardProbs, posteriorScale);
    delete [] forwardProbs;		// not needed

    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::countNgrams: "
 	     << "computing ngram counts\n";
    }

    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    if (numReachable != numNodes) {
      if (debug(DebugPrintOutLoop)) {
	dout() << "Lattice::countNgrams: warning: called with unreachable nodes\n";
      }
    }

    Map2<NodeIndex, const NBestWordInfo *, LogP2> forwardProbMap;
    FloatCount countSum = 0.0;

    // prime forward probability map with initial node
    NBestWordInfo context[2];
    context[0].word = vocab.ssIndex();
    context[1].word = Vocab_None;
    ngramCountAccumulator(this, context, 1.0, &counts);

    if (order > 1) {
	*forwardProbMap.insert(initial, context) = LogP_One;
    } else {
	// unigram counts required empty all nodes mapped to null context
	*forwardProbMap.insert(initial, &context[1]) = LogP_One;
    }

    for (unsigned i = 0; i < numReachable; i++) {
    	countSum +=
	    countNgramsAtNode(sortedNodes[i], order,
			      backwardProbs, posteriorScale, forwardProbMap,
			      (Lattice::NgramAccumulatorFunction *)&ngramCountAccumulator,
			      &counts);
    }

    delete [] backwardProbs;
    delete [] sortedNodes;

    return countSum; 
}

