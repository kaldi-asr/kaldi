/*
 * LatticeIndex.cc --
 *	Indexing of N-grams in lattice
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2006-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeIndex.cc,v 1.11 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "Lattice.h"

#include "Map2.h"
#include "LHash.cc"
#include "Array.cc"

#define DebugPrintFatalMessages         0 

#define DebugPrintFunctionality         1 
// for large functionality listed in options of the program
#define DebugPrintOutLoop               2
// for out loop of the large functions or small functions
#define DebugPrintInnerLoop             3
// for inner loop of the large functions or outloop of small functions


typedef struct {
    NBestTimestamp maxPause;
    NBestTimestamp timeTolerance;
    LHash<const NBestWordInfo *, Prob> counts;
} IndexParams;

static NBestTimestamp
maxInternalPause(const NBestWordInfo *ngram)
{
    if (ngram[0].word == Vocab_None) {
    	return 0.0;
    }

    NBestTimestamp maxPause = 0.0;

    for (unsigned i = 1; ngram[i].word != Vocab_None; i ++) {
    	if (ngram[i-1].valid() && ngram[i].valid()) {
	    NBestTimestamp pause = ngram[i].start -
				    (ngram[i-1].start + ngram[i-1].duration);
	    if (pause > maxPause) {
		maxPause = pause;
	    }
	}
    }

    return maxPause;
}

//
// Callback function for Lattice::countNgramsAtNode():
//	sums ngram counts
//
static void
ngramIndexAccumulator(Lattice *lat, const NBestWordInfo *ngram, Prob count,
							IndexParams *params)
{
    // discard ngram if internal pause is too long
    if (params->maxPause == 0.0 || maxInternalPause(ngram) <= params->maxPause)
    {
    	if (params->timeTolerance == 0.0) {
	    *params->counts.insert(ngram) += count;
	} else {
	    // round timestamps according to tolerance specified
	    unsigned len = NBestWordInfo::length(ngram);

	    makeArray(NBestWordInfo, roundedNgram, len + 1);

	    NBestWordInfo::copy(roundedNgram, ngram);

	    // round start time
	    if (ngram[0].valid()) {
	    	roundedNgram[0].start =
			    rint(ngram[0].start / params->timeTolerance) *
							params->timeTolerance;
	    }

	    // round end time
	    if (ngram[len-1].valid()) {
	    	roundedNgram[len-1].start =
			    rint(ngram[len-1].start / params->timeTolerance) *
							params->timeTolerance;
	    	
	    	roundedNgram[len-1].duration =
			rint((ngram[len-1].start + ngram[len-1].duration)
						    / params->timeTolerance) *
							params->timeTolerance
			    - roundedNgram[len-1].start;
	    }

	    *params->counts.insert(roundedNgram) += count;
	}
    }
}

//
// Output ngram index to file
//
static void
writeIndex(File &file, Lattice &lat, Prob mincount, NBestTimestamp maxPause,
				    LHash<const NBestWordInfo *, Prob> &counts, Vocab *keywords = NULL)
{
    LHashIter<const NBestWordInfo *, Prob>
    					iter(counts, SArray_compareKey);

    Prob *count;
    const NBestWordInfo *ngram;
    while ((count = iter.next(ngram))) {

        unsigned len = NBestWordInfo::length(ngram);
        // ignore if below threshold
        // if we supply a keyword vocabulary, only print out keywords
        if ((mincount == 0.0 || *count > mincount) && (keywords == NULL || keywords->getIndex(lat.vocab.getWord(ngram[0].word)) != Vocab_None)) {
            if (keywords == NULL) {
                // Regular n-gram Index format
                file.fprintf("%s", lat.getName());
                file.fprintf(ngram[0].valid() ? " %.2f" : " ?", (float)ngram[0].start);
                file.fprintf(ngram[len-1].valid() ? " %.2f" : " ?", (float)(ngram[len-1].start + ngram[len-1].duration));
                file.fprintf(" %.*lg", FloatCount_Precision, (double)*count);
                for (unsigned i = 0; i < len; i ++) {
                    file.fprintf(" %s", lat.vocab.getWord(ngram[i].word));
                }
            } else {
                // KWS results file format
	        // fprintf(stderr,"Keyword Found: %s\n", lat.vocab.getWord(ngram[0].word));
                for (unsigned i = 0; i < len; i ++) {
                    file.fprintf("%s ", lat.vocab.getWord(ngram[i].word));
                }
                file.fprintf("%s 1 ", lat.getName());
                file.fprintf(ngram[0].valid() ? "%.2f " : "? ", (float)ngram[0].start);
                file.fprintf(ngram[len-1].valid() ? "%.2f " : "? ", (float)(ngram[len-1].start + ngram[len-1].duration));
                file.fprintf("%.*lg ", FloatCount_Precision, (double)*count);
            }
            file.fprintf("\n");
        }
    }
}

FloatCount
Lattice::indexNgrams(unsigned order, File &file, Prob minCount,
		     NBestTimestamp maxPause, NBestTimestamp timeTolerance,
		     double posteriorScale, Vocab *keywords)
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
	dout() << "Lattice::indexNgrams: warning: called with unreachable nodes\n";
      }
    }

    Map2<NodeIndex, const NBestWordInfo *, LogP2> forwardProbMap;
    Prob countSum = 0.0;

    // prime forward probability map with initial node
    NBestWordInfo context[2];
    context[0].word = vocab.ssIndex();
    context[0].start = context[0].duration = 0.0;
    context[1].word = Vocab_None;

    if (order > 1) {
	*forwardProbMap.insert(initial, context) = LogP_One;
    } else {
	// unigram counts required empty all nodes mapped to null context
	*forwardProbMap.insert(initial, &context[1]) = LogP_One;
    }

    // set up data to pass to accumulator function
    IndexParams params;
    params.maxPause = maxPause;
    params.timeTolerance = timeTolerance;

    for (unsigned i = 0; i < numReachable; i++) {
    	countSum +=
	    countNgramsAtNode(sortedNodes[i], order,
			      backwardProbs, posteriorScale, forwardProbMap,
			      (Lattice::NgramAccumulatorFunction *)&ngramIndexAccumulator,
			      &params, true);
    }

    // write accumulated counts to file
    writeIndex(file, *this, minCount, maxPause, params.counts, keywords);

    delete [] backwardProbs;
    delete [] sortedNodes;

    return countSum; 
}

