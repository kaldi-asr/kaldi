/*
 * CacheLM.cc --
 *	Unigram cache language model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/CacheLM.cc,v 1.10 2014-04-08 03:04:52 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <math.h>

#include "CacheLM.h"

#include "LHash.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(VocabIndex,double);
#endif

/*
 * Debug levels used
 */
#define DEBUG_CACHE_HITS	2

CacheLM::CacheLM(Vocab &vocab, unsigned historyLength)
    : LM(vocab), historyLength(historyLength),
      wordHistory(0, historyLength), wordCounts(0)
{
   flushCache();
}

/*
 * Forget all that is in the cache
 */
void
CacheLM::flushCache()
{
    /*
     * Initialize word history.
     */
    for (unsigned i = 0; i < historyLength; i++) {
	wordHistory[i] = Vocab_None;
    }
    historyEnd = 0;
    totalCount = 0.0;

    /*
     * Reset word counts to zero
     */
    LHashIter<VocabIndex,double> wordIter(wordCounts);
    VocabIndex word;
    double *wordCount;

    while ((wordCount = wordIter.next(word))) {
	*wordCount = 0.0;
    }
}

LogP
CacheLM::wordProb(VocabIndex word, const VocabIndex *context)
{
    /*
     * We don't cache unknown words unless <unk> is treated as a regular word.
     */
    if (word == vocab.unkIndex() && !vocab.unkIsWord()) {
	return LogP_Zero;
    }

    /*
     * Return the maximum likelihood estimate based on all words
     * in the history.  Return prob 0 for the very first word.
     */
    double *wordCount = wordCounts.insert(word);

    Prob wordProb =
	totalCount == 0.0 ? 0.0 : (*wordCount / totalCount);

    if (running() && debug(DEBUG_CACHE_HITS)) {
	dout() << "[cache=" << wordProb << "]";
    }

    /*
     * Update history and counts
     */
    if (running() && historyLength > 0) {
	VocabIndex oldWord = wordHistory[historyEnd];
	if (oldWord == Vocab_None) {
	    totalCount ++;
	} else {
	    double *oldWordCount = wordCounts.find(oldWord);
	    assert(oldWordCount != 0);

	    *oldWordCount -= 1.0;
	}

	wordHistory[historyEnd] = word;
	*wordCount += 1.0;

	historyEnd = (historyEnd + 1) % historyLength;
    }

    return ProbToLogP(wordProb);
}

