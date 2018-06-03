/*
 * StopNgramStats.h --
 *	N-gram statistics with contexts excluding stop words
 *
 * Copyright (c) 1996,2002 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/StopNgramStats.h,v 1.4 2003/11/14 18:20:27 stolcke Exp $
 *
 */

#ifndef _StopNgramStats_h_
#define _StopNgramStats_h_

#include "NgramStats.h"
#include "SubVocab.h"

class StopNgramStats: public NgramStats
{
public:
    StopNgramStats(Vocab &vocab, SubVocab &stopWords, unsigned maxOrder);

    virtual unsigned countSentence(const VocabIndex *words)
	{ return countSentence(words, (NgramCount)1); };
    virtual unsigned countSentence(const VocabIndex *words, NgramCount factor);

    SubVocab &stopWords;		/* stop word set */

protected:
    void incrementCounts(const VocabIndex *words, NgramCount factor);
};

#endif /* _StopNgramStats_h_ */

