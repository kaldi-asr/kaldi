/*
 * CacheLM.h
 *	Unigram cache language model.
 *
 * The model keeps track of the last N words (across sentences)
 * and computes a maximum likelihood unigram distribution from them.
 * (This is typically used in interpolating with other LMs.)
 *
 * Copyright (c) 1995, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/CacheLM.h,v 1.1 1995/08/23 21:57:27 stolcke Exp $
 *
 */

#ifndef _CacheLM_h_
#define _CacheLM_h_

#include "LM.h"
#include "LHash.h"
#include "Array.h"

class CacheLM: public LM
{
public:
    CacheLM(Vocab &vocab, unsigned historyLength);

    unsigned historyLength;	/* context length used for cacheLength */

    /*
     * LM interface
     */
    virtual LogP wordProb(VocabIndex word, const VocabIndex *context);

protected:
    double totalCount;			/* total number of words in history */
    unsigned historyEnd;		/* index into wordHistory */
    Array<VocabIndex> wordHistory;	/* the last historyLength words */
    LHash<VocabIndex,double> wordCounts;/* (fractional) word counts */

    void flushCache();			/* forget history */
};


#endif /* _CacheLM_h_ */
