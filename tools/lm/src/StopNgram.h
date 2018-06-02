/*
 * StopNgram.h --
 *	N-gram LM with stop words removed from context
 *
 * Copyright (c) 1996,2002 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/StopNgram.h,v 1.3 2002/08/25 17:27:45 stolcke Exp $
 *
 */

#ifndef _StopNgram_h_
#define _StopNgram_h_

#include "Ngram.h"
#include "SubVocab.h"

class StopNgram: public Ngram
{
public:
    StopNgram(Vocab &vocab, SubVocab &stopWords, unsigned int order);

    /*
     * LM interface
     */
    LogP wordProb(VocabIndex word, const VocabIndex *context);
    void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    LogP contextBOW(const VocabIndex *context, unsigned length);

    SubVocab &stopWords;			/* stop word set */

protected:
    unsigned removeStopWords(const VocabIndex *context,
			    VocabIndex *usedContext, unsigned usedLength);
};

#endif /* _StopNgram_h_ */
