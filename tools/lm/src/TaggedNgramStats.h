/*
 * TaggedNgramStats.h --
 *	N-gram statistics on word/tag pairs
 *
 * Copyright (c) 1995,2002 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/TaggedNgramStats.h,v 1.3 2003/07/01 07:41:48 stolcke Exp $
 *
 */

#ifndef _TaggedNgramStats_h_
#define _TaggedNgramStats_h_

#include "NgramStats.h"
#include "TaggedVocab.h"

class TaggedNgramStats: public NgramStats
{
public:
    TaggedNgramStats(TaggedVocab &vocab, unsigned int maxOrder);

    virtual unsigned countSentence(const VocabIndex *words)
	    { return countSentence(words, (NgramCount)1); };
    virtual unsigned countSentence(const VocabIndex *words, NgramCount factor);

    TaggedVocab &vocab;			/* vocabulary */

protected:
    void incrementTaggedCounts(const VocabIndex *words, NgramCount factor);
};

#endif /* _TaggedNgramStats_h_ */

