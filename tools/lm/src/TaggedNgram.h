/*
 * TaggedNgram.h --
 *	Tagged N-gram backoff language models
 *
 * Copyright (c) 1995, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/TaggedNgram.h,v 1.1 1995/08/23 03:19:35 stolcke Exp $
 *
 */

#ifndef _TaggedNgram_h_
#define _TaggedNgram_h_

#include "Ngram.h"
#include "TaggedVocab.h"

class TaggedNgram: public Ngram
{
public:
    TaggedNgram(TaggedVocab &vocab, unsigned int order);

    TaggedVocab &vocab;			/* vocabulary */

protected:
    virtual LogP wordProbBO(VocabIndex word, const VocabIndex *context,
							unsigned int clen);
    virtual void recomputeBOWs();
};

#endif /* _TaggedNgram_h_ */
