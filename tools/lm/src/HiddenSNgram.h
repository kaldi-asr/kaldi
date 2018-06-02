/*
 * hiddenSNgram.h --
 *	N-gram backoff language model with hidden sentence boundaries
 *
 * Copyright (c) 1996, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/speech/stolcke/project/srilm/devel/lm/src/RCS/DFNgram.h,v 
1.5 1995/11/07 08:37:12 stolcke Exp $
 *
 */

#ifndef _HiddenSNgram_h_
#define _HiddenSNgram_h_

#include "Vocab.h"
#include "Ngram.h"

const VocabString	HiddenSent = "<#s>";

class HiddenSNgram: public Ngram
{
public:
    HiddenSNgram(Vocab &vocab, unsigned order);

    /*
     * LM interface
     */
    LogP wordProb(VocabIndex word, const VocabIndex *context);

    VocabIndex hiddenSIndex;		/* <#s> index */
};

#endif /* _HiddenSNgram_h_ */
