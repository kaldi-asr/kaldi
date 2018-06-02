/*
 * VarNgram.h --
 *	Variable N-gram backoff language model
 *
 * Copyright (c) 1995, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/VarNgram.h,v 1.1 1995/08/03 03:31:29 stolcke Exp $
 *
 */

#ifndef _VarNgram_h_
#define _VarNgram_h_

#include <stdio.h>

#include "Ngram.h"

class VarNgram: public Ngram
{
public:
    VarNgram(Vocab &vocab, unsigned int order, double alpha = 0.0);

    /*
     * Estimation
     */
    Boolean estimate(NgramStats &stats, Discount **discounts);
    Boolean pruneNgram(NgramStats &stats,
			VocabIndex word, NgramCount ngramCount,
			const VocabIndex *context, NgramCount contextCount);

    double pruneAlpha;		/* pruning threshold */
};

#endif /* _VarNgram_h_ */
