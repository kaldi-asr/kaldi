/*
 * SkipNgram.h --
 *	N-gram backoff language model with context skips
 *
 * Copyright (c) 1996-2007 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/speech/stolcke/project/srilm/devel/lm/src/RCS/DFNgram.h,v 
1.5 1995/11/07 08:37:12 stolcke Exp $
 *
 */

#ifndef _SkipNgram_h_
#define _SkipNgram_h_

#include "Ngram.h"
#include "LHash.h"

class SkipNgram: public Ngram
{
public:
    SkipNgram(Vocab &vocab, unsigned order);

    /*
     * LM interface
     */
    LogP wordProb(VocabIndex word, const VocabIndex *context);

    Boolean read(File &file, Boolean limitVocab = false);
    Boolean write(File &file);

    /*
     * Estimation
     */
    Boolean estimate(NgramStats &stats, Discount **discounts);

    unsigned maxEMiters;		/* max number of EM iterations */
    double minEMdelta;			/* min log likelihood delta */
    Prob initialSkipProb;		/* default initial skip probability */

    void memStats(MemStats &stats);

protected:
    LHash<VocabIndex, Prob> skipProbs;		/* word skipping probs */

    LogP estimateEstepNgram(VocabIndex *ngram, NgramCount ngramCount,
			    NgramStats &stats,
			    NgramCounts<FloatCount> &ngramExps,
			    LHash<VocabIndex, double> &skipExps);
    LogP estimateEstep(NgramStats &stats,
		       NgramCounts<FloatCount> &ngramExps,
		       LHash<VocabIndex, double> &skipExps);
    Boolean estimateMstep(NgramStats &stats,
		       NgramCounts<FloatCount> &ngramExps,
		       LHash<VocabIndex, double> &skipExps,
		       Discount **discounts);
};

#endif /* _SkipNgram_h_ */
