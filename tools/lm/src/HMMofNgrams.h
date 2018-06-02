/*
 * HMMofNgrams.h --
 *	Hidden Markov Model of Ngram distributions
 * 
 * The model consists of an HMM whose output distributions are Ngram
 * LMs.  Each state generates a word string from its Ngram model,
 * then transitions to another state according to the HMM state transition
 * matrix.
 *
 * Copyright (c) 1997,2003 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/HMMofNgrams.h,v 1.8 2014-04-22 06:57:46 stolcke Exp $
 *
 */

#ifndef _HMMofNgrams_h_
#define _HMMofNgrams_h_

#include "LM.h"
#include "Ngram.h"
#include "LHash.h"
#include "Trellis.h"
#include "Array.h"

typedef VocabIndex HMMIndex;

class HMMState
{
    friend class HMMofNgrams;

public:
    HMMState() : ngram(0), ngramName(0) {};
    ~HMMState() { delete ngram; if (ngramName) free(ngramName); };

private:
    Ngram *ngram;
    char *ngramName;
    LHash<HMMIndex, Prob> transitions;
};

class HMMofNgrams: public LM
{
public:
    HMMofNgrams(Vocab &vocab, unsigned order);
    ~HMMofNgrams();

    /*
     * LM interface
     */
    LogP wordProb(VocabIndex word, const VocabIndex *context);
    LogP wordProbRecompute(VocabIndex word, const VocabIndex *context);
    LogP sentenceProb(const VocabIndex *sentence, TextStats &stats);

    Boolean read(File &file, Boolean limitVocab = false);
    Boolean write(File &file);

    void setState(const char *state);	/* re-read HMM from file */

    /*
     * Propagate changes to Debug state to component models
     */
    void debugme(unsigned level);
    ostream &dout() const { return Debug::dout(); };
    ostream &dout(ostream &stream);

protected:
    unsigned order;
    Vocab stateVocab;
    HMMIndex initialState, finalState;
    LHash<HMMIndex,HMMState> states;

    Trellis<HMMIndex> trellis;		/* for DP over HMM states */
    const VocabIndex *prevContext;	/* context from last DP */
    unsigned prevPos;			/* position from last DP */
    LogP prefixProb(VocabIndex word, const VocabIndex *context,
		LogP &contextProb, TextStats &stats); /* prefix probability */
    Array<VocabIndex> savedContext;	/* saved, rev'd copy of last context */
    unsigned savedLength;		/* length of saved context above */
};

#endif /* _HMMofNgrams_h_ */
