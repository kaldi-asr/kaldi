/*
 * DFNgram.h --
 *	N-gram backoff language model for disfluencies
 *
 * Copyright (c) 1995-2002 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/DFNgram.h,v 1.11 2002/08/25 17:27:45 stolcke Exp $
 *
 */

#ifndef _DFNgram_h_
#define _DFNgram_h_

#include <stdio.h>

#include "Ngram.h"
#include "Trellis.h"
#include "Array.h"

/*
 * The disfluencies are modeled as hidden (unobservable) events
 * in the word stream, with probabilities encoded directly in the
 * ngram model, identified by special tokens.
 * Only the filled pause events (UH and UM) correspond to actual
 * observable words.
 */
const VocabString UHstring = "UH";
const VocabString UMstring = "UM";
const VocabString SDELstring = "@SDEL";
const VocabString DEL1string = "@DEL1";
const VocabString DEL2string = "@DEL2";
const VocabString REP1string = "@REP1";
const VocabString REP2string = "@REP2";

/*
 * Set of DF states used in the forward DP algorithm
 */
typedef enum {
    NODF, FP, SDEL, DEL1, DEL2, REP1, REP2, REP21, DFMAX
} DFstate;

const VocabString DFnames[DFMAX+1] = {
    "NODF", "FP", "SDEL", "DEL1", "DEL2", "REP1", "REP2", "REP21", 0
};

class DFNgram: public Ngram
{
public:
    DFNgram(Vocab &vocab, unsigned int order);
    ~DFNgram();

    /*
     * LM interface
     */
    LogP wordProb(VocabIndex word, const VocabIndex *context);
    LogP wordProbRecompute(VocabIndex word, const VocabIndex *context);
    void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    LogP contextBOW(const VocabIndex *context, unsigned length);
    Boolean isNonWord(VocabIndex word);
    LogP sentenceProb(const VocabIndex *sentence, TextStats &stats);

    VocabIndex UHindex;			/* filled pause event (UH) */
    VocabIndex UMindex;			/* filled pause event (UM) */
    VocabIndex SDELindex;		/* sentence deletion event */
    VocabIndex DEL1index;		/* 1-word deletion event */
    VocabIndex DEL2index;		/* 2-word deletion event */
    VocabIndex REP1index;		/* 1-word repetition event */
    VocabIndex REP2index;		/* 2-word repetition event */

protected:
    Trellis<DFstate> trellis;		/* for DP on hidden events */
    const VocabIndex *prevContext;	/* context from last DP */
    unsigned prevPos;			/* position from last DP */
    LogP prefixProb(VocabIndex word, const VocabIndex *context,
			LogP &contextProb); /* prefix probability */
    Array<VocabIndex> savedContext;	/* saved, rev'd copy of last context */
    unsigned savedLength;		/* length of saved context above */
};

#endif /* _DFNgram_h_ */
