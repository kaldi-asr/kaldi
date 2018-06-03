/*
 * HiddenNgram.h --
 *	N-gram model with hidden between-word events
 *
 * Copyright (c) 1999-2007 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/HiddenNgram.h,v 1.11 2012-03-06 20:23:56 stolcke Exp $
 *
 */

#ifndef _HiddenNgram_h_
#define _HiddenNgram_h_

#include <stdio.h>

#include "Ngram.h"
#include "Trellis.h"
#include "SubVocab.h"
#include "Array.h"

/* 
 * The DP trellis to evaluate a hidden ngram contains the N-gram context,
 * as well as a pointer indicating what words, if any, are to be repeated.
 * repeatFrom > 0 means that the word at context[repeatFrom-1] is to be 
 * repeated in the current input.  This way we handle disfluenct repeatitions.
 */
typedef const VocabIndex *VocabContext;
typedef struct {
	VocabContext context;
	unsigned repeatFrom;
	VocabIndex event;		/* for Viterbi decoding */
} HiddenNgramState;

ostream &operator<< (ostream &, const HiddenNgramState &state);

/*
 * Properties of hidden event vocabulary items
 */
typedef struct {
    unsigned deleteWords:8;		// words to delete from context
    unsigned repeatWords:8;		// words to repeat
    Boolean isObserved:1;		// event is overt
    Boolean omitFromContext:1;		// do not put event tag in context
    VocabIndex insertWord;		// tag to insert (typically <s>)
} HiddenVocabProps;

/*
 * A N-gram language model that sums over hidden events between the
 * observable words, in the style of Stolcke et al., Automatic Detection of
 * Sentence Boundaries and Disfluencies based on Recognized Words. Proc.
 * Intl. Conf. on Spoken Language Processing, vol. 5, pp.  2247-2250, Sydney,
 * 1998.
 */
class HiddenNgram: public Ngram
{
public:
    HiddenNgram(Vocab &vocab, SubVocab &hiddenVocab, unsigned order,
						    Boolean nothidden = false);
    ~HiddenNgram();

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

    Boolean read(File &file, Boolean limitVocab = false);
    Boolean write(File &file);

    Boolean readHiddenVocab(File &file);
    Boolean writeHiddenVocab(File &file);

    const HiddenVocabProps &getProps(VocabIndex word);

protected:
    Trellis<HiddenNgramState> trellis;	/* for DP on hidden events */
    const VocabIndex *prevContext;	/* context from last DP */
    unsigned prevPos;			/* position from last DP */
    LogP prefixProb(VocabIndex word, const VocabIndex *context,
				LogP &contextProb, TextStats &stats);
					/* prefix probability */
    Array<VocabIndex> savedContext;	/* saved, rev'd copy of last context */
    unsigned savedLength;		/* length of saved context above */

    SubVocab &hiddenVocab;		/* the hidden event vocabulary */
    VocabIndex noEventIndex;		/* the "no-event" event */
    Boolean notHidden;			/* overt event mode */
    LHash<VocabIndex, HiddenVocabProps> vocabProps;
					/* properties of vocabulary items */
};

#endif /* _HiddenNgram_h_ */
