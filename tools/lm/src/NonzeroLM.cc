/*
 * NonzeroLM.cc --
 *	Wrapper language model to ensure nonzere probabilities
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2011 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NonzeroLM.cc,v 1.1 2011/01/14 01:19:00 stolcke Exp $";
#endif

#include <stdlib.h>

#include "NonzeroLM.h"

NonzeroLM::NonzeroLM(Vocab &vocab, LM &lm, VocabString zerowordString)
    : LM(vocab), lm(lm)
{
    zeroword = vocab.addWord(zerowordString);
}

LogP
NonzeroLM::wordProb(VocabIndex word, const VocabIndex *context)
{
    LogP prob = lm.wordProb(word, context);

    if (prob == LogP_Zero && word != zeroword) {
	prob = lm.wordProb(zeroword, context);
    }

    return prob;
}

void *
NonzeroLM::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
{
    if (word == Vocab_None) {
	return lm.contextID(word, context, length);
    } else {
	LogP prob = lm.wordProb(word, context);

	if (prob == LogP_Zero && word != zeroword) {
	    return lm.contextID(zeroword, context, length);
	} else {
	    return lm.contextID(word, context, length);
	}
    }
}

LogP
NonzeroLM::contextBOW(const VocabIndex *context, unsigned length)
{
    return lm.contextBOW(context, length);
}

Boolean
NonzeroLM::isNonWord(VocabIndex word)
{
    return lm.isNonWord(word);
}

void
NonzeroLM::setState(const char *state)
{
    /*
     * Global state changes are propagated to the underlying model
     */
    lm.setState(state);
}

Boolean
NonzeroLM::addUnkWords()
{
    return lm.addUnkWords();
}

