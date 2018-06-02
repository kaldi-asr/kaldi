/*
 * StopNgram.cc --
 *	N-gram LM with stop words removed from context
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1996, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/StopNgram.cc,v 1.3 2002/08/25 17:27:45 stolcke Exp $";
#endif

#include "StopNgram.h"

/*
 * Debug levels used
 */
#define DEBUG_NGRAM_HITS 2		/* from Ngram.cc */

StopNgram::StopNgram(Vocab &vocab, SubVocab &stopWords, unsigned neworder)
    : Ngram(vocab, neworder), stopWords(stopWords)
{
}

/*
 * Remove stop-words from a string, and return the number of words removed
 */
unsigned
StopNgram::removeStopWords(const VocabIndex *context,
			VocabIndex *usedContext, unsigned usedLength)
{
    unsigned i, j = 0;
    for (i = 0; i < usedLength - 1 && context[i] != Vocab_None ; i++) {
	if (!stopWords.getWord(context[i])) {
	    usedContext[j ++] = context[i];
	}
    }
    usedContext[j] = Vocab_None;
    return i - j;
}

/*
 * The only difference to a standard Ngram model is that stop words are
 * removed from the context before conditional probabilities are computed.
 */
LogP
StopNgram::wordProb(VocabIndex word, const VocabIndex *context)
{
    VocabIndex usedContext[maxNgramOrder + 1];
    removeStopWords(context, usedContext, sizeof(usedContext));

    return Ngram::wordProb(word, usedContext);
}

void *
StopNgram::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
{
    VocabIndex usedContext[maxNgramOrder + 1];
    unsigned deleted =
		removeStopWords(context, usedContext, sizeof(usedContext));

    void *result = Ngram::contextID(word, usedContext, length);

    /*
     * To be safe, add the number of deleted stop words to the used context
     * length (this may be an overestimate).
     */
    length += deleted;
    return result;
}

LogP
StopNgram::contextBOW(const VocabIndex *context, unsigned length)
{
    VocabIndex usedContext[maxNgramOrder + 1];
    removeStopWords(context, usedContext, sizeof(usedContext));

    return Ngram::contextBOW(usedContext, length);
}

