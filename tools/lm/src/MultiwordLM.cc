/*
 * MultiwordLM.cc --
 *	Multiword wrapper language model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2001-2006 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/MultiwordLM.cc,v 1.7 2007/12/05 00:31:04 stolcke Exp $";
#endif

#include <stdlib.h>

#include "MultiwordLM.h"

#include "Array.cc"

LogP
MultiwordLM::wordProb(VocabIndex word, const VocabIndex *context)
{
    /*
     * buffer holding expanded context, with room to prepend expanded
     * word
     */
    VocabIndex expandedBuffer[2 * maxWordsPerLine + 1];

    /*
     * expand the context with all multiwords
     */
    VocabIndex *expandedContext = &expandedBuffer[maxWordsPerLine];
    unsigned expandedContextLength =
	vocab.expandMultiwords(context, expandedContext, maxWordsPerLine, true);

    VocabIndex multiWord[2];
    multiWord[0] = word;
    multiWord[1] = Vocab_None;

    VocabIndex expandedWord[maxWordsPerLine + 1];
    unsigned expandedWordLength =
	    vocab.expandMultiwords(multiWord, expandedWord, maxWordsPerLine);

    LogP prob = LogP_One;
    for (unsigned j = 0; j < expandedWordLength; j ++) {
	prob += lm.wordProb(expandedWord[j],
			    &expandedBuffer[maxWordsPerLine - j]);

	expandedBuffer[maxWordsPerLine - 1 - j] = expandedWord[j];
    }

    return prob;
}

void *
MultiwordLM::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
{
    VocabIndex expandedContext[maxWordsPerLine + 1];
    VocabIndex expandedWord[maxWordsPerLine + 1];

    unsigned clen = Vocab::length(context);
    makeArray(unsigned, expansionLengths, clen);

    unsigned expandedContextLength =
	vocab.expandMultiwords(context, expandedContext, maxWordsPerLine, true,
							    expansionLengths);

    if (word == Vocab_None) {
	expandedWord[0] = Vocab_None;
    } else {
	VocabIndex multiWord[2];
	multiWord[0] = word;
	multiWord[1] = Vocab_None;

	unsigned expandedWordLength =
	    vocab.expandMultiwords(multiWord, expandedWord, maxWordsPerLine);
    }

    unsigned usedLength;
    void *cid = lm.contextID(expandedWord[0], expandedContext, usedLength);

    /*
     * translate the context-used length for the non-mw LM back to multiwords
     */
    unsigned usedMWLength = 0;
    unsigned sumOfExpandedLengths = 0;
    while (sumOfExpandedLengths < usedLength && usedMWLength < clen) {
	sumOfExpandedLengths += expansionLengths[usedMWLength++];
    }

    length = usedMWLength;
    return cid;
}

LogP
MultiwordLM::contextBOW(const VocabIndex *context, unsigned length)
{
    VocabIndex expandedContext[maxWordsPerLine + 1];

    unsigned clen = Vocab::length(context);
    makeArray(unsigned, expansionLengths, clen);

    vocab.expandMultiwords(context, expandedContext, maxWordsPerLine, true,
							    expansionLengths);
    /* 
     * Compute the length value in terms of expanded words
     */
    unsigned usedLength = 0;
    for (unsigned i = 0; i < length && i < clen; i ++) {
	usedLength += expansionLengths[i];
    }

    return lm.contextBOW(expandedContext, usedLength);
}

Boolean
MultiwordLM::isNonWord(VocabIndex word)
{
    /*
     * Map candidate word to underlying LM vocab, and check if it is 
     * a non-word there.
     */
    VocabIndex oneWord[2];
    oneWord[0] = word;
    oneWord[1] = Vocab_None;

    VocabIndex expanded[2];
    unsigned expandedLength = vocab.expandMultiwords(oneWord, expanded, 2);

    return (expandedLength == 1) && lm.isNonWord(expanded[0]);
}

void
MultiwordLM::setState(const char *state)
{
    /*
     * Global state changes are propagated to the underlying models
     */
    lm.setState(state);
}

