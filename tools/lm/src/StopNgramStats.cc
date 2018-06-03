/*
 * StopNgramStats.cc --
 *	N-gram statistics with contexts excluding stop words
 *
 */

#ifndef lint
static char TaggedNgramStats_Copyright[] = "Copyright (c) 1996-2006 SRI International.  All Rights Reserved.";
static char TaggedNgramStats_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/StopNgramStats.cc,v 1.7 2006/08/12 06:46:11 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>

#include "StopNgramStats.h"

#include "Array.cc"

StopNgramStats::StopNgramStats(Vocab &vocab, SubVocab &stopWords,
							unsigned maxOrder)
    : NgramStats(vocab, maxOrder), stopWords(stopWords)
{
}

void
StopNgramStats::incrementCounts(const VocabIndex *words, NgramCount factor)
{
    while (*words != Vocab_None) {
	counts.insertTrie(words ++)->value() += factor;
    }
}

unsigned
StopNgramStats::countSentence(const VocabIndex *words, NgramCount factor)
{
    unsigned sentLength = Vocab::length(words);

    makeArray(VocabIndex, countWords, sentLength + 1);

    unsigned countPos = 0;
    for (unsigned nextPos = 0; nextPos < sentLength; nextPos++) {
	/*
	 * Count an ngram that has the current word as the last item,
	 * and is preceded the non-stop words found so far.
	 */
	countWords[countPos] = words[nextPos];
	countWords[countPos + 1] = Vocab_None;
	if (countPos + 1 >= order) {
	    incrementCounts(&countWords[countPos + 1 - order], factor);
	} else {
	    incrementCounts(countWords, factor);
	}

	/*
	 * Check if the next word is a non-stop one, and if so
	 * include it in the context for the following ngrams
	 */
	if (!stopWords.getWord(words[nextPos])) {
	    countWords[countPos ++] = words[nextPos];
	}
    }

    /*
     * keep track of word and sentence counts
     */
    stats.numWords += sentLength;
    if (words[0] == vocab.ssIndex()) {
	stats.numWords --;
    }
    if (sentLength > 0 && words[sentLength-1] == vocab.seIndex()) {
	stats.numWords --;
    }

    stats.numSentences ++;

    return sentLength;
}

