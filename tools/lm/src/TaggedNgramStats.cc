/*
 * TaggedNgramStats.cc --
 *	N-gram counting for word/tag pairs
 *
 */

#ifndef lint
static char TaggedNgramStats_Copyright[] = "Copyright (c) 1995-2006 SRI International.  All Rights Reserved.";
static char TaggedNgramStats_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/TaggedNgramStats.cc,v 1.6 2006/08/12 06:46:11 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>

#include "TaggedNgramStats.h"

TaggedNgramStats::TaggedNgramStats(TaggedVocab &vocab, unsigned int maxOrder)
    : NgramStats(vocab, maxOrder), vocab(vocab)
{
}

void
TaggedNgramStats::incrementTaggedCounts(const VocabIndex *words,
							NgramCount factor)
{
    VocabIndex wbuffer[maxWordsPerLine + 1];

    unsigned i;
    for (i = 0; i < order && words[i] != Vocab_None; i++) {
	wbuffer[i] = TaggedVocab::unTag(words[i]);
    }
    wbuffer[i] = Vocab_None;

    incrementCounts(wbuffer, 1, factor);

    for (i = 0; i < order && words[i] != Vocab_None; i++) {
	VocabIndex tag = TaggedVocab::getTag(words[i]);

	if (tag != Tag_None) {
	    wbuffer[i] = TaggedVocab::tagWord(Tagged_None, tag);
	    incrementCounts(wbuffer, i + 1, factor);
	}
    }
}

unsigned
TaggedNgramStats::countSentence(const VocabIndex *words, NgramCount factor)
{
    unsigned int start;

    for (start = 0; words[start] != Vocab_None; start++) {
        incrementTaggedCounts(words + start, factor);
    }

    /*
     * keep track of word and sentence counts
     */
    stats.numWords += start;
    if (words[0] == vocab.ssIndex()) {
	stats.numWords --;
    }
    if (start > 0 && words[start-1] == vocab.seIndex()) {
	stats.numWords --;
    }

    stats.numSentences ++;

    return start;
}

