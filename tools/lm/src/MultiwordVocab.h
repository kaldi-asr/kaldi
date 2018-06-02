/*
 * MultiwordVocab.h --
 *	Vocabulary containing multiwords
 *
 * A multiword is a vocabulary element consisting of words joined by
 * underscores (or some other delimiter), e.g., "i_don't_know".
 * This class provides support for splitting such words into their components.
 *
 * Copyright (c) 2001,2004 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/MultiwordVocab.h,v 1.3 2004/10/25 02:43:20 stolcke Exp $
 *
 */

#ifndef _MultiwordVocab_h_
#define _MultiwordVocab_h_

#include "Vocab.h"

extern const char *MultiwordSeparator;

class MultiwordVocab: public Vocab
{
public:
    MultiwordVocab(VocabIndex start, VocabIndex end,
				    const char *multiChar = MultiwordSeparator)
	: Vocab(start, end), multiChar(multiChar) {};
    MultiwordVocab(const char *multiChar = MultiwordSeparator)
	: multiChar(multiChar) {};
    ~MultiwordVocab();

    /*
     * Modified Vocab methods
     */
    virtual VocabIndex addWord(VocabString name);
    virtual void remove(VocabString name);
    virtual void remove(VocabIndex index);

    /*
     * Expansion of vocab strings into their components
     */
    unsigned expandMultiwords(const VocabIndex *words, VocabIndex *expanded,
				unsigned maxExpanded, Boolean reverse = false,
				unsigned *lengths = 0);

private:
    LHash< VocabIndex, VocabIndex * > multiwordMap;

    const char *multiChar;
};

#endif /* _MultiwordVocab_h_ */
