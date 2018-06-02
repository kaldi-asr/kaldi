/*
 * SimpleClassNgram.h --
 *	N-gram model over word classes that are unambiguous
 *
 * Copyright (c) 2002 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/SimpleClassNgram.h,v 1.2 2002/08/25 17:27:45 stolcke Exp $
 *
 */

#ifndef _SimpleClassNgram_h_
#define _SimpleClassNgram_h_

#include <stdio.h>

#include "ClassNgram.h"

/*
 * This is a specialization of ClassNgram where each word is member of
 * at most one class, and the classes expand to single words only.
 */
class SimpleClassNgram: public ClassNgram
{
public:
    SimpleClassNgram(Vocab &vocab, SubVocab &classVocab, unsigned order)
	: ClassNgram(vocab, classVocab, order), haveClassDefError(false) { };

    /*
     * LM interface
     */
    LogP wordProb(VocabIndex word, const VocabIndex *context);
    LogP wordProbRecompute(VocabIndex word, const VocabIndex *context);
    void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    LogP contextBOW(const VocabIndex *context, unsigned length);
    LogP sentenceProb(const VocabIndex *sentence, TextStats &stats);

    /*
     * I/O of class definitions
     */
    Boolean readClasses(File &file);

protected:
    LogP replaceWithClass(VocabIndex word, VocabIndex &clasz);
    LogP replaceWithClass(const VocabIndex *words, VocabIndex *classes,
					unsigned maxWords = maxWordsPerLine);
					/* replace words with classes */

    Boolean haveClassDefError;
};

#endif /* _SimpleClassNgram_h_ */
