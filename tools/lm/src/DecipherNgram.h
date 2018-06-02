/*
 * DecipherNgram.h --
 *	Approximate N-gram backoff language model used in Decipher recognizer
 *
 * Copyright (c) 1995, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/DecipherNgram.h,v 1.4 2000/08/07 07:42:15 stolcke Exp $
 *
 */

#ifndef _DecipherNgram_h_
#define _DecipherNgram_h_

#include <stdio.h>

#include "Ngram.h"

class DecipherNgram: public Ngram
{
public:
    DecipherNgram(Vocab &vocab, unsigned int order = 2,
					    Boolean backoffHack = true);

protected:
    virtual LogP wordProbBO(VocabIndex word, const VocabIndex *context,
							unsigned int clen);

    Boolean backoffHack;	/* take backoff path if higher scoring */
};

#endif /* _DecipherNgram_h_ */
