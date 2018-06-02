/*
 * NullLM.h
 *	The Null Language Model
 *
 * This model always computes a constant log probability of zero.
 * It is thus not a real model, but it can be used as a dummy when
 * an LM should not perform any real function.
 *
 * Copyright (c) 1995,2002 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/NullLM.h,v 1.3 2002/08/25 17:27:45 stolcke Exp $
 *
 */

#ifndef _NullLM_h_
#define _NullLM_h_

#include "LM.h"

class NullLM: public LM
{
public:
    NullLM(Vocab &vocab) : LM(vocab) {};

    LogP wordProb(VocabIndex word, const VocabIndex *context)
	{ return 0.0; }

    void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
	{ length = 0; return (void *)1; };
};

#endif /* _NullLM_h_ */
