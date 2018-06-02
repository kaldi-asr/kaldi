/*
 * DecipherNgram.cc --
 *	Approximate N-gram backoff language model used in Decipher recognizer
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2006 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/DecipherNgram.cc,v 1.11 2007/12/04 18:05:43 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>

#include "DecipherNgram.h"

#define DEBUG_NGRAM_HITS 2	/* from Ngram.cc */

DecipherNgram::DecipherNgram(Vocab &vocab, unsigned order, Boolean backoffHack)
      : Ngram(vocab, order), backoffHack(backoffHack)
{
}

/*
 * Simulate the rounding going on from the original LM LogP scores to the
 * bytelogs in the recognizer:
 * - PFSGs encode LogP as intlogs
 * - Nodearray compiler maps intlogs to bytelogs
 */
#define RoundToBytelog(x)	BytelogToLogP(IntlogToBytelog(LogPtoIntlog(x)))

/*
 * The backoff algorithm is similar to Ngram:wordProbBO(),
 * with the following modifications:
 * - bows and probs are truncated to ByteLog precision
 * - the backed-off probability is returned if it is larger than the
 *   direct ngram probability.
 */
LogP
DecipherNgram::wordProbBO(VocabIndex word, const VocabIndex *context, unsigned int clen)
{
    LogP result;

    TruncatedContext usedContext(context, clen);

    LogP *prob = findProb(word, usedContext);
    LogP boProb = LogP_Zero;

    /*
     * Compute the backed-off probability always
     */
    if (clen > 0) {
	LogP *bow = findBOW(usedContext);

	if (bow) {
	   boProb = RoundToBytelog(*bow) +
			wordProbBO(word, usedContext, clen - 1);
	} else {
	   boProb = wordProbBO(word, usedContext, clen - 1);
	}
    }
    
    if (prob) {
	result = RoundToBytelog(*prob);

	if (running() && debug(DEBUG_NGRAM_HITS)) {
	    dout() << "[" << (clen + 1) << "gram]";
	}

	if (backoffHack && boProb > result) {
	    result = boProb;

	    if (running() && debug(DEBUG_NGRAM_HITS)) {
		dout() << "[bo]";
	    }
	}
    } else {
	result = boProb;
    }

    return result;
}

