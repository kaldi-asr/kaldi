/*
 * Bleu.cc --
 *	BLEU score computation for MT evaluation
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2007 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Bleu.cc,v 1.1 2007/10/30 05:15:32 stolcke Exp $";
#endif

#include <math.h>

#include "Bleu.h"

double 
computeBleu(unsigned n, unsigned correct[], unsigned total[],
            unsigned length, unsigned rlength)
{
    double brevityPenalty = 1.0;
    double logPrecision = 0.0;

    if (length < rlength) {
	brevityPenalty = exp(1.0 - (double)rlength / length);
    }
    
    for (unsigned i = 0; i < n; i++) {
	if (correct[i] == 0) {
          if (total[i] == 0) 
            break;
	  logPrecision += -9999;
	} else {
	  logPrecision += log((double) correct[i] / total[i]);    
	}
    }

    return brevityPenalty * exp(logPrecision / n);
}

