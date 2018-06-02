/*
 * Bleu.h --
 *	BLEU computation
 *
 * Copyright (c) 2007, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/Bleu.h,v 1.1 2007/10/30 05:15:32 stolcke Exp $
 *
 */

#ifndef _Bleu_H
#define	_Bleu_H

#define MAX_BLEU_NGRAM 4

double 
computeBleu(unsigned n, unsigned correct[], unsigned total[],
            unsigned length, unsigned rlength);

#endif	/* _Bleu_H */

