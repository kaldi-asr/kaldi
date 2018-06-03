/*
 * WordAlign.h --
 *	Word alignment and error computation
 *
 * Copyright (c) 1996,1997 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/WordAlign.h,v 1.5 2012/10/29 17:25:05 mcintyre Exp $
 *
 */

#ifndef _WORD_ALIGN_H_
#define _WORD_ALIGN_H_

#include "Vocab.h"

/*
 * Error types
 */
typedef enum {
	CORR_ALIGN, SUB_ALIGN, DEL_ALIGN, INS_ALIGN, END_ALIGN
} WordAlignType;

/*
 * Costs for individual error types.  These are the conventional values
 * used in speech recognition word alignment.
 */
const unsigned SUB_COST = 4;
const unsigned DEL_COST = 3;
const unsigned INS_COST = 3;

unsigned wordError(const VocabIndex *hyp, const VocabIndex *ref,
			unsigned &sub, unsigned &ins, unsigned &del,
			WordAlignType *alignment = 0);
					/* computes total word error */
void wordError_freeThread();
#endif /* _WORD_ALIGN_H_ */

