/*
 * WordAlign.cc --
 *	Word alignment and error computation
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2012 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/WordAlign.cc,v 1.13 2012/10/29 17:25:05 mcintyre Exp $";
#endif

#include <assert.h>

#include "TLSWrapper.h"
#include "WordAlign.h"

typedef struct {
    unsigned cost;			// minimal cost of partial alignment
    WordAlignType error;		// best predecessor
} ChartEntry;

/*
 * wordError --
 *	Compute word error counts between hyp and a ref word string
 *
 * Result:
 *	Total word error count
 *
 * Side effect:
 *	sub, ins, del are set to the number of substitution, insertion and
 *	deletion errors, respectively.
 *	If alignment != 0 the array is filled in with the error/alignment
 *	type of each hyp word (CORR_ALIGN, SUB_ALIGN, INS_ALIGN).  Deleted
 *	words are indicated by DEL_ALIGN.  The alignment is terminated by
 *	END_ALIGN.
 */

static TLSW(unsigned, maxHypLengthTLS);
static TLSW(unsigned, maxRefLengthTLS);
static TLSW(ChartEntry**, chartTLS);

void
freeChart(ChartEntry** chart)
{
    unsigned &maxRefLength = TLSW_GET(maxRefLengthTLS);

    for (unsigned i = 0; i <= maxRefLength; i ++) {
        delete [] chart[i];
    }
    delete [] chart;
}

unsigned
wordError(const VocabIndex *ref, const VocabIndex *hyp,
			unsigned &sub, unsigned &ins, unsigned &del,
			WordAlignType *alignment)
{
    unsigned hypLength = Vocab::length(hyp);
    unsigned refLength = Vocab::length(ref);

    /* 
     * Allocate chart statically, enlarging on demand
     */
    unsigned &maxHypLength = TLSW_GET(maxHypLengthTLS);
    unsigned &maxRefLength = TLSW_GET(maxRefLengthTLS);
    ChartEntry** &chart    = TLSW_GET(chartTLS);

    if (chart == 0 || hypLength > maxHypLength || refLength > maxRefLength) {
	/*
	 * Free old chart
	 */
	if (chart != 0) {
            freeChart(chart); 
	}

	/*
	 * Allocate new chart
	 */
	maxHypLength = hypLength;
	maxRefLength = refLength;
    
	chart = new ChartEntry*[maxRefLength + 1];
	assert(chart != 0);

	unsigned i, j;

	for (i = 0; i <= maxRefLength; i ++) {
	    chart[i] = new ChartEntry[maxHypLength + 1];
	    assert(chart[i] != 0);
	}

	/*
	 * Initialize the 0'th row and column, which never change
	 */
	chart[0][0].cost = 0;
	chart[0][0].error = CORR_ALIGN;

	/*
	 * Initialize the top-most row in the alignment chart
	 * (all words inserted).
	 */
	for (j = 1; j <= maxHypLength; j ++) {
	    chart[0][j].cost = chart[0][j-1].cost + INS_COST;
	    chart[0][j].error = INS_ALIGN;
	}

	for (i = 1; i <= maxRefLength; i ++) {
	    chart[i][0].cost = chart[i-1][0].cost + DEL_COST;
	    chart[i][0].error = DEL_ALIGN;
	}
    }

    /*
     * Fill in the rest of the chart, row by row.
     */
    for (unsigned i = 1; i <= refLength; i ++) {

	for (unsigned j = 1; j <= hypLength; j ++) {
	    unsigned minCost;
	    WordAlignType minError;

	    if (hyp[j-1] == ref[i-1]) {
		minCost = chart[i-1][j-1].cost;
		minError = CORR_ALIGN;
	    } else {
		minCost = chart[i-1][j-1].cost + SUB_COST;
		minError = SUB_ALIGN;
	    }

	    unsigned delCost = chart[i-1][j].cost + DEL_COST;
	    if (delCost < minCost) {
		minCost = delCost;
		minError = DEL_ALIGN;
	    }

	    unsigned insCost = chart[i][j-1].cost + INS_COST;
	    if (insCost < minCost) {
		minCost = insCost;
		minError = INS_ALIGN;
	    }

	    chart[i][j].cost = minCost;
	    chart[i][j].error = minError;
	}
    }

    /*
     * Backtrace
     */
    unsigned totalErrors;

    {
	unsigned i = refLength;
	unsigned j = hypLength;
	unsigned k = 0;

	sub = del = ins = 0;

	while (i > 0 || j > 0) {

	    switch (chart[i][j].error) {
	    case CORR_ALIGN:
		i --; j --;
		if (alignment != 0) {
		    alignment[k] = CORR_ALIGN;
		}
		break;
	    case SUB_ALIGN:
		i --; j --;
		sub ++;
		if (alignment != 0) {
		    alignment[k] = SUB_ALIGN;
		}
		break;
	    case DEL_ALIGN:
		i --;
		del ++;
		if (alignment != 0) {
		    alignment[k] = DEL_ALIGN;
		}
		break;
	    case INS_ALIGN:
		j --;
		ins ++;
		if (alignment != 0) {
		    alignment[k] = INS_ALIGN;
		}
		break;
	    case END_ALIGN:
		assert(0);
		break;
	    }

	    k ++;
	}

	/*
	 * Now reverse the alignment to make the order correspond to words
	 */
	if (alignment) {
	    int k1, k2;	/* k2 can get negative ! */

	    for (k1 = 0, k2 = k - 1; k1 < k2; k1++, k2--) {
		WordAlignType x = alignment[k1];
		alignment[k1] = alignment[k2];
		alignment[k2] = x;
	    }

	    alignment[k] = END_ALIGN;
	}
	
	totalErrors = sub + del + ins;
    }

    return totalErrors;
}

void
wordError_freeThread()
{
    ChartEntry** &chart = TLSW_GET(chartTLS);

    if (chart != 0)
        freeChart(chart);

    TLSW_FREE(maxHypLengthTLS);
    TLSW_FREE(maxRefLengthTLS);
    TLSW_FREE(chartTLS);
}

