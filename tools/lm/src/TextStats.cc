/*
 * TextStats.cc --
 *	Text statistics
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2011 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/TextStats.cc,v 1.10 2016/04/08 23:34:42 stolcke Exp $";
#endif

#include "TextStats.h"

/*
 * Increments from other source
 */
TextStats &
TextStats::increment(const TextStats &stats, FloatCount weight)
{
    numSentences += stats.numSentences * weight;
    numWords += stats.numWords * weight;
    numOOVs += stats.numOOVs * weight;
    prob += stats.prob * weight;
    zeroProbs += stats.zeroProbs * weight;

    /* 
     * Ranking and loss metrics
     */
    r1  += stats.r1 * weight;
    r5  += stats.r5 * weight;
    r10 += stats.r10 * weight;

    r1se  += stats.r1se * weight;
    r5se  += stats.r5se * weight;
    r10se += stats.r10se * weight;

    rTotal += stats.rTotal * weight;

    posQuadLoss += stats.posQuadLoss * weight;
    posAbsLoss += stats.posAbsLoss * weight;
      
    return *this;
}

/*
 * Format stats for stream output
 */
ostream &
operator<< (ostream &stream, const TextStats &stats)
{

    unsigned oldprec = stream.precision();

    // output count values with maximal precision
    stream.precision(FloatCount_Precision);
    stream << stats.numSentences << " sentences, " 
           << stats.numWords << " words, "
	   << stats.numOOVs << " OOVs" << endl;

    if (stats.numWords + stats.numSentences > 0) {
	stream << stats.zeroProbs << " zeroprobs, ";

	// set precision for LogP-based values following
	stream.precision(LogP_Precision);
	stream << "logprob= " << stats.prob;

	double denom = stats.numWords - stats.numOOVs - stats.zeroProbs
							+ stats.numSentences;

	if (denom > 0) {
	    stream << " ppl= " << LogPtoPPL(stats.prob / denom);
	} else {
	    stream << " ppl= undefined";
	}

	denom -= stats.numSentences;

	if (denom > 0) {
	    stream << " ppl1= " << LogPtoPPL(stats.prob / denom);
	} else {
	    stream << " ppl1= undefined";
	}

	/*
	 * Ranking and loss metrics
	 */
	if (stats.rTotal > 0) {
	    FloatCount denom1 = stats.rTotal - stats.numSentences;
	    FloatCount denom2 = stats.rTotal;

	    if (denom2 > 0) {
		stream.precision(FloatCount_Precision);
		stream << endl << denom1 << " words,";
		stream << " rank1= " << (denom1 > 0 ? stats.r1 / denom1 : 0.0);
		stream << " rank5= " << (denom1 > 0 ? stats.r5 / denom1 : 0.0);
		stream << " rank10= " << (denom1 > 0 ? stats.r10 / denom1 : 0.0);

		stream << endl << denom2 << " words+sents,";
		stream << " rank1wSent= " << (stats.r1 + stats.r1se) / denom2;
		stream << " rank5wSent= " << (stats.r5 + stats.r5se) / denom2;
		stream << " rank10wSent= " << (stats.r10 + stats.r10se) / denom2;
		stream << " qloss= " << sqrt(stats.posQuadLoss / denom2);
		stream << " absloss= " << stats.posAbsLoss / denom2;
	    }
        }

	stream << endl;
    } 
    stream.precision(oldprec);

    return stream;
}

