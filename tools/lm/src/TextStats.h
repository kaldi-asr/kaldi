/*
 * TextStats.h --
 *	Text source statistics
 *
 * TextStats objects are used to pass and accumulate various 
 * statistics of text sources (training or test).
 *
 * Copyright (c) 1995-2009 SRI International, 2012-2015 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/TextStats.h,v 1.9 2015-10-13 21:04:27 stolcke Exp $
 *
 */

#ifndef _TextStats_h_
#define _TextStats_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif

#include "Prob.h"
#include "Counts.h"

class TextStats
{
public:
    TextStats() : prob(0.0), zeroProbs(0.0),
	numSentences(0.0), numWords(0.0), numOOVs(0.0),
	r1(0), r5(0), r10(0), r1se(0), r5se(0), r10se(0),
	rTotal(0), posQuadLoss(0), posAbsLoss(0)
	{};

    void reset() { prob = 0.0, zeroProbs = 0.0,
	numSentences = numWords = numOOVs = 0.0;
	r1 = r5 = r10 = r1se = r5se = r10se = rTotal = 0;
	posQuadLoss = posAbsLoss = 0.0; };
    TextStats &increment(const TextStats &stats, FloatCount weight = 1.0);

    LogP2 prob;
    FloatCount zeroProbs;
    FloatCount numSentences;
    FloatCount numWords;
    FloatCount numOOVs;

    /*
     * Ranking and loss metrics
     */
    FloatCount r1;	// rank <= 1
    FloatCount r5;	// rank <= 5
    FloatCount r10;	// rank <= 10
    
    FloatCount r1se;	// same, but for </s>
    FloatCount r5se;
    FloatCount r10se;
    FloatCount rTotal;	// total tokens ranked

    double posQuadLoss;	// quadratic loss
    double posAbsLoss;  // absolute loss
};

ostream &operator<<(ostream &, const TextStats &stats);

#endif /* _TextStats_h_ */

