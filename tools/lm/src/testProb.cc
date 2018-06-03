/*
 * testProb --
 *	Test arithmetic with log probabilities
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2000-2006 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testProb.cc,v 1.13 2016/04/09 06:53:01 stolcke Exp $";
#endif


#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits>

#include "Prob.h"
#include "Counts.h"

/*
 * Simulate the rounding going on from the original LM LogP scores to the
 * bytelogs in the recognizer:
 * - PFSGs encode LogP as intlogs
 * - Nodearray compiler maps intlogs to bytelogs
 */
#define RoundToBytelog(x)	BytelogToLogP(IntlogToBytelog(LogPtoIntlog(x)))

int
main(int argc, char **argv)
{
    if (argc < 2) {
    	cerr << "usage: testProb p1 [p2]\n";
	exit(2);
    }

    cout << "log(0) = " << LogP_Zero << " ; isfinite = " << isfinite(LogP_Zero) << endl;
    cout << "log(inf) = " << LogP_Inf << " ; isfinite = " << isfinite(LogP_Inf) << endl;
    double n = 0.0;
    n = n / 0.0;
    cout << "NaN = " << n << " ; isnan = " << isnan(n) << endl;

    if (argc < 3) {
    	Prob p;

	if (!parseProb(argv[1], p)) {
	    cerr << "bad prob value " << argv[1] << endl;
	    exit(1);
	}
	LogP lp = ProbToLogP(p);

    	cout << "log(p) = " << lp << " " << LogPtoProb(lp) << endl;

	char buffer[200];
	LogP lp2;

	sprintf(buffer, "%.*lf ", LogP_Precision, lp);
	if (parseLogP(buffer, lp2)) {
		cout << "lp read back = " << lp2 << endl;
	} else {
		cout << "lp read back FAILED\n";
	}

    	cout << "Decipher log(p) = " << RoundToBytelog(lp)
		<< " " << LogPtoProb(RoundToBytelog(lp))
		<< " " << LogPtoIntlog(lp)
		<< " " << IntlogToBytelog(LogPtoIntlog(lp)) << endl;
    } else {
    	Prob p, q;

	if (!parseProb(argv[1], p)) {
	    cerr << "bad prob value " << argv[1] << endl;
	    exit(1);
	}
	if (!parseProb(argv[2], q)) {
	    cerr << "bad prob value " << argv[2] << endl;
	    exit(1);
	}

	LogP lp = ProbToLogP(p);
	LogP lq = ProbToLogP(q);
	LogP lpq = AddLogP(lp,lq);

    	cout << "log(p + q) = " << lpq << " " << LogPtoProb(lpq) << endl;

	if (lp >= lq) {
	    lpq = SubLogP(lp,lq);

	    cout << "log(p - q) = " << lpq << " " << LogPtoProb(lpq) << endl;
	}
    }

    cout << "LogP_Precision = " << LogP_Precision << endl;
    cout << "Prob_Precision = " << Prob_Precision << endl;
    cout << "FloatCount_Precision = " << FloatCount_Precision << endl;

    exit(0);
}
