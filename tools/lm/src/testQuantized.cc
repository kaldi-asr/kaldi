/*
 * testQuantized --
 *	Test for prob codebooks and quantization
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2014 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testQuantized.cc,v 1.3 2014-05-28 00:04:02 stolcke Exp $";
#endif

#include <stdio.h>

#include "File.h"
#include "Prob.h"

int
main (int argc, char *argv[])
{
    PQCodebook cb;

    if (argc >= 2) {
	File file(argv[1], "r");

	cb.read(file);
    } else {
	cerr << "usage: " << argv[0] << " codebook [data [nbins]]\n";
	exit(1);
    }

    {
	File file(stderr);

	cb.write(file);
    }

    if (argc >= 3) {
	SArray<LogP, FloatCount> data;

	File file(argv[2], "r");

	char *line;
	while ((line = file.getline())) {
	    LogP val;
	    if (parseLogP(line, val)) {
	         //cerr << " val = " << val << endl;
		*data.insert((LogP)val) += 1;
	    }
	}

	SArrayIter<LogP, FloatCount> dataIter(data);

	FloatCount *count;
	LogP val;
	while ((count = dataIter.next(val))) {
	    unsigned bin = cb.getBin(val);

	    cerr << "val = " << val
		 << " count = " << *count
		 << " bin = " << bin
		 << " mean = " << cb.getProb(bin)
		 << endl;
	}

	unsigned numbins;

	if (argc >= 4 && sscanf(argv[3], "%u", &numbins) == 1) {
	    cerr << "numbins = " << numbins << endl;
	} else {
	    numbins = 256;
	}

	if (cb.estimate(data, numbins)) {
	    File file(stdout);
	    cb.write(file);

	} else {
	    cerr << "codebook estimation failed\n";
	    exit(1);
	}
    }

    exit(0);
}
