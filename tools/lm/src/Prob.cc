/*
 * Prob.cc --
 *	Functions for handling Probs
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2011 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Prob.cc,v 1.30 2016/06/19 04:36:59 stolcke Exp $";
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <limits>

#include "Prob.h"

#include "Array.cc"
#include "SArray.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_SARRAY(LogP,FloatCount);
#endif

const LogP LogP_Zero = -HUGE_VAL;		/* log(0) */
const LogP LogP_Inf = HUGE_VAL;			/* log(Inf) */
const LogP LogP_One = 0.0;			/* log(1) */

const unsigned LogP_Precision = numeric_limits<LogP>::digits10 + 1;
const unsigned LogP2_Precision = numeric_limits<LogP2>::digits10 + 1;
const unsigned Prob_Precision = numeric_limits<Prob>::digits10 + 1;

const Prob Prob_Epsilon = 3e-06;/* probability sums less than this in
				 * magnitude are effectively considered 0
				 * (assuming they were obtained by summing
				 * LogP's) */

const unsigned CodebookMaxIter = 10000;
				/* max iterations in codebook estimation */

/*
 * parseProb --
 *	Parse a Prob (double prec float) from a string.
 *	We don't enforce that 0 <= prob <= 1, since the values
 *	could be a sum or difference or probs etc.
 *
 * Results:
 *	true if string can be parsed as a float, false otherwise.
 *
 * Side effects:
 *	Result is set to double value if successful.
 *
 */
Boolean
parseProb(const char *str, Prob &result)
{
     double prob;
     if (sscanf(str, "%lf", &prob)) {
	result = prob;
	return true;
     } else {
	return false;
     }
}

/*
 * parseLogP --
 *	Fast parsing of floats representing log probabilities
 *
 * Results:
 *	true if string can be parsed as a float, false otherwise.
 *
 * Side effects:
 *	Result is set to float value if successful.
 *
 */
Boolean
parseLogP(const char *str, LogP2 &result)
{
    const unsigned maxDigits = 8;	// number of decimals in an integer

    const char *cp = str;
    const char *cp0;
    Boolean minus = false;

    if (*cp == '\0') {
	/* empty input */
	return false;
    }

    /*
     * Log probabilities are typically negative values of magnitude > 0.0001,
     * and thus are usually formatted without exponential notation.
     * We parse this type of format using integer arithmetic for speed,
     * and fall back onto scanf() in all other cases.
     * We also use scanf() when there are too many digits to handle with
     * integers.
     * Finally, we also parse +/- infinity values as they are printed by 
     * printf().  These are "[Ii]nf" or "[Ii]nfinity" or "1.#INF".
     */

    /*
     * Parse optional sign
     */
    if (*cp == '-') {
	minus = true;
	cp++;
    } else if (*cp == '+') {
	cp++;
    }
    cp0 = cp;

    unsigned long digits = 0;		// total value of parsed digits
    unsigned long decimals = 1;		// scaling factor from decimal point
    unsigned precision = 0;		// total number of parsed digits

    /*
     * Parse digits before decimal point
     */
    while (isdigit(*cp)) {
	digits = digits * 10 + (*(cp++) - '0');
	precision ++;
    }

    if (*cp == '.') {
	cp++;

	/*
	 * Parse digits after decimal point
	 */
	while (isdigit(*cp)) {
	    digits = digits * 10 + (*(cp++) - '0');
    	    precision ++;
	    decimals *= 10;
	}
    }

    /*
     * If we're at the end of the string then we're done.
     * Otherwise there was either an error or some format we can't
     * handle, so fall back on scanf(), after checking for infinity
     * values.
     */
    if (*cp == '\0' && precision <= maxDigits) {
	result = (minus ? - (LogP2)digits : (LogP2)digits) / (LogP2)decimals;
	return true;
    } else if ((*cp0 == 'i' || *cp0 == 'I' || 
	        (cp0[0] == '1' && cp0[1] == '.' && cp0[2] == '#')) &&
		(strncmp(cp0, "Inf", 3) == 0 || strncmp(cp0, "inf", 3) == 0 ||
		 strncmp(cp0, "1.#INF", 6) == 0))
    {
	result = (minus ? LogP_Zero : LogP_Inf);
	return true;
    } else {
	return (sscanf(str, "%lf", &result) == 1);
    }
}


/* 
 * Codebooks for quantized log probs
 */

Boolean
PQCodebook::read(File &file)
{
    char *line;
    char buffer[10];

    line = file.getline();

    if (!line || sscanf(line, "VQSize %u", &numBins) != 1) {
	file.position() << "missing VQSize spec\n";
	return false;
    }

    // Minimal sanity checking; can lower check on cap as appropriate
    const unsigned MAX_NUM_BINS = 0xFFFFFFFE;
    if (numBins > MAX_NUM_BINS) {
	file.position() << "numBins too large\n";
	return false;
    }

    for (unsigned i = 0; i < numBins; i ++) {
	binMeans[i] = LogP_Inf;
	binCounts[i] = 0;
    }

    binsAreSorted = false;
     
    line = file.getline();
    if (!line || 
	(sscanf(line, "Codeword Mean %9s", buffer) != 1 &&
	 sscanf(line, "Codword Mean %9s", buffer) != 1) ||
 	strcmp(buffer, "Count") != 0)
    {
	file.position() << "malformed Codeword header\n";
	return false;
    }

    while ((line = file.getline())) {
        unsigned bin;
	double prob;
	double count;
	if (sscanf(line, "%u %lf %lg", &bin, &prob, &count) != 3) {
	    file.position() << "malformed codeword line\n";
	    return false;
	}

	if (bin >= numBins) {
	    file.position() << "codeword index out of range\n";
	    return false;
	}

	/*
	 * Codebook means are encoded as natural logs -- convert to base 10.
	 */
	binMeans[bin] = prob / M_LN10;	
	binCounts[bin] = count;
     }

     return true;
}

Boolean
PQCodebook::write(File &file)
{
    file.fprintf("VQSize %u\n", numBins);
    file.fprintf("%8s %20s %12s\n", "Codeword", "Mean", "Count");

    for (unsigned i = 0; i < numBins; i ++) {
	file.fprintf("%8d %20.16lg %12lg\n", 
			i, (double)(binMeans[i] * M_LN10),
			(double)binCounts[i]);
    }

    return true;
}

LogP2
PQCodebook::getProb(unsigned bin)
{
    if (bin < numBins) {
	return binMeans[bin];
    } else {
	return LogP_Inf;
    }
}

static PQCodebook *sorting;

static int binCompare(const void *bin1, const void *bin2)
{
    unsigned index1 = *(unsigned *)bin1;
    unsigned index2 = *(unsigned *)bin2;

    LogP2 mean1 = sorting->getProb(index1);
    LogP2 mean2 = sorting->getProb(index2);

    if (mean1 < mean2) return -1;
    else if (mean1 > mean2) return +1;
    else return 0;
}

void
PQCodebook::sortBins()
{
    for (unsigned i = 0; i < numBins; i ++) {
	binOrder[i] = i;
    }

    sorting = this;

    qsort(binOrder, numBins, sizeof(binOrder[0]), binCompare);
    binsAreSorted = true;
}

unsigned
PQCodebook::getBin(LogP2 prob)
{
    if (!binsAreSorted) {
	sortBins();
    
#if 0
	for (unsigned i = 0; i < numBins; i ++) {
	    cerr << i << " " << getProb(binOrder[i]) << endl;
	}
#endif
    }
    
    if (numBins == 0) {
	return 0;
    }

    if (prob == LogP_Zero) {
	return binOrder[0];
    }

    if (prob < binMeans[binOrder[0]]) {
	return binOrder[0];
    }
    if (prob > binMeans[binOrder[numBins-1]]) {
	return binOrder[numBins-1];
    }

    /*
     * Perform binary search
     */
    unsigned low = 0;
    unsigned high = numBins - 1;

    while (low + 1 < high) {
	unsigned mid = (low + high)/2;

	if (binMeans[binOrder[mid]] >= prob) {
	    high = mid;
	} else {
	    low = mid;
	}
    }

    if (low == high) {
	return binOrder[low];
    } else {
	/* low + 1 == high */
        if (prob - binMeans[binOrder[low]] < binMeans[binOrder[high]] - prob) {
	    return binOrder[low];
	} else {
	    return binOrder[high];
	}
    }
}

Boolean
PQCodebook::estimateInit(SArray<LogP, FloatCount> &data, unsigned nbins)
{
    FloatCount totalCount = 0;

    SArrayIter<LogP, FloatCount> dataIter(data);

    /*
     * Determine total count
     */
    FloatCount *count;
    LogP val;
    while ((count = dataIter.next(val))) {
	totalCount += *count;
    }

    if (totalCount == 0) {
	return false;
    }

    /*
     * Clear old parameters
     */
    binMeans.clear();
    binCounts.clear();
    binOrder.clear();
    numBins = 0;

    dataIter.init();

    FloatCount totalCountLeft = totalCount;
    FloatCount avgCountPerBin = totalCount / nbins;

    FloatCount countInBin = 0;
    LogP2 sumInBin = 0.0;
    Boolean keepGoing = true;

    /*
     * Collect data into bins while there is some left
     */
    while (keepGoing) {
	count = dataIter.next(val);

	if (count == 0) {
	    keepGoing = false;
	} else {
	    countInBin += *count;
	    sumInBin = *count * val;
	}

	if (sumInBin == LogP_Zero || 
	    countInBin >= avgCountPerBin)
	{
	    /*
	     * Make a new bin based on data so far
	     */
	    binMeans[numBins] = sumInBin / countInBin;
	    binCounts[numBins] = countInBin;
	    binOrder[numBins] = numBins;
	    numBins += 1;

	    totalCountLeft -= countInBin;
	    if (nbins - numBins > 0) {
		avgCountPerBin = totalCountLeft / (nbins - numBins);
	    }

	    countInBin = 0;
	    sumInBin = 0.0;
	}
    }

    if (countInBin > 0) {
	binMeans[numBins] = sumInBin / countInBin;
	binCounts[numBins] = countInBin;
	binOrder[numBins] = numBins;
	numBins += 1;
    }
	
    binsAreSorted = true;
    
    return true;
}

LogP2
PQCodebook::estimateUpdate(SArray<LogP, FloatCount> &data, unsigned nbins)
{
    SArrayIter<LogP, FloatCount> dataIter(data);

    LogP2 totalError= 0.0;
    FloatCount totalCount = 0;

    Array<LogP> binSums;
    unsigned i;

    for (i = 0; i < numBins; i ++) {
	binCounts[i] = 0;
	binSums[i] = 0.0;
    }

    /*
     * Reassign all parameters to best bin
     */
    FloatCount *count;
    LogP val;
    while ((count = dataIter.next(val))) {
	unsigned bin = getBin(val);

	LogP2 diff = val - binMeans[bin];

	if (val == LogP_Zero && binMeans[bin] == LogP_Zero) {
	    diff = 0.0;
	}

	binCounts[bin] += *count;

	totalCount += *count;
	totalError += *count * diff * diff;
	binSums[bin] += val * *count;
    }

    for (i = 0; i < numBins; i ++) {
	if (binCounts[i] > 0) {
	    binMeans[i] = binSums[i] / binCounts[i];
	} else {
	    binMeans[i] = 0.0;
	}
    }

    sortBins();

    return totalError / totalCount;
}

Boolean
PQCodebook::estimate(SArray<LogP, FloatCount> &data, unsigned nbins)
{
    if (!estimateInit(data, nbins)) {
	return false;
    }

    LogP2 mse = 0.0, lastError;

    for (unsigned k = 1; k <= CodebookMaxIter; k ++) {
	lastError = sqrt(mse);
	mse = estimateUpdate(data, nbins);
	cerr << "iter " << k
	     << " mse " << mse
	     << " error " << sqrt(mse) << endl;

	if (k > 1 && lastError - sqrt(mse) <= Prob_Epsilon) {
	    break;
	}
    }

    return true;
}

