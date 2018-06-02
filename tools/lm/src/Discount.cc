/*
 * Discount.cc --
 *	Discounting methods
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2011 SRI International, 2011-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Discount.cc,v 1.33 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <math.h>

#include "Prob.h"
#include "Discount.h"

#include "Array.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_ARRAY(double);
#endif

/*
 * Determine the true vocab size, i.e., the number of true event
 * tokens, by enumeration. Used in various smoothing methods.
 */
unsigned
Discount::vocabSize(Vocab &vocab)
{
    VocabIter viter(vocab);
    VocabIndex wid;

    unsigned total = 0;
    while (viter.next(wid)) {
	if (!vocab.isNonEvent(wid)) {
	    total ++;
	}
    }

    return total;
}

/*
 * Good-Turing discounting
 */
GoodTuring::GoodTuring(unsigned mincount, unsigned maxcount)
   : minCount(mincount), maxCount(maxcount), discountCoeffs(0)
{
   /*
    * a zero count cannot be discounted
    */
   discountCoeffs[0] = 1.0;
}

/*
 * Debug levels used here
 */
#define DEBUG_ESTIMATE_DISCOUNT	1

/*
 * GT discounting uses the formula
 *
 *   c_discounted = (c + 1) * n_(c+1)/n_c
 *
 * where n_c is the count of count c .
 */
double
GoodTuring::discount(Count count, Count totalCount, Count observedVocab)
{
    if (count <= 0) {
	return 1.0;
    } else if (count < minCount) {
	return 0.0;
    } else if (count > maxCount) {
	return 1.0;
    } else {
	return discountCoeffs[count];
    }
}

/*
 * GT Discounting is effectively disabled if the upper cutoff is 0 or less
 * and the minimum count is no greater than 1 (i.e., no ngrams are excluded).
 */
Boolean
GoodTuring::nodiscount()
{
    return (minCount <= 1 && maxCount <= 0);
}

/*
 * Write all internal parameters to file
 */
void
GoodTuring::write(File &file)
{
    file.fprintf("mincount %s\n", countToString(minCount));
    file.fprintf("maxcount %s\n", countToString(maxCount));

    for (unsigned i = 1; !file.error() && i <= maxCount; i++) {
	file.fprintf("discount %u %.*lg\n", i, Prob_Precision, discountCoeffs[i]);
    }
}

/*
 * Read parameters back from file
 */
Boolean
GoodTuring::read(File &file)
{
    char *line;

    while ((line = file.getline())) {
	char buffer[100];
	unsigned count;
	double coeff;
	
	if (sscanf(line, "mincount %99s", buffer) == 1 &&
	    stringToCount(buffer, minCount))
	{
	    continue;
	} else if (sscanf(line, "maxcount %99s", buffer) == 1 &&
	    stringToCount(buffer, maxCount))
	{
	    if (maxCount > maxNgramOrder) {
	        file.position() << "maxcount value out of range\n";
		return false;
	    }

	    /*
	     * Zero all old discount coeffs
	     */
	    for (Count n = 0; n <= maxCount; n++) {
		discountCoeffs[n] = 0.0;
	    }
	} else if (sscanf(line, "discount %u %lf", &count, &coeff) == 2) {
	    /*
	     * It's okay for the count value to be larger than maxCount,
	     * we just make sure it's not unreasonably large
	     */
	    if (count <= maxNgramOrder) {
	        discountCoeffs[count] = coeff;
	    } else {
	        file.position() << "warning: count value out of range\n";
	    }
	} else {
	    file.position() << "unrecognized parameter\n";
	    return false;
	}
    }

    // Add 2nd check in case "&& stringToCount()" failed
    if (maxCount > maxNgramOrder) {
	file.position() << "maxcount value out of range\n";
	return false;
    }

    for (Count n = minCount; n <= maxCount; n++) {
	if (discountCoeffs[n] == 0.0) {
	    file.position() << "warning: discount coefficient " << n
			    << " = 0.0\n";
	}
    }
    return true;
}

/*
 * Estimation of discount coefficients from ngram count-of-counts
 *
 * The Good-Turing formula for this is
 *
 *	d(c) = (c+1)/c * n_(c+1)/n_c
 */
Boolean
GoodTuring::estimate(NgramStats &counts, unsigned order)
{
    Array<Count> countOfCounts;

    /*
     * First tabulate count-of-counts for the given order of ngrams 
     * Note we need GT count for up to maxCount + 1 inclusive, to apply
     * the GT formula for counts up to maxCount.
     */
    makeArray(VocabIndex, wids, order + 1);

    NgramsIter iter(counts, wids, order);
    NgramCount *count;
    Count i;

    for (i = 0; i <= maxCount + 1; i++) {
	countOfCounts[i]  = 0;
    }

    while ((count = iter.next())) {
	if (counts.vocab.isNonEvent(wids[order - 1])) {
	    continue;
	} else if (counts.vocab.isMetaTag(wids[order - 1])) {
	    unsigned type = counts.vocab.typeOfMetaTag(wids[order - 1]);

	    /*
	     * process count-of-count
	     */
	    if (type > 0 && type <= maxCount + 1) {
		countOfCounts[type] += *count;
	    }
	} else if (*count <= maxCount + 1) {
	    countOfCounts[*count] ++;
	}
    }

    if (debug(DEBUG_ESTIMATE_DISCOUNT)) {
	dout() << "Good-Turing discounting " << order << "-grams\n";
	for (i = 0; i <= maxCount + 1; i++) {
	    dout() << "GT-count [" << i << "] = " << countOfCounts[i] << endl;
	}
    }

    if (countOfCounts[1] == 0) {
	cerr << "warning: no singleton counts\n";
	maxCount = 0;
    }

    while (maxCount > 0 && countOfCounts[maxCount + 1] == 0) {
	cerr << "warning: count of count " << maxCount + 1 << " is zero "
	     << "-- lowering maxcount\n";
	maxCount --;
    }

    if (maxCount <= 0) {
	cerr << "GT discounting disabled\n";
    } else {
	double commonTerm = (maxCount + 1) *
				(double)countOfCounts[maxCount + 1] /
				    (double)countOfCounts[1];

	for (i = 1; i <= maxCount; i++) {
	    double coeff;

	    if (countOfCounts[i] == 0) {
		cerr << "warning: count of count " << i << " is zero\n";
		coeff = 1.0;
	    } else {
		double coeff0 = (i + 1) * (double)countOfCounts[i+1] /
					    (i * (double)countOfCounts[i]);
		coeff = (coeff0 - commonTerm) / (1.0 - commonTerm);
		if (!isfinite(coeff) || coeff <= Prob_Epsilon || coeff0 > 1.0) {
		    cerr << "warning: discount coeff " << i
			 << " is out of range: " << coeff << "\n";
		    coeff = 1.0;
		}
	    }
	    discountCoeffs[i] = coeff;
	}
    }

    return true;
}


/*
 * Eric Ristad's Natural Law of Succession --
 *	The discount factor d is identical for all counts,
 *
 *	d = ( n(n+1) + q(1-q) ) / 
 *	    ( n^2 + n + 2q ) 
 *
 *  where n is the total number events tokens, q is the number of observed
 *  event types.  If q equals the vocabulary size no discounting is 
 *  performed.
 */

double
NaturalDiscount::discount(Count count, Count totalCount, Count observedVocab)
{
    double n = totalCount;
    double q = observedVocab;

    if (count < _mincount) {
	return 0.0;
    } else if (q == _vocabSize) {
	return 1.0;
    } else {
	return (n * (n+1) + q * (1 - q)) / (n * (n + 1) + 2 * q);
    }
}

Boolean
NaturalDiscount::estimate(NgramStats &counts, unsigned order)
{
    _vocabSize = vocabSize(counts.vocab);
    return true;
}


/*
 * Unmodified (i.e., regular) Kneser-Ney discounting
 */
double
KneserNey::discount(Count count, Count totalCount, Count observedVocab)
{
    if (count <= 0) {
	return 1.0;
    } else if (count < minCount) {
	return 0.0;
    } else {
	return (count - discount1) / count;
    }
}

double
KneserNey::lowerOrderWeight(Count totalCount, Count observedVocab,
					    Count min2Vocab, Count min3Vocab)
{
    return (discount1 * observedVocab / totalCount);
}

void
KneserNey::write(File &file)
{
    file.fprintf("mincount %s\n", countToString(minCount));
    file.fprintf("discount1 %.*lg\n", Prob_Precision, discount1);
}

Boolean
KneserNey::read(File &file)
{
    char *line;

    while ((line = file.getline())) {
	char buffer[100];
	unsigned count;
	double coeff;
	
	if (sscanf(line, "mincount %99s", buffer) == 1 &&
	    stringToCount(buffer, minCount))
	{
	    continue;
	} else if (sscanf(line, "discount1 %lf", &discount1) == 1) {
	    continue;
	} else {
	    file.position() << "unrecognized parameter\n";
	    return false;
	}
    }
    return true;
}

Boolean
KneserNey::estimate(NgramStats &counts, unsigned order)
{
    if (!prepareCountsAtEnd) {
	prepareCounts(counts, order, counts.getorder());
    }

    /*
     * First tabulate count-of-counts
     */
    Count n1 = 0;
    Count n2 = 0;

    makeArray(VocabIndex, wids, order + 1);
    NgramsIter iter(counts, wids, order);
    NgramCount *count;

    while ((count = iter.next())) {
	if (counts.vocab.isNonEvent(wids[order - 1])) {
	    continue;
	} else if (counts.vocab.isMetaTag(wids[order - 1])) {
	    unsigned type = counts.vocab.typeOfMetaTag(wids[order - 1]);

	    /*
	     * process count-of-count
	     */
	    if (type == 1) {
		n1 ++;
	    } else if (type == 2) {
		n2 ++;
	    }
	} else if (*count == 1) {
	    n1 ++;
	} else if (*count == 2) {
	    n2 ++;
	}
    }
	    
    if (debug(DEBUG_ESTIMATE_DISCOUNT)) {
	dout() << "Kneser-Ney smoothing " << order << "-grams\n"
	       << "n1 = " << n1 << endl
	       << "n2 = " << n2 << endl;
    }

    if (n1 == 0 || n2 == 0) {
	cerr << "one of required KneserNey count-of-counts is zero\n";
	return false;
    }

    discount1 = n1/((double)n1 + 2*n2);

    if (debug(DEBUG_ESTIMATE_DISCOUNT)) {
      dout() << "D = " << discount1 << endl;
    }

    if (prepareCountsAtEnd) {
	prepareCounts(counts, order, counts.getorder());
    }
    return true;
}

void
KneserNey::prepareCounts(NgramCounts<NgramCount> &counts, unsigned order,
							 unsigned maxOrder)
{
    if (countsAreModified || order >= maxOrder) {
	return;
    }

    if (debug(DEBUG_ESTIMATE_DISCOUNT)) {
	dout() << "modifying "
	       << order << "-gram counts for Kneser-Ney smoothing\n";
    }

    /*
     * For the lower-order counts in KN discounting we need to replace the
     * counts to reflect the number of contexts in which a given N-gram
     * occurs.  Specifically,
     *
     *		c(w2,...,wN) = number of N-grams w1,w2,...wN with count > 0
     */
    makeArray(VocabIndex, ngram, order + 2);

    /*
     * clear all counts of given order 
     * Note: exclude N-grams starting with non-events (such as <s>) since there
     * are usually no words preceeding them.
     */
    {
	NgramCountsIter<NgramCount> iter(counts, ngram, order);
	NgramCount *count;

	while ((count = iter.next())) {
	    if (!counts.vocab.isNonEvent(ngram[0])) {
		*count = 0;
	    }
	}
    }

    /*
     * Now accumulate new counts
     */
    {
	NgramCountsIter<NgramCount> iter(counts, ngram, order + 1);
	NgramCount *count;

	while ((count = iter.next())) {
	    if (*count > 0 && !counts.vocab.isNonEvent(ngram[1])) {
		NgramCount *loCount = counts.findCount(&ngram[1]);

		if (loCount) {
		    (*loCount) += 1;
		}
	    }
	}
    }

    countsAreModified = true;
}


/*
 * Modified Kneser-Ney discounting
 *	as described in S. F. Chen & J. Goodman, An Empirical Study of 
 *	Smoothing Techniques for Language Modeling, TR-10-98, Computer
 *	Science Group, Harvard University, Cambridge, MA, August 1998.
 */
double
ModKneserNey::discount(Count count, Count totalCount, Count observedVocab)
{
    if (count <= 0) {
	return 1.0;
    } else if (count < minCount) {
	return 0.0;
    } else if (count == 1) {
	return (count - discount1) / count;
    } else if (count == 2) {
	return (count - discount2) / count;
    } else {
	return (count - discount3plus) / count;
    }
}

double
ModKneserNey::lowerOrderWeight(Count totalCount, Count observedVocab,
					    Count min2Vocab, Count min3Vocab)
{
    return (discount1 * (observedVocab - min2Vocab) +
	    discount2 * (min2Vocab - min3Vocab) +
	    discount3plus * min3Vocab) / totalCount;
}

void
ModKneserNey::write(File &file)
{
    file.fprintf("mincount %s\n", countToString(minCount));
    file.fprintf("discount1 %.*lf\n", Prob_Precision, discount1);
    file.fprintf("discount2 %.*lf\n", Prob_Precision, discount2);
    file.fprintf("discount3+ %.*lf\n", Prob_Precision, discount3plus);
}

Boolean
ModKneserNey::read(File &file)
{
    char *line;

    while ((line = file.getline())) {
	char buffer[100];
	unsigned count;
	double coeff;
	
	if (sscanf(line, "mincount %99s", buffer) == 1 &&
	    stringToCount(buffer, minCount))
	{
	    continue;
	} else if (sscanf(line, "discount1 %lf", &discount1) == 1) {
	    continue;
	} else if (sscanf(line, "discount2 %lf", &discount2) == 1) {
	    continue;
	} else if (sscanf(line, "discount3+ %lf", &discount3plus) == 1) {
	    continue;
	} else {
	    file.position() << "unrecognized parameter\n";
	    return false;
	}
    }
    return true;
}

Boolean
ModKneserNey::estimate(NgramStats &counts, unsigned order)
{
    if (!prepareCountsAtEnd) {
	prepareCounts(counts, order, counts.getorder());
    }

    /*
     * First tabulate count-of-counts
     */
    Count n1 = 0;
    Count n2 = 0;
    Count n3 = 0;
    Count n4 = 0;

    makeArray(VocabIndex, wids, order + 1);
    NgramsIter iter(counts, wids, order);
    NgramCount *count;

    while ((count = iter.next())) {
	if (counts.vocab.isNonEvent(wids[order - 1])) {
	    continue;
	} else if (counts.vocab.isMetaTag(wids[order - 1])) {
	    unsigned type = counts.vocab.typeOfMetaTag(wids[order - 1]);

	    /*
	     * process count-of-count
	     */
	    if (type == 1) {
		n1 ++;
	    } else if (type == 2) {
		n2 ++;
	    } else if (type == 3) {
		n3 ++;
	    } else if (type == 4) {
		n4 ++;
	    }
	} else if (*count == 1) {
	    n1 ++;
	} else if (*count == 2) {
	    n2 ++;
	} else if (*count == 3) {
	    n3 ++;
	} else if (*count == 4) {
	    n4 ++;
	}
    }
	    
    if (debug(DEBUG_ESTIMATE_DISCOUNT)) {
	dout() << "Kneser-Ney smoothing " << order << "-grams\n"
	       << "n1 = " << n1 << endl
	       << "n2 = " << n2 << endl
	       << "n3 = " << n3 << endl
	       << "n4 = " << n4 << endl;
    }

    if (n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0) {
	cerr << "one of required modified KneserNey count-of-counts is zero\n";
	return false;
    }

    /*
     * Compute discount constants (Ries 1997, Chen & Goodman 1998)
     */
    double Y = (double)n1/(n1 + 2 * n2);

    discount1 = 1 - 2 * Y * n2 / n1;
    discount2 = 2 - 3 * Y * n3 / n2;
    discount3plus = 3 - 4 * Y * n4 / n3;

    if (debug(DEBUG_ESTIMATE_DISCOUNT)) {
	dout() << "D1 = " << discount1 << endl
	       << "D2 = " << discount2 << endl
	       << "D3+ = " << discount3plus << endl;
    }

    if (discount1 < 0.0 || discount2 < 0.0 || discount3plus < 0.0) {
	cerr << "one of modified KneserNey discounts is negative\n";
	return false;
    }

    if (prepareCountsAtEnd) {
	prepareCounts(counts, order, counts.getorder());
    }
    return true;
}

