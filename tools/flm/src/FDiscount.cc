/*
 * FDiscount.cc --
 *	Discounting methods for factored LMs
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/FDiscount.cc,v 1.13 2010/06/02 05:51:57 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>

#include "hexdec.h"

#include "FDiscount.h"
#include "Array.cc"

/*
 * Debug levels used here
 */
#define DEBUG_ESTIMATE_DISCOUNT	1
#define DEBUG_PRINT_CONTEXTS_WO_COUNTS 8
#define DEBUG_PRINT_WARNING_CONTEXTS_WO_COUNTS 0

// from defined in FNgramLM.cc
// TODO: place all bit routines in separate file.
extern unsigned int bitGather(unsigned int mask,unsigned int bitv);

/*
 * For factored models, same as GoodTuring::estimate(), except different
 * interface.
 *
 * Estimation of discount coefficients from ngram count-of-counts
 *
 * The Good-Turing formula for this is
 *
 *	d(c) = (c+1)/c * n_(c+1)/n_c
 */
Boolean
FGoodTuring::estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab)
{
    Array<Count> countOfCounts;
    FNgramSpecs<FNgramCount>::FNgramSpec::ParentSubset& counts =
						spec.parentSubsets[node];

    /*
     * First tabulate count-of-counts for the given order of ngrams 
     * Note we need GT count for up to maxCount + 1 inclusive, to apply
     * the GT formula for counts up to maxCount.
     */
    makeArray(VocabIndex, wids, counts.order + 1);

    FNgramSpecs<FNgramCount>::FNgramSpec::PSIter iter(counts, wids);
    NgramCount *count;
    Count i;

    for (i = 0; i <= maxCount + 1; i++) {
	countOfCounts[i]  = 0;
    }

    while ((count = iter.next())) {
	if (vocab.isNonEvent(wids[counts.order - 1])) {
	    continue;
	} else if (vocab.isMetaTag(wids[counts.order - 1])) {
	    unsigned type = vocab.typeOfMetaTag(wids[counts.order - 1]);

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

    if (FDiscount::debug(DEBUG_ESTIMATE_DISCOUNT)) {
	FDiscount::dout()
	    << "Good-Turing discounting " << counts.order << "-grams\n";
	for (i = 0; i <= maxCount + 1; i++) {
	    FDiscount::dout()
		<< "GT-count [" << i << "] = " << countOfCounts[i] << endl;
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
		if (coeff <= Prob_Epsilon || coeff0 > 1.0) {
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
Boolean
FNaturalDiscount::estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab)
{
    _vocabSize = Discount::vocabSize(vocab);
    return true;
}


/*
 * Unmodified (i.e., regular) Kneser-Ney discounting
 */
Boolean
FKneserNey::estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
			    unsigned int node, FactoredVocab& vocab)
{
    if (!prepareCountsAtEnd) {
	prepareCounts(spec, node, vocab);
    }

    /*
     * First tabulate count-of-counts
     */
    Count n1 = 0;
    Count n2 = 0;

    FNgramSpecs<FNgramCount>::FNgramSpec::ParentSubset& counts =
						spec.parentSubsets[node];

    VocabIndex wids[maxNumParentsPerChild + 1];
    FNgramSpecs<FNgramCount>::FNgramSpec::PSIter iter(counts, wids);
    NgramCount *count;

    while ((count = iter.next())) {
	if (vocab.isNonEvent(wids[counts.order - 1])) {
	    continue;
	} else if (vocab.isMetaTag(wids[counts.order - 1])) {
	    unsigned type = vocab.typeOfMetaTag(wids[counts.order - 1]);

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
	    
    if (FDiscount::debug(DEBUG_ESTIMATE_DISCOUNT)) {
	FDiscount::dout()
	       << "Kneser-Ney smoothing " << HEX << node << DEC << "-grams\n"
	       << "n1 = " << n1 << endl
	       << "n2 = " << n2 << endl;
    }

    if (n1 == 0 || n2 == 0) {
	cerr << "warning: one of required KneserNey count-of-counts is zero\n";
	return false;
    }

    discount1 = n1/((double)n1 + 2*n2);

    if (FDiscount::debug(DEBUG_ESTIMATE_DISCOUNT)) {
	FDiscount::dout() << "D = " << discount1 << endl;
    }

    if (prepareCountsAtEnd) {
	prepareCounts(spec, node, vocab);
    }
    return true;
}

void
FKneserNey::prepareCounts(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab)
{
    if (countsAreModified || node == (spec.numSubSets-1)) {
	return;
    }

    // if (debug(DEBUG_ESTIMATE_DISCOUNT)) {
    // nothing like a simple way to print out a lousy hex number!
    // dout() << "Modifying node 0x" << HEX << node << DEC
    // << "-gram counts for Kneser-Ney smoothing\n";
    // }

    /*
     * For the lower-order counts in KN discounting we need to replace the
     * counts to reflect the number of contexts in which a given N-gram
     * occurs.  Specifically,
     *
     *		c(w2,...,wN) = number of N-grams w1,w2,...wN with count > 0
     *                       = |{ w1 : c(w1,w2,...wN) > 0 }|
     *                       = N1+(*,w2,...,wN)
     *                       = number of different words that preceed (w2,...,wN) in
     *                         training data.
     *
     */
    VocabIndex ngram[maxNumParentsPerChild + 2];

    /*
     * clear all counts of given order 
     * Note: exclude N-grams starting with non-events (such as <s>) since
     * there usually are no words preceeding them.
     */
    {
        FNgramSpecs<FNgramCount>::FNgramSpec::PSIter iter(spec.parentSubsets[node], ngram);
	FNgramCount *count;

	while ((count = iter.next())) {
	    if (!vocab.Vocab::isNonEvent(ngram[0]) &&
		!(vocab.nullIsWord() && ngram[0] == vocab.nullIndex))
	    {
		*count = 0;
	    }
	}
    }

    /*
     * Now accumulate new counts, but first choose a parent to get counts
     * from.
     */


    // TODO: the selection process here chooses only one of many
    // possible choises to get the counts from. Ideally, KN-smoothing
    // should be re-thought for general BG models. For now, choose the
    // parent with the highest bit, so we'll at least get the same
    // results as with normal word models with temporally constrained
    // backoff.

    unsigned int max_parent_node=0;
    if ((int)spec.parentSubsets[node].knCountParent == ~0x0) {      
      FNgramSpecs<FNgramCount>::FNgramSpec::BGParentIter piter(spec.numParents,
							       node);
      unsigned int parent_node;
      while (piter.next(parent_node)) {
	// not interested in parents that do not exist
	if (spec.parentSubsets[parent_node].counts == NULL)
	  continue;
	if (parent_node > max_parent_node)
	  max_parent_node = parent_node;
      }
    } else {
      max_parent_node=spec.parentSubsets[node].knCountParent;
    }


    // fprintf(stderr,"in prepcounts: node = 0x%X, max_par = 0x%X\n",node,max_parent_node);
    // if (node == 0x1)
    // fprintf(stderr,"in prepcounts: node = 0x%X, max_par = 0x%X\n",node,max_parent_node);

    {
      FNgramSpecs<FNgramCount>::FNgramSpec::PSIter iter(spec.parentSubsets[max_parent_node],ngram);
      FNgramCount *count;

      while ((count = iter.next())) {
	if (*count > 0) {
	  FNgramCount *loCount = 
	    spec.parentSubsets[node].
	    findCountSubCtx(ngram,
			    bitGather(max_parent_node,node)<<1|0x1);


	  if (loCount) {
	    (*loCount) += 1;
	  }
	}
      }
    }

    // check to make sure that all guys have a non-zero count.
    // TODO: this is arguably a debugging option, and could be turned off.
    //       This check is useful, however, if the -no-virtual-begin-sentence
    //       option has been used.
    {
        FNgramSpecs<FNgramCount>::FNgramSpec::PSIter iter(spec.parentSubsets[node], ngram);
	FNgramCount *count;
	unsigned total=0;
	unsigned thoseWoCounts=0;
	while ((count = iter.next())) {
	  total++;
	  if (!vocab.Vocab::isNonEvent(ngram[0]) &&
	      !(vocab.nullIsWord() && ngram[0] == vocab.nullIndex))
	  {
	    if (*count == 0) {
	      thoseWoCounts++;
	      if (FDiscount::debug(DEBUG_PRINT_CONTEXTS_WO_COUNTS)) {
		fprintf(stderr, "Node 0x%X, Parent 0x%X: A context does not have a count\n",node,max_parent_node);
		cerr << "Context: " << ngram << endl;
	      }
	    }
	  }
	}
	if (FDiscount::debug(DEBUG_PRINT_WARNING_CONTEXTS_WO_COUNTS) && thoseWoCounts > 0) {	
	  fprintf(stderr,"Node 0x%X, Parent 0x%X: Found %d/%d contexts with zero counts\n",node,max_parent_node,thoseWoCounts,total);
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
Boolean
FModKneserNey::estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab)
{
    if (!FKneserNey::prepareCountsAtEnd) {
	FKneserNey::prepareCounts(spec, node, vocab);
    }

    /*
     * First tabulate count-of-counts
     */
    Count n1 = 0;
    Count n2 = 0;
    Count n3 = 0;
    Count n4 = 0;

    FNgramSpecs<FNgramCount>::FNgramSpec::ParentSubset& counts =
						spec.parentSubsets[node];

    VocabIndex wids[maxNumParentsPerChild + 1];
    FNgramSpecs<FNgramCount>::FNgramSpec::PSIter iter(counts, wids);
    NgramCount *count;

    while ((count = iter.next())) {
	if (vocab.isNonEvent(wids[counts.order - 1])) {
	    continue;
	} else if (vocab.isMetaTag(wids[counts.order - 1])) {
	    unsigned type = vocab.typeOfMetaTag(wids[counts.order - 1]);

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
	    
    if (FKneserNey::FDiscount::debug(DEBUG_ESTIMATE_DISCOUNT)) {
      FKneserNey::FDiscount::dout()
	       << "Mod Kneser-Ney smoothing " << HEX << node << DEC << "-grams\n"
	       << "n1 = " << n1 << endl
	       << "n2 = " << n2 << endl
	       << "n3 = " << n3 << endl
	       << "n4 = " << n4 << endl;
    }

    if (n1 == 0 || n2 == 0 || n3 == 0 || n4 ==0) {
	cerr << "warning: one of required modified KneserNey count-of-counts is zero\n";
	return false;
    }

    /*
     * Compute discount constants (Ries 1997, Chen & Goodman 1998)
     */
    double Y = (double)n1/(n1 + 2 * n2);

    // use ModKneserNey:discount1 since we use it in ModKneserNey::discount()
    ModKneserNey::discount1 = 1 - 2 * Y * n2 / n1;
    discount2 = 2 - 3 * Y * n3 / n2;
    discount3plus = 3 - 4 * Y * n4 / n3;

    if (FKneserNey::FDiscount::debug(DEBUG_ESTIMATE_DISCOUNT)) {
	FKneserNey::FDiscount::dout()
	       << "D1 = " << ModKneserNey::discount1 << endl
	       << "D2 = " << discount2 << endl
	       << "D3+ = " << discount3plus << endl;
    }
    if (FKneserNey::prepareCountsAtEnd) {
	FKneserNey::prepareCounts(spec, node, vocab);
    }
    return true;
}

