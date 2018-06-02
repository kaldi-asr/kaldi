/*
 * VarNgram.cc --
 *	Variable length N-gram backoff language model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/VarNgram.cc,v 1.20 2014-08-29 21:35:48 frandsen Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <math.h>

#include "VarNgram.h"

#include "Array.cc"

#define DEBUG_ESTIMATE_WARNINGS	1
#define DEBUG_PRUNE_HOEFFDING 2

VarNgram::VarNgram(Vocab &vocab, unsigned neworder, double alpha)
    : Ngram(vocab, neworder), pruneAlpha(alpha)
{
}

/*
 * This is mostly a copy of Ngram::estimate(), except that 
 * pruneNgram() is consulted to determine when to exclude an ngram from
 * the model.
 */
Boolean
VarNgram::estimate(NgramStats &stats, Discount **discounts)
{
    /*
     * For all ngrams, compute probabilities and apply the discount
     * coefficients.
     */
    makeArray(VocabIndex, context, order);
    unsigned vocabSize = Ngram::vocabSize();

    /*
     * Remove all old contexts ...
     */
    clear();

    /*
     * ... but save time by allocating unigram probabilities for all words in
     * the vocabulary.
     */
    {
	VocabIndex emptyContext = Vocab_None;
	contexts.find(&emptyContext)->probs.setsize(vocab.numWords());
    }

    /*
     * Ensure <s> unigram exists (being a non-event, it is not inserted
     * in distributeProb(), yet is assumed by much other software).
     */
    if (vocab.ssIndex() != Vocab_None) {
	context[0] = Vocab_None;

	*insertProb(vocab.ssIndex(), context) = LogP_Zero;
    }

    for (unsigned i = 1; i <= order; i++) {
	NgramCount *contextCount;
	NgramsIter contextIter(stats, context, i-1);

	/*
	 * This enumerates all contexts, i.e., i-1 grams.
	 */
	while ((contextCount = contextIter.next())) {
	    /*
	     * Skip contexts ending in </s>.  This typically only occurs
	     * with the doubling of </s> to generate trigrams from
	     * bigrams ending in </s>.
	     * If <unk> is not real word, also skip context that contain
	     * it.
	     */
	    if ((i > 1 && context[i-2] == vocab.seIndex()) ||
	        (vocab.isNonEvent(vocab.unkIndex()) &&
				 vocab.contains(context, vocab.unkIndex())))
	    {
		continue;
	    }

	    VocabIndex word[2];	/* the follow word */
	    NgramsIter followIter(stats, context, word, 1);
	    NgramCount *ngramCount;

	    /*
	     * Total up the counts for the denominator
	     * (the lower-order counts may not be consistent with
	     * the higher-order ones, so we can't just use *contextCount)
	     * Only if the trustTotal flag is set do we override this
	     * with the count from the context ngram.
	     */
	    NgramCount totalCount = 0;
	    Count observedVocab = 0;
	    while ((ngramCount = followIter.next())) {
		if (vocab.isNonEvent(word[0])) {
		    continue;
		}
		totalCount += *ngramCount;
		observedVocab ++;
	    }
	    if (i > 1 && trustTotals()) {
		totalCount = *contextCount;
	    }

	    if (totalCount == 0) {
		continue;
	    }

	    /*
	     * reverse the context ngram since that's how
	     * the BO nodes are indexed.
	     */
	    Vocab::reverse(context);

	    /*
	     * Compute the discounted probabilities
	     * from the counts and store them in the backoff model.
	     */
	retry:
	    followIter.init();
	    Prob totalProb = 0.0;

	    /*
	     * check if discounting is disabled for this round
	     */
	    Boolean noDiscount =
			    (discounts == 0) ||
			    (discounts[i-1] == 0) ||
			    discounts[i-1]->nodiscount();

	    // Move outside of loop and set to something to
	    // avoid potential use of uninitialized value.
	    LogP lprob = LogP_Zero;
	    while ((ngramCount = followIter.next())) {
		Prob prob;
		double discount;

		if (vocab.isNonEvent(word[0])) {
		    /*
		     * Discard all pseudo-word probabilities,
		     * except for unigrams.  For unigrams, assign
		     * probability zero.  This will leave them with
		     * prob zero in all case, due to the backoff
		     * algorithm.
		     * Also discard the <unk> token entirely in closed
		     * vocab models, its presence would prevent OOV
		     * detection when the model is read back in.
		     */
		    if (i > 1 || word[0] == vocab.unkIndex()) {
			continue;
		    }

		    lprob = LogP_Zero;
		    discount = 1.0;
		} else if (pruneNgram(stats, word[0], *ngramCount,
						context, totalCount))
		{
		    /*
		     * Right now, pruning just means replacing an
		     * ngram with its backed-off estimate.
		     */
		    discount = 0.0;
		} else {
		    /*
		     * Ths discount array passed may contain 0 elements
		     * to indicate no discounting at this order.
		     */
		    if (noDiscount) {
			discount = 1.0;
		    } else {
			discount =
			    discounts[i-1]->discount(*ngramCount, totalCount,
								observedVocab);
		    }
		    Prob prob = (discount * *ngramCount) / totalCount;
		    lprob = ProbToLogP(prob);
		    totalProb += prob;
		}
		    
		/*
		 * A discount coefficient of zero indicates this ngram
		 * should be omitted entirely (presumably to save space).
		 */
		if (discount != 0.0) {
		    *insertProb(word[0], context) = lprob;
		} 
	    }

	    /*
	     * This is a hack credited to Doug Paul (by Roni Rosenfeld in
	     * his CMU tools).  It may happen that no probability mass
	     * is left after totalling all the explicit probs, typically
	     * because the discount coefficients were out of range and
	     * forced to 1.0.  Unless we have seen all vocabulary words in
	     * this context, to arrive at some non-zero backoff mass,
	     * we try incrementing the denominator in the estimator by 1.
	     * Another hack: If the discounting method uses interpolation 
	     * we first try disabling that because interpolation removes
	     * probability mass.
	     */
	    if (!noDiscount && totalCount > 0 &&
		observedVocab < vocabSize &&
		totalProb > 1.0 - Prob_Epsilon)
	    {
		totalCount += 1;

		if (debug(DEBUG_ESTIMATE_WARNINGS)) {
		    cerr << "warning: no backoff probability mass left for \""
			 << (vocab.use(), context)
			 << "\" -- incrementing denominator"
			 << endl;
		}
		goto retry;
	    }

	    /*
	     * Undo the reversal above so the iterator can continue correctly
	     */
	    Vocab::reverse(context);
	}
    }

    /*
     * With all the probs in place, BOWs are obtained simply by the usual
     * normalization.
     */
    recomputeBOWs();

    return true;
}

/*
 * Decide if a the ngram consisting of the context (in reverse order)
 * and the word should be omitted from the ngram model
 *
 * The criterion used here is that the difference in relative frequency
 * between the higher and the lower-order estimate doesn't exceed a
 * Hoeffding bound
 *
 *	| f1/n1 - f2/n2 | < sqrt( 1/2 log(2/alpha) ) (1/sqrt(n1) + 1/sqrt(n2))
 *
 * which is true with probability > 1-alpha
 * (alpha is a confidence level parameter).
 *
 * Here, alpha is set by the user, and we check that that the difference
 *
 *	delta = | f1/n1 - f2/n2 |
 *
 * satisfies
 *
 *      2 exp (- 2 sqr(delta / (1/sqrt(n1) + 1/sqrt(n2)))) < alpha
 */
Boolean
VarNgram::pruneNgram(NgramStats &stats,
			VocabIndex word, NgramCount ngramCount,
			const VocabIndex *context, NgramCount contextCount)
{
    if (context[0] == Vocab_None) {
	return false;
    } else {
	/*
	 * Reverse the context to make count lookups easy
	 */
	Vocab::reverse((VocabIndex *)context);

	NgramCount f1 = ngramCount;
	NgramCount n1 = contextCount;
	NgramCount *f2 = stats.findCount(context + 1, word);
	NgramCount *n2 = stats.findCount(context + 1);

	Boolean prune;

        if (n1 == 0 || f2 == 0 || n2 == 0 || *n2 == 0) {
	    prune = false;
	} else {
	    double delta = fabs(f1/(double)n1 - (double)(*f2)/(double)*n2);
	    double dev = 1.0 / (1.0/sqrt((double)n1) + 1.0/sqrt((double)*n2));

	    double val = 2.0 * exp(-2.0 * (delta * delta) * (dev * dev));
    
	    prune = (val > pruneAlpha);

	    if (prune && debug(DEBUG_PRUNE_HOEFFDING)) {
		dout() << "pruning ngram \"" 
		       << (vocab.use(), context)
		       << " " << vocab.getWord(word)
		       << "\" (f1 = " << f1 << "/" << n1
		       << " f2 = " << *f2 << "/" << *n2
		       << "; d = " << val << ")\n" ;
	    }
	}

	Vocab::reverse((VocabIndex *)context);

	return prune;
    }
}
