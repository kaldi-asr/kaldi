/*
 * TaggedNgram.cc --
 *	Tagged N-gram backoff language models
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/TaggedNgram.cc,v 1.7 2010/06/02 05:49:58 stolcke Exp $";
#endif

#include "TaggedNgram.h"

#include "Array.cc"

/*
 * Debug levels used
 */
#define DEBUG_NGRAM_HITS 2		/* from Ngram.cc */

TaggedNgram::TaggedNgram(TaggedVocab &vocab, unsigned neworder)
    : Ngram(vocab, neworder), vocab(vocab)
{
}

/*
 * The tagged ngram LM uses the following backoff hierarchy:
 * 
 *	- try word n-gram
 *	- try n-grams obtained by replacing the most distant word with its tag
 *	- try (n-1)-grams (recursively)
 */
LogP
TaggedNgram::wordProbBO(VocabIndex word, const VocabIndex *context, unsigned int clen)
{
    LogP result;
    VocabIndex usedContext[maxNgramOrder];
    VocabIndex untaggedWord = vocab.unTag(word);

    /*
     * Extract the word ngram from the context
     */
    unsigned i;
    for (i = 0; i < clen; i++) {
	usedContext[i] = vocab.unTag(context[i]);
    }
    usedContext[i] = Vocab_None;

    LogP *prob = findProb(untaggedWord, usedContext);

    if (prob) {
	if (running() && debug(DEBUG_NGRAM_HITS)) {
	    dout() << "[" << (clen + 1) << "gram]";
	}
	result = *prob;
    } else if (clen > 0) {
	/*
	 * Backoff weight from word to tag-ngram
	 */
	LogP *bow = findBOW(usedContext);
        LogP totalBOW = bow ? *bow : 0.0;

	/*
	 * Now replace the last word with its tag
	 */
	usedContext[clen - 1] = vocab.tagWord(Tagged_None,
				    vocab.getTag(context[clen - 1]));

	prob = findProb(untaggedWord, usedContext);

	if (prob) {
	    if (running() && debug(DEBUG_NGRAM_HITS)) {
		dout() << "[" << clen << "+Tgram]";
	    }
	    result = totalBOW + *prob;
	} else {
	    /*
	     * No tag-ngram, so back off to shorter context.
	     */
	    bow = findBOW(usedContext);

	    totalBOW += bow ? *bow : 0.0;
	    result = totalBOW + wordProbBO(word, context, clen - 1);
	}
    } else {
	result = LogP_Zero;
    }

    return result;
}

/*
 * Renormalize language model by recomputing backoff weights.
 *
 * The BOW(c) for a context c is computed to be
 *
 *	BOW(c) = (1 - Sum p(x | c)) /  (1 - Sum p_BO(x | c))
 *
 * where Sum is a summation over all words x with explicit probabilities
 * in context c, p(x|c) is that probability, and p_BO(x|c) is the 
 * probability for that word according to the backoff algorithm.
 */
void TaggedNgram::recomputeBOWs()
{
    makeArray(VocabIndex, context, order + 1);

    /*
     * Here it is important that we compute the backoff weights in
     * increasing order, since the higher-order ones refer to the
     * lower-order ones in the backoff algorithm.
     * Note that this will only generate backoff nodes for those
     * contexts that have words with explicit probabilities.  But
     * this is precisely as it should be.
     */
    unsigned i;
    for (i = 0; i < order; i++) {
	BOnode *node;
	NgramBOsIter iter1(*this, context, i);
	
	while ((node = iter1.next())) {
	    NgramProbsIter piter(*node);
	    VocabIndex word;
	    LogP *prob;

	    double numerator = 1.0;
	    double denominator = 1.0;

	    while ((prob = piter.next(word))) {
		numerator -= LogPtoProb(*prob);
		if (i > 0) {
		    denominator -=
			LogPtoProb(wordProbBO(word, context, i - 1));
		}
	    }

	    /*
	     * Avoid some predicatble anomalies due to rounding errors
	     */
	    if (numerator < 0.0 && numerator > -Prob_Epsilon) {
		numerator = 0.0;
	    }
	    if (denominator < 0.0 && denominator > -Prob_Epsilon) {
		denominator = 0.0;
	    }

	    if (numerator < 0.0) {
		cerr << "BOW numerator for context \""
		     << (vocab.use(), context)
		     << "\" is " << numerator << " < 0\n";
	    } else if (denominator <= 0.0) {
		if (numerator > Prob_Epsilon) {
		    cerr << "BOW denominator for context \""
			 << (vocab.use(), context)
			 << "\" is " << denominator << " <= 0"
			 << endl;
		} else {
		    node->bow = 0.0;
		}
	    } else {
		/*
		 * If unigram probs leave a non-zero probability mass
		 * then we should give that mass to the zero-order (uniform)
		 * distribution for zeroton words.  However, the ARPA
		 * BO format doesn't support a "null context" BOW.
		 * We simluate the intended distribution by spreading the
		 * left-over mass uniformly over all vocabulary items that
		 * have a zero probability.
		 * NOTE: We used to do this only if there was prob mass left,
		 * but some ngram software requires all words to appear as
		 * unigrams, which we achieve by giving them zero probability.
		 */
		if (i == 0 /*&& numerator > 0.0*/) {
		    distributeProb(numerator, context);
		} else {
		    node->bow = ProbToLogP(numerator) - ProbToLogP(denominator);
		}
	    }
	}
    }
}
