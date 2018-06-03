/*
 * SkipNgram.cc --
 *	N-gram backoff language model with context skips
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International, 2013-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/SkipNgram.cc,v 1.19 2016/04/09 06:53:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <assert.h>

#include "SkipNgram.h"

#include "Array.cc"

#define DEBUG_ESTIMATE_WARNINGS	1	/* from Ngram.cc */
#define DEBUG_PRINT_WORD_PROBS  2       /* from LM.cc */
#define DEBUG_PRINT_LIKELIHOODS	0

SkipNgram::SkipNgram(Vocab &vocab, unsigned order)
    : Ngram(vocab, order),
      maxEMiters(100), minEMdelta(0.0001),
      initialSkipProb(0.5), skipProbs(vocab.numWords())
{
}

void
SkipNgram::memStats(MemStats &stats)
{
    Ngram::memStats(stats);
    skipProbs.memStats(stats);
}

/*
 * LM interface
 */

LogP
SkipNgram::wordProb(VocabIndex word, const VocabIndex *context)
{
    unsigned int clen = Vocab::length(context);

    if (skipOOVs()) {
	/*
	 * Backward compatibility with the old broken perplexity code:
	 * return prob 0 if any of the context-words have an unknown
	 * word.
	 */
	if (word == vocab.unkIndex() ||
	    (order > 1 && context[0] == vocab.unkIndex()) ||
	    (order > 2 && context[2] == vocab.unkIndex()))
	{
	    return LogP_Zero;
	}
    }

    /*
     * If the context is empty we use only the straight Ngram prob.
     * Otherwise, interpolate the regular Ngram prob with the one
     * resulting from skipping the last word.
     */
    if (clen == 0) {
	return Ngram::wordProbBO(word, context, 0);
    } else {
	Prob *skipProb = skipProbs.find(context[0]);

	/*
	 * Avoid useless lookup if skipProb is zero.
	 */
	if (skipProb == 0 || *skipProb == 0.0) {
		return Ngram::wordProbBO(word, context,
				(clen > order - 1) ? order - 1 : clen);
	} else {
	    return MixLogP(Ngram::wordProbBO(word, &context[1],
				(clen - 1 > order - 1) ? order - 1 : clen - 1),
			   Ngram::wordProbBO(word, context,
				(clen > order - 1) ? order - 1 : clen),
			   *skipProb);
	}
    }
}

Boolean
SkipNgram::read(File &file, Boolean limitVocab)
{
    /*
     * First read the ngram data in standard format
     */
    if (!Ngram::read(file, limitVocab)) {
	return false;
    }
	
    /*
     * Now parse the skipProbs as a two-column list
     */

    char *line;

    while ((line = file.getline())) {
	
	VocabString words[3];
	VocabIndex wid;

	/*
	 * Parse a line of the form
	 *	<w>	<prob>
	 */
	if (Vocab::parseWords(line, words, 3) != 2) {
	    file.position() << "illegal skip prob line\n";
	    return false;
	}

	wid = vocab.addWord(words[0]);

	Prob prob;
	if (!parseProb(words[1], prob)) {
	    file.position() << "bad skip prob value " << words[1] << endl;
	    return false;
	}

	*skipProbs.insert(wid) = prob;
    }

    return true;
}

Boolean
SkipNgram::write(File &file)
{
    /*
     * First write out the Ngram parameters in the usual format
     */
    if (!Ngram::write(file)) {
	return false;
    }
    
    file.fprintf("\n");

    LHashIter<VocabIndex, Prob> skipProbsIter(skipProbs, vocab.compareIndex());

    VocabIndex wid;
    Prob *prob;

    while ((prob = skipProbsIter.next(wid))) {
	file.fprintf("%s %.*lg\n", vocab.getWord(wid),
				   Prob_Precision, (double)*prob);
    }

    file.fprintf("\n");

    return true;
}

/*
 * Estimation
 */

Boolean
SkipNgram::estimate(NgramStats &stats, Discount **discounts)
{
    /*
     * Initialize undefined skipProbs
     */
    VocabIter vocabIter(vocab);
    VocabIndex wid;

    while (vocabIter.next(wid)) {
	Boolean foundP;
	Prob *skipProb = skipProbs.insert(wid, foundP);

	if (!foundP) {
	    *skipProb = (wid == vocab.ssIndex()) ? 0.0 : initialSkipProb;
	}
    }

    LogP like;

    for (unsigned i = 0; i < maxEMiters; i ++) {
	NgramCounts<FloatCount> ngramExps(vocab, order);
	LHash<VocabIndex, double> skipExps(vocab.numWords());

	LogP newLike = estimateEstep(stats, ngramExps, skipExps);

	//cerr << "ngram stats:\n";
	//ngramExps.write((File)stderr);

	if (debug(DEBUG_PRINT_LIKELIHOODS)) {
	   dout() << "iteration " << i << ": "
		  << "log likelihood = " << newLike
		  << endl;
	}

	if (i > 1 && fabs((newLike - like)/like) < minEMdelta) {
	    break;
	}

	if (!estimateMstep(stats, ngramExps, skipExps, discounts)) {
	    return false;
	}
	like = newLike;
    }

    return true;
}

/*
 * E-step (expected count computation)
 *	returns the log likelihood of the training counts
 */
LogP
SkipNgram::estimateEstepNgram(VocabIndex *ngram, NgramCount ngramCount,
			      NgramStats &stats,
			      NgramCounts<FloatCount> &ngramExps,
			      LHash<VocabIndex, double> &skipExps)
{
    unsigned ngramLength = Vocab::length(ngram);
    assert(ngramLength > 1);

    VocabIndex word = ngram[ngramLength - 1];
    VocabIndex skipped = ngram[ngramLength - 2];

    //cerr << "doing ngram " << (vocab.use(), ngram) << endl;

    Prob *skipProb = skipProbs.find(skipped);
    Prob skipPr = skipProb ? *skipProb : 0.0;

    /*
     * temporarily reverse ngram for wordProb call
     */
    Vocab::reverse(ngram);

    LogP l1 = Ngram::wordProbBO(ngram[0], &ngram[2], ngramLength - 2);
    LogP l2 = Ngram::wordProbBO(ngram[0], &ngram[1], ngramLength - 2);

    /*
     * If both likelihoods are zero the model is not yet initialized,
     * so we provide some dummy values
     */
    if (l1 == LogP_Zero && l2 == LogP_Zero) {
	l1 = l2 = -10.0;
    }

    Prob p1 = skipPr * LogPtoProb(l1);
    Prob p2 = (1.0 - skipPr) * LogPtoProb(l2);

    LogP logSum = ProbToLogP(p1 + p2);

    Vocab::reverse(ngram);

    //cerr << "l1 = " << l1 << " l2 = " << l2 << endl;
    //cerr << "p1 = " << p1 << " p2 = " << p2 << endl;

    /*
     * Increment expected count for ngrams with skip
     * NOTE: Do not add ngrams that weren't observed in the data.
     */
    ngram[ngramLength - 2] = Vocab_None;
    unsigned i;

    for (i = (ngramLength - 1 > order) ? ngramLength - 1 - order : 0;
	 i < ngramLength - 1;
	 i ++)
    {
	if (stats.findCount(&ngram[i], word)) {
	    //cerr << " incrementing " << (vocab.use(), &ngram[i])
	    //     << " " << vocab.getWord(word) << endl;
	    *ngramExps.insertCount(&ngram[i], word) += skipPr * ngramCount;
	}
    }
    ngram[ngramLength - 2] = skipped;

    /*
     * Increment expected count for ngrams without skip
     */
    for (i = (ngramLength > order) ? ngramLength - order : 0;
	 i < ngramLength;
	 i ++)
    {
	//cerr << " incrementing " << (vocab.use(), &ngram[i]) << endl;
	*ngramExps.insertCount(&ngram[i]) += (1.0 - skipPr) * ngramCount;
    }

    /*
     * Increment expected skip count
     */
    *skipExps.insert(skipped) += p1 / (p1 + p2);

    return ngramCount * logSum;
}

LogP
SkipNgram::estimateEstep(NgramStats &stats,
			 NgramCounts<FloatCount> &ngramExps,
			 LHash<VocabIndex, double> &skipExps)
{
    LogP totalLikelihood = 0.0;
    makeArray(VocabIndex, ngram, order + 2);
    makeArray(VocabIndex, context, order);

    /*
     * Enumerate all n+1 grams
     */
    NgramsIter ngramIter(stats, ngram, order + 1);
    NgramCount *ngramCount;

    while ((ngramCount = ngramIter.next())) {
	totalLikelihood += estimateEstepNgram(ngram, *ngramCount,
						stats, ngramExps, skipExps);
    }

    /*
     * Enumerate the 2...n grams starting with <s>
     * (they were omitted in the iteration above)
     */
    VocabIndex start[2];
    start[0] = vocab.ssIndex();
    start[1] = Vocab_None;
    ngram[0] = vocab.ssIndex();

    for (unsigned j = order; j > 1; j --) {
	NgramsIter ngramIter(stats, start, &ngram[1], j - 1);

	while ((ngramCount = ngramIter.next())) {
	    totalLikelihood += estimateEstepNgram(ngram, *ngramCount,
						   stats, ngramExps, skipExps);
	}
    }

    return totalLikelihood;
}

/*
 * M-step (likelihood maximization):
 *	This is virtually identical to Ngram::estimate(), except that
 *	the cound are floats.
 *	We also estimate the skip probabilities using ML.
 */
Boolean
SkipNgram::estimateMstep(NgramStats &stats,
		         NgramCounts<FloatCount> &ngramExps,
		         LHash<VocabIndex,double> &skipExps,
		         Discount **discounts)
{
    /*
     * First, estimate the skip probabilities using maximum likelihood
     */
    LHashIter<VocabIndex, double> skipExpsIter(skipExps);

    VocabIndex wid;
    double *skipCount;

    while ((skipCount = skipExpsIter.next(wid))) {
	NgramCount *total = stats.findCount(0, wid);
	assert(total != 0);

	//cerr << "skip(" << vocab.getWord(wid) << ") = "
	//	<< *skipCount << "/"
	//	<< *total << endl;
	*skipProbs.insert(wid) = *skipCount / *total;
    }

    //{
    //    File file("SKIP-NGRAM-EXPECTATIONS", "w");
    //    ngramExps.write(file, 0);
    //}

    /*
     * Reestimate discounting parameters from expected counts
     */
    for (unsigned i = 1; i <= order; i++) {
	if (!discounts[i-1]->estimate(ngramExps, i)) {
	    dout() << "warning: skip-ngram discount estimator for order "
		   << i << " failed -- using old parameters\n";
	}
    }

    /*
     * Reestimate probs from expected counts
     */
    return Ngram::estimate(ngramExps, discounts);
}

