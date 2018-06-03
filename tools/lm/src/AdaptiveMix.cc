/*
 * AdaptiveMix.cc --
 *	Adaptive Bayesian mixture language model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1998-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/AdaptiveMix.cc,v 1.20 2016/04/09 06:53:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "AdaptiveMix.h"
#include "Ngram.h"
#include "Array.cc"

/*
 * Debug levels used
 */
#define DEBUG_MIX_WEIGHTS	2
#define DEBUG_ESTIMATE		3

AdaptiveMix::AdaptiveMix(Vocab &vocab, double decay, double llscale,
							unsigned maxIters)
    : LM(vocab), decay(decay), llscale(llscale), maxIters(maxIters),
						numComps(0), accumulating(true)
{
}

/*
 * read mixture model
 *	file format is:
 *		mixture-weight-1	ngram-file-1	ngram-order-1
 *		mixture-weight-2	ngram-file-2	ngram-order-2
 *		...
 */
Boolean
AdaptiveMix::read(File &file, Boolean limitVocab)
{
    /*
     * dispose of old component models
     */
    for (unsigned i = 0; i < numComps; i ++) {
	delete compLMs[i];
    }
    numComps = 0;
	
    char *line;
    while ((line = file.getline())) {
	double weight;
	char filename[100];
	unsigned order = defaultNgramOrder;
	
	if (sscanf(line, "%lf %99s %u", &weight, filename, &order) < 2) {
		file.position() << "ill-formed input\n";
		return false;
	}

	priors[numComps] = weight;

	compLMs[numComps] = new Ngram(vocab, order);
	assert(compLMs[numComps] != 0);
	compLMs[numComps]->debugme(debuglevel());

	Boolean ok;
	if (strcmp(filename, "-") == 0) {
	    ok = compLMs[numComps]->read(file, limitVocab);
	} else {
	    File lmFile(filename, "r");

	    ok = compLMs[numComps]->read(lmFile, limitVocab);
	}

	if (!ok) {
	    file.position() << "error reading ngram\n";
	    numComps ++;
	    return false;
	}

	numComps ++;
    }

    if (numComps == 0) {
	file.position() << "no mixture components found\n";
	return false;
    }

    initPosteriors();
    return true;
}

Boolean
AdaptiveMix::write(File &file)
{
    for (unsigned i = 0; i < numComps; i ++) {
	file.fprintf("%.*lf -\n", Prob_Precision, (double)priors[i]);

	compLMs[i]->write(file);
	file.fprintf("\n");
    }

    return true;
}

void
AdaptiveMix::initPosteriors()
{
    endOfHistory = endOfSentence = 0;

    for (unsigned i = 0; i < numComps; i ++) {
	posteriors[i] = priors[i];
    }
}

void
AdaptiveMix::computePosteriors()
{
    const double MINDECAY = 0.001;
    const double MIN_LL_CHANGE = 0.01;

    /*
     * Compute posteriors
     */

    LogP oldLikelihood, newLikelihood;

    for (unsigned iter = 0; iter < maxIters; iter ++) {
	Array<Prob> counts;
	Array<Prob> tmpCounts;

	for (unsigned i = 0; i < numComps; i++) {
	    counts[i] = 0.0;
	}

	double numSamples = 0;
	LogP newLikelihood = 0.0;

	double totalDecay = 1.0;

	for (unsigned h = endOfSentence; h > 0; )  {

	    h--;

	    Prob totalCount = 0.0;

	    for (unsigned i = 0; i < numComps; i ++) {
		tmpCounts[i] = posteriors[i] * histProbs[i][h];
		totalCount += tmpCounts[i];
	    }

	    if (totalCount != 0.0) {
		for (unsigned i = 0; i < numComps; i ++) {
		    counts[i] += totalDecay * tmpCounts[i] / totalCount;
		}

		numSamples += totalDecay;
		newLikelihood += totalDecay * ProbToLogP(totalCount);
	    }

	    totalDecay *= decay;

	    if (totalDecay < MINDECAY) {
		break;
	    }
	}

	/*
	 * Normalize
	 */
	if (numSamples > 0.0) {
	    for (unsigned i = 0; i < numComps; i ++) {
		posteriors[i] = counts[i] / numSamples;
	    }
	}

	if (debug(DEBUG_ESTIMATE)) {
	    dout() << "[iter" << iter
		   << "_ll=" << newLikelihood << "]";
	}

	if (iter > 0 &&
	    fabs((newLikelihood - oldLikelihood)/oldLikelihood) < MIN_LL_CHANGE)
	{
	    break;
	}

	oldLikelihood = newLikelihood;
    } 
}

LogP
AdaptiveMix::wordProb(VocabIndex word, const VocabIndex *context)
{
    unsigned i;

    if (running() && debug(DEBUG_MIX_WEIGHTS)) {
	dout() << "[post=" << posteriors[0];
	for (i = 1; i < numComps; i ++) {
	    dout() << "," << posteriors[i];
	}
	dout() << "]";
    }

    Prob mixProb = 0.0;

    for (i = 0; i < numComps; i ++) {
	LogP lp = compLMs[i]->wordProb(word, context);

	mixProb += posteriors[i] * LogPtoProb(lp);

	/* 
	 * record component probabilities for posterity
	 */
	if (running()) {
	    histProbs[i][endOfSentence] =
		lp == LogP_Zero ? 0.0 : LogPtoProb(llscale * lp);
	}
    }

    /*
     * When in accumulating mode, add the sentence likelihood to
     * the history at the end of sentence
     */
    if (running()) {
	endOfSentence ++;

	if (word == vocab.seIndex()) {
	    if (accumulating) {
		endOfHistory = endOfSentence;
	    } else {
		endOfSentence = endOfHistory;
	    }
	}
	computePosteriors();
    }

    return ProbToLogP(mixProb);
}

void *
AdaptiveMix::contextID(VocabIndex word, const VocabIndex *context,
							    unsigned &length)
{
    unsigned maxLen = 0;
    void *maxCid = 0;

    /*
     * Return the context ID of the component model that uses the longest
     * context.  We must use longest context regardless of predicted word 
     * because mixture models don't support contextBOW().
     */
    for (unsigned i = 0; i < numComps; i ++) {
	unsigned clen;
	void *cid = compLMs[i]->contextID(context, clen);

	if (i == 0 || clen > maxLen) {
	    maxLen = clen;
	    maxCid = cid;
	}
    }

    length = maxLen;
    return maxCid;
}

/*
 * Command strings used for <LMstate>
 */
#define STATE_RESET		"RESET"
#define STATE_ACCUMULATE	"ACCUMULATE"
#define STATE_NOACCUMULATE	"NOACCUMULATE"
#define STATE_ADD		"ADD"

void
AdaptiveMix::setState(const char *state)
{
    /*
     * First see if the state change is for us
     */
#define LS(x) (sizeof(x)-1)		// static string length
    if (strncmp(state, STATE_RESET, LS(STATE_RESET)) == 0) {
	/*
	 * Initialize likelihoods and posteriors
	 */
	initPosteriors();
    } else if (strncmp(state, STATE_ACCUMULATE, LS(STATE_ACCUMULATE)) == 0) {
	/*
	 * Start accumulating running likelihoods
	 */
	accumulating = true;
    } else if (strncmp(state, STATE_NOACCUMULATE, LS(STATE_NOACCUMULATE)) == 0){
	/*
	 * Stop accumulating running likelihoods
	 */
	accumulating = false;
    } else if (strncmp(state, STATE_ADD, LS(STATE_ADD)) == 0) {
	/*
	 * Add likelihood of sentence
	 */
	char *sentString = strdup(state + LS(STATE_ADD));
	assert(sentString != 0);

	VocabString sentence[maxWordsPerLine + 1];
	unsigned numWords = vocab.parseWords(sentString, sentence,
							maxWordsPerLine + 1);
       if (numWords == maxWordsPerLine + 1) {
            cerr << "AdaptiveMix::setState: too many words per sentence\n";
        } else {
	    unsigned dlevel = debuglevel();
	    debugme(0);

	    Boolean wasAccumulating = accumulating;
	    accumulating = true;

	    TextStats sentenceStats;
	    (void)sentenceProb(sentence, sentenceStats);

	    debugme(dlevel);
	    accumulating = wasAccumulating;
	}
	free(sentString);
    } else {
	/*
	 * Other state changes are propagated to the component models
	 */
	for (unsigned i = 0; i < numComps; i ++) {
	    compLMs[i]->setState(state);
	}
    }
}

