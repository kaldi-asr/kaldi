/*
 * segment-nbest --
 *	Appply a hidden segment boundary model to a sequence of N-best lists,
 *	across utterance boundaries.
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International, 2015 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: segment-nbest.cc,v 1.31 2015-06-26 08:06:36 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

#include "option.h"
#include "version.h"
#include "File.h"

#include "Vocab.h"
#include "NBest.h"
#include "VocabMap.h"
#include "Ngram.h"
#include "DecipherNgram.h"
#include "BayesMix.h"
#include "Trellis.cc"
#include "Array.cc"
#include "MStringTokUtil.h"

#define DEBUG_PROGRESS		1
#define DEBUG_TRANSITIONS	2

static int version = 0;
static unsigned order = 3;
static unsigned debug = 0;
static char *lmFile = 0;
static const char *sTag = 0;
static const char *startTag = 0;
static const char *endTag = 0;
static double bias = 1.0;
static int toLower = 0;

static char *nbestFiles = 0;
static unsigned maxNbest = 0;
static unsigned maxRescore = 0;
static char *decipherLM = 0;
static double decipherLMW = 8.0;
static double decipherWTW = 0.0;
static double rescoreLMW = 8.0;
static double rescoreWTW = 0.0;
static int noReorder = 0;
static int fbRescore = 0;
static char *writeNbestDir = 0;
static char *noiseTag = 0;
static char *noiseVocabFile = 0;

static char *mixFile  = 0;
static int bayesLength = -1;
static double bayesScale = 1.0;
static double mixLambda = 0.5;

const LogP LogP_PseudoZero = -100;

static unsigned contextLength = 0;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_STRING, "lm", &lmFile, "hidden token sequence model" },
    { OPT_UINT, "order", &order, "ngram order to use for lm" },
    { OPT_UINT, "debug", &debug, "debugging level for lm" },
    { OPT_STRING, "stag", &sTag, "segment tag to use in output" },
    { OPT_STRING, "start-tag", &startTag, "tag to insert in front of N-best hyps" },
    { OPT_STRING, "end-tag", &endTag, "tag to insert at end of N-best hyps" },
    { OPT_FLOAT, "bias", &bias, "bias for segment model" },
    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },

    { OPT_STRING, "nbest-files", &nbestFiles, "list of n-best filenames" },
    { OPT_UINT, "max-nbest", &maxNbest, "maximum number of hyps to consider" },
    { OPT_UINT, "max-rescore", &maxRescore, "maximum number of hyps to rescore" },
    { OPT_TRUE, "no-reorder", &noReorder, "don't reorder N-best hyps before rescoring" },
    { OPT_STRING, "decipher-lm", &decipherLM, "DECIPHER(TM) LM for nbest list generation" },
    { OPT_FLOAT, "decipher-lmw", &decipherLMW, "DECIPHER(TM) LM weight" },
    { OPT_FLOAT, "decipher-wtw", &decipherWTW, "DECIPHER(TM) word transition weight" },
    { OPT_FLOAT, "rescore-lmw", &rescoreLMW, "rescoring LM weight" },
    { OPT_FLOAT, "rescore-wtw", &rescoreWTW, "rescoring word transition weight" },
    { OPT_STRING, "noise", &noiseTag, "noise tag to skip" },
    { OPT_STRING, "noise-vocab", &noiseVocabFile, "noise vocabulary to skip" },
    { OPT_TRUE, "fb-rescore", &fbRescore, "rescore N-best lists with forward-backward algorithm" },
    { OPT_STRING, "write-nbest-dir", &writeNbestDir, "output directory for rescored N-best lists" },
    { OPT_UINT, "bayes", &bayesLength, "context length for Bayes mixture LM" },
    { OPT_FLOAT, "bayes-scale", &bayesScale, "log likelihood scale for -bayes" },
    { OPT_STRING, "mix-lm", &mixFile, "LM to mix in" },
    { OPT_FLOAT, "lambda", &mixLambda, "mixture weight for -mix-lm" },
    { OPT_DOC, 0, 0, "plus any number of additional n-best list filenames" },
};

typedef enum {
	NOS, S, NOSTATE
} SegmentState;

const char *stateNames[] = {
	"NOS", "S"
};

/*
 * Define null key for SegmentState trellis
 */
inline void
Map_noKey(SegmentState &state)
{
    state = NOSTATE;
}

inline Boolean
Map_noKeyP(const SegmentState &state)
{
    return state == NOSTATE;
}

/*
 * Segment a sentence by finding the SegmentState sequence 
 * that has the highest probability
 */
void
segmentHyp(NBestHyp &hyp, const VocabIndex *leftContext, LM &lm,
			LogP &NOSscore, LogP &Sscore, SegmentState lastState)
{
    unsigned len = Vocab::length(hyp.words);
    LogP logBias = ProbToLogP(bias);

    if (len == 0) {
	NOSscore = Sscore = 0.0;
	if (lastState != NOSTATE) {
	    cout << endl;
	}
	return;
    }

    Trellis<SegmentState> trellis(len);

    Vocab &vocab = lm.vocab;

    VocabIndex sContext[2];
    sContext[0] = vocab.ssIndex();
    sContext[1] = Vocab_None;
    VocabIndex *noContext = &sContext[1];

    /*
     * Replace zero probs with a small non-zero prob, so the Viterbi
     * can still distinguish between paths of different probabilities.
     */
#define Z(p) (isZero ? LogP_PseudoZero : (p))

    /*
     * Prime the trellis as follows using the context passed to us.
     */
    {
	Boolean isZero = (lm.wordProb(hyp.words[0], leftContext) == LogP_Zero);

	if (bias > 0.0) {
	    LogP probS = lm.wordProb(vocab.seIndex(), leftContext) +
			 Z(lm.wordProb(hyp.words[0], sContext));
	    trellis.setProb(S, probS + logBias);
	}

	LogP probNOS = Z(lm.wordProb(hyp.words[0], leftContext));
        trellis.setProb(NOS, probNOS);

	if (debug >= DEBUG_TRANSITIONS) {
	    cerr << 0 << ": p(NOS) = " << trellis.getProb(NOS)
			<< ", P(S) = " << trellis.getProb(S) << endl;
	}
    }

    unsigned pos = 1;
    while (hyp.words[pos] != Vocab_None) {
	
	trellis.step();

	Boolean isZero = (lm.wordProb(hyp.words[pos], noContext) == LogP_Zero);

	/*
	 * Set up the contet for next word prob
	 */
	makeArray(VocabIndex, context, contextLength + 1);
	unsigned i;

	for (i = 0; i < contextLength && i < pos; i ++) {
	    context[i] = hyp.words[pos - i - 1];
	}
	for (unsigned k = 0; i < contextLength; i ++, k ++) {
	    context[i] = leftContext[k];
	}
	context[i] = Vocab_None;

	/*
	 * Iterate over all combinations of hidden tags for the previous and
	 * the current word
	 */
	trellis.update(NOS, NOS, Z(lm.wordProb(hyp.words[pos], context)));

	if (bias > 0.0) {
	    trellis.update(NOS, S, lm.wordProb(vocab.seIndex(), context) +
				   Z(lm.wordProb(hyp.words[pos], sContext)));

	    context[1] = vocab.ssIndex();
	    context[2] = Vocab_None;

	    trellis.update(S, NOS,
			Z(lm.wordProb(hyp.words[pos], context)) + logBias);
	    trellis.update(S, S, lm.wordProb(vocab.seIndex(), context) +
			Z(lm.wordProb(hyp.words[pos], sContext)) + logBias);
	}

	if (debug >= DEBUG_TRANSITIONS) {
	    cerr << pos << ": p(NOS) = " << trellis.getProb(NOS)
			<< ", P(S) = " << trellis.getProb(S) << endl;
	}

	pos ++;
    }

    NOSscore = trellis.getLogP(NOS);
    Sscore = trellis.getLogP(S);

    /*
     * If caller gave a final state to backtrace from, do so and print
     * the hyp with their segmentation.
     */
    if (lastState != NOSTATE) {
	makeArray(SegmentState, segs, len);

	if (trellis.viterbi(segs, len, lastState) != len) {
	    cerr << "trellis.viterbi failed\n";
	} else {
	    for (unsigned i = 0; i < len; i++) {
		if (segs[i] == S) {
		    cout << sTag << " ";
		}
		cout << vocab.getWord(hyp.words[i]);
		if (i != len - 1) {
		    cout << " ";
		}
	    }
	    cout << endl;
	}
    }
}

/*
 * Segment a list of nbest lists, i.e., find the sequences of
 * hyps that has the highest joint probability.
 * Algorithm:
 *	- Do forward dynamic programming on a trellis representing
 *	the possible hyp sequences (the `nbest trellis').  For each
 *	nbest segment, there are two states for each hyp.
 *	The first one represents the hyp ending in a NOS state, the
 *	second one represents the hyp ending in an S state (i.e.,
 *	the last word of the hyp is the first word of a linguistic segment).
 *	- To compute the local cost for each hyp, we do a forward
 *	dynamic programming on a second trellis (the `hyp trellis')
 *	representing the possible segmentations of that hyp.
 *	- Do a Viterbi traceback through the nbest trellis to find the
 *	best sequence of hyps overall.
 *	- For each hyp in that sequence, do a Viterbi traceback on the
 *	hyp trellis to recover the best segmentation for it.
 */

/*
 * The following macros construct and access aggregate states for the nbest
 * trellis, encoding a hyp number and the segmentation info for the last
 * word.
 */
#define NBEST_STATE(j,seg)	(((j) << 1) | (seg))
#define NBEST_SEG(state)	((SegmentState)((state) & 1))
#define NBEST_HYP(state)	((state) >> 1)

/*
 * Forward probability compution on nbest trellis
 */
Trellis<unsigned> *
forwardNbest(Array<NBestList *> &nbestLists, unsigned numLists,
	     LM &lm, double lmw, double wtw)
{
    Vocab &vocab = lm.vocab;

    assert(numLists > 0);

    Trellis<unsigned> *nbestTrellis = new Trellis<unsigned>(numLists + 1);
    assert(nbestTrellis != 0);

    if (debug >= DEBUG_PROGRESS) {
	cerr << "Forward pass...0";
    }

    /*
     * Initialize the trellis with the scores from the first nbest list
     * The first context is a sentence start.
     */
    {
	VocabIndex context[3];	/* word context from preceding nbest hyp */
	context[0] = vocab.ssIndex();
	context[1] = Vocab_None;

	for (unsigned j = 0;
	     j < nbestLists[0]->numHyps() &&
		 (maxRescore <= 0 || j < maxRescore);
	     j ++)
	{
	    NBestHyp &hyp = nbestLists[0]->getHyp(j);

	    LogP NOSscore, Sscore;
	    segmentHyp(hyp, context, lm, NOSscore, Sscore, NOSTATE);

	    nbestTrellis->setProb(NBEST_STATE(j, NOS),
		    hyp.acousticScore + lmw * NOSscore + wtw * hyp.numWords);
	    if (bias > 0.0) {
		nbestTrellis->setProb(NBEST_STATE(j, S),
			hyp.acousticScore + lmw * Sscore + wtw * hyp.numWords);
	    }
		

//cerr << "FORW:list " << 0 << " hyp " << j
//	<< " forw " << nbestTrellis->getLogP(NBEST_STATE(j, NOS))
//	<< " " << nbestTrellis->getLogP(NBEST_STATE(j, S))
//	<< endl;

	}
    }

    for (unsigned i = 1; i < numLists; i ++) {
	makeArray(VocabIndex, context, contextLength + 1);
				/* word context from preceding nbest hyp */

	if (debug >= DEBUG_PROGRESS) {
	    cerr << "..." << i;
	}

	nbestTrellis->step();

	for (unsigned j = 0;
	     j < nbestLists[i]->numHyps() &&
		(maxRescore <= 0 || j < maxRescore);
	     j ++)
	{
	    NBestHyp &hyp = nbestLists[i]->getHyp(j);

	    for (unsigned k = 0;
		 k < nbestLists[i-1]->numHyps() &&
		    (maxRescore <= 0 || k < maxRescore);
		 k ++)
	    {
		NBestHyp &prevHyp = nbestLists[i-1]->getHyp(k);
		unsigned prevLen = Vocab::length(prevHyp.words);

		LogP NOSscore, Sscore;

		/*
		 * Set up the last words from the previous hyp
		 * as context for this one.
		 */
		{
		    unsigned k;

		    for (k = 0; k < contextLength && k < prevLen; k ++) {
			context[k] = prevHyp.words[prevLen - 1 - k];
		    }
		    context[k] = Vocab_None;
		}

		segmentHyp(hyp, context, lm, NOSscore, Sscore, NOSTATE);

		nbestTrellis->update(NBEST_STATE(k, NOS), NBEST_STATE(j, NOS),
			hyp.acousticScore + lmw * NOSscore + wtw * hyp.numWords);
		if (bias > 0.0) {
		    nbestTrellis->update(NBEST_STATE(k, NOS), NBEST_STATE(j, S),
			hyp.acousticScore + lmw * Sscore + wtw * hyp.numWords);

		    /*
		     * Now the same for the case where the previous hyp's
		     * last word starts a sentence.
		     */
		    if (prevLen == 0) {
			context[0] = Vocab_None;
		    } else {
			context[0] = prevHyp.words[prevLen - 1];
			context[1] = vocab.ssIndex();
			context[2] = Vocab_None;
		    }
		    segmentHyp(hyp, context, lm, NOSscore, Sscore, NOSTATE);

		    nbestTrellis->update(NBEST_STATE(k, S), NBEST_STATE(j, NOS),
		       hyp.acousticScore + lmw * NOSscore + wtw * hyp.numWords);
		    nbestTrellis->update(NBEST_STATE(k, S), NBEST_STATE(j, S),
		       hyp.acousticScore + lmw * Sscore + wtw * hyp.numWords);
		}
	    }
//cerr << "FORW:list " << i << " hyp " << j
//	<< " forw " << nbestTrellis->getLogP(NBEST_STATE(j, NOS))
//	<< " " << nbestTrellis->getLogP(NBEST_STATE(j, S))
//	<< endl;
	}
    }

    if (debug >= DEBUG_PROGRESS) {
	cerr << endl;
    }

    return nbestTrellis;
}

/*
 * Forward-backward on nbest trellis
 */
void
forwardBackNbest(Array<NBestList *> &nbestLists, Array<char *> &nbestNames,
		 unsigned numLists, LM &lm, double lmw, double wtw)
{
    Vocab &vocab = lm.vocab;

    Trellis<unsigned> *nbestTrellis =
			forwardNbest(nbestLists, numLists, lm, lmw, wtw);

    if (debug >= DEBUG_PROGRESS) {
	cerr << "Backward pass..." << numLists - 1;
    }

    nbestTrellis->initBack();

    /*
     * Initialize the backward probs scores for the last nbest list
     */
    {
	for (unsigned j = 0;
	     j < nbestLists[numLists - 1]->numHyps() &&
		(maxRescore <= 0 || j < maxRescore);
	     j ++)
	{
	    NBestHyp &hyp = nbestLists[numLists - 1]->getHyp(j);

	    /*
	     * We don't assume that the final utterance is a complete
	     * sentence, so initialize the beta's to 1.
	     * If we did, this should be p(</s> | hyp) .
	     */
	    nbestTrellis->setBackProb(NBEST_STATE(j, NOS), LogP_One);
	    if (bias > 0.0) {
		nbestTrellis->setBackProb(NBEST_STATE(j, S), LogP_One);
	    }
	}
    }

    for (int i = numLists - 2; i >= 0; i --) {
	makeArray(VocabIndex, context, contextLength + 1);
				/* word context from preceding nbest hyp */

	if (debug >= DEBUG_PROGRESS) {
	    cerr << "..." << i;
	}

	nbestTrellis->stepBack();

	for (unsigned j = 0;
	     j < nbestLists[i]->numHyps() &&
		(maxRescore <= 0 || j < maxRescore);
	     j ++)
	{
	    NBestHyp &hyp = nbestLists[i]->getHyp(j);
	    unsigned hypLen = Vocab::length(hyp.words);

	    for (unsigned k = 0;
		 k < nbestLists[i+1]->numHyps() &&
		    (maxRescore <= 0 || k < maxRescore);
		 k ++)
	    {
		NBestHyp &nextHyp = nbestLists[i+1]->getHyp(k);

		LogP NOSscore, Sscore;

		/*
		 * Set up the last two words from the previous hyp
		 * as context for this one.
		 */
		{
		    unsigned k;

		    for (k = 0; k < contextLength && k < hypLen; k ++) {
			context[k] = hyp.words[hypLen - 1 - k];
		    }
		    context[k] = Vocab_None;
		}

		segmentHyp(nextHyp, context, lm, NOSscore, Sscore, NOSTATE);

//cerr << "j = " << j << " k = " << k << " context  = " << (vocab.use(), context) << " : " << NOSscore << " " << Sscore << endl;


		nbestTrellis->updateBack(NBEST_STATE(j, NOS),
						NBEST_STATE(k, NOS),
			nextHyp.acousticScore + lmw * NOSscore +
						wtw * nextHyp.numWords);
		if (bias > 0.0) {
		    nbestTrellis->updateBack(NBEST_STATE(j, NOS),
						    NBEST_STATE(k, S),
			    nextHyp.acousticScore + lmw * Sscore +
						    wtw * nextHyp.numWords);

		    /*
		     * Now the same for the case where the previous hyp's
		     * last word starts a sentence.
		     */
		    if (hypLen == 0) {
			context[0] = Vocab_None;
		    } else {
			context[0] = hyp.words[hypLen - 1];
			context[1] = vocab.ssIndex();
			context[2] = Vocab_None;
		    }
		    segmentHyp(nextHyp, context, lm, NOSscore, Sscore, NOSTATE);
//cerr << "j = " << j << " k = " << k << " context  = " << (vocab.use(), context) << " : " << NOSscore << " " << Sscore << endl;

		    nbestTrellis->updateBack(NBEST_STATE(j, S),
						    NBEST_STATE(k, NOS),
			    nextHyp.acousticScore + lmw * NOSscore +
						    wtw * nextHyp.numWords);
		    nbestTrellis->updateBack(NBEST_STATE(j, S),
						    NBEST_STATE(k, S),
			    nextHyp.acousticScore + lmw * Sscore +
						    wtw * nextHyp.numWords);
		}
	    }
//cerr << "BACK:list " << i << " hyp " << j
//	<< " back " << nbestTrellis->getBackProb(NBEST_STATE(j, NOS))
//	<< " " << nbestTrellis->getBackProb(NBEST_STATE(j, S))
//	<< endl;
	}
    }

    if (debug >= DEBUG_PROGRESS) {
	cerr << endl;
    }

    /*
     * Reset the LM scores in the nbest lists to reflect the posterior
     * probabilities computes from alphas/betas.
     * The posterior of a hyp is proportional to alpha(hyp) * beta(hyp).
     * We divide by the hyp's acoustic score and wt penalty to get back
     * a lm score.
     */
    unsigned h;
    for (h = 0; h < numLists ; h ++) {
	for (unsigned j = 0;
	     j < nbestLists[h]->numHyps() &&
		(maxRescore <= 0 || j < maxRescore);
	     j ++)
	{
	    NBestHyp &hyp = nbestLists[h]->getHyp(j);

	    LogP2 posterior =
		AddLogP(nbestTrellis->getLogP(NBEST_STATE(j, S), h) +
		            nbestTrellis->getBackLogP(NBEST_STATE(j, S), h),
		        nbestTrellis->getLogP(NBEST_STATE(j, NOS), h) +
		            nbestTrellis->getBackLogP(NBEST_STATE(j, NOS), h));

//cerr << "BACK:list " << h << " hyp " << j
//	<< " forw " << nbestTrellis->getLogP(NBEST_STATE(j, NOS), h)
//	<< " " << nbestTrellis->getLogP(NBEST_STATE(j, S), h)
//	<< " back " << nbestTrellis->getBackLogP(NBEST_STATE(j, NOS), h)
//	<< " " << nbestTrellis->getBackLogP(NBEST_STATE(j, S), h)
//	<< " post " << posterior << endl;

	    hyp.languageScore = (posterior -
				 hyp.acousticScore -
				 wtw * hyp.numWords) / lmw;
	}
    }

    /*
     * Dump out the new nbest scores
     */
    for (h = 0; h < numLists; h ++) {
	if (writeNbestDir) {
	    makeArray(char, outputName, strlen(writeNbestDir) + 2 + 256);

	    char *rootname = strrchr(nbestNames[h], '/');
	    if (rootname) {
		rootname += 1;
	    } else {
		rootname = nbestNames[h];
	    }
	    sprintf(outputName, "%s/%.255s", writeNbestDir, rootname);

	    File file(outputName, "w");

	    nbestLists[h]->write(file, false, maxRescore);
	} else {
	    cout << "<nbestlist " << (h + 1) << ">\n";

	    File sout(stdout);

	    nbestLists[h]->write(sout, false, maxRescore);
	}
    }

    delete nbestTrellis;
}

/*
 * Viterbi on nbest trellis
 */
void
segmentNbest(Array<NBestList *> &nbestLists, unsigned numLists,
	     LM &lm, double lmw, double wtw)
{
    Vocab &vocab = lm.vocab;

    Trellis<unsigned> *nbestTrellis =
			forwardNbest(nbestLists, numLists, lm, lmw, wtw);

    makeArray(unsigned, bestStates, numLists);

    if (nbestTrellis->viterbi(bestStates, numLists) != numLists) {
	cerr << "nbestTrellis->viterbi failed\n";
	delete nbestTrellis;
	return;
    }

    if (debug >= DEBUG_TRANSITIONS) {
	cerr << "Best hyps states: ";
	for (unsigned i = 0; i < numLists; i++) {
	    cerr << NBEST_HYP(bestStates[i]) << "("
		 << stateNames[NBEST_SEG(bestStates[i])] << ") " ;
	}
	cerr << endl;
    }

    if (debug >= DEBUG_PROGRESS) {
	cerr << "Viterbi...0";
    }

    /*
     * Do a viterbi traceback on the best hyps
     */
    {
	makeArray(VocabIndex, context, contextLength + 1);
				/* word context from preceding nbest hyp */
	LogP NOSscore, Sscore;	/* dummies required by segmentHyp() */

	context[0] = vocab.ssIndex();
	context[1] = Vocab_None;

	NBestHyp &hyp = nbestLists[0]->getHyp(NBEST_HYP(bestStates[0]));
	SegmentState seg = NBEST_SEG(bestStates[0]);

	segmentHyp(hyp, context, lm, NOSscore, Sscore, seg);

	for (unsigned i = 1; i < numLists; i ++) {
	    NBestHyp &prevHyp =
			nbestLists[i-1]->getHyp(NBEST_HYP(bestStates[i-1]));
	    unsigned prevLen = Vocab::length(prevHyp.words);
	    SegmentState prevSeg = NBEST_SEG(bestStates[i-1]);

	    NBestHyp &hyp = nbestLists[i]->getHyp(NBEST_HYP(bestStates[i]));
	    SegmentState seg = NBEST_SEG(bestStates[i]);

	    if (debug >= DEBUG_PROGRESS) {
		cerr << "..." << i;
	    }

	    /*
	     * Set up the last two words from the previous hyp
	     * as context for this one.
	     */
	    if (prevLen == 0) {
		context[0] = Vocab_None;
	    } else {
		if (prevSeg == S) {
		    context[0] = prevHyp.words[prevLen - 1];
		    context[1] = vocab.ssIndex();
		    context[2] = Vocab_None;
		} else {
		    unsigned k;

		    for (k = 0; k < contextLength && k < prevLen; k ++) {
			context[k] = prevHyp.words[prevLen - 1 - k];
		    }
		    context[k] = Vocab_None;
		}
	    }
	    segmentHyp(hyp, context, lm, NOSscore, Sscore, seg);
	}
	if (debug >= DEBUG_PROGRESS) {
	    cerr << endl;
	}
    }
	    
    delete nbestTrellis;
}

void insertStartEndTags(NBestList &nb, VocabIndex first, VocabIndex last)
{
    if (first == Vocab_None && last == Vocab_None) {
	return;
    }

    for (unsigned h = 0; h < nb.numHyps(); h++) {
	NBestHyp &hyp = nb.getHyp(h);

	unsigned numWords = Vocab::length(hyp.words);

	VocabIndex *newWords = new VocabIndex[numWords + 3];
	assert(newWords != 0);

	unsigned i = 0;

	if (first != Vocab_None) {
	    newWords[i ++] = first;
	}
	    
	for (unsigned j = 0; j < numWords; j ++) { 
	    newWords[i ++] = hyp.words[j];
	}

	if (last != Vocab_None) {
	    newWords[i ++] = last;
	}
	newWords[i] = Vocab_None;

	delete [] hyp.words;
	hyp.words = newWords;
    }
}

void
processNbestLists(const char *fileList, LM &lm, LM &oldLM,
					VocabIndex firstTag, VocabIndex lastTag)
{
    Vocab &vocab = lm.vocab;

    Array<char *> nbestFileList;
    unsigned numNbestLists = 0;

    if (debug >= DEBUG_PROGRESS) {
	cerr << "Processing file list " << fileList << endl;
    }

    /*
     * Read list of nbest filenames
     */
    {
	File file(fileList, "r");

	char *line;
        char *strtok_ptr = NULL;
	while ((line = file.getline())) {
	    strtok_ptr = NULL;
	    char *fname = MStringTokUtil::strtok_r(line, wordSeparators, &strtok_ptr);
	    if (fname) {
		nbestFileList[numNbestLists] = strdup(fname);
		assert(nbestFileList[numNbestLists] != 0);

		numNbestLists ++;
	    }
	}
    }

    /*
     * Read nbest lists
     */
    Array<NBestList *> nbestListArray(0, numNbestLists);

    unsigned i;
    for (i = 0; i < numNbestLists; i ++) {
	File file(nbestFileList[i], "r");

	if (debug >= DEBUG_PROGRESS) {
	    cerr << "Reading " << nbestFileList[i] << endl;
	}

	nbestListArray[i] = new NBestList(vocab, maxNbest);
	assert(nbestListArray[i] != 0);

	nbestListArray[i]->read(file);    

	/*
	 * Compute acoustic-only scores from aggregate recognizer scores
	 */
	if (decipherLM) {
	    nbestListArray[i]->decipherFix(oldLM, decipherLMW, decipherWTW);
	}

	/*
	 * Reorder N-best hyps so -max-rescore can look at the top ones
	 */
	if (!noReorder) {
	    nbestListArray[i]->reweightHyps(rescoreLMW, rescoreWTW);
	    nbestListArray[i]->sortHyps();
	}

	/*
	 * Normalize acoustic scores to avoid exponent overflow in 
	 * exponentiating logscores
	 */
	nbestListArray[i]->acousticNorm();

	/*
	 * Remove noise and pauses which would disrupt the DP procedure
	 */
	nbestListArray[i]->removeNoise(lm);

	/*
	 * Insert start/end tags
	 */
	insertStartEndTags(*nbestListArray[i], firstTag, lastTag);
    }

    if (fbRescore) {
	forwardBackNbest(nbestListArray, nbestFileList, numNbestLists,
		         lm, rescoreLMW, rescoreWTW);
    } else {
	segmentNbest(nbestListArray, numNbestLists,
		     lm, rescoreLMW, rescoreWTW);
    }

    /*
     * Free storage
     */
    for (i = 0; i < numNbestLists; i ++) {
	free(nbestFileList[i]);
	delete nbestListArray[i];
    }
}

int
main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    /*
     * Set length of N-gram context;
     * use at least 2 to hold previous word and <s> tags
     */
    contextLength = order - 1;
    if ((int)contextLength < 2) {
	contextLength = 2;
    }

    /*
     * Construct language model
     */
    Vocab vocab;
    LM    *useLM;

    if (!lmFile) {
	cerr << "need a language model\n";
	exit(1);
    }

    vocab.toLower() = toLower ? true : false;

    VocabIndex startTagIndex, endTagIndex;
    if (startTag) {
	startTagIndex = vocab.addWord(startTag);
    } else {
	startTagIndex = Vocab_None;
    }
    if (endTag) {
	endTagIndex = vocab.addWord(endTag);
    } else {
	endTagIndex = Vocab_None;
    }

    Ngram *ngramLM;
	
    {
	File file(lmFile, "r");

	ngramLM = new Ngram(vocab, order);
	assert(ngramLM != 0);

	ngramLM->debugme(debug);
	ngramLM->read(file);
    }

    if (bayesLength >= 0) {
	/*
	 * create a Bayes mixture LM
	 */
	Ngram *lm1 = ngramLM;
	Ngram *lm2 = new Ngram(vocab, order);
	assert(lm2 != 0);

	lm2->debugme(debug);

	if (!mixFile) {
	    cerr << "no mix-lm file specified\n";
	    exit(1);
	}

	File file2(mixFile, "r");

	if (!lm2->read(file2)) {
	    cerr << "format error in mix-lm file\n";
	    exit(1);
	}

	/*
	 * create a Bayes mixture from the two
	 */
	useLM = new BayesMix(vocab, *lm1, *lm2,
				bayesLength, mixLambda, bayesScale);

	assert(useLM != 0);
	useLM->debugme(debug);
    } else {
	useLM = ngramLM;
    }

    if (!sTag) {
	sTag = vocab.getWord(vocab.ssIndex());
	if (!sTag) {
		cerr << "couldn't find a segment tag in LM\n";
		exit(1);
	}
    }

    /*
     * Skip noise tags in scoring
     */
    if (noiseVocabFile) {
	File file(noiseVocabFile, "r");
	useLM->noiseVocab.read(file);
    }
    if (noiseTag) {				/* backward compatibility */
	useLM->noiseVocab.addWord(noiseTag);
    }

    /*
     * Read recognizer LM
     */
    DecipherNgram oldLM(vocab, 2);

    if (decipherLM) {
	oldLM.debugme(debug);

	File file(decipherLM, "r");

	if (!oldLM.read(file)) {
	    cerr << "format error in Decipher LM\n";
	    exit(1);
	}
    }

    if (nbestFiles) {
	processNbestLists(nbestFiles, *useLM, oldLM,
					startTagIndex, endTagIndex);
    }

    /*
     * Process all remaining args as nbest list lists.
     * Print delimiter info
     */
    for (unsigned i = 1; argv[i] != 0; i ++) {
	if (!(fbRescore && writeNbestDir)) {
	    cout << "<nbestfile " << argv[i] << ">\n";
	}
	processNbestLists(argv[i], *useLM, oldLM, startTagIndex, endTagIndex);
    }

#ifdef DEBUG
    if (ngramLM != useLM) {
	delete ngramLM;
    }
    delete useLM;
    return 0;
#endif /* DEBUG */

    exit(0);
}

