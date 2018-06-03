/*
 * disambig --
 *	Disambiguate text tokens using a hidden ngram model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International, 2013-2015 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: disambig.cc,v 1.55 2015-03-05 07:48:14 stolcke Exp $";
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
#include "VocabMap.h"
#include "LMClient.h"
#include "NullLM.h"
#include "Ngram.h"
#include "NgramCountLM.h"
#include "ProductNgram.h"
#include "BayesMix.h"
#include "Trellis.cc"
#include "LHash.cc"
#include "Array.cc"

typedef const VocabIndex *VocabContext;

#define DEBUG_ZEROPROBS		1
#define DEBUG_TRANSITIONS	2
#define DEBUG_TRELLIS	        3

static int version = 0;
static unsigned numNbest = 1;
static unsigned order = 2;
static unsigned debug = 0;
static int scale = 0;
static char *lmFile = 0;
static char *useServer = 0;
static int cacheServedNgrams = 0;
static int useCountLM = 0;
static int factored = 0;
#define MAX_MIX_LMS 10
static char *mixFile[MAX_MIX_LMS] =
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static int bayesLength = -1;
static double bayesScale = 1.0;
static char *contextPriorsFile = 0;
static double mixLambda[MAX_MIX_LMS] =
	{ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
static char *vocab1File = 0;
static char *vocab2File = 0;
static char *vocab2AliasFile = 0;
static char *mapFile = 0;
static char *classesFile = 0;
static char *mapWriteFile = 0;
static char *textFile = 0;
static char *textMapFile = 0;
static char *countsFile = 0;
static int keepUnk = 0;
static int keepnull = 1;
static int tolower1 = 0;
static int tolower2 = 0;
static double lmw = 1.0;
static double mapw = 1.0;
static double prune = -1.0;
static int fb = 0;
static int fwOnly = 0;
static int posteriors = 0;
static int totals = 0;
static int logMap = 0;
static int noEOS = 0;
static int continuous = 0;
static char *escape = 0;

const LogP LogP_PseudoZero = -100;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_STRING, "lm", &lmFile, "hidden token sequence model" },
    { OPT_STRING, "use-server", &useServer, "port@host to use as LM server" },
    { OPT_TRUE, "cache-served-ngrams", &cacheServedNgrams, "enable client side caching" },
    { OPT_TRUE, "count-lm", &useCountLM, "use a count-based LM" },
    { OPT_TRUE, "factored", &factored, "use a factored LM" },
    { OPT_UINT, "bayes", &bayesLength, "context length for Bayes mixture LM" },
    { OPT_FLOAT, "bayes-scale", &bayesScale, "log likelihood scale for -bayes" },
    { OPT_STRING, "mix-lm", &mixFile[1], "LM to mix in" },
    { OPT_FLOAT, "lambda", &mixLambda[0], "mixture weight for -lm" },
    { OPT_STRING, "mix-lm2", &mixFile[2], "second LM to mix in" },
    { OPT_FLOAT, "mix-lambda2", &mixLambda[2], "mixture weight for -mix-lm2" },
    { OPT_STRING, "mix-lm3", &mixFile[3], "third LM to mix in" },
    { OPT_FLOAT, "mix-lambda3", &mixLambda[3], "mixture weight for -mix-lm3" },
    { OPT_STRING, "mix-lm4", &mixFile[4], "fourth LM to mix in" },
    { OPT_FLOAT, "mix-lambda4", &mixLambda[4], "mixture weight for -mix-lm4" },
    { OPT_STRING, "mix-lm5", &mixFile[5], "fifth LM to mix in" },
    { OPT_FLOAT, "mix-lambda5", &mixLambda[5], "mixture weight for -mix-lm5" },
    { OPT_STRING, "mix-lm6", &mixFile[6], "sixth LM to mix in" },
    { OPT_FLOAT, "mix-lambda6", &mixLambda[6], "mixture weight for -mix-lm6" },
    { OPT_STRING, "mix-lm7", &mixFile[7], "seventh LM to mix in" },
    { OPT_FLOAT, "mix-lambda7", &mixLambda[7], "mixture weight for -mix-lm7" },
    { OPT_STRING, "mix-lm8", &mixFile[8], "eighth LM to mix in" },
    { OPT_FLOAT, "mix-lambda8", &mixLambda[8], "mixture weight for -mix-lm8" },
    { OPT_STRING, "mix-lm9", &mixFile[9], "ninth LM to mix in" },
    { OPT_FLOAT, "mix-lambda9", &mixLambda[9], "mixture weight for -mix-lm9" },
    { OPT_STRING, "context-priors", &contextPriorsFile, "context-dependent mixture weights file" },
    { OPT_UINT, "nbest", &numNbest, "number of nbest hypotheses to generate in Viterbi search" },
    { OPT_UINT, "order", &order, "ngram order to use for lm" },
    { OPT_STRING, "write-vocab1", &vocab1File, "output observable vocabulary" },
    { OPT_STRING, "write-vocab2", &vocab2File, "output hidden vocabulary" },
    { OPT_STRING, "map", &mapFile, "mapping from observable to hidden tokens" },
    { OPT_STRING, "classes", &classesFile, "mapping in class expansion format" },
    { OPT_TRUE, "logmap", &logMap, "map file contains log probabilities" },
    { OPT_STRING, "escape", &escape, "escape prefix to pass data through to output" },
    { OPT_STRING, "write-map", &mapWriteFile, "output map file (for validation)" },
    { OPT_STRING, "write-counts", &countsFile, "output substitution counts" },
    { OPT_TRUE, "scale", &scale, "scale map probabilities by unigram probs" },
    { OPT_TRUE, "keep-unk", &keepUnk, "preserve unknown words" },
    { OPT_FALSE, "nonull", &keepnull, "remove <NULL> in factored LM" },
    { OPT_TRUE, "tolower1", &tolower1, "map observable vocabulary to lowercase" },
    { OPT_TRUE, "tolower2", &tolower2, "map hidden vocabulary to lowercase" },
    { OPT_STRING, "vocab-aliases", &vocab2AliasFile, "hidden vocab alias file" },
    { OPT_STRING, "text", &textFile, "text file to disambiguate" },
    { OPT_STRING, "text-map", &textMapFile, "text+map file to disambiguate" },
    { OPT_FLOAT, "lmw", &lmw, "language model weight" },
    { OPT_FLOAT, "mapw", &mapw, "map log likelihood weight" },
    { OPT_TRUE, "fb", &fb, "use forward-backward algorithm" },
    { OPT_TRUE, "fwOnly", &fwOnly, "use forward probabilities only" },
    { OPT_FLOAT, "prune", &prune, "apply pruning cutoff to forward computation" },
    { OPT_TRUE, "posteriors", &posteriors, "output posterior probabilities" },
    { OPT_TRUE, "totals", &totals, "output total string probabilities" },
    { OPT_TRUE, "no-eos", &noEOS, "don't assume end-of-sentence token" },
    { OPT_TRUE, "continuous", &continuous, "read input without line breaks" },
    { OPT_UINT, "debug", &debug, "debugging level for lm" },
};

/*
 * Disambiguate a sentence by finding the hidden tag sequence compatible with
 * it that has the highest probability
 */
unsigned
disambiguateSentence(Vocab &vocab, VocabIndex *wids, VocabIndex *hiddenWids[],
		     LogP totalProb[], VocabMap &map, LM &lm, VocabMap *counts,
		     unsigned numNbest, Boolean positionMapped = false)
{
    static VocabIndex emptyContext[] = { Vocab_None };
    unsigned len = Vocab::length(wids);
    assert(len > 0);

    Trellis<VocabContext> trellis(len, numNbest);

    /*
     * Prime the trellis with the tag likelihoods of the first position
     * (No transition probs for this one.)
     */
    {
	VocabMapIter iter(map, positionMapped ? (VocabIndex)0 : wids[0]);

	VocabIndex context[2];
	Prob prob;

	context[1] = Vocab_None;
	while (iter.next(context[0], prob)) {
	    LogP localProb = logMap ? prob : ProbToLogP(prob);
	    trellis.setProb(context, localProb);
	}
    }

    unsigned pos = 1;
    while (wids[pos] != Vocab_None) {

	trellis.step();

	/*
	 * Iterate over all combinations of hidden tags for the current
	 * position and contexts for the previous position.
	 */
	VocabMapIter currIter(map,
				positionMapped ? (VocabIndex)pos : wids[pos]);
	VocabIndex currWid;
	Prob currProb;

	while (currIter.next(currWid, currProb)) {
	    LogP localProb = logMap ? currProb : ProbToLogP(currProb);
	    LogP unigramProb = lm.wordProb(currWid, emptyContext);
	    
	    if (debug >= DEBUG_ZEROPROBS &&
		unigramProb == LogP_Zero &&
		currWid != map.vocab2.unkIndex())
	    {
		cerr << "warning: map token " << map.vocab2.getWord(currWid)
		     << " has prob zero\n";
	    }

	    if (scale && unigramProb != LogP_Zero) {
		/*
		 * Scale the map probability p(tag|word)
		 * by the tag prior probability p(tag), so the
		 * local distance is a scaled version of the
		 * likelihood p(word|tag)
		 */
		localProb -= unigramProb;
	    }

	    /*
	     * Set up new context to transition to
	     */
	    VocabIndex newContext[maxWordsPerLine + 2];
	    newContext[0] = currWid;

	    TrellisIter<VocabContext> prevIter(trellis, pos - 1);
	    VocabContext prevContext;
	    LogP prevProb;

	    while (prevIter.next(prevContext, prevProb)) {
		LogP transProb = lm.wordProb(currWid, prevContext);

		if (transProb == LogP_Zero && unigramProb == LogP_Zero) {
		    /*
		     * Zero probability due to an OOV
		     * would equalize the probabilities of all paths
		     * and make the Viterbi backtrace useless.
		     */
		    transProb = LogP_PseudoZero;
		}

		/*
		 * Determined new context
		 */
		unsigned i;
		for (i = 0; i < maxWordsPerLine && 
					prevContext[i] != Vocab_None; i ++)
		{
		    newContext[i + 1] = prevContext[i];
		}
		newContext[i + 1] = Vocab_None;

		/*
		 * Truncate context to what is actually used by LM,
		 * but keep at least one word so we can recover words later.
		 */
		unsigned usedLength;
		lm.contextID(newContext, usedLength);
		newContext[usedLength > 0 ? usedLength : 1] = Vocab_None;

		if (debug >= DEBUG_TRANSITIONS) {
		    cerr << "position = " << pos
			 << " from: " << (map.vocab2.use(), prevContext)
			 << " to: " << (map.vocab2.use(), newContext)
			 << " trans = " << transProb
			 << " local = " << localProb
			 << endl;
		}

		trellis.update(prevContext, newContext,
			       weightLogP(lmw, transProb) +
			       weightLogP(mapw, localProb));
	    }
	}

	pos ++;

	if (prune >= 0.0) {
	    cerr << "pruned " << trellis.prune(prune) << endl;
	}
    }

    if (debug >= DEBUG_TRELLIS) {
	cerr << "Trellis:\n" << trellis << endl;
    }

    /*
     * Check whether viterbi-nbest (default) is asked for.  If so,
     * do it and return. Otherwise do forward-backward.
     */
    if (!fb && !fwOnly && !posteriors) {
	//VocabContext hiddenContexts[numNbest][len + 1];
	makeArray(VocabContext *, hiddenContexts, numNbest);

	for (unsigned n = 0; n < numNbest; n++) {
	    hiddenContexts[n] = new VocabContext[len + 1];
	    assert(hiddenContexts[n] != 0);

	    if (trellis.nbest_viterbi(hiddenContexts[n], len,
						n, totalProb[n]) != len)
	    {
		return n;
	    }
	      
	    /*
	     * Transfer the first word of each state (word context) into the
	     * word index string
	     */
	    for (unsigned i = 0; i < len; i ++) {
		hiddenWids[n][i] = hiddenContexts[n][i][0];
	    }
	    hiddenWids[n][len] = Vocab_None;
	}

	/* 
	 * update v1-v2 counts if requested 
	 */
	if (counts) {
	    for (unsigned i = 0; i < len; i++) {
		counts->put(wids[i], hiddenWids[0][i],
			    counts->get(wids[i], hiddenWids[0][i]) + 1);
	    }
	}

	for (unsigned i = 0; i < numNbest; i++) {
	    delete [] hiddenContexts[i];
	}

	return numNbest;
    } else {
	/*
	 * Viterbi-nbest wasn't asked for.  We run the Forward-backward
	 * algorithm: compute backward scores, but only get the 1st best.
	 */
	pos --;
	trellis.initBack(pos);

	/* 
	 * Initialize backward probabilities
	 */
	{
	    TrellisIter<VocabContext> iter(trellis, pos);
	    VocabContext context;
	    LogP prob;
	    while (iter.next(context, prob)) {
		trellis.setBackProb(context, LogP_One);
	    }
	}

	while (pos > 0) {
	    trellis.stepBack();

	    /*
	     * Iterate over all combinations of hidden tags for the current
	     * position and contexts for the previous position.
	     */
	    VocabMapIter currIter(map,
				positionMapped ? (VocabIndex)pos : wids[pos]);
	    VocabIndex currWid;
	    Prob currProb;

	    while (currIter.next(currWid, currProb)) {
		LogP localProb = logMap ? currProb : ProbToLogP(currProb);
		LogP unigramProb = lm.wordProb(currWid, emptyContext);
		
		if (debug >= DEBUG_ZEROPROBS &&
		    unigramProb == LogP_Zero &&
		    currWid != map.vocab2.unkIndex())
		{
		    cerr << "warning: map token " << map.vocab2.getWord(currWid)
			 << " has prob zero\n";
		}

		if (scale && unigramProb != LogP_Zero) {
		    /*
		     * Scale the map probability p(tag|word)
		     * by the tag prior probability p(tag), so the
		     * local distance is a scaled version of the
		     * likelihood p(word|tag)
		     */
		    localProb -= unigramProb;
		}

		/*
		 * Set up new context to transition to
		 */
		VocabIndex newContext[maxWordsPerLine + 2];
		newContext[0] = currWid;

		TrellisIter<VocabContext> prevIter(trellis, pos - 1);
		VocabContext prevContext;
		LogP prevProb;

		while (prevIter.next(prevContext, prevProb)) {
		    LogP transProb = lm.wordProb(currWid, prevContext);

		    if (transProb == LogP_Zero && unigramProb == LogP_Zero) {
			/*
			 * Zero probability due to an OOV
			 * would equalize the probabilities of all paths
			 * and make the Viterbi backtrace useless.
			 */
			transProb = LogP_PseudoZero;
		    }

		    /*
		     * Determined new context
		     */
		    unsigned i;
		    for (i = 0; i < maxWordsPerLine && 
					    prevContext[i] != Vocab_None; i ++)
		    {
			newContext[i + 1] = prevContext[i];
		    }
		    newContext[i + 1] = Vocab_None;

		    /*
		     * Truncate context to what is actually used by LM,
		     * but keep at least one word so we can recover words later.
		     */
		    unsigned usedLength;
		    lm.contextID(newContext, usedLength);
		    newContext[usedLength > 0 ? usedLength : 1] = Vocab_None;

		    trellis.updateBack(prevContext, newContext,
				       weightLogP(lmw, transProb) +
				       weightLogP(mapw, localProb));

		    if (debug >= DEBUG_TRANSITIONS) {
			cerr << "backward position = " << pos
			     << " from: " << (map.vocab2.use(), prevContext)
			     << " to: " << (map.vocab2.use(), newContext)
			     << " trans = " << transProb
			     << " local = " << localProb
			     << endl;
		    }
		}
	    }
	    pos --;
	}

	/*
	 * Compute posterior symbol probabilities and extract most
	 * probable symbol for each position
	 */
	for (pos = 0; pos < len; pos ++) {
	    /*
	     * Compute symbol probabilities by summing posterior probs
	     * of all states corresponding to the same symbol
	     */
	    LHash<VocabIndex,LogP2> symbolProbs;

	    TrellisIter<VocabContext> iter(trellis, pos);
	    VocabContext context;
	    LogP forwProb;
	    while (iter.next(context, forwProb)) {
		LogP2 posterior;
		
		if (fwOnly) {
		    posterior = forwProb;
		} else if (fb) {
		    posterior = forwProb + trellis.getBackLogP(context, pos);
		} else {
		    LogP backmax;
		    posterior = trellis.getMax(context, pos, backmax);
		    posterior += backmax;
		}

		Boolean foundP;
		LogP2 *symbolProb = symbolProbs.insert(context[0], foundP);
		if (!foundP) {
		    *symbolProb = posterior;
		} else {
		    *symbolProb = AddLogP(*symbolProb, posterior);
		}
	    }

	    /*
	     * Find symbol with highest posterior
	     */
	    LogP2 totalPosterior = LogP_Zero;
	    LogP2 maxPosterior = LogP_Zero;
	    VocabIndex bestSymbol = Vocab_None;

	    LHashIter<VocabIndex,LogP2> symbolIter(symbolProbs);
	    LogP2 *symbolProb;
	    VocabIndex symbol;
	    while ((symbolProb = symbolIter.next(symbol))) {
		if (bestSymbol == Vocab_None || *symbolProb > maxPosterior) {
		    bestSymbol = symbol;
		    maxPosterior = *symbolProb;
		}
		totalPosterior = AddLogP(totalPosterior, *symbolProb);
	    }

	    if (bestSymbol == Vocab_None) {
		cerr << "no forward-backward state for position "
		     << pos << endl;
		return 0;
	    }

	    hiddenWids[0][pos] = bestSymbol;

	    /*
	     * Print posterior probabilities
	     */
	    if (posteriors) {
		cout << vocab.getWord(wids[pos]) << "\t";

		symbolIter.init();
		while ((symbolProb = symbolIter.next(symbol))) {
		    LogP2 posterior = *symbolProb - totalPosterior;

		    cout << " " << map.vocab2.getWord(symbol)
			 << " " << (logMap ? posterior : LogPtoProb(posterior));
		}
		cout << endl;
	    }

	    /* 
	     * update v1-v2 counts if requested 
	     */
	    if (counts) {
		symbolIter.init();
		while ((symbolProb = symbolIter.next(symbol))) {
		    LogP2 posterior = *symbolProb - totalPosterior;

		    counts->put(wids[pos], symbol,
			        counts->get(wids[pos], symbol) +
					LogPtoProb(posteriors));
		}
	    }
	}

        /*
	 * Return total string probability summing over all paths
	 */
	totalProb[0] = trellis.sumLogP(len-1);
	hiddenWids[0][len] = Vocab_None;
	return 1;
    }
}

/*
 * Get one input sentences at a time, map it to wids, 
 * disambiguate it, and print out the result
 */
void
disambiguateFile(File &file, VocabMap &map, LM &lm, VocabMap *counts)
{
    char *line;
    VocabString sentence[maxWordsPerLine];

    unsigned escapeLen = escape ? strlen(escape) : 0;

    while ((line = file.getline())) {
	/*
	 * Pass escaped lines through unprocessed
	 */
        if (escape && strncmp(line, escape, escapeLen) == 0) {
	    cout << line;
	    continue;
	}

	unsigned numWords = Vocab::parseWords(line, sentence, maxWordsPerLine);
	if (numWords == maxWordsPerLine) {
	    file.position() << "too many words per sentence\n";
	} else {
	    VocabIndex wids[maxWordsPerLine + 2];
	    unsigned startPos;

	    map.vocab1.getIndices(sentence, &wids[1], maxWordsPerLine,
						    map.vocab1.unkIndex());

	    if (wids[1] == map.vocab1.ssIndex()) {
		startPos = 1;
	    } else {
		wids[0] = map.vocab1.ssIndex();
		startPos = 0;
	    }

	    if (noEOS || wids[numWords] == map.vocab1.seIndex()) {
		wids[numWords + 1] = Vocab_None;
	    } else {
		wids[numWords + 1] = map.vocab1.seIndex();
		wids[numWords + 2] = Vocab_None;
	    }

	    makeArray(VocabIndex *, hiddenWids, numNbest);
	    makeArray(VocabString,  hiddenWords, maxWordsPerLine + 2);

	    for (unsigned n = 0; n < numNbest; n++) {
		hiddenWids[n] = new VocabIndex[maxWordsPerLine + 2];
	    }

	    makeArray(LogP, totalProb, numNbest);
	    unsigned numHyps =
		disambiguateSentence(map.vocab1, &wids[startPos], hiddenWids,
					totalProb, map, lm, counts, numNbest);
	    if (!numHyps) {
		file.position() << "Disambiguation failed\n";
	    } else if (totals) {
		cout << totalProb[0] << endl;
	    } else if (!posteriors) {
		for (unsigned n = 0; n < numHyps; n++) {
		    map.vocab2.getWords(hiddenWids[n], hiddenWords,
							maxWordsPerLine + 2);
		    if (numNbest > 1) {
		      cout << "NBEST_" << n << " " << totalProb[n] << " ";
		    }

		    if (keepUnk) {
			/*
			 * Look for <unk> symbols in the output and replace
			 * them with the corresponding input tokens
			 */
			for (unsigned i = 0;
			     hiddenWids[n][i] != Vocab_None;
			     i++)
			{
			    if (i > 0 &&
				hiddenWids[n][i] == map.vocab2.unkIndex())
			    {
				hiddenWords[i] = sentence[i - 1];
			    }
			}
		    }
		    cout << (map.vocab2.use(), hiddenWords) << endl;
		}
	    }

	    for (unsigned n = 0; n < numNbest; n++) {
		delete [] hiddenWids[n];
	    }
	}
    }
}

/*
 * Read entire file ignoring line breaks, map it to wids, 
 * disambiguate it, and print out the result
 */
void
disambiguateFileContinuous(File &file, VocabMap &map, LM &lm,
							VocabMap *counts)
{
    char *line;
    Array<VocabIndex> wids;

    unsigned escapeLen = escape ? strlen(escape) : 0;
    unsigned lineStart = 0; // index into the above to mark the offset for the 
			    // current line's data

    while ((line = file.getline())) {
	/*
	 * Pass escaped lines through unprocessed
	 * (although this is pretty useless in continuous mode)
	 */
        if (escape && strncmp(line, escape, escapeLen) == 0) {
	    cout << line;
	    continue;
	}

	VocabString words[maxWordsPerLine];
	unsigned numWords =
		Vocab::parseWords(line, words, maxWordsPerLine);

	if (numWords == maxWordsPerLine) {
	    file.position() << "too many words per line\n";
	} else {
	    // This effectively allocates more space
	    wids[lineStart + numWords] = Vocab_None;

	    map.vocab1.getIndices(words, &wids[lineStart], numWords,
							map.vocab1.unkIndex());
	    lineStart += numWords;
	}
    }

    if (lineStart == 0) {		  // empty input -- nothing to do
	return;
    }

    makeArray(VocabIndex *, hiddenWids, numNbest);
    makeArray(VocabString, hiddenWords, maxWordsPerLine + 2);

    for (unsigned n = 0; n < numNbest; n++) {
	hiddenWids[n] = new VocabIndex[lineStart + 1];
	hiddenWids[n][lineStart] = Vocab_None;
    }

    makeArray(LogP, totalProb, numNbest);
    unsigned numHyps = disambiguateSentence(map.vocab1, &wids[0], hiddenWids,
					totalProb, map, lm, counts, numNbest);

    if (!numHyps) {
	file.position() << "Disambiguation failed\n";
    } else if (totals) {
	cout << totalProb[0] << endl;
    } else if (!posteriors) {
	for (unsigned n = 0; n < numHyps; n++) {
	    map.vocab2.getWords(hiddenWids[n], hiddenWords,
							maxWordsPerLine + 2);
	    if (numNbest > 1) {
	      cout << "NBEST_" << n << " " << totalProb[n] << " ";
	    }

	    for (unsigned i = 0; hiddenWids[n][i] != Vocab_None; i++) {
		// XXX: keepUnk not implemented yet.
		cout << map.vocab2.getWord(hiddenWids[n][i]) << " ";
	    }
	    cout << endl;
	}
    }

    for (unsigned n = 0; n < numNbest; n++) {
	delete [] hiddenWids[n];
    }
}

/*
 * Read a "text-map" file, containing one word per line, followed by
 * map entries;
 * disambiguate it, and print out the result
 */
void
disambiguateTextMap(File &file, Vocab &vocab, LM &lm, VocabMap *counts)
{
    char *line;

    unsigned escapeLen = escape ? strlen(escape) : 0;

    while ((line = file.getline())) {
	/*
	 * Hack alert! We pass the map entries associated with the word
	 * instances in a VocabMap, but we encode the word position (not
	 * its identity) as the first VocabIndex.
	 */
	PosVocabMap map(lm.vocab);

	unsigned numWords = 0;
	Array<VocabIndex> wids;

	/*
	 * Process one sentence
	 */
	Boolean haveEscape = false;

	do {
	    /*
	     * Pass escaped lines through unprocessed
	     * We also terminate an "sentence" whenever an escape line is found,
	     * but printing the escaped line has to be deferred until we're
	     * done processing the sentence.
	     */
	    if (escape && strncmp(line, escape, escapeLen) == 0) {
		haveEscape = true;
		break;
	    }

	    /*
	     * Read map line
	     */
	    VocabString mapFields[maxWordsPerLine];

	    unsigned howmany =
			Vocab::parseWords(line, mapFields, maxWordsPerLine);

	    if (howmany == maxWordsPerLine) {
		file.position() << "text map line has too many fields\n";
		return;
	    }

	    /*
	     * First field is the V1 word
	     * Note we use addWord() since V1 words are by definition
	     * only found in the textmap, so there are no OOVs here.
	     */
	    wids[numWords] = vocab.addWord(mapFields[0]);
	    
	    /*
	     * Parse the remaining words as either probs or V2 words
	     */
	    unsigned i = 1;

	    while (i < howmany) {
		double prob;

		VocabIndex w2 = lm.vocab.addWord(mapFields[i++]);

		if (i < howmany && parseProbOrLogP(mapFields[i], prob, logMap)) {
		    i ++;
		} else {
		    prob = logMap ? LogP_One : 1.0;
		}

		map.put((VocabIndex)numWords, w2, prob);
	    }
	} while (wids[numWords ++] != vocab.seIndex() &&
		 (line = file.getline()));

	if (numWords > 0) {
	    wids[numWords] = Vocab_None;

	    makeArray(VocabIndex *, hiddenWids, numNbest);
	    for (unsigned n = 0; n < numNbest; n++) {
		hiddenWids[n] = new VocabIndex[numWords + 1];
		hiddenWids[n][numWords] = Vocab_None;
	    }

	    makeArray(LogP, totalProb, numNbest);
	    unsigned numHyps =
		    disambiguateSentence(vocab, &wids[0], hiddenWids, totalProb,
					    map, lm, counts, numNbest, true);

	    if (!numHyps) {
		file.position() << "Disambiguation failed\n";
	    } else if (totals) {
		cout << totalProb[0] << endl;
	    } else if (!posteriors) {
		for (unsigned n = 0; n < numHyps; n++) {
		    if (numNbest > 1) {
			cout << "NBEST_" << n << " " << totalProb[n] << " ";
		    }

		    for (unsigned i = 0; hiddenWids[n][i] != Vocab_None; i++) {
			cout << map.vocab2.getWord(hiddenWids[n][i]) << " ";
		    }
		    cout << endl;
		}
	    }

	    for (unsigned n = 0; n < numNbest; n++) {
		delete [] hiddenWids[n];
	    }
	}

	if (haveEscape) {
	    cout << line;
	}
    }
}

BayesMix *
makeBayesMixLM(char *filenames[], Vocab &vocab, unsigned order,
	       LM *oldLM, double lambdas[])
{
    Array<LM *> allLMs;
    allLMs[0] = oldLM;

    Array<Prob> allLambdas;
    allLambdas[0] = lambdas[0];

    for (unsigned i = 1; i < MAX_MIX_LMS && filenames[i]; i++) {
	File file(filenames[i], "r");

	allLambdas[i] = lambdas[i];

	LM *lm;

	if (useServer && strchr(filenames[i], '@') && !strchr(filenames[i], '/')) {
	    /*
	     * Filename looks like a network LM spec -- create LMClient object
	     */
	    lm = new LMClient(vocab, filenames[i], order,
						cacheServedNgrams ? order : 0);
	    assert(lm != 0);
	} else {
	    File file(filenames[i], "r");

	    lm = factored ? 
			  new ProductNgram((ProductVocab &)vocab, order) :
			    new Ngram(vocab, order);
	    assert(lm != 0);

	    if (!lm->read(file, !textMapFile)) {
		cerr << "format error in mix-lm file " << filenames[i] << endl;
		exit(1);
	    }
	    // @kw false positive: RH.LEAK (lm->serverSocket)
	}

	allLMs[i] = lm;
    }

    BayesMix *newLM = new BayesMix(vocab, allLMs, allLambdas,
			           bayesLength, bayesScale);
    assert(newLM != 0);

    if (contextPriorsFile) {
        File file(contextPriorsFile, "r");

	if (!newLM->readContextPriors(file)) {
	    cerr << "error reading context priors\n";
	    exit(1);
	}
    }

    return newLM;
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
     * Construct language model
     */
    LM::initialDebugLevel = debug;

    Vocab *hiddenVocab;
    Vocab vocab;
    LM    *hiddenLM;

    if (useCountLM + factored > 1) {
	cerr << "NgramCountLM and factored LMs are mutually exclusive\n";
	exit(2);
    }

    if (lmFile && useServer) {
	cerr << "-lm and -use-server are mutually exclusive\n";
	exit(2);
    }

    hiddenVocab = factored ? new ProductVocab : new Vocab;
    assert(hiddenVocab != 0);

    VocabMap map(vocab, *hiddenVocab, logMap);

    vocab.toLower() = tolower1? true : false;
    hiddenVocab->toLower() = tolower2 ? true : false;
    hiddenVocab->unkIsWord() = keepUnk ? true : false;

    if (vocab2AliasFile) {
	File file(vocab2AliasFile, "r");
	hiddenVocab->readAliases(file);
    }

    if (factored) {
	((ProductVocab *)hiddenVocab)->nullIsWord() = keepnull ? true : false;
    }

    if (mapFile) {
	File file(mapFile, "r");

	if (!map.read(file)) {
	    cerr << "format error in map file\n";
	    exit(1);
	}
    }

    if (classesFile) {
	File file(classesFile, "r");

	if (!map.readClasses(file)) {
	    cerr << "format error in classes file\n";
	    exit(1);
	}
    }

    if (useServer) {
    	hiddenLM = new LMClient(*hiddenVocab, useServer, order,
						cacheServedNgrams ? order : 0);
	assert(hiddenLM != 0);
    } else if (lmFile) {
	File file(lmFile, "r");

	if (factored) {
	    hiddenLM = new ProductNgram(*(ProductVocab *)hiddenVocab, order);
	} else if (useCountLM) {
	    hiddenLM = new NgramCountLM(*hiddenVocab, order);
	} else {
	    hiddenLM = new Ngram(*hiddenVocab, order);
	}
	assert(hiddenLM != 0);

	/*
	 * Limit LM reading to the words found in the map file if 
	 * -text-map is NOT used
	 */
	hiddenLM->read(file, !textMapFile);
    } else {
	hiddenLM = new NullLM(*hiddenVocab);
	assert(hiddenLM != 0);
    }

    /*
     * Build the full LM used for hidden event decoding
     */
    LM *useLM = hiddenLM;

    if (mixFile[1]) {
	/*
	 * create a Bayes mixture LM 
	 */
	mixLambda[1] = 1.0 - mixLambda[0];
	for (unsigned i = 2; i < MAX_MIX_LMS; i++) {
	    mixLambda[1] -= mixLambda[i];
	}

	if (bayesLength < 0) {
	    if (contextPriorsFile) {
		/*
		 * User wants context-dependent priors but not Bayesian posteriors
		 */
		bayesLength = order - 1;
		bayesScale = 0.0;
	    }
	} else {
	    /*
	     * Default is not to use context
	     */
	    bayesLength = 0;
        }

	useLM = makeBayesMixLM(mixFile, *hiddenVocab, order,
			       useLM, mixLambda);
    }

    VocabMap *counts;
    if (countsFile) {
	counts = new VocabMap(vocab, *hiddenVocab);
	assert(counts != 0);

	counts->remove(vocab.ssIndex(), hiddenVocab->ssIndex());
	counts->remove(vocab.seIndex(), hiddenVocab->seIndex());
	counts->remove(vocab.unkIndex(), hiddenVocab->unkIndex());
    } else {
	counts = 0;
    }

    if (textFile) {
	File file(textFile, "r");

	if (continuous) {
	    disambiguateFileContinuous(file, map, *useLM, counts);
	} else {
	    disambiguateFile(file, map, *useLM, counts);
	}
    }

    if (textMapFile) {
	File file(textMapFile, "r");

	disambiguateTextMap(file, vocab, *useLM, counts);
    }

    if (countsFile) {
	File file(countsFile, "w");

	counts->writeBigrams(file);
    }

    if (mapWriteFile) {
	File file(mapWriteFile, "w");
	map.write(file);
    }

    if (vocab1File) {
	File file(vocab1File, "w");
	hiddenVocab->write(file);
    }
    if (vocab2File) {
	File file(vocab2File, "w");
	vocab.write(file);
    }

#ifdef DEBUG
    delete hiddenLM;
    delete hiddenVocab;
    return 0;
#endif /* DEBUG */

    exit(0);
}

