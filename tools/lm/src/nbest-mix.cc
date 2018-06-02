/*
 * nbest-mix --
 *	Interpolate N-Best lists
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1998-2004 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: nbest-mix.cc,v 1.6 2004/12/03 05:35:34 stolcke Exp $";
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <locale.h>

#include "option.h"
#include "version.h"
#include "File.h"
#include "Prob.h"
#include "Vocab.h"
#include "NBest.h"
#include "Array.cc"

#define DEBUG_ERRORS	1

static int version = 0;
static unsigned debug = 0;
static char *writeNbestFile = 0;
static unsigned maxNbest = 0;
static double rescoreLMW = 8.0;
static double rescoreWTW = 0.0;
static double posteriorScale = 0.0;
static int setAMscores = 0;
static int setLMscores = 0;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_UINT, "debug", &debug, "debugging level" },
    { OPT_STRING, "write-nbest", &writeNbestFile, "output n-best list" },
    { OPT_UINT, "max-nbest", &maxNbest, "maximum number of hyps to consider" },
    { OPT_FLOAT, "rescore-lmw", &rescoreLMW, "rescoring LM weight" },
    { OPT_FLOAT, "rescore-wtw", &rescoreWTW, "rescoring word transition weight" },
    { OPT_FLOAT, "posterior-scale", &posteriorScale, "divisor for log posterior estimates" },
    { OPT_TRUE, "set-am-scores", &setAMscores, "set AM scores in output n-best list" },
    { OPT_TRUE, "set-lm-scores", &setLMscores, "set LM scores in output n-best list" },
    { OPT_DOC, 0, 0, "weight1 nbest1 weight2 nbest2 ..." }
};

NBestList *
mixNbestFiles(unsigned nlists, Array<Prob> &weights,
				Array<NBestList *> &nbestLists)
{
    /*
     * create result nbest list by copying first of the input lists
     */
    NBestList *result = new NBestList(*nbestLists[0]);
    assert(result != 0);

    unsigned i;

    /*
     * compute hyp posteriors for all lists
     * also check that all lists have same number of hyps
     */
    for (i = 0; i < nlists; i ++) {
	nbestLists[i]->computePosteriors(rescoreLMW, rescoreWTW,
							posteriorScale);

	if (nbestLists[i]->numHyps() != result->numHyps()) {
	    cerr << "nbest list " << (i + 1)
		 << " has inconsistent number of hyps\n";
	    exit(1);
	}
    }

    LogP firstScore;

    /*
     * combine hyp posteriors
     */
    for (unsigned j = 0; j < result->numHyps(); j ++) {
	Prob totalPosterior = 0.0;

	for (i = 0; i < nlists; i ++) {
	    totalPosterior += weights[i] * nbestLists[i]->getHyp(j).posterior;
	}

	NBestHyp &resultHyp = result->getHyp(j);

	resultHyp.posterior = totalPosterior;
	resultHyp.totalScore = ProbToLogP(totalPosterior);

	if (setAMscores) {
	    assert(nlists > 0);

	    /*
	     * set LM scores and IP from the first n-best list, and 
	     * the AM score so that after score combination the posteriors
	     * correspond to the mixture of all lists
	     */
	    resultHyp.numWords = nbestLists[0]->getHyp(j).numWords;
	    resultHyp.languageScore = nbestLists[0]->getHyp(j).languageScore;
	    resultHyp.acousticScore = 
			resultHyp.totalScore * posteriorScale -
			resultHyp.languageScore * rescoreLMW -
			resultHyp.numWords * rescoreWTW;

	    /*
	     * Scale output AM scores to zero offset
	     */
	    if (j == 0) {
		firstScore = resultHyp.acousticScore;
	    }
	    resultHyp.acousticScore -= firstScore;
	} else if (setLMscores) {
	    assert(nlists > 0);

	    /*
	     * set acoustic scores and IP from the first n-best list, and 
	     * the LM score so that after score combination the posteriors
	     * correspond to the mixture of all lists
	     */
	    resultHyp.acousticScore = nbestLists[0]->getHyp(j).acousticScore;
	    resultHyp.numWords = nbestLists[0]->getHyp(j).numWords;
	    resultHyp.languageScore = 
			(resultHyp.totalScore * posteriorScale -
			resultHyp.acousticScore -
			resultHyp.numWords * rescoreWTW) / rescoreLMW;

	    /*
	     * Scale output LM scores to zero offset
	     */
	    if (j == 0) {
		firstScore = resultHyp.languageScore;
	    }
	    resultHyp.languageScore -= firstScore;
	} else {
	    /*
	     * set result scores so that acoustic scores reflect log posteriors
	     */
	    resultHyp.acousticScore = resultHyp.totalScore;
	    resultHyp.languageScore = 0.0;
	    resultHyp.numWords = 0;
	}
    }

    return result;
}

int
main (int argc, char *argv[])
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    argc = Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    if (setAMscores && setLMscores) {
	cerr << "cannot set both AM and LM scores\n";
	exit(2);
    }

    Vocab vocab;

    /*
     * Posterior scaling:  if not specified (= 0.0) use LMW for
     * backward compatibility.
     */
    if (posteriorScale == 0.0) {
	posteriorScale = (rescoreLMW == 0.0) ? 1.0 : rescoreLMW;
    }

    if ((argc - 1) % 2) {
	cerr << "number of arguments must be even\n";
	exit(1);
    }

    unsigned nlists = (argc - 1)/2;

    if (nlists < 1) {
	cerr << "need at least one input nbest file\n";
	exit(1);
    }

    Array<Prob> weights;
    Array<NBestList *> nbestLists;

    for (unsigned i = 0; i < nlists; i++) {

	weights[i] = atof(argv[2 * i + 1]);

	nbestLists[i] = new NBestList(vocab, maxNbest);
	assert(nbestLists[i] != 0);
	nbestLists[i]->debugme(debug);

	File input(argv[2 * i + 2], "r");
	if (!nbestLists[i]->read(input)) {
	    cerr << "format error in nbest list " <<  (i + 1) << endl;
	    exit(1);
	}
    }

    NBestList *result = mixNbestFiles(nlists, weights, nbestLists);

    if (writeNbestFile) {
	File output(writeNbestFile, "w");

	result->write(output, false);
    } else {
	result->sortHyps();
	cout << (vocab.use(), result->getHyp(0).words) << endl;
    }
 
    exit(0);
}
