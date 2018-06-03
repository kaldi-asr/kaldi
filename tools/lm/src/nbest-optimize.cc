/*
 * nbest-optimize --
 *	Optimize score combination for N-best rescoring
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2000-2012 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: nbest-optimize.cc,v 1.78 2016/05/03 05:50:25 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
# include <sstream.h>
#else
# include <iostream>
# include <sstream>
using namespace std;
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#ifndef _MSC_VER
#include <unistd.h>
#define GETPID getpid
#else
#include <process.h>
#define GETPID _getpid
#endif
#include <math.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <assert.h>

#ifdef NEED_RAND48
extern "C" {
    void srand48(long);
    double drand48();
}
#endif

#ifndef SIGALRM
#define NO_TIMEOUT
#endif

#include "option.h"
#include "version.h"
#include "Prob.h"
#include "File.h"
#include "Vocab.h"
#include "mkdir.h"

#include "NullLM.h"
#include "RefList.h"
#include "NBestSet.h"
#include "WordAlign.h"
#include "WordMesh.h"
#include "VocabDistance.h"
#include "Bleu.h"
#include "MultiwordVocab.h"	// for MultiwordSeparator

#include "Array.cc"
#include "LHash.cc"

#define DEBUG_TRAIN 1
#define DEBUG_ALIGNMENT	2
#define DEBUG_SCORES 3
#define DEBUG_RANKING 4

unsigned numScores;				/* number of score dimensions */
unsigned numFixedWeights;			/* number of fixed weights */
LHash<RefString,NBestScore **> nbestScores;	/* matrices of nbest scores,
						 * one matrix per nbest list */
LHash<RefString,WordMesh *> nbestAlignments;	/* nbest alignments */

Array<double> lambdas;				/* score weights */
Array<double> lambdaDerivs;			/* error derivatives wrt same */
Array<double> prevLambdaDerivs;
Array<double> prevLambdaDeltas;
Array<Boolean> fixLambdas;			/* lambdas to keep constant */
unsigned numRefWords;				/* number of train reference words */
unsigned numXvalWords = 0;			/* number of xval reference words */
unsigned totalError;				/* number of word errors */
double totalLoss;				/* smoothed loss */
Array<double> bestLambdas;			/* lambdas at lowest error */
unsigned bestError;				/* lowest train set error count */
unsigned bestXvalError = 0;			/* lowest xval set error count */

Array<double> lambdaSteps;			/* simplex step sizes  */
Array<double> simplex;				/* current simplex points  */

static int version = 0;
static int oneBest = 0;				/* optimize 1-best error */
static int oneBestFirst = 0;			/* 1-best then nbest error */
static int noReorder = 0;
static unsigned debug = 0;
static char *vocabFile = 0;
static int toLower = 0;
static int multiwords = 0;
static const char *multiChar = MultiwordSeparator;
static char *noiseTag = 0;
static char *noiseVocabFile = 0;
static char *hiddenVocabFile = 0;
static char *dictFile = 0;
static char *distanceFile = 0;
static char *refFile = 0;
static char *antiRefFile = 0;
static double antiRefWeight = 0.0;
static char *errorsDir = 0;
static char *nbestFiles = 0;
static char *xvalFiles = 0;
static unsigned maxNbest = 0;
static char *printHyps = 0;
static unsigned printTopN = 0;
static unsigned printUniqueHyps = 0;
static unsigned printOldRanks = 0;
static char *nbestDirectory = 0;
static char **scoreDirectories = 0;
static char *writeRoverControl = 0;
static int quickprop = 0;
static int skipopt = 0;

static double rescoreLMW = 8.0;
static double rescoreWTW = 0.0;
static double posteriorScale = 0.0;
static double posteriorScaleStep = 1.0;
static int combineLinear = 0;
static int nonNegative = 0;

static char *initLambdas = 0;
static char *initSimplex = 0;
static char *initPowell = 0;
static double alpha = 1.0;
static double epsilon = 0.1;
static double epsilonStepdown = 0.0;
static double minEpsilon = 0.0001;
static double minLoss = 0;
static double maxDelta = 1000;
static unsigned maxIters = 100000;
static double converge = 0.0001;
static unsigned maxBadIters = 10;
static unsigned maxAmoebaRestarts = 100000;
static unsigned maxTime = 0;
static unsigned srinterpFormat = 0;
static char * srinterpCountsFile = 0;

static double insertionWeight = 1.0;
static char *wordWeightFile = 0;
static LHash<VocabIndex, double> wordWeights;

// for BLEU optimization
static unsigned numReferences = 0;
static int useAvgRefLeng = 0;
static int useMinRefLeng = 0;
static int useClosestRefLeng = 0;
static unsigned optimizeBleu = 0;
static unsigned bleuRefLength = 0;
static unsigned bleuNgram = 0;
static char *bleuCountsDir = 0;
const double bleuScale = 1000000;
static double errorBleuRatio = 0;

// oracle Bleu
static unsigned oracleBleuIters = 1;
static char *printOracleHyps = 0;
static int computeOracle = 0;

// for powell quick search
Array<double> lambdaMins;
Array<double> lambdaMaxs;
Array<double> lambdaInitials;
unsigned numPowellRuns = 20;
unsigned useDynamicRandomSeries = 0;

// for srinterp format
Array<double>	  featScales;
Array<RefString>  featNames;

struct DeltaBleu {
    short correct[MAX_BLEU_NGRAM];
    short total[MAX_BLEU_NGRAM];
    int length, closestRefLeng;

    const DeltaBleu &operator += (const DeltaBleu &o) {
	for (unsigned k = 0; k < bleuNgram; k ++) {
	    correct[k] += o.correct[k];
	    total[k]   += o.total[k];
	}
	length += o.length;
        closestRefLeng += o.closestRefLeng;
	return *this;
    }
};

struct DeltaWerr {
    double numErr;
    int numWrd;

    const DeltaWerr &operator += (const DeltaWerr &o) {
	numErr += o.numErr;
	numWrd += o.numWrd;
	return *this;
    }
};

union DeltaCounts {
    DeltaBleu bleu;
    DeltaWerr werr;

    const DeltaCounts &operator += (const DeltaCounts &o) {
	if (!optimizeBleu) {
	    werr += o.werr;
	} else {
	    bleu += o.bleu;
	}
	return *this;
    }

    void clear() { memset(this, 0, sizeof(DeltaCounts)); }
};

static int optRest;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_STRING, "refs", &refFile, "reference transcripts" },

    { OPT_STRING, "nbest-files", &nbestFiles, "list of training N-best files" },
    { OPT_STRING, "xval-files", &xvalFiles, "list of cross-validation N-best files" },
    { OPT_TRUE, "srinterp-format", &srinterpFormat, "use SRInterp n-best format" },
    { OPT_STRING, "srinterp-counts", &srinterpCountsFile, "read BLEU/TER counts from a SRInterp counts file" },
    { OPT_UINT, "max-nbest", &maxNbest, "maximum number of hyps to consider" },
    { OPT_TRUE, "1best", &oneBest, "optimize 1-best error" },
    { OPT_TRUE, "1best-first", &oneBestFirst, "optimize 1-best error before full optimization" },
    { OPT_TRUE, "no-reorder", &noReorder, "don't reorder N-best hyps before aligning and align refs first" },
    { OPT_STRING, "errors", &errorsDir, "directory containing error counts" },
    { OPT_STRING, "bleu-counts", &bleuCountsDir, "directory containing bleu counts" },
    { OPT_TRUE, "averge-bleu-reference", &useAvgRefLeng, "use averge reference length for bleu brevity penalty computation (default)" },
    { OPT_TRUE, "minimum-bleu-reference", &useMinRefLeng, "use minimum reference length for bleu brevity penalty computation" },
    { OPT_TRUE, "closest-bleu-reference", &useClosestRefLeng, "use closest reference length for bleu brevity penalty computation" },
    { OPT_FLOAT, "error-bleu-ratio", &errorBleuRatio, "scale of error rate when combined with bleu for optimization" },
    { OPT_STRING, "vocab", &vocabFile, "set vocabulary" },
    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },
    { OPT_TRUE, "multiwords", &multiwords, "split multiwords in N-best hyps" },
    { OPT_STRING, "multi-char", &multiChar, "multiword component delimiter" },
    { OPT_STRING, "noise", &noiseTag, "noise tag to skip" },
    { OPT_STRING, "noise-vocab", &noiseVocabFile, "noise vocabulary to skip" },
    { OPT_STRING, "hidden-vocab", &hiddenVocabFile, "subvocabulary to be kept separate in mesh alignment" },
    { OPT_STRING, "dictionary", &dictFile, "dictionary to use in mesh alignment" },
    { OPT_STRING, "distances", &distanceFile, "word distance matrix to use in mesh alignment" },
    { OPT_FLOAT, "insertion-weight", &insertionWeight, "relative weight of insertion errors" },
    { OPT_STRING, "word-weights", &wordWeightFile, "word weights for error computation" },
    { OPT_STRING, "anti-refs", &antiRefFile, "anti-reference transcripts (for decorrelation)" },
    { OPT_FLOAT, "anti-ref-weight", &antiRefWeight, "anti-reference error weight" },
    { OPT_STRING, "write-rover-control", &writeRoverControl, "nbest-rover control output file" },
    { OPT_FLOAT, "rescore-lmw", &rescoreLMW, "rescoring LM weight" },
    { OPT_FLOAT, "rescore-wtw", &rescoreWTW, "rescoring word transition weight" },
    { OPT_FLOAT, "posterior-scale", &posteriorScale, "divisor for log posterior estimates" },
    { OPT_TRUE, "combine-linear", &combineLinear, "combine scores linearly (not log-linearly" },
    { OPT_TRUE, "non-negative", &nonNegative, "limit search to non-negative weights" },
    { OPT_STRING, "init-lambdas", &initLambdas, "initial lambda values" },
    { OPT_STRING, "init-amoeba-simplex", &initSimplex, "initial amoeba simplex points" },
    { OPT_STRING, "init-powell-range", &initPowell, "initial powell weight range" },
    { OPT_UINT, "num-powell-runs", &numPowellRuns, "number of random runs for quick powell grid search (default: 20)" },
    { OPT_TRUE, "-dynamic-random-series", &useDynamicRandomSeries, "use dynamic random series for powell search (result not repeatable)" },
    { OPT_TRUE, "compute-oracle", &computeOracle, "find best possible hyps in n-best list"},
    { OPT_UINT, "oracle-bleu-iters", &oracleBleuIters, "number of iterations to compute oracle Bleu value (default: 1)" },
    { OPT_STRING, "print-oracle-hyps", &printOracleHyps, "output file for oracle hyps"},
    { OPT_FLOAT, "alpha", &alpha, "sigmoid slope parameter" },
    { OPT_FLOAT, "epsilon", &epsilon, "learning rate parameter" },
    { OPT_FLOAT, "epsilon-stepdown", &epsilonStepdown, "epsilon step-down factor" },
    { OPT_FLOAT, "min-epsilon", &minEpsilon, "minimum epsilon after step-down" },
    { OPT_FLOAT, "min-loss", &minLoss, "samples with loss below this are ignored" },
    { OPT_FLOAT, "max-delta", &maxDelta, "threshold to filter large deltas" },
    { OPT_UINT, "maxiters", &maxIters, "maximum number of learning iterations" },
    { OPT_UINT, "max-bad-iters", &maxBadIters, "maximum number of iterations without improvement" },
    { OPT_UINT, "max-amoeba-restarts", &maxAmoebaRestarts, "maximum number of Amoeba restarts" },
#ifndef NO_TIMEOUT
    { OPT_UINT, "max-time", &maxTime, "abort search after this many seconds" },
#endif
    { OPT_FLOAT, "converge", &converge, "minimum relative change in objective function" },
    { OPT_STRING, "print-hyps", &printHyps, "output file or directory (when print-top-n specified) for final top hyps" },
    { OPT_TRUE, "skipopt",  &skipopt, "skip optimization (useful if you only want to print top hyps)" },
    { OPT_UINT, "print-top-n", &printTopN, "output top N rescored hypotheses" },
    { OPT_TRUE, "print-unique-hyps", &printUniqueHyps, "output unique hyptheses" },
    { OPT_TRUE, "print-old-ranks", &printOldRanks, "output old ranks of hypotheses before rescoring, instead of probabilities" },
    { OPT_TRUE, "quickprop", &quickprop, "use QuickProp gradient descent" },
    { OPT_UINT, "debug", &debug, "debugging level" },
    { OPT_REST, "-", &optRest, "indicate end of option list" },
};

static Boolean abortSearch = false;

#ifndef NO_TIMEOUT
/*
 * deal with different signal hander types
 */
#ifndef _sigargs
#define _sigargs int
#endif

void catchAlarm(_sigargs)
{
    abortSearch = true;
}
#endif /* !NO_TIMEOUT */

double
sigmoid(double x)
{
    return 1/(1 + exp(- alpha * x));
}

void
dumpScores(ostream &str, NBestSet &nbestSet)
{
    NBestSetIter iter(nbestSet);
    NBestList *nbest;
    RefString id;

    while ((nbest = iter.next(id))) {
	str << "id = " << id << endl;

	NBestScore ***scores = nbestScores.find(id);

	if (!scores) {
	    str << "no scores found!\n";
	} else {
	    for (unsigned j = 0; j < nbest->numHyps(); j ++) {
		str << "Hyp " << j << ":" ;
		for (unsigned i = 0; i < numScores; i ++) {
		    str << " " << (*scores)[i][j];
		}
		str << endl;
	    }
	}
    }
}

void
dumpAlignment(ostream &str, WordMesh &alignment)
{
    for (unsigned pos = 0; pos < alignment.length(); pos ++) {
	Array<HypID> *hypMap;
	VocabIndex word;

	str << "position " << pos << endl;

	WordMeshIter iter(alignment, pos);
	while ((hypMap = iter.next(word))) {
	    str << "  word = " << alignment.vocab.getWord(word) << endl;

	    for (unsigned k = 0; k < hypMap->size(); k ++) {
		str << " " << (*hypMap)[k];
	    }
	    str << endl;
	}
    }
}

/*
 * compute hypothesis score (weighted sum of log scores)
 */
LogP
hypScore(unsigned hyp, NBestScore **scores)
{
    static NBestScore **lastScores = 0;
    static Array<LogP> *cachedScores = 0;

    if (scores != lastScores) {
	delete cachedScores;
	cachedScores = new Array<LogP>;
	assert(cachedScores != 0);
	lastScores = scores;
    }

    if (hyp < cachedScores->size()) {
	if ((*cachedScores)[hyp] != 0.0) {
	    return (*cachedScores)[hyp];
	}
    } else {
	for (unsigned j = cachedScores->size(); j < hyp; j ++) {
	    (*cachedScores)[j] = 0.0;
	}
    }

    LogP score;

    double *weights = lambdas.data(); /* bypass index range check for speed */

    if (combineLinear) {
	/* linear combination, even though probabilities are encoded as logs */
	Prob prob = 0.0;
	for (unsigned i = 0; i < numScores; i ++) {
	    prob += weights[i] * LogPtoProb(scores[i][hyp]);
	}
	score = ProbToLogP(prob);
    } else {
	/* log-linear combination */
	score = 0.0;
	for (unsigned i = 0; i < numScores; i ++) {
	    score += weightLogP(weights[i], scores[i][hyp]);
	}
    }
    return ((*cachedScores)[hyp] = score);
}

/*
 * compute summed hyp scores (sum of unnormalized posteriors of all hyps
 *	containing a word)
 *	isCorrect is set to true if hyps contains the reference (refID)
 *	The last parameter is used to collect auxiliary sums needed for
 *	derivatives
 */
Prob
wordScore(Array<HypID> &hyps, NBestScore **scores, Boolean &isCorrect,
								Prob *a = 0)
{
    Prob totalScore = 0.0;
    isCorrect = false;

    if (a != 0) {
	for (unsigned i = 0; i < numScores; i ++) {
	    a[i] = 0.0;
	}
    }

    for (unsigned k = 0; k < hyps.size(); k ++) {
	if (hyps[k] == refID) {
	    /*
	     * This hyp represents the correct word string, but doesn't 		     
             * contribute to the posterior probability for the word.
	     */
	    isCorrect = true;
	} else {
	    Prob score = LogPtoProb(hypScore(hyps[k], scores));

	    totalScore += score;
	    if (a != 0) {
		for (unsigned i = 0; i < numScores; i ++) {
		    a[i] += weightLogP(score, scores[i][hyps[k]]);
		}
	    }
	}
    }

    return totalScore;
}


/*
 * compute loss and derivatives for a single nbest list
 */
void
computeDerivs(RefString id, NBestScore **scores, WordMesh &alignment)
{
    /* 
     * process all positions in alignment
     */
    for (unsigned pos = 0; pos < alignment.length(); pos++) {
	VocabIndex corWord = Vocab_None;
	Prob corScore = 0.0;
	Array<HypID> *corHyps;

	VocabIndex bicWord = Vocab_None;
	Prob bicScore = 0.0;
	Array<HypID> *bicHyps = 0;

	if (debug >= DEBUG_RANKING) {
	    cerr << "   position " << pos << endl;
	}

	WordMeshIter iter(alignment, pos);

	Array<HypID> *hypMap;
	VocabIndex word;
	while ((hypMap = iter.next(word))) {
	    /*
	     * compute total score for word and check if it's the correct one
	     */
	    Boolean isCorrect;
	    Prob totalScore = wordScore(*hypMap, scores, isCorrect);

	    if (isCorrect) {
		corWord = word;
		corScore = totalScore;
		corHyps = hypMap;
	    } else {
		if (bicWord == Vocab_None || bicScore < totalScore) {
		    bicWord = word;
		    bicScore = totalScore;
		    bicHyps = hypMap;
		}
	    }
	}

	/*
	 * There must be a correct hyp
	 */
	assert(corWord != Vocab_None);

	if (debug >= DEBUG_RANKING) {
	    cerr << "      cor word = " << alignment.vocab.getWord(corWord)
		 << " score = " << corScore << endl;
	    cerr << "      bic word = " << (bicWord == Vocab_None ? "NONE" :
					    alignment.vocab.getWord(bicWord))
		 << " score = " << bicScore << endl;
	}

	unsigned wordError = (bicScore > corScore);
	double smoothError = 
			sigmoid(ProbToLogP(bicScore) - ProbToLogP(corScore));

	totalError += wordError;
	totalLoss += smoothError;

	/*
	 * If all word hyps are correct or incorrect, or loss is below a set
	 * threshold, then this sample cannot help us and we exclude it from
	 * the derivative computation
	 */
	if (bicScore == 0.0 || corScore == 0.0 || smoothError < minLoss) {
	    continue;
	}

	assert(bicHyps != 0);

	/*
	 * Compute the auxiliary vectors for derivatives
	 */
	Boolean dummy;
	makeArray(Prob, corA, numScores);
	wordScore(*corHyps, scores, dummy, corA);

	makeArray(Prob, bicA, numScores);
	wordScore(*bicHyps, scores, dummy, bicA);

	/*
	 * Accumulate derivatives
	 */
	double sigmoidDeriv = alpha * smoothError * (1 - smoothError);

	for (unsigned i = 0; i < numScores; i ++) {
	    double delta = (bicA[i] / bicScore - corA[i] / corScore);

	    if (fabs(delta) > maxDelta) {
		cerr << "skipping large delta " << delta
		     << " at id " << id
		     << " position " << pos
		     << " score " << i
		     << endl;
	    } else {
		lambdaDerivs[i] += sigmoidDeriv * delta;
	    }
	}
    }
}

/*
 * do a single pass over all nbest lists, computing loss function
 * and accumulating derivatives for lambdas.
 */
void
computeDerivs(NBestSet &nbestSet)
{
    /*
     * Initialize error counts and derivatives
     */
    totalError = 0;
    totalLoss = 0.0;

    for (unsigned i = 0; i < numScores; i ++) {
	lambdaDerivs[i] = 0.0;
    }

    NBestSetIter iter(nbestSet);
    NBestList *nbest;
    RefString id;

    while ((nbest = iter.next(id))) {
	NBestScore ***scores = nbestScores.find(id);
	assert(scores != 0);
	WordMesh **alignment = nbestAlignments.find(id);
	assert(alignment != 0);

	computeDerivs(id, *scores, **alignment);
    }
}

/*
 * accumulate bleu counts
 */
void
accumulateBleuCounts(RefString id, NBestScore **scores, NBestList &nbest, 
                     unsigned *correct, unsigned *total, unsigned &length, 
		     unsigned &closestRefLeng)
{
    unsigned numHyps = nbest.numHyps();
    unsigned bestHyp;
    LogP bestScore;

    if (numHyps == 0) {
	return;
    }

    if (id == 0 && scores == 0) {

        bestHyp = 0;

    } else {

        /*
	 * Find hyp with highest score
	 */
        unsigned i;
	for (i = 0; i < numHyps; i ++) {
	    LogP score = hypScore(i, scores);

	    if (i == 0 || score > bestScore) {
	        bestScore = score;
		bestHyp = i;
	    }
	}
    }

    NBestHyp &h = nbest.getHyp(bestHyp);
    length += h.numWords;
    closestRefLeng += h.closestRefLeng;

    for (unsigned i = 0; i < bleuNgram; i++) {
        correct[i] += h.bleuCount->correct[i];
        unsigned n = h.numWords;
        if (n > i) {
	    n -= i;
        } else {
	    n = 0;
	}

        total[i] += n;      
    }
}

/*
 * compute 1-best word error for a single nbest list
 * Note: uses global lambdas variable (yuck!)
 */
double
compute1bestErrors(RefString id, NBestScore **scores, NBestList &nbest)
{
    unsigned numHyps = nbest.numHyps();
    unsigned bestHyp;
    LogP bestScore;

    if (numHyps == 0) {
	return 0.0;
    }

    /*
     * Find hyp with highest score
     */
    for (unsigned i = 0; i < numHyps; i ++) {
	LogP score = hypScore(i, scores);

	if (i == 0 || score > bestScore) {
	    bestScore = score;
	    bestHyp = i;
	}
    }

    return nbest.getHyp(bestHyp).numErrors;
}

/*
 * Compute error contribution of a single word confusion (or insertion/deletion)
 */
double
computeWordConfusionError(WordMesh &alignment, VocabIndex ref, VocabIndex hyp)
{
    if (ref == hyp) {
    	return 0.0;
    } else if (wordWeightFile) {
    	// weighted insertion/deletion errors:
	// a substitution counts as the sum of deletion and an insertion
	// default weight for unknown words is 1

	double error = 0.0;

	if (ref != alignment.deleteIndex) {
	    double *refWeight = wordWeights.find(ref);

	    error += (refWeight ? *refWeight : 1.0);
	}

	if (hyp != alignment.deleteIndex) {
	    double *hypWeight = wordWeights.find(hyp);

	    error += insertionWeight * (hypWeight ? *hypWeight : 1.0);
	}

	return error;

    } else {
    	// traditional word error
	if (ref == alignment.deleteIndex) {
	    // insertion error;
	    return insertionWeight;
	} else if (hyp == alignment.deleteIndex) {
	    // deletion error;
	    return 1.0;
	} else {
	    // substitution error
	    return 1.0;
	}
    }
}

/*
 * compute sausage word error for a single nbest list
 * Note: uses global lambdas variable (yuck!)
 */
double
computeSausageErrors(RefString id, NBestScore **scores, WordMesh &alignment)
{
    double result = 0.0;

    /* 
     * process all positions in alignment
     */
    for (unsigned pos = 0; pos < alignment.length(); pos++) {
	VocabIndex corWord = Vocab_None;
	Prob corScore = 0.0;

	VocabIndex bicWord = Vocab_None;
	Prob bicScore = 0.0;

	if (debug >= DEBUG_RANKING) {
	    cerr << "   position " << pos << endl;
	}

	WordMeshIter iter(alignment, pos);
	
	Array<HypID> *hypMap;
	VocabIndex word;
	while ((hypMap = iter.next(word))) {
	    /*
	     * compute total score for word and check if it's the correct one
	     */
	    Boolean isCorrect;
	    Prob totalScore = wordScore(*hypMap, scores, isCorrect);
	    
	    if (isCorrect) {
		corWord = word;
		corScore = totalScore;
	    } else {
		if (bicWord == Vocab_None || bicScore < totalScore) {
		    bicWord = word;
		    bicScore = totalScore;
		}
	    }

	    if (corWord != Vocab_None &&
		bicWord != Vocab_None &&
		bicScore > corScore)
	    {
		result +=
			computeWordConfusionError(alignment, corWord, bicWord);
		break;
	    }
	}
    }

    return result;
}

/*
 * Compute word error/1-bleu for vector of weights
 */
double
computeErrors(NBestSet &nbestSet, double *weights)
{
    double result = 0.0;

    Array<double> savedLambdas;

    unsigned i;
    for (i = 0; i < numScores; i ++) {
        savedLambdas[i] = lambdas[i];
        lambdas[i] = weights[i];
    }

    NBestSetIter iter(nbestSet);
    NBestList *nbest;
    RefString id;

    if (!optimizeBleu) {
	while ((nbest = iter.next(id))) {
	    NBestScore ***scores = nbestScores.find(id);
	    assert(scores != 0);

	    if (oneBest) {
		result += (int) compute1bestErrors(id, *scores, *nbest);
	    } else {
		WordMesh **alignment = nbestAlignments.find(id);
		assert(alignment != 0);

		result += computeSausageErrors(id, *scores, **alignment);
	    }
	}
    } else {
        unsigned correct[MAX_BLEU_NGRAM], total[MAX_BLEU_NGRAM], length = 0;
	unsigned closestRefLeng = 0;

	for (i = 0; i < bleuNgram; i++) {
	    correct[i] = 0;
	    total[i] = 0;
	}

	while ((nbest = iter.next(id))) {
	    NBestScore ***scores = nbestScores.find(id);
	    assert(scores != 0);

	    accumulateBleuCounts(id, *scores, *nbest, correct, total, length, closestRefLeng);
	    
	    if (errorBleuRatio != 0) {
	        result += compute1bestErrors(id, *scores, *nbest);
	    }
	} 

	if (useClosestRefLeng) 
	  bleuRefLength = closestRefLeng;

	double bleu =
		computeBleu(bleuNgram, correct, total, length, bleuRefLength);

	if (errorBleuRatio != 0)
	  result = ((1.0 - bleu) + (result / bleuRefLength) * errorBleuRatio) * bleuScale;
	else
	  result = (1.0 - bleu) * bleuScale;
    }

    for (i = 0; i < numScores; i ++) {
        lambdas[i] = savedLambdas[i];
    }

    return result;
}

/*
 * print lambdas, and optionally write nbest-rover control file
 */
void
printLambdas(ostream &str, Array<double> &lambdas, const char *controlFile = 0)
{
    unsigned i;
    double normalizer = 0.0;

    if (!optimizeBleu) {
	str << "   weights =";
	for (i = 0; i < numScores; i ++) {
	    if (normalizer == 0.0 && lambdas[i] != 0.0) {
		normalizer = lambdas[i];
	    }

	    str << " " << lambdas[i];
	}
        str << endl;
    } else {
	normalizer = 1.0;
    }

    str << "   normed =";
    for (i = 0; i < numScores; i ++) {
	str << " " << lambdas[i]/normalizer;
    }
    str << endl;

    str << "   scale = " << 1/normalizer
	<< endl;

    if (controlFile) {

        File file(controlFile, "w");

	if (srinterpFormat) {

	    double maxScale = 0;
	    for (i = 0; i < numScores; i++) {
	        if  (maxScale < featScales[i])
		  maxScale = featScales[i];
	    }
	    
	    for (i = 0; i < numScores; i++) {
	        file.fprintf("%s:%lg ", featNames[i], (lambdas[i] / featScales[i]) * maxScale);
	    }
	    file.fprintf("\n");

	} else {
	    /*
	     * write additional score dirs and weights
	     */
	    for (i = 3; i < numScores; i ++) {
	        file.fprintf("%s\t%lg +\n",
			scoreDirectories[i-3],
			lambdas[i]/normalizer);
	    }

	    /*
	     * write main score dir and weights
	     */
	    if (!optimizeBleu) {
	        file.fprintf("%s\t%lg %lg 1.0 %d %lg\n",
			nbestDirectory,
			lambdas[1]/normalizer,
			lambdas[2]/normalizer,
			maxNbest,
			/*
 			 * In -1best mode, output a dummy posterior scale
 			 * that simulates one-best decoding.
 			 */
			oneBest ? 0.1 : 1/normalizer);
	    } else {
	        file.fprintf("%s\t%lg %lg 1.0 %d %lg\n",
			nbestDirectory,
			lambdas[1]/normalizer,
			lambdas[2]/normalizer,
			maxNbest,
			lambdas[0]/normalizer);          
	    }
	}
    }
}

/*
 * One step of gradient descent on loss function
 */
void
updateLambdas()
{
    static Boolean havePrev = false;

    for (unsigned i = 0; i < numScores; i ++) {
	if (!fixLambdas[i]) {
	    double delta;

	    if (!havePrev || !quickprop ||
		lambdaDerivs[i]/prevLambdaDerivs[i] > 0)
	    {
		delta = - epsilon * lambdaDerivs[i] / numRefWords;
	    } else {
		/*
		 * Use QuickProp update rule
		 */
		delta = prevLambdaDeltas[i] * lambdaDerivs[i] /
			    (prevLambdaDerivs[i] - lambdaDerivs[i]);
	    }
	    lambdas[i] += delta;

	    prevLambdaDeltas[i] = delta;
	    prevLambdaDerivs[i] = lambdaDerivs[i];
	}
    }
	
    havePrev = true;
}

/*
 * Iterate gradient descent on loss function
 */
void
train(NBestSet &nbestSet, NBestSet &xvalSet)
{
    unsigned iter = 0;
    unsigned badIters = 0;
    double oldLoss = 0.0;
    unsigned xvalError;

    if (debug >= DEBUG_TRAIN) {
	printLambdas(cerr, lambdas);
    }

    while (iter++ < maxIters) {
	/*
	 * Compute the xval error before the training set, since this also
	 * updates the derivatives and we do gradient descent on the training set
	 */
	if (numXvalWords) {
	    computeDerivs(xvalSet);
	    xvalError = totalError;
	}

	computeDerivs(nbestSet);

	if (iter > 1 && fabs(totalLoss - oldLoss)/oldLoss < converge) {
	    cerr << "stopping due to convergence\n";
	    break;
	}

	if (debug >= DEBUG_TRAIN) {
	    cerr << "iteration " << iter << ":"
		 << " errors = " << totalError
		 << " (" << ((double)totalError/numRefWords) << "/word)"
		 << " loss = " << totalLoss
		 << " (" << (totalLoss/numRefWords) << "/word)";
	    if (numXvalWords) {
		cerr << " xval errors = " << xvalError
		     << " (" << ((double)xvalError/numXvalWords) << "/word)";
	    }
	    cerr << endl;
	}

	if (iter == 1 || (isfinite(totalLoss) && totalError < bestError)) {
	    cerr << "NEW BEST ERROR: " << totalError 
		 << " (" << ((double)totalError/numRefWords) << "/word)";

	    if (numXvalWords) {
		cerr <<  " XVAL ERROR: " << xvalError
		     << " (" << ((double)xvalError/numXvalWords) << "/word)";

		if (iter > 1 && xvalError > bestXvalError) {
		    cerr << "\nstopping due to cross-validation\n";
		    cerr << "bestXvalError = " << bestXvalError << endl;
		    break;
		}
	    }
	    cerr << endl;

	    printLambdas(cerr, lambdas, writeRoverControl);

	    bestError = totalError;
	    if (numXvalWords) {
		bestXvalError = xvalError;
	    }
	    bestLambdas = lambdas;
	    badIters = 0;

#ifndef NO_TIMEOUT
	    if (maxTime) {
		alarm(maxTime);
	    }
#endif /* !NO_TIMEOUT */
	} else {
	    badIters ++;
	}

	if (abortSearch) {
	    cerr << "search timed out after " << maxTime << " seconds\n";
	    break;
	}

	if (badIters > maxBadIters || !isfinite(totalLoss)) {
	    if (epsilonStepdown > 0.0) {
		epsilon *= epsilonStepdown;
		if (epsilon < minEpsilon) {
		    cerr << "minimum epsilon reached\n";
		    break;
		}
		cerr << "setting epsilon to " << epsilon
		     << " due to lack of error decrease\n";

		/*
		 * restart descent at last best point, and 
		 * disable QuickProp for the next iteration
		 */
		prevLambdaDerivs = lambdaDerivs;	
		lambdas = bestLambdas;
		badIters = 0;
	    } else {
		cerr << "stopping due to lack of error decrease\n";
		break;
	    }
	} else {
	    updateLambdas();
	}

	if (debug >= DEBUG_TRAIN) {
	    printLambdas(cerr, lambdas);
	}

	oldLoss = totalLoss;
    }
}

/*
 * Evaluate a single point in the (unconstrained) parameter space
 */
double
amoebaComputeErrors(NBestSet &nbestSet, NBestSet &xvalSet, double *p)
{
    int i, j;
    Array <double> weights;

    if (p[0] < 0.5) {
	/*
	 * This prevents posteriors to go through the roof, leading 
	 * to numerical problems.  Since the scaling of posteriors is 
	 * a redundant dimension this doesn't constrain the result.
	 */
      if (!optimizeBleu)
	return numRefWords;
      else
        return bleuScale;
    }

    for (unsigned i = 0, j = 1; i < numScores; i++) {
	if (fixLambdas[i]) {
	    weights[i] = lambdas[i] / p[0];
	} else {
	    weights[i] = p[j] / p[0];
	    j++;
	}

	/*
	 * Check for negative weights if -non-negative is in effect.
	 * Return large error count for disallowed values.
	 */
	if (nonNegative && weights[i] < 0.0) {
	    return numRefWords;
	}
    }

    double error = computeErrors(nbestSet, weights.data());

    if (error < bestError) {
	double xvalError;

	if (numXvalWords) {
	    xvalError = computeErrors(xvalSet, weights.data());
	}

	cerr << "NEW BEST ERROR: " << error
	     << " (" << ((double)error/numRefWords) << "/word)";
	if (numXvalWords) {
	    cerr << " XVAL ERROR: " << xvalError
		 << " (" << ((double)xvalError/numXvalWords) << "/word)";

	    if (xvalError > bestXvalError) {
		/*
		 * Ignore this new point by returning a large error
		 */
		cerr << " IGNORED\n";
		return numRefWords;
	    }
	}
	cerr << endl;
	printLambdas(cerr, weights, writeRoverControl);

	bestError = (int) error;
	if (numXvalWords) {
	    bestXvalError = (int) xvalError;
	}
	bestLambdas = weights;

#ifndef NO_TIMEOUT
	if (maxTime) {
	    alarm(maxTime);
	}
#endif /* !NO_TIMEOUT */
    }

    return error;
}

/*
 * Try moving a single simplex corner
 */
double
amoebaEval(NBestSet &nbest, NBestSet &xval,
	   double **p, double *y, double *psum, unsigned ndim,
	   double (*funk) (NBestSet &, NBestSet &, double[]), unsigned ihi, double fac)
{
    makeArray(double, ptry, ndim);
    double fac1 = (1.0 - fac) / ndim;
    double fac2 = fac1 - fac;

    for (unsigned j = 0; j < ndim; j++) {
	ptry[j] = psum[j] * fac1 - p[ihi][j] * fac2;
    }
    double ytry = (*funk) (nbest, xval, ptry);
    if (ytry < y[ihi]) {
	y[ihi] = ytry;
	for (unsigned j = 0; j < ndim; j++) {
	    psum[j] += ptry[j] - p[ihi][j];
	    p[ihi][j] = ptry[j];
	}
    }
    return ytry;
}

inline void
computeSum(unsigned ndim, unsigned mpts, double *psum, double **p)
{
    for (unsigned j = 0; j < ndim; j++) {
	double sum = 0.0;
	for (unsigned i = 0; i < mpts; i++) {
	    sum += p[i][j];
	}
	psum[j] = sum;
    }
}

inline void
swap(double &a, double &b)
{
    double h = a;
    a = b;
    b = h;
}

/*
 * Run Amoeba optimization
 */
void
amoeba(NBestSet &nbest, NBestSet &xval, double **p, double *y, unsigned ndim, double ftol,
       double (*funk) (NBestSet &, NBestSet &, double[]), unsigned &nfunk)
{
    unsigned ihi, inhi, mpts = ndim + 1;
    makeArray(double, psum, ndim);

    if (debug >= DEBUG_TRAIN) {
	cerr << "Starting amoeba with " << ndim << " dimensions" << endl;
    }

    computeSum(ndim, mpts, psum, p);
    
    double rtol = 10000.0;

    unsigned ilo = 0;
    unsigned unchanged = 0;

    while (true) {
	double ysave, ytry;
	double ylo_pre = y[ilo];

	ilo = 0;

	ihi = y[0] > y[1] ? (inhi = 1, 0) : (inhi = 0, 1);
	for (unsigned i = 0; i < mpts; i++) {
	    if (y[i] <= y[ilo]) {
		ilo = i;
	    }
	    if (y[i] > y[ihi]) {
		inhi = ihi;
		ihi = i;
	    } else if (y[i] > y[inhi] && i != ihi)
		inhi = i;
	}

	if (debug >= DEBUG_TRAIN) {
	    cerr << "Current low " << y[ilo] << ": ";
	    cerr << "Current high " << y[ihi] << ":";
	    /*
	     * for (unsigned j=0; j<ndim; j++)
	     *	   cerr << " " << p[ihi][j] ; cerr << endl; 
	     */
	    cerr << "Current next high " << y[inhi] << endl;
	}

	double denom = fabs(y[ihi]) + fabs(y[ilo]);
	if (denom == 0.0) {
	    rtol = 0.0;
	} else {
	    rtol = 2.0 * fabs(y[ihi] - y[ilo]) / denom;
	}

	if (ylo_pre == y[ilo] && rtol < converge) {
	    unchanged++;
	} else {
	    unchanged = 0;
	    if (ylo_pre > y[ilo]) {
		int k;

		if (debug >= DEBUG_TRAIN) {
		    cerr << "scale " << p[ilo][0] << endl;
		    for (unsigned j = 1, k = 0; k < numScores && j < ndim; k++)
		    {
			if (!fixLambdas[k])
			    cerr << "lambda_" << j - 1
				 << " " << p[ilo][j] << endl;
			    j ++;
		    }
		}
	    }
	}

	if (unchanged > maxBadIters) {
	    swap(y[0], y[ilo]);
	    for (unsigned i = 0; i < ndim; i++) {
		swap(p[0][i], p[ilo][i]);
	    }
	    break;
	}

	if (debug >= DEBUG_TRAIN) {
	    cerr << " fractional range " << rtol << endl;
	    cerr << " limit range " << ftol << endl;
	}

	if (rtol <= ftol) {
	    swap(y[0], y[ilo]);
	    for (unsigned i = 0; i < ndim; i++) {
		swap(p[0][i], p[ilo][i]);
	    }
	    break;
	}

	if (abortSearch) {
	    break;
	}

	nfunk += 1;

	// Try a reflection
	ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, -1.0);
	if (debug >= DEBUG_TRAIN) {
	    cerr << " Reflected amoeba returned " << ytry << endl;
	}

	// If successful try more
	if (ytry <= y[ilo]) {
	    nfunk += 1;
	    ysave = ytry;
	    ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, 1.5);
	    if (debug >= DEBUG_TRAIN) {
		cerr << " Expanded amoeba by 1.5 returned " << ytry <<
		    endl;
	    }

	    // If successful try more
	    if (ytry <= ysave) {
		ysave = ytry;
		nfunk += 1;
		ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, 2.0);
		if (debug >= DEBUG_TRAIN) {
		    cerr << " Expanded amoeba by 2.0 returned " << ytry <<
			endl;
		}
	    }

	    // If successful try more
	    if (ytry <= ysave) {
		nfunk += 1;
		ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, 3.0);
		if (debug >= DEBUG_TRAIN) {
		    cerr << " Expanded amoeba by 3.0 returned " << ytry << endl;
		}
	    }
	} else if (ytry >= y[inhi]) {
	    // If failed shrink
	    ysave = y[ihi];

	    // shrink half
	    nfunk += 1;
	    ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, 0.7);
	    if (debug >= DEBUG_TRAIN) {
		cerr << " Shrunken amoeba by 0.7 returned " << ytry << endl;
	    }

	    // try again opposite direction
	    if (ytry >= ysave) {
		nfunk += 1;
		ytry =
		    amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, -0.7);
		if (debug >= DEBUG_TRAIN) {
		    cerr << " Shrunken reflected amoeba by -0.7 returned "
			 << ytry << endl;
		}
	    }

	    // try again 
	    if (ytry >= ysave) {
		nfunk += 1;
		ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, 0.5);
		if (debug >= DEBUG_TRAIN) {
		    cerr << " Shrunken amoeba by 0.5 returned " << ytry << endl;
		}
	    }

	    // try again opposite direction
	    if (ytry >= ysave) {
		nfunk += 1;
		ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, -0.5);
		if (debug >= DEBUG_TRAIN) {
		    cerr << " Shrunken reflected amoeba by -0.5 returned "
			 << ytry << endl;
		}
	    }
	    if (ytry >= ysave) {
		nfunk += 1;
		ytry = amoebaEval(nbest, xval, p, y, psum, ndim, funk, ihi, 0.3);
		if (debug >= DEBUG_TRAIN) {
		    cerr << " Shrunken amoeba by 0.3 returned " << ytry << endl;
		}
	    }

	    // if failed to get rid of high contract everything by 0.7
	    if (ytry >= ysave) {
		for (unsigned i = 0; i < mpts; i++) {
		    if (i != ilo) {
			for (unsigned j = 0; j < ndim; j++)
			    p[i][j] = psum[j] =
				0.7 * p[i][j] + 0.3 * p[ilo][j];
			y[i] = (*funk) (nbest, xval, psum);
		    }
		}
		nfunk += ndim;
		computeSum(ndim, mpts, psum, p);
	    }
	} else {
	    --nfunk;
	}
    }
}

/*
 * Amoeba optimization with restarts
 */
void
trainAmoeba(NBestSet &nbestSet, NBestSet &xvalSet)
{
    // Before training reset the lambdas to their unscaled version
    for (unsigned i = 0; i < numScores; i++) {
	lambdas[i] *= posteriorScale;
    }

    // Initialize ameoba points
    // There is one search dimension per score dimension, excluding fixed
    // weights, but adding one for the posterior score (which is stored in
    // the vector even if it is kept fixed).
    unsigned numFreeWeights = numScores - numFixedWeights + 1;
    assert((int)numFreeWeights > 0);

    makeArray(double *, points, numFreeWeights + 1);
    for (unsigned i = 0; i <= numFreeWeights; i++) {
	points[i] = new double[numFreeWeights];
	assert(points[i] != 0);
    }

    makeArray(double, errs, numFreeWeights + 1);
    makeArray(double, prevPoints, numFreeWeights);

    prevPoints[0] = points[0][0] = posteriorScale;
    simplex[0] = posteriorScaleStep;
    for (unsigned i = 0, j = 1; i < numScores; i++) {
	if (!fixLambdas[i]) {
	    prevPoints[j] = points[0][j] = lambdas[i];
	    simplex[j] = lambdaSteps[i];
	    j++;
	}
    }

    makeArray(double *, dir, numFreeWeights + 1);
    for (unsigned i = 0; i <= numFreeWeights; i++) {
	dir[i] = new double[numFreeWeights];
	assert(dir[i] != 0);
	for (unsigned j = 0; j < numFreeWeights; j++) {
	    dir[i][j] = 0.0;
	}
    }

    unsigned nevals = 0;
    unsigned loop = 1;

    unsigned same = 0;
    unsigned shift = 0;
    unsigned reps = 0;

    /* force an improvement */
    bestError = (unsigned)-1;
    bestXvalError = (unsigned)-1;

    while (loop) {
	reps++;

	for (unsigned i = 1; i <= numFreeWeights; i++) {
	    unsigned k = 0;
	    dir[i][k] += (((k + loop + shift - 1) % numFreeWeights) + 1 == i) ?
			    loop * simplex[k] : 0.0;
	    k++;
	    for (unsigned j = 0; j < numScores; j++) {
		if (!fixLambdas[j]) {
		    dir[i][k] +=
			(((k + loop + shift - 1) % numFreeWeights) + 1 == i) ?
			    loop * simplex[k] : 0.0;
		    k++;
		}
	    }
	}

	if (debug >= DEBUG_TRAIN) {
	    cerr << "Simplex points:" << endl;
	}

	for (unsigned i = 0; i <= numFreeWeights; i++) {
	    for (unsigned j = 0; j < numFreeWeights; j++) {
		points[i][j] = points[0][j] + dir[i][j];
	    }
	    errs[i] = amoebaComputeErrors(nbestSet, xvalSet, points[i]);

	    if (debug >= DEBUG_TRAIN) {
		cerr << "Point " << i << " : ";

		for (unsigned j = 0; j < numFreeWeights; j++) {
		    cerr << points[i][j] << " ";
		}
		cerr << "errors = " << errs[i] << endl;
	    }
	}

	long prevErrors = (int) errs[0];

	/*
	 * The objective function fractional tolerance is
	 * decreasing with each retry.
	 */
	amoeba(nbestSet, xvalSet, points, errs, numFreeWeights, converge / reps,
					       amoebaComputeErrors, nevals);

	if ((int) errs[0] < prevErrors) {
	    loop++;
	    same = 0;
	} else if (same < numFreeWeights) {
	    loop++;
	    same++;
	} else {
	    loop = 0;
	}

	if (loop > numFreeWeights / 3) {
	    loop = 1;
	    for (unsigned i = 0; i <= numFreeWeights; i++) {
		for (unsigned j = 0; j < numFreeWeights; j++) {
		    dir[i][j] = 0.0;
		}
	    }
	    shift++;
	}

	// reset step sizes  
	posteriorScale = points[0][0];
	if (loop == 1) {
	    simplex[0] = posteriorScaleStep;
	} else if (fabs(prevPoints[0] - points[0][0])
					> 1.3 * fabs(simplex[0]))
	{
	    simplex[0] = points[0][0] - prevPoints[0]; 
	} else {
	    simplex[0] = simplex[0] * 1.3;
	}
	prevPoints[0] = points[0][0];

	unsigned j = 1;
	for (unsigned i = 0; i < numScores; i++) {
	    if (!fixLambdas[i]) {
		lambdas[i] = points[0][j];
		if (loop == 1) {
		    simplex[j] = lambdaSteps[i];
		} else if (fabs(prevPoints[j] - points[0][j])
					> 1.3 * fabs(simplex[j]))
		{
		    simplex[j] = points[0][j] - prevPoints[j];
		} else {
		    simplex[j] = simplex[j] * 1.3;
		}
		prevPoints[j] = points[0][j];

		if (debug >= DEBUG_TRAIN) {
		    cerr << "lambda_" << i << " " << points[0][j]
			  << " " << simplex[j] << endl;
		}

		j++;
	    }
	}

	if (debug >= DEBUG_TRAIN) {
	    cerr << "scale " << points[0][0] << endl;
	    cerr << "errors " << errs[0] << endl;
	    cerr << "unchanged for " << same << " iterations " << endl;
	}

	if (nevals >= maxIters) {
	    cerr << "maximum number of iterations exceeded" << endl;
	    loop = 0;
	}

	if (reps > maxAmoebaRestarts) {
	    cerr << "maximum number of Amoeba restarts reached" << endl;
	    loop = 0;
	}

	if (abortSearch) {
	    cerr << "search timed out after " << maxTime << " seconds\n";
	    loop = 0;
	}
    }

    for (unsigned i = 0; i <= numFreeWeights; i++) {
	delete [] points[i];
	delete [] dir[i];
    }

    // Scale the lambdas back
    for (unsigned i = 0; i < numScores; i++) {
	lambdas[i] /= posteriorScale;
    }
}

/*
 * output 1-best hyp
 */
void
printTop1bestHyp(File &file, RefString id, NBestScore **scores,
							NBestList &nbest)
{
    unsigned numHyps = nbest.numHyps();
    unsigned bestHyp;
    LogP bestScore;

    file.fprintf("%s", id);

    /*
     * Find hyp with highest score
     */
    for (unsigned i = 0; i < numHyps; i ++) {
	LogP score = hypScore(i, scores);

	if (i == 0 || score > bestScore) {
	    bestScore = score;
	    bestHyp = i;
	}
    }

    if (numHyps > 0) {
	VocabIndex *hyp = nbest.getHyp(bestHyp).words;

	for (unsigned j = 0; hyp[j] != Vocab_None; j ++) {
	    file.fprintf(" %s", nbest.vocab.getWord(hyp[j]));
	}
    }
    file.fprintf("\n");
}

struct ISPair {
    int index;
    LogP score;
};

static int 
myISPsorter(const void *p1, const void *p2)
{
    const ISPair * i1 = (const ISPair *) p1;
    const ISPair * i2 = (const ISPair *) p2;

    if (i1->score < i2->score) {
	return 1;
    } else if (i1->score == i2->score) {
	return 0;
    } else {
	return -1;
    }
}

/*
 * output rescored n-best hyps
 */
void
printTopNbestHyps(RefString id, NBestList &nbest, unsigned num, char * dirname)
{
  
    unsigned numHyps = nbest.numHyps();

    LHash<const char *, int> uniqueHyps;

    NBestScore ***scores = nbestScores.find(id);
    assert(scores != 0);

    char filename[1024];
    sprintf(filename, "%s/%s", dirname, id);
    File file(filename, "w");

    makeArray(ISPair, pa, numHyps);

    /*
     * Find n-hyp with highest score
     */
    for (unsigned i = 0; i < numHyps; i ++) {
	LogP score = hypScore(i, *scores);
	pa[i].index = i;
	pa[i].score = score;

    }

    if (numHyps > 0) {
      
        qsort(pa, numHyps, sizeof(ISPair), myISPsorter);

	unsigned numPrinted = 0;

	for (unsigned k = 0; k < numHyps; k++) {      
	  
	    ostringstream oss;
	    VocabIndex *hyp = nbest.getHyp(pa[k].index).words;
	    for (unsigned j = 0; hyp[j] != Vocab_None; j ++) {
		oss << " " << nbest.vocab.getWord(hyp[j]);
	    }
	    oss << endl;
	    const char *str = oss.str().c_str();

	    Boolean dontPrint = false;
	    if (printUniqueHyps) {
		uniqueHyps.insert(str, dontPrint);
	    }
	    
	    if (!dontPrint) {
		if (printOldRanks)  {
		    file.fprintf("or=%d%s", pa[k].index + 1, str);
		} else {
		    file.fprintf("pr=%.*g%s", LogP_Precision, pa[k].score, str);
		}
		numPrinted++;

		if (numPrinted >= num) break;
	    }
	}
    }
}


unsigned
findThresholdPoints(NBestSet &nbestSet, unsigned feat,
		    LHash<const char *, DeltaCounts> *thresholds, 
                    DeltaCounts &firstCounts) 
{
    assert(feat < numScores);
    firstCounts.clear();
    thresholds->clear();
    NBestSetIter iter(nbestSet);
    NBestList *nbest;
    RefString id;
    unsigned i, j;
    char dbuf[128];

    int num = 0;

    Array<double> fixedScores;  

    while ((nbest = iter.next(id))) {
	NBestScore ***scores = nbestScores.find(id);     
	if (!scores) continue;
	
	double minFeatScore = 1.0e100;
	double bestScore = -1.0e100;
	unsigned best = 0;
	for (j = 0; j < nbest->numHyps(); j ++) {
	    double score = 0.0;
	    for (i = 0; i < numScores; i ++) {
		if (i != feat) {
		    score += (*scores)[i][j] * lambdas[i];
		}
	    }

	    double fscore = (*scores)[feat][j];
	    if (fscore < minFeatScore ||
		(fscore == minFeatScore && score > bestScore))
	    {
		minFeatScore    = fscore;
		bestScore       = score;
		best            = j;
	    }

	    fixedScores[j] = score;
	}
	    
	double bestFixedScore = fixedScores[best];
	double lastThresh = -1e100;
	NBestHyp *bestHyp = &(nbest->getHyp(best));
	if (!optimizeBleu) {
	    firstCounts.werr.numErr += bestHyp->numErrors;
	    firstCounts.werr.numWrd += bestHyp->numWords;
	} else {
	    unsigned k;
	    for (k = 0; k < bleuNgram; k ++) {
		firstCounts.bleu.correct[k] +=
			bestHyp->bleuCount->correct[k];
		firstCounts.bleu.total[k] +=
			(bestHyp->numWords > k ? bestHyp->numWords - k : 0);
	    }
	    firstCounts.bleu.length += bestHyp->numWords;
	    firstCounts.bleu.closestRefLeng += bestHyp->closestRefLeng;
	}

	double bestFeatScore = minFeatScore;
	while (1) {
	    unsigned h = 0;
	    double thresh = 1e100;
	    double featScore = 0.0;
	    for (j = 0; j < nbest->numHyps(); j ++) {        
		double fscore = (*scores)[feat][j];
		double fxscore = fixedScores[j];
		if (fscore > bestFeatScore) {
		    double t = - (bestFixedScore - fxscore) /
		    			(bestFeatScore - fscore);
		  
		    if (t > lastThresh) {
			if (t < thresh || (t == thresh && fscore > featScore)) {
			    h = j;
			    thresh = t;
			    featScore = fscore;
			}
		    }
		}
	    }
	    
	    if (thresh == 1e100) break;

	    // put thresh into hash
	    Boolean foundP;
	    sprintf(dbuf, "%.10f", thresh);
	    DeltaCounts *pi = thresholds->insert(dbuf, foundP);
	    if (!foundP) {
		pi->clear();
	    }

	    NBestHyp *o = bestHyp;
	    NBestHyp *n = &(nbest->getHyp(h));
	    if (!optimizeBleu) {
		double deltaErr = (int) n->numErrors - (int) o->numErrors;
		int deltaCnt = (int) n->numWords - (int) o->numWords;
		pi->werr.numErr += deltaErr;
		pi->werr.numWrd += deltaCnt;
	    } else {
		unsigned k;
		int nt = n->numWords;
		int ot = o->numWords;
		int nl = n->closestRefLeng;
		int ol = o->closestRefLeng;

		pi->bleu.length += nt - ot;
		pi->bleu.closestRefLeng +=  nl - ol;

		for (k = 0; k < bleuNgram; k++) {
		    pi->bleu.correct[k] += ((int) n->bleuCount->correct[k] 
					    - (int) o->bleuCount->correct[k]);
		    pi->bleu.total[k] += (nt - ot);
		    
		    if (nt > 0) nt --;
		    if (ot > 0) ot --;
		}
	    }

	    best      = h;
	    bestHyp   = n;
	    bestFixedScore = fixedScores[h];
	    bestFeatScore  = (*scores)[feat][h];
	    lastThresh = thresh;
	}
    }
    
    return thresholds->numEntries();
}

static int
dblCompare(const char *sp1, const char *sp2) 
{
    double p1 = atof(sp1);
    double p2 = atof(sp2);
    if (p1 < p2) {
	return -1;
    } else if (p1 == p2) {
	return 0;
    } else {
	return 1;  
    }
}

double
computeError(DeltaCounts &counts) 
{
    double error;

    if (!optimizeBleu) {
	error = counts.werr.numErr;
    } else {
	unsigned correct[MAX_BLEU_NGRAM], total[MAX_BLEU_NGRAM];
	unsigned length = counts.bleu.length;

	for(unsigned i = 0; i < bleuNgram; i ++) {
	    correct[i] = counts.bleu.correct[i];
	    total[i]   = counts.bleu.total[i];
	    if (counts.bleu.correct[i] < 0) {
		fprintf(stderr, "warning: correct count less than 0!\n");
		correct[i] = 0;
	    }
	}

	if (useClosestRefLeng) 
	  bleuRefLength = counts.bleu.closestRefLeng;

	double bleu =
		computeBleu(bleuNgram, correct, total, length, bleuRefLength);

	error = bleuScale * (1 - bleu);
    }
    return error;
}

double
findMinError(LHash<const char *, DeltaCounts> *thresholds,
             DeltaCounts &firstCounts, double &lambda)
{
    LHashIter<const char *, DeltaCounts> iter(*thresholds, dblCompare);
    int first = 1;
    double lastThresh = 0.0;

    NBestList *nbest;
    RefString id;

    DeltaCounts *delta, counts = firstCounts;
    double minError = 1.0e100, error;
    const char *pkey;
    double thresh;

    while ((delta = iter.next(pkey))) {
	thresh = atof(pkey);
	if (first) {
	    first = 0;
	    lastThresh = thresh - 0.2;
	}
	
	error = computeError(counts);

	if (error < minError) {
	    lambda = (thresh + lastThresh) / 2;
	    minError = error;
	}

	// increment counts
	counts += *delta;

	lastThresh = thresh;
    }

    error = computeError(counts);

    if (error < minError) {    
	lambda = lastThresh + 0.1;
	minError = error;
    }

    return minError;
}

void
initializePowell(unsigned run)
{
    unsigned i = 0;

    if (debug >= DEBUG_TRAIN) {
	cerr << "Initial lambdas for run " << run << ": " << endl;    
    }

    if (run == 0) {
	// use initial value for lambdas
        srand48(useDynamicRandomSeries ? time(0) + (GETPID() << 8) : 0);
	return;
    }

    for (i = 0; i < numScores; i ++) {
	if (fixLambdas[i]) continue;
	lambdas[i] = (lambdaMins[i] +
			drand48() * (lambdaMaxs[i] - lambdaMins[i])) /
								posteriorScale;
	if (debug >= DEBUG_TRAIN) {
	    cerr << "lambdas[" << i << "] = " << lambdas[i] << endl;
	}
    }
}

/*
 * quick powell grid search
 */
void
trainPowellQuick(NBestSet &nbestSet)
{
    double minError, globalMinError;
    LHash<const char *, DeltaCounts> thresholds;

    // get start error and copy the initial lambdas
    globalMinError = computeErrors(nbestSet, lambdas.data());
    bestLambdas = lambdas;
      
    unsigned i, j;

    for (i = 0; i < numPowellRuns; i ++) {
	initializePowell(i);
	int loop = 0;
	minError = computeErrors(nbestSet, lambdas.data()); 

	while (1) {
	    int bestDimen = -1;
	    double bestWeight = 0.0;
	    DeltaCounts firstCounts;
	    
	    for (j = 2; j < numScores; j ++) {
		if (fixLambdas[j]) continue;
		
		findThresholdPoints(nbestSet, j, &thresholds, firstCounts);
		
		double lambda;
		double merr = findMinError(&thresholds, firstCounts, lambda);
		
		if (merr < minError - 0.000001) {
		    minError = merr;
		    bestWeight = lambda;
		    bestDimen = j;
		}
	    }

	    if (debug >= DEBUG_TRAIN) {
		cerr << "run(" << i << "), loop(" << loop << ") : dim : "
		     << bestDimen;
		cerr << "; lambda : " << bestWeight << "; error : " << minError;
		cerr << "; global : " << globalMinError << endl;
	    }

	    loop ++;

	    if (bestDimen < 0) break;
	    lambdas[bestDimen] = bestWeight;

	    // normalize weights
	    if (oneBest && numFixedWeights == 0) {
		double norm = 0;
		unsigned k;
		for (k = 0; k < numScores; k ++) {
		    norm += lambdas[k] * lambdas[k];
		}
		norm = sqrt(norm);
		for (k = 0; k < numScores; k ++) {
		    lambdas[k] = lambdas[k] / norm;
		}
	    }
	}
	
	if (debug >= DEBUG_TRAIN) {
	    unsigned k;
	    cerr << "Run (" << i << ") Error : " << minError << endl;
	    for (k = 0; k < numScores; k ++) {
		cerr << "lambdas[" << k << "] = " << lambdas[k] << endl;
	    }     
	}
	
	if (minError < globalMinError) {
	    globalMinError = minError;
	    bestLambdas = lambdas;
	}
    }
      
    if (debug >= DEBUG_TRAIN) {
	unsigned k;
	cerr << "Final Error : " << globalMinError << endl;
	for (k = 0; k < numScores; k ++) {
	    cerr << "lambdas[" << k << "] = " << bestLambdas[k] << endl;
	}
    }
    bestError = (unsigned)globalMinError;
}

/*
 * output best sausage hypotheses
 */
void
printTopSausageHyp(File &file, RefString id, NBestScore **scores,
							WordMesh &alignment)
{
    file.fprintf("%s", id);

    /* 
     * process all positions in alignment
     */
    for (unsigned pos = 0; pos < alignment.length(); pos++) {
	VocabIndex bestWord = Vocab_None;
	Prob bestScore = 0.0;

	WordMeshIter iter(alignment, pos);

	Array<HypID> *hypMap;
	VocabIndex word;
	while ((hypMap = iter.next(word))) {
	    /*
	     * compute total score for word and check if it's the correct one
	     */
	    Boolean dummy;
	    Prob totalScore = wordScore(*hypMap, scores, dummy);

	    if (bestWord == Vocab_None || totalScore > bestScore) {
		bestWord = word;
		bestScore = totalScore;
	    }
	}

	assert(bestWord != Vocab_None);

	if (bestWord != alignment.deleteIndex) {
	    file.fprintf(" %s", alignment.vocab.getWord(bestWord));
	}
    }
    file.fprintf("\n");
}

void
printTopHyps(File &file, NBestSet &nbestSet)
{
    NBestSetIter iter(nbestSet);
    NBestList *nbest;
    RefString id;

    while ((nbest = iter.next(id))) {
	NBestScore ***scores = nbestScores.find(id);
	assert(scores != 0);

	if (oneBest) {
	    printTop1bestHyp(file, id, *scores, *nbest);
	} else {
	    WordMesh **alignment = nbestAlignments.find(id);
	    assert(alignment != 0);

	    printTopSausageHyp(file, id, *scores, **alignment);
	}
    }
}

/*
 * Align N-best lists
 */

typedef struct {
    LogP score;
    unsigned rank;
} HypRank;			/* used in sorting nbest hyps by score */

static int
compareHyps(const void *h1, const void *h2)
{
    LogP score1 = ((HypRank *)h1)->score;
    LogP score2 = ((HypRank *)h2)->score;
    
    return score1 > score2 ? -1 :
		score1 < score2 ? 1 : 0;
}

void
alignNbest(NBestSet &nbestSet, RefList &refs, VocabDistance &distance)
{
    NBestSetIter iter(nbestSet);
    NBestList *nbest;
    RefString id;

    while ((nbest = iter.next(id))) {
	VocabIndex *ref = refs.findRef(id);

	assert(ref != 0);

	unsigned numHyps = nbest->numHyps();

	/*
	 * Sort hyps by initial scores.  (Combined initial scores are
	 * stored in acousticScore from before.)
	 * Keep hyp order outside of N-best lists, since scores must be
	 * kept in sync.
	 */
	makeArray(HypRank, reordering, numHyps);
	NBestScore ***scores = nbestScores.find(id);
	assert(scores != 0);

	/*
	 * Copy combined scores back into N-best list acoustic score for
	 * posterior probability computation (since computePosteriors()
	 * doesn't take additional scores).
	 */
	for (unsigned j = 0; j < numHyps; j ++) {
	    reordering[j].rank = j;
	    reordering[j].score = 
		nbest->getHyp(j).acousticScore = hypScore(j, *scores);
	}

	if (!noReorder) {
	    qsort(reordering, numHyps, sizeof(HypRank), compareHyps);
	}
	
	/*
	 * compute posteriors for passing to alignWords().
	 * Note: these now reflect all scores and initial lambdas.
	 */
	nbest->computePosteriors(0.0, 0.0, 1.0);

	/*
	 * create word-mesh for multiple alignment
	 */
	WordMesh *alignment;
	if (dictFile || hiddenVocabFile || distanceFile) {
	    alignment = new WordMesh(nbestSet.vocab, 0, &distance);
	} else {
	    alignment = new WordMesh(nbestSet.vocab);
	}
	assert(alignment != 0);

	*nbestAlignments.insert(id) = alignment;

	/*
	 * Default is to start alignment with hyps strings,
	 * or with the reference if -align-refs-first was given.
	 *	Note we give reference posterior 1 only to constrain the
	 *	alignment. The loss computation in training ignores the
	 *	posteriors assigned to hyps at this point.
	 */
	HypID hypID;

	if (noReorder) {
	    hypID = refID;
	    alignment->alignWords(ref, 1.0, 0, &hypID);
	}

	/*
	 * Now align all N-best hyps, in order of decreasing scores
	 */
	for (unsigned j = 0; j < numHyps; j ++) {
	    unsigned hypRank = reordering[j].rank;
	    NBestHyp &hyp = nbest->getHyp(hypRank);

	    hypID = hypRank;

	    /*
	     * Check for overflow in the hypIDs
	     */
	    if ((unsigned)hypID != hypRank || hypID == refID) {
		cerr << "Sorry, too many hypotheses in " << id << endl;
		exit(2);
	    }

	    alignment->alignWords(hyp.words, hyp.posterior, 0, &hypID);
	}

	if (!noReorder) {
	    hypID = refID;
	    alignment->alignWords(ref, 1.0, 0, &hypID);
	}

	if (debug >= DEBUG_ALIGNMENT) {
	    dumpAlignment(cerr, *alignment);
	}
    }
}

/*
 * Read a single score file into a column of the score matrix
 */
Boolean
readScoreFile(const char *scoreDir, RefString id, NBestScore *scores,
							unsigned numHyps) 
{
    makeArray(char, fileName,
	      strlen(scoreDir) + 1 + strlen(id) + strlen(GZIP_SUFFIX) + 1);
					
    sprintf(fileName, "%s/%s", scoreDir, id);

    /* 
     * If plain file doesn't exist try gzipped version
     */
    FILE *fp = 0;
    if ((fp = fopen(fileName, "r")) == NULL) {
	strcat(fileName, GZIP_SUFFIX);
    } else {
	fclose(fp);
    }

    File file(fileName, "r", 0);

    char *line;
    unsigned hypNo = 0;
    Boolean decipherScores = false;

    while (!file.error() && (line = file.getline())) {
	if (strncmp(line, nbest1Magic, sizeof(nbest1Magic)-1) == 0 ||
	    strncmp(line, nbest2Magic, sizeof(nbest2Magic)-1) == 0)
	{
	    decipherScores = true;
	    continue;
	}

	if (hypNo >= numHyps) {
	    break;
	}

	/*
	 * parse the first word as a score
	 */
	double score;

	if (decipherScores) {
	    /*
	     * Read decipher scores as floats event though they are supposed
	     * to be int's.  This way we accomodate some preexisting rescoring
	     * programs.
	     */
	    if (sscanf(line, "(%lf)", &score) != 1) {
		file.position() << "bad Decipher score: " << line << endl;
		break;
	    } else  {
		scores[hypNo ++] = BytelogToLogP((int)score);
	    }
	} else {
	    if (sscanf(line, "%lf", &score) != 1) {
		file.position() << "bad score: " << line << endl;
		break;
	    } else  {
		scores[hypNo ++] = score;
	    }
	}
    }

    /* 
     * Set missing scores to zero
     */
    if (!file.error() && hypNo < numHyps) {
	cerr << "warning: " << (numHyps - hypNo) << " scores missing from "
	     << fileName << endl;
    }
	
    while (hypNo < numHyps) {
	scores[hypNo ++] = 0;
    }

    return !file.error();
}

/*
 * Read error counts file
 */
Boolean
readErrorsFile(const char *errorsDir, RefString id, NBestList &nbest,
							unsigned &numWords)
{
    unsigned numHyps = nbest.numHyps();
    makeArray(char, fileName,
	      strlen(errorsDir) + 1 + strlen(id) + strlen(GZIP_SUFFIX) + 1);
					
    sprintf(fileName, "%s/%s", errorsDir, id);

    /* 
     * If plain file doesn't exist try gzipped version
     */
    FILE *fp;
    if ((fp = fopen(fileName, "r")) == NULL) {
	strcat(fileName, GZIP_SUFFIX);
    } else {
	fclose(fp);
    }

    File file(fileName, "r", 0);

    char *line;
    unsigned hypNo = 0;

    while (!file.error() && (line = file.getline())) {

	if (hypNo >= numHyps) {
	    break;
	}

	/*
	 * parse errors line
	 */
	float corrRate, errRate, numErrs;
	unsigned numSub, numDel, numIns, numWds;

	if (sscanf(line, "%f %f %u %u %u %g %u", &corrRate, &errRate,
			 &numSub, &numDel, &numIns, &numErrs, &numWds) != 7)
	{
	    file.position() << "bad errors: " << line << endl;
	    return 0;
	} else {
	    if (hypNo == 0) {
		numWords = numWds;
	    } else if (numWds != numWords) {
		/*
 		 * Warn about changing numbers of words, which may be the result
 		 * of sclite's alternative reference strings, but allow it.
 		 */
	        file.position() << "warning: number of words changed from "
		                << numWords << " to " << numWds << endl;
	    }
	    nbest.getHyp(hypNo ++).numErrors = numErrs;
	}
    }

    if (hypNo < numHyps) {
	file.position() << "too few errors lines" << endl;
	return 0;
    }

    return !file.error();
}

/*
 * Read word weights file
 */
Boolean
readWordWeights(File &file, Vocab &vocab, LHash<VocabIndex, double> &weights)
{
    char *line;

    while ((line = file.getline())) {
        char buffer[1001];
	double weight;

	if (sscanf(line, "%1000s %lf", buffer, &weight) != 2) {
	    return false;
	}

	*weights.insert(vocab.addWord(buffer)) = weight;
    }

    return true;
}

#define MAX_NUM_REFS 256

/*
 * Read error counts file
 */
Boolean
readBleuCountsFile(const char *countsDir, RefString id, NBestList &nbest,
		   unsigned &numWords)
{
    unsigned numHyps = nbest.numHyps();

    makeArray(char, fileName,
     	      strlen(bleuCountsDir) + 1 + strlen(id) + strlen(GZIP_SUFFIX) + 1);
					
    sprintf(fileName, "%s/%s", bleuCountsDir, id);

    /* 
     * If plain file doesn't exist try gzipped version
     */
    FILE *fp;
    if ((fp = fopen(fileName, "r")) == NULL) {
	strcat(fileName, GZIP_SUFFIX);
    } else {
	fclose(fp);
    }

    numWords = 0;
    File file(fileName, "r", 0);

    char *line, *p;
    unsigned hypNo = 0;
    unsigned reflen[MAX_NUM_REFS];
    reflen[0] = 0;

    // read the first line
    if (!file.error() && (line = file.getline())) {
	int pos;
	unsigned nl, nr, i, rl;

	if (sscanf(line, "%u%u%n", &nl, &nr, &pos) != 2) {
	    file.position() << "format error: " << line << endl;
	}

	if (nr > MAX_NUM_REFS) {
	    cerr << "too many references (" << nr << "), can handle only up to "
	    
	         << MAX_NUM_REFS << "!\n";
	}
	
	// check consistency
	if (numHyps != nl) {
	    cerr << "number of count lines does not match nbest list: " << nl
	         << " versus " << numHyps << endl;
	    return 0;
	}

	if (numReferences == 0) {
	    if (nr == 0) {
		cerr << "0 reference !" << endl;
		return 0;
	    }
	    numReferences = nr;

	} else if (nr != numReferences) {
	    cerr << "number of reference mismatch: " << nr
	         << " versus " << numReferences << endl;
	    return 0;
	}

	unsigned minLen = 1000000;
	// @kw false positive: SV.TAINTED.INDEX_ACCESS (pos)
	for (p = line + pos, i = 0; i < nr; i ++, p += pos) {
	    if (sscanf(p, "%u%n", &rl, &pos) != 1) {
		file.position() << "format error: " << line << endl;
		return 0;
	    } else {	        
		
		if (minLen > rl)
		  minLen = rl;

		numWords += rl;

		reflen[i] = rl;	
	    }
	}

	if (useMinRefLeng) 
	  numWords = minLen;
	else
	  numWords /= nr;
	
    }

    while (!file.error() && (line = file.getline())) {
	if (hypNo >= numHyps) {
	    break;
	}

	/*
	 * parse count line
	 */
        unsigned corr[MAX_BLEU_NGRAM], numWds;
        unsigned n = 0;
	int pos = 0;
        if (sscanf(line, "%u%n", &numWds, &pos) != 1) {
	    file.position() << "format error: " << line << endl;
	    return 0;
        }
        NBestHyp & h = nbest.getHyp(hypNo ++);
        if (numWds != h.numWords) {
	    cerr << "inconsistent number of words for hyp " << hypNo 
		 << " : " << numWds << " versus " << h.numWords << endl;
	    return 0;
        }

        char *p = line + pos;
        while (sscanf(p, "%u%n", &(corr[n]), &pos) == 1) {
	    p += pos;
	    n ++;
	    if (n >= MAX_BLEU_NGRAM) break;
        }

        if (bleuNgram == 0) {
	    bleuNgram = n;
        } else if (n != bleuNgram) {
	    file.position() << "inconsistent bleu ngram length : " << line 
			    << " : " << n << " versus " << bleuNgram << endl;
	    return 0;
        }

        if (n) {
	    h.bleuCount = new BleuCount;
	    for (unsigned i = 0; i < bleuNgram; i++) {
		h.bleuCount->correct[i] = corr[i];
	    }
        } else {
	    file.position() << "failed to read bleu counts: " << line << endl;
	    return 0;
        }

	unsigned l = reflen[0];
	unsigned diff = abs((int) (numWds - l));

	for (unsigned i = 1; i < numReferences; i++) {
	    unsigned d = abs((int) (numWds - reflen[i]));
	    if (d < diff) {
		diff = d;
		l = reflen[i];
	    } else if (d == diff && reflen[i] < l) {
		// for the same difference, use the smaller one
		l = reflen[i];
	    }
	}
	
	h.closestRefLeng = l;	
    }

    if (hypNo < numHyps) {
	file.position() << "too few count lines" << endl;
	return 0;
    }

    return !file.error();
}

struct MYSTRUCT {
  int index;
  int best;
  double value;
};

static int 
mysorter (const MYSTRUCT * d1, const MYSTRUCT * d2)
{
  double diff = d1->value - d2->value;
  
  if (diff < 0) 
    return -1;
  else if (diff > 0)
    return 1;
  else 
    return 0;    
}

typedef int (*MYSORTER) (const void *, const void *);

void
getInitialBleuStatistics(unsigned numSentences, NBestList **lists,
				unsigned correct[], unsigned total[],
				unsigned &length, unsigned &refLen)
{
    length = 0;
    refLen = 0;
    unsigned closestRefLeng = 0;
    
    memset(correct, 0, sizeof(unsigned) * bleuNgram);
    memset(total, 0, sizeof(unsigned) * bleuNgram);
    
    for (unsigned i = 0; i < numSentences; i++) {
	NBestList & nbest = *(lists[i]);
	accumulateBleuCounts(0, 0, nbest, correct, total, length,
							      closestRefLeng);
    }

    if (useClosestRefLeng) {
	refLen = closestRefLeng;
    } else {
	refLen = bleuRefLength;
    }
}

void
sortHypsBySentenceBleu(NBestSet & nbestSet)
{
    RefString id;
    NBestList * nbest;
    NBestSetIter iter(nbestSet);
    while ((nbest = iter.next(id))) {
        nbest->sortHypsBySentenceBleu(bleuNgram);
    }
}

double
findOracleBleu(NBestSet &nbestSet, int numIters, unsigned *hypIdxs = 0,
				RefString *sids = 0, NBestList **lists = 0)
{
    sortHypsBySentenceBleu(nbestSet);
    
    // initialize random series
    srand48(useDynamicRandomSeries ? time(0) + (GETPID() << 8) : 0);

    unsigned numSentences = nbestSet.numElements();
  
    MYSTRUCT *data = new MYSTRUCT [ numSentences ];

    if (hypIdxs) memset(hypIdxs, 0, sizeof(int) * numSentences);

    int allocLists = 0;
    if (lists == 0) {
	lists = new NBestList * [ numSentences ];
	allocLists = 1;
    }

    unsigned i;

    RefString id;
    NBestSetIter iter(nbestSet);

    i = 0;
    NBestList *nbest;
    while ((nbest = iter.next(id))) {
        if (sids) sids[i] = id;
        lists[i++] = nbest;	
    }

    // get initial total counts 
    unsigned correct[MAX_BLEU_NGRAM], total[MAX_BLEU_NGRAM];
    unsigned refLen, length;

    unsigned newCorr[MAX_BLEU_NGRAM], newTotl[MAX_BLEU_NGRAM];
    unsigned newRL, newLength;
     
    getInitialBleuStatistics(numSentences, lists, correct, total, length, refLen);

    double initBleu = computeBleu(bleuNgram, correct, total, length, refLen);
    
    double bestBleu = initBleu; 
  
    assert(i == numSentences);
  
    for (int it = 0; it < numIters ; it++) {
        memcpy(newCorr, correct, sizeof(unsigned) * bleuNgram);
	memcpy(newTotl, total, sizeof(unsigned) * bleuNgram);
	newLength = length;
	newRL = refLen;

	double newBleu = initBleu;
    
	for (i = 0; i < numSentences; i++) {
            data[i].index = i;
	    data[i].best = 0;
	    data[i].value = drand48();
	}

	qsort(data, numSentences, (unsigned) sizeof(MYSTRUCT), (MYSORTER) mysorter);

	DeltaBleu delta;

	double startBleu;
	   
	do {
	    startBleu = newBleu;

	    for (i = 0; i < numSentences; i++) {
	        unsigned best = data[i].best;
		int index = data[i].index;
		nbest = lists[index];
		NBestHyp & bestHyp = nbest->getHyp(best);
		int ot = bestHyp.numWords;
		int ol = bestHyp.closestRefLeng;
		unsigned short * oc = bestHyp.bleuCount->correct;
	
		unsigned numHyps = nbest->numHyps();
		for (unsigned j = 0; j < numHyps; j++) {
		    if (j == best)
		      continue;
		  
		    unsigned corr[MAX_BLEU_NGRAM], totl[MAX_BLEU_NGRAM];
		    unsigned len, rl;

		    NBestHyp & hyp = nbest->getHyp(j);
		    int nt = hyp.numWords;
		    int nl = hyp.closestRefLeng;
		    unsigned short * nc = hyp.bleuCount->correct;

		    len = newLength + nt - ot;
		    if (useClosestRefLeng) {
			rl = newRL + nl - ol;
		    } else {
			rl = newRL;
		    }

		    for(unsigned k = 0; k < bleuNgram; k++) {
		        corr[k] = newCorr[k] + nc[k] - oc[k];
			totl[k] = newTotl[k] + nt - ot;
			if (nt > 0) nt--;
			if (ot > 0) ot--;	  
		    }
                    ot = bestHyp.numWords;
                                       
		    double bleu = computeBleu(bleuNgram, corr, totl, len, rl);
	  
		    if (bleu > newBleu) {
		        data[i].best = j;
		        newBleu = bleu;
		    }
		}
	    }
	} while (newBleu > startBleu);

	if (newBleu > bestBleu) {
	    bestBleu = newBleu;

	    if (hypIdxs) {
	        for (i = 0; i < numSentences; i++) {
		    hypIdxs[data[i].index] = data[i].best;
		}
	    }
	}    

	cout << "iteration " << (it + 1) << ", oracle bleu: " << bestBleu << endl;
    }

    delete [] data;
    if (allocLists) {
	delete [] lists;
    }

    return bestBleu;
}

double
findOracleError(NBestSet &nbestSet)
{
    double totalErrors = 0;
    RefString id;
    NBestList *nbest;
    NBestSetIter iter(nbestSet);
    unsigned i = 0;
    while ((nbest = iter.next(id))) {
        totalErrors += nbest->sortHypsByErrorRate(); 
    }
    
    return (totalErrors / numRefWords);   
}

void
outputHyps(NBestSet &trainSet)
{
    if (printTopN) {
	// printHyps is a dir
	if (MKDIR(printHyps) < 0 && errno != EEXIST) {
	    perror(printHyps);
	    exit(1);
	}
	    
	NBestSetIter iter(trainSet);
	NBestList *nbest;
	RefString id;
	
	while ((nbest = iter.next(id))) {
	    printTopNbestHyps(id, *nbest, printTopN, printHyps);
	}
    } else {
	File file(printHyps, "w");

	lambdas = bestLambdas;
	printTopHyps(file, trainSet);
    }
}

void
outputOracle(NBestSet &trainSet)
{
    if (optimizeBleu) {
        unsigned numSentences = trainSet.numElements();
	RefString *sids = 0;
	unsigned *hids = 0;
	NBestList **lsts = 0;

	if (printOracleHyps) {
	    sids = new RefString [ numSentences ];
	    hids = new unsigned [ numSentences ];
	    lsts = new NBestList * [ numSentences ];
	}  
      
	double obleu =
		    findOracleBleu(trainSet, oracleBleuIters, hids, sids, lsts);
	
	printf("oracleBleu = %g\n", obleu);
	
	if (sids && hids && lsts) {
	    File file(printOracleHyps, "w");

	    for (unsigned i = 0; i < numSentences; i++) {
		file.fprintf("%s", sids[i]);
	    
		NBestList *nbest = lsts[i];
		VocabIndex *hyp = nbest->getHyp(hids[i]).words;

		for (unsigned j = 0; hyp[j] != Vocab_None; j++) {
		    file.fprintf(" %s", nbest->vocab.getWord(hyp[j]));
		}
		file.fprintf("\n");
	    }
	}

	delete [] lsts;
	delete [] hids;
	delete [] sids;
    } else {
        double oerr = findOracleError(trainSet);
        
        printf("oracleErrorRate = %g\n", oerr);
        
        if (printOracleHyps) {
            File file(printOracleHyps, "w");
        
            RefString id;
            NBestList *nbest;
            NBestSetIter iter(trainSet);
            while ((nbest = iter.next(id))) {
                file.fprintf("%s", id);
                VocabIndex *hyp = nbest->getHyp(0).words;

                for (unsigned j = 0; hyp[j] != Vocab_None; j++) {
                    file.fprintf(" %s", nbest->vocab.getWord(hyp[j]));
		}
                file.fprintf("\n");
            }
        }       
    }
}

Boolean
readScores(NBestSet &nbestSet, unsigned numScoreDirs, char *scoreDirs[])
{
    NBestSetIter iter(nbestSet);
    RefString id;
    NBestList *nbest;

    while ((nbest = iter.next(id))) {
	/*
	 * Allocate score matrix for this nbest list
	 */
	NBestScore **scores = new NBestScore *[numScores];
	assert(scores != 0);

	for (unsigned i = 0; i < numScores; i ++) {
	    scores[i] = new NBestScore[nbest->numHyps()];
	    assert(scores[i] != 0);
	}

	/*
	 * Transfer the standard scores from N-best list to score matrix
	 */
	for (unsigned j = 0; j < nbest->numHyps(); j ++) {
	    scores[0][j] = nbest->getHyp(j).acousticScore;
	    scores[1][j] = nbest->getHyp(j).languageScore;
	    scores[2][j] = (NBestScore) nbest->getHyp(j).numWords;
	}

	/*
	 * Read additional scores
	 */
	for (unsigned i = 1; i < numScoreDirs; i ++) {
	    if (!readScoreFile(scoreDirs[i], id, scores[i + 2], nbest->numHyps())) {
		cerr << "warning: error reading scores for " << id
		     << " from " << scoreDirs[i] << endl;
	    }
	}

	/*
	 * Scale scores to help prevent underflow
	 */
	if (!combineLinear && !optimizeBleu) {
	    for (unsigned i = 0; i < numScores; i ++) {
		for (unsigned j = nbest->numHyps(); j > 0; j --) {
		    scores[i][j-1] -= scores[i][0];
		}
	    }
	}
    
	/* 
	 * save score matrix under nbest id
	 */
	*nbestScores.insert(id) = scores;
    }
    return true;
}

unsigned 
prepareErrorCounts(NBestSet &nbestSet, RefList &refs, RefList &antiRefs, NullLM &nullLM)
{
    NBestSetIter iter(nbestSet);
    RefString id;
    NBestList *nbest;

    unsigned totalRefWords = 0;

    /*
     * Compute hyp errors
     */
    while ((nbest = iter.next(id))) {
	unsigned numWords;
	VocabIndex *ref = refs.findRef(id);
	VocabIndex *antiRef = antiRefs.findRef(id);

	if (!(ref || ((oneBest && !oneBestFirst) &&
		      (errorsDir || bleuCountsDir))))
	{
	    cerr << "missing reference for " << id << endl;
	    exit(1);
	}

	/*
	 * Remove pauses and noise from nbest hyps since these would
	 * confuse the inter-hyp alignments.
	 */
	nbest->removeNoise(nullLM);

	/*
	 * In 1-best mode we only need the error counts for each hypothesis;
	 * in sausage (default) mode we need to construct multiple alignment
	 * of reference and all n-best hyps.
	 */
	if (errorsDir) {
	    /*
	     *  read error counts 
	     */
	    if (!readErrorsFile(errorsDir, id, *nbest, numWords)) {
		cerr << "couldn't get error counts for " << id << endl;
		exit(2);
	    }
	}

	if (bleuCountsDir) {
	    /*
	     * read bleu counts
	     */
	    if (!readBleuCountsFile(bleuCountsDir, id, *nbest, numWords)) {
		cerr << "couldn't get bleu counts for " << id << endl;
		exit(2);
	    }
	} 

	if (!errorsDir && !bleuCountsDir && !srinterpCountsFile) {
	    /*
	     * need to recompute hyp errors (after removeNoise() above)
	     */
	    unsigned sub, ins, del;
	    nbest->wordError(ref, sub, ins, del);
	    numWords = Vocab::length(ref);
	}

	if (antiRefWeight != 0.0) {
	    if (antiRef == 0) {
		cerr << "warning: missing anti-reference for " << id << endl;
	    } else {
		/*
		 * Add anti-ref error to error counts
		 */
		unsigned sub, ins, del;
		nbest->wordError(antiRef, sub, ins, del, antiRefWeight);
	    }
	}

	/*
	 * compute total length of references for later normalizations
	 */
	totalRefWords += numWords;
    }

    return totalRefWords;
}

int
main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    argc = Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    if (!nbestFiles) {
	cerr << "cannot proceed without nbest files and feature score file\n";
	exit(2);
    }

    if (bleuCountsDir || srinterpCountsFile) {
	optimizeBleu = 1;
	oneBest = 1;
	oneBestFirst = 0;
	if (srinterpCountsFile) srinterpFormat = 1;

	if (!initSimplex && !initPowell) {
	    cerr << "bleu optimization only supported in simplex or powell model\n";
	    exit (2);
	} else if (errorBleuRatio == 0) {
	    cerr << "will optimize BLEU score instead of error rate!\n";
	} else {
	  cerr << "will optimize combined metric: BLEU - ERR * " << errorBleuRatio << endl;
	}

	if (useAvgRefLeng) {
	    cerr << "use average bleu reference length" << endl;
	} else if (useMinRefLeng) {
	    cerr << "use minimum bleu reference length" << endl;
	} else if (useClosestRefLeng) {
	    cerr << "use closest bleu reference length" << endl;
	} else {
	    cerr << "did not specify reference length method, use average bleu referenc length" << endl;
	    useAvgRefLeng = 1;
	}	  
    }

    if (!oneBest && !refFile && !skipopt) {
	cerr << "cannot proceed without references\n";
	exit(2);
    }
    if (oneBest && !refFile && !errorsDir && !optimizeBleu && !skipopt) {
	cerr << "cannot proceed without references or error counts\n";
	exit(2);
    }

    if ((oneBest || oneBestFirst) && !initSimplex && !initPowell && !skipopt) {
	cerr << "1-best optimization only supported in simplex or powell mode\n";
	exit(2);
    }

    if (srinterpFormat) {
        bool foundP;
	LHash<RefString, int> seen(argc);
        // now the argvs contain feature names instead of score directories
        for (int i = 1; i < argc; i++) {
	    // format is "name#scale"
	    char * feature = argv[i];
	    char * p = strchr(feature, '#');
	    if (p) {
	        *p = '\0';
		featNames[i-1] = feature;
		featScales[i-1] = atof(p+1);
	    } else {
	        featNames[i-1] = feature;
		featScales[i-1] = 1.f;
	    }
	    *seen.insert(feature, foundP) = 1;
	    if (foundP) {
	      cerr << "Feature \"" << feature << "\" appears more than once!" << endl;
	      exit (-1);
	    }
	}
    }


    Vocab vocab;
    NullLM nullLM(vocab);
    RefList refs(vocab);
    RefList antiRefs(vocab);

    NBestSet trainSet(vocab, refs, maxNbest, false, multiwords ? multiChar : 0);
    trainSet.debugme(debug);
    trainSet.warn = false;	// don't warn about missing refs

    NBestSet xvalSet(vocab, refs, maxNbest, false, multiwords ? multiChar : 0);
    xvalSet.debugme(debug);
    xvalSet.warn = false;	// don't warn about missing refs

    if (vocabFile) {
	File file(vocabFile, "r");
	vocab.read(file);
    }

    vocab.toLower() = toLower ? true : false;

    /*
     * Skip noise tags in scoring
     */
    if (noiseVocabFile) {
	File file(noiseVocabFile, "r");
	nullLM.noiseVocab.read(file);
    }
    if (noiseTag) {				/* backward compatibility */
	nullLM.noiseVocab.addWord(noiseTag);
    }

    /* 
     * Read optional dictionary to help in word alignment
     */
    Vocab dictVocab;
    VocabMultiMap dictionary(vocab, dictVocab);

    if (dictFile) {
	File file(dictFile, "r");

	if (!dictionary.read(file)) {
	    cerr << "format error in dictionary file\n";
	    exit(1);
	}
    }
    DictionaryAbsDistance dictDistance(vocab, dictionary);

    /* 
     * Read optional word distance matrix to direct word alignment
     */
    VocabMap distanceMatrix(vocab, vocab);

    if (distanceFile) {
	File file(distanceFile, "r");

	if (!distanceMatrix.read(file)) {
	    cerr << "format error in distance matrix\n";
	    exit(1);
	}
    }
    MatrixDistance matrixDistance(vocab, distanceMatrix);

    /*
     * Optionally read a subvocabulary that is to be kept separate from
     * regular words during alignment
     */
    SubVocab hiddenVocab(vocab);
    if (hiddenVocabFile) {
	File file(hiddenVocabFile, "r");

	if (!hiddenVocab.read(file)) {
	    cerr << "error in hidden vocab file\n";
	    exit(1);
	}
    }
    SubVocabDistance subvocabDistance(vocab, hiddenVocab);

    /* 
     * Read word-specific error weights
     */
    if (wordWeightFile) {
    	File file(wordWeightFile, "r");

	cerr << "reading word weights...\n";

	if (!readWordWeights(file, vocab, wordWeights)) {
	    cerr << "error in word weights file\n";
	    exit(1);
	}
    }

    /*
     * Posterior scaling:  if not specified (= 0.0) use LMW for
     * backward compatibility.
     */
    if (posteriorScale == 0.0) {
	posteriorScale = (rescoreLMW == 0.0) ? 1.0 : rescoreLMW;
    }

    if (refFile) {
	cerr << "reading references...\n";
	File file(refFile, "r");

	refs.read(file, true);	 // add reference words to vocabulary
    }

    if (antiRefFile) {
	cerr << "reading anti-references...\n";
	File file(antiRefFile, "r");

	antiRefs.read(file, true);	 // add words to vocabulary
    }

    {
	cerr << "reading nbest lists...\n";
	File file(nbestFiles, "r");
	if (srinterpFormat) {

  	    numScores = argc - 1;
	    numFixedWeights = 0;

	    if (!trainSet.readSRInterpFormat(file, nbestScores, numScores, featNames, featScales)) {
	        cerr << "failed to read SRInterp-format n-best lists" << endl;
	        exit (2);
	    }

	    for (unsigned i = 0; i < numScores; i ++) {
	        lambdas[i] = 0.0;
	        fixLambdas[i] = false;
	        lambdaSteps[i] = 1.0;
	    }

	} else {
	    trainSet.read(file);

	    if (xvalFiles) {
		File file(xvalFiles, "r");

		cerr << "reading xval nbest lists...\n";
		xvalSet.read(file);
	    }
	  
	    /*
	     * there are three scores in the N-best list, plus as many as 
	     * user supplies in separate directories on the command line
	     */
	    numScores = 3 + argc - 1;
	    numFixedWeights = 0;
	    
	    lambdas[0] = 1/posteriorScale;
	    lambdas[1] = rescoreLMW/posteriorScale;
	    lambdas[2] = rescoreWTW/posteriorScale;
	    
	    for (unsigned i = 0; i < 3; i ++) {
	        fixLambdas[i] = false;
		lambdaSteps[i] = 1.0;
	    }
	    
	    for (unsigned i = 3; i < numScores; i ++) {
	        lambdas[i] = 0.0;
	        fixLambdas[i] = false;
	        lambdaSteps[i] = 1.0;
	    }
	}
    }

    /*
     * Store directory names needed to write nbest-rover file
     */
    if (!srinterpFormat) {
	/*
	 * infer nbest directory name from first file in list
	 */
	NBestSetIter iter(trainSet);
	RefString id;
	const char *nbestFilename = iter.nextFile(id);
	if (nbestFilename) {
	    nbestDirectory = strdup(nbestFilename);
	    assert(nbestDirectory != 0);

	    char *basename = strrchr(nbestDirectory, '/');
	    if (basename != 0) {
		*basename = '\0';
	    } else {
		strcpy(nbestDirectory, ".");
	    }
	} else {
	    nbestDirectory = strdup(".");
	    assert(nbestDirectory != 0);
	}

	scoreDirectories = &argv[1];
    }
    /*
     * Initialize lambdas from command line values if specified
     */
    if (initLambdas) {
	unsigned offset = 0;

	for (unsigned i = 0; i < numScores; i ++) {
	    int consumed = 0;
	    if (sscanf(&initLambdas[offset], " =%lf%n",
						&lambdas[i], &consumed) > 0)
	    {
                lambdaInitials[i] = lambdas[i];
	        lambdas[i] /= posteriorScale;
		fixLambdas[i] = true;
		numFixedWeights++;
	    } else if (sscanf(&initLambdas[offset], "%lf%n",
						&lambdas[i], &consumed) > 0)
	    {
                lambdaInitials[i] = lambdas[i];
	        lambdas[i] /= posteriorScale;
		lambdaSteps[i] = 1.0;
	    } else {
		break;
	    }
	    offset += consumed;
	}
    }

    /*
     * Initialize simplex points
     */
    if (initSimplex) {
	unsigned offset = 0;

	int consumed = 0;
	for (unsigned i = 0; i < numScores; i++) {

	    if (!fixLambdas[i]) {
	        if (sscanf(&initSimplex[offset], "%lf%n",
					&lambdaSteps[i], &consumed) <= 0)
		{
		    break;
		}

	        if (lambdaSteps[i] == 0.0) {
		    cerr << "Fixing " << i << "th parameter\n";
		    fixLambdas[i] = true;
		    numFixedWeights++;
		}

		offset += consumed;
	    }
	}

	sscanf(&initSimplex[offset], "%lf%n", &posteriorScaleStep, &consumed);
    }

    /*
     * Initialize Powell quick grid search
     */
    if (initPowell) {
	unsigned offset = 0;

	char buf[512];
	int consumed = 0;
	for (unsigned i = 0; i < numScores; i++) {
	    if (!fixLambdas[i]) {
		if (sscanf(&initPowell[offset], "%lf,%lf%n",
			   &lambdaMins[i], &lambdaMaxs[i], &consumed) != 2) {

		    cerr << "failed to parse powell initialization : \"" << initPowell << "\"!" << endl;
		    exit (-1);
		}

		if (lambdaMins[i] == lambdaMaxs[i] && 
		    lambdaMins[i] == lambdaInitials[i])
		{
		    cerr << "Fixing " << i << "th parameter\n";
		    fixLambdas[i] = true;
		    numFixedWeights++;
		}

		offset += consumed;
	    } else {
		lambdaMins[i] = lambdaMaxs[i] = lambdaInitials[i];
	    }
	}
    }

    /*
     * Set up the score matrices
     */
    if (!srinterpFormat) {
        cerr << "reading scores...\n";

 	readScores(trainSet, argc, argv);

	if (xvalFiles) {
	    readScores(xvalSet, argc, argv);
	}
    }

    if (debug >= DEBUG_SCORES) {
	dumpScores(cerr, trainSet);
    }

    if (skipopt ) {
	if (printHyps) {
	    oneBest = 1;
	    bestLambdas = lambdaInitials;        
	    outputHyps(trainSet);
	}

	if (!computeOracle) exit(0);
    }

    cerr << ((errorsDir || bleuCountsDir || srinterpCountsFile) ? "reading" : "computing") << " error counts...\n";

    if (srinterpCountsFile) {
        File file(srinterpCountsFile, "r");

	if (!trainSet.readSRInterpCountsFile(file, numRefWords, bleuNgram)) {
	    cerr << "failed to read SRInterp counts file from " << srinterpCountsFile << endl;
	    exit (2);
	}
    } else {
	/*
	 * Compute hyp errors
	 */
	numRefWords = prepareErrorCounts(trainSet, refs, antiRefs, nullLM);

	if (xvalFiles) {
	    numXvalWords = prepareErrorCounts(xvalSet, refs, antiRefs, nullLM);
	}
    }

    if (optimizeBleu) {
        bleuRefLength = numRefWords;
    }
    
    cerr << numRefWords << " reference words\n";
    if (xvalFiles) {
	cerr << numXvalWords << " xval reference words\n";
    }

    /*
     * preemptive trouble avoidance: prevent division by zero
     */
    if (numRefWords == 0) {
	numRefWords = 1;        
    }

    if (skipopt && computeOracle) {
	outputOracle(trainSet);
	exit(0);
    }
    
#ifndef NO_TIMEOUT
    /*
     * set up search time-out handler
     */
    if (maxTime) {
	signal(SIGALRM, catchAlarm);
    }
#endif /* !NO_TIMEOUT */

    double oldPosteriorScaleStep = posteriorScaleStep;
 
    if (oneBest || oneBestFirst) {
    	oneBest = true;
	posteriorScaleStep = 0.0;
	
	cerr << "Posterior scale step size set to " << posteriorScaleStep
	     << endl;

	unsigned errors, xvalErrors;

	errors = (int) computeErrors(trainSet, lambdas.data());
	if (xvalFiles) {
	    xvalErrors = (int) computeErrors(xvalSet, lambdas.data());
	}

	printLambdas(cout, lambdas);

	if (initSimplex == 0 && initPowell == 0) {
	    train(trainSet, xvalSet);
	} else if (initSimplex) {
	    trainAmoeba(trainSet, xvalSet);
	} else {          
	    trainPowellQuick(trainSet);
	}

        if (!optimizeBleu) {
	    cout << "original errors = " << errors
		 << " (" << ((double)errors/numRefWords) << "/word)"
		 << endl;
	    cout << "best errors = " << bestError
		 << " (" << ((double)bestError/numRefWords) << "/word)" 
		 << endl;
	    if (numXvalWords) {
		cout << "original xval errors = " << xvalErrors
		     << " (" << ((double)xvalErrors/numXvalWords) << "/word)"
		     << endl;
		cout << "xval errors = " << bestXvalError
		     << " (" << ((double)bestXvalError/numXvalWords) << "/word)" 
		     << endl;
	    }
        } else if (errorBleuRatio == 0) {
	    double bleu = 1.0 - (errors / bleuScale);
	    double bestBleu = 1.0 - (bestError / bleuScale);

	    cout << "original bleu = " << bleu << endl;
	    cout << "best bleu = " << bestBleu << endl;          
        } else {
	    double met = 1.0 - (errors / bleuScale);
	    double bestMet = 1.0 - (bestError / bleuScale);
	    
	    cout << "original metric = " << met << endl;
	    cout << "best metric = " << bestMet << endl;
	}
    }

    if (oneBestFirst) {
	// restart search at best point found in 1-best search
	lambdas = bestLambdas;

	// scale weights to LMW==1
	if (lambdas[1] != 0.0) {
	    posteriorScale = lambdas[1];
	    for (unsigned i = 0; i < numScores; i ++) {
		lambdas[i] /= posteriorScale;
	    }
	}
    }

    if (!oneBest || oneBestFirst) {
	unsigned errors, xvalErrors;

    	oneBest = false;

        posteriorScaleStep = oldPosteriorScaleStep;
	cerr << "Posterior scale step size set to " << posteriorScaleStep
	     << endl;

	cerr << "aligning nbest lists...\n";
	if (dictFile) {
	    alignNbest(trainSet, refs, dictDistance);
	    if (xvalFiles) {
		alignNbest(xvalSet, refs, dictDistance);
	    }
	} else if (distanceFile) {
	    alignNbest(trainSet, refs, matrixDistance);
	    if (xvalFiles) {
		alignNbest(xvalSet, refs, matrixDistance);
	    }
	} else {
	    // last argument is only used if hiddenVocabFile != 0
	    alignNbest(trainSet, refs, subvocabDistance);
	    if (xvalFiles) {
		alignNbest(xvalSet, refs, subvocabDistance);
	    }
	}
 
	errors = (int) computeErrors(trainSet, lambdas.data());
	if (xvalFiles) {
	    xvalErrors = (int) computeErrors(xvalSet, lambdas.data());
	}
	printLambdas(cout, lambdas);

	if (initSimplex == 0) {
	    train(trainSet, xvalSet);
	} else {
	    trainAmoeba(trainSet, xvalSet);
	}
	cout << "original errors = " << errors
	     << " (" << ((double)errors/numRefWords) << "/word)"
	     << endl;
	cout << "best errors = " << bestError
	     << " (" << ((double)bestError/numRefWords) << "/word)" 
	     << endl;
	if (numXvalWords) {
	    cout << "original xval errors = " << xvalErrors
		 << " (" << ((double)xvalErrors/numXvalWords) << "/word)"
		 << endl;
	    cout << "best xval errors = " << bestXvalError
		 << " (" << ((double)bestXvalError/numXvalWords) << "/word)" 
		 << endl;
	}
    }

    printLambdas(cout, bestLambdas, writeRoverControl);

    if (printHyps) {
	outputHyps(trainSet);
    }

    if (computeOracle) {
	outputOracle(trainSet);
    }
    
    exit(0);
}

