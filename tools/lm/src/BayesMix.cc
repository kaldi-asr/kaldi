/*
 * BayesMix.cc --
 *	Bayesian mixture language model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2006 SRI International, 2012-2013 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/BayesMix.cc,v 1.26 2014-08-29 21:35:48 frandsen Exp $";
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
#include <assert.h>

#include "option.h"

#include "BayesMix.h"
#include "NullLM.h"
#include "ClassNgram.h"
#include "SimpleClassNgram.h"
#include "NgramCountLM.h"
#include "MEModel.h"
#include "LMClient.h"
#include "MSWebNgramLM.h"

#include "Array.cc"

/*
 * Debug levels used
 */
#define DEBUG_MIX_WEIGHTS	2
#define DEBUG_LM_PROBS		3

BayesMix::BayesMix(Vocab &vocab, unsigned int clength, double llscale)
    : LM(vocab), numLMs(1), priors(0, 1), prior(priors[0]),
      clength(clength), llscale(llscale), subLMs(0, 1), deleteSubLMs(true),
      useContextPriors(false), contextPriors(vocab, clength, 1)
{
    /*
     * Initialize with a dummy LM
     */
    subLMs[0] = new NullLM(vocab);
    assert(subLMs[0] != 0);

    priors[0] = 1.0;
}

BayesMix::BayesMix(Vocab &vocab, LM &lm1, LM &lm2,
			    unsigned int clength, Prob pr, double llscale)
    : LM(vocab), numLMs(2), priors(0, 2), prior(priors[0]),
      clength(clength), llscale(llscale), subLMs(0, 2), deleteSubLMs(false),
      useContextPriors(false), contextPriors(vocab, clength, 2)
{
    if (pr < 0.0 || pr > 1.0) {
	cerr << "warning: mixture prior out of range: " << pr << endl;
	pr = 0.5;
    }

    subLMs[0] = &lm1;
    subLMs[1] = &lm2;

    priors[0] = pr;
    priors[1] = 1.0 - pr;
}

BayesMix::BayesMix(Vocab &vocab, Array<LM *> &lms, Array<Prob> &priors,
			    unsigned int clength, double llscale)
    : LM(vocab), numLMs(lms.size()), priors(priors), prior(priors[0]),
      clength(clength), llscale(llscale), subLMs(0, numLMs), deleteSubLMs(false),
      useContextPriors(false), contextPriors(vocab, clength, numLMs)
{
    assert(numLMs > 0);

    /*
     * Check priors for sanity and normalize
     */
    unsigned i;
    Prob priorSum = 0.0;
    for (i = 0; i < numLMs; i++) {
        if (priors[i] < 0.0 || priors[i] > 1.0) {
	    cerr << "warning: mixture prior out of range: " << priors[i] << endl;
	    priors[i] = priors[i] < 0.0 ? 0.0 : 1.0;
	}
	priorSum += priors[i];
    }
    for (i = 0; i < numLMs; i++) {
	priors[i] /= priorSum;
	subLMs[i] = lms[i];
    }
}

BayesMix::~BayesMix()
{
    if (deleteSubLMs) {
	for (unsigned i = 0; i < numLMs; i++) {
	    delete subLMs[i];
	}
    }
}

Boolean
BayesMix::readMixLMs(File &file, Boolean limitVocab, Boolean ngramOnly)
{
    char *line;
    unsigned numMixtures = 0;
    Prob priorSum = 0.0;

    if (deleteSubLMs) {
	for (unsigned i = 0; i < numLMs; i++) {
	    delete subLMs[i];
	}
	deleteSubLMs = false;
    }

    subLMs.clear();
    priors.clear();

    /*
     * Context priors cannot be guaranteed to be consistent with new component LMs,
     * so clear them.
     */
    contextPriors.clear();
    useContextPriors = false;

    while ((line = file.getline())) {
	VocabString argv[maxWordsPerLine + 1];

	unsigned argc = Vocab::parseWords(line, argv, maxWordsPerLine + 1);
	if (argc > maxWordsPerLine) {
	    file.position() << "too many words in line\n";
	    return false;
	}

	if (argc == 0) {
	    continue;
	}

	/*
	 * First word on line defines the LM file name
	 */
	const char *lmFile = argv[0];
	unsigned lmOrder = 3;
	double lmWeight = 1.0;
	const char *lmType = "ARPA";	// the default
	const char *classesFile = 0;
	int simpleClasses = 0;
	int cacheServedNgrams = 0;

	Option options[] = {
		{ OPT_UINT, "order", &lmOrder, "lm ngram order" },
		{ OPT_FLOAT, "weight", &lmWeight, "lm prior weight" },
		{ OPT_STRING, "type", &lmType, "lm type" },
		{ OPT_STRING, "classes", &classesFile, "class definitions" },
		{ OPT_TRUE, "simple-classes", &simpleClasses, "use unique class model" },
		{ OPT_TRUE, "cache-served-ngrams", &cacheServedNgrams, "enable client side caching" },
	};

	if (Opt_Parse(argc, (char **)argv, options, Opt_Number(options),
						    OPT_UNKNOWN_IS_ERROR) != 1)
	{
	    file.position() << "allowed options for mixture LM " << lmFile << " are\n";
	    Opt_PrintUsage(NULL, options, Opt_Number(options));
	    return false;
	}

	LM *lm = 0;

	if (strcmp(lmType, "ARPA") == 0) {
	    /*
	     * Read Ngram LM in ARPA format
	     */
	    SubVocab *classVocab = 0;
	    if (classesFile != 0) {
		classVocab = new SubVocab(vocab);
		assert(classVocab != 0);
	    }

	    Ngram *ngram = classesFile != 0 ?
			    (simpleClasses ?
				new SimpleClassNgram(vocab, *classVocab, lmOrder) :
				new ClassNgram(vocab, *classVocab, lmOrder)) :
			    new Ngram(vocab, lmOrder);
	    assert(ngram != 0);

	    File file(lmFile, "r");
	    if (!ngram->read(file, limitVocab)) {
		file.position() << "error in ngram lm" << lmFile << endl;
		delete ngram;
		delete classVocab;
		return false;
	    }

	    if (classesFile != 0) {
		File cfile(classesFile, "r");
		if (!((ClassNgram *)ngram)->readClasses(cfile)) {
		    file.position() << "error in class defintions lm"
			 	    << classesFile << endl;
		    delete ngram;
		    delete classVocab;
		    return false;
		}
	    }
	    lm = ngram;
	} else if (!ngramOnly && strcmp(lmType, "COUNTLM") == 0) {
	    /*
	     * Read an Ngram-count LM
	     */
	    File file(lmFile, "r");
	    NgramCountLM *countlm = new NgramCountLM(vocab, lmOrder);
	    assert(countlm != 0);

	    if (!countlm->read(file, limitVocab)) {
		cerr << "error in count-lm " << lmFile << endl;
		delete countlm;
		return false;
	    }
	    lm = countlm;
	} else if (strcmp(lmType, "MAXENT") == 0) {
	    /*
	     * Read a Maxent LM
	     */
	    File file(lmFile, "r");
	    MEModel *meLM = new MEModel(vocab);
	    assert(meLM != 0);

	    if (!meLM->read(file, limitVocab)) {
		cerr << "error in maxent lm " << lmFile << endl;
		delete meLM;
		return false;
	    }

	    if (ngramOnly) {
		lm = meLM->getNgramLM();
		delete meLM;
	    } else {
		lm = meLM;
	    }
	} else if (!ngramOnly && strcmp(lmType, "LMCLIENT") == 0) {
	    /*
	     * Create an LM client -- the "filename" is the network address
	     */
    	    LMClient *lmClient = new LMClient(vocab, lmFile, lmOrder,
						cacheServedNgrams ? lmOrder : 0);
	    assert(lmClient != 0);
	    lm = lmClient;
	} else if (!ngramOnly && strcmp(lmType, "MSWEBLM") == 0) {
	    /*
	     * Read a MS Web-Ngram LM
	     */
	    File file(lmFile, "r");
	    MSWebNgramLM *weblm = new MSWebNgramLM(vocab, lmOrder,
						cacheServedNgrams ? lmOrder : 0);
	    assert(weblm != 0);

	    if (!weblm->read(file, limitVocab)) {
		cerr << "error in creating MS Web-Ngram LM " << lmFile << endl;
		delete weblm;
		return false;
	    }
	    lm = weblm;
	} else {
	    file.position() << lmType << " is not a valid LM type\n";
	    return false;
	}

	subLMs[numMixtures] = lm;
	priors[numMixtures] = lmWeight;
	priorSum += lmWeight;

	numMixtures += 1;
	// @kw false positive: RH.LEAK (lmClient->serverSocket)
    }

    numLMs = numMixtures;
    contextPriors.setdim(numLMs);
    deleteSubLMs = true;

    for (unsigned i = 0; i < numLMs; i++) {
	priors[i] /= priorSum;
    }

    return true;
}

Boolean
BayesMix::readContextPriors(File &file, Boolean limitVocab)
{
    useContextPriors = contextPriors.read(file, clength, limitVocab);

    /*
     * XXX: should normalize all priors before use
     */
    return useContextPriors;
}

Array<Prob> &
BayesMix::findPriors(const VocabIndex *context)
{
    if (useContextPriors) {
	unsigned contextLen = Vocab::length(context);

	if (contextLen > clength) contextLen = clength;

	TruncatedContext usedContext(context, contextLen);

	unsigned depth;
	Array<Prob> *cdPriors = contextPriors.findPrefixProbs(usedContext, depth);
	assert(cdPriors != 0);

	/*
	 * If priors are all zero, fall back on global priors
	 * This will keep us from using contexts that were
	 * added implicitly into the prior trie.
	 */
	for (unsigned i = 0; i < numLMs; i++) {
	    if ((*cdPriors)[i] != 0.0) {
		return *cdPriors;
	    }
	}
    }

    return priors;
}

LogP
BayesMix::wordProb(VocabIndex word, const VocabIndex *context)
{
    makeArray(Prob, lmProbs, numLMs);
    makeArray(Prob, lmWeights, numLMs);

    unsigned i;
    Prob lmWeightSum = 0.0;
    Boolean allZeroWeights = true;

    Array<Prob> &usePriors = findPriors(context);

    for (i = 0; i < numLMs; i++) {
	lmProbs[i] = LogPtoProb(subLMs[i]->wordProb(word, context));

	lmWeights[i] = usePriors[i];
	if (llscale > 0.0) {
	    lmWeights[i] *=
		LogPtoProb(llscale * subLMs[i]->contextProb(context, clength));
	}

	if (lmWeights[i] != 0.0) {
	    allZeroWeights = false;
	}
	lmWeightSum += lmWeights[i];
    }

    /*
     * If none of the LMs know this context revert to the prior
     */
    if (allZeroWeights) {
        lmWeightSum = 0.0;
	for (i = 0; i < numLMs; i++) {
	    lmWeights[i] = usePriors[i];
	    lmWeightSum += lmWeights[i];
	}
    }

    if (running() && debug(DEBUG_MIX_WEIGHTS)) {
	if (clength > 0) {
	    dout() << "[post=";
	    if (i > 0) dout() << (lmWeights[0]/lmWeightSum);
	    for (unsigned i = 1; i < numLMs; i ++) {
		dout() << "," << (lmWeights[i]/lmWeightSum);
	    }
	    dout() << "]";
	}
    }
    if (running() && debug(DEBUG_LM_PROBS)) {
	dout() << "[probs=";
	if (i > 0) dout() << lmProbs[0];
	for (unsigned i = 1; i < numLMs; i ++) {
	    dout() << "," << lmProbs[i];
	}
	dout() << "]";
    }

    Prob totalProb = 0.0;
    for (i = 0; i < numLMs; i++) {
	totalProb += lmWeights[i] * lmProbs[i];
    }

    return ProbToLogP(totalProb / lmWeightSum);
}

void *
BayesMix::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
{
    /*
     * Return the context ID of the component model that uses the longer
     * context. We must use longest context regardless of predicted word
     * because mixture models don't support contextBOW().
     */
    void *contextID = 0;
    unsigned maxContextLen = 0;

    for (unsigned i = 0; i < numLMs; i++) {
 	unsigned clen;
	void *cid = subLMs[i]->contextID(context, clen);

	if (clen > maxContextLen) {
	    maxContextLen = clen;
	    contextID = cid;
	}
    }

    length = maxContextLen;
    return contextID;
}

Boolean
BayesMix::isNonWord(VocabIndex word)
{
    /*
     * A non-word in either of our component models is a non-word.
     * This ensures that state names, hidden vocabulary, etc. are not
     * treated as regular words in the respectively other component.
     */
    for (unsigned i = 0; i < numLMs; i++) {
	if (subLMs[i]->isNonWord(word)) {
	    return true;
	}
    }
    return false;
}

void
BayesMix::setState(const char *state)
{
    /*
     * Global state changes are propagated to the component models
     */
    for (unsigned i = 0; i < numLMs; i++) {
	subLMs[i]->setState(state);
    }
}

Boolean
BayesMix::addUnkWords()
{
    for (unsigned i = 0; i < numLMs; i++) {
	if (subLMs[i]->addUnkWords()) {
	    return true;
	}
    }
    return false;
}

Boolean
BayesMix::running(Boolean newstate)
{
    Boolean old = _running; _running = newstate; 

    /*
     * Propagate changes to running state to component models
     */
    for (unsigned i = 0; i < numLMs; i++) {
	subLMs[i]->running(newstate);
    }
    return old;
}

void
BayesMix::debugme(unsigned level)
{
    /*
     * Propagate changes to Debug state to component models
     */
    for (unsigned i = 0; i < numLMs; i++) {
	subLMs[i]->debugme(level);
    }

    Debug::debugme(level);
}

ostream &
BayesMix::dout(ostream &stream)
{
    /*
     * Propagate dout changes to sub-lms
     */
    for (unsigned i = 0; i < numLMs; i++) {
	subLMs[i]->dout(stream);
    }
    return Debug::dout(stream);
}

unsigned
BayesMix::prefetchingNgrams()
{
    /*
     * Propagate prefetching protocol to component models
     */
    unsigned maxpf = 0;

    for (unsigned i = 0; i < numLMs; i++) {
	unsigned pf = subLMs[i]->prefetchingNgrams();

        if (pf > maxpf) maxpf = pf;
    }
    return maxpf;
}

Boolean
BayesMix::prefetchNgrams(NgramCounts<Count> &ngrams)
{
    Boolean result = true;

    for (unsigned i = 0; i < numLMs; i++) {
	result = result && subLMs[i]->prefetchNgrams(ngrams);
    }
    return result;
}

Boolean
BayesMix::prefetchNgrams(NgramCounts<XCount> &ngrams)
{
    Boolean result = true;

    for (unsigned i = 0; i < numLMs; i++) {
	result = result && subLMs[i]->prefetchNgrams(ngrams);
    }
    return result;
}

Boolean
BayesMix::prefetchNgrams(NgramCounts<FloatCount> &ngrams)
{
    Boolean result = true;

    for (unsigned i = 0; i < numLMs; i++) {
	result = result && subLMs[i]->prefetchNgrams(ngrams);
    }
    return result;
}

