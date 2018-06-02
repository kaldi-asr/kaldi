/*
 * fngram --
 *	Create and manipulate fngram models
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2009 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: fngram.cc,v 1.76 2013/03/05 05:54:17 stolcke Exp $";
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
#include <assert.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <time.h>

#ifdef NEED_RAND48
extern "C" {
    void srand48(long);
}
#endif

#include "option.h"
#include "version.h"
#include "File.h"
#include "Vocab.h"
#include "SubVocab.h"
#include "NBest.h"
#include "Ngram.h"
#include "FNgram.h"
#include "FNgramStats.h"
#include "NullLM.h"
#include "DecipherNgram.h"
#include "hexdec.h"

static int version = 0;
static unsigned order = defaultNgramOrder;
static unsigned debug = 0;
static char *pplFile = 0;
static char *escape = 0;
static char *countFile = 0;
static unsigned countOrder = 0;
static char *vocabFile = 0;
static char *nonEvent = 0;
static char *noneventFile = 0;
static int reverseSents = 0;
static int writeLM  = 0;
static char *writeVocab  = 0;
static int memuse = 0;
static int skipOOVs = 0;
static int seed = 0;  /* default dynamically generated in main() */
static int toLower = 0;
static int keepunk = 0;
static int keepnull = 1;
static char *noiseTag = 0;
static char *noiseVocabFile = 0;
static int virtualBeginSentence = 1;
static int virtualEndSentence = 1;
static int noScoreSentenceBoundaryMarks = 0;
/*
 * N-Best related variables
 */

static char *factorFile = 0;
static char *nbestFile = 0;
static unsigned maxNbest = 0;
static char *rescoreFile = 0;
static double rescoreLMW = 8.0;
static double rescoreWTW = 0.0;
static int combineLMScores = 1;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_STRING, "factor-file", &factorFile, "build a factored LM, use factors given in file" },
    { OPT_UINT, "debug", &debug, "debugging level for lm" },
    { OPT_TRUE, "skipoovs", &skipOOVs, "skip n-gram contexts containing OOVs" },
    { OPT_TRUE, "unk", &keepunk, "vocabulary contains <unk>" },
    { OPT_FALSE, "nonull", &keepnull, "remove <NULL> in LM" },
    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },

    { OPT_STRING, "ppl", &pplFile, "text file to compute perplexity from" },
    { OPT_STRING, "escape", &escape, "escape prefix to pass data through -ppl" },
    { OPT_INT, "seed", &seed, "seed for randomization" },
    { OPT_STRING, "vocab", &vocabFile, "vocab file" },
    { OPT_STRING, "non-event", &nonEvent, "non-event word" },
    { OPT_STRING, "nonevents", &noneventFile, "non-event vocabulary file" },

    { OPT_FALSE, "no-virtual-begin-sentence", &virtualBeginSentence, "Do *not* use a virtual start sentence context at the sentence begin"},
    { OPT_FALSE, "no-virtual-end-sentence", &virtualEndSentence, "Do *not* use a virtual end sentence context at the sentence end"},

    { OPT_TRUE, "no-score-sentence-marks", &noScoreSentenceBoundaryMarks, "Do *not* score the sentence boundary marks <s> </s>, score only words in-between"},
    { OPT_TRUE, "write-lm", &writeLM, "re-write LM to file" },
    { OPT_STRING, "write-vocab", &writeVocab, "write LM vocab to file" },
    // not currently implemented
    //    { OPT_TRUE, "memuse", &memuse, "show memory usage" },
    { OPT_STRING, "rescore", &rescoreFile, "hyp stream input file to rescore" },
    { OPT_FALSE, "separate-lm-scores", &combineLMScores, "print separate lm scores in n-best file" },
    { OPT_FLOAT, "rescore-lmw", &rescoreLMW, "rescoring LM weight" },
    { OPT_FLOAT, "rescore-wtw", &rescoreWTW, "rescoring word transition weight" },
    { OPT_STRING, "noise", &noiseTag, "noise tag to skip" },
    { OPT_STRING, "noise-vocab", &noiseVocabFile, "noise vocabulary to skip" },
};

int
main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    /* set default seed for randomization */
#ifndef _MSC_VER
    seed = time(NULL) + getpid();
#else
	seed = time(NULL);
#endif

    // print 0x in front of hex numbers.
    SHOWBASE(cout);
    SHOWBASE(cerr);

    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    /*
     * Set random seed
     */
    srand48((long)seed);

    /*
     * Construct language model
     */

    if (factorFile == 0) {
	fprintf(stderr,"Error: must specify factor file\n");
	exit(-1);
    }

    FactoredVocab *vocab = new FactoredVocab;
    assert(vocab != 0);

    FNgramSpecs<FNgramCount>* fnSpecs = 0; 
    File f(factorFile,"r");
    fnSpecs = new FNgramSpecs<FNgramCount>(f,*vocab,debug);
    if (!fnSpecs) {
	fprintf(stderr,"Error creating fnspecs object");
	exit(-1);
    }
    
    vocab->unkIsWord() = keepunk ? true : false;
    vocab->nullIsWord() = keepnull ? true : false;
    vocab->toLower() = toLower ? true : false;

    // for now, load in the stats object since we need the counts
    // to make decisions. Ultimately, this will be entirely contained
    // within the LM file.
    FNgramStats *factoredStats = new FNgramStats(*vocab, *fnSpecs);
    assert(factoredStats != 0);

    factoredStats->debugme(debug);

    if (vocabFile) {
	File file(vocabFile, "r");
	factoredStats->vocab.read(file);
	factoredStats->openVocab = false;
    }

    if (noneventFile) {
	/*
	 * create temporary sub-vocabulary for non-event words
	 */
	SubVocab nonEvents(*vocab);

	File file(noneventFile, "r");
	nonEvents.read(file);

	vocab->addNonEvents(nonEvents);
    }
    if (nonEvent) {
	vocab->addNonEvent(nonEvent);
    }

    FNgram* fngramLM = new FNgram(*vocab,*fnSpecs);
    assert(fngramLM != 0);

    fngramLM->debugme(debug);

    fngramLM->virtualBeginSentence = virtualBeginSentence ? true : false;
    fngramLM->virtualEndSentence = virtualEndSentence ? true : false;
    fngramLM->noScoreSentenceBoundaryMarks = noScoreSentenceBoundaryMarks ? true : false;

    if (skipOOVs) {
	fngramLM->skipOOVs = true;
    }

    /*
     * Read just a single LM
     */

    // readin the counts, we need to do this for now.
    // TODO: change so that counts are not needed for ppl/rescoring.
    if (!factoredStats->read()) {
	cerr << "error reading in counts in factor file\n";
	exit(1);
    }

    // We need to do this here so that we get the
    // same GBO strategy that we got when the LM was estimated.
    // TODO: put the resulting counts used when the LM was trained
    // into the lm file, making the LM file self contained.

    factoredStats->estimateDiscounts();
    factoredStats->computeCardinalityFunctions();
    factoredStats->sumCounts();

    if (!fngramLM->read()) {
	cerr << "format error in lm file\n";
	exit(1);
    }

    /*
     * Reverse words in scoring
     */
    if (reverseSents) {
	fngramLM->reverseWords = true;
    }

    /*
     * Skip noise tags in scoring
     */
    if (noiseVocabFile) {
	File file(noiseVocabFile, "r");
	fngramLM->noiseVocab.read(file);
    }

    if (noiseTag) {				/* backward compatibility */
	fngramLM->noiseVocab.addWord(noiseTag);
    }

#if 0
    if (memuse) {
        // TODO: get all the memuse stuff working in all factored model code
	MemStats memuse;
	fngramLM->memStats(memuse);
	memuse.print();
    }
#endif

    /*
     * Compute perplexity on a text file, if requested
     */
    if (pplFile) {
	File file(pplFile, "r");
	TextStats stats;
	/*
	 * Send perplexity info to stdout 
	 */
	fngramLM->dout(cout);
	fngramLM->pplFile(file, stats, escape);
	fngramLM->pplPrint(cout, pplFile);
	fngramLM->dout(cerr);
    }

    /*
     * Compute perplexity on a count file, if requested
     */
    if (countFile && 0) { // TODO: not yet implemented
	TextStats stats;
	File file(countFile, "r");
	/*
	 * Send perplexity info to stdout 
	 */
	fngramLM->dout(cout);
	fngramLM->pplCountsFile(file, countOrder ? countOrder : order,
							    stats, escape);
	fngramLM->dout(cerr);

	cout << "file " << countFile << ": " << stats;
    }

    // TODO: add generate option.

    /*
     * Rescore stream of N-best hyps, if requested
     */
    if (rescoreFile) {
	File file(rescoreFile, "r");
	NullLM nullLM(fngramLM->vocab);

	fngramLM->combineLMScores = combineLMScores;
	fngramLM->rescoreFile(file, rescoreLMW, rescoreWTW, 
			      nullLM, 0, 0,
			      escape);
    }

    if (writeLM) {
	fngramLM->write();
    }

    if (writeVocab) {
	File file(writeVocab, "w");
	vocab->write(file);
    }

#ifdef DEBUG
    delete fngramLM;
    delete vocab;
    delete fnSpecs;
    delete factoredStats;
    return 0;
#endif /* DEBUG */

    exit(0);
}

