/*
 * fngram-count --
 *	factored ngram counts program, counts factored language files
 *      and builds factored language models based on them using
 *      a variety of general graph-based backoff algorithms.
 *
 *  Jeff Bilmes <bilmes@ee.washington.edu>, but uses code from ngram-count.cc
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2009 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: fngram-count.cc,v 1.57 2012/05/17 06:46:49 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <locale.h>
#include <assert.h>

#include "option.h"
#include "version.h"
#include "File.h"
#include "Vocab.h"
#include "SubVocab.h"
#include "Ngram.h"
#include "VarNgram.h"
#include "TaggedNgram.h"
#include "SkipNgram.h"
#include "StopNgram.h"
#include "NgramStats.h"
#include "TaggedNgramStats.h"
#include "StopNgramStats.h"
#include "FNgramStats.h"
#include "FNgramSpecs.h"
#include "FNgram.h"
#include "FDiscount.h"
#include "hexdec.h"

const unsigned maxorder = 9;		/* this is only relevant to the 
					 * the -gt<n> and -write<n> flags */
static int version = 0;
static int writeCounts = 0;
static int writeCountsAfterLM = 0;
static int lm = 0;
static char *filetag = 0;
static unsigned order = 3;
static unsigned debug = 0;
static char *textFile = 0;
static int textFileHasWeights = 0;
static int readCounts = 0;

static int knCountsModified = 0;
static int virtualBeginSentence = 1;
static int virtualEndSentence = 1;

static int addStartSentenceToken = 1;
static int addEndSentenceToken = 1;

static char *lmFile = 0;
static char *initLMFile = 0;

static char *vocabFile = 0;
static char *nonEvent = 0;
static char *noneventFile = 0;
static char *writeVocab = 0;
static int sortNgrams = 0;
static int keepunk = 0;
static int keepnull = 1;
static char *factorFile = 0;
static int tagged = 0;
static int toLower = 0;
static int trustTotals = 0;
static double prune = 0.0;
static unsigned minprune = 2;
static int useFloatCounts = 0;

static double varPrune = 0.0;

static int skipNgram = 0;
static double skipInit = 0.5;
static unsigned maxEMiters = 100;
static double minEMdelta = 0.001;

static char *stopWordFile = 0;
static char *metaTag = 0;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_STRING, "factor-file", &factorFile, "build a factored LM, use factors given in file" },
    { OPT_UINT, "debug", &debug, "debugging level for LM" },
    { OPT_TRUE, "sort", &sortNgrams, "sort ngrams output" },
    { OPT_STRING, "text", &textFile, "text file to read" },
    { OPT_TRUE, "text-has-weights", &textFileHasWeights, "text file contains count weights" },
    { OPT_TRUE, "read-counts", &readCounts, "try to read counts first" },
    { OPT_TRUE, "write-counts", &writeCounts, "write counts to file(s)" },
    { OPT_TRUE, "write-counts-after-lm-train", &writeCountsAfterLM, "write counts to file(s) after LM training" },
    { OPT_TRUE, "lm", &lm, "estimate and write lm to file(s)" },
    { OPT_TRUE, "kn-counts-modified", &knCountsModified, "input counts already modified for KN smoothing"},

    { OPT_FALSE, "no-virtual-begin-sentence", &virtualBeginSentence, "Do *not* use a virtual start sentence context at the sentence begin"},
    { OPT_FALSE, "no-virtual-end-sentence", &virtualEndSentence, "Do *not* use a virtual end sentence context at the sentence end"},

    { OPT_FALSE, "no-add-start-sentence-token", &addStartSentenceToken, "Do *not* add a start sentence token to count if it doesn't exist in text file"},
    { OPT_FALSE, "no-add-end-sentence-token", &addEndSentenceToken, "Do *not* add an end sentence token to count if it doesn't exist in text file"},


    { OPT_TRUE, "unk", &keepunk, "keep <unk> in LM" },
    { OPT_FALSE, "nonull", &keepnull, "remove <NULL> in LM" },
    { OPT_STRING, "meta-tag", &metaTag, "meta tag used to input count-of-count information" },
    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },
    { OPT_STRING, "vocab", &vocabFile, "vocab file" },
    { OPT_STRING, "non-event", &nonEvent, "non-event word" },
    { OPT_STRING, "nonevents", &noneventFile, "non-event vocabulary file" },
    { OPT_STRING, "write-vocab", &writeVocab, "write vocab to file" },
    { OPT_DOC, 0, 0, "the default action is to write counts to stdout" }
};

int
main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    // print 0x in front of hex numbers.
    SHOWBASE(cout);
    SHOWBASE(cerr);

    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    if (factorFile == 0) {
	fprintf(stderr,"Error: must specify factor file\n");
	exit(-1);
    }
    
    FactoredVocab *vocab = new FactoredVocab;
    assert(vocab);

    FNgramStats *factoredStats = 0;
    FNgramSpecs<FNgramCount>* fnSpecs = 0; 
    File f(factorFile,"r");
    fnSpecs = new FNgramSpecs<FNgramCount>(f,*vocab,debug);
    if (!fnSpecs) {
	fprintf(stderr,"Error creating fnspecs objecdt");
	exit(-1);
    }

    vocab->unkIsWord() = keepunk ? true : false;
    vocab->nullIsWord() = keepnull ? true : false;
    vocab->toLower() = toLower ? true : false;

    /*
     * Meta tag is used to input count-of-count information
     */
    if (metaTag) {
	vocab->metaTag() = metaTag;
    }

    factoredStats = new FNgramStats(*vocab, *fnSpecs);

    factoredStats->debugme(debug);
    factoredStats->virtualBeginSentence = virtualBeginSentence ? true : false;
    factoredStats->virtualEndSentence = virtualEndSentence ? true : false;

    factoredStats->addStartSentenceToken = addStartSentenceToken ? true : false;
    factoredStats->addEndSentenceToken = addEndSentenceToken ? true : false;

    if (vocabFile) {
	File file(vocabFile, "r");
	factoredStats->vocab.read(file);
	factoredStats->openVocab = false;
    }

    if (noneventFile) {
	/*
	 * create temporary sub-vocabulary for non-event words
	 */
	SubVocab nonEvents(factoredStats->vocab);

	File file(noneventFile, "r");
	nonEvents.read(file);

	factoredStats->vocab.addNonEvents(nonEvents);
    }
    if (nonEvent) {
        factoredStats->vocab.addNonEvent(nonEvent);
    }

    if (readCounts) {
        factoredStats->read();
    }

    if (textFile) {
	File file(textFile, "r");
	factoredStats->countFile(file, textFileHasWeights);
    }

    if (writeCounts) {
	factoredStats->write(sortNgrams);
    }

    if (lm) {
	// estimateDiscounts might change the count tries if it 
	// is kn smoothing
	factoredStats->estimateDiscounts();
	// sumCounts will change lower levels in the count tries, and is
	// needed for certain graph backoff strategies.
	factoredStats->computeCardinalityFunctions();

	if (writeCountsAfterLM) {
	    // estimate the discounts which might effect the count 
	    // if we don't have a lm.
	    printf("** Writing counts after Discount estimation **\n");
	    factoredStats->write(sortNgrams);
	}
	// sum up the individual count tries.
	factoredStats->sumCounts(); 
	FNgram flm(*vocab,*fnSpecs); 
	flm.debugme(debug);
	flm.estimate();
	flm.write();
    }

    if (writeVocab) {
	File file(writeVocab, "w");
	vocab->write(file);
    }

#ifdef DEBUG
    /*
     * Free all objects
     */

    delete vocab;
    return(0);
#endif /* DEBUG */

    exit(0);
}

