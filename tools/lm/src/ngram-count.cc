/*
 * ngram-count --
 *	Create and manipulate word ngram counts
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2011 SRI International, 2012-2015 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: ngram-count.cc,v 1.79 2015-10-13 07:08:49 stolcke Exp $";
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
#include "Discount.h"
#include "NgramCountLM.h"
#include "Array.cc"
#include "MEModel.h"
#include "BlockMalloc.h"

const unsigned maxorder = 9;		/* this is only relevant to the 
					 * the -gt<n> and -write<n> flags */
static int version = 0;
static char *filetag = 0;
static unsigned order = 3;
static unsigned debug = 0;
static char *textFile = 0;
static int textFileHasWeights = 0;
static int noSOS = 0;
static int noEOS = 0;
static char *readFile = 0;
static char *intersectFile = 0;
static int readWithMincounts = 0;
static char *readGoogleDir = 0;

static unsigned writeOrder = 0;		/* default is all ngram orders */
static char *writeFile[maxorder+1];
static char *writeBinaryFile;

static double gtmin[maxorder+1] = {1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
static unsigned gtmax[maxorder+1] = {5, 1, 7, 7, 7, 7, 7, 7, 7, 7};

static double cdiscount[maxorder+1] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
static double addsmooth[maxorder+1] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
static int ndiscount[maxorder+1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int wbdiscount[maxorder+1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int kndiscount[maxorder+1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int ukndiscount[maxorder+1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int knCountsModified = 0;
static int knCountsModifyAtEnd = 0;
static int interpolate[maxorder+1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static char *gtFile[maxorder+1];
static char *knFile[maxorder+1];
static char *lmFile = 0;
static int writeBinaryLM = 0;
static char *initLMFile = 0;

static char *vocabFile = 0;
static char *vocabAliasFile = 0;
static char *noneventFile = 0;
static int limitVocab = 0;
static char *writeVocab = 0;
static char *writeVocabIndex = 0;
static char *writeTextFile = 0;
static int memuse = 0;
static int recompute = 0;
static int sortNgrams = 0;
static int keepunk = 0;
static char *mapUnknown = 0;
static int tagged = 0;
static int toLower = 0;
static int trustTotals = 0;
static double prune = 0.0;
static unsigned minprune = 2;
static int useFloatCounts = 0;

static double varPrune = 0.0;

static int useCountLM = 0;
static int skipNgram = 0;
static double skipInit = 0.5;
static unsigned maxEMiters = 100;
static double minEMdelta = 0.001;

static char *stopWordFile = 0;
static char *metaTag = 0;

static int useMaxentLM = 0;
static double maxentAlpha = 0.5;
static double maxentSigma2 = 0;
static int maxentConvertToArpa = 0;


static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_UINT, "order", &order, "max ngram order" },
    { OPT_FLOAT, "varprune", &varPrune, "pruning threshold for variable order ngrams" },
    { OPT_UINT, "debug", &debug, "debugging level for LM" },
    { OPT_TRUE, "recompute", &recompute, "recompute lower-order counts by summation" },
    { OPT_TRUE, "sort", &sortNgrams, "sort ngrams output" },
    { OPT_UINT, "write-order", &writeOrder, "output ngram counts order" },
    { OPT_STRING, "tag", &filetag, "file tag to use in messages" },
    { OPT_STRING, "text", &textFile, "text file to read" },
    { OPT_TRUE, "text-has-weights", &textFileHasWeights, "text file contains count weights" },
    { OPT_TRUE, "no-sos", &noSOS, "don't insert start-of-sentence tokens" },
    { OPT_TRUE, "no-eos", &noEOS, "don't insert end-of-sentence tokens" },
    { OPT_STRING, "read", &readFile, "counts file to read" },
    { OPT_STRING, "intersect", &intersectFile, "intersect counts with this file" },
    { OPT_TRUE, "read-with-mincounts", &readWithMincounts, "apply minimum counts when reading counts file" },
    { OPT_STRING, "read-google", &readGoogleDir, "Google counts directory to read" },

    { OPT_STRING, "write", &writeFile[0], "counts file to write" },
    { OPT_STRING, "write1", &writeFile[1], "1gram counts file to write" },
    { OPT_STRING, "write2", &writeFile[2], "2gram counts file to write" },
    { OPT_STRING, "write3", &writeFile[3], "3gram counts file to write" },
    { OPT_STRING, "write4", &writeFile[4], "4gram counts file to write" },
    { OPT_STRING, "write5", &writeFile[5], "5gram counts file to write" },
    { OPT_STRING, "write6", &writeFile[6], "6gram counts file to write" },
    { OPT_STRING, "write7", &writeFile[7], "7gram counts file to write" },
    { OPT_STRING, "write8", &writeFile[8], "8gram counts file to write" },
    { OPT_STRING, "write9", &writeFile[9], "9gram counts file to write" },

    { OPT_STRING, "write-binary", &writeBinaryFile, "binary counts file to write" },

    { OPT_FLOAT, "gtmin", &gtmin[0], "lower GT discounting cutoff" },
    { OPT_UINT, "gtmax", &gtmax[0], "upper GT discounting cutoff" },
    { OPT_FLOAT, "gt1min", &gtmin[1], "lower 1gram discounting cutoff" },
    { OPT_UINT, "gt1max", &gtmax[1], "upper 1gram discounting cutoff" },
    { OPT_FLOAT, "gt2min", &gtmin[2], "lower 2gram discounting cutoff" },
    { OPT_UINT, "gt2max", &gtmax[2], "upper 2gram discounting cutoff" },
    { OPT_FLOAT, "gt3min", &gtmin[3], "lower 3gram discounting cutoff" },
    { OPT_UINT, "gt3max", &gtmax[3], "upper 3gram discounting cutoff" },
    { OPT_FLOAT, "gt4min", &gtmin[4], "lower 4gram discounting cutoff" },
    { OPT_UINT, "gt4max", &gtmax[4], "upper 4gram discounting cutoff" },
    { OPT_FLOAT, "gt5min", &gtmin[5], "lower 5gram discounting cutoff" },
    { OPT_UINT, "gt5max", &gtmax[5], "upper 5gram discounting cutoff" },
    { OPT_FLOAT, "gt6min", &gtmin[6], "lower 6gram discounting cutoff" },
    { OPT_UINT, "gt6max", &gtmax[6], "upper 6gram discounting cutoff" },
    { OPT_FLOAT, "gt7min", &gtmin[7], "lower 7gram discounting cutoff" },
    { OPT_UINT, "gt7max", &gtmax[7], "upper 7gram discounting cutoff" },
    { OPT_FLOAT, "gt8min", &gtmin[8], "lower 8gram discounting cutoff" },
    { OPT_UINT, "gt8max", &gtmax[8], "upper 8gram discounting cutoff" },
    { OPT_FLOAT, "gt9min", &gtmin[9], "lower 9gram discounting cutoff" },
    { OPT_UINT, "gt9max", &gtmax[9], "upper 9gram discounting cutoff" },

    { OPT_STRING, "gt", &gtFile[0], "Good-Turing discount parameter file" },
    { OPT_STRING, "gt1", &gtFile[1], "Good-Turing 1gram discounts" },
    { OPT_STRING, "gt2", &gtFile[2], "Good-Turing 2gram discounts" },
    { OPT_STRING, "gt3", &gtFile[3], "Good-Turing 3gram discounts" },
    { OPT_STRING, "gt4", &gtFile[4], "Good-Turing 4gram discounts" },
    { OPT_STRING, "gt5", &gtFile[5], "Good-Turing 5gram discounts" },
    { OPT_STRING, "gt6", &gtFile[6], "Good-Turing 6gram discounts" },
    { OPT_STRING, "gt7", &gtFile[7], "Good-Turing 7gram discounts" },
    { OPT_STRING, "gt8", &gtFile[8], "Good-Turing 8gram discounts" },
    { OPT_STRING, "gt9", &gtFile[9], "Good-Turing 9gram discounts" },

    { OPT_FLOAT, "cdiscount", &cdiscount[0], "discounting constant" },
    { OPT_FLOAT, "cdiscount1", &cdiscount[1], "1gram discounting constant" },
    { OPT_FLOAT, "cdiscount2", &cdiscount[2], "2gram discounting constant" },
    { OPT_FLOAT, "cdiscount3", &cdiscount[3], "3gram discounting constant" },
    { OPT_FLOAT, "cdiscount4", &cdiscount[4], "4gram discounting constant" },
    { OPT_FLOAT, "cdiscount5", &cdiscount[5], "5gram discounting constant" },
    { OPT_FLOAT, "cdiscount6", &cdiscount[6], "6gram discounting constant" },
    { OPT_FLOAT, "cdiscount7", &cdiscount[7], "7gram discounting constant" },
    { OPT_FLOAT, "cdiscount8", &cdiscount[8], "8gram discounting constant" },
    { OPT_FLOAT, "cdiscount9", &cdiscount[9], "9gram discounting constant" },

    { OPT_TRUE, "ndiscount", &ndiscount[0], "use natural discounting" },
    { OPT_TRUE, "ndiscount1", &ndiscount[1], "1gram natural discounting" },
    { OPT_TRUE, "ndiscount2", &ndiscount[2], "2gram natural discounting" },
    { OPT_TRUE, "ndiscount3", &ndiscount[3], "3gram natural discounting" },
    { OPT_TRUE, "ndiscount4", &ndiscount[4], "4gram natural discounting" },
    { OPT_TRUE, "ndiscount5", &ndiscount[5], "5gram natural discounting" },
    { OPT_TRUE, "ndiscount6", &ndiscount[6], "6gram natural discounting" },
    { OPT_TRUE, "ndiscount7", &ndiscount[7], "7gram natural discounting" },
    { OPT_TRUE, "ndiscount8", &ndiscount[8], "8gram natural discounting" },
    { OPT_TRUE, "ndiscount9", &ndiscount[9], "9gram natural discounting" },

    { OPT_FLOAT, "addsmooth", &addsmooth[0], "additive smoothing constant" },
    { OPT_FLOAT, "addsmooth1", &addsmooth[1], "1gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth2", &addsmooth[2], "2gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth3", &addsmooth[3], "3gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth4", &addsmooth[4], "4gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth5", &addsmooth[5], "5gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth6", &addsmooth[6], "6gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth7", &addsmooth[7], "7gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth8", &addsmooth[8], "8gram additive smoothing constant" },
    { OPT_FLOAT, "addsmooth9", &addsmooth[9], "9gram additive smoothing constant" },

    { OPT_TRUE, "wbdiscount", &wbdiscount[0], "use Witten-Bell discounting" },
    { OPT_TRUE, "wbdiscount1", &wbdiscount[1], "1gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount2", &wbdiscount[2], "2gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount3", &wbdiscount[3], "3gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount4", &wbdiscount[4], "4gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount5", &wbdiscount[5], "5gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount6", &wbdiscount[6], "6gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount7", &wbdiscount[7], "7gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount8", &wbdiscount[8], "8gram Witten-Bell discounting"},
    { OPT_TRUE, "wbdiscount9", &wbdiscount[9], "9gram Witten-Bell discounting"},

    { OPT_TRUE, "kndiscount", &kndiscount[0], "use modified Kneser-Ney discounting" },
    { OPT_TRUE, "kndiscount1", &kndiscount[1], "1gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount2", &kndiscount[2], "2gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount3", &kndiscount[3], "3gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount4", &kndiscount[4], "4gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount5", &kndiscount[5], "5gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount6", &kndiscount[6], "6gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount7", &kndiscount[7], "7gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount8", &kndiscount[8], "8gram modified Kneser-Ney discounting"},
    { OPT_TRUE, "kndiscount9", &kndiscount[9], "9gram modified Kneser-Ney discounting"},

    { OPT_TRUE, "ukndiscount", &ukndiscount[0], "use original Kneser-Ney discounting" },
    { OPT_TRUE, "ukndiscount1", &ukndiscount[1], "1gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount2", &ukndiscount[2], "2gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount3", &ukndiscount[3], "3gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount4", &ukndiscount[4], "4gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount5", &ukndiscount[5], "5gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount6", &ukndiscount[6], "6gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount7", &ukndiscount[7], "7gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount8", &ukndiscount[8], "8gram original Kneser-Ney discounting"},
    { OPT_TRUE, "ukndiscount9", &ukndiscount[9], "9gram original Kneser-Ney discounting"},

    { OPT_STRING, "kn", &knFile[0], "Kneser-Ney discount parameter file" },
    { OPT_STRING, "kn1", &knFile[1], "Kneser-Ney 1gram discounts" },
    { OPT_STRING, "kn2", &knFile[2], "Kneser-Ney 2gram discounts" },
    { OPT_STRING, "kn3", &knFile[3], "Kneser-Ney 3gram discounts" },
    { OPT_STRING, "kn4", &knFile[4], "Kneser-Ney 4gram discounts" },
    { OPT_STRING, "kn5", &knFile[5], "Kneser-Ney 5gram discounts" },
    { OPT_STRING, "kn6", &knFile[6], "Kneser-Ney 6gram discounts" },
    { OPT_STRING, "kn7", &knFile[7], "Kneser-Ney 7gram discounts" },
    { OPT_STRING, "kn8", &knFile[8], "Kneser-Ney 8gram discounts" },
    { OPT_STRING, "kn9", &knFile[9], "Kneser-Ney 9gram discounts" },

    { OPT_TRUE, "kn-counts-modified", &knCountsModified, "input counts already modified for KN smoothing"},
    { OPT_TRUE, "kn-modify-counts-at-end", &knCountsModifyAtEnd, "modify counts after discount estimation rather than before"},

    { OPT_TRUE, "interpolate", &interpolate[0], "use interpolated estimates"},
    { OPT_TRUE, "interpolate1", &interpolate[1], "use interpolated 1gram estimates"},
    { OPT_TRUE, "interpolate2", &interpolate[2], "use interpolated 2gram estimates"},
    { OPT_TRUE, "interpolate3", &interpolate[3], "use interpolated 3gram estimates"},
    { OPT_TRUE, "interpolate4", &interpolate[4], "use interpolated 4gram estimates"},
    { OPT_TRUE, "interpolate5", &interpolate[5], "use interpolated 5gram estimates"},
    { OPT_TRUE, "interpolate6", &interpolate[6], "use interpolated 6gram estimates"},
    { OPT_TRUE, "interpolate7", &interpolate[7], "use interpolated 7gram estimates"},
    { OPT_TRUE, "interpolate8", &interpolate[8], "use interpolated 8gram estimates"},
    { OPT_TRUE, "interpolate9", &interpolate[9], "use interpolated 9gram estimates"},

    { OPT_STRING, "lm", &lmFile, "LM to estimate" },
    { OPT_TRUE, "write-binary-lm", &writeBinaryLM, "output LM in binary format" },
    { OPT_STRING, "init-lm", &initLMFile, "initial LM for EM estimation" },
    { OPT_TRUE, "unk", &keepunk, "keep <unk> in LM" },
    { OPT_STRING, "map-unk", &mapUnknown, "word to map unknown words to" },
    { OPT_STRING, "meta-tag", &metaTag, "meta tag used to input count-of-count information" },
    { OPT_TRUE, "float-counts", &useFloatCounts, "use fractional counts" },
    { OPT_TRUE, "tagged", &tagged, "build a tagged LM" },
    { OPT_TRUE, "count-lm", &useCountLM, "train a count-based LM" },
    { OPT_TRUE, "skip", &skipNgram, "build a skip N-gram LM" },
    { OPT_FLOAT, "skip-init", &skipInit, "default initial skip probability" },
    { OPT_UINT, "em-iters", &maxEMiters, "max number of EM iterations" },
    { OPT_FLOAT, "em-delta", &minEMdelta, "min log likelihood delta for EM" },
    { OPT_STRING, "stop-words", &stopWordFile, "stop-word vocabulary for stop-Ngram LM" },

    { OPT_TRUE, "maxent", &useMaxentLM, "Estimate maximum entropy model" },
    { OPT_FLOAT, "maxent-alpha", &maxentAlpha, "The L1 regularisation constant for max-ent estimation" },
    { OPT_FLOAT, "maxent-sigma2", &maxentSigma2, "The L2 regularisation constant for max-ent estimation (default: 6 for estimation, 0.5 for adaptation)" },
    { OPT_TRUE, "maxent-convert-to-arpa", &maxentConvertToArpa, "Save estimated max-ent model as a regular ARPA backoff model" },

    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },
    { OPT_TRUE, "trust-totals", &trustTotals, "trust lower-order counts for estimation" },
    { OPT_FLOAT, "prune", &prune, "prune redundant probs" },
    { OPT_UINT, "minprune", &minprune, "prune only ngrams at least this long" },
    { OPT_STRING, "vocab", &vocabFile, "vocab file" },
    { OPT_STRING, "vocab-aliases", &vocabAliasFile, "vocab alias file" },
    { OPT_STRING, "nonevents", &noneventFile, "non-event vocabulary" },
    { OPT_TRUE, "limit-vocab", &limitVocab, "limit count reading to specified vocabulary" },
    { OPT_STRING, "write-vocab", &writeVocab, "write vocab to file" },
    { OPT_STRING, "write-vocab-index", &writeVocabIndex, "write vocab index map to file" },
    { OPT_STRING, "write-text", &writeTextFile, "write input text to file (for validation)" },
    { OPT_TRUE, "memuse", &memuse, "show memory usage" },

    { OPT_DOC, 0, 0, "the default action is to write counts to stdout" },
};

static Boolean
copyFile(File &in, File &out)
{
    char *line;

    while ((line = in.getline())) {
	out.fputs(line);
    }
    return !(in.error());
}

int
main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    Boolean written = false;

    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    if (useFloatCounts + tagged + skipNgram +
	(stopWordFile != 0) + (varPrune != 0.0) > 1)
    {
	cerr << "fractional counts, variable, tagged, stop-word Ngram and skip N-gram models are mutually exclusive\n";
	exit(2);
    }

    /*
     * Detect inconsistent discounting options
     */
    if (ndiscount[0] +
	wbdiscount[0] +
	(cdiscount[0] != -1.0) +
	(addsmooth[0] != -1.0) +
	ukndiscount[0] + 
	(knFile[0] != 0 || kndiscount[0]) +
	(gtFile[0] != 0) > 1)
    {
	cerr << "conflicting default discounting options\n";
	exit(2);
    }

    Vocab *vocab = tagged ? new TaggedVocab : new Vocab;
    assert(vocab);

    vocab->unkIsWord() = keepunk ? true : false;
    vocab->toLower() = toLower ? true : false;

    /*
     * Change unknown word string if requested
     */
    if (mapUnknown) {
	vocab->remove(vocab->unkIndex());
	vocab->unkIndex() = vocab->addWord(mapUnknown);
    }

    /*
     * Meta tag is used to input count-of-count information
     */
    if (metaTag) {
	vocab->metaTag() = metaTag;
    }

    SubVocab *stopWords = 0;

    if (stopWordFile != 0) {
	stopWords = new SubVocab(*vocab);
	assert(stopWords);
    }

    /*
     * The skip-ngram model requires count order one higher than
     * the normal model.
     */
    NgramStats *intStats =
	(stopWords != 0) ? new StopNgramStats(*vocab, *stopWords, order) :
	   tagged ? new TaggedNgramStats(*(TaggedVocab *)vocab, order) :
	      useFloatCounts ? 0 :
	         new NgramStats(*vocab, skipNgram ? order + 1 : order);
    NgramCounts<FloatCount> *floatStats =
	      !useFloatCounts ? 0 :
		 new NgramCounts<FloatCount>(*vocab, order);

#define USE_STATS(what) (useFloatCounts ? floatStats->what : intStats->what)

    if (useFloatCounts) {
	assert(floatStats != 0);
    } else {
	assert(intStats != 0);
    }

    USE_STATS(debugme(debug));

    if (vocabFile) {
	File file(vocabFile, "r");
	USE_STATS(vocab.read(file));
	USE_STATS(openVocab) = false;
    }

    if (noSOS) {
        USE_STATS(addSentStart) = false;
    }
    if (noEOS) {
        USE_STATS(addSentEnd) = false;
    }

    if (vocabAliasFile) {
	File file(vocabAliasFile, "r");
	USE_STATS(vocab.readAliases(file));
    }

    if (stopWordFile) {
	File file(stopWordFile, "r");
	stopWords->read(file);
    }

    if (noneventFile) {
	/*
	 * create temporary sub-vocabulary for non-event words
	 */
	SubVocab nonEvents(USE_STATS(vocab));

	File file(noneventFile, "r");
	nonEvents.read(file);

	USE_STATS(vocab).addNonEvents(nonEvents);
    }

    if (intersectFile) {
	File file(intersectFile, "r");

        USE_STATS(read(file, order, limitVocab));
	USE_STATS(setCounts(0));
	USE_STATS(intersect) = true;
    }

    if (readFile) {
	File file(readFile, "r");

	unsigned countOrder = USE_STATS(getorder());

	if (readWithMincounts) {
	    makeArray(Count, minCounts, countOrder);

	    /* construct min-counts array from -gtNmin options */
	    unsigned i;
	    for (i = 0; i < countOrder && i < maxorder; i ++) {
		minCounts[i] = (Count)gtmin[i + 1];
	    }
	    for ( ; i < countOrder; i ++) {
		minCounts[i] = (Count)gtmin[0];
	    }
	    USE_STATS(readMinCounts(file, countOrder, minCounts));
	} else {
	    USE_STATS(read(file, countOrder, limitVocab));
	}
    }

    if (readGoogleDir) {
    	if (!USE_STATS(readGoogle(readGoogleDir, order, limitVocab))) {
	    cerr << "error reading Google counts from "
	         << readGoogleDir << endl;
	    exit(1);
	}
    }

    if (textFile) {
	File file(textFile, "r");
	if (writeTextFile) {
	    File outFile(writeTextFile, "w");
	    copyFile(file, outFile);
	} else {
	    USE_STATS(countFile(file, textFileHasWeights));
	}
    }

    if (memuse) {
	MemStats memuse;
	USE_STATS(memStats(memuse));

	if (debug == 0)  {
	    memuse.clearAllocStats();
	}
	memuse.print();

    	if (debug > 0) {
	    BM_printstats();
	}
    }

    if (recompute) {
	if (useFloatCounts)
	    floatStats->sumCounts(order);
	else
	    intStats->sumCounts(order);
    }

    /*
     * While ngrams themselves can have order 0 (they will always be empty)
     * we need order >= 1 for LM estimation.
     */
    if (order == 0) {
	cerr << "LM order must be positive -- set to 1\n";
	order = 1;
    }

    /*
     * This stores the discounting parameters for the various orders
     * Note this is only needed when estimating an LM
     */
    Discount **discounts = new Discount *[order];
    assert(discounts != 0);

    unsigned i;
    for (i = 0; i < order; i ++) {
	discounts[i] = 0;
    }

    /*
     * Estimate discounting parameters 
     * Note this is only required if 
     * - the user wants them written to a file
     * - we also want to estimate a LM later
     */
    for (i = 1; !useCountLM && !useMaxentLM && i <= order; i++) {
	/*
	 * Detect inconsistent options for this order
	 */
	if (i <= maxorder &&
	    ndiscount[i] + wbdiscount[i] +
	    (cdiscount[i] != -1.0) + (addsmooth[i] != -1.0) +
	    ukndiscount[i] + (knFile[i] != 0 || kndiscount[i]) +
	    (gtFile[i] != 0) > 1)
	{
	    cerr << "conflicting discounting options for order " << i << endl;
	    exit(2);
	}

	/*
	 * Inherit default discounting method where needed
	 */
	if (i <= maxorder &&
	    !ndiscount[i] && !wbdiscount[i] &&
	    cdiscount[i] == -1.0 && addsmooth[i] == -1.0 &&
	    !ukndiscount[i] && knFile[i] == 0 && !kndiscount[i] &&
	    gtFile[i] == 0)
	{
	    if (ndiscount[0]) ndiscount[i] = ndiscount[0];
	    else if (wbdiscount[0]) wbdiscount[i] = wbdiscount[0]; 
	    else if (cdiscount[0] != -1.0) cdiscount[i] = cdiscount[0];
	    else if (addsmooth[0] != -1.0) addsmooth[i] = addsmooth[0];
	    else if (ukndiscount[0]) ukndiscount[i] = ukndiscount[0];
	    else if (kndiscount[0]) kndiscount[i] = kndiscount[0];

	    if (knFile[0] != 0) knFile[i] = knFile[0];
	    else if (gtFile[0] != 0) gtFile[i] = gtFile[0];
	}

	/*
	 * Choose discounting method to use
	 *
	 * Also, check for any discounting parameter files.
	 * These have a dual interpretation.
	 * If we're not estimating a new LM, simple WRITE the parameters
	 * out.  Otherwise try to READ them from these files.
	 *
	 * Note: Test for ukndiscount[] before knFile[] so that combined use 
	 * of -ukndiscountN and -knfileN will do the right thing.
	 */
	unsigned useorder = (i > maxorder) ? 0 : i;
	Discount *discount = 0;

	if (ndiscount[useorder]) {
	    if (debug) cerr << "using NaturalDiscount for " << i << "-grams";
	    discount = new NaturalDiscount(gtmin[useorder]);
	    assert(discount);
	} else if (wbdiscount[useorder]) {
	    if (debug) cerr << "using WittenBell for " << i << "-grams";
	    discount = new WittenBell(gtmin[useorder]);
	    assert(discount);
	} else if (cdiscount[useorder] != -1.0) {
	    if (debug) cerr << "using ConstDiscount for " << i << "-grams";
	    discount = new ConstDiscount(cdiscount[useorder], gtmin[useorder]);
	    assert(discount);
	} else if (addsmooth[useorder] != -1.0) {
	    if (debug) cerr << "using AddSmooth for " << i << "-grams";
	    discount = new AddSmooth(addsmooth[useorder], gtmin[useorder]);
	    assert(discount);
	} else if (ukndiscount[useorder]) {
	    if (debug) cerr << "using KneserNey for " << i << "-grams";
	    discount = new KneserNey((unsigned)gtmin[useorder], knCountsModified, knCountsModifyAtEnd);
	    assert(discount);
	} else if (knFile[useorder] || kndiscount[useorder]) {
	    if (debug) cerr << "using ModKneserNey for " << i << "-grams";
	    discount = new ModKneserNey((unsigned)gtmin[useorder], knCountsModified, knCountsModifyAtEnd);
	    assert(discount);
	} else if (gtFile[useorder] || (i <= order && lmFile)) {
	    if (debug) cerr << "using GoodTuring for " << i << "-grams";
	    discount = new GoodTuring((unsigned)gtmin[useorder], gtmax[useorder]);
	    assert(discount);
	}
	if (debug && discount != 0) cerr << endl;

	/*
	 * Now read in, or estimate the discounting parameters.
	 * Also write them out if no language model is being created.
	 */
	if (discount) {
	    discount->debugme(debug);

	    if (interpolate[0] || interpolate[useorder]) {
		discount->interpolate = true;
	    }

	    if (knFile[useorder] && lmFile) {
		File file(knFile[useorder], "r");

		if (!discount->read(file)) {
		    cerr << "error in reading discount parameter file "
			 << knFile[useorder] << endl;
		    exit(1);
		}
	    } else if (gtFile[useorder] && lmFile) {
		File file(gtFile[useorder], "r");

		if (!discount->read(file)) {
		    cerr << "error in reading discount parameter file "
			 << gtFile[useorder] << endl;
		    exit(1);
		}
	    } else {
		/*
		 * Estimate discount params, and write them only if 
		 * a file was specified, but no language model is
		 * being estimated.
		 */
		if (!(useFloatCounts ? discount->estimate(*floatStats, i) :
				       discount->estimate(*intStats, i)))
		{
		    cerr << "error in discount estimator for order "
			 << i << endl;
		    exit(1);
		}
		if (knFile[useorder]) {
		    File file(knFile[useorder], "w");
		    discount->write(file);
		    written = true;
		} else if (gtFile[useorder]) {
		    File file(gtFile[useorder], "w");
		    discount->write(file);
		    written = true;
		}
	    }

	    discounts[i-1] = discount;
	}
    }

    /*
     * Estimate a new model from the existing counts,
     */
    if (useCountLM && lmFile) {
    	/*
	 * count-LM estimation is different 
	 * - read existing model from file
	 * - set estimation parameters
	 * - estimate 
	 * - write updated model
	 */
	NgramCountLM *lm;

	lm = new NgramCountLM(*vocab, order);
	assert(lm != 0);

	lm->maxEMiters = maxEMiters;
	lm->minEMdelta = minEMdelta;

	/*
	 * Set debug level on LM object
	 */
	lm->debugme(debug);

	/*
	 * Read initial LM parameters
	 */
	if (initLMFile) {
	    File file(initLMFile, "r");

	    if (!lm->read(file, limitVocab)) {
		cerr << "format error in init-lm file\n";
		exit(1);
	    }
	} else {
	    cerr << "count-lm estimation needs initial model\n";
	    exit(1);
	}
        
	if (useFloatCounts) {
	    cerr << "cannot use -float-counts with count-lm\n";
	    exit(1);
	}

        if (!lm->estimate(*intStats)) {
	    cerr << "LM estimation failed\n";
	    exit(1);
	} else {
	    /*
	     * Write updated parameters, but avoid writing out the counts,
	     * which are unchanged.
	     */
	    lm->writeCounts = false;
	    if (writeBinaryLM) {
		File file(lmFile, "wb");
		lm->writeBinary(file);
	    } else {
		File file(lmFile, "w");
		lm->write(file);
	    }
	}

	written = true;

	// XXX: don't free the lm since this itself may take a long time
	// and we're going to exit anyways.
#ifdef DEBUG
	delete lm;
#endif
    } else if (useMaxentLM && lmFile) {
    	/*
    	 * MaxEnt model estimation
	 * -init-lm serves as model prior
    	 */
    	MEModel *lm = new MEModel(*vocab, order);
    	lm->debugme(debug);
    	if (initLMFile) {
	    File file(initLMFile, "r");

    	    if (!lm->read(file)) {
		cerr << "format error in maxent prior (-init-lm) file\n";
		exit(1);
    	    }

    	    // Use default value 0.5 for L2 smoothing
    	    (useFloatCounts ? lm->adapt(*floatStats, maxentAlpha, maxentSigma2 == 0.0 ? 0.5 : maxentSigma2) :
    	    		lm->adapt(*intStats, maxentAlpha, maxentSigma2 == 0.0 ? 0.5 : maxentSigma2));
    	} else {
	    // Use default value 6.0 for L2 smoothing
	    if (!(useFloatCounts ? lm->estimate(*floatStats, maxentAlpha, maxentSigma2 == 0.0 ? 6.0 : maxentSigma2) :
				   lm->estimate(*intStats, maxentAlpha, maxentSigma2 == 0.0 ? 6.0 : maxentSigma2)))
	    {
		cerr << "Maxent LM estimation failed\n";
		exit(1);
	    }
    	}
	if (maxentConvertToArpa) {
	    Ngram *ngram = lm->getNgramLM();
	    ngram->debugme(debug);

	    /*
	     * Remove redundant probs (perplexity increase below threshold)
	     */
	    if (prune != 0.0) {
		ngram->pruneProbs(prune, minprune);
	    }

	    if (writeBinaryLM) {
		File file(lmFile, "wb");
		ngram->writeBinary(file);
	    } else {
		File file(lmFile, "w");
		ngram->write(file);
	    }
#ifdef DEBUG
	    delete ngram;
#endif
	} else {
	    File file(lmFile, "w");
	    lm->write(file);
	}
	written = true;
#ifdef DEBUG
	delete lm;
#endif
    } else if (lmFile) {
        /*
	 * Backoff ngram LM estimation:
	 * either using a default discounting scheme, or the GT parameters
	 * read in from files
	 */
	Ngram *lm;
	
	if (varPrune != 0.0) {
	    lm = new VarNgram(*vocab, order, varPrune);
	    assert(lm != 0);
	} else if (skipNgram) {
	    SkipNgram *skipLM =  new SkipNgram(*vocab, order);
	    assert(skipLM != 0);

	    skipLM->maxEMiters = maxEMiters;
	    skipLM->minEMdelta = minEMdelta;
	    skipLM->initialSkipProb = skipInit;

	    lm = skipLM;
	} else {
	    lm = (stopWords != 0) ? new StopNgram(*vocab, *stopWords, order) :
		       tagged ? new TaggedNgram(*(TaggedVocab *)vocab, order) :
			  new Ngram(*vocab, order);
	    assert(lm != 0);
	}

	/*
	 * Set debug level on LM object
	 */
	lm->debugme(debug);

	/*
	 * Read initial LM parameters in case we're doing EM
	 */
	if (initLMFile) {
	    File file(initLMFile, "r");

	    if (!lm->read(file, limitVocab)) {
		cerr << "format error in init-lm file\n";
		exit(1);
	    }
	}
        
	if (trustTotals) {
	    lm->trustTotals() = true;
	}
	if (!(useFloatCounts ? lm->estimate(*floatStats, discounts) :
			       lm->estimate(*intStats, discounts)))
	{
	    cerr << "LM estimation failed\n";
	    exit(1);
	} else {
	    /*
	     * Remove redundant probs (perplexity increase below threshold)
	     */
	    if (prune != 0.0) {
		lm->pruneProbs(prune, minprune);
	    }

	    if (writeBinaryLM) {
		File file(lmFile, "wb");
		lm->writeBinary(file);
	    } else {
		File file(lmFile, "w");
		lm->write(file);
	    }
	}
	written = true;

	// XXX: don't free the lm since this itself may take a long time
	// and we're going to exit anyways.
#ifdef DEBUG
	delete lm;
#endif
    }

    if (writeVocab) {
	File file(writeVocab, "w");
	vocab->write(file);
	written = true;
    }

    if (writeVocabIndex) {
	File file(writeVocabIndex, "w");
	vocab->writeIndexMap(file);
	written = true;
    }

    /*
     * Write counts of a specific order
     */
    for (i = 1; i <= maxorder; i++) {
	if (writeFile[i]) {
	    File file(writeFile[i], "w");
	    USE_STATS(write(file, i, sortNgrams));
	    written = true;
	}
    }

    /*
     * Write binary counts
     */
    if (writeBinaryFile) {
	File file(writeBinaryFile, "wb");
	if (!USE_STATS(writeBinary(file, writeOrder))) {
	    cerr << "error writing " << writeBinaryFile << endl;
	}
	written = true;
    }

    /*
     * If nothing has been written out so far, make it the default action
     * to dump the counts 
     *
     * Note: This will write the modified rather than the original counts
     * if KN discounting was used.
     */
    if (writeFile[0] || !written) {
	File file(writeFile[0] ? writeFile[0] : "-", "w");
	USE_STATS(write(file, writeOrder, sortNgrams));
    }

#ifdef DEBUG
    /*
     * Free all objects
     */
    for (i = 0; i < order; i ++) {
	delete discounts[i];
	discounts[i] = 0;
    }
    delete [] discounts;

    delete intStats;
    delete floatStats;

    if (stopWords != 0) {
	delete stopWords;
    }

    delete vocab;
    return(0);
#endif /* DEBUG */

    exit(0);
}

