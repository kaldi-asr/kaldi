/*
 * anti-ngram --
 *	Compute Anti-Ngram-LM from N-best lists
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1999-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: anti-ngram.cc,v 1.21 2014-08-29 21:35:48 frandsen Exp $";
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
#include "RefList.h"
#include "NBestSet.h"
#include "NgramStats.h"
#include "Ngram.h"
#include "ClassNgram.h"
#include "MultiwordVocab.h"	// for MultiwordSeparator
#include "Array.cc"

static int version = 0;
static unsigned order = 3;
static char *vocabFile = 0;
static char *lmFile = 0;
static char *classesFile = 0;
static double rescoreLMW = 8.0;
static double rescoreWTW = 0.0;
static double posteriorScale = 0.0;

static int toLower = 0;
static int multiwords = 0;
static const char *multiChar = MultiwordSeparator;

static char *refFile = 0;
static char *nbestFiles = 0;
static unsigned maxNbest = 0;

static char *readCounts = 0;
static char *writeCounts = 0;
static double minCount = 0.0;
static int sortNgrams = 0;
static int allNgrams = 0;
static unsigned debug = 0;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_STRING, "refs", &refFile, "reference transcripts" },
    { OPT_STRING, "nbest-files", &nbestFiles, "list of training Nbest files" },
    { OPT_UINT, "max-nbest", &maxNbest, "maximum number of hyps to consider" },

    { OPT_UINT, "order", &order, "max ngram order" },
    { OPT_STRING, "lm", &lmFile, "N-gram model in ARPA LM format" },
    { OPT_STRING, "classes", &classesFile, "class definitions" },
    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },
    { OPT_TRUE, "multiwords", &multiwords, "split multiwords in N-best hyps" },
    { OPT_STRING, "multi-char", &multiChar, "multiword component delimiter" },
    { OPT_FLOAT, "rescore-lmw", &rescoreLMW, "rescoring LM weight" },
    { OPT_FLOAT, "rescore-wtw", &rescoreWTW, "rescoring word transition weight" },
    { OPT_FLOAT, "posterior-scale", &posteriorScale, "divisor for log posterior estimates" },
    { OPT_TRUE, "all-ngrams", &allNgrams, "include reference ngrams" },
    { OPT_STRING, "read-counts", &readCounts, "read N-gram stats from file" },
    { OPT_STRING, "write-counts", &writeCounts, "write N-gram stats to file" },
    { OPT_FLOAT, "min-count", &minCount, "prune counts below this value" },
    { OPT_TRUE, "sort", &sortNgrams, "sort ngrams output" },
    { OPT_UINT, "debug", &debug, "debugging level" },
};

typedef double DiscNgramCount;		// fractional count type

/*
 * Add ngram counts
 */
void
addStats(NgramCounts<DiscNgramCount> &stats,
         NgramCounts<DiscNgramCount> &add, NgramStats &exclude)
{
    makeArray(VocabIndex, ngram, order + 1);

    for (unsigned i = 1; i <= order; i++) {
	DiscNgramCount *count;
	NgramCountsIter<DiscNgramCount> countIter(add, ngram, i);

	/*
	 * This enumerates all ngrams
	 */
	while ((count = countIter.next())) {
	    if (!exclude.findCount(ngram)) {
		*stats.insertCount(ngram) += *count;
	    }
	}
    }
}

/*
 * Add <s> and </s> tokens to a word string
 *	(returns pointer to static buffer)
 */
VocabIndex *
makeSentence(VocabIndex *words, Vocab &vocab)
{
    static VocabIndex buffer[maxWordsPerLine + 3];
    unsigned j = 0;

    if (words[0] != vocab.ssIndex()) {
	buffer[j++] = vocab.ssIndex();
    }

    unsigned i;
    for (i = 0; words[i] != Vocab_None; i ++) {
	assert(i < maxWordsPerLine);

	buffer[j++] = words[i];
    }
    if (buffer[j-1] != vocab.seIndex()) {
	buffer[j++] = vocab.seIndex();
    }
    if (j < maxWordsPerLine + 3) {
	buffer[j] = Vocab_None;
    } // else unexpected error

    return buffer;
}

/*
 * Process an N-best list for training:
 *	- compute LM scores
 *	- compute posteriors
 *	- update ngram counts
 */
void
countNBestList(NBestList &nbest, VocabIndex *ref, LM *lm, 
			NgramCounts<DiscNgramCount> &stats)
{
    if (nbest.numHyps() == 0) {
	return;
    }

    /*
     * Recompute LM scores, using unit weights
     */
    if (lm) {
    	nbest.rescoreHyps(*lm, 1.0, 0.0);
    }

    /*
     * Compute posterior probs, using chosen weights
     */
    nbest.computePosteriors(rescoreLMW, rescoreWTW, posteriorScale);

    /*
     * Count ngrams in reference
     */
    NgramStats refNgrams(stats.vocab, order);
    if (!allNgrams) {
	refNgrams.countSentence(makeSentence(ref, stats.vocab), 1);
    }

    /* 
     * count ngrams in N-best list, weighted by posteriors
     */
    NgramCounts<DiscNgramCount> hypNgrams(stats.vocab, order);
  
    unsigned h;
    for (h = 0; h < nbest.numHyps(); h ++) {
	hypNgrams.countSentence(makeSentence(nbest.getHyp(h).words,
						stats.vocab),
						    nbest.getHyp(h).posterior);
    }

    /*
     * Add hyp ngram counts to overall stats, excluding the ref ngrams.
     */
    addStats(stats, hypNgrams, refNgrams);
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

    if (!nbestFiles) {
	cerr << "cannot proceed without nbest files\n";
	exit(2);
    }

    /*
     * Posterior scaling:  if not specified (= 0.0) use LMW for
     * backward compatibility.
     */
    if (posteriorScale == 0.0) {
	posteriorScale = rescoreLMW;
    }

    Vocab vocab;

    vocab.toLower() = toLower ? true : false;

    RefList refs(vocab);

    NBestSet trainSet(vocab, refs, maxNbest, true, multiwords ? multiChar : 0);
    trainSet.debugme(debug);
    trainSet.warn = false;

    NgramCounts<DiscNgramCount> trainStats(vocab, order);
    trainStats.debugme(debug);

    SubVocab *classVocab = 0;
    if (classesFile != 0) {
	classVocab = new SubVocab(vocab);
	assert(classVocab);
    }

    Ngram *ngram = 0;

    if (lmFile) {
	cerr << "reading LM...\n";
	File file(lmFile, "r");

	/*
	 * create class-ngram if -classes were specified, otherwise a regular
	 * ngram
	 */
	ngram = (classVocab != 0) ?
	  new ClassNgram(vocab, *classVocab, order) :
	  new Ngram(vocab, order);
	assert(ngram != 0);

	ngram->debugme(debug);
	ngram->read(file);

	/*
	 * read class vocabulary if specified
	 */
	if (classVocab != 0) {
	    File file(classesFile, "r");
	    ((ClassNgram *)ngram)->readClasses(file);
	}
    }

    /*
     * Read reference file after LM, so we have the vocabulary
     * loaded and can replace unknown words with <unk>
     */
    if (refFile) {
	cerr << "reading references...\n";
	File file(refFile, "r");
	refs.read(file);
    }

    if (readCounts) {
	cerr << "reading prior counts...\n";
	File file(readCounts, "r");
	trainStats.read(file);
    }

    {
	cerr << "reading nbest lists...\n";
	File file(nbestFiles, "r");
	trainSet.read(file);
    }

    /*
     * Accumulate counts over nbest set
     */
    NBestSetIter iter(trainSet);

    RefString id;
    NBestList *nbest;
    while ((nbest = iter.next(id))) {
	VocabIndex *ref = refs.findRef(id);

	if (!ref && !allNgrams) {
	    cerr << "no reference found for " << id << endl;
	} else if (ref) {
	    // ref can't be NULL since will get dereferenced
	    countNBestList(*nbest, ref, ngram, trainStats);
	} else {
	    cerr << "ref NULL for " << id << endl;
	}
    }

    /*
     * prune counts if specified
     */
    if (minCount > 0.0) {
	trainStats.pruneCounts(minCount);
    }

    if (writeCounts) {
	File file(writeCounts, "w");
	trainStats.write(file, 0, sortNgrams);
    } else {
	File file(stdout);
	trainStats.write(file, 0, sortNgrams);
    }

    exit(0);
}

