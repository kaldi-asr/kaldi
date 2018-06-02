/*
 * nbest-pron-score --
 *	Score pronunciations and pauses in N-best hypotheses
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2002-2010 SRI International, 2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: nbest-pron-score.cc,v 1.17 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <locale.h>
#include <assert.h>

#include "option.h"
#include "version.h"
#include "File.h"

#include "Array.cc"
#include "Prob.h"
#include "MultiwordVocab.h"
#include "NBest.h"
#include "Ngram.h"
#include "VocabMultiMap.h"
#include "RefList.h"
#include "MStringTokUtil.h"

#define DEBUG_SCORES	2

static int version = 0;
static unsigned debug = 0;
static int toLower = 0;
static int multiwords = 0;
static const char *multiChar = MultiwordSeparator;
static char *rescoreFile = 0;
static char *nbestFiles = 0;
static char *pauseLMFile = 0;
static char *dictFile = 0;
static char *pronScoreDir = 0;
static char *pauseScoreDir = 0;
static double pauseScoreWeight = 0.0;
static unsigned maxNbest = 0;
static int intlogs = 0;

static char *noPauseTag = (char *)"<nopause>";
static char *shortPauseTag = (char *)"<shortpause>";
static char *longPauseTag = (char *)"<longpause>";
static double minPauseDur = 0.06;
static double longPauseDur = 0.6;
static VocabIndex noPauseIndex;
static VocabIndex shortPauseIndex;
static VocabIndex longPauseIndex;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_UINT, "debug", &debug, "debugging level" },
    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },
    { OPT_TRUE, "multiwords", &multiwords, "split multiwords in N-best hyps" },
    { OPT_STRING, "multi-char", &multiChar, "multiword component delimiter" },
    { OPT_STRING, "rescore", &rescoreFile, "hyp stream input file to rescore" },
    { OPT_STRING, "nbest", &rescoreFile, "same as -rescore" },
    { OPT_STRING, "nbest-files", &nbestFiles, "list of n-best filenames" },
    { OPT_UINT, "max-nbest", &maxNbest, "maximum number of hyps to consider" },
    { OPT_STRING, "dictionary", &dictFile, "pronunciation dictionary" },
    { OPT_STRING, "pause-lm", &pauseLMFile, "pause language model" },
    { OPT_STRING, "pron-score-dir", &pronScoreDir, "pronunciation score directory" },
    { OPT_STRING, "pause-score-dir", &pauseScoreDir, "pause score directory" },
    { OPT_FLOAT, "pause-score-weight", &pauseScoreWeight, "pause score weight for adding with pron scores" },
    { OPT_TRUE, "intlogs", &intlogs, "dictionary uses intlog probabilities" },

    { OPT_STRING, "no-pause", &noPauseTag, "no pause tag" },
    { OPT_STRING, "short-pause", &shortPauseTag, "short pause tag" },
    { OPT_STRING, "long-pause", &longPauseTag, "long pause tag" },
    { OPT_FLOAT, "min-pause-dur", &minPauseDur, "minumum pause duration" },
    { OPT_FLOAT, "long-pause-dur", &longPauseDur, "long pause duration" },
};


VocabIndex
getPauseTag(NBestTimestamp pauseLength)
{
    if (pauseLength < minPauseDur) {
	return noPauseIndex;
    } else if (pauseLength >= longPauseDur) {
	return longPauseIndex;
    } else {
	return shortPauseIndex;
    }
}

void
writeScores(LogP *scores, unsigned numScores, const char *filename)
{

   File file(filename, "w");

   for (unsigned i = 0; i < numScores; i ++) {
	file.fprintf("%.*g\n", LogP_Precision, scores[i]);
   }
}

/*
 * Rescore one N-best list with pronunciation and pause models
 */
void
processNbest(const char *nbestFile, MultiwordVocab &vocab,
			VocabMultiMap &dictionary, Ngram &pauseLM,
			const char *pronScoreFile, const char *pauseScoreFile,
			double pauseScoreWeight)
{
    Vocab &phoneVocab = dictionary.vocab2;

    NBestList nbest(vocab, maxNbest, false, true);

    {
	File file(nbestFile, "r");

	if (!nbest.read(file)) {
	    cerr << "error reading nbest file\n";
	    return;
	}
    }

    unsigned numHyps = nbest.numHyps();

    if (numHyps == 0) {
	cerr << "warning: N-best list " << nbestFile << " is empty\n";
	return;
    }

    makeArray(LogP, pronScores, numHyps);
    makeArray(LogP, pauseScores, numHyps);

    Boolean warning = false;
    
    for (unsigned h = 0; h < numHyps; h ++) {
	NBestHyp &hyp = nbest.getHyp(h);

	if (hyp.wordInfo == 0) {
	    if (!warning) {
		cerr << "warning: N-best hyp " << h << " in "
		     << nbestFile << " does not contain backtrace info\n";
		warning = true;
	    }
	    continue;
	}

	/*
	 * compute pronunciation score:
	 *	sum of pronunciation log probabilites of all words in hyp
	 */
	if (pronScoreFile) {
	    LogP pronScore = LogP_One;

	    for (unsigned i = 0; hyp.words[i] != Vocab_None; i ++) {
		/*
		 * If pronunciation info is missing there is nothing we 
		 * can score. Assume pronunciation prob = 1.
		 */
		if (hyp.wordInfo[i].phones == 0) {
		    if (debug >= DEBUG_SCORES) {
			cerr << "WORD " << vocab.getWord(hyp.words[i])
			     << " PRON missing\n";
		    }

		    continue;
		}

		/*
		 * copy phone string to buffer for parsing
		 */
		makeArray(char, phoneString,
			  strlen(hyp.wordInfo[i].phones) + 1);
		strcpy(phoneString, hyp.wordInfo[i].phones);

		/*
		 * convert phone string to index string
		 */
		Array<VocabIndex> phones;
		unsigned numPhones = 0;
                char *strtok_ptr = NULL;

		for (char *s = MStringTokUtil::strtok_r(phoneString, phoneSeparator, &strtok_ptr);
		     s != 0;
		     s = MStringTokUtil::strtok_r(NULL, phoneSeparator, &strtok_ptr), numPhones ++)
		{
		    phones[numPhones] = phoneVocab.addWord(s);
		}
		phones[numPhones] = Vocab_None;

		/*
		 * find pronunciations prob
		 */
		Prob p = dictionary.get(hyp.words[i], phones.data());

		if (debug >= DEBUG_SCORES) {
		    cerr << "WORD " << vocab.getWord(hyp.words[i])
			 << " PRON " << (phoneVocab.use(), phones.data())
			 << " PROB " << p << endl;
		}

		if (p != 0.0) {
		    if (intlogs) {
			pronScore += IntlogToLogP(p);
		    } else {
			pronScore += ProbToLogP(p);
		    }
		}

	    }

	    pronScores[h] = pronScore;
	} else {
	    pronScores[h] = LogP_One;
	}

	/*
	 * compute pause score:
	 *	sum of pause LM log probabilites of all pauses in hyp
	 */
	if (pauseScoreFile || pauseScoreWeight != 0) {
	    LogP pauseScore = LogP_One;

	    VocabIndex lastWord = Vocab_None; 
	    NBestTimestamp pauseLength = 0;

	    for (unsigned i = 0; hyp.words[i] != Vocab_None; i ++) {
		if (hyp.words[i] == vocab.pauseIndex()) {
		    pauseLength += hyp.wordInfo[i].duration;
		} else {
		    VocabIndex context[3];
		    context[0] = Vocab_None;

		    VocabIndex firstPart, lastPart;

		    if (!multiwords ||
			pauseLM.findProb(hyp.words[i], context) != 0)
		    {
			firstPart = lastPart = hyp.words[i];
		    } else {
			context[0] = hyp.words[i];
			context[1] = Vocab_None;

			VocabIndex expanded[maxWordsPerLine + 1];
			unsigned n = vocab.expandMultiwords(context, expanded,
							    maxWordsPerLine);
			firstPart = expanded[0];
			lastPart = expanded[n - 1];
		    }

		    if (lastWord != Vocab_None) {
			context[0] = lastWord;
			context[1] = firstPart;
			context[2] = Vocab_None;

			VocabIndex pauseTag = getPauseTag(pauseLength);

			LogP pauseProb = pauseLM.wordProb(pauseTag, context);

			if (debug >= DEBUG_SCORES) {
			    cerr << "PAUSE " << vocab.getWord(pauseTag)
				 << " DUR " << pauseLength
				 << " CONTEXT " << (vocab.use(), context)
				 << " PROB " << pauseProb << endl;
			}

			pauseScore += pauseProb;
		    }

		    pauseLength = 0.0;
		    lastWord = lastPart;
		}
	    }

	    if (pauseScoreFile) {
		pauseScores[h] = pauseScore;
	    }
	    if (pauseScoreWeight != 0.0) {
		pronScores[h] += pauseScoreWeight * pauseScore;
	    }
	}
    }

    if (pronScoreFile) {
	writeScores(pronScores, numHyps, pronScoreFile);
    }

    if (pauseScoreFile) {
	writeScores(pauseScores, numHyps, pauseScoreFile);
    }
}

int
main (int argc, char *argv[])
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    MultiwordVocab vocab(multiChar);
    vocab.toLower() = toLower ? true : false;

    noPauseIndex = vocab.addWord(noPauseTag);
    shortPauseIndex = vocab.addWord(shortPauseTag);
    longPauseIndex = vocab.addWord(longPauseTag);

    Vocab dictVocab;
    VocabMultiMap dictionary(vocab, dictVocab, intlogs);

    /* 
     * Read optional dictionary to help in word alignment
     */
    if (dictFile) {
	File file(dictFile, "r");

	if (!dictionary.read(file)) {
	    cerr << "format error in dictionary file\n";
	    exit(1);
	}
    }

    Ngram pauseLM(vocab, 3);
    pauseLM.debugme(debug);

    if (pauseLMFile) {
	File file(pauseLMFile, "r");

	if (!pauseLM.read(file)) {
	    cerr << "format error in pause LM\n";
	    exit(1);
	}
    }

    /*
     * Process single nbest file
     */
    if (rescoreFile) {
	processNbest(rescoreFile, vocab, dictionary, pauseLM,
				dictFile ? "-" : 0, pauseLMFile ? "-" : 0,
				pauseScoreWeight);
    }

    /*
     * Read list of nbest filenames
     */
    if (nbestFiles) {
	File file(nbestFiles, "r");
	char *line;
        char *strtok_ptr = NULL;
	while ((line = file.getline())) {
	    strtok_ptr = NULL;
	    char *fname = MStringTokUtil::strtok_r(line, wordSeparators, &strtok_ptr);
	    if (!fname) continue;

	    RefString sentid = idFromFilename(fname);

	    makeArray(char, pronScoreFile,
		      (pronScoreDir ? strlen(pronScoreDir) : 0) + 1
				 + strlen(sentid) + strlen(GZIP_SUFFIX) + 1);
	    if (pronScoreDir) {
		sprintf(pronScoreFile, "%s/%s%s", pronScoreDir, sentid,
								GZIP_SUFFIX);
	    }

	    makeArray(char, pauseScoreFile,
		      (pauseScoreDir ? strlen(pauseScoreDir) : 0) + 1
				+ strlen(sentid) + strlen(GZIP_SUFFIX) + 1);
	    if (pauseScoreDir) {
		sprintf(pauseScoreFile, "%s/%s%s", pauseScoreDir, sentid,
								GZIP_SUFFIX);
	    }

	    processNbest(fname, vocab, dictionary, pauseLM,
				    pronScoreDir ? (char *)pronScoreFile : 0,
				    pauseScoreDir ? (char *)pauseScoreFile : 0,
				    pauseScoreWeight);
	}
    }

    exit(0);
}
