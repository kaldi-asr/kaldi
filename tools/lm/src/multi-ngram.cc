/*
 * multi-ngram --
 *	Assign probabilities of multiword N-gram models
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2000-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: multi-ngram.cc,v 1.12 2010/06/02 05:49:58 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <assert.h>

#include "option.h"
#include "version.h"
#include "File.h"
#include "Vocab.h"
#include "Ngram.h"
#include "MultiwordVocab.h"

#include "Array.cc"
#include "LHash.cc"

static int version = 0;
static unsigned order = defaultNgramOrder;
static unsigned multiOrder = defaultNgramOrder;
static unsigned debug = 0;
static char *vocabFile = 0;
static char *lmFile  = 0;
static char *multiLMfile  = 0;
static char *writeLM  = (char *)"-";
static const char *multiChar = MultiwordSeparator;
static int pruneUnseenNgrams = false;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_UINT, "order", &order, "max ngram order" },
    { OPT_UINT, "multi-order", &multiOrder, "max multiword ngram order" },
    { OPT_UINT, "debug", &debug, "debugging level for lm" },
    { OPT_STRING, "vocab", &vocabFile, "(multiword) vocabulary to be added" },
    { OPT_STRING, "lm", &lmFile, "ngram LM to model" },
    { OPT_STRING, "multi-lm", &multiLMfile, "multiword ngram LM" },
    { OPT_STRING, "write-lm", &writeLM, "multiword ngram output file" },
    { OPT_STRING, "multi-char", &multiChar, "multiword component delimiter" },
    { OPT_TRUE, "prune-unseen-ngrams", &pruneUnseenNgrams, "do not add unseen multiword ngrams" }
};

/*
 * Reassign parameters to the multiword ngrams in multiLM according the
 * probabilities in ngramLM.  vocab is the multiword vocabulary used to 
 * expand ngrams.
 */
void
assignMultiProbs(MultiwordVocab &vocab, Ngram &multiLM, Ngram &ngramLM)
{
    unsigned order = multiLM.setorder();

    for (unsigned i = 0; i < order; i++) {
	BOnode *node;
        makeArray(VocabIndex, context, order + 1);
	NgramBOsIter iter(multiLM, context, i);
	
	while ((node = iter.next())) {
	    /*
	     * buffer holding expanded context, with room to prepend expanded
	     * word
	     */
	    VocabIndex expandedBuffer[2 * maxWordsPerLine + 1];

	    /*
	     * Expand the backoff context with all multiwords
	     */
	    VocabIndex *expandedContext = &expandedBuffer[maxWordsPerLine];
	    unsigned expandedContextLength =
			    vocab.expandMultiwords(context, expandedContext,
							maxWordsPerLine, true);

	    /*
	     * Find the corresponding context in the old LM
	     */
	    unsigned usedLength;
	    ngramLM.contextID(expandedContext, usedLength);
	    expandedContext[usedLength] = Vocab_None;

	    LogP *bow = ngramLM.findBOW(expandedContext);
	    assert(bow != 0);

	    /*
	     * Assign BOW from old LM to new LM context
	     */
	    node->bow = *bow;

	    NgramProbsIter piter(*node);
	    VocabIndex word;
	    LogP *multiProb;
		
	    while ((multiProb = piter.next(word))) {

		VocabIndex multiWord[2];
		multiWord[0] = word;
		multiWord[1] = Vocab_None;

		VocabIndex expandedWord[maxWordsPerLine + 1];
		unsigned expandedWordLength =
			    vocab.expandMultiwords(multiWord, expandedWord,
							    maxWordsPerLine);

		LogP prob = LogP_One;
		for (unsigned j = 0; j < expandedWordLength; j ++) {
		    prob += ngramLM.wordProb(expandedWord[j],
					&expandedBuffer[maxWordsPerLine - j]);

		    expandedBuffer[maxWordsPerLine - 1 - j] = expandedWord[j];
		}

		/*
		 * Set new LM prob to aggregate old LM prob
		 */
		*multiProb = prob;
	    }
	}
    }
}

/*
 * Check a multiword ngram for whether its component ngrams are all 
 * contained in the reference model
 * (used to limit the set of additional ngrams inserted)
 */
Boolean
haveNgramsFor(VocabIndex word, VocabIndex *context, Ngram &ngramLM,
						    MultiwordVocab &vocab)
{
    makeArray(VocabIndex, multiwordNgram, Vocab::length(context) + 2);
    makeArray(VocabIndex, expandedNgram, 2 * maxWordsPerLine + 1);

    /*
     * Assemble complete reversed N-gram of multiword and context
     */
    multiwordNgram[0] = word;
    Vocab::copy(&multiwordNgram[1], context);

    /*
     * Expand the reversed ngram with all multiwords
     */
    unsigned expandedLength =
		vocab.expandMultiwords(multiwordNgram, expandedNgram,
						2 * maxWordsPerLine, true);

    /*
     * Check that all maximal N-grams are contained in reference model
     */
    Boolean ok = true;
    unsigned ngramOrder = ngramLM.setorder();

    if (expandedLength < ngramOrder) {
	if (!ngramLM.findProb(expandedNgram[0], &expandedNgram[1])) {
	    ok = false;
	}
    } else if (expandedLength > 1) {
	for (VocabIndex *startNgram =
				&expandedNgram[expandedLength - ngramOrder];
	     startNgram >= expandedNgram;
	     startNgram --)
	{
	    startNgram[ngramOrder] = Vocab_None;
	    if (!ngramLM.findProb(startNgram[0], &startNgram[1])) {
		ok = false;
		break;
	    }
	}
    }

    return ok;
}

/* 
 * Populate multi-ngram LM with a superset of original ngrams.
 */
void
populateMultiNgrams(MultiwordVocab &vocab, Ngram &multiLM, Ngram &ngramLM)
{
    unsigned order = ngramLM.setorder();
    unsigned multiOrder = multiLM.setorder();

    /*
     * don't exceed the multi-ngram order
     */
    if (order > multiOrder) {
	order = multiOrder;
    }

    /*
     * construct mapping to multiwords that start/end with given words
     */
    LHash<VocabIndex, Array<VocabIndex> > startMultiwords;
    LHash<VocabIndex, Array<VocabIndex> > endMultiwords;

    VocabIndex **expansion;

    VocabIter viter(vocab);
    VocabIndex word;

    while (viter.next(word)) {
	VocabIndex oneWord[2];
	oneWord[0] = word;
	oneWord[1] = Vocab_None;

	VocabIndex expansion[maxWordsPerLine + 1];
	if (vocab.expandMultiwords(oneWord, expansion, maxWordsPerLine) > 1) {
	    VocabIndex startWord = expansion[0];
	    VocabIndex endWord = expansion[Vocab::length(expansion) - 1];

	    Array<VocabIndex> &startIndex = *startMultiwords.insert(startWord);
	    Array<VocabIndex> &endIndex = *endMultiwords.insert(endWord);

	    startIndex[startIndex.size()] = word;
	    endIndex[endIndex.size()] = word;
	}
    }

    /*
     * Populate multi-ngram LM
     */
    for (unsigned i = 0; i < order; i++) {
	BOnode *node;
	makeArray(VocabIndex, context, order + 1);
	NgramBOsIter iter(ngramLM, context, i);
	
	while ((node = iter.next())) {
	    /*
	     * copy BOW to multi ngram
	     */
	    *multiLM.insertBOW(context) = node->bow;
		
	    /*
	     * copy probs to multi ngram
	     */
	    NgramProbsIter piter(*node);
	    VocabIndex word;
	    LogP *prob;

	    while ((prob = piter.next(word))) {
		*multiLM.insertProb(word, context) = *prob;

		Array<VocabIndex> *startIndex = startMultiwords.find(word);
		if (startIndex) {
		    for (unsigned k = 0; k < startIndex->size(); k ++) {
			/*
			 * don't worry, the prob value will be reset later
			 */
			if (!pruneUnseenNgrams ||
		            haveNgramsFor((*startIndex)[k], context,
							ngramLM, vocab))
			{
			    *multiLM.insertProb((*startIndex)[k], context) =
									*prob;
			}
		    }
		}
	    }

	    Array<VocabIndex> *endIndex;
	    if (i > 0 && (endIndex = endMultiwords.find(context[i - 1]))) {
		VocabIndex oldWord = context[i - 1];

		for (unsigned j = 0; j < endIndex->size(); j ++) {
		    context[i - 1] = (*endIndex)[j];

		    Boolean haveNgrams = false;

		    /*
		     * repeat the procedure above for the new context
		     */
		    NgramProbsIter piter(*node);
		    VocabIndex word;
		    LogP *prob;

		    while ((prob = piter.next(word))) {
			*multiLM.insertProb(word, context) = *prob;

			Array<VocabIndex> *startIndex =
						startMultiwords.find(word);
			if (startIndex) {
			    for (unsigned k = 0; k < startIndex->size(); k ++) {
				if (!pruneUnseenNgrams ||
				    haveNgramsFor((*startIndex)[k], context,
							ngramLM, vocab))
				{
				    *multiLM.insertProb((*startIndex)[k],
							    context) = *prob;
				    haveNgrams = true;
				}
			    }
			}
		    }

		    /*
		     * Only insert new context if needed
		     */
		    if (haveNgrams) {
			*multiLM.insertBOW(context) = node->bow;
		    }
		}

		/*
		 * restore old context
		 */
		context[i - 1] = oldWord;
	    }
	}
    }

    /*
     * Remove ngrams made redundant by multiwords
     */
    viter.init();
    while (viter.next(word)) {
	VocabIndex oneWord[2];
	oneWord[0] = word;
	oneWord[1] = Vocab_None;

	VocabIndex expansion[maxWordsPerLine + 1];
	if (vocab.expandMultiwords(oneWord, expansion, maxWordsPerLine) > 1) {
	    Vocab::reverse(expansion);
	    multiLM.removeProb(expansion[0], &expansion[1]);
	    Vocab::reverse(expansion);
	}
    }

    /*
     * Free auxiliary data (XXX: due to flaw in LHash)
     */
    viter.init();
    while (viter.next(word)) {
	Array<VocabIndex> *starts = startMultiwords.find(word);
	if (starts) {
	    starts->~Array();
	}
	Array<VocabIndex> *ends = endMultiwords.find(word);
	if (ends) {
	    ends->~Array();
	}
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

    if (!lmFile) {
	cerr << "-lm must be specified\n";
	exit(2);
    }
    
    /*
     * Construct language models
     */
    MultiwordVocab vocab(multiChar);
    Ngram ngramLM(vocab, order);
    Ngram multiNgramLM(vocab, multiOrder);

    ngramLM.debugme(debug);
    multiNgramLM.debugme(debug);

    if (vocabFile) {
	File file(vocabFile, "r");

	if (!vocab.read(file)) {
	    cerr << "format error in vocab file\n";
	    exit(1);
	}
    }

    /*
     * Read LMs
     */
    {
	File file(lmFile, "r");

	if (!ngramLM.read(file)) {
	    cerr << "format error in lm file\n";
	    exit(1);
	}
    }

    if (multiLMfile) {
	File file(multiLMfile, "r");

	if (!multiNgramLM.read(file)) {
	    cerr << "format error in multi-lm file\n";
	    exit(1);
	}
    }

    /*
     * If a vocabulary was specified assume that we want to add new ngrams
     * containing multiwords.
     */
    if (vocabFile) {
	populateMultiNgrams(vocab, multiNgramLM, ngramLM);
    }

    assignMultiProbs(vocab, multiNgramLM, ngramLM);

    {
	File file(writeLM, "w");
	multiNgramLM.write(file);
    }

    exit(0);
}
