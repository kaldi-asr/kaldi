/*
 * SimpleClassNgram.cc --
 *	N-gram model over word classes with unique class membership
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2002-2012 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/SimpleClassNgram.cc,v 1.8 2012/10/18 20:55:22 mcintyre Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>

#include "SimpleClassNgram.h"
#include "Trellis.cc"
#include "LHash.cc"
#include "Array.cc"
#include "Map2.cc"
#include "NgramStats.cc"

#define DEBUG_ESTIMATE_WARNINGS		1	/* from Ngram.cc */
#define DEBUG_PRINT_WORD_PROBS          2	/* from LM.cc */
#define DEBUG_NGRAM_HITS		2	/* from Ngram.cc */
#define DEBUG_TRANSITIONS		4
#define DEBUG_ESTIMATES			4	/* from Ngram.cc */

/* 
 * replace words with classes
 */

static const VocabIndex emptyContext[] = { Vocab_None };
LogP
SimpleClassNgram::replaceWithClass(const VocabIndex *words, VocabIndex *classes,							      unsigned maxWords)
{
    LogP xprob = LogP_One;		// sum of class expansions log probs
    
    unsigned i;
    for (i = 0; i < maxWords && words[i] != Vocab_None; i ++) {
	/*
	 * Find class for word
	 */
	Map2Iter2<VocabIndex,ClassExpansion,Prob>
				expandIter(classDefsByWord, words[i]);
	ClassExpansion classAndExpansion;
	Prob *expansionProb = expandIter.next(classAndExpansion);

	/*
	 * If the word is not part of a class expansion, or if the class 
	 * is not defined in the LM, then keep the word;
	 * otherwise replace it with its class.
	 */

	if (expansionProb == 0 ||
	    findProb(classAndExpansion[0], emptyContext) == 0)
	{
	    classes[i] = words[i];
	} else {
	    classes[i] = classAndExpansion[0]; 
	    xprob += ProbToLogP(*expansionProb);
	}
    }
    classes[i] = Vocab_None;

    return xprob;
}

LogP
SimpleClassNgram::replaceWithClass(VocabIndex word, VocabIndex &clasz)
{
    VocabIndex words[2];
    VocabIndex classes[2];

    words[0] = word;
    words[1] = Vocab_None;

    LogP xprob = replaceWithClass(words, classes, 1);
    clasz = classes[0];

    return xprob;
}

void *
SimpleClassNgram::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
{
    if (simpleNgram) {
	return Ngram::contextID(word, context, length);
    } else {
	VocabIndex wordClass;

	if (word == Vocab_None) {
	    wordClass = Vocab_None;
	} else {
	    replaceWithClass(word, wordClass);
	}

	makeArray(VocabIndex, classes, vocab.length(context) + 1);
	replaceWithClass(context, classes, order - 1);

	return Ngram::contextID(wordClass, classes, length);
    } 
}

LogP
SimpleClassNgram::contextBOW(const VocabIndex *context, unsigned length)
{
    if (simpleNgram) {
	return Ngram::contextBOW(context, length);
    } else {
	makeArray(VocabIndex, classes, vocab.length(context) + 1);
	replaceWithClass(context, classes, order - 1);

	return Ngram::contextBOW(classes, length);
    } 
}

/*
 * The conditional word probability is computed as
 *	p(w1 .... wk)/p(w1 ... w(k-1)
 */
LogP
SimpleClassNgram::wordProb(VocabIndex word, const VocabIndex *context)
{
    if (simpleNgram) {
	LogP result = Ngram::wordProb(word, context);
	return result;
    } else {
	VocabIndex wordClass;
	LogP xprob = replaceWithClass(word, wordClass);

	// expand savedContext cache to length needed
	savedContext[vocab.length(context)] = Vocab_None;
	replaceWithClass(context, &savedContext[0], order - 1);

	return xprob + Ngram::wordProb(wordClass, &savedContext[0]);
    }
}

LogP
SimpleClassNgram::wordProbRecompute(VocabIndex word, const VocabIndex *context)
{
    if (simpleNgram) {
	return Ngram::wordProbRecompute(word, context);
    } else {
	VocabIndex wordClass;
	LogP xprob = replaceWithClass(word, wordClass);

	// reuse class context in savedContext
	return xprob + Ngram::wordProb(wordClass, &savedContext[0]);
    }
}

/*
 * Sentence probabilities from indices
 *	This version computes the result directly using prefixProb to
 *	avoid recomputing prefix probs for each prefix.
 */
LogP
SimpleClassNgram::sentenceProb(const VocabIndex *sentence, TextStats &stats)
{

    /*
     * The debugging machinery is not duplicated here, so just fall back
     * on the general code for that.
     */
    if (simpleNgram || debug(DEBUG_PRINT_WORD_PROBS)) {
	return Ngram::sentenceProb(sentence, stats);
    } else {
	Boolean wasSimpleNgram = simpleNgram;

	unsigned len = vocab.length(sentence);
	makeArray(VocabIndex, classes, len + 1);
	LogP xprob = replaceWithClass(sentence, classes, len);

	simpleNgram = true;
	LogP classProb = Ngram::sentenceProb(classes, stats);
	simpleNgram = wasSimpleNgram;

	stats.prob += xprob;

	return xprob + classProb;
    }
}

Boolean
SimpleClassNgram::readClasses(File &file)
{
    if (!ClassNgram::readClasses(file)) {
	return false;
    }

    if (haveClassDefError) {
	return false;
    }

    /*
     * Check that class expansions conform to SimpleClassNgram constraints
     */
    VocabIndex word;
    Map2Iter<VocabIndex,ClassExpansion,Prob> expandIter(classDefsByWord);

    while (expandIter.next(word)) {
	if (classDefsByWord.numEntries(word) > 1) {
	    file.position() << "word " << vocab.getWord(word)
			    << " has multiple class memberships\n";
	    haveClassDefError = true;
	}

	Map2Iter2<VocabIndex,ClassExpansion,Prob>
				expandIter2(classDefsByWord, word);
	ClassExpansion classAndExpansion;
	Prob *expansionProb = expandIter2.next(classAndExpansion);
	assert(expansionProb != 0);

	if (vocab.length(classAndExpansion) > 2) {
	    file.position() << "class " << vocab.getWord(classAndExpansion[0])
			    << " expands to string of more than one word\n";
	    haveClassDefError = true;
	}
    }
	
    return !haveClassDefError;
}

