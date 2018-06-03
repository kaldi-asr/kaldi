/*
 * ClassNgram.h --
 *	N-gram model over word classes
 *
 * Copyright (c) 1999-2010 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/ClassNgram.h,v 1.13 2010/06/02 05:49:58 stolcke Exp $
 *
 */

#ifndef _ClassNgram_h_
#define _ClassNgram_h_

#include <stdio.h>

#include "Ngram.h"
#include "Trellis.h"
#include "SubVocab.h"
#include "Map2.h"
#include "Array.h"

/* 
 * The DP trellis to evaluate a class ngram contains the N-gram context
 * over classes, as well as the string of words left to be expanded in the
 * current class.
 */
typedef const VocabIndex *ClassExpansion;

typedef struct {
	const VocabIndex *classContext;
	ClassExpansion classExpansion;
} ClassNgramState;

ostream &operator<< (ostream &, const ClassNgramState &state);

class ClassNgramExpandIter;		// forward declaration

/*
 * A class N-gram language model that allows words to be members in multiple
 * classes, and class expansions to contain strings of > 1 words.
 * The vocabulary contains both words and classes, with classes being
 * identified in a SubVocab.
 */
class ClassNgram: public Ngram
{
    friend class ClassNgramExpandIter;

public:
    ClassNgram(Vocab &vocab, SubVocab &classVocab, unsigned order);
    ~ClassNgram();

    /*
     * LM interface
     */
    LogP wordProb(VocabIndex word, const VocabIndex *context);
    LogP wordProbRecompute(VocabIndex word, const VocabIndex *context);
    void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    LogP contextBOW(const VocabIndex *context, unsigned length);

    Boolean isNonWord(VocabIndex word);
    LogP sentenceProb(const VocabIndex *sentence, TextStats &stats);

    Boolean read(File &file, Boolean limitVocab = false);
    Boolean write(File &file);

    /*
     * I/O of class definitions
     */
    virtual Boolean readClasses(File &file);
    void writeClasses(File &file);

    /*
     * Compile class-ngram into word-ngram
     */
    Ngram *expand(unsigned newOrder = 0, unsigned expandExact = 0);

protected:
    void clearClasses();		/* remove class definitions */

    Map2<VocabIndex, ClassExpansion, Prob> classDefs;
					/* class expansions:
					 *	Class x Wordstring -> Prob */
    Map2<VocabIndex, ClassExpansion, Prob> classDefsByWord;
					/* reverse index to find possible 
					 * expansions starting with given word:
					 *	FirstWord x (Class,WordString)
					 *		-> Prob */
    
    Trellis<ClassNgramState> trellis;	/* for DP on hidden class expansions */
    const VocabIndex *prevContext;	/* context from last DP */
    unsigned prevPos;			/* position from last DP */
    LogP prefixProb(VocabIndex word, const VocabIndex *context,
					LogP &contextProb, TextStats &stats);
					/* prefix probability */
    Array<VocabIndex> savedContext;	/* saved, rev'd copy of last context */
    unsigned savedLength;		/* length of saved context above */

    SubVocab &classVocab;		/* the class vocabulary */

    Boolean simpleNgram;		/* compute word probs w/o DP */
};

/*
 * Enumeration of all class expansions
 */
class ClassNgramExpandIter
{
public:
    ClassNgramExpandIter(ClassNgram &ngram, const VocabIndex *classes,
							VocabIndex *buffer);
    ~ClassNgramExpandIter();

    VocabIndex *next(LogP &prob, unsigned &firstLen, unsigned &lastLen);
						/* return next expansion and
						 * and total expansion prob */

private:
    ClassNgram &ngram;				/* model defining classes */
    const VocabIndex *classes;			/* input string */
    VocabIndex *buffer;				/* expanded result string */

    unsigned firstClassPos;			/* position of first class */
    unsigned firstClassLen;			/* length of first expansion */
    Map2Iter2<VocabIndex,ClassExpansion,Prob> *expandIter;
						/* expansions of first class */
    LogP prob1;					/* their log probability */
    ClassNgramExpandIter *subIter;		/* recursive expansion of
						 * remaining substring */
    Boolean done;				/* are we, yet ? */
};

#endif /* _ClassNgram_h_ */
