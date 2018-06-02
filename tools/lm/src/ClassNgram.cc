/*
 * ClassNgram.cc --
 *	N-gram model over word classes
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1999-2010 SRI International, 2013 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/ClassNgram.cc,v 1.40 2016/04/09 06:53:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>

#include "ClassNgram.h"
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
 * We use pairs of strings over VocabIndex (type ClassNgramState)
 * as keys into the trellis.  Define the necessary support functions.
 */

static inline size_t
LHash_hashKey(const ClassNgramState &key, unsigned maxBits)
{
    return (LHash_hashKey(key.classContext, maxBits) +
	    (key.classExpansion == 0 ? 0
		: LHash_hashKey(key.classExpansion, maxBits)))
	   & hashMask(maxBits);
}

static inline int
SArray_compareKey(const ClassNgramState &key1, const ClassNgramState &key2)
{
   int c = SArray_compareKey(key1.classContext, key2.classContext);
   if (c != 0) {
	return c;
   } else {
	return SArray_compareKey(key1.classExpansion, key2.classExpansion);
   }
}

static inline ClassNgramState
Map_copyKey(const ClassNgramState &key)
{
    ClassNgramState copy;

    /*
     * We need to copy the class context since it is created dynamically,
     * but not the class expansion strings, since they reside statically in
     * the LM.
     */
    copy.classContext = Map_copyKey(key.classContext);
    copy.classExpansion = key.classExpansion;
    return copy;
}

static inline void
#if __GNUC__ == 2 && __GNUC_MINOR__ <= 7
Map_freeKey(ClassNgramState key)
#else	// gcc 2.7 doesn't match f(&T) with f(T) here
Map_freeKey(ClassNgramState &key) // gcc 2.7 has problem matching this
#endif
{
    Map_freeKey(key.classContext);
}

static inline Boolean
#if __GNUC__ == 2 && __GNUC_MINOR__ <= 7
LHash_equalKey(const ClassNgramState key1, const ClassNgramState key2)
#else	// gcc 2.7 doesn't match f(&T,&T) with f(T,T) here
LHash_equalKey(const ClassNgramState &key1, const ClassNgramState &key2)
#endif
{
    /*
     * Note class expansions point into a shared structure, so can be
     * compared as pointers.
     */
    return LHash_equalKey(key1.classContext, key2.classContext) &&
	   key1.classExpansion == key2.classExpansion;
}

static inline Boolean
Map_noKeyP(const ClassNgramState &key)
{
    return (key.classContext == 0);
}

static inline void
Map_noKey(ClassNgramState &key)
{
    key.classContext = 0;
}

/*
 * output operator
 */
ostream &
operator<< (ostream &stream, const ClassNgramState &state)
{
    stream << "<" << state.classContext << ",";
    if (state.classExpansion != 0 && state.classExpansion[0] != Vocab_None) {
	stream << state.classExpansion;
    } else {
	stream << "NULL";
    }
    stream  << ">";
    return stream;
}

/*
 * LM code 
 */

ClassNgram::ClassNgram(Vocab &vocab, SubVocab &classVocab, unsigned order)
    : Ngram(vocab, order),
      trellis(maxWordsPerLine + 2 + 1, 0), savedLength(0),
      classVocab(classVocab), simpleNgram(false)
{
    /*
     * Make sure the classes are subset of base vocabulary 
     */
    assert(&vocab == &classVocab.baseVocab());
}

ClassNgram::~ClassNgram()
{
    clearClasses();
}

void *
ClassNgram::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
{
    if (simpleNgram) {
	return Ngram::contextID(word, context, length);
    } else {
	/*
	 * Due to the DP algorithm, we alway use the full context
	 * (don't inherit Ngram::contextID()).
	 */
	return LM::contextID(word, context, length);
    } 
}

LogP
ClassNgram::contextBOW(const VocabIndex *context, unsigned length)
{
    if (simpleNgram) {
	return Ngram::contextBOW(context, length);
    } else {
	/*
	 * Due to the DP algorithm, we alway use the full context
	 * (don't inherit Ngram::contextBOW()).
	 */
	return LM::contextBOW(context, length);
    } 
}

Boolean
ClassNgram::isNonWord(VocabIndex word)
{
    /*
     * classes are not words: duh!
     */
    return Ngram::isNonWord(word) || 
	    (!simpleNgram && classVocab.getWord(word) != 0);
}

/*
 * Compute the prefix probability of word string (in reversed order)
 * taking hidden events into account.
 * This is done by dynamic programming, filling in the trellis[]
 * array from left to right.
 * Entry trellis[i][state].prob is the probability that of the first
 * i words while being in state.
 * The total prefix probability is the column sum in trellis[],
 * summing over all states.
 * For efficiency reasons, we specify the last word separately.
 * If context == 0, reuse the results from the previous call.
 */
LogP
ClassNgram::prefixProb(VocabIndex word, const VocabIndex *context,
					    LogP &contextProb, TextStats &stats)
{
    /*
     * pos points to the column currently being computed (we skip over the
     *     initial <s> token)
     * prefix points to the tail of context that is used for conditioning
     *     the current word.
     */
    unsigned pos;
    int prefix;

    Boolean wasRunning = running(false);

    if (context == 0) {
	/*
	 * Reset the computation to the last iteration in the loop below
	 */
	pos = prevPos;
	prefix = 0;
	context = prevContext;

	trellis.init(pos);
    } else {
	unsigned len = Vocab::length(context);
	assert(len <= maxWordsPerLine);

	/*
	 * Save these for possible recomputation with different
	 * word argument, using same context
	 */
	prevContext = context;
	prevPos = 0;

	/*
	 * Initialization: The 0 column corresponds to the <s> prefix.
	 */
	VocabIndex initialContext[2];
	if (len > 0 && context[len - 1] == vocab.ssIndex()) {
	    initialContext[0] = context[len - 1];
	    initialContext[1] = Vocab_None;
	    prefix = len - 1;

	} else {
	    initialContext[0] = Vocab_None;
	    prefix = len;
	}

	ClassNgramState initialState;
	initialState.classContext = initialContext;
	initialState.classExpansion = 0;

	/*
	 * Start the DP from scratch if the context has less than two items
	 * (including <s>).  This prevents the trellis from accumulating states
	 * over multiple sentences (which computes the right thing, but
	 * slowly).
	 */
	if (len > 1 &&
	    savedLength > 0 && savedContext[0] == initialContext[0])
	{
	    /*
	     * Skip to the position where the saved
	     * context diverges from the new context.
	     */
	    for (pos = 1;
		 pos < savedLength && prefix > 0 &&
		     context[prefix - 1] == savedContext[pos];
		 pos ++, prefix --)
	    {
		prevPos = pos;
	    }

	    savedLength = pos;
	    trellis.init(pos);
	} else {
	    /*
	     * Start a DP from scratch
	     */
	    trellis.clear();
	    trellis.setProb(initialState, LogP_One);
	    trellis.step();

	    savedContext[0] = initialContext[0];
	    savedLength = 1;
	    pos = 1;
	}
    }

    LogP logSum = LogP_Zero;

    for ( ; prefix >= 0; pos++, prefix--) {
	/*
	 * Keep track of the fact that at least one state has positive
	 * probability, needed for handling of OOVs below.
	 */
	Boolean havePosProb = false;

        VocabIndex currWord;
	
	if (prefix == 0) {
	    currWord = word;
	} else {
	    currWord = context[prefix - 1];

	    /*
	     * Cache the context words for later shortcut processing
	     */
	    savedContext[savedLength ++] = currWord;
	}

	/*
	 * Put underlying Ngram in "running" state (for debugging etc.)
	 * only when processing the last (current) word to prevent
	 * redundant output.
	 */
	if (prefix == 0) {
	    running(wasRunning);
	}

	/*
	 * Iterate over all states for the previous position in trellis
	 */
	TrellisIter<ClassNgramState> prevIter(trellis, pos - 1);

	ClassNgramState prevState;
	LogP prevProb;

	while (prevIter.next(prevState, prevProb)) {
	    ClassNgramState nextState;
	    VocabIndex newContext[maxWordsPerLine + 1];

	    /*
	     * There are two ways to extend a previous state:
	     * - if the state is an incomplete class expansion, then the
	     *   current word has to match the next word in the expansion,
	     *   and the follow-state consists of the same class context and
	     *   the current expansion with the current word removed.
	     * - if the state is a complete class context, then the class 
	     *   context is extended either with the current word, or by
	     *   starting a class expansion that matches the current word.
	     */
	    if (prevState.classExpansion != 0 &&
		prevState.classExpansion[0] != Vocab_None)
	    {
		/*
		 * Skip expansions that don't match
		 */
		if (prevState.classExpansion[0] != currWord) {
		    continue;
		}

		nextState.classContext = prevState.classContext;
		nextState.classExpansion = prevState.classExpansion + 1;

		if (debug(DEBUG_TRANSITIONS)) {
		    cerr << "POSITION = " << pos
			 << " FROM: " << (vocab.use(), prevState)
			 << " TO: " << (vocab.use(), nextState)
			 << " WORD = " << vocab.getWord(currWord)
			 << endl;
		}

		if (prevProb != LogP_Zero) {
		    havePosProb = true;
		}

		/*
		 * For efficiency reasons we don't update the trellis
		 * when at the final word.  In that case we just record
		 * the total probability.
		 */
		if (prefix > 0) {
		    trellis.update(prevState, nextState, LogP_One);
		} else {
		    logSum = AddLogP(logSum, prevProb);
		}
	    } else {
		/*
		 * Set up the extended context.  
		 */
		unsigned i;
		for (i = 0; i < maxWordsPerLine-1 && 
				prevState.classContext[i] != Vocab_None; i ++)
		{
		    newContext[i + 1] = prevState.classContext[i];
		}
		newContext[i + 1] = Vocab_None;

		nextState.classContext = newContext;
		nextState.classExpansion = 0;

		/*
		 * Extend context with current word
		 */
		newContext[0] = currWord;

		/*
		 * Transition prob out of previous context to current word:
		 */
		LogP wordProb =
			Ngram::wordProb(currWord, prevState.classContext);

		if (prevProb != LogP_Zero && wordProb != LogP_Zero) {
		    havePosProb = true;
		}
		
                /*
                 * Truncate context to what is actually used by LM.
                 */
                unsigned usedLength;
                Ngram::contextID(Vocab_None, newContext, usedLength);

                VocabIndex truncatedContextWord = newContext[usedLength];
                newContext[usedLength] = Vocab_None;
	   
		if (debug(DEBUG_TRANSITIONS)) {
		    cerr << "POSITION = " << pos
			 << " FROM: " << (vocab.use(), prevState)
			 << " TO: " << (vocab.use(), nextState)
			 << " WORD = " << vocab.getWord(currWord)
			 << " PROB = " << wordProb
			 << endl;
		}

		/*
		 * For efficiency reasons we don't update the trellis
		 * when at the final word.  In that case we just record
		 * the total probability.
		 */
		if (prefix > 0) {
		    trellis.update(prevState, nextState, wordProb);
		} else {
		    logSum = AddLogP(logSum, prevProb + wordProb);
		}

                /*
                 * Restore newContext
                 */
                newContext[usedLength] = truncatedContextWord;

		/*
		 * Now extend context by all class expansions that can start
		 * with the current word
		 */
		Map2Iter2<VocabIndex,ClassExpansion,Prob>
					expandIter(classDefsByWord, currWord);
		Prob *expansionProb;
		ClassExpansion classAndExpansion;

		while ((expansionProb = expandIter.next(classAndExpansion))) {

		    VocabIndex clasz = classAndExpansion[0]; 

		    /*
		     * Prepend new class to context
		     */ 
		    newContext[0] = clasz;

		    /*
		     * Transition prob out of previous context to new class
		     */
		    LogP classProb =
			    Ngram::wordProb(clasz, prevState.classContext);

		    /*
		     * Truncate context to what is actually used by LM
		     */
		    unsigned usedLength;
		    Ngram::contextID(Vocab_None, newContext, usedLength);

		    VocabIndex truncatedContextWord = newContext[usedLength];
		    newContext[usedLength] = Vocab_None;

		    /*
		     * Discard the class itself and the first word,
		     * which is already consumed by the current position.
		     */
		    nextState.classExpansion = classAndExpansion + 2;

		    if (debug(DEBUG_TRANSITIONS)) {
			cerr << "POSITION = " << pos
			     << " FROM: " << (vocab.use(), prevState)
			     << " TO: " << (vocab.use(), nextState)
			     << " WORD = " << vocab.getWord(currWord)
			     << " PROB = " << classProb
			     << " EXPANDPROB = " << *expansionProb
			     << endl;
		    }

		    if (classProb != LogP_Zero && *expansionProb != 0.0) {
			havePosProb = true;
		    }

		    /*
		     * For efficiency reasons we don't update the trellis
		     * when at the final word.  In that case we just record
		     * the total probability.
		     */
		    if (prefix > 0) {
			trellis.update(prevState, nextState,
				    classProb + ProbToLogP(*expansionProb));
		    } else {
			logSum = AddLogP(logSum, prevProb + 
				    classProb + ProbToLogP(*expansionProb));
		    }

		    /*
		     * Restore newContext
		     */
		    newContext[usedLength] = truncatedContextWord;
		}
            }
	}

	/*
	 * Set noevent state probability to the previous total prefix
	 * probability if the current word had probability zero in all
	 * states, and we are not yet at the end of the prefix.
	 * This allows us to compute conditional probs based on
	 * truncated contexts, and to compute the total sentence probability
	 * leaving out the OOVs, as required by sentenceProb().
	 * We include the words in the state so that context cues (e.g., <s>)
	 * can still be used down the line.
	 */
	if (prefix > 0 && !havePosProb) {
	    ClassNgramState newState;
	    newState.classContext = &context[prefix - 1];
	    newState.classExpansion = 0;

	    trellis.init(pos);
	    trellis.setProb(newState, trellis.sumLogP(pos - 1));

	    if (currWord == vocab.unkIndex()) {
		stats.numOOVs ++;
	    } else {
	        stats.zeroProbs ++;
	    }
	}
	
	trellis.step();
	prevPos = pos;
    }

    if (prevPos > 0) {
	contextProb = trellis.sumLogP(prevPos - 1);
    } else { 
	contextProb = LogP_One;
    }
    return logSum;
}

/*
 * The conditional word probability is computed as
 *	p(w1 .... wk)/p(w1 ... w(k-1)
 */
LogP
ClassNgram::wordProb(VocabIndex word, const VocabIndex *context)
{
    if (simpleNgram) {
	return Ngram::wordProb(word, context);
    } else {
	LogP cProb;
	TextStats stats;
	LogP pProb = prefixProb(word, context, cProb, stats);

	if (cProb == LogP_Zero && pProb == LogP_Zero) {
	    return LogP_Zero;
	} else {
	    return pProb - cProb;
	}
    }
}

LogP
ClassNgram::wordProbRecompute(VocabIndex word, const VocabIndex *context)
{
    if (simpleNgram) {
	return Ngram::wordProbRecompute(word, context);
    } else {
	LogP cProb;
	TextStats stats;
	LogP pProb = prefixProb(word, 0, cProb, stats);

	if (cProb == LogP_Zero && pProb == LogP_Zero) {
	    return LogP_Zero;
	} else {
	    return pProb - cProb;
	}
    }
}

/*
 * Sentence probabilities from indices
 *	This version computes the result directly using prefixProb to
 *	avoid recomputing prefix probs for each prefix.
 */
LogP
ClassNgram::sentenceProb(const VocabIndex *sentence, TextStats &stats)
{
    /*
     * The debugging machinery is not duplicated here, so just fall back
     * on the general code for that.
     */
    if (simpleNgram || debug(DEBUG_PRINT_WORD_PROBS)) {
	return Ngram::sentenceProb(sentence, stats);
    } else {
	unsigned int len = vocab.length(sentence);
	LogP totalProb;

	makeArray(VocabIndex, reversed, len + 2 + 1);

	/*
	 * Contexts are represented most-recent-word-first.
	 * Also, we have to prepend the sentence-begin token,
	 * and append the sentence-end token.
	 */
	len = prepareSentence(sentence, reversed, len);

	/*
	 * Invalidate cache (for efficiency only)
	 */
	savedLength = 0;

	LogP contextProb;
	totalProb = prefixProb(reversed[0], reversed + 1, contextProb, stats);

	/* 
	 * OOVs and zeroProbs are updated by prefixProb()
	 */
	stats.numSentences ++;
	stats.prob += totalProb;
	stats.numWords += len;

	return totalProb;
    }
}

Boolean
ClassNgram::read(File &file, Boolean limitVocab)
{
    /*
     * First read the ngram data in standard format
     */
    if (!Ngram::read(file, limitVocab)) {
	return false;
    }
	
    /*
     * Now read class definitions
     */
    return readClasses(file);
}

Boolean
ClassNgram::write(File &file)
{
    /*
     * First write out the Ngram parameters in the usual format
     */
    if (!Ngram::write(file)) {
	return false;
    }
    
    file.fprintf("\n");

    /*
     * Now write the class definitions
     */
    writeClasses(file);

    file.fprintf("\n");

    return true;
}

void
ClassNgram::clearClasses()
{
    /* 
     * Remove all class definitions
     */
    classDefs.clear();
    classDefsByWord.clear();
}

Boolean
ClassNgram::readClasses(File &file)
{
    char *line;
    Boolean classesCleared = false;

    while ((line = file.getline())) {
	VocabString words[maxWordsPerLine];

	/*
	 * clear old class definitions only when encountering first new
	 * class definition
	 */
	if (!classesCleared) {
	    clearClasses();
	    classesCleared = true;
	}

	unsigned howmany = Vocab::parseWords(line, words, maxWordsPerLine);

	if (howmany == maxWordsPerLine) {
	    file.position() << "class definition has too many fields\n";
	    return false;
	}

	/*
	 * First word contains class name
	 */
	VocabIndex clasz = classVocab.addWord(words[0]);

	Prob prob = 1.0;
	VocabString *expansionWords;

	/*
	 * If second word is numeral, assume it's the class expansion prob
	 */
	if (howmany > 1 && parseProb(words[1], prob)) {
	    expansionWords = &words[2];
	} else {
	    expansionWords = &words[1];
	}

	/*
	 * Add expansion words to vocabulary and store.
	 * The first position in the string is reserved for the class itself
	 * (for use in classDefsByWord).
	 */
	VocabIndex classAndExpansion[maxWordsPerLine + 1];
	classAndExpansion[0] = clasz;
	if (vocab.addWords(expansionWords, &classAndExpansion[1],
						maxWordsPerLine) == 0)
	{
	    file.position() << "class expansion contains no words\n";
	    return false;
	}

	*classDefs.insert(clasz, &classAndExpansion[1]) = prob;

	/*
	 * Index the class and its expansion by the first word.
	 */
	*classDefsByWord.insert(classAndExpansion[1], classAndExpansion) = prob;
    }
	
    return true;
}

void
ClassNgram::writeClasses(File &file)
{
    VocabIndex clasz;
    Map2Iter<VocabIndex, ClassExpansion, Prob>
				classIter(classDefs, vocab.compareIndex());

    while (classIter.next(clasz)) {
	Map2Iter2<VocabIndex, ClassExpansion, Prob>
				iter(classDefs, clasz, vocab.compareIndices());
	ClassExpansion expansion;
	Prob *prob;

	while ((prob = iter.next(expansion))) {
	    file.fprintf("%s %.*lf", vocab.getWord(clasz), Prob_Precision, *prob);

	    for (unsigned i = 0; expansion[i] != Vocab_None; i ++) {
		file.fprintf(" %s", vocab.getWord(expansion[i]));
	    }
	    file.fprintf("\n");
	}
    }
    file.fprintf("\n");
}

/*
 * Compile class-ngram into word-ngram model by expanding classes
 * Algorithm:
 * 1 - Compute joint probabilities for expanded word-ngrams
 * 2 - Compute conditional word-ngram probabilities from joint probs.
 * 3 - Compute backoff weights
 * The second argument gives the length of expanded ngrams whose
 * conditional probability should be computed using the forward algorithm.
 * This is more expensive but gives better results for ngrams much longer
 * than those contained in the original model.
 */
Ngram *
ClassNgram::expand(unsigned newOrder, unsigned expandExact)
{
    NgramCounts<LogP> ngramProbs(vocab, maxWordsPerLine);
					// accumulators for joint ngram probs

    unsigned maxNgramLength = 0;	// to determine final ngram order

    VocabIndex wordNgram[maxWordsPerLine];
    wordNgram[0] = Vocab_None;

    makeArray(VocabIndex, context, order + 2);

    /*
     * Turn off the DP for the computation of joint context probabilities
     */
    simpleNgram = true;

    unsigned i;
    for (i = 0 ; i < order; i ++) {
	BOnode *node;
	NgramBOsIter iter(*this, context, i);

	while ((node = iter.next())) {
	    LogP jointContextProb = contextProb(context);

	    /*
	     * Flip context to give regular ngram order and allow appending
	     * final word.
	     */
	    Vocab::reverse(context);

	    /*
	     * Enumerate all follow ngrams
	     */
	    NgramProbsIter piter(*node);
	    VocabIndex clasz;
	    LogP *ngramProb;

	    while ((ngramProb = piter.next(clasz))) {
		context[i] = clasz;
		context[i+1] = Vocab_None;

		/*
		 * Expand the full ngram.
		 */
		ClassNgramExpandIter expandIter(*this, context, wordNgram);

		LogP expandProb;
		unsigned firstLen, lastLen;
		while (expandIter.next(expandProb, firstLen, lastLen)) {
		    unsigned expandedLen = Vocab::length(wordNgram);

		    if (expandedLen > maxNgramLength) {
			maxNgramLength = expandedLen;
		    }

		    LogP newProb = jointContextProb + *ngramProb + expandProb;

		    /*
		     * Increment the total joint probability for all 
		     * ngrams resulting from the last class expansion.
		     * (Shorter prefixes of the expansion are taken care
		     * of automatically as a result of expanding prefixes
		     * of the context.)
		     */
		    for (unsigned j = 0; j < lastLen; j ++) {
			for (unsigned k = 0; k < firstLen; k ++) {
			    /*
			     * Truncate the N-gram starting from the back
			     */
			    wordNgram[expandedLen - j] = Vocab_None;

			    LogP *oldProb =
					ngramProbs.insertCount(&wordNgram[k]);
			    if (*oldProb == 0.0) {
				*oldProb = newProb;
			    } else {
				*oldProb = AddLogP(*oldProb, newProb);
			    }
			}
		    }
		}
	    }

	    context[i] = Vocab_None;
	    Vocab::reverse(context);
	}
    }

    /*
     * Unless requested otherwise, include all ngrams in the new model
     */
    if (newOrder == 0) {
	newOrder = maxNgramLength;
    } else if (newOrder > maxNgramLength) {
	newOrder = maxNgramLength;
    }

    /*
     * Default is to not use exact expansion at all
     */
    if (expandExact == 0) {
	expandExact = newOrder + 1;
    }

    /*
     * Compute joint probs for word ngrams that were expanded in the
     * above step, but are not contained in the original model.
     */
    for (i = 1; i < expandExact; i++) {
	LogP *oldProb;

	NgramCountsIter<LogP> ngramIter(ngramProbs, wordNgram, i);

	/*
	 * This enumerates all i-grams.
	 */
	while ((oldProb = ngramIter.next())) {
	    /*
	     * destructively extract context portion of ngram
	     */
	    Vocab::reverse(wordNgram);

	    if (findProb(wordNgram[0], &wordNgram[1]) == 0) {
		/*
		 * ngram is not in old model:
		 * compute joint probability for this ngram, excluding classes
		 */
		LogP newProb = contextProb(wordNgram);

		if (*oldProb == 0.0) {
		    *oldProb = newProb;
		} else {
		    *oldProb = AddLogP(*oldProb, newProb);
		}
	    }
	    Vocab::reverse(wordNgram);
	}
    }

    simpleNgram = false;

    /*
     * Copy all regular (non-class) words to the new vocabulary,
     * including special tokens.
     */
    SubVocab *newVocab = new SubVocab(vocab);
    assert(newVocab);

    VocabIter viter(vocab);
    VocabIndex wordIndex;
    VocabString wordString;

    while ((wordString = viter.next(wordIndex))) {
	if (!classVocab.getWord(wordIndex)) {
	    newVocab->addWord(wordString);
	} else {
	    /* 
	     * ensure all words in the class expansion are in the new vocab:
	     * this includes classes that occur in expansions of other classes,
	     * even though we currently don't support "context-free" rules
	     */
	    Map2Iter2<VocabIndex, ClassExpansion, Prob>
						iter(classDefs, wordIndex);
	    ClassExpansion expansion;

	    while (iter.next(expansion)) {
		for (i = 0; expansion[i] != Vocab_None; i ++) {
		    VocabString className = classVocab.getWord(expansion[i]);

		    if (className) {
			cerr << "warning: expansion of " << wordString 
			     << " -> " << (vocab.use(), expansion)
			     << " refers to another class: " << className
			     << endl;
			newVocab->addWord(className);
		    }
		}
	    }
	}
    }

    /*
     * Duplicate special word indices in new vocab
     */
    newVocab->unkIndex() = vocab.unkIndex();
    newVocab->ssIndex() = vocab.ssIndex();
    newVocab->seIndex() = vocab.seIndex();
    newVocab->pauseIndex() = vocab.pauseIndex();
    newVocab->addNonEvent(vocab.ssIndex());
    newVocab->addNonEvent(vocab.pauseIndex());

    /*
     * Create new ngram model (inherit debug level from class ngram)
     */
    Ngram *ngram = new Ngram(*newVocab, newOrder);
    assert(ngram != 0);
    ngram->debugme(debuglevel());

    /*
     * For all ngrams, compute probabilities
     */
    for (i = 0; i < newOrder; i++) {
	LogP *contextProb;
	NgramCountsIter<LogP> contextIter(ngramProbs, wordNgram, i);

	/*
	 * This enumerates all contexts, i.e., i-grams.
	 */
	while ((contextProb = contextIter.next())) {
	    /*
	     * The probability of <s> is -Inf in the model, but it should be
	     * P(</s>) for purposes of normalization when computing the
	     * conditional probs below (consistent with LM::contextProb()).
	     */
	    if (i == 1 && wordNgram[0] == vocab.ssIndex()) {
		VocabIndex emptyContext = Vocab_None;
		*contextProb = wordProb(vocab.seIndex(), &emptyContext);
	    }

	    VocabIndex word[2];	/* the follow word */
	    NgramCountsIter<LogP> followIter(ngramProbs, wordNgram, word, 1);
	    LogP *ngramProb;

	    /*
	     * reverse context words in preparation for ngram prob insertion
	     */
	    Vocab::reverse(wordNgram);

	    if (i + 1 >= expandExact) {
		/*
		 * Exact conditional probability for expanded ngram:
		 * Run the forward algorithm (by way of wordProb).
		 */
		while ((ngramProb = followIter.next())) {
		    LogP lprob = wordProb(word[0], wordNgram);

		    if (lprob > LogP_One) {
			if (LogPtoProb(lprob) - 1.0 > Prob_Epsilon) {
			    cerr << "bad conditional prob for \""
				 << (vocab.use(), wordNgram) << "\": "
				 << LogPtoProb(lprob) << " > 1\n";
			}
			lprob = LogP_One;
		    }

		    if (debug(DEBUG_ESTIMATES)) {
			dout() << "CONTEXT " << (vocab.use(), wordNgram)
			       << " WORD " << vocab.getWord(word[0])
			       << " EXACT LPROB " << lprob
			       << endl;
		    }
		    *ngram->insertProb(word[0], wordNgram) = lprob;
		}
	    } else {
		/*
		 * Compute sum of all ngram probs
		 */
		LogP probSum = LogP_Zero;

		while ((ngramProb = followIter.next())) {
		    probSum = AddLogP(probSum, *ngramProb);
		}

		/*
		 * Compute the sum of ngram probs
		 * - because it needs to be computed for the empty context
		 * - to check for abnormal conditions
		 */
		if (i == 0 || probSum > *contextProb) {
		    if (i > 0 &&
			LogPtoProb(probSum) - LogPtoProb(*contextProb)
							    > Prob_Epsilon &&
			debug(DEBUG_ESTIMATE_WARNINGS))
		    {
			cerr << "warning: prob for context \""
			     << (vocab.use(), wordNgram)
			     << "\" lower than total ngram prob for words "
			     << "(" << *contextProb << " < " << probSum << ")"
			     << endl;
		    }
		    *contextProb = probSum;
		}

		/*
		 * Enumerate all words that can follow this context
		 */
		followIter.init();

		while ((ngramProb = followIter.next())) {
		    LogP lprob = *ngramProb - *contextProb;

		    if (debug(DEBUG_ESTIMATES)) {
			dout() << "CONTEXT " << (vocab.use(), wordNgram)
			       << " WORD " << vocab.getWord(word[0])
			       << " NUMER " << *ngramProb
			       << " DENOM " << *contextProb
			       << " LPROB " << lprob
			       << endl;
		    }
		    *ngram->insertProb(word[0], wordNgram) = lprob;
		}
	    }

	    Vocab::reverse(wordNgram);
	}
    }

    /*
     * Complete new model estimation by filling in backoff weights
     */
    ngram->recomputeBOWs();

    return ngram;
}

/*
 * Enumerate all class expansions in a mixed word/class token string
 */

ClassNgramExpandIter::ClassNgramExpandIter(ClassNgram &ngram,
		const VocabIndex *classes, VocabIndex *buffer)
    : ngram(ngram), classes(classes), buffer(buffer),
      expandIter(0), subIter(0), done(false)
{
    /*
     * Find the first class token in classes string
     */
    for (firstClassPos = 0;
	 classes[firstClassPos] != Vocab_None;
	 firstClassPos++)
    {
	if (ngram.classVocab.getWord(classes[firstClassPos]) != 0) {
	    break;
	}
    }
		
    /*
     * If there is a class token, set up the iterator over its expansions
     */
    if (classes[firstClassPos] != Vocab_None) {
	expandIter = new Map2Iter2<VocabIndex,ClassExpansion,Prob>
				    (ngram.classDefs, classes[firstClassPos]);
	assert(expandIter != 0);
    }

    /*
     * Copy the words preceding the first class into the buffer
     */
    for (unsigned i = 0; i < firstClassPos; i ++) {
	buffer[i] = classes[i];
    }
    buffer[firstClassPos] = Vocab_None;
}

ClassNgramExpandIter::~ClassNgramExpandIter()
{
    delete expandIter;
    delete subIter;
}

/*
 * Return the next class-expanded word string.
 * Also return the aggregate probability of all expansions in the current
 * string (prob), the expanded length of the first input token (firstLen),
 * and the expanded length of the last input token (lastLen).
 */
VocabIndex *
ClassNgramExpandIter::next(LogP &prob, unsigned &firstLen, unsigned &lastLen)
{
    if (done) {
	return 0;
    } else if (expandIter == 0) {
	/*
	 * If the class iterator is not active, we have an all-words string
	 * and just return it.
	 */
	done = true;
	prob = LogP_One;
	firstLen = lastLen = (classes[0] == Vocab_None ? 0 : 1);
	return buffer;
    } else {
	while (1) {
	    if (subIter == 0) {
		/*
		 * The sub-iteration is done, advance to the next 
		 * expansion of the first class.
		 */
		ClassExpansion expansion;
		Prob *expandProb = expandIter->next(expansion);

		if (expandProb == 0) {
		    done = true;
		    return 0;
		} else {
		    /*
		     * Remember across invocations
		     */
		    prob1 = ProbToLogP(*expandProb);

		    /*
		     * Append expansion to buffer, and record its length
		     */
		    for (firstClassLen = 0;
			 expansion[firstClassLen] != Vocab_None;
			 firstClassLen ++)
		    {
			buffer[firstClassPos + firstClassLen] =
						expansion[firstClassLen];
		    }

		    /*
		     * Create recursive iterator to expand the 
		     * remaining string
		     */
		    subIter = new ClassNgramExpandIter(ngram,
					&classes[firstClassPos + 1],
					&buffer[firstClassPos + firstClassLen]);
		    assert(subIter != 0);
		}
	    }

	    LogP subProb;
	    unsigned subFirstLen, subLastLen;
	    if (subIter->next(subProb, subFirstLen, subLastLen) == 0) {
		/*
		 * expansion of rest string exhausted
		 * -- continue expanding first class 
		 */
		delete subIter;
		subIter = 0;
	    } else {
		prob = prob1 + subProb;
		firstLen = (firstClassPos == 0 ? firstClassLen : 1);
		lastLen = (classes[firstClassPos + 1] == Vocab_None ? 
						firstClassLen : subLastLen);
		return buffer;
	    }
	}
    }
}

