/*
 * Ngram.h --
 *	N-gram backoff language models
 *
 * Copyright (c) 1995-2012 SRI International, 2012-2014 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/Ngram.h,v 1.57 2014-05-28 00:04:02 stolcke Exp $
 *
 */

#ifndef _Ngram_h_
#define _Ngram_h_

#include <stdio.h>

#include "LM.h"
#include "NgramStats.h"
#include "Discount.h"

#ifdef USE_SARRAY

# define PROB_INDEX_T   SArray
# define PROB_ITER_T    SArrayIter
# include "SArray.h"

#else /* ! USE_SARRAY */

# define PROB_INDEX_T   LHash
# define PROB_ITER_T    LHashIter
# include "LHash.h"

#endif /* USE_SARRAY */

#include "Trie.h"

typedef struct {
    LogP			bow;		/* backoff weight */
    PROB_INDEX_T<VocabIndex,LogP>	probs;	/* word probabilities */
} BOnode;

typedef Trie<VocabIndex,BOnode> BOtrie;

const unsigned defaultNgramOrder = 3;

class NgramBayesMix;				/* forward declaration */

class Ngram: public LM
{
    friend class NgramBOsIter;

public:
    Ngram(Vocab &vocab, unsigned order = defaultNgramOrder);
    virtual ~Ngram() {};

    unsigned setorder(unsigned neworder = 0);   /* change/return ngram order */

    /*
     * LM interface
     */
    virtual LogP wordProb(VocabIndex word, const VocabIndex *context);
    virtual void *contextID(const VocabIndex *context, unsigned &length)
	{ return contextID(Vocab_None, context, length); };
    virtual void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    virtual LogP contextBOW(const VocabIndex *context, unsigned length);

    virtual Boolean read(File &file, Boolean limitVocab = false);
    virtual Boolean write(File &file);
    virtual Boolean writeWithOrder(File &file, unsigned int order);
    Boolean writeBinaryV1(File &file);

    virtual Boolean &skipOOVs() { return _skipOOVs; };	
				/* backward compatiability: return
				 * zero prob if <unk> is in context */
    virtual Boolean &trustTotals() { return _trustTotals; }
				/* use lower-order counts for ngram totals */

    virtual void useCodebook(PQCodebook &cb)
	{ codebook = &cb; };	/* start using VQ codebook */
    virtual void useCodebook()
	{ codebook = 0; };	/* stop using VQ codebook */

    /*
     * Estimation
     */
    virtual Boolean estimate(NgramStats &stats,
			Count *mincount = 0,
			Count *maxcounts = 0);
    virtual Boolean estimate(NgramStats &stats, Discount **discounts);
    virtual Boolean estimate(NgramCounts<FloatCount> &stats,
							Discount **discounts);
    virtual void mixProbs(Ngram &lm2, double lambda);
    virtual void mixProbs(Ngram &lm1, Ngram &lm2, double lambda);
    virtual void mixProbs(NgramBayesMix &mixLMs);
    virtual void recomputeBOWs();
    virtual void pruneProbs(double threshold, unsigned minorder = 2,
							LM *historyLM = 0);
    virtual void pruneLowProbs(unsigned minorder = 2);
    virtual void rescoreProbs(LM &lm);

    /*
     * Statistics
     */
    virtual Count numNgrams(unsigned int n) const;
    virtual void memStats(MemStats &stats);
    virtual Count countParams(SArray<LogP, FloatCount> &params);

    /*
     * Low-level access
     */
    LogP *findBOW(const VocabIndex *context) const;
    LogP *insertBOW(const VocabIndex *context);
    LogP *findProb(VocabIndex word, const VocabIndex *context) const;
    LogP *insertProb(VocabIndex word, const VocabIndex *context);
    void removeBOW(const VocabIndex *context);
    void removeProb(VocabIndex word, const VocabIndex *context);

    void clear();				/* remove all parameters */

protected:
    BOtrie contexts;				/* n-1 gram context trie */
    unsigned int order; 			/* maximal ngram order */

    Boolean _skipOOVs;
    Boolean _trustTotals;

    PQCodebook *codebook;			/* optional VQ codebook */

    /*
     * Helper functions
     */
    virtual LogP wordProbBO(VocabIndex word, const VocabIndex *context,
							unsigned int clen);
    virtual unsigned vocabSize();
    template <class CountType>
	Boolean estimate2(NgramCounts<CountType> &stats, Discount **discounts);
    virtual void fixupProbs();
    virtual void distributeProb(Prob mass, VocabIndex *context);
    virtual Boolean computeBOW(BOnode *node, const VocabIndex *context, 
			    unsigned clen, Prob &numerator, Prob &denominator);
    virtual Boolean computeBOWs(unsigned order);

    /*
     * Binary format support 
     */
    Boolean writeBinaryNgram(File &file);
    Boolean writeBinaryNode(BOtrie &node, unsigned level, File &file,
							long long &offset);
    Boolean writeBinaryV1Node(BOtrie &trie, File &idx, File &dat,
    			      long long &offset, unsigned myOrder);
    Boolean readBinary(File &file, Boolean limitVocab);
    Boolean readBinaryNode(BOtrie &node, unsigned order, unsigned maxOrder,
					File &file, long long &offset,
					Boolean limitVocab,
					Array<VocabIndex> &vocabMap);
    Boolean readBinaryV1(File &file, Boolean limitVocab);
    Boolean readBinaryV1Node(BOtrie &trie, File &idx, File &dat,
			     Boolean limitVocab, Array<VocabIndex> & vocabMap,
			     unsigned myOrder);
    Boolean skipToNextTrie(File &idx, unsigned myOrder);
};

/*
 * Iteration over all backoff nodes of a given order
 */
class NgramBOsIter
{
public:
    NgramBOsIter(const Ngram &lm, VocabIndex *keys, unsigned order,
			int (*sort)(VocabIndex, VocabIndex) = 0)
	 : myIter(lm.contexts, keys, order, sort) {};

    void init() { myIter.init(); };
    BOnode *next()
	{ Trie<VocabIndex,BOnode> *node = myIter.next();
	  return node ? &(node->value()) : 0; }
private:
    TrieIter2<VocabIndex,BOnode> myIter;
};

/*
 * Iteration over all probs at a backoff node
 */
class NgramProbsIter
{
public:
    NgramProbsIter(const BOnode &bonode, 
			int (*sort)(VocabIndex, VocabIndex) = 0)
	: myIter(bonode.probs, sort) {};

    void init() { myIter.init(); };
    LogP *next(VocabIndex &word) { return myIter.next(word); };

private:
    PROB_ITER_T<VocabIndex,LogP> myIter;
};

#endif /* _Ngram_h_ */
