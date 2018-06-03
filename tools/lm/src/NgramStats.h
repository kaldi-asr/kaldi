/*
 * NgramStats.h --
 *	N-gram statistics
 *
 * Copyright (c) 1995-2012 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStats.h,v 1.41 2012/10/29 17:25:05 mcintyre Exp $
 *
 */

#ifndef _NgramStats_h_
#define _NgramStats_h_

#include <stdio.h>

#include "XCount.h"
#include "LMStats.h"

#include "Array.h"
#include "Trie.h"
#include "TLSWrapper.h"

const unsigned int      maxLineLength = 10000;
const unsigned int	maxNgramOrder = 100;	/* Used in allocating various
						 * data structures.  For all
						 * practical purposes, this
						 * should infinite. */

extern TLSW_DECL_ARRAY(VocabIndex, countSentenceWidsTLS, maxWordsPerLine +3);
extern TLSW_DECL_ARRAY(char, writeBufferTLS, maxLineLength);

#define NgramNode	Trie<VocabIndex,CountT>

template <class CountT> class NgramCountsIter;	// forward declaration

template <class CountT>
class NgramCounts: public LMStats
{
    friend class NgramCountsIter<CountT>;

public:
    NgramCounts(Vocab &vocab, unsigned int order);
    virtual ~NgramCounts() {};

    unsigned getorder() { return order; };

    /*
     * Individual word/ngram lookup and insertion
     */
    CountT *findCount(const VocabIndex *words)
	{ return counts.find(words); };
    CountT *findCount(const VocabIndex *words, VocabIndex word1)
	{ NgramNode *node = counts.findTrie(words);
	  return node ? node->find(word1) : 0; }
    CountT *insertCount(const VocabIndex *words)
	{ return counts.insert(words); };
    CountT *insertCount(const VocabIndex *words, VocabIndex word1)
	{ NgramNode *node = counts.insertTrie(words);
	  return node->insert(word1); };
    Boolean removeCount(const VocabIndex *words, CountT *removedData = 0)
	{ return counts.remove(words, removedData); };
    Boolean removeCount(const VocabIndex *words, VocabIndex word1, CountT *removedData = 0)
	{ NgramNode *node = counts.findTrie(words);
	  return node->remove(word1, removedData); };

    virtual unsigned countSentence(const VocabString *word)
	{ return countSentence(word, (CountT)1); };
    virtual unsigned countSentence(const VocabString *word, const char *factor);
    virtual unsigned countSentence(const VocabString *word, CountT factor);
    virtual unsigned countSentence(const VocabIndex *word)
	{ return countSentence(word, (CountT)1); };
    virtual unsigned countSentence(const VocabIndex *word, CountT factor);

    void incrementCounts(const VocabIndex *words,
				unsigned minOrder = 1, CountT factor = 1);

    Boolean read(File &file) { return read(file, order); };
    Boolean read(File &file, unsigned int order, Boolean limitVocab = false);
    Boolean readGoogle(const char *dir, unsigned int order,
    						Boolean limitVocab = false);
    Boolean readMinCounts(File &file, unsigned order, Count *minCounts,
    						Boolean limitVocab = false);
    void write(File &file) { write(file, order); };
    void write(File &file, unsigned int order, Boolean sorted = false);
    Boolean writeBinary(File &file) { return writeBinary(file, order); };
    Boolean writeBinary(File &file, unsigned order);

    static unsigned int parseNgram(char *line,
				  VocabString *words, unsigned int max,
				  CountT &count);
    static unsigned int readNgram(File &file,
				  VocabString *words, unsigned int max,
				  CountT &count);
					/* read one ngram count from file */
    static unsigned int writeNgram(File &file, const VocabString *words,
				   CountT count);
					/* write ngram count to file */

    CountT sumCounts() { return sumCounts(order); };
    CountT sumCounts(unsigned int order);
					/* sum child counts on parent nodes */
    unsigned pruneCounts(CountT minCount);
					/* remove low-count N-grams */
    void setCounts(CountT value = 0);	/* set all counts to constant */

    void dump();			/* debugging dump */
    void memStats(MemStats &stats);	/* compute memory stats */
    void clear() { counts.clear(); };	/* delete all counts */

    Boolean intersect;			/* intersect ngrams upon reading */

protected:
    unsigned int order;
    NgramNode counts;
    void addCounts(const VocabIndex *prefix,
			const LHash<VocabIndex, CountT> &set);
    void writeNode(NgramNode &node, File &file, char *buffer, char *bptr,
	    unsigned int level, unsigned int order, Boolean sorted);
    CountT sumNode(NgramNode &node, unsigned level, unsigned order);

    /*
     * Binary format support
     */
    Boolean readBinary(File &file, unsigned order, Boolean limitVocab);
    Boolean readBinaryNode(NgramNode &node,
    					unsigned order, unsigned maxOrder,
					File &file, long long &offset,
					Boolean limitVocab,
					Array<VocabIndex> &vocabMap);
    Boolean writeBinaryNode(NgramNode &node, unsigned level, unsigned order,
    						File &file, long long &offset);
};

/*
 * Iteration over all counts of a given order
 */
template <class CountT>
class NgramCountsIter
{
public:
    NgramCountsIter(NgramCounts<CountT> &ngrams, VocabIndex *keys, 
				unsigned order = 1,
				int (*sort)(VocabIndex, VocabIndex) = 0)
	 : myIter(ngrams.counts, keys, order, sort) {};
					/* all ngrams of length order, starting
					 * at root */
    
    NgramCountsIter(NgramCounts<CountT> &ngrams, const VocabIndex *start,
				VocabIndex *keys, unsigned order = 1,
				int (*sort)(VocabIndex, VocabIndex) = 0)
	 : myIter(*(ngrams.counts.insertTrie(start)), keys, order, sort) {};
					/* all ngrams of length order, rooted
					 * at node indexed by start */

    void init() { myIter.init(); };
    CountT *next()
	{ NgramNode *node = myIter.next();
	  return node ? &(node->value()) : 0; };

private:
    TrieIter2<VocabIndex,CountT> myIter;
};


/*
 * Instantiate the count trie for integer and float count types
 */

#ifdef USE_XCOUNTS
typedef XCount NgramCount;
#else
# ifdef USE_LONGLONG_COUNTS
typedef unsigned long long NgramCount;
# else
typedef unsigned long NgramCount;
# endif
#endif

class NgramStats: public NgramCounts<NgramCount>
{
public:
    NgramStats(Vocab &vocab, unsigned int order)
	: NgramCounts<NgramCount>(vocab, order) {};
    virtual ~NgramStats() {};
};

class NgramsIter: public NgramCountsIter<NgramCount>
{
public:
    NgramsIter(NgramStats &ngrams, VocabIndex *keys, unsigned order = 1,
				int (*sort)(VocabIndex, VocabIndex) = 0)
	: NgramCountsIter<NgramCount>(ngrams, keys, order, sort) {};
    NgramsIter(NgramStats &ngrams, const VocabIndex *start,
				VocabIndex *keys, unsigned order = 1,
				int (*sort)(VocabIndex, VocabIndex) = 0)
	: NgramCountsIter<NgramCount>(ngrams, start, keys, order, sort) {};
};

#endif /* _NgramStats_h_ */

