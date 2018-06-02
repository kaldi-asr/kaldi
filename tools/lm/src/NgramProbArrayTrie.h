/*
 * NgramProbArrayTrie.h --
 *	Trie indexing Prob vectors with Ngrams
 *
 * Copyright (c) 2013 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramProbArrayTrie.h,v 1.5 2014-05-26 03:00:28 stolcke Exp $
 *
 */

#ifndef _NgramProbArrayTrie_h_
#define _NgramProbArrayTrie_h_

#include <stdio.h>

#include "Vocab.h"
#include "Prob.h"
#include "Trie.h"
#include "Array.cc"

#define NgramProbArrayTrieNode	Trie<VocabIndex,ZeroArray<Prob> >

class NgramProbArrayTrie
{
    friend class NgramProbArrayTrieIter;

public:
    NgramProbArrayTrie(Vocab &vocab, unsigned order, unsigned dim);
    virtual ~NgramProbArrayTrie() {};

    unsigned getorder() { return order; };
    unsigned getdim() { return dimension; };
    unsigned setdim(unsigned dim) { return (dimension = dim > 0 ? dim : 1); };

    /*
     * Individual word/ngram lookup and insertion
     */
    ZeroArray<Prob> *findProbs(const VocabIndex *words)
	{ return probs.find(words); };
    ZeroArray<Prob> *findPrefixProbs(const VocabIndex *words, unsigned &depth)
	{ return probs.findPrefix(words, depth); };
    ZeroArray<Prob> *insertProbs(const VocabIndex *words)
	{ return probs.insert(words); };
    Boolean removeProbs(const VocabIndex *words,
			ZeroArray<Prob> *removedData = 0)
	{ return probs.remove(words, removedData); };

    Boolean read(File &file) { return read(file, order); };
    Boolean read(File &file, unsigned order, Boolean limitVocab = false);
    void write(File &file) { write(file, order); };
    void write(File &file, unsigned order, Boolean sorted = false);

    int parseNgram(char *line, VocabString *words, unsigned max,
				  ZeroArray<Prob> &probs);
    int readNgram(File &file, VocabString *words, unsigned max,
				  ZeroArray<Prob> &probs);
				/* read one ngram probs vector from file */
    unsigned writeNgram(File &file, const VocabString *words,
				  ZeroArray<Prob> &probs);
				/* write ngram probs vector to file */

    void dump();			/* debugging dump */
    void memStats(MemStats &stats);	/* compute memory stats */
    void clear();			/* delete all data */
protected:
    Vocab &vocab;
    unsigned order;
    unsigned dimension;
    NgramProbArrayTrieNode probs;
    void writeNode(NgramProbArrayTrieNode &node, File &file, char *buffer,
	    char *bptr, unsigned level, unsigned order, Boolean sorted);
};

/*
 * Iteration over all vectors of a given order
 */
class NgramProbArrayTrieIter
{
public:
    NgramProbArrayTrieIter(NgramProbArrayTrie &trie, VocabIndex *keys, 
				unsigned order = 1,
				int (*sort)(VocabIndex, VocabIndex) = 0)
	 : myIter(trie.probs, keys, order, sort) {};
					/* all ngrams of length order, starting
					 * at root */
    
    NgramProbArrayTrieIter(NgramProbArrayTrie &trie, const VocabIndex *start,
				VocabIndex *keys, unsigned order = 1,
				int (*sort)(VocabIndex, VocabIndex) = 0)
	 : myIter(*(trie.probs.insertTrie(start)), keys, order, sort) {};
					/* all ngrams of length order, rooted
					 * at node indexed by start */

    void init() { myIter.init(); };
    ZeroArray<Prob> *next()
	{ NgramProbArrayTrieNode *node = myIter.next();
	  return node ? &(node->value()) : 0; };

private:
    TrieIter2<VocabIndex,ZeroArray<Prob> > myIter;
};

#endif /* _NgramProbArrayTrie_h_ */

