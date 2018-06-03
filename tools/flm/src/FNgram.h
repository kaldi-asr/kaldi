/*
 * FNgram.h --
 *	N-gram backoff language models
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 *       but based on some code from NgramLM.cc and Ngram.h (we therefore
 *       retain the Copyright)
 *
 * Copyright (c) 1995-2009 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/FNgram.h,v 1.12 2012/10/29 17:24:59 mcintyre Exp $
 *
 */

#ifndef _FNgram_h_
#define _FNgram_h_

#include <stdio.h>

#include "LM.h"
#include "FNgramStats.h"
#include "FNgramSpecs.h"
#include "SubVocab.h"

#ifdef USE_SARRAY

# define PROB_INDEX_T   SArray
# define PROB_ITER_T    SArrayIter
# include "SArray.h"

#else /* ! USE_SARRAY */

# define PROB_INDEX_T   LHash
# define PROB_ITER_T    LHashIter
# include "LHash.h"

#endif /* USE_SARRAY */

class FactoredVocab; // forw. ref.

template <class CountT> class FNgramSpecs;
typedef FNgramSpecs<FNgramCount> FNgramSpecsType;

class FNgram: public LM
{
  class BOsIter;
  friend class FNgram::BOsIter;
  class ProbsIter;
  friend class FNgram::ProbsIter;
  friend class FNgramSpecs<FNgramCount>;
  friend class FNgramSpecs<class T>;

  // general backoff-graph language models

public:
  struct ProbEntry {
    LogP prob;              // probability
    unsigned int cnt;       // also store the count here.
    ProbEntry() {
      cnt = ~0x0;
    };
  };

  struct BOnode {
    LogP bow;
    PROB_INDEX_T<VocabIndex,ProbEntry>	probs;	/* word probabilities */
  };
  
  typedef Trie<VocabIndex,BOnode> BOtrie;

  struct ParentSubset {
    Boolean active;
    // Here, order == number of parents in this context + child
    // I.e., if we are doing P(f|p1,p2,p3), order = 4..
    // in general, order >= 1.
    unsigned int order; 
    BOtrie contexts;
    ParentSubset() { active = false; }

    /*
     * Low-level access
     */

    LogP *findProb(VocabIndex word, const VocabIndex *context);
    LogP *findBOW(const VocabIndex *context);

    LogP *insertProb(VocabIndex word, const VocabIndex *context);
    LogP *insertProbAndCNT(VocabIndex word, const VocabIndex *context, 
			   const unsigned int cnt);
    LogP *insertBOW(const VocabIndex *context);

    void removeProb(VocabIndex word, const VocabIndex *context);
    void removeBOW(const VocabIndex *context);

    // versions that ignore a subset of the context, based on
    // the zeros in the bit vector, 'bits'.  The effective
    // length of the context is the number of bits set in 'bits'.
    // It is always assumed that num_words_in(context) >= (highest_order_bit_pos(bits)+1),
    // where highest_order_bit_pos(0x1)==0.
    BOnode* findTrieNodeSubCtx(const VocabIndex *context,
			       unsigned int bits);
    LogP *findBOWSubCtx(const VocabIndex *context,
			const unsigned bits = ~0x0);
    LogP *findProbSubCtx(VocabIndex word, const VocabIndex *context,
			 const unsigned bits = ~0x0);
    Boolean findBOWProbSubCtx(VocabIndex word, const VocabIndex *context,
			      LogP*& prob, LogP*& bow,
			      const unsigned bits = ~0x0);


    BOnode* insertTrieNodeSubCtx(const VocabIndex *context,
				 unsigned int bits,
				 Boolean &foundP);
    LogP *insertBOWSubCtx(const VocabIndex *context,
			  const unsigned bits = ~0x0);
    LogP *insertProbSubCtx(VocabIndex word, const VocabIndex *context,
			   const unsigned bits = ~0x0);

    void removeBOWSubCtx(const VocabIndex *context,
			 const unsigned bits = ~0x0);
    void removeProbSubCtx(VocabIndex word, const VocabIndex *context,
			  const unsigned bits = ~0x0);
    
  };

  struct FNgramLM {
    // each LM has:
    ParentSubset* parentSubsets;
    unsigned int parentSubsetsSize;
  };

  static void freeThread();

private:

  // the set of language models for this FLM.
  // TODO: we use only one count type for now, make this more gene
  FNgramLM* fNgrams;
  unsigned int fNgramsSize;

  FNgramSpecsType& fngs;

  /*
   * Iteration over all backoff contexts of a given node
   */
  class BOsIter
  {
  public:
    BOsIter(FNgram &lm, 
	    const unsigned int specNum,
	    const unsigned int parSpec,
	    VocabIndex *keys, 
	    unsigned order,
	    int (*sort)(VocabIndex, VocabIndex) = 0)
    {
      assert (order < lm.fNgrams[specNum].parentSubsets[parSpec].order);
      if (specNum >= lm.fNgramsSize || 
	  parSpec >= lm.fNgrams[specNum].parentSubsetsSize)
	myIter = NULL;
      else {
	myIter = new TrieIter2<VocabIndex,BOnode>(lm.fNgrams[specNum].parentSubsets[parSpec].contexts,
						  keys,
						  order,
						  sort);
      }
    }

    BOsIter(FNgram &lm, 
		  const unsigned int specNum,
		  const unsigned int parSpec,
		  VocabIndex *keys, 
		  int (*sort)(VocabIndex, VocabIndex) = 0)
    {
      if (specNum >= lm.fNgramsSize || parSpec >= lm.fNgrams[specNum].parentSubsetsSize)
	myIter = NULL;
      else {
	myIter = new TrieIter2<VocabIndex,BOnode>(lm.fNgrams[specNum].parentSubsets[parSpec].contexts,
						  keys,
						  lm.fNgrams[specNum].parentSubsets[parSpec].order-1,
						  sort);
      }
    }


    ~BOsIter() { delete myIter; }
    void init() { if (myIter) myIter->init(); };
    BOnode *next()
    { if (!myIter) return 0;
    Trie<VocabIndex,BOnode> *node = myIter->next();
    return node ? &(node->value()) : 0; }
  private:
    TrieIter2<VocabIndex,BOnode> *myIter;
  };


  /*
   * Iteration over all probs at a backoff node
   */
  class ProbsIter
  {
  public:
    ProbsIter(BOnode &bonode, 
		    int (*sort)(VocabIndex, VocabIndex) = 0)
      : myIter(bonode.probs, sort) {};

    void init() { myIter.init(); };
    LogP *next(VocabIndex &word) { 
      ProbEntry *pe = myIter.next(word);
      return pe ? &(pe->prob) : NULL;
    }
    LogP *next(VocabIndex &word, unsigned int* &cnt) { 
      ProbEntry *pe = myIter.next(word);
      if (!pe)
	return NULL;
      cnt = &(pe->cnt);
      return &(pe->prob);
    }

  private:
    PROB_ITER_T<VocabIndex,ProbEntry> myIter;
  };

public:
    FNgram(FactoredVocab &vocab, FNgramSpecsType& _fngs);
    virtual ~FNgram();

    /*
     * LM interface
     */
#ifdef USE_SHORT_VOCAB
    LogP wordProb(VocabIndex word, const VocabIndex *context) { return 0.0; };
    	// never used, needed to instantiate abstract virtual function
#endif

    LogP wordProb(VocabIndex word, 
		  const VocabIndex *context,
		  const unsigned nWrtwCip, 
		  const unsigned int specNum,
		  const unsigned int node);

    virtual Boolean read();
    virtual Boolean read(const unsigned int specNum, File &file);
    virtual Boolean read(File &file, Boolean limitVocab = false)
    	{ return false; };
    virtual void write();
    virtual void write(unsigned int specNum, File &file);
    virtual Boolean write(File &file)
        { return false; };		// TODO

    Boolean virtualBeginSentence;
    Boolean virtualEndSentence;
    Boolean noScoreSentenceBoundaryMarks;


    Boolean skipOOVs;		/* backward compatiability: return
				 * zero prob if <unk> is in context */
    Boolean trustTotals;	/* use lower-order counts for ngram totals */

    /*
     * Estimation
     */
    virtual void estimate();

    virtual void recomputeBOWs();
    virtual void storeBOcounts(unsigned int specNum, unsigned int node);

    /*
     * Statistics
     */
    virtual void memStats(MemStats &stats);

    virtual LogP wordProb(unsigned int i, const VocabIndex*v)
	{ assert(0); return 0; };	// TODO

    virtual Prob wordProbSum(const VocabIndex *context)
	{ return LM::wordProbSum(context); };


    LogP wordProb(unsigned int specNum,
		  WidMatrix& wm,
		  const unsigned int childPos, // position of child in wm
		  const unsigned int length);

    LogP wordProb(WidMatrix& wm,
		  const unsigned int childPos, // position of child in wm
		  const unsigned int length);

    LogP sentenceProb(const VocabIndex *sentence, TextStats &stats)
	{ assert(0); return 0; };	// TODO

    LogP sentenceProb(WordMatrix& wordMatrix,
		      unsigned int howmany,
		      const Boolean addWords = false,
		      LogP* parr = NULL);
    LogP sentenceProb(unsigned int specNum,
		      WidMatrix& wm,
		      const unsigned int start,
		      const unsigned int end,
		      TextStats& stats);


    virtual unsigned int pplFile(File &file, TextStats &stats,
    						const char *escapeString);
    virtual unsigned int pplPrint(ostream &stream,char *);
    virtual LogP wordProbBO(VocabIndex word, const VocabIndex *context,
			    const unsigned nWrtwCip, 
			    const unsigned specNum, 
			    const unsigned node);
    virtual LogP bgChildProbBO(VocabIndex word, const VocabIndex *context,
			       const unsigned nWrtwCip, 
			       const unsigned specNum, 
			       const unsigned node);


    virtual unsigned rescoreFile(File &file, double lmScale, double wtScale,
    				LM &oldLM, double oldLmScale, double oldWtScale,
				const char *escapeString = 0);
    Boolean combineLMScores;	// whether to output separate scores

protected:
    BOtrie contexts;				/* n-1 gram context trie */

    void clear();				/* remove all parameters */
    void clear(unsigned int specNum);


    /*
     * Helper functions
     */

    virtual unsigned vocabSize();
    virtual void estimate(unsigned int specNum);
    virtual void distributeProb(unsigned int specNum,
				unsigned int bg_node,
				Prob mass, 
				VocabIndex *context,
				const unsigned nWrtwCip);
    virtual Boolean computeBOW(BOnode *node, const VocabIndex *context, 
			       const unsigned nWrtwCip,
			       unsigned int specNum,
			       unsigned int bg_node,
			       Prob &numerator, Prob &denominator);
    virtual Boolean computeBOW1child(BOnode *node, const VocabIndex *context,
			       const unsigned nWrtwCip,
			       unsigned int specNum,
			       unsigned int bg_node,
			       Prob &numerator, Prob &denominator);
    virtual unsigned boNode(const VocabIndex word,
			    const VocabIndex *context,
			    const unsigned nWrtwCip, // node number r.w.t which context is packed. 
			    const unsigned int specNum,
			    const unsigned int node);
    virtual Boolean computeBOWs(unsigned int specNum, unsigned int node);
    unsigned int numFNgrams(const unsigned int specNum, const unsigned int node);

    // compute the sum over all words in current tag set for current context.
    Prob wordProbSum(const VocabIndex *context, 
		     // nWrtwCip: Node number With respect to (w.r.t.) 
		     // which Context is packed. 
		     const unsigned nWrtwCip, 
		     const unsigned int specNum,
		     const unsigned int node);

    Prob bgChildProbSum(const VocabIndex *context, 
		      // nWrtwCip: Node number With respect to (w.r.t.) 
		      // which Context is packed. 
		      const unsigned nWrtwCip, 
		      const unsigned int specNum,
		      const unsigned int node);

    Boolean wordProbSum();
};

#endif /* _FNgram_h_ */
