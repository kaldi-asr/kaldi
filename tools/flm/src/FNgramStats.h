/*
 * FNgramStats.h --
 *	Factored (i.e., Cross-stream, multi-stream, morpholoigically factored,
 *	etc.) N-gram statistics class
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 * (based on code from Stolcke@SRI so copyright is preserved)
 *
 * Copyright (c) 1995-2009 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/FNgramStats.h,v 1.19 2012/10/29 17:24:59 mcintyre Exp $
 *
 */

#ifndef _FNgramStats_h_
#define _FNgramStats_h_

#include <stdio.h>

#include "TLSWrapper.h"
#include "LMStats.h"
#include "XCount.h"
#include "SubVocab.h"

#ifdef USE_XCOUNTS
typedef XCount FNgramCount;
#else
# ifdef USE_LONGLONG_COUNTS
typedef unsigned long long FNgramCount;
# else
typedef unsigned long FNgramCount;
# endif
#endif

#include "FactoredVocab.h"
#include "FNgramSpecs.h"
#include "wmatrix.h"
#include "NgramStats.h"		// for maxLineLength

#include "Trie.h"
#include "Array.h"

const unsigned int	maxFNgramOrder = 100;	/* Used in allocating various
						 * data structures.  For all
						 * practical purposes, this
						 * should be infinite. */

class FactoredVocab;				// forward declaration
template <class CountT> class FNgramSpecs;	// forward declaration
template <class CountT> class FNgramCountsIter;	// forward declaration

extern TLSW_DECL_ARRAY(VocabIndex, countSentenceWids, maxNumParentsPerChild+2);
extern TLSW_DECL(WordMatrix, countSentenceWordMatrix); 
extern TLSW_DECL(WidMatrix, countSentenceWidMatrix); 
extern TLSW_DECL_ARRAY(VocabString, readWords, maxNumParentsPerChild+1);
extern TLSW_DECL_ARRAY(VocabIndex, readWids, maxNumParentsPerChild+1);
extern TLSW_DECL_ARRAY(Boolean, readTagsFound, maxNumParentsPerChild+1);
extern TLSW_DECL_ARRAY(char, writeSpecBuffer, maxLineLength);

#ifndef FNgramNode
#define FNgramNode        Trie<VocabIndex,CountT>
#endif

template <class CountT>
class FNgramCounts: public LMStats
{
  friend class FNgramCountsIter<CountT>;

  FNgramSpecs<CountT>& fnSpecs;

  virtual unsigned countSentence(const VocabString *words)
          { return countSentence(words, (CountT)1); };
  virtual unsigned countSentence(const VocabIndex *words)
          { return countSentence(words, (CountT)1); };
  virtual unsigned countSentence(const VocabIndex *words, CountT factor)
          { fprintf(stderr,"Error: FNgramStats::countSentence(const VocabIndex *words, CountT factor) not implemented\n"); exit(-1); return false; }

public:
    // this vocab and order is for the entire word (not CS) vocab.
    // CS ngrams given in file
  FNgramCounts(FactoredVocab &vocab, FNgramSpecs<CountT>& fngs);

  virtual ~FNgramCounts() { /* FINISH ME */ }

  virtual unsigned int countSentence(const unsigned int start,
				     const unsigned int end,
				     WidMatrix& wm,
				     CountT factor);
  virtual unsigned int countSentence(const VocabString *words, 
				     const char *factor);
  virtual unsigned int countSentence(const VocabString *words, 
				     CountT factor);



  static unsigned int parseFNgram(char *line,
				  VocabString *words, unsigned int max,
				  CountT &count,
				  unsigned int& parSpec,
				  Boolean &ok);
  static unsigned int readFNgram(File &file,
				 VocabString *words, unsigned int max,
				 CountT &count,
				 unsigned int& parSpec,
				 Boolean &ok);
					/* read one ngram count from file */
  Boolean read(unsigned int specNum,File &file);
  Boolean read();
  // version that ignores the file argument.
  Boolean read(File &file) { return read(); }
  Boolean readMinCounts(File &file, unsigned order, Count *minCounts) 
  { fprintf(stderr,"Error: FNgramStats::readMinCounts(file,order,min) not implemented\n"); exit(-1); return false; }

  static unsigned int writeFNgram(File &file, const VocabString *words,
				  CountT count,
				  unsigned int parSpec);
  /* write ngram count to file */
  void writeSpec(File &file, 
		 const unsigned int specNum, 
		 const Boolean sorted = false);
  void write(const Boolean sorted = false);
  void write(File &file, 
	     const unsigned int order, 
	     const Boolean sorted = false)  
                            { write(sorted); }
  void write(File &file) { write(); }

  CountT sumCounts();
  CountT sumCounts(unsigned int specNum);
  CountT sumCounts(unsigned int specNum,unsigned int node);
         /* sum child counts on parent nodes, and store the result */
  
  void dump();			/* debugging dump */
  void memStats(MemStats &stats);	/* compute memory stats */


  // LM support
  void estimateDiscounts();
  void computeCardinalityFunctions();

  virtual unsigned int countFile(File &file, Boolean weighted = false);

  Boolean virtualBeginSentence;
  Boolean virtualEndSentence;

  Boolean addStartSentenceToken;
  Boolean addEndSentenceToken;

  FactoredVocab &vocab;		// specialized from Vocab to FactorVocab

protected:
  void incrementCounts(FNgramNode* counts,
		       const VocabIndex *words,
		       const unsigned order,
		       const unsigned minOrder = 1, 
		       const CountT factor = 1);

  void writeNode(FNgramNode *node, unsigned int parSpec, File &file, char *buffer, char *bptr,
		 unsigned int level, unsigned int order, Boolean sorted);
};

/*
 * Instantiate the count trie for integer types
 */

class FNgramStats: public FNgramCounts<FNgramCount>
{
public:
    FNgramStats(FactoredVocab &vocab, FNgramSpecs<FNgramCount>& fngs)
	: FNgramCounts<FNgramCount>(vocab, fngs) {};
    virtual ~FNgramStats() {};
};

#endif /* _FNgramStats_h_ */

