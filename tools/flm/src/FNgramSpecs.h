/*
 * FngramSpecs.h --
 * 
 *	structure to store the specifications of a factored LM
 *
 *  Jeff Bilmes  <bilmes@ee.washington.edu>
 * 
 * Copyright (c) 1995-2010 SRI International.  All Rights Reserved.

 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/FNgramSpecs.h,v 1.15 2012/10/20 00:22:26 mcintyre Exp $
 *
 */

#ifndef _FNgramSpecs_h_
#define _FNgramSpecs_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>

#include "LHash.cc"
#include "Trie.cc"
#include "Array.cc"

#include "LMStats.h"
#include "TLSWrapper.h"


/*
 * Debug levels used
 */
#define DEBUG_TAG_WARNINGS   8
#define DEBUG_VERY_VERBOSE   10
#define DEBUG_EVERY_SENTENCE_INFO   12
#define DEBUG_EXTREME        20
#define DEBUG_WARN_DUP_TAG   1
#define DEBUG_BG_PRINT 1
#define DEBUG_MISSING_FIRST_LAST_WORD   1

const unsigned int maxNumParentsPerChild = 32;
const unsigned int maxExtraWordsPerLine = 3;

// output files with these names are never written
const VocabString FNGRAM_DEV_NULL_FILE = "_";

#include "FactoredVocab.h"

#ifndef FNgramNode
#define FNgramNode	Trie<VocabIndex,CountT>
#endif

template <class CountT> class FNgramCounts;	// forward declaration
class FDiscount;
class FNgram;

class WidMatrix;
class WordMatrix;

enum BackoffNodeStrategy { 
  CountsNoNorm=0,           // use absolute counts
  CountsSumCountsNorm=1,    // norm by sum counts at level (== event MI maximization)
  CountsSumNumWordsNorm=2,  // norm by num words
  CountsProdCardinalityNorm=3,  // norma by prod. cardinality of gram
  CountsSumCardinalityNorm=4,   // norma by sum. cardinality of gram
  CountsSumLogCardinalityNorm=5, // norma by sum log cardinality of gram
  BogNodeProb=6 // backoff graph node probability
  // To add your favorite strategy here, add code 1) to boNode() in
  // FNgramLM.cc, 2) to backoffValueRSubCtxW(), and 3) to the node
  // option reading code in constructor for FNgramSpecs. Also make sure
  // to check the (great)grandchild score summing code in 
  // FNgram::boNode() to make sure that it still makes sense with your
  // strategy.
};

// combination methods
enum BackoffNodeCombine { 
  MaxBgChild=0,
  MinBgChild=1,
  SumBgChild=2,
  ProdBgChild=3,
  AvgBgChild=4,
  GmeanBgChild=5,
  WmeanBgChild=6
};

// useful defines.
const char FNGRAM_WORD_TAG = 'W';
const VocabString FNGRAM_WORD_TAG_STR = "W";
const unsigned FNGRAM_WORD_TAG_STR_LEN = 2;
// stuff will break if next const is changed to something other than 0
const unsigned FNGRAM_WORD_TAG_POS = 0;
const VocabString FNGRAM_WORD_TAG_SEP_STR = "-";
const char FNGRAM_WORD_TAG_SEP = '-';
const VocabString FNGRAM_WORD_TAG_NULL_SPEC_STR = Vocab_NULL;
const char FNGRAM_FACTOR_SEPARATOR = ':';
const VocabString FNGRAM_FACTOR_SEPARATOR_STR = ":";

class FactoredVocab;

template <class CountT>
class FNgramSpecs : public Debug {

  friend class FNgram;
  friend class FactoredVocab;

  FactoredVocab& fvocab;

  // the template&friend stuff does not seem to work for the gcc 2.95.3
  // so we make everything public for now.
public:

  // data structure for specifying which factored forms are to be
  // counted
  struct FNgramSpec {
    friend class FactoredVocab;
    FNgramSpec() { child = countFileName = NULL;
                   numParents = 0; }
    VocabString child;
    unsigned childPosition;
    unsigned numParents;
    unsigned numSubSets;
    Array<VocabString> parents;
    Array<unsigned> parentPositions;
    Array<int> parentOffsets; 

    unsigned int parseNodeString(char *str,Boolean& success);
    Boolean printNodeString(FILE* f,unsigned int node);

    // for each set of subsets of parents (all 2^N of them, where
    // N is the number of parents), (i.e., each node) we have the following
    // counts-like structure.
    struct ParentSubset {
      // 1. the "n-gram" for each parent subset has its own counts
      FNgramNode* counts;
      // Here, the "order" in the sence that P(C|P1,P2,...,PN) is
      // a distribution, then order = (N+1), so normal notion of N-gram order,
      // but for an arbitrary set of parents. order >= 1
      unsigned int order;
      // Number of BG children, but based on the constraints (i.e., for
      // certain constraints, some BG children might not be used and
      // have counts == NULL). This must be >= 1 to have a well-defined
      // model.
      unsigned int numBGChildren;

      // Discount object for this node.
      FDiscount *discount;

      // options for this node, pretty much the same thing you'll find
      // on the command line of ngram-counts.cc
      unsigned int backoffConstraint;
      BackoffNodeStrategy backoffStrategy;
      BackoffNodeCombine backoffCombine;
      unsigned int gtmin;
      unsigned int gtmax;
      char *gtFile;
      double cdiscount;
      Boolean ndiscount;
      Boolean wbdiscount;
      Boolean kndiscount;
      Boolean ukndiscount;
      char *knFile;
      Boolean knCountsModified;
      Boolean knCountsModifyAtEnd;
      unsigned int knCountParent;
      Boolean interpolate;
      char *writeFile;      // counts file for this node
      LogP2 *wmean;

      double prodCardinalities;
      double sumCardinalities;
      double sumLogCardinalities;

      Boolean requiresGenBackoff() {
	return !((numBGChildren <= 1)
		 ||
		 (backoffCombine == AvgBgChild)
		 || 
		 (backoffCombine == WmeanBgChild));
      }

      // initialize to some sensible values, gt will be default.
      ParentSubset() { 
	counts = NULL;
	order = 0;
	numBGChildren = 0;
	discount = NULL;
	backoffConstraint = ~0x0;
	backoffStrategy = CountsSumCountsNorm;
	backoffCombine = MaxBgChild;
	// TODO: make it possible for these defaults to be node dependent
	//       like in ngram-counts.cc
	gtmin = 1;
	gtmax = 7;
	gtFile = 0;
	cdiscount = -1.0;
	ndiscount = false;
	wbdiscount = false;
	kndiscount = false;
	ukndiscount = false;
	knFile = NULL;
	knCountsModified = false;
	knCountsModifyAtEnd = false;
	knCountParent = ~0x0;
	interpolate = false;
	writeFile = NULL;
	wmean = NULL;
      };

      /*
       * Individual word/ngram lookup and insertion,
       * caller should make sure counts != NULL.
       */
      CountT *findCount(const VocabIndex *words)
          { return counts->find(words); };
      CountT *findCount(const VocabIndex *words, VocabIndex word1)
          { FNgramNode *node = counts->findTrie(words);
	    return node ? node->find(word1) : 0; }
      CountT *findCountR(const VocabIndex *words, VocabIndex word1);

      // Fast routines which are used given a BG-parent node context
      // but we wish to find the count of a child, given by the bit vector.
      // These routines should probably live in Trie.{cc,h}.
      CountT *findCountSubCtx(const VocabIndex *words, unsigned int bits = ~0x0);
      CountT *findCountSubCtxW(const VocabIndex *words, VocabIndex word1,
			      unsigned int bits = ~0x0);
      // Same as above two, but indexes into words in the reverse order. This
      // is because LMs use wids in Tries in reverse order. These
      // routines give faster count trie lookups when wids are in reversed order
      // rather than having to reverse the wids list first.
      CountT *findCountRSubCtx(const VocabIndex *words, unsigned int bits = ~0x0);
      CountT *findCountRSubCtxW(const VocabIndex *words, VocabIndex word1,
				unsigned int bits = ~0x0);

      // return a value giving a "score" for the backoff context.
      // the caller might want to maximize that score to choose which context to use.
      double backoffValueRSubCtxW(VocabIndex word1,
				  const VocabIndex*words,
				  unsigned int nWrtwCip,
				  BackoffNodeStrategy parentsStrategy,
				  FNgram& fngram,
				  unsigned int specNum,
				  unsigned int node);


      // other routines for manipulating count objects.
      CountT *insertCount(const VocabIndex *words)
          { return counts->insert(words); };
      CountT *insertCount(const VocabIndex *words, VocabIndex word1)
          { FNgramNode *node = counts->insertTrie(words);
	    return node->insert(word1); };
      Boolean removeCount(const VocabIndex *words, CountT *removedData)
          { return counts->remove(words, removedData); };
      Boolean removeCount(const VocabIndex *words, VocabIndex word1, CountT *removedData)
          { FNgramNode *node = counts->findTrie(words);
	    return node ? node->remove(word1, removedData) : false; };

      CountT accumulateCounts(FNgramNode* counts);
      CountT accumulateCounts()
      { return accumulateCounts(counts); }

    };
    
    // iteration over counts entries in a parent subset object
    // Note that since count tries in this case only store counts for a given
    // level, we can get the "order" directly from the object itself. We
    // still give the option, however, to give a different order and iterate
    // over a context even though all of those counts will always be zero
    // (this is necessary in LM estimation)
    class PSIter {
    public:
      /* all ngrams of length order, starting
       * at root */
      PSIter(ParentSubset& pss, 
	     VocabIndex *keys,
	     unsigned order = ~0x0,
	     int (*sort)(VocabIndex,VocabIndex) = 0) {
	if (order == ~0x0U) order = pss.order;
	if (pss.counts == NULL) {
	  myIter = NULL;
	} else {
	  myIter = new TrieIter2<VocabIndex,CountT>(*(pss.counts), keys, order, sort);
	}
      }
      /* all count grams of length order for this parent subset, rooted
       * at node indexed by start */    
      PSIter(ParentSubset& pss,
	     const VocabIndex *start,
	     VocabIndex *keys, 
	     unsigned order = ~0x0,
	     int (*sort)(VocabIndex, VocabIndex) = 0) {
	if (order == ~0x0U) order = pss.order;
	if (pss.counts == NULL) {
	  myIter = NULL;
	} else {
	  myIter = new TrieIter2<VocabIndex,CountT>(*(pss.counts->insertTrie(start)), 
						    keys,order,sort);
	}
      }
      ~PSIter() { delete myIter; }
      void init() { if (myIter) myIter->init(); }
      CountT *next()
      { if (!myIter) return 0;
      FNgramNode *node = myIter->next();
      return node ? &(node->value()) : 0; }
    private:
      TrieIter2<VocabIndex,CountT> *myIter;
    };

    // iteration over counts object above of a given order.
    // probably not needed.
    class CountsIter {
    public:
      CountsIter(FNgramNode& counts, VocabIndex *keys,
		 unsigned order = 1,
		 int (*sort)(VocabIndex,VocabIndex) = 0)
	: myIter(counts, keys, order, sort) {};
					/* all ngrams of length order, starting
					 * at root */
    
      CountsIter(FNgramNode& counts, const VocabIndex *start,
		 VocabIndex *keys, unsigned order = 1,
		 int (*sort)(VocabIndex, VocabIndex) = 0)
	 : myIter(*(counts.insertTrie(start)), keys, order, sort) {};
					/* all ngrams of length order, rooted
					 * at node indexed by start */

      void init() { myIter.init(); };
      CountT *next()
      { FNgramNode *node = myIter.next();
        return node ? &(node->value()) : 0; };
    private:
      TrieIter2<VocabIndex,CountT> myIter;
      
    };


    // the set of parent subsets, the array index determines which 
    // parents are used.
    Array<ParentSubset> parentSubsets;

    // Iters (defined below) over parent subsets by level, parents,
    // children, ancestors, and descendants in THE BACKOFF GRAPH
    // (BG). Note here that by "parents", "children", etc., we do
    // *NOT* mean the same parents and children that are specified in
    // the FLM specification file (and what the member variables
    // 'numParents', 'child', etc. refer to above).  Rather, here we
    // mean parents and children of a node in the backoff graph.  This
    // overloading of terminology is confusing indeed. 
    // 
    // Perhaps we should use "Eltern", "Kinder", "Nachfahren", and
    // "Vorfahren" for this and stick with English for the above??? As
    // much as I (JB, obviously not AS) need to practice German, how
    // about simply "BGParents" for "backoff graph parents", etc. This
    // was suggested by Katrin (who also speaks German).
    // 
    // Anyway, here is a simple backoff graph, for a multi-backoff
    // trigram. The bit vector shown in the node of the graph gives
    // the set of parents that are used in that node.
    // 
    // Level 2     11		.
    //            /  \		.
    // Level 1   10   01	.
    //            \  /		.
    // Level 0     00		.
    // 
    // Note that, by convention, level numbers *increase* we *add*
    // bits going down from 00 to 11. The iters and code below use
    // this convention.
    // 
    // Here is a BG for a standard trigram p(w_t|w_{t-1},w_{t-2})
    //
    // Level 2     11		.
    //               \ 		.
    // Level 1        01	.
    //               /		.
    // Level 0     00		.
    //
    // This means that the first parent (w_{t-1}) correspond to the
    // low-order bit 0x01, and the second parent (w_{t_2}) corresponds
    // to the higher order bit 0x10. In general, when we specify in
    // the factored language model (FLM) file a distribution of the
    // form f(c|p0,p1,...,pn), where c = child, and pi = i'th parent,
    // then p0 corresponds to bit 0, p1 to bit 1, etc.
    //
    // Here is a BG for a "reverse-context" trigram p(w_t|w_{t-1},w_{t-2})
    //
    // Level 2     11		.
    //            /		.
    // Level 1   10		.
    //            \ 		.
    // Level 0     00		.
    //
    // This means that p(W_t=w_t|W_{t-1}=w_{t-1},W_{t-2}=w_{t-2})
    // first backs off to p(W_t=w_t|W_{t-2}=w_{t-2}) and then to
    // p(W_t=w_t). This is different from p(W_t=w_t|W_{t-1}=w_{t-2})
    // which would change the history but uses the same distribution
    // (as similar to SkipNgram.cc)
    // 
    //
    // Here is another more interesting case for a four-parent CPT
    // f(c|p0,p1,p2,p3) (i.e., a 5-gram if they were words). The graph
    // here only shows the nodes not the edges for clarity. We have an
    // edge from a node in level_i to a node in level_{i+1} if all
    // bits in the node in level_{i+1} are contained also in the
    // parent in level_i.
    //
    //
    // L4                                    1111(0xF)
    // 
    // L3                 0111(0x7)    1011(0xB)    1101(0xD)    11110(0xE)
    // 
    // L2    0011(0x3)    0101(0x5)    0110(0x6)    1001(0x9)    1010(0x10)  1100(0xC)
    // 
    // L1                 1000(0x8)    0100(0x4)    0010(0x2)    0001(0x1)
    // 
    // L0                                     0000(0x0)
    // 
    // 
    // Note that since we're using unsigned ints for bitvectors, this means we can not
    // have more than 32 parents (i.e., 33-grams) for now, assuming a 32-bit machine.
    // TODO: possibly use an unlimited size bitvector class, but perhaps this
    //       will be not necessary since by the time we can train 33-grams, we'll
    //       all be using 64-bit machines.
    // 
    // The following iters make it easy to navigate around in the above backoff graphs (BGs).

    // iterate accross a given level
    class LevelIter {
      const unsigned int numParents;
      const unsigned int numNodes;
      unsigned int state;

      const unsigned int level;
    public:
      LevelIter(const unsigned int _numParents,const unsigned int _level)
	: numParents(_numParents),numNodes(1<<_numParents),level(_level) 
      { init(); }
      void init() { state = 0; }
      Boolean next(unsigned int&node);
    };

    // iterate over parents of a BG node
    class BGParentIter {
      const unsigned int numParents;
      const unsigned int numNodes;
      unsigned int state;

      const unsigned int homeNode;
      const unsigned int numBitsSetOfHomeNode;
    public:
      BGParentIter(const unsigned int _numParents,const unsigned int _homeNode);
      void init() { state = (homeNode+1); }
      Boolean next(unsigned int&node);
    };

    // iterate over (great) grandparents of a BG node
    class BGGrandParentIter {
      const unsigned int numParents;
      const unsigned int numNodes;
      unsigned int state;

      const unsigned int homeNode;
      const unsigned int numBitsSetOfHomeNode;
      const unsigned int great; // grandparent(great=0), greatgrandparent(great=1), etc.
    public:
      BGGrandParentIter(const unsigned int _numParents,const unsigned int _homeNode,
			const unsigned int _great=0);
      void init() { state = (homeNode+((1U<<(great+1))-1)); }
      Boolean next(unsigned int&node);
    };
    
    // iterate over all ancestors of a BG node
    class BGAncestorIter {
      const unsigned int numParents;
      const unsigned int numNodes;
      unsigned int state;

      const unsigned int homeNode;
      const unsigned int numBitsSetOfHomeNode;
    public:
      BGAncestorIter(const unsigned int _numParents,const unsigned int _homeNode);
      void init() { state = (homeNode+1); }
      Boolean next(unsigned int&node);
    };

    // Child Iter, no constraints (i.e., all children)
    class BGChildIter {
      const unsigned int numParents;
      const unsigned int numNodes;
      int state;

      const unsigned int homeNode;
      const unsigned int numBitsSetOfHomeNode;
    public:
      BGChildIter(const unsigned int _numParents,const unsigned int _homeNode);
      void init() { state = ((int)homeNode-1); }
      Boolean next(unsigned int&node);
    };

    // Child Iter with BO constraints
    class BGChildIterCnstr {
      const unsigned int numParents;
      const unsigned int numNodes;
      int state;

      const unsigned int homeNode;
      const unsigned int bo_constraints;
      const unsigned int numBitsSetOfHomeNode;
    public:
      BGChildIterCnstr(const unsigned int _numParents,
		       const unsigned int _homeNode,
		       const unsigned int _bo_constraints);
      void init() { state = ((int)homeNode-1); }
      Boolean next(unsigned int&node);
    };

    // etc. 
    class BGGrandChildIter {
      const unsigned int numParents;
      const unsigned int numNodes;
      int state;

      const unsigned int homeNode;
      const unsigned int numBitsSetOfHomeNode;
      const unsigned int great;
    public:
      BGGrandChildIter(const unsigned int _numParents,const unsigned int _homeNode, 
		       const unsigned int _great=0);
      void init() { state = ((int)homeNode-((1<<(great+1))-1)); }
      Boolean next(unsigned int&node);
    };


    class BGDescendantIter {
      const unsigned int numParents;
      const unsigned int numNodes;
      int state;

      const unsigned int homeNode;
      const unsigned int numBitsSetOfHomeNode;
    public:
      BGDescendantIter(const unsigned int _numParents,const unsigned int _homeNode);
      void init() { state = ((int)homeNode-1); }
      Boolean next(unsigned int&node);
    };


    // unsigned int numParents(const unsigned int _numParents,const unsigned int _homeNode);
    // unsigned int numChildren(const unsigned int _numParents,const unsigned int _homeNode);

    /* training data stats for this LM */
    TextStats stats;

    char *countFileName;
    char *lmFileName;
    char *initLMFile; // intial LM file

  };

  // Collection of conditional distributions that will be
  // computed.
  Array <FNgramSpec> fnSpecArray;

  // Hash that maps from tag name to array position where the tag
  // value will be stored when parsing a tagged text file. This
  // also gives the number of tags we're currently working with.
  LHash<VocabString,unsigned> tagPosition;

public:
  
  void printFInfo();

  // constructor
  FNgramSpecs(File& f,
	      FactoredVocab& fv,
	      unsigned debuglevel = 0);

  // TODO: finish destructor
  virtual ~FNgramSpecs() {}

  unsigned int loadWordFactors(const VocabString *words,
			       WordMatrix& wm,
			       unsigned int max);

  void estimateDiscounts(FactoredVocab& vocab);

  void computeCardinalityFunctions(FactoredVocab& vocab);

  // Boolean readCounts(); -- Not yet implemented

  static inline unsigned int numBitsSet(unsigned u) {
    unsigned count=0;
    while (u) { count += (u&0x1); u >>= 1; }
    return count;
  }

  static VocabString getTag(VocabString a);
  static VocabString wordTag();
private:
#define FNgramSpecs_BUF_SZ 1024
  static TLSW_DECL_ARRAY(char, FNgramSpecsBuff, FNgramSpecs_BUF_SZ);
};

#endif /* _FNgramSpecs_h_ */

