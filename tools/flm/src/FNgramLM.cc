/*
 * FNgramLM.cc --
 *	factored N-gram general graph backoff language models
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 *       but based on some code from NgramLM.cc and Ngram.h
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2012 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/FNgramLM.cc,v 1.27 2014-04-23 00:25:52 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <new.h>
# include <iostream.h>
#else
# include <new>
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "FNgram.h"
#include "FactoredVocab.h"
#include "FDiscount.h"
#include "File.h"
#include "Array.cc"
#include "Trie.cc"
#include "hexdec.h"
#include "TLSWrapper.h"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_TRIE(VocabIndex,FNgram::BOnode);
INSTANTIATE_ARRAY(FNgram::FNgramLM); 
INSTANTIATE_ARRAY(FNgram::ParentSubset); 

#ifdef USE_SARRAY
#include "SArray.cc"
INSTANTIATE_SARRAY(VocabIndex,FNgram::ProbEntry); 
#else
#include "LHash.cc"
INSTANTIATE_LHASH(VocabIndex,FNgram::ProbEntry); 
#endif
#endif /* INSTANTIATE_TEMPLATES */

/*
 * Debug levels used
 */
#define DEBUG_ESTIMATE_WARNINGS	1
#define DEBUG_FIXUP_WARNINGS 3
#define DEBUG_PRINT_GTPARAMS 2
#define DEBUG_READ_STATS 1
#define DEBUG_WRITE_STATS 1
#define DEBUG_NGRAM_HITS 2
#define DEBUG_ESTIMATES 4
#define DEBUG_ESTIMATE_LM 4
#define DEBUG_BOWS 4
#define DEBUG_EXTREME 20

/* these are the same as in LM.cc */
#define DEBUG_PRINT_SENT_PROBS		1
#define DEBUG_PRINT_WORD_PROBS		2
#define DEBUG_PRINT_PROB_SUMS		3

const LogP LogP_PseudoZero = -99.0;	/* non-inf value used for log 0 */

/*
 * Low level methods to access context (BOW) nodes and probs
 */

void
FNgram::memStats(MemStats &stats)
{
  // TODO: finish this function.
}

FNgram::FNgram(FactoredVocab &vocab, FNgramSpecsType& _fngs)
  : LM(vocab), fngs(_fngs),
    virtualBeginSentence(true), virtualEndSentence(true), noScoreSentenceBoundaryMarks(false),
    skipOOVs(false), trustTotals(false), combineLMScores(true)
{
  // we could pre-allocate the fngrams arrays here
  // but Array objects do not export alloc.
  fNgrams = new FNgramLM[fngs.fnSpecArray.size()];
  fNgramsSize = fngs.fnSpecArray.size();
  for (unsigned specNum=0;specNum<fngs.fnSpecArray.size();specNum++) {  
    fNgrams[specNum].parentSubsets =
      new ParentSubset[fngs.fnSpecArray[specNum].numSubSets];
    fNgrams[specNum].parentSubsetsSize = fngs.fnSpecArray[specNum].numSubSets;
    for (unsigned node=0;node<fngs.fnSpecArray[specNum].numSubSets;node++) {
      fNgrams[specNum].parentSubsets[node].active =
	(fngs.fnSpecArray[specNum].parentSubsets[node].counts != NULL);
      // make copy for convenient access
      fNgrams[specNum].parentSubsets[node].order =
	fngs.fnSpecArray[specNum].parentSubsets[node].order;
    }
  }
}

FNgram::~FNgram()
{
  clear();
  for (unsigned specNum=0;specNum<fNgramsSize;specNum++) {  
    delete [] fNgrams[specNum].parentSubsets;
  }
  delete [] fNgrams;
}


/*
 * Locate a BOW entry in the n-gram trie
 */
LogP *
FNgram::ParentSubset::findBOW(const VocabIndex *context)
{
    BOnode *bonode = contexts.find(context);
    if (bonode) {
	return &(bonode->bow);
    } else {
	return 0;
    }
}

/*
 * Locate a prob entry in the n-gram trie.
 */
LogP *
FNgram::ParentSubset::findProb(VocabIndex word, const VocabIndex *context)
{
    BOnode *bonode = contexts.find(context);
    if (bonode) {
	return &(bonode->probs.find(word)->prob);
    } else {
	return 0;
    }
}



/*
 * Locate or create a BOW entry in the n-gram trie
 */
LogP *
FNgram::ParentSubset::insertBOW(const VocabIndex *context)
{
    Boolean found;
    BOnode *bonode = contexts.insert(context, found);

    if (!found) {
	/*
	 * initialize the index in the BOnode
	 */
	new (&bonode->probs) PROB_INDEX_T<VocabIndex,ProbEntry>(0);
    }
    return &(bonode->bow);
}

/*
 * Locate or create a prob entry in the n-gram trie
 */
LogP *
FNgram::ParentSubset::insertProb(VocabIndex word, const VocabIndex *context)
{
    Boolean found;
    BOnode *bonode = contexts.insert(context, found);

    // fprintf(stderr,"inserting word %d context starting with %d\n",word,*context);
    if (!found) {
      // fprintf(stderr,"not found\n");
	/*
	 * initialize the index in the BOnode
	 */
	new (&bonode->probs) PROB_INDEX_T<VocabIndex,ProbEntry>(0);
    }
    ProbEntry* res = bonode->probs.insert(word);
    // fprintf(stderr,"inserted into probs, got back 0x%X, size now %d\n",
    // res,bonode->probs.numEntries());
    return res ? &(res->prob) : NULL;
    // return bonode->probs.insert(word);
}


/*
 * Locate or create a prob entry in the n-gram trie
 */
LogP *
FNgram::ParentSubset::insertProbAndCNT(VocabIndex word, const VocabIndex *context,
				       const unsigned int cnt)
{
    Boolean found;
    BOnode *bonode = contexts.insert(context, found);

    if (!found) {
	/*
	 * initialize the index in the BOnode
	 */
	new (&bonode->probs) PROB_INDEX_T<VocabIndex,ProbEntry>(0);
    }
    ProbEntry* res = bonode->probs.insert(word);
    if (res) {
      res->cnt = cnt;
      return &(res->prob);
    }
    return NULL;
}


/*
 * Remove a BOW node (context) from the n-gram trie
 */
void
FNgram::ParentSubset::removeBOW(const VocabIndex *context)
{
    contexts.removeTrie(context);
}

/*
 * Remove a prob entry from the n-gram trie
 */
void
FNgram::ParentSubset::removeProb(VocabIndex word, const VocabIndex *context)
{
    BOnode *bonode = contexts.find(context);

    if (bonode) {
	bonode->probs.remove(word);
    }
}


FNgram::BOnode*
FNgram::ParentSubset::findTrieNodeSubCtx(const VocabIndex *context,
					 unsigned int bits)
{
  // layout of arguments
  // variables: p1 p2 p3 p4 (i.e., parent number)
  // bits:      b1 b2 b3 b4
  // context[i]: 0  1  2  3

  // From model p(c|p1,p2,p3,p4)
  // LM tries have p1 at the root tree level, then p2, p3, and p4 at
  // the tree leaf level. Therefore, this routine indexes lm tries
  // in ascending context array order.

  const unsigned wlen = Vocab::length(context);
  assert (FNgramSpecsType::numBitsSet(bits) <= wlen);

  BOtrie* boTrie = &contexts;
  VocabIndex word[2];
  word[1] = Vocab_None;
  for (unsigned i = 0; i < wlen && bits; i++) {
    if (bits & 0x1) {
      word[0] = context[i];
      if ((boTrie = boTrie->findTrie(word)) == NULL)
	return NULL;
    }
    bits >>= 1;
  }
  return boTrie ? &(boTrie->value()) : NULL;
}


/*
 * Locate a prob entry in the n-gram trie.
 */
LogP *
FNgram::ParentSubset::findProbSubCtx(VocabIndex word1,
				     const VocabIndex *context,
				     unsigned int bits)
{
  BOnode* bonode = findTrieNodeSubCtx(context,bits);
  if (!bonode)
    return NULL;
  ProbEntry* pe = bonode->probs.find(word1);
  return pe ? &(pe->prob) : NULL;  
}

/*
 * Locate a bow in the n-gram trie.
 */
LogP *
FNgram::ParentSubset::findBOWSubCtx(const VocabIndex *context,
				    unsigned int bits)
{
  BOnode* bonode = findTrieNodeSubCtx(context,bits);
  return bonode ? &(bonode->bow) : NULL;  
}



/*
 * Locate both prob entry and bow in the n-gram trie.
 */
Boolean
FNgram::ParentSubset::findBOWProbSubCtx(VocabIndex word1,
					const VocabIndex *context,
					LogP*& prob, LogP*& bow,
					unsigned int bits)
{
  BOnode* bonode = findTrieNodeSubCtx(context,bits);
  if (bonode) {
    prob = &(bonode->probs.find(word1)->prob);
    bow = &(bonode->bow);
    return true;
  } else
    return false;
}




FNgram::BOnode*
FNgram::ParentSubset::insertTrieNodeSubCtx(const VocabIndex *context,
					   unsigned int bits,
					   Boolean& foundP)
{
  // same as findTrieNodeSubCtx except we do inserts rather than finds.
  const unsigned wlen = Vocab::length(context);
  assert (FNgramSpecsType::numBitsSet(bits) <= wlen);

  BOtrie* boTrie = &contexts;
  VocabIndex word[2];
  word[1] = Vocab_None;
  for (unsigned i = 0; i < wlen && bits; i++) {
    if (bits & 0x1) {
      word[0] = context[i];
      if ((boTrie = boTrie->insertTrie(word,foundP)) == NULL)
	return NULL;
    }
    bits >>= 1;
  }
  foundP = true;
  return boTrie ? &(boTrie->value()) : NULL;
}



/*
 * Locate or create a BOW entry in the n-gram trie
 */
LogP *
FNgram::ParentSubset::insertBOWSubCtx(const VocabIndex *context,
				      unsigned int bits)
{
    Boolean found;
    BOnode *bonode = insertTrieNodeSubCtx(context, bits, found);

    if (!found) {
	/*
	 * initialize the index in the BOnode
	 */
	new (&bonode->probs) PROB_INDEX_T<VocabIndex,ProbEntry>(0);
    }
    return &(bonode->bow);
}



/*
 * Locate or create a BOW entry in the n-gram trie
 */
LogP *
FNgram::ParentSubset::insertProbSubCtx(VocabIndex word,
				       const VocabIndex *context,
				       unsigned int bits)
{
    Boolean found;
    BOnode *bonode = insertTrieNodeSubCtx(context, bits, found);

    if (!found) {
	/*
	 * initialize the index in the BOnode
	 */
	new (&bonode->probs) PROB_INDEX_T<VocabIndex,ProbEntry>(0);
    }
    ProbEntry* pe = bonode->probs.insert(word);
    return pe ? &(pe->prob) : NULL;
}



void
FNgram::ParentSubset::removeBOWSubCtx(const VocabIndex *context,
				      unsigned int bits)
{
  // same as findTrieNodeSubCtx except that we pack context
  // into a local words array rather than do the trie search
  // explicitly. This is because this routine
  // it is probably not called very often and
  // doesn't need to be as fast as the others.


  const unsigned wlen = Vocab::length(context);
  assert (FNgramSpecsType::numBitsSet(bits) <= wlen);
  VocabIndex words[maxNumParentsPerChild+2];

  unsigned j = 0;
  for (unsigned i = 0; i < wlen && bits; i++) {
    if (bits & 0x1) {
      words[j] = context[i];
    }
    bits >>= 1;
  }
  words[j] = Vocab_None;
  removeBOW(words);
}



void
FNgram::ParentSubset::removeProbSubCtx(VocabIndex word1,
				       const VocabIndex *context,
				       unsigned int bits)
{
  const unsigned wlen = Vocab::length(context);
  assert (FNgramSpecsType::numBitsSet(bits) <= wlen);
  VocabIndex words[maxNumParentsPerChild+2];

  unsigned j = 0;
  for (unsigned i = 0; i < wlen && bits; i++) {
    if (bits & 0x1) {
      words[j] = context[i];
    }
    bits >>= 1;
  }
  words[j] = Vocab_None;
  removeProb(word1,words);
}







/*
 * Remove all probabilities and contexts from n-gram trie
 */
void
FNgram::clear(unsigned int specNum)
{
    VocabIndex context[maxNumParentsPerChild+2];

    BOnode *node;

    if (specNum >= fNgramsSize)
      return;

    /*
     * Remove a ngram probabilities
     */
    for (unsigned i = 0; i < fNgrams[specNum].parentSubsetsSize; i++) {
      BOnode *node;
      BOsIter iter(*this,specNum,i,context);
      while ((node = iter.next())) {
	node->probs.clear(0);
      }
    }
}

void
FNgram::clear()
{
  for (unsigned i=0;i<fNgramsSize;i++)
    clear(i);
}

// TODO: but this and other bit routines in one
// separate pair of .cc .h files.
unsigned int 
bitGather(register unsigned int mask,register unsigned int bitv)
{
  // gather (or pack) together the bits in bitv that
  // correspond to the 1 bits in mask, and place them
  // at the low end of the word 'res'
  register unsigned res = 0;
  register unsigned res_pos = 0;

  while (mask && bitv) {
    if (mask & 0x1) {
      if (bitv & 0x1) {
	res |= (1<<res_pos);
      }
      res_pos++;
    }
    mask >>= 1;
    bitv >>= 1;
  }
  return res;
}

/*
 * For a given node in the BG, compute the child node that
 * we should backoff to. The 'context' argument is assumed
 * to be in reverse order with respect to the cound tries.
 *
 * Note: This routine contains the backoff algorithm.
 * TODO: at some point make a version of this w/o a nWrtwCip argument
 *       for fast ppl and rescoring.
 */
unsigned
FNgram::boNode(const VocabIndex word,
	       const VocabIndex *context,
	       const unsigned nWrtwCip, // node number w.r.t which context is packed. 
	       const unsigned int specNum,
	       const unsigned int node)
{
  assert (node != 0); // no reason to call this routine if node == 0.

  // all nodes with one parent back off to the unigram.
  const unsigned int nbitsSet = FNgramSpecsType::numBitsSet(node);
  if (nbitsSet == 1)
    return 0;

  const unsigned int numParents = fngs.fnSpecArray[specNum].numParents;
  FNgramSpecsType::FNgramSpec::BGChildIterCnstr
    citer(numParents,node,fngs.fnSpecArray[specNum].parentSubsets[node].backoffConstraint);

  unsigned bg_child; // backoff-graph child
  unsigned chosen_bg_child = ~0x0U;
  unsigned a_bg_child = ~0x0U; // arbitrary child, if everything fails, we use this.
  const Boolean domax =  // do min if domax == false.
    (fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine == MaxBgChild);
  const double initScore = domax ? -1e220 : 1e220;
  double bestScore = initScore;
  // find child node with the largest counts
  while (citer.next(bg_child)) {
 
    // We max/min over the BO value. The BO value is determined by the
    // BO algorithm (i.e., counts, normalized counts, etc.)  in the
    // specs object for this node.

    double score =
      fngs.fnSpecArray[specNum].parentSubsets[bg_child].
      backoffValueRSubCtxW(word,context,nWrtwCip,
			   fngs.fnSpecArray[specNum].
			   parentSubsets[node].backoffStrategy,
			   *this,
			   specNum,
			   bg_child);
			   
    if (score == -1e200) // TODO: change this to NaN or Inf, and a #define (also see below)
      continue; // continue presumably because of a NULL counts object
    if (a_bg_child == ~0x0U)
      a_bg_child = bg_child;


    if ((domax && score > bestScore) || (!domax && score < bestScore)) {
      chosen_bg_child = bg_child;
      bestScore = score;
    }
  }

  // make sure that we have at least one valid child
  assert (a_bg_child != ~0x0U);

  // if we only have one child, or if we have two children and have
  // not found a best child, just return an arbitrary child node.
  if ((fngs.fnSpecArray[specNum].parentSubsets[node].numBGChildren == 1)
      || (chosen_bg_child == ~0x0U && nbitsSet == 2))
    return a_bg_child;


  if (chosen_bg_child == ~0x0U) {
    // Then we did not found any BG-child with a score for this
    // context. We back off to the child that has the best
    // combined schore of its children. We keep
    // doing this, as long as possible. 

    unsigned int great;
    for (great=0; (great+2)< nbitsSet ; great++) {
      
      citer.init();
      while (citer.next(bg_child)) {

	double score = initScore;
	// get score for child
	if (great == 0) {
	  // children of child iter
	  FNgramSpecsType::FNgramSpec::BGChildIterCnstr
	    gciter(numParents,bg_child,fngs.fnSpecArray[specNum].parentSubsets[bg_child].backoffConstraint);
	  unsigned bg_grandchild;
	  while (gciter.next(bg_grandchild)) {
	    double tmp=fngs.fnSpecArray[specNum].parentSubsets[bg_grandchild].
	      backoffValueRSubCtxW(word,context,nWrtwCip,
				   fngs.fnSpecArray[specNum].
				   parentSubsets[node].backoffStrategy,
				   *this,
				   specNum,
				   bg_grandchild);
	    // compute local max min of offspring
	    if ((domax && tmp > score) || (!domax && tmp < score)) {
	      score = tmp;
	    }
	  }
	} else {
	  fprintf(stderr,"WARNING: might produce unnormalized LM. Check with -debug 3\n");
	  // grandchildren of child iter
	  FNgramSpecsType::FNgramSpec::BGGrandChildIter
	    descendant_iter(numParents,bg_child,great-1);
	  unsigned bg_grandchild;
	  while (descendant_iter.next(bg_grandchild)) {
	    double tmp=fngs.fnSpecArray[specNum].parentSubsets[bg_grandchild].
	      backoffValueRSubCtxW(word,context,nWrtwCip,
				   fngs.fnSpecArray[specNum].
				   parentSubsets[node].backoffStrategy,
				   *this,
				   specNum,
				   bg_grandchild);
	    if ((domax && tmp > score) || (!domax && tmp < score)) {
	      score = tmp;
	    }
	  }
	}

	if (score == -1e200)  // TODO: change this to NaN or Inf, and a #define (also see above)
	  continue; // presumably because of a NULL counts objects
	if ((domax && score > bestScore) || (!domax && score < bestScore)) {
	  chosen_bg_child = bg_child;
	  bestScore = score;
	}
      }
      if (chosen_bg_child != ~0x0U)
	break;

    }

    // still not found, chose an arbitrary child
    if (chosen_bg_child == ~0x0U)
      chosen_bg_child = a_bg_child;
  }
  return chosen_bg_child;
}



//
// General algorithm for the "BG child probability backoff"
LogP
FNgram::bgChildProbBO(VocabIndex word, 
		      const VocabIndex *context, 
		      const unsigned nWrtwCip, 
		      const unsigned int specNum,
		      const unsigned int node)
{
  // general graph backoff algorithm for multiple BG children.

  // should never be called at node 0 since there is nothing to
  // backoff to.
  assert ( node != 0 );

  // special case numBGChildren == 1 for speed.
  if (fngs.fnSpecArray[specNum].parentSubsets[node].numBGChildren == 1) {
    unsigned int bg_child;
    FNgramSpecsType::FNgramSpec::BGChildIterCnstr
      citer(fngs.fnSpecArray[specNum].numParents,
	    node,
	    fngs.fnSpecArray[specNum].parentSubsets[node].backoffConstraint);
    while (citer.next(bg_child)) {
      if (fngs.fnSpecArray[specNum].parentSubsets[bg_child].counts != NULL) {
	// return immediately since we've found the bg child
	return wordProbBO(word,context,nWrtwCip,specNum,bg_child);
      }
    }
  }

  // still here? Do the general case.
  LogP bo_prob;
  if (fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine
      == ProdBgChild
      ||
      fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine
      == GmeanBgChild) {
    bo_prob = LogP_One;
    unsigned int bg_child;
    FNgramSpecsType::FNgramSpec::BGChildIterCnstr
      citer(fngs.fnSpecArray[specNum].numParents,
	    node,
	    fngs.fnSpecArray[specNum].parentSubsets[node].backoffConstraint);
    while (citer.next(bg_child)) {
      if (fngs.fnSpecArray[specNum].parentSubsets[bg_child].counts != NULL) {
	// multiply the probs (add the log probs)
	bo_prob += wordProbBO(word,context,nWrtwCip,specNum,bg_child);
      }
    }
    if (fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine
	== GmeanBgChild)
      bo_prob /= fngs.fnSpecArray[specNum].
	parentSubsets[node].numBGChildren;
  } else if (fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine
	     == SumBgChild ||
	     fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine
	     == AvgBgChild) {
    bo_prob = LogP_Zero;
    unsigned int bg_child;
    FNgramSpecsType::FNgramSpec::BGChildIterCnstr
      citer(fngs.fnSpecArray[specNum].numParents,
	    node,
	    fngs.fnSpecArray[specNum].parentSubsets[node].backoffConstraint);
    while (citer.next(bg_child)) {
      if (fngs.fnSpecArray[specNum].parentSubsets[bg_child].counts != NULL) {
	// add the probs
	bo_prob = 
	  AddLogP(bo_prob,wordProbBO(word,context,nWrtwCip,specNum,bg_child));
      }
    }
    if (fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine
	== AvgBgChild)
      bo_prob -= log10((double)fngs.fnSpecArray[specNum].parentSubsets[node].numBGChildren);
  } else if (fngs.fnSpecArray[specNum].parentSubsets[node].backoffCombine
	     == WmeanBgChild) { 
    bo_prob = LogP_Zero;
    unsigned int bg_child;
    FNgramSpecsType::FNgramSpec::BGChildIterCnstr
      citer(fngs.fnSpecArray[specNum].numParents,
	    node,
	    fngs.fnSpecArray[specNum].parentSubsets[node].backoffConstraint);
    unsigned cpos = 0;
    while (citer.next(bg_child)) {
      if (fngs.fnSpecArray[specNum].parentSubsets[bg_child].counts != NULL) {
	// add the probs by weights
	bo_prob = 
	  AddLogP(bo_prob, 
		  // log10(0.5) +
		  fngs.fnSpecArray[specNum].parentSubsets[node].wmean[cpos] +
		  wordProbBO(word,context,nWrtwCip,specNum,bg_child));
	cpos++;
      }
    }
  } else {
    // choose only one backoff node
    unsigned chosen_descendant = boNode(word,context,nWrtwCip,specNum,node);
    bo_prob = wordProbBO(word,context,nWrtwCip,specNum,chosen_descendant);
  }
  return bo_prob;
}



/*
 * This method implements the backoff algorithm in a straightforward,
 * recursive manner.
 * Note, this function likes context in reverse order.
 * I.e., if word = child = word_t, then
 *    context[0] = parent1 word_{t-1}
 *    context[1] = parent2 word_{t-2}
 *    context[2] = parent3 word_{t-3}
 * In the model p(child|parent1,parent2,parent3) = p(w_t|w_t-1,w_t-2,w_t-3)
 */


// TODO: make a version of this routine that assumes context is
// packed w.r.t. 0x1111 node, used for ppl and rescoring.
LogP
FNgram::wordProbBO(VocabIndex word, 
		   const VocabIndex *context, 
		   // nWrtwCip: Node number With respect to (w.r.t.) 
		   // which Context is packed. 
		   const unsigned nWrtwCip, 
		   const unsigned int specNum,
		   const unsigned int node)
{
    LogP result;
    
    const unsigned packedBits = bitGather(nWrtwCip,node);

    LogP *prob = 
      fNgrams[specNum].parentSubsets[node].
      findProbSubCtx(word,context,packedBits);

    if (prob) {
      if (running() && debug(DEBUG_NGRAM_HITS)) {
	// note that this message will occur multiple times when doing
	// certain strategies and/or combination methods. The true
	// "hit" is the last (right most) one that is printed.
	char buff[1024];
	sprintf(buff,"[0x%X gram]",node);
	dout() << buff;
      }
      result = *prob;
    } else {
      if (node == 0) {
	if (running() && debug(DEBUG_NGRAM_HITS)) {
	    dout() << "[OOV]";
	}
	result = LogP_Zero;
      } else {
	LogP *bow = 
	  fNgrams[specNum].parentSubsets[node].
	  findBOWSubCtx(context,packedBits);
	if (bow) {
	  // fprintf(stderr,"found bow = %f\n",*bow);
	  result = *bow + 
	    bgChildProbBO(word,context,nWrtwCip,specNum,node);
	} else {
	  if (!fngs.fnSpecArray[specNum].parentSubsets[node].requiresGenBackoff()) {
	    result = bgChildProbBO(word,context,nWrtwCip,specNum,node);
	  } else {
	    // need to compute BOW for this context (i.e., normalize
	    // the BO distribution).
	    Prob sum = bgChildProbSum(context,nWrtwCip,specNum,node);
	    LogP lsum = ProbToLogP(sum);
	    // insert sum as a BOW to speed up access on future access.
	    // NOTE: This next code is one of the reasons BG LM estimation runs so slowly.
	    // TODO: could always recompute to save memory but slow things down.
	    // TODO: this slows things down significantly, think of a way to make
	    // this part run faster! (perhaps compute all bows during training,
	    // but that makes the files large).
	    // NOTE: this code will also cause memory usage to grow, it will
	    // look like a memory leak, but it is not, it is just
	    // the insertion of bows into the lm trie.
	    *(fNgrams[specNum].parentSubsets[node].
	      insertBOWSubCtx(context,packedBits)) = - lsum;
	    result = bgChildProbBO(word,context,nWrtwCip,specNum,node)
	      - lsum;
	  }
	}
      }
    }
    return result;
}



LogP
FNgram::wordProb(VocabIndex word, 
		 const VocabIndex *context,
		 // nWrtwCip: Node number With respect to (w.r.t.) which Context is packed. 
		 const unsigned nWrtwCip, 
		 const unsigned int specNum,
		 const unsigned int node)
{
  if (skipOOVs) {
    /*
     * Backward compatibility with the old broken perplexity code:
     * TODO: figure out what is meant by broken here.
     * return prob 0 if any of the context-words have an unknown
     * word.
     */
    if (word == vocab.unkIndex())
      return LogP_Zero;
    unsigned int clen = Vocab::length(context);
    for (unsigned j=0;j<clen; j++) {
      if (context[j] == vocab.unkIndex())
	return LogP_Zero;
    }
  }

  /*
   * Perform the backoff algorithm for a context lenth that is 
   * the minimum of what we were given and the maximal length of
   * the contexts stored in the LM
   */
  return wordProbBO(word, context, nWrtwCip, specNum, node);
}


// reads in an ARPA-file inspired format -
//   - there are node-grams rather than n-grams, where
//     node is the bit vector (hex integer) giving LM parents that
//     to be used in that gram in the *LM* graph
//   - there might be multiple bows, depending on from where
//     the backoff is coming from.

// TODO: make sure that read sets all variables in FNgram that
// constructor is supposed to do (e.g., "order", and other member variables).
Boolean
FNgram::read(const unsigned int specNum, File &file)
{
    char *line;
    Array<unsigned int> numGrams;   /* the number of n-grams for each parent set */
				
    Array<unsigned int> numRead;      /* Number of n-grams actually read */
    int state = -2 ;		/* section of file being read:
				 * -2 - pre-header, -1 - header,
				 * 0 - unigrams, 1 - node-1 bigrams, 
				   2 - node-2 bigrams, etc, ... */
    unsigned int numBitsInState = 0;
    unsigned int state_order = 0;
    unsigned int max_state = 0;

    Boolean warnedAboutUnk = false; /* at most one warning about <unk> */

    const unsigned int numParents = fngs.fnSpecArray[specNum].numParents;
    const unsigned int numSubSets = fngs.fnSpecArray[specNum].numSubSets;

    for (unsigned i = 0; i < numSubSets; i++) {
	numGrams[i] = 0;
	numRead[i] = 0;
    }

    clear(specNum);

    /*
     * The ARPA format implicitly assumes a zero-gram backoff weight of 0.
     * This has to be properly represented in the BOW trie so various
     * recursive operation work correctly.
     */
    VocabIndex nullContext[1];
    nullContext[0] = Vocab_None;


    *fNgrams[specNum].parentSubsets[0].insertBOW(nullContext) = LogP_Zero;


    while ((line = file.getline())) {
	
	Boolean backslash = (line[0] == '\\');

	switch (state) {

	case -2: 	/* looking for start of header */
	    if (backslash && strncmp(line, "\\data\\", 6) == 0) {
		state = -1;
		continue;
	    }
	    /*
	     * Everything before "\\data\\" is ignored
	     */
	    continue;

	case -1:		/* ngram header */
	    unsigned thisNode;
	    int nFNgrams;

	    // expect this to be of the form "0xC-grams" where "C"
	    // is a hex digit string indicating the BG node
	    if (backslash && sscanf(line, "\\%i-grams", &state) == 1) {
		/*
		 * start reading grams
		 */
		if ((unsigned)state >= numSubSets) {
		  file.position() << "invalid ngram order " << HEX << state << "\n" << DEC;
		  return false;
		}

	        if (debug(DEBUG_READ_STATS)) {
		  dout() << "reading " << numGrams[state] << " " << HEX
			 << state << "-grams\n" << DEC;
		}
		numBitsInState = FNgramSpecsType::numBitsSet(state);
		state_order = numBitsInState+1;
		continue;
	    } else if (sscanf(line, "ngram %i=%d", (int *)&thisNode, &nFNgrams) == 2) {
		/*
		 * scanned a line of the form
		 *	ngram <0xCCCC>=<howmany>
		 * now perform various sanity checks
		 */
		if (thisNode >= numSubSets) {
		  file.position() << "gram node " << HEX << thisNode
				  << " out of range\n" << DEC;
		  return false;
		}
		if (nFNgrams < 0) {
		    file.position() << "gram count number " << nFNgrams
				    << " out of range\n";
		    return false;
		}
		if (thisNode > max_state) {
		  max_state = thisNode;
		}
		numGrams[thisNode] = nFNgrams;
		continue;
	    } else {
		file.position() << "unexpected input\n";
		return false;
	    }

	default:	/* reading n-grams, where n == state */

	    if (backslash && sscanf(line, "\\%i-grams", &state) == 1) {
		if ((unsigned)state >= numSubSets) {
		  file.position() << "invalid ngram order " << HEX << state << "\n" << DEC;
		  return false;
		}

	        if (debug(DEBUG_READ_STATS)) {
		  dout() << "reading " << numGrams[state] << " " << HEX
			 << state << "-grams\n" << DEC;
		}
		numBitsInState = FNgramSpecsType::numBitsSet(state);
		// state order is the number of strings we expect to see on a line
		// i.e., the parents + child.
		state_order = numBitsInState+1;
		/*
		 * start reading more n-grams
		 */
		continue;
	    } else if (backslash && strncmp(line, "\\end\\", 5) == 0) {
		/*
		 * Check that the total number of ngrams read matches
		 * that found in the header
		 */
		for (unsigned i = 0; i <= max_state ; i++) {
		    if (numGrams[i] != numRead[i]) {
		      file.position() << "warning: " << numRead[i] << " " << HEX
				      << i << "-grams read, expected "
				      << numGrams[i] << "\n" << DEC;
		    }
		}

		return true;
	    } else if ((unsigned)state > numSubSets) {
		/*
		 * Save time and memory by skipping ngrams outside
		 * the order range of this model. This is moot
		 * right now as we signal an error condition above.
		 * TODO: change so that file can have higher-order
		 *     than model specifies, where we ignore those.
		 */
		continue;
	    } else {
	        VocabString words[1                         // probability
				  + maxNumParentsPerChild   // longest gram
				  + 1                       // child
				  + 1                       // backoff location
				  + 1                       // optional bow
				  + 1                       // end marker
		];
				/* result of parsing an n-gram line,
				 * some elements are actually
				 * numerical parameters, but so what?  
				 */
		VocabIndex wids[maxNumParentsPerChild + 2];
				/* gram+child+endmark as word indices */
		LogP prob, bow;

		/*
		 * Parse a line of the form
		 *	<prob>	<w1> <w2> ... <wN> <c> [<hx_bg_location>] [<bow>]
		 *
		 * I.e., We can have zero or one <bow>s, We do not
		 * have a hex bg location only when state == 0.  Note,
		 * that unlike the ARPA format, here the <bow> here is
		 * associated with the backoff of <w1> <w2> ... <wN>
		 * <c> at hx_bg_location rather than context
		 * consisting of <w1> <w2> ... <wN> <c>
		 */
		unsigned int howmany =
		  Vocab::parseWords(line, words, maxNumParentsPerChild + 5);

		// unsigned int have_cnt = (numBitsInState > 1) ? 1 : 0;
		unsigned int have_cnt = 0;

		if ((howmany < state_order+1+have_cnt) || (howmany > state_order+2+have_cnt)) {
		  file.position() << "error, ngram line has invalid number (" << howmany
				  << ") of fields, expecting either " << state_order+1+have_cnt 
				  << " or " << state_order+2+have_cnt << "\n";
		  return false;
		}

		/*
		 * Parse prob
		 */
		if (!parseLogP(words[0], prob)) {
		    file.position() << "bad prob \"" << words[0] << "\"\n";
		    return false;
		} else if (prob > LogP_One || prob != prob) {
		    file.position() << "warning: questionable prob \""
				    << words[0] << "\"\n";
		} else if (prob == LogP_PseudoZero) {
		    /*
		     * convert pseudo-zeros back into real zeros
		     */
		    prob = LogP_Zero;
		}

		/*
		 * 
		 * get backoff graph location
		 *
		 */

		unsigned int gram_cnt = 0; // support for counts in lm file, not finished.
		if (have_cnt) {
		  char *endptr = (char *)words[state_order+1];
		  gram_cnt = (unsigned)strtol(words[state_order+1],&endptr,0);
		  if (endptr == words[state_order+1] || gram_cnt == ~0x0U) {
		    file.position() << "warning: invalid backoff graph location\""
				    << words[state_order+1] << "\"\n";
		  }
		}

		/* 
		 * Parse bow, if any
		 */
		if (howmany == state_order + 2 + have_cnt) {
		    /*
		     * Parsing floats strings is the most time-consuming
		     * part of reading in backoff models.  We therefore
		     * try to avoid parsing bows where they are useless,
		     * i.e., for contexts that are longer than what this
		     * model uses.  We also do a quick sanity check to
		     * warn about non-zero bows in that position.
		     */
		    if (state == 0) {
		      // unlike normal ARPA files, here we 
		      // would not expect a bow for the unigram.
			if (words[state_order + 1 + have_cnt][0] != '0') {
			    file.position() << "ignoring non-zero bow \""
					    << words[state_order + 1 + have_cnt]
					    << "\" for minimal ngram\n";
			}
		    } else if (!parseLogP(words[state_order + 1 + have_cnt], bow)) {
			file.position() << "bad bow \"" << 
			  words[state_order + 1 + have_cnt] << "\"\n";
			return false;
		    } else if (bow == LogP_Inf || bow != bow) {
			file.position() << "warning: questionable bow \""
		                    	<< words[state_order + 1 + have_cnt] << "\"\n";
		    } else if (bow == LogP_PseudoZero) {
			/*
			 * convert pseudo-zeros back into real zeros
			 */
			bow = LogP_Zero;
		    }
		}

		/* 
		 * Terminate the words array after the last word,
		 * then translate it to word indices.  We also
		 * reverse the ngram since that's how we'll need it
		 * to index the trie.
		 */
		words[state_order + 1] = 0;
		vocab.addWords(&words[1], wids, maxNumParentsPerChild+2);
		Vocab::reverse(wids);

		/*
		 * Store bow, if any
		 */
		if (howmany == state_order + 2 + have_cnt && state != 0) {
		  *(fNgrams[specNum].parentSubsets[state].insertBOW(&wids[1])) = bow;
		}

		/*
		 * Save the last word (which is now the first, due to reversal)
		 * then use the first n-1 to index into
		 * the context trie, storing the prob.
		 */
		if (!warnedAboutUnk &&
		    wids[0] == vocab.unkIndex() &&
		    prob != LogP_Zero &&
		    !vocab.unkIsWord())
		  {
		    file.position() << "warning: found non-zero LM probability for "
				    << vocab.getWord(vocab.unkIndex())
				    << " in closed-vocabulary LM\n";
		    warnedAboutUnk = true;
		  }
		*(fNgrams[specNum].parentSubsets[state].
		  insertProbAndCNT(wids[0], &wids[1],gram_cnt)) = prob;
		/*
		 * Hey, we're done with this ngram!
		 */
		numRead[state] ++;
	    }
	}
    }

    /*
     * we reached a premature EOF
     */
    file.position() << "reached EOF before \\end\\\n";
    return false;
}

void
FNgram::write()
{
  for (unsigned specNum=0;specNum<fNgramsSize;specNum++) {  
    if (fngs.fnSpecArray[specNum].lmFileName && 
	!(*fngs.fnSpecArray[specNum].lmFileName == '_'
	  && !*(fngs.fnSpecArray[specNum].lmFileName+1))) {
      File f(fngs.fnSpecArray[specNum].lmFileName,"w");
      if (debug(DEBUG_WRITE_STATS)) {
	dout() << "writing FLM to " << fngs.fnSpecArray[specNum].lmFileName << "\n";
      }
      write(specNum,f);
    }
  }
}


Boolean
FNgram::read()
{
  for (unsigned specNum=0;specNum<fNgramsSize;specNum++) {  
    if (fngs.fnSpecArray[specNum].lmFileName && 
	!(*fngs.fnSpecArray[specNum].lmFileName == '_'
	  && !*(fngs.fnSpecArray[specNum].lmFileName+1))) {
      File f(fngs.fnSpecArray[specNum].lmFileName,"r");
      if (debug(DEBUG_READ_STATS)) {
	dout() << "reading FLM from " << fngs.fnSpecArray[specNum].lmFileName << "\n";
      }
      if  (!read(specNum,f))
	return false;
    }
  }
  if (debug(DEBUG_ESTIMATE_LM))
    wordProbSum();
  return true;
}




// writes out in an ARPA-inspired file format -
//   - there are node-grams rather than n-grams, where
//     node is the bit vector giving LM parents in the LM graph
//   - the bow for a context is located with the ngram that
//     could use that bow (rather than for the backoff graph)
void
FNgram::write(unsigned int specNum,File &file)
{
    Array<unsigned> howmanyFNgrams;
    VocabIndex context[maxNumParentsPerChild + 2];
    VocabString scontext[maxNumParentsPerChild + 2];

    file.fprintf("\n\\data\\\n");


    const unsigned numParents = fngs.fnSpecArray[specNum].numParents;
    // starting with the unigram, and moving up to the all LM-parents case.
    for (unsigned level = 0; level <= numParents; level ++) {
      FNgramSpecsType::FNgramSpec::LevelIter liter(numParents,level);
      unsigned int node;
      while (liter.next(node)) {
	howmanyFNgrams[node] = numFNgrams(specNum,node);
	file.fprintf("ngram 0x%X=%d\n",node, howmanyFNgrams[node]);
      }
    }

    for (unsigned level = 0; level <= numParents; level++) {
      FNgramSpecsType::FNgramSpec::LevelIter liter(numParents,level);
      unsigned int node;
      while (liter.next(node)) {
	file.fprintf("\n\\0x%X-grams:\n", node);

        if (debug(DEBUG_WRITE_STATS)) {
	  char buff[1024];
	  sprintf(buff,"writing %d 0x%X-grams\n",howmanyFNgrams[node],node);
	  dout() << buff;
	  // for some reason, gcc3.1.1 produces spurious "2"s with the folowing
	  //dout() << "writing " << DEC << howmanyFNgrams[node] << " 0x" << HEX
	  //<< node << "-grams" << DEC << "\n";
	}
        
	BOsIter citer(*this, specNum, node, context, vocab.compareIndex());
	BOnode *tr_node;


	while ((tr_node = citer.next())) {

	  // TODO: write out BOWs that have contexts but no words
	  // since for gen BW, spent time computing them.

	    vocab.getWords(context, scontext, maxNumParentsPerChild + 1);
	    Vocab::reverse(scontext);

	    ProbsIter piter(*tr_node, vocab.compareIndex());
	    VocabIndex pword;
	    LogP *prob;
	    unsigned*cnt;
	    
	    Boolean first_word = true;
	    while ((prob = piter.next(pword,cnt))) {
		if (file.error()) {
		    return;
		}

		file.fprintf("%.*lg\t", LogP_Precision,
				(double)(*prob == LogP_Zero ?
						LogP_PseudoZero : *prob));
		Vocab::write(file, scontext);
		file.fprintf("%s%s", (node != 0 ? " " : ""), vocab.getWord(pword));

		// if (node > 0 && *cnt != ~0x0U)
		// file.fprintf("\t0x%X", *cnt);

		if (first_word && level > 0) {
		  // write BOW for the context right here rather than
		  // (as in ARPA file) with the n-1 gram because we're
		  // not even guaranteed in a FLM with different symbol sets
		  // for the random varibles that there will be an n-1 gram
		  // at which to store the BOW.
		  LogP *bow =
		    fNgrams[specNum].parentSubsets[node].findBOW(context);
		  if (bow) {
		    file.fprintf("\t%.*lg",LogP_Precision,
			    (double)(*bow == LogP_Zero ?
				       LogP_PseudoZero : *bow));
		  } else {
		    // there should always be a bow for a real context.
		    // in the structures.
		    assert(0);
		  }
		  first_word = false;
		}
		file.fprintf("\n");
	    }
	}
      }
    }
    file.fprintf("\n\\end\\\n");
}

unsigned int
FNgram::numFNgrams(const unsigned int specNum,const unsigned int node)
{
  VocabIndex context[maxNumParentsPerChild + 2];
  unsigned int howmany = 0;

  BOsIter iter(*this,specNum,node,context);
  BOnode *bo_node;

  while ((bo_node = iter.next())) {
    howmany += bo_node->probs.numEntries();
  }
  return howmany;
}

/*
 * Estimation
 */


/*
 * Count number of vocabulary items that get probability mass
 * for the current tag set.
 */
unsigned
FNgram::vocabSize()
{
    unsigned numWords = 0;
    FactoredVocab::TagIter viter(*((FactoredVocab*)&vocab));
    VocabIndex word;

    while (viter.next(word)) {
	if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
	    numWords ++;
	}
    }
    return numWords;
}



void
FNgram::estimate()
{
  for (unsigned specNum=0;specNum<fNgramsSize;specNum++) {  
    estimate(specNum);
  }
  if (debug(DEBUG_ESTIMATE_LM))
    wordProbSum();
}

void
FNgram::estimate(const unsigned int specNum)
{
    assert ( specNum < fNgramsSize );

    // this is necessary so that estimate() doesn't smooth
    // over the wrong random variable value set.
    ((FactoredVocab*)&vocab)->setCurrentTagVocab(fngs.fnSpecArray[specNum].child);

    /*
     * For all ngrams, compute probabilities and apply the discount
     * coefficients.
     */
    VocabIndex context[maxNumParentsPerChild+2];
    unsigned vocabSize = FNgram::vocabSize();

    /*
     * Remove all old contexts
     */
    clear(specNum);

    /*
     * Ensure <s> unigram exists (being a non-event, it is not inserted
     * in distributeProb(), yet is assumed by much other software).
     */

    if (vocab.ssIndex() != Vocab_None) {
	context[0] = Vocab_None;
	*(fNgrams[specNum].parentSubsets[0].insertProb(vocab.ssIndex(), context))
	  = LogP_Zero;
    }

    // By convention, the lowest numeric level in the backoff graph (BG)
    // (level zero) has one node corresponding to all LM parents, so this
    // node has value (2^(numParents)-1) (all bits on). The highest numeric level
    // (level numParents) has one node with no LM parents, so this node
    // has value 0 (all bits off). All other BG nodes have some intermediary
    // number of bits, and the bit pattern determines the number of LM parents
    // that are currently active. See the ASCII pictures in FNgramSpecs.h.

    // We learn the LM by doing a reverse BG level level-iterator,
    // starting with the unigram, and moving up to the all LM-parents case.
    const unsigned numParents = fngs.fnSpecArray[specNum].numParents;
    for (unsigned level = 0; level <= numParents; level++) {
      FNgramSpecsType::FNgramSpec::LevelIter iter(numParents,level);
      unsigned int node;
      while (iter.next(node)) {

	// check if backoff constraints turned off this node, if
	// so we don't learn it.
	if (fngs.fnSpecArray[specNum].parentSubsets[node].counts == NULL)
	  continue;

	const unsigned numBitsSetInNode = FNgramSpecsType::numBitsSet(node);

	unsigned noneventContexts = 0;
	unsigned noneventFNgrams = 0;
	unsigned discountedFNgrams = 0;

	/*
	 * check if discounting is disabled for this round
	 */
	Boolean noDiscount =
	  (fngs.fnSpecArray[specNum].parentSubsets[node].discount == 0) ||
	  fngs.fnSpecArray[specNum].parentSubsets[node].discount->nodiscount();

	// fprintf(stderr,"noDiscount = %d\n",noDiscount);

	Boolean interpolate =
	  (fngs.fnSpecArray[specNum].parentSubsets[node].discount != 0) &&
	  fngs.fnSpecArray[specNum].parentSubsets[node].discount->interpolate;

	/*
	 * assume counts are already "prepared" (which is always true for factored counts)
	 */

	/*
	 * This enumerates all parent set contexts, i.e., i-1 "grams"
	 */
	FNgramCount *contextCount;
	assert ( fngs.fnSpecArray[specNum].parentSubsets[node].order > 0 );
	FNgramSpecsType::FNgramSpec::PSIter 
	  contextIter(fngs.fnSpecArray[specNum].parentSubsets[node],
		      context,
		      fngs.fnSpecArray[specNum].parentSubsets[node].order-1);
	while ((contextCount = contextIter.next())) {
	    /*
	     * If <unk> is not real word, skip contexts that contain
	     * it. Note that original code checked for </s> here in
	     * context. We do not do this here because:
	     *  1) count tries here are for conditional distributions where
	     *     the leaves of the trie is the child, and the non-leaves
	     *     are a set of parent variables.
	     *  2) a context might contain a </s>, say in P(M_t|R_t,R_t-1)
	     *     where the context comes from the same time point.
	     *  3) It is not clear why you would need to do doubling here
	     *     since short word strings are effectively elongated
	     *     in FNgramCounts<CountT>::countSentence() by
	     *     having any context that goes earlier than the beginning
	     *     of the sentence be replaced with <s> (so we can deal
	     *     with sentences shorter than the gram temporal width,
	     *     but this is only the case if the -no-virtual-begin-sentence 
	     *     option is not set).
	     * Note also, that this means that this code will have
	     * fewer pseudo-events when comparing with ngram-count.cc.
	     */
	  if (vocab.isNonEvent(vocab.unkIndex()) &&
	      vocab.contains(context, vocab.unkIndex()))
	    {
	      noneventContexts ++;
	      continue;
	    }

	    VocabIndex word[2];	/* the follow word */
	    FNgramSpecsType::FNgramSpec::PSIter 
	      followIter(fngs.fnSpecArray[specNum].parentSubsets[node],
			 context,word,1);
	    FNgramCount *ngramCount;

	    /*
	     * Total up the counts for the denominator
	     * (the lower-order counts are not consistent with
	     * the higher-order ones, so we can't just use *contextCount)
	     * I.e., we assume here that trustTotal flag is always false.
	     */
	    FNgramCount totalCount = 0;
	    Count observedVocab = 0, min2Vocab = 0, min3Vocab = 0;
	    while ((ngramCount = followIter.next())) {
		if (vocab.isNonEvent(word[0]) ||
		    ngramCount == 0 ||
		    (node == 0 && vocab.isMetaTag(word[0])))
		{
		    continue;
		}

		if (!vocab.isMetaTag(word[0])) {
		    totalCount += *ngramCount;
		    observedVocab ++;
		    if (*ngramCount >= 2) {
			min2Vocab ++;
		    }
		    if (*ngramCount >= 3) {
			min3Vocab ++;
		    }
		} else {
		    /*
		     * Process meta-counts
		     */
		    unsigned type = vocab.typeOfMetaTag(word[0]);
		    if (type == 0) {
			/*
			 * a count total: just add to the totalCount
			 * the corresponding type count can't be known,
			 * but it has to be at least 1
			 */
			totalCount += *ngramCount;
			observedVocab ++;
		    } else {
			/*
			 * a count-of-count: increment the word type counts,
			 * and infer the totalCount
			 */
			totalCount += type * *ngramCount;
			observedVocab += (Count)*ngramCount;
			if (type >= 2) {
			    min2Vocab += (Count)*ngramCount;
			}
			if (type >= 3) {
			    min3Vocab += (Count)*ngramCount;
			}
		    }
		}
	    }

	    if (totalCount == 0) {
		continue;
	    }

	    /*
	     * reverse the context ngram since that's how
	     * the BO nodes are indexed. More specifically,
	     * normal count tries are organized as follows:
	     *    x1 y1 z1 c
	     *    x1 y1 z2 c
	     *    ...
	     *    x1 y1 zn c
	     *    x1 y2 z1 c
	     *    x1 y2 z2 c
	     *    ...
	     *    x1 y2 zn c
	     * and so on, where c is the count for the context,
	     * and where z follows y which follows x in the input
	     * files, so we might want to compute p(z|x,y). I.e.,
	     * the x's are at the root, and the z's are at the leaves
	     * of the count trie.
	     *
	     * When we get one of these 'context' wid strings, we are
	     * iterating over the x_i and y_i's that exist in the
	     * corresponding trie object. (the z's are the follow iter above)
	     *
	     * When LM tries are created, however, we insert them using
	     * a reversed context, so what gets inserted used to index
	     * the triee is "y x" and then z is in the hash table at that node.
	     * 
	     *
	     * In flm file, we've got
	     *
	     *      C | P1 P2 P3
	     * 
	     * here, P1 = bit0, P1 = bit1, P3 = bit2
	     * where a bit vector is organzied in an integer as:
	     *
	     *                     [...  bit2 bit1 bit0]
	     *
	     * In count file, contexts are given as 
	     *
	     *    word[0] = P3, word[1] = P2, word[2] = P1 [ word[3] = C ]
	     *       = bit2        bit1            bit0    
	     *
	     * when we reverse the context, we've got
	     *
	     *    word[0] = P1, word[1] = P2, word[2] = P3
	     *       = bit0         bit1         bit2
	     *
	     * or if we include the child,
	     *
	     *    word[0] = C, word[1] = P1, word[2] = P2, word[3] = P3
	     *                      = bit0         bit1         bit2
	     *
	     * In a normal LM (as is done in NgramLM), the reason for
	     * reversing things in the LM but not in the counts (I
	     * believe) is that it is faster to backoff to a parent
	     * node in the trie rather than having to re-index down
	     * along a completely different path to another node in
	     * the trie. 
	     *
	     * For a general graph backoff, however, this reversal
	     * isn't really necessary since we need to index up into
	     * an entirely different LM trie when we backoff so the
	     * speed advantage is lost. We keep doing the reversal
	     * here, however, just to stay consistent with the
	     * code in NgramLM.cc. This means, however, that 
	     * we need to reverse wid count lookup routines (as
	     * defined in NgramSpecs.h).
	     *     
	    */
	    Vocab::reverse(context);

	    /*
	     * Compute the discounted probabilities
	     * from the counts and store them in the backoff model.
	     */
	retry:
	    followIter.init();
	    Prob totalProb = 0.0;

	    while ((ngramCount = followIter.next())) {
		LogP lprob;
		double discountCoeff;

		/*
		 * zero counts,
		 * Officially, we shouldn't see this for a FLM count trie,
		 * except possibly at node 0.
		 */
		if (node != 0 && *ngramCount == 0) {
		  fprintf(stderr,"WARNING: FNgramLM:  node = 0x%X, *ngramCount = %d\n",
			  node, (unsigned)*ngramCount);
		  cerr << "context: " << (vocab.use(), context) << endl;
		  cerr << "word: " << word << endl;
		  // assert (0);
		  // NOTE: if this happens, check the count
		  // modification code in discount.cc. This might
		  // also be a result of running fngram-count with
		  // the "-no-virtual-begin-sentence" option.
		}

		if (vocab.isNonEvent(word[0]) || vocab.isMetaTag(word[0])) {
		    /*
		     * Discard all pseudo-word probabilities,
		     * except for unigrams.  For unigrams, assign
		     * probability zero.  This will leave them with
		     * prob zero in all cases, due to the backoff
		     * algorithm.
		     * Also discard the <unk> token entirely in closed
		     * vocab models, its presence would prevent OOV
		     * detection when the model is read back in.
		     */
		    if (node != 0 ||
			word[0] == vocab.unkIndex() ||
			vocab.isMetaTag(word[0]))
		    {
			noneventFNgrams ++;
			continue;
		    }

		    lprob = LogP_Zero;
		    discountCoeff = 1.0;
		} else {
		    /*
		     * Ths discount array passed may contain 0 elements
		     * to indicate no discounting at this order.
		     */

		    if (noDiscount) {
			discountCoeff = 1.0;
			// fprintf(stderr,"setting discount to 1.0\n");
		    } else {
			discountCoeff =
			  fngs.fnSpecArray[specNum].
			  parentSubsets[node].discount->discount(*ngramCount, totalCount,
								 observedVocab);
			// fprintf(stderr,"*ngramCount = %d, setting discount to %f\n",
			// *ngramCount,discountCoeff);
		    }
		    Prob prob = (discountCoeff * *ngramCount) / totalCount;

		    /*
		     * For interpolated estimates we compute the weighted 
		     * linear combination of the high-order estimate
		     * (computed above) and the lower-order estimate.
		     * The high-order weight is given by the discount factor,
		     * the lower-order weight is obtained from the Discount
		     * method (it may be 0 if the method doesn't support
		     * interpolation).
		     */
		    double lowerOrderWeight;
		    LogP lowerOrderProb;
		    if (interpolate) {
			lowerOrderWeight = 
			  fngs.fnSpecArray[specNum].
			  parentSubsets[node].discount->lowerOrderWeight(totalCount,
									 observedVocab,
									 min2Vocab,
									 min3Vocab);
			if (node > 0) {
			  lowerOrderProb = 
			    bgChildProbBO(word[0],context,node,specNum,node);
			} else {
			    lowerOrderProb = - log10((double)vocabSize);
			}

			prob += lowerOrderWeight * LogPtoProb(lowerOrderProb);
		    }



		    if (discountCoeff != 0.0) {
			totalProb += prob;
		    }

		    lprob = ProbToLogP(prob);
		    if (discountCoeff != 0.0 && debug(DEBUG_ESTIMATES)) {
			dout() << "CONTEXT " << (vocab.use(), context)
			       << " WORD " << vocab.getWord(word[0])
			       << " NUMER " << *ngramCount
			       << " DENOM " << totalCount
			       << " DISCOUNT " << discountCoeff;

			if (interpolate) {
			    dout() << " LOW " << lowerOrderWeight
				   << " LOLPROB " << lowerOrderProb;
			}
			dout() << " LPROB " << lprob << endl;
		    }
		}

		/*
		 * A discount coefficient of zero indicates this ngram
		 * should be omitted entirely (to save space and to 
		 * ensure that BOW code below works).
		 */
		if (discountCoeff == 0.0) {
		    discountedFNgrams ++;
		    fNgrams[specNum].parentSubsets[node].
		      removeProb(word[0],context);
		} else {
		  // fprintf(stderr,"inserting word %d\n",word[0]);
		  *(fNgrams[specNum].parentSubsets[node].
		    insertProb(word[0],context)) = lprob;
		  // fprintf(stderr,"size now = %d\n",numFNgrams(specNum,node));
		}
	    }

	    /*
	     * This is a hack credited to Doug Paul (by Roni Rosenfeld in
	     * his CMU tools).  It may happen that no probability mass
	     * is left after totalling all the explicit probs, typically
	     * because the discount coefficients were out of range and
	     * forced to 1.0.  To arrive at some non-zero backoff mass
	     * we try incrementing the denominator in the estimator by 1.
	     */
	    if (!noDiscount && totalCount > 0 &&
		totalProb > 1.0 - Prob_Epsilon)
	    {
		totalCount += 1;

		if (debug(DEBUG_ESTIMATE_WARNINGS)) {
		    cerr << "warning: " << (1.0 - totalProb)
			 << " backoff probability mass left for " << 
		      HEX << node << DEC << " context \"" 
			 << (vocab.use(), context)
			 << "\" -- incrementing denominator"
			 << endl;
		}
		goto retry;
	    }

	    /*
	     * Undo the reversal above so the iterator can continue correctly
	     */
	    Vocab::reverse(context);
	}

	if (debug(DEBUG_ESTIMATE_WARNINGS)) {
 	    // use stupid C++ I/O to print in hex
	    if (noneventContexts > 0) {
	      dout() << "discarded " << noneventContexts << " " << HEX
		     << node << "-gram contexts containing pseudo-events\n" << DEC;
	    }
	    if (noneventFNgrams > 0) {
	      dout() << "discarded " << noneventFNgrams << " " << HEX
		     << node << "-gram probs predicting pseudo-events\n" << DEC;
	    }
	    if (discountedFNgrams > 0) {
	      dout() << "discarded " << discountedFNgrams << " " << HEX
		     << node << "-gram probs discounted to zero\n" << DEC;
	    }
	}

	// not finished with this yet (this has to do with keeping the counts in the LM file)
	// storeBOcounts(specNum,node);

	/*
	 * With all the probs in place, BOWs are obtained simply by the usual
	 * normalization.
	 * We do this right away before computing probs of higher order since 
	 * the estimation of higher-order N-grams can refer to lower-order
	 * ones (e.g., for interpolated estimates).
	 */
	// fprintf(stderr,"size now = %d\n",numFNgrams(0,0));
	computeBOWs(specNum,node);
      }
    }
    // fprintf(stderr,"done computing LM\n");

}


/*
 * Compute the numerator and denominator of a backoff weight, checking
 * for sanity.  Returns true if values make sense, and prints a
 * warning if not. This version assumes that there is only one
 * possible BG child in the node and can therefore compute the
 * denominator as 1-\sum_{grams with >0 counts} , which will
 * significantly speed things up in this case.
 */
Boolean
FNgram::computeBOW1child(BOnode *bo_node,  // back off node
			 const VocabIndex *context,
			 const unsigned int nWrtwCip,
			 unsigned int specNum, 
			 unsigned int node, // back-off *graph* node
			 Prob &numerator, 
			 Prob &denominator)
{

    /*
     * The BOW(c) for a context c is computed to be
     *
     *	BOW(c) = (1 - Sum p(x | c)) /  (1 - Sum p_BO(x | c))
     *
     * where Sum is a summation over all words x with explicit
     * probabilities in context c, p(x|c) is that probability, and
     * p_BO(x|c) is the probability for that word according to the
     * backoff graph algorithm. In this case, it is assumed that
     * there is only one possible BG child, which is why
     * we can do the denominator in this way.
     */

    LogP *prob;
    numerator = 1.0;
    denominator = 1.0;
    ProbsIter piter(*bo_node);
    VocabIndex word;
    while ((prob = piter.next(word))) {
      numerator -= LogPtoProb(*prob);
      if (node != 0)
	denominator -= LogPtoProb(bgChildProbBO(word,context,nWrtwCip,specNum,node));
    }

    /*
     * Avoid some predictable anomalies due to rounding errors
     */
    if (numerator < 0.0 && numerator > -Prob_Epsilon) {
	numerator = 0.0;
    }
    if (denominator < 0.0 && denominator > -Prob_Epsilon) {
	denominator = 0.0;
    }

    if (numerator < 0.0) {
	cerr << "BOW numerator for context \""
	     << (vocab.use(), context)
	     << "\" is " << numerator << " < 0\n";
	return false;
    } else if (denominator <= 0.0) {
	if (numerator > Prob_Epsilon) {
	    cerr << "BOW denominator for context \""
		 << (vocab.use(), context)
		 << "\" is " << denominator << " <= 0,"
		 << "numerator is " << numerator
		 << endl;
	    return false;
	} else {
	    numerator = 0.0;
	    denominator = 0.0;	/* will give bow = 1 */
	    return true;
	}
    } else {
	return true;
    }
}



/*
 * Compute the numerator and denominator of a backoff weight,
 * checking for sanity.  Returns true if values make sense,
 * and prints a warning if not. 
 * 
 * Note: This version allows for an arbitrary backoff strategy, so
 * computes the denominator as \sum_{ngrams where counts are zero}.
 * This means the training algorithm runs much slower, but thats
 * what you pay for doing general backoff. Note that there is
 * no penalty once the language model has been trained.
 */
Boolean
FNgram::computeBOW(BOnode *bo_node,  // back off node
		   const VocabIndex *context,
		   const unsigned int nWrtwCip,
		   unsigned int specNum, 
		   unsigned int node, // back-off *graph* node
		   Prob &numerator, 
		   Prob &denominator)
{

    /*
     * The BOW(c) for a context c is computed to be
     *
     *	BOW(c) = (1 - Sum p(x | c)) /  Sum' p_BO(x | c))
     *
     * where Sum is a summation over all words x with explicit probabilities
     * in context c, Sum' is over all words without explicit probs in
     * context c, p(x|c) is that probability, and p_BO(x|c) is the 
     * probability for that word according to the backoff algorithm.
     */

    LogP *prob;
    numerator = 1.0;
    denominator = 0.0;
    // double tmp_denominator = 1.0;
    ProbsIter piter(*bo_node);
    VocabIndex word;
    while ((prob = piter.next(word))) {
      numerator -= LogPtoProb(*prob);
      // if (node != 0)
      // tmp_denominator -= LogPtoProb(bgChildProbBO(word,context,nWrtwCip,specNum,node));
    }

    if (node == 0) {
      // no reason to backoff to nothing.
      denominator = 0;
    } else {
      denominator = 0;
      /***************************************************************************
       *
       *    NOTE: This next loop is one of the reasons BG LM estimation runs so slowly.
       *
       ***************************************************************************
       */
      FactoredVocab::TagIter titer(*((FactoredVocab*)&vocab));
      while (titer.next(word)) {
	// accumulate only when count(word,context) == 0
	// Count *cnt = fngs.fnSpecArray[specNum].parentSubsets[node].findCountR(context,word);
	// if (!cnt || *cnt == 0) {
	Boolean found;
	ProbEntry *pe = bo_node->probs.find(word,found);
	if (!found) {
	  const LogP lp = bgChildProbBO(word,context,nWrtwCip,specNum,node);
	  const Prob p = LogPtoProb(lp);
	  denominator += p;
	}
      }
    }
    // fprintf(stderr,"tmp_denominator = %f\n",tmp_denominator);
    // denominator = tmp_denominator;

    /*
     * Avoid some predictable anomalies due to rounding errors
     */
    if (numerator < 0.0 && numerator > -Prob_Epsilon) {
	numerator = 0.0;
    }

    if (numerator < 0.0) {
	cerr << "BOW numerator for context \""
	     << (vocab.use(), context)
	     << "\" is " << numerator << " < 0\n";
	return false;
    } else {
	return true;
    }
}


/*
 * Recompute backoff weight for all contexts of a given order
 */
Boolean
FNgram::computeBOWs(unsigned int specNum, unsigned int node)
{
    Boolean result = true;

    /*
     * Note that this will only generate backoff nodes for those
     * contexts that have words with explicit probabilities.  But
     * this is precisely as it should be.
     */
    BOnode *bo_node;
    VocabIndex context[maxNumParentsPerChild + 2];
    
    // fprintf(stderr,"computebow size now = %d\n",numFNgrams(0,0));
    BOsIter iter1(*this, specNum, node, context);

    unsigned totalNumContexts = 0;
    if (fngs.fnSpecArray[specNum].parentSubsets[node].requiresGenBackoff()) {
      // print out message since this will take a while.
      fprintf(stderr, "Starting estimation of general graph-backoff node: LM %d Node 0x%X, children:",specNum,node);
      FNgramSpecsType::FNgramSpec::BGChildIterCnstr
	citer(fngs.fnSpecArray[specNum].numParents,node,
	      fngs.fnSpecArray[specNum].parentSubsets[node].backoffConstraint);
      unsigned child;
      while (citer.next(child)) {
	if (fngs.fnSpecArray[specNum].parentSubsets[child].counts != NULL)
	  fprintf(stderr, " 0x%X",child);
      }
      fprintf(stderr, "\n");
      if (totalNumContexts == 0) {
	// it is actually worth it to count the number of contexts here
	// to report a informative status message below.
	while (iter1.next()) {
	  totalNumContexts++;
	}
	iter1.init();
      }
    }

    unsigned iter = 0;
    while ((bo_node = iter1.next())) {

      if (debug(DEBUG_BOWS) && fngs.fnSpecArray[specNum].parentSubsets[node].numBGChildren > 1)
	fprintf(stderr,"in computeBOWs, bo_node = 0x%llX, specNum=%d, node = 0x%X, context = 0x%llX, *context = %d, iter = %d, cword = %s, nc = %d\n",
		(unsigned long long)(size_t)bo_node, specNum, node,
		(unsigned long long)(size_t)context, *context, iter, vocab.getWord(*context),
		fngs.fnSpecArray[specNum].parentSubsets[node].numBGChildren);

      double numerator, denominator;
      if (((fngs.fnSpecArray[specNum].parentSubsets[node].requiresGenBackoff())) &&
	  computeBOW(bo_node, context, node,specNum, node,numerator,denominator)) {
	/*
	 * If unigram probs leave a non-zero probability mass
	 * then we should give that mass to the zero-order (uniform)
	 * distribution for zeroton words.  However, the ARPA
	 * BO format doesn't support a "null context" BOW.
	 * We simluate the intended distribution by spreading the
	 * left-over mass uniformly over all vocabulary items that
	 * have a zero probability.
	 *
	 * NOTE: We used to do this only if there was prob mass left,
	 * but some ngram software requires all words to appear as
	 * unigrams, which we achieve by giving them zero probability.
	 *
	 * NOTE2: since FNgramLM.cc doesn't use ARPA format files (it
	 * uses ARPA format "inspired" files), this step really isn't
	 * required any longer, as we could easily define a special
	 * null context. We nevertheless keep the same convention here
	 * for the unigram.
	 *
	 */
	if (node == 0 /*&& numerator > 0.0*/) {
	  distributeProb(specNum,node,numerator,context,node);
	} else if (numerator == 0.0) {
	  bo_node->bow = LogP_One;
	} else {
	  bo_node->bow = ProbToLogP(numerator) - ProbToLogP(denominator);
	  // fprintf(stderr,"got bow, num = %f, den = %f, bow = %f\n",
	  // numerator,denominator,bo_node->bow);

	}
	// should print out some message since this will be taking a long time.
	if (iter % 1000 == 0)
	  fprintf(stderr,
		 "Computing BOWs for LM %d, BG node 0x%X, Context num %d/%d [%.2f%%]\n",
		 specNum,
		 node,
		 iter,totalNumContexts,100*(double)iter/totalNumContexts);
      } else if (!fngs.fnSpecArray[specNum].parentSubsets[node].requiresGenBackoff()
 		 && computeBOW1child(bo_node,context,node,specNum,node,numerator,denominator)) {
 	if (node == 0 /*&& numerator > 0.0*/) {
 	  distributeProb(specNum,node,numerator,context,node);
 	} else if (numerator == 0.0 && denominator == 0) {
 	  bo_node->bow = LogP_One;
 	} else {
 	  bo_node->bow = ProbToLogP(numerator) - ProbToLogP(denominator);
	}
      } else {
	/*
	 * Dummy value for improper models
	 */
	bo_node->bow = LogP_Zero;
	result = false;
      }
      iter++;

    }

    if (fngs.fnSpecArray[specNum].parentSubsets[node].numBGChildren > 1) {
      fprintf(stderr, "Finished estimation of multi-child graph-backoff node: LM %d Node 0x%X\n",specNum,node);
    }

    return result;
}



void
FNgram::storeBOcounts(unsigned int specNum, unsigned int node)
{
  if (node == 0 || FNgramSpecsType::numBitsSet(node) == 1)
    return;
  VocabIndex context[maxNumParentsPerChild + 2];
  BOsIter iter1(*this, specNum, node, context);
  BOnode *bo_node;
  while ((bo_node = iter1.next())) {
    ProbsIter piter(*bo_node);
    VocabIndex word;
    unsigned int* cnt;
    LogP *prob;
    while ((prob = piter.next(word,cnt))) {
      // get count and store it
      // *cnt = count for this node.
    }
  }
}


/*
 * Renormalize language model by recomputing backoff weights.
 */
void
FNgram::recomputeBOWs()
{
    /*
     * Here it is important that we compute the backoff weights in
     * increasing BG level order, since the higher-order ones refer to the
     * lower-order ones in the backoff algorithm.
     * Note that this will only generate backoff nodes for those
     * contexts that have words with explicit probabilities.  But
     * this is precisely as it should be.
     */
  for (unsigned specNum=0;specNum<fNgramsSize;specNum++) {
    const unsigned numParents = fngs.fnSpecArray[specNum].numParents;
    for (unsigned level = 0; level <= numParents; level++) {
      FNgramSpecsType::FNgramSpec::LevelIter iter(numParents,level);
      unsigned int node;
      while (iter.next(node)) {
	computeBOWs(specNum,node);
      }
    }
  }
}

/*
 * Redistribute probability mass over ngrams of given context
 */
void
FNgram::distributeProb(unsigned int specNum,
		       unsigned int node,
		       Prob mass, 
		       VocabIndex *context,
		       const unsigned nWrtwCip)
{
    /*
     * First enumerate the vocabulary to count the number of
     * items affected
     */
    unsigned numWords = 0;
    unsigned numZeroProbs = 0;
    const unsigned packedBits = bitGather(nWrtwCip,node);


    FactoredVocab::TagIter viter(*((FactoredVocab*)&vocab));
    VocabIndex word;

    while (viter.next(word)) {
	if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
	    numWords ++;

	    LogP *prob = 
	      fNgrams[specNum].parentSubsets[node].findProbSubCtx(word,
								  context,
								  packedBits);
	    if (!prob || *prob == LogP_Zero) {
		numZeroProbs ++;
	    }
	    /*
	     * create zero probs so we can update them below
	     */
	    if (!prob) {
	      *(fNgrams[specNum].parentSubsets[node].insertProbSubCtx(word, 
								      context,
								      packedBits))
		= LogP_Zero;
	    }
	}
    }

    /*
     * If there are no zero-probability words 
     * then we add the left-over prob mass to all unigrams.
     * Otherwise do as described above.
     */
    viter.init();

    if (numZeroProbs > 0) {
	if (debug(DEBUG_ESTIMATE_WARNINGS)) {
	    cerr << "warning: distributing " << mass
		 << " left-over probability mass over "
		 << numZeroProbs << " zeroton words" << endl;
	}
	Prob add = mass / numZeroProbs;

	while (viter.next(word)) {
	    if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
		LogP *prob =
		  fNgrams[specNum].parentSubsets[node].insertProbSubCtx(word,
									context,
									packedBits);
		if (*prob == LogP_Zero) {
		    *prob = ProbToLogP(add);
		}
	    }
	}
    } else {
	if (mass > 0.0 && debug(DEBUG_ESTIMATE_WARNINGS)) {
	    cerr << "warning: distributing " << mass
		 << " left-over probability mass over all "
		 << numWords << " words" << endl;
	}
	Prob add = mass / numWords;

	while (viter.next(word)) {
	    if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
		LogP *prob =
		  fNgrams[specNum].parentSubsets[node].insertProbSubCtx(word,
									context,
									packedBits);
		*prob = ProbToLogP(LogPtoProb(*prob) + add);
	    }
	}
    }
}


//
// simple interface to get a word probability for a word
// for a given spec with word and its context loaded into a WidMatrix.
LogP
FNgram::wordProb(unsigned int specNum,
		 WidMatrix& wm,
		 const unsigned int childPos, // position of child in wm
		 const unsigned int length) // length of wm

{
  assert (length > 0);
  // the word is in position (length - 1)
  // the context is in earlier positions, so
  // for a tri-gram (length 3), we have:
  // w(t-2) w(t-1) w_t in wm[0] wm[1] wm[2]


  VocabIndex wids[maxNumParentsPerChild + 2];  

  Boolean unk_detected = false;
  const int numSubSets = fngs.fnSpecArray[specNum].numSubSets;
  const unsigned int context_bits = numSubSets-1;

  const VocabIndex child = wm[childPos][fngs.fnSpecArray[specNum].childPosition];

  if (child == vocab.unkIndex())
    unk_detected = true;

  // if (child == vocab.pauseIndex() || noiseVocab.getWord(child))
  // continue;

  // construct context in increasing order. I.e., if we have mode
  // W : P1 P2 P3
  // then wids[0] = P1, wids[1] = P2, wids[2] = P3
  unsigned wid_index = 0;
  for (unsigned p = 0; p < fngs.fnSpecArray[specNum].numParents; p++) {
    const int parent_time_loc = fngs.fnSpecArray[specNum].parentOffsets[p] + (int)childPos;
    if ((int)parent_time_loc < (int)0) {
      // load with something that will never occur so will cause
      // a backoff
      VocabIndex tmp;
      if (virtualBeginSentence)
	tmp = vocab.ssIndex();
      else // force a backoff
	tmp = (*((FactoredVocab*)&vocab)).emptySlot;
      wids[wid_index++] = tmp;
    } else if ((int)parent_time_loc > (int)(length-1)) {
      VocabIndex tmp;
      if (virtualEndSentence)
	tmp = vocab.seIndex();
      else // force a backoff
	tmp = (*((FactoredVocab*)&vocab)).emptySlot;
      wids[wid_index++] = tmp;
    } else {
      const VocabIndex tmp =
	wm[parent_time_loc][fngs.fnSpecArray[specNum].parentPositions[p]];
      if (tmp == vocab.unkIndex())
	unk_detected = true;
      wids[wid_index++] = tmp;
    }
  }
  wids[wid_index] = Vocab_None;

  if (debug(DEBUG_PRINT_WORD_PROBS)) {
    dout() << "\tp( " << vocab.getWord(child) << " | ";
    for (unsigned p = 0; p < fngs.fnSpecArray[specNum].numParents; p++) {
      dout() << vocab.getWord(wids[p]) << 
	((p+1)<fngs.fnSpecArray[specNum].numParents?",":"");
    }      
    dout() << ") \t= " ;
  }

  LogP prob;
  if (unk_detected && skipOOVs)
    prob = LogP_Zero;
  else 
    prob = wordProbBO(child,wids,context_bits,specNum,context_bits);

  if (debug(DEBUG_PRINT_WORD_PROBS)) {
    dout() << " " << LogPtoProb(prob) << " [ " << prob << " ]";
    dout() << endl;
  }
  if (debug(DEBUG_PRINT_PROB_SUMS)) {
    Prob p = wordProbSum(wids,context_bits,specNum,context_bits);
    dout() << "\tSum_w p(w|context) = " << p << endl;
  }
  return prob;
}


//
// product form of wordProb
LogP
FNgram::wordProb(WidMatrix& wm,
		 const unsigned int childPos, // position of child in wm
		 const unsigned int length) // length of wm

{


  assert (length > 0);
  // the word is in position (length - 1)
  // the context is in earlier positions, so
  // for a tri-gram (length 3), we have:
  // w(t-2) w(t-1) w_t in wm[0] wm[1] wm[2]

  VocabIndex wids[maxNumParentsPerChild + 2];  

  Boolean unk_detected = false;
#if 0
  LogP totalProb = 0.0;
  unsigned totalOOVs = 0;
  unsigned totalZeros = 0;
#endif
  
  LogP prob = LogP_One;
  for (unsigned specNum=0;specNum<fngs.fnSpecArray.size();specNum++) {  
    ((FactoredVocab*)&vocab)->setCurrentTagVocab(fngs.fnSpecArray[specNum].child);
    LogP p = wordProb(specNum,wm,childPos,length);

    prob += p;
  }

#if 0
  if (prob == LogP_Zero) {
    if (child == vocab.unkIndex())
      totalOOVs++;
    else 
      totalZeros++;
  } else {
    totalProb = prob;
  }  
  stats.numSentences ++;
  stats.numWords += 1;
  stats.numOOVs += totalOOVs;
  stats.zeroProbs += totalZeros;
  stats.prob += totalProb;
#endif

  return prob;
}


LogP
FNgram::sentenceProb(unsigned int specNum,
		     WidMatrix& wm,
		     const unsigned int start, // pos of first token
		     const unsigned int end,   // pos of last token
		     TextStats& stats)
{
  VocabIndex wids[maxNumParentsPerChild + 2];  

  LogP totalProb = 0.0;
  unsigned totalOOVs = 0;
  unsigned totalZeros = 0;

    /*
     * Indicate to lm methods that we're in sequential processing
     * mode.
     */
  Boolean wasRunning = running(true);

  // TODO: move this stuff to specs.
  Boolean allParentsFromPast = true;
  Boolean allParentsFromFuture = true;
  for (unsigned p = 0; p < fngs.fnSpecArray[specNum].numParents; p++) {
    const int parent_time_offset = fngs.fnSpecArray[specNum].parentOffsets[p];
    if (parent_time_offset > 0)
      allParentsFromPast = false;
    else if (parent_time_offset < 0)
      allParentsFromFuture = false;
  }

  // TODO: assume that position 0 is word position, i.e., FNGRAM_WORD_TAG_POS
  // which for now is defined in FNgramSpecs.cc but should be placed
  // in FactoredVocab.h
  const unsigned int numSubSets = fngs.fnSpecArray[specNum].numSubSets;
  const unsigned int context_bits = numSubSets-1;
  // skip initial word which is assumed to contain a <s>
  unsigned realStart;
  unsigned realEnd;
  unsigned numRealWords;
  if (noScoreSentenceBoundaryMarks) {
    realStart = start+1;
    realEnd = end-1;
  } else if (allParentsFromPast) {
    realStart = start+1;
    realEnd = end;
  } else if (allParentsFromFuture) {
    realStart = start;
    realEnd = end-1;
  } else {
    // have parents both from past and future, so essentially force
    // the noScoreSentenceBoundaryMarks condition.
    realStart = start+1;
    realEnd = end-1;
  }

  for (unsigned i = realStart; i <= realEnd ; i++) {
    Boolean unk_detected = false;

    const VocabIndex child = wm[i][fngs.fnSpecArray[specNum].childPosition];

    if (child == vocab.unkIndex())
      unk_detected = true;

    if (child == vocab.pauseIndex() || noiseVocab.getWord(child))
      continue;

    // construct context in increasing order. I.e., if we have mode
    // W : P1 P2 P3
    // then wids[0] = P1, wids[1] = P2, wids[2] = P3
    unsigned wid_index = 0;
    for (unsigned p = 0; p < fngs.fnSpecArray[specNum].numParents; p++) {
      const int parent_time_loc = fngs.fnSpecArray[specNum].parentOffsets[p] + (int)i;
      if ((int)parent_time_loc < (int)start) {
	// load with something that will never occur so will cause
	// a backoff
	VocabIndex tmp;
	if (virtualBeginSentence)
	  tmp = vocab.ssIndex();
	else // force a backoff
	  tmp = (*((FactoredVocab*)&vocab)).emptySlot;
	wids[wid_index++] = tmp;
      } else if ((int)parent_time_loc > (int)end) {
	VocabIndex tmp;
	if (virtualEndSentence)
	  tmp = vocab.seIndex();
	else // force a backoff
	  tmp = (*((FactoredVocab*)&vocab)).emptySlot;
	wids[wid_index++] = tmp;
      } else {
	const VocabIndex tmp =
	  wm[parent_time_loc][fngs.fnSpecArray[specNum].parentPositions[p]];
	if (tmp == vocab.unkIndex())
	  unk_detected = true;
	wids[wid_index++] = tmp;
      }
    }
    wids[wid_index] = Vocab_None;

    if (debug(DEBUG_PRINT_WORD_PROBS)) {
      dout() << "\tp( " << vocab.getWord(child) << " | ";
      for (unsigned p = 0; p < fngs.fnSpecArray[specNum].numParents; p++) {
	dout() << vocab.getWord(wids[p]) << 
	  ((p+1)<fngs.fnSpecArray[specNum].numParents?",":"");
      }      
      dout() << ") \t= " ;
    }

    LogP prob;
    if (unk_detected && skipOOVs)
      prob = LogP_Zero;
    else 
      prob = wordProbBO(child,wids,context_bits,specNum,context_bits);

    if (debug(DEBUG_PRINT_WORD_PROBS)) {
      dout() << " " << LogPtoProb(prob) << " [ " << prob << " ]";
      dout() << endl;
    }
    if (debug(DEBUG_PRINT_PROB_SUMS)) {
      Prob p = wordProbSum(wids,context_bits,specNum,context_bits);
      dout() << "\tSum_w p(w|context) = " << p << endl;
    }
    if (prob == LogP_Zero) {
      if (child == vocab.unkIndex())
	totalOOVs++;
      else 
	totalZeros++;
    } else {
      totalProb += prob;
    }
  }
  
  running(wasRunning);

  stats.numSentences ++;
  // stats.numWords += (i-start-2); // don't count ss & se
  // TODO: should check that ss and se are there since we now have
  // command line options regarding this.
  stats.numWords += (end-start+1-2); // don't count ss & se
  stats.numOOVs += totalOOVs;
  stats.zeroProbs += totalZeros;
  stats.prob += totalProb;

  return totalProb;
}

static TLSWC(WidMatrix, widMatrixTLS);

LogP
FNgram::sentenceProb(WordMatrix& wordMatrix,
		     unsigned int howmany,
		     const Boolean addWords,
		     LogP* parr)
{
  WidMatrix &widMatrix = TLSW_GET(widMatrixTLS);

  ::memset(widMatrix[0],0,(maxNumParentsPerChild+1)*sizeof(VocabIndex));
  for (unsigned i = 0; i < howmany; i++) {
    ::memset(widMatrix[i+1],0,(maxNumParentsPerChild+1)*sizeof(VocabIndex));
    if (addWords) 
      vocab.addWords(wordMatrix[i],widMatrix[i+1],maxNumParentsPerChild+1);
    else
      vocab.getIndices(wordMatrix[i],widMatrix[i+1],maxNumParentsPerChild+1,
		       vocab.unkIndex());
  }
  unsigned start;
  unsigned end;
  if (widMatrix[1][0] == vocab.ssIndex()) {
    for (unsigned j=0;j<(maxNumParentsPerChild+1);j++) {
      widMatrix[1][j] = vocab.ssIndex();
    }
    start = 1;
  } else {
    // add extra start sentence token
    for (unsigned j=0;j<(maxNumParentsPerChild+1);j++) {
      widMatrix[0][j] = vocab.ssIndex();
    }
    start = 0;
  }

  if (widMatrix[howmany][0] == vocab.seIndex()) {
    for (unsigned j=0;j<(maxNumParentsPerChild+1);j++) {
      widMatrix[howmany][j] = vocab.seIndex();
      widMatrix[howmany+1][j] = Vocab_None;
    }
    end = howmany;
  } else {
    // need to add se index.
    for (unsigned j=0;j<(maxNumParentsPerChild+1);j++) {
      widMatrix[howmany+1][j] = vocab.seIndex();
      widMatrix[howmany+2][j] = Vocab_None;
    }
    end = howmany+1;
  }
  
  if (debug(DEBUG_EXTREME)) {
    fprintf(stderr,"Done converting word string matrix into wid matrix\n");
  }
  
  LogP prob = LogP_One;
  for (unsigned specNum=0;specNum<fngs.fnSpecArray.size();specNum++) {  
    TextStats lm_stats;
    ((FactoredVocab*)&vocab)->setCurrentTagVocab(fngs.fnSpecArray[specNum].child);
    LogP p = sentenceProb(specNum,widMatrix,start,end,lm_stats);
    
    if (parr != NULL)
      parr[specNum] = p;

    if (debug(DEBUG_PRINT_SENT_PROBS)) {
      dout() << "LM(" << specNum << ")\n";
      dout() << lm_stats << endl;
    }
    // multiply probs together
    prob += p;

    fngs.fnSpecArray[specNum].stats.increment(lm_stats);
  }
  return prob;
}


/*
 * Perplexity from text
 *	The escapeString is an optional line prefix that marks information
 *	that should be passed through unchanged.  This is useful in
 *	constructing rescoring filters that feed hypothesis strings to
 *	pplFile(), but also need to pass other information to downstream
 *	processing.
 */
static TLSWC(WordMatrix, pplFileWordMatrixTLS);
unsigned int
FNgram::pplFile(File &file, TextStats &stats, const char *escapeString)
{
    char *line;
    unsigned escapeLen = escapeString ? strlen(escapeString) : 0;
    unsigned stateTagLen = stateTag ? strlen(stateTag) : 0;
    VocabString sentence[maxWordsPerLine + 1];
    unsigned totalWords = 0;
    unsigned sentNo = 0;
    unsigned int howmany;
    WordMatrix &wordMatrix = TLSW_GET(pplFileWordMatrixTLS);

    for (unsigned specNum=0;specNum<fngs.fnSpecArray.size();specNum++) {
      fngs.fnSpecArray[specNum].stats.reset();
    }

    while ((line = file.getline())) {

	if (escapeString && strncmp(line, escapeString, escapeLen) == 0) {
	    dout() << line;
	    continue;
	}

	/*
	 * check for directives to change the global LM state
	 */
	if (stateTag && strncmp(line, stateTag, stateTagLen) == 0) {
	    /*
	     * pass the state info the lm to let it do whatever
	     * it wants with it
	     */
	    setState(&line[stateTagLen]);
	    continue;
	}

	sentNo ++;

	unsigned int numWords =
			vocab.parseWords(line, sentence, maxWordsPerLine + 1);

	if (numWords == maxWordsPerLine + 1) {
	    file.position() << "too many words per sentence\n";
	} else {

	    if (debug(DEBUG_PRINT_SENT_PROBS)) {
	      dout() << "----- Sentence: " << sentNo << endl;
	      dout() << sentence << endl;
	    }

	    howmany = fngs.loadWordFactors(sentence,wordMatrix,maxWordsPerLine + 1);

	    LogP prob = sentenceProb(wordMatrix,howmany);

	    stats.numSentences ++;
	    stats.numWords += numWords;
	    stats.prob += prob;
	    // TODO: update numOOVs and zeroProbs

	    totalWords += numWords;
	}
    }
    return totalWords;
}


unsigned int
FNgram::pplPrint(ostream &stream, char *fileName)
{
  for (unsigned specNum=0;specNum<fngs.fnSpecArray.size();specNum++) {
    stream << "Perplexity on file: " << fileName << endl;
    stream << "Language model: " << specNum << endl;
    stream << fngs.fnSpecArray[specNum].stats;
  }
  return 0;
}



//
// this routine ONLY supports n-best files of type 3 (i.e., it does
// not support the old decipher file formats)
// TODO: support more than 1 file format for n-best rescoring .
static const unsigned maxFieldsPerLine = maxWordsPerLine + 4;
static TLSWC(WordMatrix, rescoreFileWordMatrixTLS);
static TLSW_ARRAY(VocabString, rescoreFileWstrings, maxFieldsPerLine);
static TLSW_ARRAY(VocabString, rescoreFileWstringsCopy, maxFieldsPerLine);

unsigned int
FNgram::rescoreFile(File &file, double lmScale, double wtScale,
		    LM &oldLM, double oldLmScale, double oldWtScale,
		    const char *escapeString)
{
  char *line;
  unsigned escapeLen = escapeString ? strlen(escapeString) : 0;
  unsigned stateTagLen = stateTag ? strlen(stateTag) : 0;
  unsigned sentNo = 0;
  LogP* parr = new LogP[fngs.fnSpecArray.size()];
  assert (parr);

  WordMatrix   wordMatrix   = TLSW_GET(rescoreFileWordMatrixTLS);
  VocabString *wstrings     = TLSW_GET_ARRAY(rescoreFileWstrings);
  VocabString *wstringsCopy = TLSW_GET_ARRAY(rescoreFileWstringsCopy);

  for (unsigned i=0;i<fngs.fnSpecArray.size();i++) {    
    fngs.fnSpecArray[i].stats.reset();
  }

  while ((line = file.getline())) {
    LogP acousticScore;
    LogP languageScore;
    unsigned numWords; // number of words field (3rd col) in file
    unsigned actualNumWords; // actual number of words there

    if (escapeString && strncmp(line, escapeString, escapeLen) == 0) {
      fputs(line, stdout);
      continue;
    }

    /*
     * check for directives to change the global LM state
     */
    if (stateTag && strncmp(line, stateTag, stateTagLen) == 0) {
      /*
       * pass the state info the lm to let let if do whatever
       * it wants with it
       */
      setState(&line[stateTagLen]);
      continue;
    }

    sentNo ++;
	


    // make copy to print out better error messages
    makeArray(char, lineCopy, strlen(line)+1);
    strcpy(lineCopy,line);

    actualNumWords =
      Vocab::parseWords(line, wstrings, maxFieldsPerLine);

    if (actualNumWords == maxFieldsPerLine) {
      file.position() << "more than " << maxFieldsPerLine
      		      << " fields per line\n";
      continue;
    }

    // assume acoustic, orig lm, and numwords fields start
    actualNumWords -= 3;

    if (actualNumWords > maxWordsPerLine) {
      file.position() << "more than " << maxWordsPerLine << " words in hyp\n";
      continue;
    }

    /*
     * Parse the first three columns as numbers
     */
    if (sscanf(wstrings[0], "%f", &acousticScore) != 1)
      {
	file.position() << "bad acoustic score: " << wstrings[0] << endl;
	continue;
      }
    if (sscanf(wstrings[1], "%f", &languageScore) != 1)
      {
	file.position() << "bad LM score: " << wstrings[1] << endl;
	continue;
      }
    if (sscanf(wstrings[2], "%u", &numWords) != 1)
      {
	file.position() << "bad word count: " << wstrings[2] << endl;
	continue;
      }

    actualNumWords = 
      fngs.loadWordFactors(wstrings+3,wordMatrix,maxWordsPerLine+1);

    languageScore = lmScale*sentenceProb(wordMatrix,
					 actualNumWords,
					 true, // add words is true
					 parr);


    // extra fields for error messages.
    Vocab::parseWords(lineCopy, wstringsCopy, maxFieldsPerLine);    
    Boolean errPrintHyp = false;
    for (unsigned i=0;i<fngs.fnSpecArray.size();i++) {    
      if (fngs.fnSpecArray[i].stats.zeroProbs > 0) {
	file.position() << "LM(" << i << ") warning: hyp " << sentNo
	                << " contains zero prob words: " << endl;
	errPrintHyp = true;
      }
      if (fngs.fnSpecArray[i].stats.numOOVs > 0) {
	file.position() << "LM(" << i << ") warning: hyp " << sentNo
	                << " contains OOV words: " << endl;
	errPrintHyp = true;
      }
      // reset for next list
      fngs.fnSpecArray[i].stats.reset();
    }
    if (errPrintHyp) {
      cerr << "hyp:";
      for (unsigned wrd=0;wrd<actualNumWords;wrd++) {
	cerr << " " << wstringsCopy[3+wrd];
      }
      cerr << endl;
    }

    // compute this just for fun.
    LogP totalScore = acousticScore + 
      languageScore + wtScale * numWords;
    fprintf(stdout, "%g ",acousticScore);
    if (combineLMScores)
      fprintf(stdout, "%g ",languageScore);
    else {
      for (unsigned i=0;i<fngs.fnSpecArray.size();i++) {
	fprintf(stdout, "%g ",lmScale*parr[i]);
      }
    }
    fprintf(stdout, "%d",numWords);
    for (unsigned i=0;i<actualNumWords;i++) {
      fprintf(stdout," %s",wstringsCopy[3+i]);
    }
    fprintf(stdout,"\n");
  }

  delete [] parr;
  return sentNo;
}


/*
 * Total probabilites
 *	For debugging purposes, compute the sum of all word probs
 *	in a context.
 */
Prob
FNgram::wordProbSum(const VocabIndex *context, 
		    // nWrtwCip: Node number With respect to (w.r.t.) 
		    // which Context is packed. 
		    const unsigned nWrtwCip, 
		    const unsigned int specNum,
		    const unsigned int node)
{
    double total = 0.0;
    VocabIndex wid;

    /*
     * prob summing interrupts sequential processing mode
     */
    Boolean wasRunning = running(false);
    FactoredVocab::TagIter titer(*((FactoredVocab*)&vocab));
    while (titer.next(wid)) {
      const double tmp = LogPtoProb(wordProbBO(wid, context, nWrtwCip, specNum, node));
      // fprintf(stderr,"Prob for word %s = %f\n",vocab.getWord(wid),tmp);
      total += tmp;
    }
    running(wasRunning);
    return total;
}

Prob
FNgram::bgChildProbSum(const VocabIndex *context, 
		       // nWrtwCip: Node number With respect to (w.r.t.) 
		       // which Context is packed. 
		       const unsigned nWrtwCip, 
		       const unsigned int specNum,
		       const unsigned int node)
{
    double total = 0.0;
    VocabIndex wid;

    /*
     * prob summing interrupts sequential processing mode
     */
    Boolean wasRunning = running(false);
    FactoredVocab::TagIter titer(*((FactoredVocab*)&vocab));
    while (titer.next(wid)) {
      total += LogPtoProb(bgChildProbBO(wid, context, nWrtwCip, specNum, node));
    }
    running(wasRunning);
    return total;
}



Boolean
FNgram::wordProbSum()
{
  Boolean rc = true;
  fprintf(stderr,"Checking that all LMs sum to unity for all training contexts\n");
  for (unsigned specNum=0;specNum<fngs.fnSpecArray.size();specNum++) {  
    fprintf(stderr," Checking LM %d\n",specNum);
    const unsigned numParents = fngs.fnSpecArray[specNum].numParents;
    ((FactoredVocab*)&vocab)->setCurrentTagVocab(fngs.fnSpecArray[specNum].child);
    for (int level=numParents;level>=0;level--) {
      fprintf(stderr,"  Checking level %d\n",level);
      FNgramSpecsType::FNgramSpec::LevelIter liter(numParents,level);
      unsigned int node;
      while (liter.next(node)) {
	unsigned iter = 0;

	VocabIndex context[maxNumParentsPerChild+2];
	FNgramSpecsType::FNgramSpec::PSIter 
	  contextIter(fngs.fnSpecArray[specNum].parentSubsets[node],
		      context,
		      fngs.fnSpecArray[specNum].parentSubsets[node].order-1);
	FNgramCount *contextCount;
	fprintf(stderr,"   Checking node 0x%X:",node);
	while ((contextCount = contextIter.next())) {
	  double sum = wordProbSum(context,node,specNum,node);
	  if (fabs(sum - 1.0) > 1e-3) {
	    fprintf(stderr,"WARNING: child in context summed to %f, node = 0x%X, lm = %d\n",
		    sum,node,specNum);
	    cerr << "context: " << (vocab.use(), context) << endl;
	    rc = false;
	  }
	  if (++iter % 1000 == 0) 
	    fprintf(stderr,".");
	}
	fprintf(stderr,"\n");
      }
    }
  }
  fprintf(stderr,"Done checking that LM sums to unity for all training contexts\n");
  return rc;
}

void
FNgram::freeThread()
{
    TLSW_FREE(widMatrixTLS);
    TLSW_FREE(pplFileWordMatrixTLS);
    TLSW_FREE(rescoreFileWordMatrixTLS);
    TLSW_FREE(rescoreFileWstrings);
    TLSW_FREE(rescoreFileWstringsCopy);
}
