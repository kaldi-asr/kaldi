/*
 * FNgramSpecs.cc --
 *	Cross-stream N-gram specification file and parameter structure
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 */

#ifndef _FNgramSpecs_cc_
#define _FNgramSpecs_cc_

#ifndef lint
static char FNgramSpecs_Copyright[] = "Copyright (c) 1995-2012 SRI International.  All Rights Reserved.";
static char FNgramSpecs_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/FNgramSpecs.cc,v 1.24 2014-08-29 21:35:47 frandsen Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <ctype.h>

#include "FNgram.h"
#include "FNgramSpecs.h"

#include "Trie.cc"
#include "Array.cc"
#include "FactoredVocab.h"
#include "Debug.h"
#include "FDiscount.h"
#include "hexdec.h"
#include "wmatrix.h"

#define INSTANTIATE_FNGRAMSPECS(CountT) \
	INSTANTIATE_ARRAY(FNgramSpecs<CountT>::FNgramSpec); \
	INSTANTIATE_ARRAY(FNgramSpecs<CountT>::FNgramSpec::ParentSubset); \
	template class FNgramSpecs<CountT>


// from defined in FNgramLM.cc
// TODO: place all bit routines in separate file.
extern unsigned int bitGather(unsigned int mask,unsigned int bitv);

// defined externally
extern long strtolplusb(const char *nptr, char **endptr, int base);

static inline unsigned int
numBitsSet(register unsigned u) 
{
  register unsigned count=0;
  while (u) {
    count += (u&0x1);
    u >>= 1;
  }
  return count;
}


static inline unsigned int
backoffGraphLevel(unsigned numParents, unsigned parentSubSet)
{
  const unsigned int nbs = numBitsSet(parentSubSet);
  assert (numParents >= nbs);
  return (numParents-nbs);
}

/*
 * return true if, given the mask, we should expand
 * the parentSubset path.
 */
static inline Boolean
expandBackoff(unsigned mask, unsigned parentSubSet)
{
  // true if all zero bits in parentSubSet have
  // a 1 in corresponding position in mask
  return ((~parentSubSet & mask) != 0);
}


template <class CountT> CountT
FNgramSpecs<CountT>::FNgramSpec::ParentSubset::accumulateCounts(FNgramNode* counts)
{
  if (counts == NULL)
    return 0;
  if (counts->numEntries() == 0)
    return counts->value();
  CountT accumulator = 0;
  FNgramNode* child;
  TrieIter<VocabIndex,CountT> iter(*counts);
  VocabIndex wid;

  while ((child = iter.next(wid))) {
    accumulator += accumulateCounts(child);
  }
  // not a leaf node, so destroy old accumulated counts
  counts->value() = accumulator;
  return accumulator;
}


//
// The next set of routines do trie queries, but only use the words
// for which the corresponding bit in 'bits' is on (in some way).
// These sorts of routine should probably live in Trie.{h,cc} rather
// than here.
//



template <class CountT> CountT*
FNgramSpecs<CountT>::FNgramSpec::ParentSubset::findCountSubCtx(const VocabIndex*words,
							       unsigned int bits)
{
  // for word string "w,x,y,z" where "p3=w,p2=x,p1=y,c=z" in normal order,
  // words is of the form
  // variables:   p3 p2 p1 c   
  // bits:        b3 b2 b1 b0
  // word[i],i=   0  1  2  3
  // count tries have p3 root level, then p2, p1, and c at the leaf level.
  // this routine indexes the count trie in ascending words order to go
  // from p3 down to a leaf.
  if (counts == NULL)
    return NULL;

  const unsigned nbs = numBitsSet(bits);
  const unsigned wlen = Vocab::length(words);
  assert (nbs <= wlen);

  unsigned bitNum = wlen-1;
  unsigned wNum = 0;
  FNgramNode* subtrie = counts;
  VocabIndex word[2];
  word[1] = Vocab_None;
  while (wNum < wlen) {
    if (bits & (1<<bitNum)) {
      word[0] = words[wNum];
      if ((subtrie = subtrie->findTrie(word)) == NULL)
	return NULL;
    }
    wNum++;
    bitNum--;
  }
  return subtrie ? &(subtrie->value()) : NULL;
}


template <class CountT> CountT*
FNgramSpecs<CountT>::FNgramSpec::ParentSubset::findCountSubCtxW(const VocabIndex*words,
								VocabIndex word1,
								unsigned int bits)
{
  // same as findCountSubCtx(words,bits) above but we have an extra
  // word1 variable that we query (irrespective of bits) at the end.
  if (counts == NULL)
    return NULL;

  const unsigned nbs = numBitsSet(bits);
  const unsigned wlen = Vocab::length(words);
  assert (nbs <= wlen);

  unsigned bitNum = wlen-1;
  unsigned wNum = 0;
  FNgramNode* subtrie = counts;
  VocabIndex word[2];
  word[1] = Vocab_None;
  while (wNum < wlen) {
    if (bits & (1<<bitNum)) {
      word[0] = words[wNum];
      if ((subtrie = subtrie->findTrie(word)) == NULL)
	return NULL;
    }
    wNum++;
    bitNum--;
  }
  return subtrie ? subtrie->find(word1) : NULL;
}


template <class CountT> CountT*
FNgramSpecs<CountT>::FNgramSpec::ParentSubset::findCountRSubCtx(const VocabIndex*words,
								unsigned int bits)
{
  // same as findCountSubCtx except that words are in reversed order. Rather than
  // reversing them again, this routine does the trie lookup in reverse word[] order of
  // what findCountSubCtx does.
  // words is of the form
  // variables:   p3 p2 p1 c 
  // bits:        b3 b2 b1 b0
  // word[i],i=   3  2  1  0
  // In a model p(c|p1,p2,p3) where bits correspond to the parent number,
  // count tries have p3 root trie level, then p2, p1, and c at the leaf level.
  // this routine indexes the count trie in descending words order to go
  // from p3 down to the root.
  if (counts == NULL)
    return NULL;

  const unsigned wlen = Vocab::length(words);
  assert (numBitsSet(bits) <= wlen);

  int wNum = wlen-1;
  FNgramNode* subtrie = counts;
  VocabIndex word[2];
  word[1] = Vocab_None;
  while (wNum >= 0) {
    if (bits & (1<<wNum)) {
      word[0] = words[wNum];
      if ((subtrie = subtrie->findTrie(word)) == NULL)
	return NULL;
    }
    wNum--;
  }
  return subtrie ? &(subtrie->value()) : NULL;
}


template <class CountT> CountT*
FNgramSpecs<CountT>::FNgramSpec::ParentSubset::findCountRSubCtxW(const VocabIndex*words,
								 VocabIndex word1,
								 unsigned int bits)
{
  // same as findCountRSubCtx, but with a word1 variable.
  if (counts == NULL)
    return NULL;

  const unsigned wlen = Vocab::length(words);
  assert (numBitsSet(bits) <= wlen);

  int wNum = wlen-1;
  FNgramNode* subtrie = counts;
  VocabIndex word[2];
  word[1] = Vocab_None;
  while (wNum >= 0) {
    if (bits & (1<<wNum)) {
      word[0] = words[wNum];
      if ((subtrie = subtrie->findTrie(word)) == NULL)
	return NULL;
    }
    wNum--;
  }
  return subtrie ? subtrie->find(word1) : NULL;
}



template <class CountT> CountT*
FNgramSpecs<CountT>::FNgramSpec::ParentSubset::findCountR(const VocabIndex*words,
							  VocabIndex word1)
{
  // like findCount, but looks up words in reverse order.
  if (counts == NULL)
    return NULL;

  const unsigned wlen = Vocab::length(words);

  int wNum = wlen-1;
  FNgramNode* subtrie = counts;
  VocabIndex word[2];
  word[1] = Vocab_None;
  while (wNum >= 0) {
    word[0] = words[wNum];
    if ((subtrie = subtrie->findTrie(word)) == NULL)
      return NULL;
    wNum--;
  }
  return subtrie ? subtrie->find(word1) : NULL;
}


/***********************************************************************************
 *
 *
 *                 The general backoff node selection strategy
 *
 *
 ***********************************************************************************/

template <class CountT> double
FNgramSpecs<CountT>::FNgramSpec::ParentSubset::
backoffValueRSubCtxW(VocabIndex word1,       // the word
		     const VocabIndex*words, // the context
		     unsigned int nWrtwCip, // node w.r.t. which context is packed
		     BackoffNodeStrategy parentsStrategy, // strategy of parent
		     FNgram& fngram,        // the related fngram
		     unsigned int specNum,  // which LM of the fngram
		     unsigned int node)     // which node of the LM
{
  // we should never select a path with null counts
  // this could occur because of backoff constraints
  if (counts == NULL)
    return -1e200; // TODO: return NaN or Inf to signal this condition (and use #define)
  
  switch (parentsStrategy) {
  case CountsNoNorm: {
    const unsigned int bits = bitGather(nWrtwCip,node);
    CountT *c = findCountRSubCtxW(words,word1,bits);
    if (c)
      return (double)*c;
    else
      return 0.0;
  }
    break;
  case CountsSumCountsNorm: {
    // choose the node with the max normalized counts,
    // this is equivalent to choosing the largest
    // maximum-likelihod (ML) prob (a max MI criterion).

    // UGLYNESS: get the numerator and denominator here.
    // just a copy of code from backoffValueRSubCtxW() 
    const unsigned int bits = bitGather(nWrtwCip,node);
    const unsigned wlen = Vocab::length(words);
    int wNum = wlen-1;
    FNgramNode* subtrie = counts;
    VocabIndex word[2];
    word[1] = Vocab_None;
    while (wNum >= 0) {
      if (bits & (1<<wNum)) {
	word[0] = words[wNum];
	if ((subtrie = subtrie->findTrie(word)) == NULL)
	  return 0.0;
      }
      wNum--;
    }
    if (!subtrie)
      return 0.0;
    CountT *cnt = subtrie->find(word1);
    if (!cnt)
      return 0.0;
    if (*cnt == 0)
      return 0.0;
    // this might not be the case depending on if kndiscount is used
    // since it significantly modifies the count tries.
    // TODO: should indicate a warning here!!!
    if (*cnt > subtrie->value()) {
      fprintf(stderr,"Warning: leaf count greater than node sum, cnt = %d, sum = %d\n", 
	      (unsigned)*cnt,
	      (unsigned)subtrie->value());
    }
    // assert (*cnt <= subtrie->value());
    double numerator = *cnt;
    double denominator = subtrie->value();
    // note that if sumCounts() wasn't called, denominator might be zero.
    // we assume here that denominator can not be zero if *cnt > 0.
    return numerator/denominator;
  }
    break;
  case CountsSumNumWordsNorm: {
    // normalize by the number of words that have occured.

    // more UGLYNESS:
    // again, a copy of code from backoffValueRSubCtxW() 
    const unsigned int bits = bitGather(nWrtwCip,node);
    const unsigned wlen = Vocab::length(words);
    int wNum = wlen-1;
    FNgramNode* subtrie = counts;
    VocabIndex word[2];
    word[1] = Vocab_None;
    while (wNum >= 0) {
      if (bits & (1<<wNum)) {
	word[0] = words[wNum];
	if ((subtrie = subtrie->findTrie(word)) == NULL)
	  return 0.0;
      }
      wNum--;
    }
    if (!subtrie)
      return 0.0;
    CountT *cnt = subtrie->find(word1);
    if (!cnt)
      return 0.0;
    if (*cnt == 0)
      return 0.0;
    double denominator = subtrie->numEntries();
    double numerator = *cnt;
    // assume that denominator can't be zero if we got count for word1
    return numerator/denominator;
  }
    break;
  case CountsProdCardinalityNorm:
  case CountsSumCardinalityNorm:
  case CountsSumLogCardinalityNorm: {
    const unsigned int bits = bitGather(nWrtwCip,node);
    const unsigned wlen = Vocab::length(words);
    int wNum = wlen-1;
    FNgramNode* subtrie = counts;
    VocabIndex word[2];
    word[1] = Vocab_None;
    while (wNum >= 0) {
      if (bits & (1<<wNum)) {
	word[0] = words[wNum];
	if ((subtrie = subtrie->findTrie(word)) == NULL)
	  return 0.0;
      }
      wNum--;
    }
    if (!subtrie)
      return 0.0;
    CountT *cnt = subtrie->find(word1);
    if (!cnt)
      return 0.0;
    if (*cnt == 0)
      return 0.0;
    double numerator = *cnt;
    double denominator;
    if (parentsStrategy == CountsProdCardinalityNorm) 
      denominator = prodCardinalities;
    else if (parentsStrategy == CountsSumLogCardinalityNorm) 
      denominator = sumCardinalities;
    else 
      denominator = sumLogCardinalities;
    return numerator/denominator;
  }
    break;
  case BogNodeProb: {
    // chose the one with the maximum smoothed probability
    // this is also a max MI criterion.
    return fngram.wordProbBO(word1,words,nWrtwCip,specNum,node);
  }
  default:
    fprintf(stderr,"Error: unknown backoff strategy, value = %d\n",parentsStrategy);
    abort();
    break;
  }
  return -1.0;
}  // of backoffValueRSubCtxW



template <class CountT> Boolean
FNgramSpecs<CountT>::FNgramSpec::LevelIter::next(unsigned int&node) {
  for (;state<numNodes;state++) {
    // if (numBitsSet(state) == (numParents - level)) {
    if (numBitsSet(state) == level) {
      node = state++;
      return true;
    }
  }
  return false;
}


/*
 * parent iter code
 */


template <class CountT>
FNgramSpecs<CountT>::FNgramSpec::BGParentIter::BGParentIter(const unsigned int _numParents,
					    const unsigned int _homeNode)
  : numParents(_numParents),numNodes(1<<_numParents),homeNode(_homeNode),
    numBitsSetOfHomeNode(numBitsSet(homeNode))
{
  init();
}


template <class CountT> Boolean
FNgramSpecs<CountT>::FNgramSpec::BGParentIter::next(unsigned int&node) 
{
  for (;state<numNodes;state++) {
    // all bits in homeNode must also be on in parent=state
    if (((homeNode & state) == homeNode) &&
	(numBitsSet(state) == (numBitsSetOfHomeNode+1))) {
      node = state++;
      return true;
    }
  }
  return false;
}


/*
 * grandparent iter code
 */


template <class CountT>
FNgramSpecs<CountT>::FNgramSpec::BGGrandParentIter::BGGrandParentIter(const unsigned int _numParents,
					    const unsigned int _homeNode,
					    const unsigned int _great)
  : numParents(_numParents),numNodes(1<<_numParents),homeNode(_homeNode),
    numBitsSetOfHomeNode(numBitsSet(homeNode)),great(_great)
{
  init();
}


template <class CountT> Boolean
FNgramSpecs<CountT>::FNgramSpec::BGGrandParentIter::next(unsigned int&node) 
{
  for (;state<numNodes;state++) {
    // all bits in homeNode must also be on in parent=state
    if (((homeNode & state) == homeNode) &&
	(numBitsSet(state) == (numBitsSetOfHomeNode+2+great))) {
      node = state++;
      return true;
    }
  }
  return false;
}



/*
 * ancestor iter code
 */


template <class CountT>
FNgramSpecs<CountT>::FNgramSpec::BGAncestorIter::BGAncestorIter(const unsigned int _numParents,
					    const unsigned int _homeNode)
  : numParents(_numParents),numNodes(1<<_numParents),homeNode(_homeNode),
    numBitsSetOfHomeNode(numBitsSet(homeNode))
{
  init();
}


template <class CountT> Boolean
FNgramSpecs<CountT>::FNgramSpec::BGAncestorIter::next(unsigned int&node) {
  for (;state<numNodes;state++) {
    // all bits in homeNode must also be on in parent=state
    if ((homeNode & state) == homeNode) {
      node = state++;
      return true;
    }
  }
  return false;
}


/*
 * child iter code
 */


template <class CountT>
FNgramSpecs<CountT>::FNgramSpec::BGChildIter::BGChildIter(const unsigned int _numParents,
					  const unsigned int _homeNode)
  : numParents(_numParents),numNodes(1<<_numParents),homeNode(_homeNode),
    numBitsSetOfHomeNode(numBitsSet(homeNode))
{
  init();
}

template <class CountT>  Boolean
FNgramSpecs<CountT>::FNgramSpec::BGChildIter::next(unsigned int&node) {
  for (;state>=0;state--) {
    // all bits in child=state must also be on in homeNode
    if (((state & (int)homeNode) == state) &&
	((1+numBitsSet(state)) == numBitsSetOfHomeNode)) {
      node = state--;
      return true;
    }
  }
  return false;
}


/*
 * child iter with constraints code
 */


template <class CountT>
FNgramSpecs<CountT>::FNgramSpec::BGChildIterCnstr::BGChildIterCnstr
(
 const unsigned int _numParents,
 const unsigned int _homeNode,
 const unsigned int _bo_constraints
 )
  : numParents(_numParents),numNodes(1<<_numParents),homeNode(_homeNode),
    bo_constraints(_bo_constraints),
    numBitsSetOfHomeNode(numBitsSet(homeNode))
{
  init();
}

template <class CountT>  Boolean
FNgramSpecs<CountT>::FNgramSpec::BGChildIterCnstr::next(unsigned int&node) {
  for (;state>=0;state--) {
    if ((// all bits in child=state must also be on in homeNode
	 ((state & (int)homeNode) == state) &&
	 ((1+numBitsSet(state)) == numBitsSetOfHomeNode))
	&&
	(// child can validly come from a parent with current BO constraints
	 ~state & 
	 (bo_constraints
	  & homeNode)))
	{
	  node = state--;
	  return true;
	}
  }
  return false;
}


/*
 * grand child iter code
 */


template <class CountT>
FNgramSpecs<CountT>::FNgramSpec::BGGrandChildIter::BGGrandChildIter(const unsigned int _numParents,
							       const unsigned int _homeNode,
							       const unsigned int _great)
  : numParents(_numParents),numNodes(1<<_numParents),homeNode(_homeNode),
    numBitsSetOfHomeNode(numBitsSet(homeNode)),great(_great)
{
  init();
}

template <class CountT>  Boolean
FNgramSpecs<CountT>::FNgramSpec::BGGrandChildIter::next(unsigned int&node) {
  for (;state>=0;state--) {
    // all bits in child=state must also be on in homeNode
    if (((state & (int)homeNode) == state) &&
	((great+2+numBitsSet(state)) == numBitsSetOfHomeNode)) {
      node = state--;
      return true;
    }
  }
  return false;
}


/*
 * descendant iter code
 */


template <class CountT>
FNgramSpecs<CountT>::FNgramSpec::BGDescendantIter::BGDescendantIter(const unsigned int _numParents,
					  const unsigned int _homeNode)
  : numParents(_numParents),numNodes(1<<_numParents),homeNode(_homeNode),
    numBitsSetOfHomeNode(numBitsSet(homeNode))
{
  init();
}

template <class CountT>  Boolean
FNgramSpecs<CountT>::FNgramSpec::BGDescendantIter::next(unsigned int&node) {
  for (;state>=0;state--) {
    // all bits in child=state must also be on in homeNode
    if ((state & (int)homeNode) == state) {
      node = state--;
      return true;
    }
  }
  return false;
}



/*
 ****************************************
 * main FNgramSpecs constructor         *
 ****************************************
 */

template <class CountT>
FNgramSpecs<CountT>::FNgramSpecs(File& f,
				 FactoredVocab& fv,
				 unsigned debuglevel)
  : Debug(debuglevel), fvocab(fv)
{
  if (f.error())
    return;

  // parse the input file which we now assume to be open and valid.
  char *line;

  // position of word is always zero
  *tagPosition.insert(FNGRAM_WORD_TAG_STR) = FNGRAM_WORD_TAG_POS;
  // this initialization assumes that FNGRAM_WORD_TAG_POS == 0
  unsigned nextPosition = 1;

  // get number of CS Ngrams
  line = f.getline();
  // printf("line = (%s)\n",line);
  if (line == 0) {
    // not sure what to do here with these errors, so we just exit with a message.
    f.position() << "Error: File::getline() returned 0 when reading FLM spec file\n";
    exit(-1);
  }
  register char *p = line;
  // skip space to next token
  while (*p && isspace(*p)) p++;

  //////////////////////////////////////////////////////////////
  // get number of LM specs that are being given here
  char* endptr = p;
  int n_csngrams = (int) strtol(p,&endptr,0);
  if (endptr == p) {
    f.position() << "Error: couldn't form int for number of factored LMs in when reading FLM spec file\n";
    exit(-1);
  }
  p = endptr;

  // each FLM gram spec is on a diff line.
  for (int i=0;i<n_csngrams;i++) {

    // Parse a chunk of text of the form: 
    // W : 4 W(-1) M(0) S(0) R(0) count_filename lm_filename num_node_specs
    //    <node_spec_1> <node_constraint_1> [optional_node_options]
    //    <node_spec_2> <node_constraint_2> [optional_node_options]
    //    ...
    //    <node_spec_N> <node_constraint_N> [optional_node_options]
    //

    char *token;
    char *parse_state;
    char tmp;
    line = f.getline();
    // TODO: allow this to be multi-line
    if (line == 0) {
      f.position() << "Error: File::getline() returned 0 when reading FLM spec file\n";
      exit(-1);
    }
    // skip blanks
    p = line; while (*p && isspace(*p)) p++;
    if (!isalnum(*p)) {
      f.position() << "Error: expecting child spec in FLM in when reading FLM spec file\n";
      exit(-1);
    }

    ////////////////////////////////////    
    // pull out the name of the child
    if (!*p) {
      f.position() << "Error: couldn't get child name when reading factor spec file\n";
      exit(-1);      
    }
    token = p;
    do { p++; } while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != ':');
    tmp = *p; *p = '\0'; // C string parsing is beautiful, isn't it.
    fnSpecArray[i].child = strdup(token); // TODO: finish destructor and free all strdups.
    *p = tmp;
    // insert the tag
    Boolean found;
    unsigned *pos_p = tagPosition.insert(fnSpecArray[i].child,found);
    if (!found)
      *pos_p = nextPosition++;
    fnSpecArray[i].childPosition = *pos_p;
    // skip to next token
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == ':')) p++;

    ////////////////////////////////////
    // get num parents
    if (!*p) {
      f.position() << "Error: couldn't get number parents when reading factor spec file\n";
      exit(-1);      
    }
    endptr = p;
    fnSpecArray[i].numParents = (int) strtol(p,&endptr,0);
    if (endptr == p) {
      f.position() << "Error: couldn't form int for number FN-grams in when reading FLM spec file\n";
      exit(-1);
    }
    p = endptr;
    if (fnSpecArray[i].numParents > maxNumParentsPerChild) {
      f.position()
	  << "Error: number parents must not be negative or greater than " 
	  << maxNumParentsPerChild << "\n"; 
    }
    // skip space to next token
    while (*p && isspace(*p)) p++;

    fnSpecArray[i].numSubSets = 1<<fnSpecArray[i].numParents;

    for (unsigned j = 0; j < fnSpecArray[i].numParents; j++) {

      ////////////////////////////////////
      // get name of parent
      if (!*p) {
	f.position() << "Error: couldn't get parent name when reading factor spec file\n";
	exit(-1);      
      }
      token = p;
      do { p++; } while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '(');
      tmp = *p; *p = '\0';
      fnSpecArray[i].parents[j] = strdup(token);
      *p = tmp;
      // insert the tag
      Boolean found;
      pos_p = tagPosition.insert(fnSpecArray[i].parents[j],found);
      if (!found)
	*pos_p = nextPosition++;
      fnSpecArray[i].parentPositions[j] = *pos_p;
      // skip to next token  (NOTE: this would accept multiple '(' chars.)
      while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '(')) p++;

      ////////////////////////////////////
      // get offset
      if (!*p) {
	f.position() << "Error: couldn't get parent offset when reading factor spec file\n";
	exit(-1);      
      }
      endptr = p;
      fnSpecArray[i].parentOffsets[j] = (int) strtol(p,&endptr,0);
      if (endptr == p) {
	f.position() << "Error: couldn't form int for number FN-grams in when reading FLM spec file\n";
	exit(-1);
      }

      // Future Language Model Support: allow offsets to come from the future as well as the past
      // if (fnSpecArray[i].parentOffsets[j] > 0) {
      //    f.position() << "Error: can't have positive parent offset in structure file\n";
      //    exit(-1);
      // }

      if (fnSpecArray[i].parentOffsets[j] == 0 && fnSpecArray[i].parentPositions[j] == fnSpecArray[i].childPosition) {
	f.position() << "Error: parent and child can not be the same\n";
	exit(-1);
      }

      // make sure that the same parent is not specified twice.
      for (unsigned l = 0; l < j; l++) {
	if ((fnSpecArray[i].parentPositions[j] == fnSpecArray[i].parentPositions[l]) &&
	    (fnSpecArray[i].parentOffsets[j] == fnSpecArray[i].parentOffsets[l])) {
	  f.position() << "Error: cannot specify same parent more than once\n";
	  exit(-1);
	}
      }

      p = endptr;
      // skip to next token (NOTE: this would accept multiple ')' chars.)
      while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == ')')) p++;
    }

    ////////////////////////////////////
    // get Count file name
    while (*p && isspace(*p)) p++;
    if (!(*p)) {
      f.position() << "Error: couldn't get count file name when reading factor spec file\n";
      exit(-1);      
    }
    token = p;
    do { p++; } while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n');
    tmp = *p; *p = '\0';    
    fnSpecArray[i].countFileName = strdup(token);
    *p = tmp;

    ////////////////////////////////////
    // get LM file name
    // skip space to next token
    while (*p && isspace(*p)) p++;
    if (!*p) {
      f.position() << "Error: couldn't get LM file name when reading factor spec file\n";
      exit(-1);
    }
    token = p;
    do { p++; } while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n');
    tmp = *p; *p = '\0';    
    fnSpecArray[i].lmFileName = strdup(token);
    *p = tmp;

    ////////////////////////////////////
    // get number of nodes that have node specs
    // skip space to next token
    while (*p && isspace(*p)) p++;
    if (!*p) {
      f.position() << "Error: couldn't get num node specs name when reading factor spec file\n";
      exit(-1);      
    }
    endptr = p;
    int numNodeSpecs = 0;
    numNodeSpecs = (int) strtol(p,&endptr,0);
    if (endptr == p || numNodeSpecs < 0) {
      f.position() << "Error: couldn't form unsigned int for number node specs when reading FLM spec file\n";
      exit(-1);
    }

    // finally! done with this line, now get node specs

    const unsigned numSubSets = 1U<<fnSpecArray[i].numParents;

    // next set of numNodeSpecs lines contain node specs
    for (unsigned j = 0; j < (unsigned)numNodeSpecs; j++) {
      // line should have the form
      // NODE_NUM BACKOFFCONSTRAINT <options>
      // options include what ngram-count.cc uses on comand line for
      // discount options. I've given up on any (even slighty) fancy 
      // C string parsing for now, so this is just a string of tokens which
      // are parsed in a very simple way.
      // TODO: do a proper multi-line tokenizer here.

      line = f.getline();
      // printf("line = (%s)\n",line);
      if (line == 0) {
	f.position() << "Error: File::getline() returned 0 when readin FLM spec file\n";
	exit(-1);
      }
      VocabString tokens[128];

      unsigned howmany = Vocab::parseWords(line,tokens,128);

      if (howmany < 2) {
	f.position() << "Error: specifier must at least specify node id and back-off constraint\n";
	exit(-1);
      }
      unsigned tok = 0;
      
      // get node id
      int nodeId = 0x0;
      Boolean success;
      nodeId = (int) fnSpecArray[i].parseNodeString((char*)tokens[tok],success);
      if (!success) {
	f.position() << "Error: couldn't form BG node specifier in string (" <<
	  tokens[tok] << ")\n";
	exit(-1);
      }
      if (nodeId < 0) {
	f.position() << "Error: couldn't form unsigned int in " <<
	  tokens[tok] << "for node specifier when reading factored spec file\n";
	exit(-1);
      }
      if ((unsigned)nodeId >= numSubSets) {
	fprintf(stderr,"Error: node specifier must be between 0x0 and 0x%x inclusive\n",
		numSubSets-1);
	exit(-1);
      }

      tok++;

      // get backoff constraint for this node
      fnSpecArray[i].parentSubsets[nodeId].backoffConstraint
	= fnSpecArray[i].parseNodeString((char*)tokens[tok],success);
      if (!success) {
	f.position() << "Error: couldn't form BG node constraint in string (" <<
	  tokens[tok] << ")\n";
	exit(-1);
      }
      tok++;

      // 
      // Current set of Node Options
      // 
      // gtmin [num]
      // gtmax [num]
      // gt [fileName string]
      // cdiscount [double]
      // ndiscount []
      // wbdiscount []
      // kndiscount []
      // ukndiscount []
      // kn-counts-modified []
      // kn-counts-modify-at-end []
      // kn [fileName string]
      // interpolate []
      // write [fileName string]
      // strategy [option]
      //    where [option] is one of:
      //            counts_no_norm
      //            counts_sum_counts_norm
      //            counts_sum_num_words_norm
      //            counts_prod_card_norm
      //            counts_sum_card_norm
      //            counts_sum_log_card_norm
      //            bog_node_prob
      // 

    startGetOptions:
      for (;tok<howmany;tok++) {
	if (strcmp(tokens[tok],"gtmin") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: gtmin argument needs a value"
		    "reading factored spec file\n");
	    exit(-1);
	  }
	  tok++;
	  // @kw false positive: SV.FMT_STR.SCAN_FORMAT_MISMATCH.BAD
	  if (sscanf(tokens[tok],"%u",&fnSpecArray[i].parentSubsets[nodeId].gtmin) != 1){
	    fprintf(stderr,"Error: gtmin argument needs integer value");
	    exit(-1);
	  }
	} else if (strcmp(tokens[tok],"gtmax") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: gtmax argument needs a value"
		    "reading factored spec file\n");
	    exit(-1);
	  }
	  tok++;
	  if (sscanf(tokens[tok],"%u",&fnSpecArray[i].parentSubsets[nodeId].gtmax) != 1){
	    fprintf(stderr,"Error: gtmax argument needs integer value");
	    exit(-1);
	  }
	} else if (strcmp(tokens[tok],"gt") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: gt argument needs a value"
		    "reading factored spec file\n");
	    exit(-1);
	  }
	  tok++;
	  delete [] fnSpecArray[i].parentSubsets[nodeId].gtFile;
	  fnSpecArray[i].parentSubsets[nodeId].gtFile = strdup(tokens[tok]);
	} else if (strcmp(tokens[tok],"cdiscount") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: cdiscount argument needs a value");
	    exit(-1);
	  }
	  tok++;
	  double tmp;
	  char *endptr;
	  tmp = strtod(tokens[tok],&endptr);
	  if (endptr == tokens[tok]) {
	    fprintf(stderr,"Error: cdiscount argument (%s) should be floating point value",
		    tokens[tok]);
	    exit(-1);
	  }
	  fnSpecArray[i].parentSubsets[nodeId].cdiscount = tmp;
	} else if (strcmp(tokens[tok],"ndiscount") == 0) {
	  fnSpecArray[i].parentSubsets[nodeId].ndiscount = true;
	} else if (strcmp(tokens[tok],"wbdiscount") == 0) {
	  fnSpecArray[i].parentSubsets[nodeId].wbdiscount = true;
	} else if (strcmp(tokens[tok],"kndiscount") == 0) {
	  fnSpecArray[i].parentSubsets[nodeId].kndiscount = true;
	} else if (strcmp(tokens[tok],"ukndiscount") == 0) {
	  fnSpecArray[i].parentSubsets[nodeId].ukndiscount = true;
	} else if (strcmp(tokens[tok],"kn-counts-modified") == 0) {
	  fnSpecArray[i].parentSubsets[nodeId].knCountsModified = true;
	} else if (strcmp(tokens[tok],"kn-counts-modify-at-end") == 0) {
	  fnSpecArray[i].parentSubsets[nodeId].knCountsModifyAtEnd= true;
	} else if (strcmp(tokens[tok],"kn-count-parent") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: kn-count-parent argument needs a parent specifier\n");
	    exit(-1);
	  }
	  tok++;
	  unsigned par = fnSpecArray[i].parseNodeString((char*)tokens[tok],success);
	  if (!success) {
	    fprintf(stderr,"Error: kn-count-parent argument invalid\n");
	    exit(-1);
	  }
	  fnSpecArray[i].parentSubsets[nodeId].knCountParent = par;
	} else if (strcmp(tokens[tok],"kn") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: kn argument needs a value"
		    "reading factored spec file\n");
	    exit(-1);
	  }
	  tok++;
	  delete [] fnSpecArray[i].parentSubsets[nodeId].knFile;
	  fnSpecArray[i].parentSubsets[nodeId].knFile = strdup(tokens[tok]);
	} else if (strcmp(tokens[tok],"interpolate") == 0) {
	  fnSpecArray[i].parentSubsets[nodeId].interpolate = true;
	} else if (strcmp(tokens[tok],"write") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: write argument needs a value"
		    "reading factored spec file\n");
	    exit(-1);
	  }
	  tok++;
	  delete [] fnSpecArray[i].parentSubsets[nodeId].writeFile;
	  fnSpecArray[i].parentSubsets[nodeId].writeFile = strdup(tokens[tok]);
	} else if (strcmp(tokens[tok],"strategy") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: strategy argument needs a value"
		    "reading factored spec file\n");
	    exit(-1);
	  }
	  tok++;
	  if (strcmp(tokens[tok],"counts_no_norm") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffStrategy
	      = CountsNoNorm;
	  } else if (strcmp(tokens[tok],"counts_sum_counts_norm") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffStrategy
	      = CountsSumCountsNorm;
	  } else if (strcmp(tokens[tok],"counts_sum_num_words_norm") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffStrategy
	      = CountsSumNumWordsNorm;
	  } else if (strcmp(tokens[tok],"counts_prod_card_norm") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffStrategy
	      = CountsProdCardinalityNorm;
	  } else if (strcmp(tokens[tok],"counts_sum_card_norm") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffStrategy
	      = CountsSumCardinalityNorm;
	  } else if (strcmp(tokens[tok],"counts_sum_log_card_norm") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffStrategy
	      = CountsSumLogCardinalityNorm;
	  } else if (strcmp(tokens[tok],"bog_node_prob") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffStrategy
	      = BogNodeProb;
	  } else {
	    fprintf(stderr,"Error: unknown strategy argument (%s) when "
		  "reading factored spec file\n",tokens[tok]);
	    exit(-1);
	  }
	} else if (strcmp(tokens[tok],"combine") == 0) {
	  if (tok+1==howmany) {
	    fprintf(stderr,"Error: combine argument needs a value "
		    "reading factored spec file\n");
	    exit(-1);
	  }
	  tok++;
	  if (strcmp(tokens[tok],"max") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffCombine
	      = MaxBgChild;
	  } else if (strcmp(tokens[tok],"min") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffCombine
	      = MinBgChild;
	  } else if ((strcmp(tokens[tok],"avg") == 0) ||
		     (strcmp(tokens[tok],"mean") == 0)) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffCombine
	      = AvgBgChild;
	  } else if (strcmp(tokens[tok],"wmean") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffCombine
	      = WmeanBgChild;
	    // next set of tokens must have a combination of (node_spec, weight)
	    // for each child.
	    // we compute this below, but we compute it here since we don't
	    // have the quantity numBGchildren yet.
	    typename FNgramSpec::BGChildIterCnstr
	      citer(fnSpecArray[i].numParents,nodeId,fnSpecArray[i].parentSubsets[nodeId].backoffConstraint); 
	    unsigned int numChildrenUsed = 0;
	    for (unsigned child;citer.next(child);) {
	      numChildrenUsed++;
	    }
	    if (tok+2*numChildrenUsed >= howmany) {
	      f.position() << "Error: combine wmean needs " << numChildrenUsed << 
		" node & weight pairs, one for each child\n";
	      exit(-1);
	    }
	    // TODO: add to destructor
	    LogP2 *wmean = new LogP2[numChildrenUsed];
	    for (unsigned cnum=0;cnum<numChildrenUsed;cnum++) { wmean[cnum] = 0.0; }
	    tok++;
	    for (unsigned cnum=0;cnum<numChildrenUsed;cnum++) {
	      double value;
	      unsigned int childSpec = fnSpecArray[i].parseNodeString((char*)tokens[tok],success);
	      if (!success) {
		f.position() << "Error: combine wmean invalid node specifier\n";
		exit(-1);
	      }
	      tok++;
	      char *endptr;
	      value = strtod(tokens[tok],&endptr);
	      if (endptr == tokens[tok] || value < 0.0) {
		f.position() << "Error: combine wmean invalid weight value\n";
		exit(-1);
	      }
	      citer.init();
	      unsigned int cpos = 0;
	      for (unsigned child;citer.next(child);) {
		if (!(~child & 
		    (fnSpecArray[i].parentSubsets[nodeId].backoffConstraint 
		     & nodeId)))
		  continue;
		if (child == childSpec)
		  break;
		cpos++;
	      }
	      if (cpos == numChildrenUsed) {
		f.position() << "Error: combine wmean, invalid child node given\n";
		exit(-1);
	      }
	      // load them in the array in the order that they will
	      // be encountered when doing a child iter.
	      wmean[cpos] = value;
	      tok++;
	    }
	    double sum = 0;
	    for (unsigned cnum=0;cnum<numChildrenUsed;cnum++) { 
	      sum += wmean[cnum];
	    }
	    // normalize and convert to logp
	    for (unsigned cnum=0;cnum<numChildrenUsed;cnum++) 
	      { wmean[cnum] = ProbToLogP(wmean[cnum]) - ProbToLogP(sum); }
	    fnSpecArray[i].parentSubsets[nodeId].wmean = wmean;
	  } else if (strcmp(tokens[tok],"sum") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffCombine
	      = SumBgChild;
	  } else if (strcmp(tokens[tok],"prod") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffCombine
	      = ProdBgChild;
	  } else if (strcmp(tokens[tok],"gmean") == 0) {
	    fnSpecArray[i].parentSubsets[nodeId].backoffCombine
	      = GmeanBgChild;
	  } else {
	    fprintf(stderr,"Error: unknown combine argument (%s) when "
		  "reading factored spec file\n",tokens[tok]);
	    exit(-1);
	  }
	} else if ((strcmp(tokens[tok],"\\") == 0) &&
		   tok == (howmany-1)) {
	  // do poor man's next line parsing.
	  line = f.getline();
	  if (line == 0) {
	    f.position() << "Error: File::getline() returned 0 when reading FLM spec file\n";
	    exit(-1);
	  }
	  howmany = Vocab::parseWords(line,tokens,128);
	  tok = 0;
	  goto startGetOptions;
	} else {
	  fprintf(stderr,"Error: unknown argument (%s) when"
		  "reading factored spec file\n",tokens[tok]);
	  exit(-1);
	}
      }
      if (fnSpecArray[i].parentSubsets[nodeId].backoffCombine == SumBgChild
	  &&
	  fnSpecArray[i].parentSubsets[nodeId].interpolate) {
	f.position() << "WARNING: using 'interpolate' and 'combine sum' together\n";
      }
    }

    if (debug(DEBUG_EXTREME)) {
      // debug all the iterators.

      for (unsigned level = fnSpecArray[i].numParents; (int)level >= 0; level--) {
	typename FNgramSpec::LevelIter iter(fnSpecArray[i].numParents,level);
	fprintf(stderr, "level 0x%X:",level);
	unsigned int node;
	while (iter.next(node)) {
	  fprintf(stderr, " node 0x%X,",node);
	}
	fprintf(stderr, "\n");
      }
      for (unsigned node = 0; node < numSubSets; node++) {      
	fprintf(stderr, "node 0x%X\n",node);

	typename FNgramSpec::BGParentIter piter(fnSpecArray[i].numParents,node);
	for (unsigned parent=0;piter.next(parent);) {
	  fprintf(stderr, "parent 0x%X,",parent);
	}
	fprintf(stderr, "\n");

	typename FNgramSpec::BGAncestorIter aiter(fnSpecArray[i].numParents,node);
	for (unsigned ancestor;aiter.next(ancestor);) {
	  fprintf(stderr, "ancestor 0x%X,",ancestor);
	}
	fprintf(stderr, "\n");

	typename FNgramSpec::BGChildIter citer(fnSpecArray[i].numParents,node);
	for (unsigned child;citer.next(child);) {
	  fprintf(stderr, "child 0x%X,",child);
	}
	fprintf(stderr, "\n");

	typename FNgramSpec::BGDescendantIter diter(fnSpecArray[i].numParents,node);
	for (unsigned des;diter.next(des);) {
	  fprintf(stderr, "descendant 0x%X,",des);
	}
	fprintf(stderr, "\n");
      }
      fflush(stderr);
    }

    // only create counts objects for nodes that are to be used
    fnSpecArray[i].parentSubsets[numSubSets-1].counts = new FNgramNode;
    fnSpecArray[i].parentSubsets[numSubSets-1].order = numBitsSet(numSubSets-1)+1;
    // descend down the BG, level by level
    for (int level=fnSpecArray[i].numParents;level>=0;level--) {

      typename FNgramSpec::LevelIter liter(fnSpecArray[i].numParents,level);
      Boolean allAreNull = true;

      for (unsigned nodeAtLevel;liter.next(nodeAtLevel);) {

	if (fnSpecArray[i].parentSubsets[nodeAtLevel].counts == NULL)
	  continue;
	allAreNull = false;

	typename FNgramSpec::BGChildIterCnstr 
	  citer(fnSpecArray[i].numParents,nodeAtLevel,fnSpecArray[i].parentSubsets[nodeAtLevel].backoffConstraint);
	unsigned int numChildrenUsed = 0;
	for (unsigned child;citer.next(child);) {
	  if (fnSpecArray[i].parentSubsets[child].counts == NULL) {
	    fnSpecArray[i].parentSubsets[child].counts = new FNgramNode;
	    fnSpecArray[i].parentSubsets[child].order = numBitsSet(child)+1;
	  }
	  numChildrenUsed++;
	  // make sure kn-count-parent has counts itself.
	  if (fnSpecArray[i].parentSubsets[child].knCountParent != ~0x0U) {
	    const unsigned kncp = fnSpecArray[i].parentSubsets[child].knCountParent;
	    if (kncp >= numSubSets || 
		fnSpecArray[i].parentSubsets[kncp].counts == NULL) {
	      f.position() << "Error: kn-counts-parent argument " << HEX << 
		kncp << DEC << " must specify a parent that exists and is in use\n";
	      exit(-1);
	    }
	  }
	}
	// everybody must have a child.
	if (nodeAtLevel > 0 && numChildrenUsed == 0) {
	  fprintf(stderr,"ERROR: backoff graph node 0x%X has no children with backoff constraint 0x%X. Must have at least one child.\n",nodeAtLevel,fnSpecArray[i].parentSubsets[nodeAtLevel].backoffConstraint);
	  exit(-1);
	}
	fnSpecArray[i].parentSubsets[nodeAtLevel].numBGChildren = numChildrenUsed;

      }

      if (allAreNull) {
	// no count object was created
	// NOTE: we might not want to consider this an error, if we for example 
	// want not to backoff to lower levels in backoff graph. In that case,
	// probabilities would become zero, however.
	fprintf(stderr,"ERROR: backoff constraints leave level %d of backoff graph "
		"entirely unexpanded, lower distribution order never reached\n",level);
	exit(-1);
      }
    }

    if (debug(DEBUG_BG_PRINT)) { 
      fprintf(stderr, "Language Model %d --------------\n",i);
      for (int level=fnSpecArray[i].numParents;level>=0;level--) {
	fprintf(stderr, "-- Level %d\n",level);
	typename FNgramSpec::LevelIter liter(fnSpecArray[i].numParents,level);
	for (unsigned nodeAtLevel;liter.next(nodeAtLevel);) {
	  if (fnSpecArray[i].parentSubsets[nodeAtLevel].counts == NULL)
	    continue;
	  fprintf(stderr, "      Node: ");
	  fnSpecArray[i].printNodeString(stderr,nodeAtLevel);
	  fprintf(stderr, " (0x%X), Constraint: ",nodeAtLevel);
	  fnSpecArray[i].printNodeString(stderr,
					 fnSpecArray[i].parentSubsets[nodeAtLevel].backoffConstraint);
	  fprintf(stderr, " (0x%X)\n",fnSpecArray[i].parentSubsets[nodeAtLevel].backoffConstraint);
	  fprintf(stderr, "         %d Children:",fnSpecArray[i].parentSubsets[nodeAtLevel].numBGChildren);
	  typename FNgramSpec::BGChildIterCnstr 
	    citer(fnSpecArray[i].numParents,nodeAtLevel,fnSpecArray[i].parentSubsets[nodeAtLevel].backoffConstraint);
	  Boolean do_comma = false;
	  for (unsigned child;citer.next(child);) {
	    if (fnSpecArray[i].parentSubsets[child].counts != NULL) {
	      // @kw false positive: SV.FMTSTR.GENERIC
	      fprintf(stderr, (do_comma?"; ":" "));
	      fnSpecArray[i].printNodeString(stderr,child);
	      fprintf(stderr, " (0x%X)",child);
	    }
	    do_comma = true;
	  }
	  fprintf(stderr, "\n");
	}
      }
    }

  }
  if (nextPosition > maxNumParentsPerChild) {
    f.position() << "Error: may only have at most " << maxNumParentsPerChild <<
      " distinct tags\n";
    exit(-1);
  }
  
  LHashIter<VocabString,unsigned> tags(tagPosition);
  VocabString tag;
  unsigned *pos;
  char buff[2048];
  while ((pos = tags.next(tag)) != NULL) {

    // TODO: WARNING: should use strncat here.
    buff[0] = '\0';
    fvocab.tagNulls[*pos] = strdup(strcat(strcat(strcat(buff,tag),FNGRAM_WORD_TAG_SEP_STR),FNGRAM_WORD_TAG_NULL_SPEC_STR));
    buff[0] = '\0';
    fvocab.tagUnks[*pos] = strdup(strcat(strcat(strcat(buff,tag),FNGRAM_WORD_TAG_SEP_STR),Vocab_Unknown));
    buff[0] = '\0';
    fvocab.tagSes[*pos] = strdup(strcat(strcat(strcat(buff,tag),FNGRAM_WORD_TAG_SEP_STR),Vocab_SentStart));
    buff[0] = '\0';
    fvocab.tagSss[*pos] = strdup(strcat(strcat(strcat(buff,tag),FNGRAM_WORD_TAG_SEP_STR),Vocab_SentEnd));
    buff[0] = '\0';
    fvocab.tagPauses[*pos] = strdup(strcat(strcat(strcat(buff,tag),FNGRAM_WORD_TAG_SEP_STR),Vocab_Pause));

  }

  if (debug(DEBUG_VERY_VERBOSE))
    printFInfo();

}

template <class CountT>  Boolean
FNgramSpecs<CountT>::FNgramSpec::printNodeString(FILE *f,unsigned int node)
{
  Boolean do_comma = false;
  for (unsigned i=0;i<numParents;i++) {
    if (node & (1<<i)) {
      // @kw false positive: SV.FMT_STR.PRINT_FORMAT_MISMATCH.BAD
      fprintf(f,"%s%s%+d",
	      (do_comma?",":""),
	      parents[i],
	      parentOffsets[i]);
      do_comma = true;
    }
  }
  return true;
}


template <class CountT>  unsigned int
FNgramSpecs<CountT>::FNgramSpec::parseNodeString(char *str, Boolean &success) 
{
  if (str == NULL)
    return 0;
  success = false;
  char *endptr;
  unsigned bits = 0;  

  endptr = str;
  bits = (unsigned) strtolplusb(str,&endptr,0);
  if (endptr != str) {
    success = true;
    return bits;
  }

  char *p = str;
  const unsigned buflen = 2047;
  char buff[buflen+1];
  unsigned par = 1;
  while (*p) {
    // parse tokens of the form TAGNUMBER,TAGNUMBER,TAGNUMBER
    // where TAG is one of the parent names, and number
    // is the negative of the parent position. For example,
    // given parents of the form
    //    W : 3 M(-1) M(-2) M(0) S(-1) S(-2) S(+2)
    // a valid string would be
    //     M1,M0,S1,S2,S+2 
    // and which could correspond to parents
    //
    // M(-1), M(0), S(-1), S(-2), S(+2)
    //
    // and be bit vector
    //      0b11101
    // which is returned.
    
    // get parent
    char *parent = p;
    while (*p && !isdigit(*p) && *p != '+' && *p != '-') {
      p++;
    }
    char tmp = *p;
    *p = '\0';
    strncpy(buff,parent,buflen);
    *p = tmp;

    // get parent position
    // NOTE: this is such that;
    //   W1 == W-1, i.e., both are the previous word.
    //   W+1 == the future word
    // this means that W1 is really an index into the past, rather than the future,
    // meaning that W1 is not the same as W+1. The reason for this is that
    // people will much more commonly use language models with histories in the past rather
    // than histories in the future, so the more common case uses fewer characters.
    bool plusMinusPresent = (*p == '+' || *p == '-');
    int parPos = strtol(p,&endptr,0);
    if (endptr == p) {
      fprintf(stderr,"Can't form integer at parent specifier %d in string (%s)\n",par,str);
      return 0; // doesn't matter what we return here.
    }
    p = endptr;

    // search for parent and position.
    unsigned i;
    for (i=0;i<numParents;i++) {
      if (parentOffsets[i] == (!plusMinusPresent?(-1):(+1))*parPos &&
	  strcmp(parents[i],buff) == 0) {
	// found 
	if (bits & (1<<i)) {
	  // already set, might be an oversite or error by user, give warning
	  fprintf(stderr,
		  "WARNING: parent specifier (%s%d) at position %d given twice in string (%s)\n",
		  buff,(!plusMinusPresent?(-1):(+1))*parPos,par,str);
	}
	bits |= (1<<i);
	break;
      }
    }
    if (i == numParents) {
      fprintf(stderr,"Can't find a valid parent specifier with (%s%d) at position %d in string (%s)\n",
	      buff,(!plusMinusPresent?(-1):(+1))*parPos,par,str);
      return 0; // doesn't matter what we return here.
    } 

    if (*p == ',')
      p++;
    par++;
  }
  success = true;
  return bits;
}




// null for null string
static inline const char *nfns(const char *const str) {
  return (str == NULL ? "NULL" : str);
}
static inline char bchar(const Boolean b) {
  return (b?'T':'F');
}

template <class CountT> void
FNgramSpecs<CountT>::printFInfo()
{
  for (unsigned i = 0; i < fnSpecArray.size(); i++) {
    fprintf(stderr, "----\nchild = [%s], %d parents\n",fnSpecArray[i].child,
	   fnSpecArray[i].numParents);
    for (unsigned j = 0; j < fnSpecArray[i].numParents; j++) {
      fprintf(stderr, "   parent %d = [%s(%d)] = [%d(%d)]\n",j,
	     fnSpecArray[i].parents[j],
	     fnSpecArray[i].parentOffsets[j],
	     fnSpecArray[i].parentPositions[j],
	     fnSpecArray[i].parentOffsets[j]);
    }
    fprintf(stderr, "   count filename = (%s)\n",fnSpecArray[i].countFileName);
    for (unsigned subset = 0; subset < (1U<<fnSpecArray[i].numParents); subset++) {
      if (fnSpecArray[i].parentSubsets[subset].counts != NULL) {
	fprintf(stderr,
	       "   node 0x%X, constraint 0x%X, count object is %s\n",subset,
	       fnSpecArray[i].parentSubsets[subset].backoffConstraint,
	       (fnSpecArray[i].parentSubsets[subset].counts != NULL ? 
		"allocated" : "unallocated"));
	fprintf(stderr,
	       "      gtmin=%d, gtmax=%d, gt=(%s), cdiscount=%f, ndiscount=%c, wbdiscount=%c,\n"
	        "      kndiscount=%c, kn=(%s), interpolate=%c, write=(%s), backoffStrategy=%d\n",
	       fnSpecArray[i].parentSubsets[subset].gtmin,
	       fnSpecArray[i].parentSubsets[subset].gtmax,
	       nfns(fnSpecArray[i].parentSubsets[subset].gtFile),
	       fnSpecArray[i].parentSubsets[subset].cdiscount,
	       bchar(fnSpecArray[i].parentSubsets[subset].ndiscount),
	       bchar(fnSpecArray[i].parentSubsets[subset].wbdiscount),
	       bchar(fnSpecArray[i].parentSubsets[subset].kndiscount),
	       nfns(fnSpecArray[i].parentSubsets[subset].knFile),
	       bchar(fnSpecArray[i].parentSubsets[subset].interpolate),
	       nfns(fnSpecArray[i].parentSubsets[subset].writeFile),
	       fnSpecArray[i].parentSubsets[subset].backoffStrategy);
      }
    }
  }
  fflush(stderr); // TODO: remove since always flushed
}



/*
 *  This breaks each word into factors or streams which presumably is formatted as:
 *
 *     <Tag1>-<factor1>:<Tag2>-<factor2>:...:<TagN>-<factorN>
 * 
 * Load them in the order that was constructed for the tags in FNgramSpecs()
 * constructor. That way, the position in the resulting matrix corresponds
 * to the tag position that several of the objects use such as FNgramStats.
 *
 * Duplicate tags for a word are ignored (the first one encountered on
 * the left is used, possibly issuing a warning). Missing tags for a
 * word are assumed to be <tag>-"<NULL>" I.e., for a morph class, a
 * missing morph tag would be assumed to be "M-<NULL>" where "<NULL>"
 * is the special null word, and "M" is the morph tag. This way, files
 * need not specify the nulls when they exist (and is the reason why NULL
 * is a special word).
 *
 * Any unknown tags are ignored (and a message is printed if the debug level is set
 * accordingly). That way, extra tags not used at the moment can stay in the file
 * without affecting anything.
 *
 * A word tag really should be there for the start/end sentence stuff
 * to work.  (if there is no word stream, a warning message will be
 * issued if debug > 0)
 *
 */
template <class CountT>
unsigned int
FNgramSpecs<CountT>::loadWordFactors(const VocabString *words,
				     WordMatrix& wm,
				     unsigned int max)
{
  if (!words)
    return 0;

  // parse words into factors
  unsigned i;
  for (i = 0; i < max && words[i] != 0; i++) {

    // TODO: call FactoredVocab::loadWordFactor() here instead of
    // code below.

    // assume we don't need to reclaim any of the word_factors[i][j], so
    // just zero out our pointers here.
    ::memset(wm.word_factors[i],0,(maxNumParentsPerChild+1)*sizeof(VocabString));

    VocabString word = words[i];
    VocabString word_p = word;

    if (debug(DEBUG_EVERY_SENTENCE_INFO))
      fprintf(stderr,"Processing word (%s)\n",word);

    // word looks like:
    // <Tag1>-<factor1>:<Tag2>-<factor2>:...:<TagN>-<factorN>
    // if a tag is missing (i.e., just <factor_n>), then we
    // assume it is a FNGRAM_WORD_TAG, which indicates 
    // it is a word.
    
    Boolean tag_assigned = false;
    // make a copy of word, for useful messages
    char word_copy[2048];
    strncpy(word_copy,word,2047);
    Boolean last_factor = false;
    while (!last_factor) {
      char *end_p = (char *)strchr(word_p,FNGRAM_FACTOR_SEPARATOR);
      if (end_p != NULL) {
	// this is not last word
	*end_p = '\0';
      } else 
	last_factor = true;

      if (debug(DEBUG_EVERY_SENTENCE_INFO))
	fprintf(stderr,"working on factor (%s)\n",word_p);
      char *sep_p = (char *)strchr(word_p,FNGRAM_WORD_TAG_SEP);
      if (sep_p == NULL) {
	// no tag, assume word tag. Note, either all words must
	// have a word tag "W-...", or no words can have a word tag. Otherwise,
	// vocab object will assign two different wids for same word, one
	// with wordtag and one without.
	wm.word_factors[i][FNGRAM_WORD_TAG_POS] = word_p;
	tag_assigned = true;
      } else {
	*sep_p = '\0';
	unsigned* pos = tagPosition.find(word_p);
	*sep_p = FNGRAM_WORD_TAG_SEP;
	if (pos == NULL) {
	  if (debug(DEBUG_TAG_WARNINGS)) {
	    fprintf(stderr,"Warning: unknown tag in factor (%s) of word (%s) when parsing file\n",
		    word_p,word_copy);
	  }
	  goto next_tag;
	}
	if (*pos == FNGRAM_WORD_TAG_POS) {
	  // TODO: normalize word so that it either always uses a "W-" tag
	  // or does not use a "W-" tag.
	}
	if (wm.word_factors[i][*pos] != NULL) {
	  if (debug(DEBUG_WARN_DUP_TAG)) 
	    fprintf(stderr,"Warning: tag given twice in word (%s) when parsing "
		    "file. Using first instance.\n",word_copy);
	} else
	  wm.word_factors[i][*pos] = word_p;
	tag_assigned = true;
      }
    next_tag:
      word_p = end_p+1;
    }
    if (!tag_assigned) {
      if (debug(DEBUG_TAG_WARNINGS)) {
	fprintf(stderr,"Warning: no known tags in word (%s), treating all tags as NULLs",
		word_copy);
      }
    }
    // store any nulls
    unsigned j;
    for (j = 0; j < tagPosition.numEntries(); j++) {
      if (wm.word_factors[i][j] == 0) {
	wm.word_factors[i][j] = fvocab.tagNulls[j];
      }
    }
    wm.word_factors[i][j] = 0;      
  }
  if (debug(DEBUG_MISSING_FIRST_LAST_WORD)) {
    // extra check for a word stream. If a word stream does not
    // exist, there is no current way that the start/end sentence
    // stuff will be added.
    if (wm.word_factors[0][FNGRAM_WORD_TAG_POS] == fvocab.tagNulls[FNGRAM_WORD_TAG_POS])
      fprintf(stderr,"Warning: using NULL for first word in sentence\n");
    if (wm.word_factors[i-1][FNGRAM_WORD_TAG_POS] == fvocab.tagNulls[FNGRAM_WORD_TAG_POS])
      fprintf(stderr,"Warning: using NULL for last word in sentence\n");
  }

  if (debug(DEBUG_EVERY_SENTENCE_INFO))
    fprintf(stderr,"%d words in sentence\n",i);
  if (i < max) {
    // zero out last one
    ::memset(wm.word_factors[i],0,(maxNumParentsPerChild+1)*sizeof(VocabString));
  }
  if (debug(DEBUG_EVERY_SENTENCE_INFO))
    wm.print(stderr);
  return i;
}

template <class CountT>
void
FNgramSpecs<CountT>::estimateDiscounts(FactoredVocab& vocab)
{
  for (unsigned i = 0; i < fnSpecArray.size(); i++) {
    // estimate the discounts in increasing level order in BG

    // Change this for loop to:
    //    for (int level=0;level<=fnSpecArray[i].numParents;level++) {
    // to get meta, meta-meta, meta-meta-meta, (etc...) counts.
    // And change this for loop to:
    //    for (int level=fnSpecArray[i].numParents;level>=0;level--) {    
    // just to get meta level counts.
#define METAMETAMETAETC 0
#if METAMETAMETAETC
    // fprintf(stderr,"Doing meta meta meta etc. counts\n");
    for (unsigned level = fnSpecArray[i].numParents; (int)level >= 0; level--) {    
#else
      // fprintf(stderr,"Doing just meta counts\n");
    for (unsigned level = 0;level <= fnSpecArray[i].numParents; level++) {
#endif
      typename FNgramSpec::LevelIter iter(fnSpecArray[i].numParents,level);
      unsigned int subset;
      while (iter.next(subset)) {
	if (fnSpecArray[i].parentSubsets[subset].counts != NULL) {
	  FDiscount *discount = 0;	
	  Boolean gt = false;
	  if (fnSpecArray[i].parentSubsets[subset].ndiscount) {
	    discount = 
	      new FNaturalDiscount(fnSpecArray[i].parentSubsets[subset].gtmin);
	    assert(discount);
	  } else if (fnSpecArray[i].parentSubsets[subset].wbdiscount) {
	    discount = new FWittenBell(fnSpecArray[i].parentSubsets[subset].gtmin);
	    assert(discount);
	  } else if (fnSpecArray[i].parentSubsets[subset].cdiscount != -1.0) {
	    discount = 
	      new FConstDiscount(fnSpecArray[i].parentSubsets[subset].cdiscount,
				 fnSpecArray[i].parentSubsets[subset].gtmin);
	    assert(discount);
	  } else if (fnSpecArray[i].parentSubsets[subset].knFile ||
		     fnSpecArray[i].parentSubsets[subset].kndiscount) {
	    discount = 
	      new FModKneserNey(fnSpecArray[i].parentSubsets[subset].gtmin, 
			        fnSpecArray[i].parentSubsets[subset].knCountsModified,
			        fnSpecArray[i].parentSubsets[subset].knCountsModifyAtEnd);
	    assert(discount);
	  } else if (fnSpecArray[i].parentSubsets[subset].knFile ||
		     fnSpecArray[i].parentSubsets[subset].ukndiscount) {
	    discount = 
	      new FKneserNey(fnSpecArray[i].parentSubsets[subset].gtmin, 
			     fnSpecArray[i].parentSubsets[subset].knCountsModified,
			     fnSpecArray[i].parentSubsets[subset].knCountsModifyAtEnd);
	    assert(discount);
	  } else {
	    gt = true;
	    discount = new FGoodTuring(fnSpecArray[i].parentSubsets[subset].gtmin,
				       fnSpecArray[i].parentSubsets[subset].gtmax);
	    assert(discount);
	  }
	  discount->debugme(debuglevel());
	  discount->interpolate = fnSpecArray[i].parentSubsets[subset].interpolate;

	  Boolean estimated = false;
	  if (fnSpecArray[i].parentSubsets[subset].knFile && 
	      fnSpecArray[i].parentSubsets[subset].kndiscount) {
	    File file(fnSpecArray[i].parentSubsets[subset].knFile,"r",0);
	    if (!file.error()) {
	      if (!discount->read(file)) {
		fprintf(stderr,"error reading kn discount file (%s)\n",
			fnSpecArray[i].parentSubsets[subset].knFile);
		exit(-1);
	      }
	      estimated = true;
	    }
	  }
	  if (!estimated && fnSpecArray[i].parentSubsets[subset].gtFile && gt) {
	    File file(fnSpecArray[i].parentSubsets[subset].gtFile,"r",0);
	    if (!file.error()) {
	      if (!discount->read(file)) {
		fprintf(stderr,"error reading gt discount file (%s)\n",
			fnSpecArray[i].parentSubsets[subset].gtFile);
		exit(-1);
	      }
	      estimated = true;
	    }
	  }
	  if (!estimated) {
	    vocab.setCurrentTagVocab(fnSpecArray[i].child);
	    if (!discount->estimate(fnSpecArray[i],
				    subset,
				    vocab)) {
	      // TODO: make better error message here.
	      fprintf(stderr,"error in discount estimator\n");
	      exit(-1);
	    }
	    estimated = true;
	    if (fnSpecArray[i].parentSubsets[subset].kndiscount && 
		fnSpecArray[i].parentSubsets[subset].knFile) {
	      File file(fnSpecArray[i].parentSubsets[subset].knFile,"w");
	      discount->write(file);
	    } else if (gt && fnSpecArray[i].parentSubsets[subset].gtFile) {
	      File file(fnSpecArray[i].parentSubsets[subset].gtFile,"w");
	      discount->write(file);
	    }
	  }
	  fnSpecArray[i].parentSubsets[subset].discount = discount;
	}
      }
    }
  }
}



template <class CountT>
void
FNgramSpecs<CountT>::computeCardinalityFunctions(FactoredVocab& vocab)
{
  for (unsigned specNum = 0; specNum < fnSpecArray.size(); specNum++) {
    // child
    vocab.setCurrentTagVocab(fnSpecArray[specNum].childPosition);
    const unsigned numChildWords = vocab.currentTagVocabCardinality();

    for (unsigned node=0;node<fnSpecArray[specNum].numSubSets;node++) {

      fnSpecArray[specNum].parentSubsets[node].prodCardinalities 
	= numChildWords;
      fnSpecArray[specNum].parentSubsets[node].sumCardinalities 
	= numChildWords;
      fnSpecArray[specNum].parentSubsets[node].sumLogCardinalities 
	= log10((double)numChildWords);

      // parents
      for (unsigned par = 0; par < fnSpecArray[specNum].numParents; par++) {
	if (node & (1<<par)) {
	  vocab.setCurrentTagVocab(fnSpecArray[specNum].parentPositions[par]);
	  const unsigned numParWords = vocab.currentTagVocabCardinality();
	  fnSpecArray[specNum].parentSubsets[node].prodCardinalities 
	    *= numParWords;
	  fnSpecArray[specNum].parentSubsets[node].sumCardinalities 
	    += numParWords;
	  fnSpecArray[specNum].parentSubsets[node].sumLogCardinalities 
	    += log((double)numParWords);
	}
      }
    }
  }
}

 
// return pointer to a static buff where
// we've got the tag of a if any.
template <class CountT>
TLSW_DEF_ARRAY(char, FNgramSpecs<CountT>::FNgramSpecsBuff, FNgramSpecs_BUF_SZ);

template <class CountT>
VocabString
FNgramSpecs<CountT>::getTag(VocabString a)
{
  // TODO: this routine is a quick hack and should be redone properly.
  if (!a)
    return NULL;
  char *buff = TLSW_GET_ARRAY(FNgramSpecsBuff);
  char* sep_p = (char *)strchr(a,FNGRAM_WORD_TAG_SEP);
  if (sep_p == NULL)
    return NULL;
  *sep_p = '\0';
  // make sure we don't overrun buffer and it's terminated
  strncpy(buff,a,FNgramSpecs_BUF_SZ - 1);
  buff[FNgramSpecs_BUF_SZ - 1] = '\0';
  *sep_p = FNGRAM_WORD_TAG_SEP;
  return buff;
}

				       
template <class CountT>
VocabString
FNgramSpecs<CountT>::wordTag()
{
  return FNGRAM_WORD_TAG_STR;
}

#endif /* _FNgramSpecs_cc_ */
