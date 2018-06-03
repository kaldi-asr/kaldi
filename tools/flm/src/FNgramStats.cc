/*
 * FNgramStats.cc --
 *	Cross-stream N-gram counting
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 * Kevin Duh <duh@ee.washington.edu>
 *
 */

#ifndef _FNgramStats_cc_
#define _FNgramStats_cc_

#ifndef lint
static char FNgramStats_Copyright[] = "Copyright (c) 1995-2012 SRI International.  All Rights Reserved.";
static char FNgramStats_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/FNgramStats.cc,v 1.22 2013-03-19 06:41:44 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <ctype.h>

#include "FNgramStats.h"

#include "LHash.cc"
#include "Trie.cc"
#include "Array.cc"
#include "FactoredVocab.h"

#define INSTANTIATE_FNGRAMCOUNTS(CountT) \
	template class FNgramCounts<CountT>


/*
 * Debug levels used
 */
#define DEBUG_VERY_VERBOSE   10
#define DEBUG_WARNINGS   1
#define DEBUG_PRINT_TEXTSTATS	1

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


template <class CountT>
void
FNgramCounts<CountT>::dump()
{
  fprintf(stderr,"FNgramCounts<CountT>::dump() undefined\n");
  exit(-1);
  // cerr << "order = " << order << endl;
  //     counts.dump();
}


template <class CountT>
void
FNgramCounts<CountT>::memStats(MemStats &stats)
{
  // TODO: update this object to reflect the new mem.
  // stats.total += sizeof(*this) - sizeof(counts) - sizeof(vocab);
  // vocab.memStats(stats);
  // counts.memStats(stats);
  fprintf(stderr, "FNgramCounts<CountT>::memStats(MemStats &stats) needs to be implemented\n:");
}

/*
 *
 * FNgramCounts constructor
 *
 */
template <class CountT>
FNgramCounts<CountT>::FNgramCounts(FactoredVocab &vocab,
				   FNgramSpecs<CountT>& fngs)
  : LMStats(vocab),fnSpecs(fngs),virtualBeginSentence(true),virtualEndSentence(true),
    addStartSentenceToken(true),addEndSentenceToken(true), vocab(vocab)
{
  LHashIter<VocabString,unsigned> tags(fnSpecs.tagPosition);
  VocabString tag;
  unsigned *pos;
  while ((pos = tags.next(tag)) != NULL) {
    vocab.addTaggedIndices(tag,FNGRAM_WORD_TAG_SEP_STR);
  }
  vocab.createTagSubVocabs(fnSpecs.tagPosition);
}


template <class CountT>
unsigned int
FNgramCounts<CountT>::countSentence(const VocabString *words, const char *factor)
{
    CountT factorCount;

    /*
     * Parse the weight string as a count
     */
    if (!stringToCount(factor, factorCount)) {
	return 0;
    }

    return countSentence(words, factorCount);
}


template <class CountT>
unsigned int
FNgramCounts<CountT>::countSentence(const VocabString *words, CountT factor)
{
    WordMatrix &wordMatrix = TLSW_GET(countSentenceWordMatrix);
    WidMatrix  &widMatrix  = TLSW_GET(countSentenceWidMatrix);
    unsigned int howmany;

    howmany = fnSpecs.loadWordFactors(words,wordMatrix,maxWordsPerLine + 1);

    /*
     * Check for buffer overflow or no data
     */
    if (howmany == maxWordsPerLine + 1 || howmany == 0) {
	return 0;
    }

    ::memset(widMatrix[0],0,(maxNumParentsPerChild+1)*sizeof(VocabIndex));
    if (openVocab) {
      for (unsigned i = 0; i < howmany; i++) {
	::memset(widMatrix[i+1],0,(maxNumParentsPerChild+1)*sizeof(VocabIndex));
	vocab.addWords(wordMatrix[i],widMatrix[i+1],maxNumParentsPerChild+1);
	for (int j=0;widMatrix[i+1][j] != Vocab_None;j++) {
	  (*((FactoredVocab*)(&vocab))).addTagWord(j,widMatrix[i+1][j]);
	}
      }
    } else {
      for (unsigned i = 0; i < howmany; i++) {
	::memset(widMatrix[i+1],0,(maxNumParentsPerChild+1)*sizeof(VocabIndex));
	vocab.getIndices(wordMatrix[i],widMatrix[i+1],maxNumParentsPerChild+1,
			 vocab.unkIndex());
	// TODO: while this code is finished for closed vocab, we need
	// still to update FactoredVocab::read() before closed vocab
	// stuff will work.
	for (unsigned j = 0; widMatrix[i+1][j] != Vocab_None; j++) {
	  if (widMatrix[i+1][j] == vocab.unkIndex()) {
	    // NOTE: since we use the same global index set for all
	    // words even if they're in different factors, you might
	    // think it would be possible for an unknown word in one
	    // factor to be known in the other and thereby give a real
	    // word index for the factor in which that word is
	    // unknown. We assume here, however, that all tokens in
	    // the word file are either without tags or with the
	    // special word tag "W-" (so are regularly words), or have
	    // a unique tag that identifies them. Therefore, this
	    // problem shouldn't happen.

	    // need to update all LMs that have this index as a child.
	    for (unsigned lm = 0; lm < fnSpecs.fnSpecArray.size(); lm++) {
	      if (j == fnSpecs.fnSpecArray[lm].childPosition)
		fnSpecs.fnSpecArray[lm].stats.numOOVs++;
	    }
	  }
	}
      }
    }

    /*
     * update OOV count, we need to do this for all factored
     * models that use the tag as a child.
     */
    if (!openVocab) {
	for (unsigned i = 1; i <= howmany; i++) {
	  for (unsigned j=0;j<fnSpecs.tagPosition.numEntries();j++) {	  
	    if (widMatrix[i][j] == vocab.unkIndex()) {
	      // TODO: update OOV counts here.
	    }
	  }
	}
    }

    /*
     * Insert begin/end sentence tokens if necessary
     */
    unsigned start;
    unsigned end;
    if (widMatrix[1][FNGRAM_WORD_TAG_POS] == vocab.ssIndex()) {
	start = 1;
	// TODO: note that if other factors at word start sentence index position
	// will be destroyed here, all being forced to start sentence.
	for (unsigned j=(FNGRAM_WORD_TAG_POS+1);j<fnSpecs.tagPosition.numEntries();j++) {
	  widMatrix[1][j] = vocab.ssIndex();
	}
    } else if (addStartSentenceToken) {
	start = 0;
	widMatrix[0][FNGRAM_WORD_TAG_POS] = vocab.ssIndex();
	for (unsigned j=(FNGRAM_WORD_TAG_POS+1);j<fnSpecs.tagPosition.numEntries();j++) {
	  widMatrix[0][j] = vocab.ssIndex();
	}
    } else {
      // just use what is given in the text file.
      start = 1;
    }

    if ((widMatrix[howmany][FNGRAM_WORD_TAG_POS] != vocab.seIndex())
	&&
	addEndSentenceToken) {
	for (unsigned j=FNGRAM_WORD_TAG_POS;j<fnSpecs.tagPosition.numEntries();j++) {
	  widMatrix[howmany+1][j] = vocab.seIndex();
	  widMatrix[howmany+2][j] = Vocab_None;
	}
	end = howmany+1;
    } else {
      // this is not done above, so we do it here.
      for (unsigned j=0;j<fnSpecs.tagPosition.numEntries();j++) {
	widMatrix[howmany+1][j] = Vocab_None;
      }
      end = howmany;
    }

    return countSentence(start, end, widMatrix, factor);
}

/*
 * Incrememt counts indexed by words, starting at node.
 *
 * Yes, Trie structures are not the most space and compute efficient way
 * to do this here, but they're easy to use and since they exist now, we
 * use them.
 * *TODO*: optimize this class to use a data-structure that is better suited
 * to the backoff-graph. One simple solution would be to use Trie structures
 * that have a count value only at the leaf nodes rather than using space
 * on thouse counts at each non-leaf and leaf node. (on the other hand,
 * perhaps this space will be useful in new forms of kn smoothing).
 * 
 */
template <class CountT>
void
FNgramCounts<CountT>::incrementCounts(FNgramNode* counts,
				      const VocabIndex *words,
				      const unsigned order,
				      const unsigned minOrder,
				      const CountT factor)
{
  FNgramNode *node = counts;

  for (unsigned i = 0; i < order; i++) {
    VocabIndex wid = words[i];
    /*
     * check of end-of-sentence
     */
    if (wid == Vocab_None) {
      break;
    } else {
      node = node->insertTrie(wid);
      if (i + 1 >= minOrder) {
	node->value() += factor;
      }
    }
  }
}

template <class CountT>
unsigned int
FNgramCounts<CountT>::countSentence(const unsigned int start, // first valid token
				    const unsigned int end,   // last valid token
				    WidMatrix& wm,
				    CountT factor)
{
  VocabIndex *wids = TLSW_GET_ARRAY(countSentenceWids);  

  if (debug(DEBUG_VERY_VERBOSE))  
    fprintf(stderr,"counting sentence\n");

  // if tag in .flm file is mentioned, but is not in file, then give
  // warning.
  
  unsigned wordNum;
  for (wordNum=start; wm[wordNum][FNGRAM_WORD_TAG_POS] != Vocab_None; wordNum++) {
    for (unsigned j = 0; j < fnSpecs.fnSpecArray.size(); j++) {
      // TODO: for efficiency, change these loops so that wids is created in 
      // reverse order (thereby reducing number of writes)
      // and then calling a version of incrementCounts() that takes
      // a wids in reverse order.
      // OR: use a level iter in reverse level order (from 0 to max).
      // Alternatively, create version of incrementCounts that
      // take a bitvector saying which word it should use, thereby
      // not using so many writes to wids.
      const int numSubSets = 1<<fnSpecs.fnSpecArray[j].numParents;
      for (int k=0;k<numSubSets;k++) {
	if (fnSpecs.fnSpecArray[j].parentSubsets[k].counts == NULL)
	  continue;
	unsigned wid_index = 0;
	for (int l=fnSpecs.fnSpecArray[j].numParents-1;l>=0;l--) {
	  if (k & (1<<l)) {
	    if (fnSpecs.fnSpecArray[j].parentOffsets[l] + (int)wordNum < (int)start) {
	      // NOTE: note that adding start sentence here will add
	      // more contexts than would exist for normal trigram
	      // models (i.e. Wt-2, Wt-1, Wt when t=0 would give a
	      // context <s> <s> w. This is harmless, in the word
	      // case.  Note that we need to do this though, because
	      // for kn smoothing we potentially need to have all BG
	      // parents have the appropriate counts for the child.
	      // NOTE: We do not get the *exact* same perplexity as
	      // ngram-count/ngram on word-only data unless we add in
	      // the following line here in the code:
	      if (!virtualBeginSentence)
		goto nextSubSet;
	      // To get the exact same perplexity, take the goto
	      // statement, and run with "-nonull" (and make sure all
	      // gtmin options are the same)
	      wids[wid_index++] = vocab.ssIndex();
	    } else if (fnSpecs.fnSpecArray[j].parentOffsets[l] + (int)wordNum > (int)end) {
	      if (!virtualEndSentence)
		goto nextSubSet;
	      wids[wid_index++] = vocab.seIndex();
	    } else {
	      wids[wid_index++] = 
		wm[wordNum + fnSpecs.fnSpecArray[j].parentOffsets[l]]
		[fnSpecs.fnSpecArray[j].parentPositions[l]];
	    }
	  }
	}
	wids[wid_index++] = wm[wordNum][fnSpecs.fnSpecArray[j].childPosition];
	wids[wid_index] = Vocab_None;
	assert (wid_index == fnSpecs.fnSpecArray[j].parentSubsets[k].order);
	incrementCounts(fnSpecs.fnSpecArray[j].parentSubsets[k].counts,
			wids,
			fnSpecs.fnSpecArray[j].parentSubsets[k].order,
			fnSpecs.fnSpecArray[j].parentSubsets[k].order,
			factor);
      nextSubSet:
	;
      }
      
    }
  }
  for (unsigned j = 0; j < fnSpecs.fnSpecArray.size(); j++) {
    // we subtract off beginning (start) and potentially more for
    // start/end of sentence tokens.
    // NOTE: NgramCounts<CountT>::countSentence() checks the
    // ssIndex/seIndex condition here so we do the same.

#if 0
    unsigned numWords = wordNum - start;
    if (wm[start][FNGRAM_WORD_TAG_POS] == vocab.ssIndex())
      numWords--;
    if (wordNum > 0 && wm[wordNum-1][FNGRAM_WORD_TAG_POS] == vocab.seIndex())
      numWords--;    
    fnSpecs.fnSpecArray[j].stats.numWords += numWords;
#endif

    fnSpecs.fnSpecArray[j].stats.numWords += (wordNum - start - 1);
    fnSpecs.fnSpecArray[j].stats.numSentences++;
  }

  if (debug(DEBUG_VERY_VERBOSE))
    fprintf(stderr,"done counting sentence\n");
  return wordNum;
}



/*****************************************************************************************
 ***************************************************************************************** 
 *****************************************************************************************/

/*
 * Type-dependent count <--> string conversions
 */

static inline Boolean
stringToGenInt(const char *str, unsigned &count)
{
  /*
   * scanf("%u") doesn't do 0x hex numbers.
   */
  char *endptr = (char*)str;
  int res = strtol(str,&endptr,0);
  if (endptr == str || res < 0) return false;
  count = (unsigned) res;
  return true;
}

/*****************************************************************************************
 ***************************************************************************************** 
 *****************************************************************************************
 */

template <class CountT>
unsigned int
FNgramCounts<CountT>::parseFNgram(char *line,
		      VocabString *words,
		      unsigned int max,
		      CountT &count,
		      unsigned int &parSpec,
		      Boolean &ok)
{
    unsigned howmany = Vocab::parseWords(line, words, max);

    if (howmany == max) {
      ok = false;
      return 0;
    }

    /*
     * Parse the last word as a count
     */
    if (!stringToCount(words[howmany - 1], count)) {
      ok = false;
      return 0;
    }

    /*
     * Parse the first word as a par spec bitmap
     */
    if (!stringToGenInt(words[0], parSpec)) {
      ok=false;
      return 0;
    }

    words[howmany-1] = 0;
    howmany -=2;

    ok = true;
    return howmany;
}

template <class CountT>
unsigned int
FNgramCounts<CountT>::readFNgram(File &file,
				 VocabString *words,
				 unsigned int max,
				 CountT &count,
				 unsigned int &parSpec,
				 Boolean& ok)
{
    char *line;
    ok = true;

    /*
     * Read next ngram count from file, skipping blank lines
     */
    line = file.getline();
    if (line == 0) {
      return 0;
    }

    unsigned howmany = parseFNgram(line, words, max, count,parSpec,ok);

    if (howmany == 0) {
      ok = false;
      file.position() << "malformed N-gram count or more than " << max - 1 << " words per line\n";
      return 0;
    }

    return howmany;
}


/*
 *
 * This reads an ngram counts file (i.e., word and word history followed
 * by a count of the occurance of that word and word history), and loads
 * it into the counts object.
 */
template <class CountT>
Boolean
FNgramCounts<CountT>::read(unsigned int specNum,File &file)
{
    VocabString *words     = TLSW_GET_ARRAY(readWords);
    VocabIndex  *wids      = TLSW_GET_ARRAY(readWids);
    Boolean     *tagsfound = TLSW_GET_ARRAY(readTagsFound);
    CountT count;
    unsigned int howmany;
    unsigned int parSpec;
    Boolean ok;

    if (specNum >= fnSpecs.fnSpecArray.size())
      return false;

    while ((howmany = readFNgram(file, words, maxFNgramOrder + 1, count, parSpec,ok))) {
      if (!ok)
	return false;

      /* 
       * Map words to indices, skip over first word since it is parspec in string form
       */
      Boolean skipNgramCount = false;
      if (openVocab) {
	vocab.addWords2(words+1, wids, maxFNgramOrder,tagsfound+1);
	for (unsigned j = 0; j < maxFNgramOrder && words[j+1] != 0; j++) {
	  if (tagsfound[j+1]==0) {
	    skipNgramCount = true;
	  }
	}
      } else {
	vocab.getIndices(words+1, wids, maxFNgramOrder, vocab.unkIndex());
      }

      /*
       *  Update the count
       */
      if (!skipNgramCount) {
	if (fnSpecs.fnSpecArray[specNum].parentSubsets[parSpec].counts == NULL) {
	  if (debug(DEBUG_WARNINGS)) {
	    fprintf(stderr,"WARNING: counts file contains counts for backoff graph node that does not exist given backoff constraints, creading counts object anyway\n");
	  }
	  fnSpecs.fnSpecArray[specNum].parentSubsets[parSpec].counts = new FNgramNode;
	  // alternatively, we could just ignore the count, since it will
	  // not be used.
	}
	*(fnSpecs.fnSpecArray[specNum].parentSubsets[parSpec].counts->insert(wids))
	  += count;
      }
    }
    return true;
}




template <class CountT>
Boolean
FNgramCounts<CountT>::read()
{
  for (unsigned i = 0; i < fnSpecs.fnSpecArray.size(); i++) {
    File f(fnSpecs.fnSpecArray[i].countFileName, "r");
    if (!read(i,f))
      return false;
  }
  return true;
}





/*****************************************************************************************
 ***************************************************************************************** 
 *****************************************************************************************
 */


template <class CountT>
unsigned int
FNgramCounts<CountT>::writeFNgram(File &file,
				  const VocabString *words,
				  CountT count,
				  unsigned int parSpec)
{
    unsigned int i;
    if (words[0]) {
      file.fprintf("0x%X\t",parSpec);
      file.fprintf("%s", words[0]);
      for (i = 1; words[i]; i++) {
	file.fprintf(" %s", words[i]);
      }
    }
    // why could we have a count w/o any words?
    file.fprintf("\t%s\n", countToString(count));

    return i;
}

/*
 * For reasons of efficiency the write() method doesn't use
 * writeFNgram()  (yet).  Instead, it fills up a string buffer 
 * as it descends the tree recursively.  this avoid having to
 * lookup shared prefix words and buffer the corresponding strings
 * repeatedly.
 */
template <class CountT>
void
FNgramCounts<CountT>::writeNode(
    FNgramNode *node,		/* the trie node we're at */
    const unsigned int parSpec, /* parent specifier */
    File &file,			/* output file */
    char *buffer,		/* output buffer */
    char *bptr,			/* pointer into output buffer */
    unsigned int level,		/* current trie level */
    unsigned int order,		/* target trie level */
    Boolean sorted)		/* produce sorted output */
{
    FNgramNode *child;
    VocabIndex wid;

    TrieIter<VocabIndex,CountT> iter(*node, sorted ? vocab.compareIndex() : 0);

    /*
     * Iterate over the child nodes at the current level,
     * appending their word strings to the buffer
     */
    while (!file.error() && (child = iter.next(wid))) {
	VocabString word = vocab.getWord(wid);

	if (word == 0) {
	   cerr << "undefined word index " << wid << "\n";
	   continue;
	}

	unsigned wordLen = strlen(word);

	if (bptr + wordLen + 1 > buffer + maxLineLength) {
	   *bptr = '0';
	   cerr << "ngram ["<< buffer << word
		<< "] exceeds write buffer\n";
	   continue;
	}
        
	strcpy(bptr, word);

	/*
	 * If this is the final level, print out the ngram and the count.
	 * Otherwise set up another level of recursion.
	 */
	if (order == 0 || level == order) {
	   file.fprintf("0x%X\t%s\t%s\n",parSpec,buffer, countToString(child->value()));
	} 
	
	if (order == 0 || level < order) {
	   *(bptr + wordLen) = ' ';
	   writeNode(child, parSpec, file, buffer, bptr + wordLen + 1, level + 1,
			order, sorted);
	}
    }
}

template <class CountT>
void
FNgramCounts<CountT>::writeSpec(File &file, 
				const unsigned int specNum, 
				const Boolean sorted)
{
  if (specNum >= fnSpecs.fnSpecArray.size())
    return;
  
  char *buffer = TLSW_GET_ARRAY(writeSpecBuffer);
  const unsigned numSubSets = 1U<<fnSpecs.fnSpecArray[specNum].numParents;
  for (unsigned int i = 0; i < numSubSets; i++) {
    if (fnSpecs.fnSpecArray[specNum].parentSubsets[i].counts != NULL) {
      writeNode(fnSpecs.fnSpecArray[specNum].parentSubsets[i].counts,
		i,file, buffer, buffer, 1, numBitsSet(i)+1, sorted);
      
    }
    // TODO: also write out individal BG node count file(s) if specified.
  }

}


template <class CountT>
void
FNgramCounts<CountT>::write(const Boolean sorted)
{
  for (unsigned i = 0; i < fnSpecs.fnSpecArray.size(); i++) {
    if (strcmp(fnSpecs.fnSpecArray[i].countFileName,FNGRAM_DEV_NULL_FILE) != 0) {
      File f(fnSpecs.fnSpecArray[i].countFileName, "w");
      writeSpec(f,i,sorted);
    }
  }
}


template <class CountT>
void
FNgramCounts<CountT>::estimateDiscounts()
{
  fnSpecs.estimateDiscounts(*((FactoredVocab*)&vocab));
}


template <class CountT>
void
FNgramCounts<CountT>::computeCardinalityFunctions()
{
  fnSpecs.computeCardinalityFunctions(*((FactoredVocab*)&vocab));
}




template <class CountT>
CountT
FNgramCounts<CountT>::sumCounts(unsigned int specNum,unsigned int node)
{
  return fnSpecs.fnSpecArray[specNum].parentSubsets[node].accumulateCounts();
}

template <class CountT>
CountT
FNgramCounts<CountT>::sumCounts(unsigned int specNum)
{
  CountT sum=0;
  for (unsigned node = 0; node < fnSpecs.fnSpecArray[specNum].numSubSets; node++) {
    sum+=sumCounts(specNum,node);
  }
  return sum;
}

template <class CountT>
CountT
FNgramCounts<CountT>::sumCounts()
{
  CountT sum=0;
  for (unsigned i = 0; i < fnSpecs.fnSpecArray.size(); i++) {
    sum+=sumCounts(i);
  }
  return sum;
}


// parse file into sentences and update stats
template <class CountT>
unsigned int
FNgramCounts<CountT>::countFile(File &file, Boolean weighted)
{
    unsigned numWords = 0;
    char *line;

    while ((line = file.getline())) {
	unsigned int howmany = countString(line, weighted);

	/*
	 * Since getline() returns only non-empty lines,
	 * a return value of 0 indicates some sort of problem.
	 */
	if (howmany == 0) {
	    file.position() << "line too long?\n";
	} else {
	    numWords += howmany;
	}
    }
    if (debug(DEBUG_PRINT_TEXTSTATS)) {
      file.position(dout()) << endl;
      for (unsigned i=0; i < fnSpecs.fnSpecArray.size(); i++) {
	dout() << "LM(" << i << ") " << fnSpecs.fnSpecArray[i].stats;
      }
      // file.position(dout()) << this -> stats;
    }
    return numWords;
}

#endif /* _FNgramStats_cc_ */
