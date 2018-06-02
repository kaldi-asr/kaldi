/*
 * FactoredVocab object, for words which can be represented as a set of
 * 'tags', where tags can either be POS or any decomposition of words.
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 * Kevin Duh <duh@ee.washington.edu>
 *
 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/FactoredVocab.h,v 1.18 2012/10/29 17:24:59 mcintyre Exp $
 *
 */

#ifndef _FactoredVocab_h_
#define _FactoredVocab_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif

#include "Boolean.h"
#include "File.h"
#include "LHash.h"
#include "Array.h"
#include "XCount.h"
#include "MemStats.h"
#include "Vocab.h"
#include "SubVocab.h"
#include "Debug.h"

const VocabString	Vocab_NULL = "<NULL>";

#include "FNgramStats.h"	// define FNgramCount

// should be a word that can never occur in any database
// const VocabString	Empty_Slot = "";
const VocabString	Empty_Slot = "*||||EMPTY||||*";


template <class CountT> class FNgramCounts;	// forward declaration
template <class CountT> class FNgramSpecs;	// forward declaration
class FNgram;
class ProductVocab;

class FactoredVocab : public  Vocab, public Debug
{
public:
  class TagIter;
  friend class TagIter;
  friend class FNgramCounts<FNgramCount>;
  friend class FNgramSpecs<FNgramCount>;
  friend class FNgram;
  friend class ProductVocab;

private:

  // array of vocabs for each tag.
  // need to turn this into another set of hash tables
  // to have fast lookup.

  struct TagVocab {
     LHash<VocabIndex,unsigned> tagMap;
  };

  Array<TagVocab> tagVocabs;
  unsigned int curTagVocab;

  LHash<VocabString,unsigned> tagPosition;

  // arrays of special tags
  Array <VocabString> tagNulls;
  Array <VocabString> tagUnks;
  Array <VocabString> tagSes;
  Array <VocabString> tagSss;
  Array <VocabString> tagPauses;

  Boolean _nullIsWord;

public:
  FactoredVocab(VocabIndex start = 0, VocabIndex end = Vocab_None-1);
  ~FactoredVocab();

  /*
   * Special (pseudo-) vocabulary tokens
   */
  VocabIndex nullIndex;		/* NULL index */
  VocabIndex emptySlot;		/* special empty slot for near beginning of sentence */
  virtual Boolean &nullIsWord() { return _nullIsWord; };

  virtual void addTaggedIndices(VocabString wordTag,VocabString separator);

  virtual void createTagSubVocabs(LHash<VocabString,unsigned>& tagPosition);
  virtual void addTagWord(unsigned tagPos, VocabIndex tag_wid);
  virtual unsigned addTagWord(VocabString tag, VocabIndex tag_wid);

  virtual VocabIndex addWord(VocabString name);
  virtual VocabIndex getIndex(VocabString name,
			      VocabIndex unkIndex = Vocab_None);
  virtual VocabIndex addWord2(VocabString name, Boolean &tagfound);
  virtual unsigned addWords2(const VocabString *words, VocabIndex *wids,
			     unsigned int max, Boolean *tagsfound);

  virtual Boolean isNonEvent(VocabIndex word) const;

  virtual void setCurrentTagVocab(unsigned i);
  virtual void setCurrentTagVocab(VocabString tag);
  virtual unsigned currentTagVocabCardinality() {
    return tagVocabs[curTagVocab].tagMap.numEntries();
  }

  // TODO: implement this, see also TODO tag in the routine
  //       FNgramCounts<CountT>::countSentence(const VocabString *words, CountT factor)

  virtual unsigned int read(File &file);

  static void freeThread();
public:

  void loadWordFactor(const VocabString word,
		      VocabString* word_factors);

  // iterator over current tag wids
  class TagIter
  {
  public:
    TagIter(FactoredVocab &vocab) 
      : myIter(vocab.tagVocabs[vocab.curTagVocab].tagMap) {}
    void init() { myIter.init(); }
    Boolean next(VocabIndex &vi) { 
      return (myIter.next(vi) != NULL); 
    };
  private:
    LHashIter<VocabIndex,unsigned> myIter;
  };

};

#endif /* _FactoredVocab_h_ */
