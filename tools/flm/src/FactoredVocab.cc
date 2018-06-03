/*
 *
 * FactoredVocab.cc --
 *	The factored vocabulary class implementation.
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 * Kevin Duh <duh@ee.washington.edu>
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2012 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/FactoredVocab.cc,v 1.20 2012/10/29 17:24:59 mcintyre Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include "File.h"
#include "FNgramStats.h"
#include "FNgramSpecs.h"
#include "FactoredVocab.h"
#include "MStringTokUtil.h"
#include "TLSWrapper.h"

#include "LHash.cc"
#include "Array.cc"

FactoredVocab::FactoredVocab(VocabIndex start,VocabIndex end)
  : Vocab(start,end) 
{
  curTagVocab = ~0x0;
  _nullIsWord = true;
  nullIndex = Vocab::addWord(Vocab_NULL);
  emptySlot = Vocab::addWord(Empty_Slot);
}

FactoredVocab::~FactoredVocab()
{
}


// Add word to vocabulary
void
FactoredVocab::addTaggedIndices(VocabString wordTag,
				VocabString separator)
{
  char buff[2048];

  // use same indices for tagged and untagged special words (unk, ss, se, and pause).
  // hopefully this is ok.
  Boolean found;
  buff[0] = '\0';
  VocabIndex *indexPtr = 
    byName.insert(strcat(strcat(strcat(buff,wordTag),separator),Vocab_NULL),found);
  *indexPtr = nullIndex;

  buff[0] = '\0';
  indexPtr = 
    byName.insert(strcat(strcat(strcat(buff,wordTag),separator),Vocab_Unknown),found);
  *indexPtr = unkIndex();

  buff[0] = '\0';
  indexPtr = 
    byName.insert(strcat(strcat(strcat(buff,wordTag),separator),Vocab_SentStart),found);
  *indexPtr = ssIndex();

  buff[0] = '\0';
  indexPtr = 
    byName.insert(strcat(strcat(strcat(buff,wordTag),separator),Vocab_SentEnd),found);  
  *indexPtr = seIndex();

  buff[0] = '\0';
  indexPtr = 
    byName.insert(strcat(strcat(strcat(buff,wordTag),separator),Vocab_Pause),found);
  *indexPtr = pauseIndex();

}

void
FactoredVocab::createTagSubVocabs(LHash<VocabString,unsigned>& _tagPosition)
{
  LHashIter<VocabString,unsigned> tags(_tagPosition);
  VocabString tag;
  unsigned *pos, maxPos = 0;

  // find the maximal position used in _tagPosition
  while ((pos = tags.next(tag))) {
    if (*pos > maxPos) maxPos = *pos;
  }

  // do a hash table copy.
  tags.init();
  while ((pos = tags.next(tag)) != NULL) {
    Boolean found;
    unsigned *u = tagPosition.insert(tag,found);
    if (!found) {
      *u = *pos;
    } else if (*u != *pos) { 
      // it's ok if a compatible position was previously recorded.
      // this happens when the multiple FNgramStats objects share the same vocab
      fprintf(stderr, "Error: FactoredVocab::createTagSubVocabs: incompatible tag position\n");
      exit(1);
    }

    // add tag to table
    if (_nullIsWord && *pos != FNGRAM_WORD_TAG_POS)
      tagVocabs[*pos].tagMap.insert(nullIndex);
    if (unkIsWord())
      tagVocabs[*pos].tagMap.insert(unkIndex());
    // add the rest which are always words
    tagVocabs[*pos].tagMap.insert(seIndex());
    // But we do not add pauseIndex, or emptySlot since
    // they should stay non events.
    tagVocabs[*pos].tagMap.insert(ssIndex());
  }
}

Boolean
FactoredVocab::isNonEvent(VocabIndex word) const
{

  // it is a non event if either it is 
  // 1. a nonevent above or
  // 2. not in the current tag set.
  // 3. null is not a word
  
  Boolean tmp =
    Vocab::isNonEvent(word) ||
    ((curTagVocab != ~0x0U) &&
     (tagVocabs[curTagVocab].tagMap.find(word) == 0)) ||
    (!_nullIsWord && (word == nullIndex));

  return tmp;
}
				  
void
FactoredVocab::addTagWord(unsigned tagPos, VocabIndex tag_wid)
{
  tagVocabs[tagPos].tagMap.insert(tag_wid);
}
				  
unsigned
FactoredVocab::addTagWord(VocabString tag, VocabIndex tag_wid)
{
  Boolean found;
  unsigned *i = tagPosition.find(tag, found);
  // assert(found);
  if (!found) {
    // should never happen, this could be an assertion.
    fprintf(stderr,"Error: FactoredVocab::const addTagWord, adding for tag %s that does not exist\n",
	    (tag?tag:"NULL"));
    abort();
  }
  tagVocabs[*i].tagMap.insert(tag_wid);
  return *i;
}

void
FactoredVocab::setCurrentTagVocab(VocabString tag)
{
  Boolean found;
  unsigned *i = tagPosition.find(tag, found);
  if (!found) {
    // should really be an assertion.
    fprintf(stderr,"Error: tag used (%s) does not exist in FVocab object\n",
	    (tag?tag:"NULL"));
    abort();
  }
  setCurrentTagVocab(*i);
}

void
FactoredVocab::setCurrentTagVocab(unsigned int i)
{
  assert (i < tagPosition.numEntries());
  curTagVocab = i;
}

VocabIndex
FactoredVocab::addWord(VocabString name)
{
  VocabIndex wid;
  // if word is tagged, make sure that we have tag entry
  // for that word. If word is not tagged, don't worry
  // about it and assume it is just has a "W-" default
  // tag (for a word tag).
  // fprintf(stderr,"FactoredVocab:: adding word %s\n",name);

  const char *tag = FNgramSpecs<FNgramCount>::getTag(name);
  if (tag == NULL) {
    // no tag, assume W- tag.
    tag = FNgramSpecs<FNgramCount>::wordTag();
  }
  // ensure that tag in word exists.
  Boolean found;
  unsigned *i = tagPosition.find(tag, found);
  if (!found) {
    fprintf(stderr,"Error: trying to add word (%s) that has unknown tag (%s)\n",
	    name,tag);
    LHashIter<VocabString,unsigned> tags(tagPosition);
    VocabString _tag;
    unsigned *_pos;
    while ((_pos = tags.next(_tag)) != NULL) {
      fprintf(stderr,"tag = (%s) at postion = %d\n",_tag,*_pos);
    }
    exit(-1);
  }
  wid = Vocab::addWord(name);
  tagVocabs[*i].tagMap.insert(wid);
  return wid;
}

// same as addWord(), but returning flag
VocabIndex
FactoredVocab::addWord2(VocabString name, Boolean &tagfound)
{
  VocabIndex wid;
  // if word is tagged, make sure that we have tag entry
  // for that word. If word is not tagged, don't worry
  // about it and assume it is just has a "W-" default
  // tag (for a word tag).
  // fprintf(stderr,"FactoredVocab:: adding word %s\n",name);

  const char *tag = FNgramSpecs<FNgramCount>::getTag(name);
  if (tag == NULL) {
    // no tag, assume W- tag.
    tag = FNgramSpecs<FNgramCount>::wordTag();
  }
  // ensure that tag in word exists.
  Boolean found;
  unsigned *i = tagPosition.find(tag, found);
  if (!found) {
      tagfound = false;
      //KEVIN's TODO: For now, uncomment the whole thing to save time. In the future, should output error/warning once to alert the user
      //fprintf(stderr,"Warning: trying to add word (%s) that has unknown tag (%s). Skipping this vocabulary item...\n", name,tag);
      return Vocab_None;
  } else {
      wid = Vocab::addWord(name);
      if (wid != Vocab_None) {
	  tagVocabs[*i].tagMap.insert(wid);
      }
      tagfound = true;
      return wid;
  }
}

// Same as addWords(), but returning a flag to indicate whether new 
// tags were found
unsigned int
FactoredVocab::addWords2(const VocabString *words, VocabIndex *wids,
					unsigned int max, Boolean *tagsfound)
{
    unsigned int i;

    for (i = 0; i < max && words[i] != 0; i++) {
	wids[i] = addWord2(words[i], tagsfound[i]);
    }
    if (i < max) {
	wids[i] = Vocab_None;
    }
    return i;
}

VocabIndex
FactoredVocab::getIndex(VocabString name,VocabIndex unkIndex)
{
  VocabIndex wid;
  // if word is tagged, make sure that we have tag entry
  // for that word. If word is not tagged, don't worry
  // about it and assume it is just has a "W-" default
  // tag (for a word tag).

  // fprintf(stderr,"FactoredVocab:: getting index for word %s\n",name);

  const char *tag = FNgramSpecs<FNgramCount>::getTag(name);
  if (tag == NULL) {
    // no tag, assume W- tag.
    tag = FNgramSpecs<FNgramCount>::wordTag();
  }
  // ensure that tag in word exists.
  Boolean found;
  unsigned *i = tagPosition.find(tag, found);
  if (!found) {
    fprintf(stderr,"Error: trying to get index of word (%s) that has unknown tag (%s)\n",
	    name,tag);
    LHashIter<VocabString,unsigned> tags(tagPosition);
    VocabString _tag;
    unsigned *_pos;
    while ((_pos = tags.next(_tag)) != NULL) {
      fprintf(stderr,"tag = (%s) at postion = %d\n",_tag,*_pos);
    }
    exit(-1);
  }
  // first try to get its wid
  wid = Vocab::getIndex(name,unkIndex);
  if (wid == unkIndex)
    return wid;
  // next see if it is in the appropriate tag set
  // and if not, return unk.
  if (tagVocabs[*i].tagMap.find(wid) == 0)
    return unkIndex;
  return wid;
}

#define word_copy_sz 2048
static TLSW_ARRAY(char, loadWordFactor_word_copy, word_copy_sz);
void
FactoredVocab::loadWordFactor(const VocabString word,
			      VocabString* word_factors)
{
  ::memset(word_factors,0,(maxNumParentsPerChild+1)*sizeof(VocabString));
  // word currently looks like:
  // <Tag1>-<factor1>:<Tag2>-<factor2>:...:<TagN>-<factorN>
  // if a tag is missing (i.e., just <factor_n>), then we
  // assume it is a FNGRAM_WORD_TAG, which indicates 
  // it is a word.
    
  Boolean tag_assigned = false;
  // make a copy of word for the work
  // use static buffer so pointers into buffer can be returned in word_factors
  char *word_copy = TLSW_GET_ARRAY(loadWordFactor_word_copy);
  strncpy(word_copy,word,word_copy_sz-1);
  word_copy[word_copy_sz-1] = '\0';
  VocabString word_p = word_copy;
  Boolean last_factor = false;
  while (!last_factor) {
    char* end_p = (char *)strchr(word_p,FNGRAM_FACTOR_SEPARATOR);
    if (end_p != NULL) {
      // this is not last word
      *end_p = '\0';
    } else 
      last_factor = true;

    char* sep_p = (char *)strchr(word_p,FNGRAM_WORD_TAG_SEP);
    if (sep_p == NULL) {
      // no tag, assume word tag. Note, either all words must
      // have a word tag "W-...", or no words can have a word tag. Otherwise,
      // vocab object will assign two different wids for same word, one
      // with wordtag and one without.
      word_factors[FNGRAM_WORD_TAG_POS] = word_p;
      tag_assigned = true;
    } else {
      *sep_p = '\0';
      unsigned* pos = tagPosition.find(word_p);
      *sep_p = FNGRAM_WORD_TAG_SEP;
      if (pos == NULL) {
	if (debug(DEBUG_TAG_WARNINGS)) {
	  fprintf(stderr,"Warning: unknown tag in factor (%s) of word (%s) when parsing file\n",
		  word_p,word);
	}
	goto next_tag;
      }
      if (*pos == FNGRAM_WORD_TAG_POS) {
	// TODO: normalize word so that it either always uses a "W-" tag
	// or does not use a "W-" tag.
      }
      if (word_factors[*pos] != NULL) {
	if (debug(DEBUG_WARN_DUP_TAG)) 
	  fprintf(stderr,"Warning: tag given twice in word (%s) when parsing "
		  "file. Using first instance.\n",word);
      } else
	word_factors[*pos] = word_p;
      tag_assigned = true;
    }
  next_tag:
    word_p = end_p+1;
  }
  if (!tag_assigned) {
    if (debug(DEBUG_TAG_WARNINGS)) {
      fprintf(stderr,"Warning: no known tags in word (%s), treating all tags as NULLs",
	      word);
    }
  }
  // store any nulls
  unsigned j;
  for (j = 0; j < tagPosition.numEntries(); j++) {
    if (word_factors[j] == 0) {
      word_factors[j] = tagNulls[j];
    }
  }
  word_factors[j] = 0;      
}

// Read vocabulary from file
// (same as Vocab::read, but using addWord2()
unsigned int
FactoredVocab::read(File &file)
{
    char *line;
    unsigned int howmany = 0;
    char *strtok_ptr = NULL;

    while ((line = file.getline())) {
	/*
	 * getline() returns only non-empty lines, so strtok()
	 * will find at least one word.  Any further ones on that
	 * line are ignored.
	 */
	strtok_ptr = NULL;
	VocabString word = MStringTokUtil::strtok_r(line, wordSeparators, &strtok_ptr);

	Boolean tagfound;
	if (addWord2(word, tagfound) == Vocab_None && tagfound) {
	    file.position() << "warning: failed to add " << word
			    << " to vocabulary\n";
	    continue;
	}
	howmany++;
    }
    return howmany;
}

void
FactoredVocab::freeThread()
{
    TLSW_FREE(loadWordFactor_word_copy);
}
