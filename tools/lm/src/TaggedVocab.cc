/*
 * TaggedVocab.cc --
 *	Tagged vocabulary implementation
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/TaggedVocab.cc,v 1.12 2013-03-19 06:41:41 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "TaggedVocab.h"
#include "LHash.cc"

const char tagSep = '/';		/* delimiter separating word from tag */

static char *
findTagSep(VocabString name)
{
    unsigned len = strlen(name);

    for (const char *p = &name[len-1]; p >= name; p --) {
	if (*p == tagSep &&
	    /*
	     * Don't mistake '/' inside SGML tags as tag separators
	     */
	    (p == name || *(p-1) != '<'))
	{
	    return (char *)p;		/* discard const */
	}
    }
    return 0;
}

/*
 * Initialize the word Vocab making sure indices will fit in the lower-order
 * bits.  Let tag indices start at one, so 0 can be used for the untagged
 * case.
 */
TaggedVocab::TaggedVocab(VocabIndex start, VocabIndex end)
    : Vocab(start, end > maxTaggedIndex ? maxTaggedIndex : end),
      _tags(1, maxTagIndex)
{
    if (end > maxTaggedIndex) {
	cerr << "warning: maximum tagged index lowered to "
	     << maxTaggedIndex << endl;
    }
}

TaggedVocab::~TaggedVocab()
{
    /*
     * Free cached word/tag strings
     */
    LHashIter<VocabIndex, VocabString> iter(taggedWords);
    VocabIndex idx;
    VocabString *tword;

    while ((tword = iter.next(idx))) {
	free((void *)*tword);
    }
}

void
TaggedVocab::memStats(MemStats &stats) const
{
    Vocab::memStats(stats);
    stats.total -= sizeof(_tags);
    _tags.memStats(stats);
}


VocabIndex
TaggedVocab::addWord(VocabString name)
{
    /*
     * Check if the string is a tagged word, and if so add both word and
     * tag string to the respective vocabs.
     */
    char saved, *sep = findTagSep(name);

    VocabIndex wordId, tagId;

    if (sep) {
	saved = *sep;
	*sep = '\0';

	/*
	 * empty words are allowed to denote tags by themselves
	 */
	wordId = (sep == name) ? Tagged_None : Vocab::addWord(name);
	tagId = _tags.addWord(sep + 1);
    } else {
	wordId = Vocab::addWord(name);
	tagId = Tag_None;
    }
	
    if (wordId == Vocab_None) {
	cerr << "maximum number of tagged words (" << maxTaggedIndex << ") exceeded\n";
	assert(wordId != Vocab_None);
    }

    if (tagId == Vocab_None) {
	cerr << "maximum number of tags (" << maxTagIndex << ") exceeded\n";
	assert(tagId != Vocab_None);
    }

    if (sep) {
	*sep = saved;
    }

    return tagWord(wordId, tagId);
}

VocabString
TaggedVocab::getWord(VocabIndex index)
{
    /*
     * First check cache of seen word/tag strings 
     */
    VocabString *cachedResult = taggedWords.find(index);

    if (cachedResult) {
	return *cachedResult;

    /*
     * Check if index contains tag, and if so construct a word/tag string
     * for it.
     */
    } else if (getTag(index) != Tag_None || unTag(index) == Tagged_None) {
	VocabString wordStr =
		isTag(index) ? "" : Vocab::getWord(unTag(index));
	VocabString tagStr =
		(getTag(index) == Tag_None) ? "" : _tags.getWord(getTag(index));

	if (wordStr == 0 || tagStr == 0) {
	    return 0;
	} else {
	    unsigned resultLen = strlen(wordStr) + 1 + strlen(tagStr) + 1;

	    char *thisResult = (char *)malloc(resultLen + 1);
	    assert(thisResult != 0);

	    sprintf(thisResult, "%s%c%s", wordStr, tagSep, tagStr);

	    *taggedWords.insert(index) = thisResult;

	    return thisResult;
	}
    } else {
	return Vocab::getWord(index);
    }
}

VocabIndex
TaggedVocab::getIndex(VocabString name, VocabIndex unkIndex)
{
    /*
     * Check if the string is a tagged word, and if so return a tagged index
     */
    char *sep = findTagSep(name);

    if (sep) {
	char saved = *sep;
	*sep = '\0';

	VocabIndex wordId =
		(sep == name) ? Tagged_None : Vocab::getIndex(name, Vocab_None);
	VocabIndex tagId = _tags.getIndex(sep + 1, Vocab_None);

	*sep = saved;

	if (wordId == Vocab_None || tagId == Vocab_None) {
	    return unkIndex;
	} else {
	    return tagWord(wordId, tagId);
	}
    } else {
	return Vocab::getIndex(name, unkIndex);
    }
}

void
TaggedVocab::remove(VocabString name)
{
    /*
     * Check if the string is a tagged word, and if so remove both word
     * and tag.
     */
    char *sep = findTagSep(name);

    if (sep) {
	char saved = *sep;
	*sep = '\0';

	if (sep != name) {
	    Vocab::remove(name);
	}
	_tags.remove(sep + 1);

	*sep = saved;
    } else {
	Vocab::remove(name);
    }
}

void
TaggedVocab::remove(VocabIndex index)
{
    /*
     * Check if index contains tag, and if so remove both word and tag.
     */
    if (getTag(index) != Tag_None) {
	if (!isTag(index)) {
	    Vocab::remove(unTag(index));
	}
	_tags.remove(getTag(index));
    } else {
	Vocab::remove(index);
    }
}

// Write vocabulary to file
void
TaggedVocab::write(File &file, Boolean sorted) const
{
    Vocab::write(file, sorted);

    VocabIter iter(_tags, sorted);
    VocabString word;

    while (!file.error() && (word = iter.next())) {
	file.fprintf("%c%s\n", tagSep, word);
    }
}

