/*
 * Vocab.cc --
 *	The vocabulary class implementation.
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2012 SRI International, 2012 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Vocab.cc,v 1.57 2014-08-29 21:35:48 frandsen Exp $";
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
#include "Vocab.h"
#include "LHash.cc"
#include "Array.cc"
#include "MStringTokUtil.h"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(VocabString,VocabIndex);
INSTANTIATE_ARRAY(VocabString);
INSTANTIATE_ARRAY(VocabIndex);		// for repeated use elsewhere
#endif

/* vocabulary implicitly used by operator<< */
TLSW_DEF(Vocab*, Vocab::outputVocabTLS);
/* implicit parameter to compare() */
TLSW_DEF(Vocab*, Vocab::compareVocabTLS);

void
Vocab::setOutputVocab(Vocab *v) {
    Vocab* &output = TLSW_GET(outputVocabTLS);
    output = v;
}

void
Vocab::setCompareVocab(Vocab *v) {
    Vocab* &compare = TLSW_GET(compareVocabTLS);
    compare = v;
}

Vocab::Vocab(VocabIndex start, VocabIndex end)
    : byIndex(start), nextIndex(start), maxIndex(end), _metaTag(0)
{
    /*
     * Vocab_None is both the non-index value and the end-token
     * for key sequences used with Tries.  Both need to be
     * the same value so the user or the LM library doesn't have
     * to deal with Map_noKey() and friends.
     */
    assert(Map_noKeyP(Vocab_None));

    /*
     * Make the last Vocab created be the default one used in
     * ostream output
     */
    Vocab* &outputVocab = TLSW_GET(outputVocabTLS);
    outputVocab = this;

    /*
     * default is a closed-vocabulary model
     */
    _unkIsWord = false;

    /*
     * do not map word strings to lowercase by defaults
     */
    _toLower = false;

    /*
     * set some special vocabulary tokens to their defaults
     */
    _unkIndex = addWord(Vocab_Unknown);
    _ssIndex = addWord(Vocab_SentStart);
    _seIndex = addWord(Vocab_SentEnd);
    _pauseIndex = addWord(Vocab_Pause);

    /*
     * declare some known non-events
     */
    addNonEvent(_ssIndex);
    addNonEvent(_pauseIndex);
}

// Compute memory usage
void
Vocab::memStats(MemStats &stats) const
{
    stats.total += sizeof(*this) - sizeof(byName) - sizeof(byIndex);
    byName.memStats(stats);
    byIndex.memStats(stats);
}

// map word string to lowercase
// returns a static buffer
static TLSW_ARRAY(char, lowerTLS, maxWordLength + 1);
VocabString
Vocab::mapToLower(VocabString name)
{
    char* lower = TLSW_GET_ARRAY(lowerTLS);

    unsigned  i;
    for (i = 0; name[i] != 0 && i < maxWordLength; i++) {
	lower[i] = tolower((unsigned char)name[i]);
    }
    lower[i] = '\0';

    assert(i < maxWordLength);

    return lower;
}

void
Vocab::freeThread() 
{
    TLSW_FREE(outputVocabTLS);
    TLSW_FREE(compareVocabTLS);
    TLSW_FREE(lowerTLS);
}
// Add word to vocabulary
VocabIndex
Vocab::addWord(VocabString name)
{
    if (_toLower) {
	name = mapToLower(name);
    }

    Boolean found;
    VocabIndex *indexPtr = byName.insert(name, found);

    if (found) {
	return *indexPtr;
    } else {
	if (nextIndex == maxIndex) {
	    return Vocab_None;
	} else {
	    *indexPtr = nextIndex;
	    byIndex[nextIndex] = byName.getInternalKey(name);

	    /*
	     * Check for metatags, and intern them into our metatag type map
	     */
	    if (_metaTag != 0) {
		unsigned metaTagLength = strlen(_metaTag);

		if (strncmp(name, _metaTag, metaTagLength) == 0) {
		    int type = -1;
		    if (name[metaTagLength] == '\0') {
			type = 0;
		    } else {
			sscanf(&name[metaTagLength], "%u", (unsigned *)&type);
		    }
		    if (type >= 0) {
			*metaTagMap.insert(nextIndex) = type;
		    }
		}
	    }

	    return nextIndex++;
	}
    } 
}

// add an alternate (alias) name for a word index
VocabIndex
Vocab::addWordAlias(VocabIndex word, VocabString name)
{
    if (_toLower) {
	name = mapToLower(name);
    }

    // make sure word is a valid index
    if (byIndex[word] == 0) {
	return Vocab_None;
    } else {
	// avoid aliasing name to itself
	if (strcmp(name, byIndex[word]) == 0) {
	    return word;
	}

	// make sure name isn't otherwise used
	remove(name);

	VocabIndex *indexPtr = byName.insert(name);

	*indexPtr = word;

	return word;
    }
}

// declare word to be a non-event
VocabIndex
Vocab::addNonEvent(VocabIndex word)
{
    /* 
     * First make sure the word is already defined
     */
    if (getWord(word) == 0) {
	return Vocab_None;
    } else {
	*nonEventMap.insert(word) = 1;
	return word;
    }
}

// declare a set of non-events
Boolean
Vocab::addNonEvents(Vocab &nonevents)
{
    VocabIter viter(nonevents);
    Boolean ok = true;

    VocabString name;
    while ((name = viter.next())) {
	if (addNonEvent(name) == Vocab_None) {
	    ok = false;
	}
    }

    return ok;
}

// remove word as a non-event
Boolean
Vocab::removeNonEvent(VocabIndex word)
{
    /* 
     * First make sure the word is already defined
     */
    if (getWord(word) == 0) {
	return false;
    } else {
	return nonEventMap.remove(word);
    }
}

// Get a word's index by its name
VocabIndex
Vocab::getIndex(VocabString name, VocabIndex unkIndex)
{
    if (_toLower) {
	name = mapToLower(name);
    }

    VocabIndex *indexPtr = byName.find(name);

    /*
     * If word is a metatag and not already interned, do it now
     */
    if (indexPtr == 0 &&
	_metaTag != 0 &&
	strncmp(name, _metaTag, strlen(_metaTag)) == 0)
    {
	return addWord(name);
    } else {
	return indexPtr ? *indexPtr : unkIndex;
    }
}

// Get the index of a metatag
VocabIndex
Vocab::metaTagOfType(unsigned type)
{
    if (_metaTag == 0) {
	return Vocab_None;
    } else {
	if (type == 0) {
	    return getIndex(_metaTag);
	} else {
	    makeArray(char, tagName, strlen(_metaTag) + 20);

	    sprintf(tagName, "%s%u", _metaTag, type);
	    return getIndex(tagName);
	}
    }
}

// Get a word's name by its index
VocabString
Vocab::getWord(VocabIndex index)
{
    if (index < (VocabIndex)byIndex.base() || index >= nextIndex) {
	return 0;
    } else {
	return (*(Array<VocabString>*)&byIndex)[index];	// discard const
    }
}

// Delete a word by name
void
Vocab::remove(VocabString name)
{
    if (_toLower) {
	name = mapToLower(name);
    }

    VocabIndex *indexPtr = byName.find(name);

    if (indexPtr == 0) {
    	return;
    } else if (strcmp(name, byIndex[*indexPtr]) != 0) {
	// name is an alias: only remove the string mapping, not the index
	byName.remove(name);
    } else {
	VocabIndex idx = *indexPtr;

	byName.remove(name);
	byIndex[idx] = 0;
	nonEventMap.remove(idx);
	metaTagMap.remove(idx);

	if (idx == _ssIndex) {
	    _ssIndex = Vocab_None;
	}
	if (idx == _seIndex) {
	    _seIndex = Vocab_None;
	}
	if (idx == _unkIndex) {
	    _unkIndex = Vocab_None;
	}
	if (idx == _pauseIndex) {
	    _pauseIndex = Vocab_None;
	}
    }
}

// Delete a word by index
void
Vocab::remove(VocabIndex index)
{
    if (index < (VocabIndex)byIndex.base() || index >= nextIndex) {
	return;
    } else {
	VocabString name = byIndex[index];
	if (name) {
	    byName.remove(name);
	    byIndex[index] = 0;
	    nonEventMap.remove(index);
	    metaTagMap.remove(index);

	    if (index == _ssIndex) {
		_ssIndex = Vocab_None;
	    }
	    if (index == _seIndex) {
		_seIndex = Vocab_None;
	    }
	    if (index == _unkIndex) {
		_unkIndex = Vocab_None;
	    }
	    if (index == _pauseIndex) {
		_pauseIndex = Vocab_None;
	    }
	}
    }
}

// Convert index sequence to string sequence
unsigned int
Vocab::getWords(const VocabIndex *wids, VocabString *words,
						    unsigned int max)
{
    unsigned int i;

    for (i = 0; i < max && wids[i] != Vocab_None; i++) {
	words[i] = getWord(wids[i]);
    }
    if (i < max) {
	words[i] = 0;
    }
    return i;
}

// Convert word sequence to index sequence, adding words if needed
unsigned int
Vocab::addWords(const VocabString *words, VocabIndex *wids, unsigned int max)
{
    unsigned int i;

    for (i = 0; i < max && words[i] != 0; i++) {
	wids[i] = addWord(words[i]);
    }
    if (i < max) {
	wids[i] = Vocab_None;
    }
    return i;
}

// Convert word sequence to index sequence (without adding words)
unsigned int
Vocab::getIndices(const VocabString *words,
		  VocabIndex *wids, unsigned int max,
		  VocabIndex unkIndex)
{
    unsigned int i;

    for (i = 0; i < max && words[i] != 0; i++) {
	wids[i] = getIndex(words[i], unkIndex);
    }
    if (i < max) {
	wids[i] = Vocab_None;
    }
    return i;
}

// Convert word sequence to index sequence, checking if words are in 
// vocabulary; return false is not, true otherwise.
Boolean
Vocab::checkWords(const VocabString *words, VocabIndex *wids, unsigned int max)
{
    unsigned int i;

    for (i = 0; i < max && words[i] != 0; i++) {
	if ((wids[i] = getIndex(words[i], Vocab_None)) == Vocab_None) {
	    return false;
	}
    }
    if (i < max) {
	wids[i] = Vocab_None;
    }
    return true;
}

// parse strings into words and update stats
unsigned int
Vocab::parseWords(char *sentence, VocabString *words, unsigned int max)
{
    char *word;
    unsigned i;
    char *strtok_ptr = NULL;

    for (i = 0, word = MStringTokUtil::strtok_r(sentence, wordSeparators, &strtok_ptr);
	 i < max && word != 0;
	 i++, word = MStringTokUtil::strtok_r(0, wordSeparators, &strtok_ptr))
    {
	words[i] = word;
    }
    if (i < max) {
	words[i] = 0;
    }

    return i;
}

/*
 * Length of Ngrams
 */
unsigned int
Vocab::length(const VocabIndex *words)
{
    unsigned int len = 0;

    while (words[len] != Vocab_None) len++;
    return len;
}

unsigned int
Vocab::length(const VocabString *words)
{
    unsigned int len = 0;

    while (words[len] != 0) len++;
    return len;
}

/*
 * Copying (a la strcpy())
 */
VocabIndex *
Vocab::copy(VocabIndex *to, const VocabIndex *from)
{
    unsigned i;
    for (i = 0; from[i] != Vocab_None; i ++) {
	to[i] = from[i];
    }
    to[i] = Vocab_None;

    return to;
}

VocabString *
Vocab::copy(VocabString *to, const VocabString *from)
{
    unsigned i;
    for (i = 0; from[i] != 0; i ++) {
	to[i] = from[i];
    }
    to[i] = 0;

    return to;
}

/*
 * Word containment
 */
Boolean
Vocab::contains(const VocabIndex *words, VocabIndex word)
{
    unsigned i;

    for (i = 0; words[i] != Vocab_None; i++) {
	if (words[i] == word) {
	    return true;
	}
    }
    return false;
}

/*
 * Reversal of Ngrams
 */
VocabIndex *
Vocab::reverse(VocabIndex *words)
{
    int i, j;	/* j can get negative ! */

    for (i = 0, j = length(words) - 1;
	 i < j;
	 i++, j--)
    {
	VocabIndex x = words[i];
	words[i] = words[j];
	words[j] = x;
    }
    return words;
}

VocabString *
Vocab::reverse(VocabString *words)
{
    int i, j;	/* j can get negative ! */

    for (i = 0, j = length(words) - 1;
	 i < j;
	 i++, j--)
    {
	VocabString x = words[i];
	words[i] = words[j];
	words[j] = x;
    }
    return words;
}

/*
 * Output of Ngrams
 */

void
Vocab::write(File &file, const VocabString *words)
{
    for (unsigned int i = 0; words[i] != 0; i++) {
	file.fprintf("%s%s", (i > 0 ? " " : ""), words[i]);
    }
}

ostream &
operator<< (ostream &stream, const VocabString *words)
{
    for (unsigned int i = 0; words[i] != 0; i++) {
	stream << (i > 0 ? " " : "") << words[i];
    }
    return stream;
}

ostream &
operator<< (ostream &stream, const VocabIndex *words)
{
    Vocab* &outputVocab = TLSW_GET(Vocab::outputVocabTLS);
    for (unsigned int i = 0; words[i] != Vocab_None; i++) {
	VocabString word = outputVocab->getWord(words[i]);

	stream << (i > 0 ? " " : "")
	       << (word ? word : "UNKNOWN");
    }
    return stream;
}

/* 
 * Sorting of words and word sequences
 */
// compare to word indices by their associated word strings
// This should be a non-static member, so we don't have to pass the
// Vocab in a global variable, but then we couldn't use this function
// with qsort() and friends.
// If compareVocab == 0 then comparison by index is performed.
int
Vocab::compare(VocabIndex word1, VocabIndex word2)
{
    Vocab* &compareVocab = TLSW_GET(compareVocabTLS);

    if (compareVocab == 0) {
	return word2 - word1;
    } else {
	return strcmp(compareVocab->getWord(word1),
		      compareVocab->getWord(word2));
    }
}

int
Vocab::compare(const VocabString *words1, const VocabString *words2)
{
    unsigned int i = 0;

    for (i = 0; ; i++) {
	if (words1[i] == 0) {
	    if (words2[i] == 0) {
		return 0;
	    } else {
		return -1;	/* words1 is shorter */
	    }
	} else {
	    if (words2[i] == 0) {
		return 1;	/* words2 is shorted */
	    } else {
		int comp = compare(words1[i], words2[i]);
		if (comp != 0) {
		    return comp;	/* they differ as pos i */
		}
	    }
	}
    }
    /*NOTREACHED*/
}

int
Vocab::compare(const VocabIndex *words1, const VocabIndex *words2)
{
    unsigned int i = 0;

    for (i = 0; ; i++) {
	if (words1[i] == Vocab_None) {
	    if (words2[i] == Vocab_None) {
		return 0;
	    } else {
		return -1;	/* words1 is shorter */
	    }
	} else {
	    if (words2[i] == Vocab_None) {
		return 1;	/* words2 is shorted */
	    } else {
		int comp = compare(words1[i], words2[i]);
		if (comp != 0) {
		    return comp;	/* they differ as pos i */
		}
	    }
	}
    }
    /*NOTREACHED*/
}

/*
 * These are convenience methods which set the implicit Vocab used
 * by the comparison functions and returns a pointer to them.
 * Suitable to generate the 'sort' argument used by the iterators.
 */
VocabIndexComparator
Vocab::compareIndex() const
{
    Vocab* &compareVocab = TLSW_GET(compareVocabTLS);
    compareVocab = (Vocab *)this;	// discard const
    return &Vocab::compare;
}

VocabIndicesComparator
Vocab::compareIndices() const
{
    Vocab* &compareVocab = TLSW_GET(compareVocabTLS);
    compareVocab = (Vocab *)this;	// discard const
    return &Vocab::compare;
}

// Write vocabulary to file
void
Vocab::write(File &file, Boolean sorted) const
{
    VocabIter iter(*this, sorted);
    VocabString word;

    while (!file.error() && (word = iter.next())) {
	file.fprintf("%s\n", word);
    }
}

// Read vocabulary from file
unsigned int
Vocab::read(File &file)
{
    char *line;
    unsigned int howmany = 0;
    char *strtok_ptr = NULL;

    while ((line = file.getline())) {
	/*
	 * getline() returns only non-empty lines, so strtok()
	 * will find at least one word.  Any further ones on that line
	 * are ignored.
	 */
	strtok_ptr = NULL;
	VocabString word = MStringTokUtil::strtok_r(line, wordSeparators, &strtok_ptr);

	if (addWord(word) == Vocab_None) {
	    file.position() << "warning: failed to add " << word
			    << " to vocabulary\n";
	    continue;
	}
	howmany++;
    }
    return howmany;
}

// Read alias mapping from file
unsigned int
Vocab::readAliases(File &file)
{
    char *line;
    unsigned int howmany = 0;
    char *strtok_ptr = NULL;

    while ((line = file.getline())) {
	/*
	 * getline() returns only non-empty lines, so strtok()
	 * will find at least one word.  Anything after the second word
	 * is ignored.
	 */
	strtok_ptr = NULL;
	VocabString alias = MStringTokUtil::strtok_r(line, wordSeparators, &strtok_ptr);
	VocabString word = MStringTokUtil::strtok_r(0, wordSeparators, &strtok_ptr);

	if (word == 0) {
	    file.position() << "warning: line contains only one token\n";
	    continue;
	}

	VocabIndex windex;

	if ((windex = addWord(word)) == Vocab_None) {
	    file.position() << "warning: failed to add " << word
			    << " to vocabulary\n";
	    continue;
	}

	if (addWordAlias(windex, alias) == Vocab_None) {
	    file.position() << "warning: failed to add alias " << alias
			    << " for word " << word
			    << " to vocabulary\n";
	    continue;
	}

	howmany++;
    }
    return howmany;
}

VocabIndex
Vocab::highIndex() const
{
    if (nextIndex == 0) {
	return Vocab_None;
    } else {
	return nextIndex - 1;
    }
}

/*
 * Check that vocabulary contains words in the range of ngrams given
 * (used for reading Google ngrams efficiently)
 * startRange == 0 means the beginning (INITIAL) of the sorting order
 * endRange == 0 means the end (FINAL) of the sorting order implicitly
 *
 * Algorithm:
 *	if empty(startRange)
 *	    return TRUE
 *	else
 *	    if first(startRange) == first(endRange) and 
 *	       first(startRange) in vocab
 *		return ngramsInRange(rest(startRange), rest(endRange))
 *	    else
 *		if first(startRange) in vocab and
 *		   ngramsInRange(rest(startRange), FINAL)
 *		    return TRUE
 *		if first(endRange) in vocab and
 *		   ngramsInRange(INITIAL, rest(endRange))
 *		    return TRUE
 *		for all w in vocab
 *		   if w > first(startRange) and w < last(endRange)
 *			return TRUE;
 */
Boolean
Vocab::ngramsInRange(VocabString *startRange, VocabString *endRange)
{
    if ((startRange && startRange[0] == 0) ||
        (endRange && endRange[0] == 0))
    {
	return true;
    } else if (startRange && endRange &&
               strcmp(startRange[0], endRange[0]) == 0 &&
               getIndex(startRange[0]) != Vocab_None)
    {
	return ngramsInRange(&startRange[1], &endRange[1]);
    } else {
    	if (startRange && getIndex(startRange[0]) != Vocab_None &&
	    ngramsInRange(&startRange[1], 0))
    	{
	    return true;
	}
	if (endRange && getIndex(endRange[0]) != Vocab_None &&
	    ngramsInRange(0, &endRange[1]))
	{
	    return true;
	}

    	/*
	 * Cycle through entire vocabulary, looking for a word that's in range
	 */
	VocabIter iter(*this);
	VocabString word;

	while ((word = iter.next())) {
	    if ((startRange == 0 || strcmp(startRange[0], word) < 0) &&
	        (endRange == 0 || strcmp(word, endRange[0]) < 0))
	    {
		return true;
	    }
	}
	return false;
    }
}

/*
 * Input/output of word-index mappings
 *	word-index mappings store the VocabString-VocabIndex mapping in
 *	binary data formats.
 *	The format is ascii with one word per line:
 *		index	string
 *	The mapping is terminated by EOF or a line consisting only of ".".
 *	If writingLM is true, omit words that should not appear in LMs.
 */
void
Vocab::writeIndexMap(File &file, Boolean writingLM)
{
    // Output index map in order of internal vocab indices.
    // This ensures that vocab strings are assigned indices in the same order
    // on reading, and ensures faster insertions into SArray-based tries.
    for (unsigned i = byIndex.base(); i < nextIndex; i ++) {
	if (byIndex[i] && !(writingLM && isMetaTag(i))) {
	    file.fprintf("%u %s\n", i, byIndex[i]);
	}
    }
    file.fprintf(".\n");
}

Boolean
Vocab::readIndexMap(File &file, Array<VocabIndex> &map, Boolean limitVocab)
{
    char *line;
    char *strtok_ptr = NULL;

    while ((line = file.getline())) {
	VocabIndex id, newid;

	/*
	 * getline() returns only non-empty lines, so strtok()
	 * will find at least one word.  Anything after the second word
	 * is ignored.
	 */
	strtok_ptr = NULL;
	VocabString idstring = MStringTokUtil::strtok_r(line, wordSeparators, &strtok_ptr);
	VocabString word = MStringTokUtil::strtok_r(0, wordSeparators, &strtok_ptr);

	if (!idstring || (idstring[0] == '.' && idstring[1] == '\0' && word == 0)) {
	    // end of vocabulary table
	    break;
	}

	if (sscanf(idstring, "%u", &id) != 1 || word == 0) {
	    file.position() << "malformed vocab index line\n";
	    return false;
	}

	if (limitVocab) {
	    newid = getIndex(word);
	} else {
	    newid = addWord(word);
	}
	// @kw false positive: SV.TAINTED.ALLOC_SIZE (id)
	map[id] = newid;
    }

    return true;
}


/*
 * Iteration
 */
VocabIter::VocabIter(const Vocab &vocab, Boolean sorted)
    : myIter(vocab.byName, !sorted ? 0 : (int(*)(const char*,const char*))strcmp)
{
}

void
VocabIter::init()
{
    myIter.init();
}

VocabString
VocabIter::next(VocabIndex &index)
{
    VocabString word;
    VocabIndex *idx;
    if ((idx = myIter.next(word))) {
	index = *idx;
	return word;
    } else {
	return 0;
    }
}

