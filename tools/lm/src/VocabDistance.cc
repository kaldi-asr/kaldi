/*
 * VocabDistance.cc --
 *	Distance metrics over vocabularies
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2000-2010 SRI International, 2012 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/VocabDistance.cc,v 1.6 2015-01-28 23:52:00 wwang Exp $";
#endif

#include "VocabDistance.h"
#include "WordAlign.h"

#include <string>
#include <vector>

#include "Map2.cc"
#ifdef INSTANTIATE_TEMPLATES
//INSTANTIATE_MAP2(VocabIndex,VocabIndex,double);
#endif

/*
 * Relative phonetic distance
 * (Levenshtein distance of pronunciations normalized by pronunciation length)
 */

DictionaryDistance::DictionaryDistance(Vocab &vocab, VocabMultiMap &dictionary)
    : dictionary(dictionary)
{
    /* 
     * Dictionary must be applicable to word vocabulary
     */
    assert(&vocab == &dictionary.vocab1);
}

double
DictionaryDistance::distance(VocabIndex w1, VocabIndex w2)
{
    if (w2 < w1) {
	VocabIndex h = w2;
	w2 = w1;
	w1 = h;
    } else if (w1 == w2) {
	return 0.0;
    }

    double *dist;
    Boolean foundP;
    dist = cache.insert(w1, w2, foundP);

    if (foundP) {
	return *dist;
    } else {
	double minDistance = -1.0;

	VocabMultiMapIter iter1(dictionary, w1);
	const VocabIndex *pron1;
	Prob p1;

	while ((pron1 = iter1.next(p1))) {
	    unsigned len1 = Vocab::length(pron1);

	    VocabMultiMapIter iter2(dictionary, w2);
	    const VocabIndex *pron2;
	    Prob p2;

	    while ((pron2 = iter2.next(p2))) {
		unsigned len2 = Vocab::length(pron2);

		unsigned maxLen = (len1 > len2) ? len1 : len2;

		unsigned sub, ins, del;
		double thisDistance =
			(double)wordError(pron1, pron2, sub, ins, del) /
					(maxLen > 0 ? maxLen : 1);

		if (minDistance < 0.0 || thisDistance < minDistance) {
		    minDistance = thisDistance;
		}
	    }
	}

	/*
	 * If no dictionary entries were found use 1 as default distance
	 */
	if (minDistance < 0.0) {
	    minDistance = 1.0;
	}

	*dist = minDistance;
	return minDistance;
    }
}

/*
 * Absolute phonetic distance
 */

const double defaultDistance = 1.0; /* for word without dictionary entries */
const char *emptyWord = "*EMPTY*WORD*";

DictionaryAbsDistance::DictionaryAbsDistance(Vocab &vocab,
						VocabMultiMap &dictionary)
    : dictionary(dictionary)
{
    /* 
     * Dictionary must be applicable to word vocabulary
     */
    assert(&vocab == &dictionary.vocab1);

    emptyIndex = vocab.addWord(emptyWord);
}

double
DictionaryAbsDistance::penalty(VocabIndex w)
{
    double *dist;
    Boolean foundP;
    dist = cache.insert(w, emptyIndex, foundP);

    if (foundP) {
	return *dist;
    } else {
	double minLength = -1.0;

	VocabMultiMapIter iter(dictionary, w);
	const VocabIndex *pron;
	Prob p;

	while ((pron = iter.next(p))) {
	    unsigned len = Vocab::length(pron);

	    if (minLength < 0.0 || len < minLength) {
		minLength = len;
	    }
	}

	/*
	 * If no dictionary entries were found use default distance
	 */
	if (minLength < 0.0) {
	    minLength = defaultDistance;
	}

	*dist = minLength;
	return minLength;
    }
}

double
DictionaryAbsDistance::distance(VocabIndex w1, VocabIndex w2)
{
    if (w2 < w1) {
	VocabIndex h = w2;
	w2 = w1;
	w1 = h;
    } else if (w1 == w2) {
	return 0.0;
    }

    double *dist;
    Boolean foundP;
    dist = cache.insert(w1, w2, foundP);

    if (foundP) {
	return *dist;
    } else {
	double minDistance = -1.0;

	VocabMultiMapIter iter1(dictionary, w1);
	const VocabIndex *pron1;
	Prob p1;

	while ((pron1 = iter1.next(p1))) {
	    VocabMultiMapIter iter2(dictionary, w2);
	    const VocabIndex *pron2;
	    Prob p2;

	    while ((pron2 = iter2.next(p2))) {
		unsigned sub, ins, del;
		double thisDistance =
			(double)wordError(pron1, pron2, sub, ins, del);

		if (minDistance < 0.0 || thisDistance < minDistance) {
		    minDistance = thisDistance;
		}
	    }
	}

	/*
	 * If no dictionary entries were found use default distance
	 */
	if (minDistance < 0.0) {
	    minDistance = defaultDistance;
	}

	*dist = minDistance;
	return minDistance;
    }
}

/*
 * Word distances defined by a matrix
 */

const VocabString deleteWord = "*DELETE*"; 

MatrixDistance::MatrixDistance(Vocab &vocab, VocabMap &map)
    : map(map)
{
    deleteIndex = vocab.addWord(deleteWord);
}

double
MatrixDistance::penalty(VocabIndex w1)
{
     return distance(w1, deleteIndex);
}

double
MatrixDistance::distance(VocabIndex w1, VocabIndex w2)
{
     Prob d = map.get(w1, w2);
     if (d == 0.0) {
	Prob d2 = map.get(w2, w1);
	if (d2 > 0.0) {
	    return d2;
	}
     }
     return d;
}

/*
 * Word distances based on prefix/stem/suffix
 */
static const string RXwhite = " \t\n";
static const string prefix = "#";
static const string suffix= "@";

// Split A_STRING into PARTS; the split is based on a separator_list;
// returns the number of fields resulting from the split
static unsigned int split(const std::string& a_string, std::vector<std::string>& parts, std::string separator_list)
{

  //assert(parts.empty() == true);
  parts.clear();
  unsigned int _ret = 0;

  unsigned int first = 0;
  unsigned int last  = 0;
  while(true){
    first = a_string.find_first_not_of(separator_list, last);
    if(first >= a_string.length()){
      break;
    } else if((first > 0) && (last == 0)){// it starts with a separator
      parts.push_back("");
      _ret++;
    }
    last  = a_string.find_first_of(separator_list, first);
    assert(first < last);
    if(last > a_string.length()){
      last = a_string.length();
    }
    string element = a_string.substr(first,last-first);
    parts.push_back(element);
    _ret++;
  }
  assert(_ret == parts.size());
  return _ret;
}

double
StemDistance::distance(VocabIndex w1, VocabIndex w2)
{   
  if (w1 == w2) return 0;

  // Decompose w1 and w2 into prefix#stem@suffix
  std::string str1 = vocab.getWord(w1);
  std::string str2 = vocab.getWord(w2);
  std::vector<std::string> tk1, tk2;

  // string string into
  split(str1, tk1, prefix);
  split(str2, tk2, prefix);

  // the last part of tk1 and tk2 must contain stem info
  std::string last1 = tk1[tk1.size()-1];
  std::string last2 = tk2[tk2.size()-1];

  tk1.clear();
  tk2.clear();

  // strip suffix
  split(last1, tk1, suffix);
  split(last2, tk2, suffix);

  // Take the first one as stem
  if (tk1[0] == tk2[0]) {
    return 0;
  } else {
    return 1;
  }
}

