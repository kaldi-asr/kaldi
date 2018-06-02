/*
 * VocabMultiMap.cc --
 *	Probabilistic mappings between a vocabulary and strings from
 *	another vocabulary (as in dictionaries).
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2000-2011 SRI International, 2013-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/VocabMultiMap.cc,v 1.10 2016/04/09 06:53:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "VocabMultiMap.h"

#include "Array.cc"
#include "Map2.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_MAP2(VocabIndex,const VocabIndex *,Prob);
#endif

VocabMultiMap::VocabMultiMap(Vocab &v1, Vocab &v2, Boolean logmap)
    : vocab1(v1), vocab2(v2), logmap(logmap)
{
}

Prob
VocabMultiMap::get(VocabIndex w1, const VocabIndex *w2)
{
    Prob *prob = map.find(w1, w2);
    if (prob) {
	return *prob;
    } else {
	return 0.0;
    }
}

void
VocabMultiMap::put(VocabIndex w1, const VocabIndex *w2, Prob prob)
{
    *map.insert(w1, w2) = prob;
}

void
VocabMultiMap::remove(VocabIndex w1, const VocabIndex *w2)
{
    (void)map.remove(w1, w2);
}

Boolean
VocabMultiMap::read(File &file, Boolean limitVocab)
{
    char *line;

    while ((line = file.getline())) {
	VocabString words[maxWordsPerLine];

	unsigned howmany = Vocab::parseWords(line, words, maxWordsPerLine);

	if (howmany == maxWordsPerLine) {
	    file.position() << "map line has too many fields\n";
	    return false;
	}

	VocabIndex w1;

	/*
	 * The first word is always the source of the map.
	 * If limitVocab is in effect and the word is OOV, skip this entry.
	 */
	if (limitVocab && !vocab1.checkWords(&words[0], &w1, 1)) {
	    continue;
	}

	w1 = vocab1.addWord(words[0]);

	/*
	 * The second word is an optional probability
	 */
	double prob;
	unsigned startWords;
	if (howmany > 1 && parseProbOrLogP(words[1], prob, logmap)) {
	    startWords = 2;
	} else {
	    prob = logmap ? LogP_One : 1.0;
	    startWords = 1;
	}

	makeArray(VocabIndex, mappedWords, howmany);
	vocab2.addWords(&words[startWords], mappedWords, howmany);

	*(map.insert(w1, mappedWords)) = prob;
    }

    return true;
}

Boolean
VocabMultiMap::write(File &file)
{
    Map2Iter<VocabIndex,const VocabIndex *,Prob>
				iter1(map, vocab1.compareIndex());

    VocabIndex w1;

    while (iter1.next(w1)) {
	VocabString word1 = vocab1.getWord(w1);
	assert(word1 != 0);

	Map2Iter2<VocabIndex,const VocabIndex *,Prob> iter2(map, w1);

	const VocabIndex *w2;
	Prob *prob;

	while ((prob = iter2.next(w2))) {
	    if (*prob == (logmap ? LogP_One : 1.0)) {
		file.fprintf("%s\t", word1);
	    } else {
		file.fprintf("%s\t%.*lg ", word1,
					   Prob_Precision, (double)*prob);
	    }

	    unsigned numWords = Vocab::length(w2);
	    makeArray(VocabString, mappedWords, numWords + 1);
	    vocab2.getWords(w2, mappedWords, numWords + 1);

	    Vocab::write(file, mappedWords);
	    file.fprintf("\n");
	}
    }
    return true;
}

/*
 * Iteration over map entries
 */

VocabMultiMapIter::VocabMultiMapIter(VocabMultiMap &vmap, VocabIndex w) :
   mapIter(vmap.map, w)
{
}

void
VocabMultiMapIter::init()
{
    mapIter.init();
}

const VocabIndex *
VocabMultiMapIter::next(Prob &prob)
{
    const VocabIndex *w;
    Prob *myprob = mapIter.next(w);

    if (myprob) {
	prob = *myprob;
	return w;
    } else {
	return 0;
    }
}

