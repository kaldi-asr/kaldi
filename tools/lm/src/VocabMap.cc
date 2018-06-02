/*
 * VocabMap.cc --
 *	Probabilistic mappings between vocabularies
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International, 2013-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/VocabMap.cc,v 1.18 2016/04/09 06:53:01 stolcke Exp $";
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

#include "VocabMap.h"

#include "Map2.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_MAP2(VocabIndex,VocabIndex,Prob);
#endif

VocabMap::VocabMap(Vocab &v1, Vocab &v2, Boolean logmap)
    : vocab1(v1), vocab2(v2), logmap(logmap)
{
    /*
     * Establish default mappings between special vocab items
     */
    if (v1.ssIndex() != Vocab_None && v2.ssIndex() != Vocab_None) {
	*map.insert(v1.ssIndex(), v2.ssIndex()) = logmap ? LogP_One : 1.0;
    }
    if (v1.seIndex() != Vocab_None && v2.seIndex() != Vocab_None) {
	*map.insert(v1.seIndex(), v2.seIndex()) = logmap ? LogP_One : 1.0;
    }
    if (v1.unkIndex() != Vocab_None && v2.unkIndex() != Vocab_None) {
	*map.insert(v1.unkIndex(), v2.unkIndex()) = logmap ? LogP_One : 1.0;
    }
}

Prob
VocabMap::get(VocabIndex w1, VocabIndex w2)
{
    Prob *prob = map.find(w1, w2);
    if (prob) {
	return *prob;
    } else {
	return 0.0;
    }
}

void
VocabMap::put(VocabIndex w1, VocabIndex w2, Prob prob)
{
    *map.insert(w1, w2) = prob;
}

void
VocabMap::remove(VocabIndex w1, VocabIndex w2)
{
    (void)map.remove(w1, w2);
}

void
VocabMap::remove(VocabIndex w1)
{
    (void)map.remove(w1);
}

Boolean
VocabMap::read(File &file)
{
    char *line;

    while ((line = file.getline())) {
	VocabString words[maxWordsPerLine];

	unsigned howmany = Vocab::parseWords(line, words, maxWordsPerLine);

	if (howmany == maxWordsPerLine) {
	    file.position() << "map line has too many fields\n";
	    return false;
	}

	/*
	 * The first word is always the source of the map
	 */
	VocabIndex w1 = vocab1.addWord(words[0]);

	if (map.numEntries(w1) > 0) {
	    file.position() << "warning: map redefining entry "
			    << words[0] << endl;
	    map.remove(w1);
	}

	/*
	 * Parse the remaining words as either probs or target words
	 */
	unsigned i = 1;

	while (i < howmany) {
	    Prob prob;

	    VocabIndex w2 = vocab2.addWord(words[i++]);

	    if (i < howmany && parseProbOrLogP(words[i], prob, logmap)) {
		i ++;
	    } else {
		prob = logmap ? LogP_One : 1.0;
	    }

	    *(map.insert(w1, w2)) = prob;
	}
    }

    return true;
}

/* 
 * Read classes(5) format file, interpreted as VocabMap
 * (mostly borrowed from ClassNgram::readClasses())
 */
Boolean
VocabMap::readClasses(File &file)
{
    char *line;

    while ((line = file.getline())) {
	VocabString words[maxWordsPerLine];

	unsigned howmany = Vocab::parseWords(line, words, maxWordsPerLine);

	if (howmany == maxWordsPerLine) {
	    file.position() << "class definition has too many fields\n";
	    return false;
	}

	/*
	 * First word contains class name
	 */
	VocabIndex clasz = vocab2.addWord(words[0]);

	Prob prob = logmap ? LogP_One : 1.0;
	unsigned numExpansionWords;

	/*
	 * If second word is numeral, assume it's the class expansion prob
	 */
	if (howmany > 1 && parseProbOrLogP(words[1], prob, logmap)) {
	    numExpansionWords = howmany - 2;
	} else {
	    numExpansionWords = howmany - 1;
	}

	if (numExpansionWords != 1) {
	    file.position() << "class definition must have exactly one word\n";
	    return false;
	}

	VocabIndex expansionWord = vocab1.addWord(words[howmany - 1]);

	*(map.insert(expansionWord, clasz)) = prob;
    }
	
    return true;
}

Boolean
VocabMap::write(File &file)
{
    Map2Iter<VocabIndex,VocabIndex,Prob> iter1(map);

    VocabIndex w1;

    while (iter1.next(w1)) {

	VocabString word1 = vocab1.getWord(w1);
	assert(word1 != 0);

	file.fprintf("%s", word1);

	Map2Iter2<VocabIndex,VocabIndex,Prob> iter2(map, w1);

	VocabIndex w2;
	Prob *prob;

	unsigned i = 0;
	while ((prob = iter2.next(w2))) {
	    VocabString word2 = vocab2.getWord(w2);
	    assert(word1 != 0);

	    char sep = (i++ == 0)  ? '\t' : ' ';

	    if (*prob == (logmap ? LogP_One : 1.0)) {
		file.fprintf("%c%s", sep, word2);
	    } else {
		file.fprintf("%c%s %.*lg", sep, word2,
					   Prob_Precision, (double)*prob);
	    }
	}
	file.fprintf("\n");
    }
    return true;
}

/*
 * Write VocabMap in bigram count file format
 */
Boolean
VocabMap::writeBigrams(File &file)
{
    Map2Iter<VocabIndex,VocabIndex,Prob> iter1(map);

    VocabIndex w1;

    while (iter1.next(w1)) {
	VocabString word1 = vocab1.getWord(w1);
	assert(word1 != 0);

	Map2Iter2<VocabIndex,VocabIndex,Prob> iter2(map, w1);

	VocabIndex w2;
	Prob *prob;

	unsigned i = 0;
	while ((prob = iter2.next(w2))) {
	    VocabString word2 = vocab2.getWord(w2);
	    assert(word1 != 0);

	    // prob = P(word1|word2), hence the bigram word order
	    file.fprintf("%s %s\t%.*lg\n", word2, word1,
					   Prob_Precision, (double)*prob);
	}
    }
    return true;
}

/*
 * Iteration over map entries
 */

VocabMapIter::VocabMapIter(VocabMap &vmap, VocabIndex w) :
   mapIter(vmap.map, w)
{
}

void
VocabMapIter::init()
{
    mapIter.init();
}

Boolean
VocabMapIter::next(VocabIndex &w, Prob &prob)
{
    Prob *myprob = mapIter.next(w);

    if (myprob) {
	prob = *myprob;
	return true;
    } else {
	return false;
    }
}

