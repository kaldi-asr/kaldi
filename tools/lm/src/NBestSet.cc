/*
 * NBestSet.cc --
 *	Set of N-best lists
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1998-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NBestSet.cc,v 1.15 2014-08-29 21:35:48 frandsen Exp $";
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

#include "NBestSet.h"
#include "NullLM.h"
#include "MultiwordVocab.h"

#include "LHash.cc"
#include "Array.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_ARRAY(LogP);
INSTANTIATE_LHASH(RefString, NBestSetElement);
INSTANTIATE_LHASH(RefString, Array<LogP>*);
INSTANTIATE_LHASH(RefString, LogP); 
#endif

/*
 * Debugging levels used
 */
#define DEBUG_READ	1

NBestSet::NBestSet(Vocab &vocab, RefList &refs,
		    unsigned maxNbest, Boolean incremental, Boolean multiwords)
    : vocab(vocab), warn(true), maxNbest(maxNbest), incremental(incremental),
      multiChar(multiwords ? MultiwordSeparator : 0), refs(refs)
{
}

NBestSet::NBestSet(Vocab &vocab, RefList &refs,
		    unsigned maxNbest, Boolean incremental,
		    const char *multiChar)
    : vocab(vocab), warn(true), maxNbest(maxNbest), incremental(incremental),
      multiChar(multiChar), refs(refs)
{
}

NBestSet::~NBestSet()
{
    LHashIter<RefString, NBestSetElement> iter(lists);

    RefString id;
    NBestSetElement *elt;
    while ((elt = iter.next(id))) {
	delete [] elt->filename;
	delete elt->nbest;
    }
}

Boolean
NBestSet::read(File &file)
{
    char *line;
    
    while ((line = file.getline())) {
	VocabString filename[2];

	/*
	 * parse filename from input file
	 */
	if (Vocab::parseWords(line, filename, 2) != 1) {
	    file.position() << "one filename expected\n";
	    return false;
	}
	
	/*
	 * Locate utterance id in filename
	 */
	RefString id = idFromFilename(filename[0]);
	NBestSetElement *elt = lists.insert(id);

	delete [] elt->filename;
	elt->filename = new char[strlen(filename[0]) + 1];
	assert(elt->filename != 0);
	strcpy((char *)elt->filename, filename[0]);

	delete elt->nbest;
	if (incremental) {
	    elt->nbest = 0;
	} else if (!readList(id, *elt)) {
	    file.position() << "error reading N-best list\n";
	    if (elt->nbest) {
		delete elt->nbest;
	    }
	    elt->nbest = 0;
	}
    }

    return true;
}

// feature names must be unique
Boolean
NBestSet::readSRInterpFormat(File &file, LHash<RefString, NBestScore **> & allScores, 
			     unsigned numScores, RefString * feats, double * scales)
{
    const unsigned initialSize = 128;
    LHash<RefString, Array<NBestScore> * > scores;

    unsigned n;
    for (n = 0; n < numScores; n++) {
	*scores.insert(feats[n]) = new Array<NBestScore>(0, initialSize);
    }

    char *line;
    while ((line = file.getline())) {
	VocabString filename[2];

	/*
	 * parse filename from input file
	 */
	if (Vocab::parseWords(line, filename, 2) != 1) {
	    file.position() << "one filename expected\n";
	    return false;
	}
	
	/*
	 * Locate utterance id in filename
	 */
	RefString id = idFromFilename(filename[0]);
	NBestSetElement *elt = lists.insert(id);

	delete [] elt->filename;
	elt->filename = new char[strlen(filename[0]) + 1];
	assert(elt->filename != 0);
	strcpy((char *)elt->filename, filename[0]);

	delete elt->nbest;
	if (incremental) {
	    elt->nbest = 0;
	} else if (!readListSRInterpFormat(id, *elt, scores)) {
	    file.position() << "error reading N-best list\n";
	    if (elt->nbest) {
		delete elt->nbest;
	    }
	    elt->nbest = 0;
	} else {
	    // copy scores;
	    NBestList * nbest = elt->nbest;
	    unsigned numHyps = nbest->numHyps();
	    NBestScore ** mat = new NBestScore * [ numScores ];
	    for (n = 0; n < numScores; n++) {	     
	        mat[n] = new NBestScore [ numHyps ];
		Array<NBestScore>** ptr = scores.find(feats[n]);
		if (!ptr || !*ptr || !**ptr) {
		    // Unexpected lookup error
		    continue;
		}
		Array<NBestScore> &array = **ptr;
		double scale = scales[n];
		for (unsigned j = 0; j < numHyps; j++) {
		    mat[n][j] = (NBestScore) (array[j]/scale);
		}
	    }
	    *allScores.insert(id) = mat;
	}
    }

    for (n = 0; n < numScores; n++) {
	Array<NBestScore>** ptr = scores.find(feats[n]);
	if (ptr && *ptr) {
	    delete *ptr;
	}
    }
    
    return true;
}

Boolean
NBestSet::readSRInterpCountsFile(File & file, unsigned & numRefWords, unsigned &bleuNgram)
{
    VocabString fields[MAX_BLEU_NGRAM * 2 + 6];
    const char * sep = "|||";
    
    char * line;
    unsigned lno = 0;
    numRefWords = 0;
    string lastSentID(".");
    unsigned nr = 0, nh = 0;

    while ((line = file.getline())) {
        string str = line;
	lno ++;
	// format is: SENTID HYPID ||| BLEU_COUNTS REF_LEN TER ||| FEATURE Key-val pairs 
	// we ignore the feature fields as they have been read with SRInterp n-best list

	// the first field is sentid
	unsigned num = Vocab::parseWords(line, fields, MAX_BLEU_NGRAM * 2 + 6);
	if (num < 8) {
	    cerr << "line (" << lno <<") : too few fields: " << str.c_str();
	    continue;
	} else if(strcmp(fields[2], sep)) {
	   cerr << "line (" << lno <<") format error -- 3rd field is not separator: " << str.c_str();
	   continue;
	}

	const char * sentid = fields[0];
	NBestSetElement * ele = lists.find(sentid);
	if (ele == 0) {
	    cerr << "cannot find sentid: " << sentid << endl;
	    continue;
	}

	if (strcmp(lastSentID.c_str(), sentid)) {
	    lastSentID = sentid;
	    if (nh) {
	      numRefWords += (nr + (nh >>1)) / nh;
	    }
	    nh = 0;
	    nr = 0;
	}
      
	// the second field is hyp rank (0-based);
	unsigned hypid = atoi(fields[1]);
	if (hypid >= ele->nbest->numHyps()) {
	    cerr << "sentence " << sentid << ": hypid (" << hypid << ") is out of range (0-" 
		 << ele->nbest->numHyps() - 1 << ")\n";
	    continue;
	}

	NBestHyp & h = ele->nbest->getHyp(hypid);      
      
	unsigned i;
	for (i = num - 1; i > 2; i--) {
	if (strcmp(fields[i], sep) == 0)
	  break;
	}
      
	if (i == 2) {
	    cerr << "line (" << lno << ") format error -- two many bleu counters: " << str.c_str();
	    continue;
	}

	unsigned n = (( i - 5 ) >> 1);
	if (bleuNgram == 0) {
	    bleuNgram = n;
	} else if (bleuNgram != n) {
	    cerr << "inconsistent bleu order in different lines" << endl;
	    return false;
	}

	unsigned numWords = atoi(fields[4]);
	if (numWords != h.numWords) {
	    cerr << "line(" << lno << "): inconsistent number of words: " 
		 << numWords << " vs " << h.numWords;
	    cerr << " in sentence: " << str.c_str(); 
	}

	if (!h.bleuCount) 
	  h.bleuCount = new BleuCount;
	for (i = 0; i < n; i++) {
	    h.bleuCount->correct[i] = (unsigned short) atoi(fields[3 + i*2]);
	}

	h.closestRefLeng = atoi(fields[3 + n*2]);
	h.numErrors = (int)(atof(fields[4 + n*2]) * h.closestRefLeng + .5);
	
	nr += h.closestRefLeng;
	nh ++;
    }

    if (nh) {
      numRefWords += (nr + (nh>>1)) / nh;
    }

    return true;
}


Boolean
NBestSet::readList(RefString id, NBestSetElement &elt)
{
    elt.nbest = new NBestList(vocab, maxNbest, multiChar);
    assert(elt.nbest != 0);

    /*
     * Read Nbest list
     */
    File nbestFile(elt.filename, "r");
    if (!elt.nbest->read(nbestFile)) {
	return false;
    }

    if (debug(DEBUG_READ)) {
	dout() << id << ": " << elt.nbest->numHyps() << " hyps\n";
    }

    /*
     * Remove pause and noise tokens
     */
    NullLM nullLM(vocab);
    elt.nbest->removeNoise(nullLM);

    /*
     * Compute word errors
     */
    VocabIndex *ref = refs.findRef(id);

    if (!ref) {
	if (warn) {
	    cerr << "No reference found for " << id << endl;
	}
    } else {
	unsigned subs, inss, dels;
	elt.nbest->wordError(ref, subs, inss, dels);
    }

    return true;
}

Boolean
NBestSet::readListSRInterpFormat(RefString id, NBestSetElement &elt, 
				 LHash<RefString, Array<NBestScore>* > & nbestScores )
{
    elt.nbest = new NBestList(vocab, maxNbest, multiChar);
    assert(elt.nbest != 0);

    /*
     * Read Nbest list
     */
    File nbestFile(elt.filename, "r");
    if (!elt.nbest->readSRInterpFormat(nbestFile, nbestScores)) {
	return false;
    }

    if (debug(DEBUG_READ)) {
	dout() << id << ": " << elt.nbest->numHyps() << " hyps\n";
    }

    return true;
}

void
NBestSet::freeList(NBestSetElement &elt)
{
    delete elt.nbest;
    elt.nbest = 0;
}

/*
 * Iteration over N-best sets
 */

NBestSetIter::NBestSetIter(NBestSet &set)
    : mySet(set), myIter(set.lists, strcmp), lastElt(0)
{
}

void
NBestSetIter::init()
{
    myIter.init();
    lastElt = 0;
}

NBestList *
NBestSetIter::next(RefString &id)
{
    if (lastElt) {
	mySet.freeList(*lastElt);
	lastElt = 0;
    }
	
    NBestSetElement *nextElt = myIter.next(id);
    if (nextElt) {
	if (mySet.incremental) {
	    if (!mySet.readList(id, *nextElt)) {
	        cerr << "error reading N-best list" << id << endl;
		return 0;
	    } else {
		lastElt = nextElt;
	    }
	}
	return nextElt->nbest;
    } else {
	return 0;
    }
}

const char *
NBestSetIter::nextFile(RefString &id)
{
    if (lastElt) {
	mySet.freeList(*lastElt);
	lastElt = 0;
    }
	
    NBestSetElement *nextElt = myIter.next(id);
    if (nextElt) {
	return nextElt->filename;
    } else {
	return 0;
    }
}

