/*
 * NBestSet.h --
 *	Sets of N-best lists
 *
 * Copyright (c) 1998-2008 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/NBestSet.h,v 1.11 2012/07/03 21:19:28 zj Exp $
 *
 */

#ifndef _NBestSet_h_
#define _NBestSet_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>

#include "Boolean.h"
#include "File.h"
#include "Vocab.h"
#include "LHash.h"
#include "NBest.h"
#include "RefList.h"
#include "Debug.h"

/*
 * An element of the NBest set: an NBest List and associated information
 */
typedef struct {
    const char *filename;
    NBestList *nbest;
} NBestSetElement;

class NBestSet: public Debug
{
    friend class NBestSetIter;

public:
    NBestSet(Vocab &vocab, RefList &refs, unsigned maxNbest = 0,
		    Boolean incremental = false, Boolean multiwords = false);
    NBestSet(Vocab &vocab, RefList &refs, unsigned maxNbest,
		    Boolean incremental, const char *multiChar);
    virtual ~NBestSet();

    Boolean read(File &file);
    Boolean readSRInterpFormat(File & file, LHash<RefString, NBestScore **> & allScores,
			       unsigned numScores, RefString * feats, double * scales);

    Boolean readSRInterpCountsFile(File & file, unsigned & numRefWords, unsigned & bleuNgrams);
    unsigned numElements() { return lists.numEntries(); };

    Vocab &vocab;
    Boolean warn;

private:
    unsigned maxNbest;
    Boolean incremental;
    const char *multiChar;
    RefList &refs;

    LHash<RefString,NBestSetElement> lists;

    Boolean readList(RefString id, NBestSetElement &elt);
    Boolean readListSRInterpFormat(RefString id, NBestSetElement & elt, 
				   LHash<RefString, Array<NBestScore> * > & nbestScores);
    void freeList(NBestSetElement &elt);
};

class NBestSetIter
{
public:
    NBestSetIter(NBestSet &set);

    void init();
    NBestList *next(RefString &id);
    const char *nextFile(RefString &id);

private:
    NBestSet &mySet;
    LHashIter<RefString,NBestSetElement> myIter;
    NBestSetElement *lastElt;
};

#endif /* _NBestSet_h_ */
