/*
 * RefList.h --
 *	Lists of words strings identified by string ids
 *
 * Copyright (c) 1998-2006 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/RefList.h,v 1.7 2012/10/29 17:25:05 mcintyre Exp $
 *
 */

#ifndef _RefList_h_
#define _RefList_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif

#include "Boolean.h"
#include "File.h"
#include "Vocab.h"
#include "LHash.h"
#include "Array.h"

typedef const char *RefString;

RefString idFromFilename(const char *filename);
void RefList_freeThread();

class RefList 
{
public:
    RefList(Vocab &vocab, Boolean haveIDs = true);
    ~RefList();

    Boolean read(File &file, Boolean addWords = false);
    Boolean write(File &file);

    VocabIndex *findRef(RefString id);
    VocabIndex *findRefByNumber(unsigned id);

private:
    Vocab &vocab;
    Boolean haveIDs;				// whether refs have string IDs
    Array<VocabIndex *> refarray;		// by number
    LHash<RefString, VocabIndex *> reflist;	// by ID
};

#endif /* _RefList_h_ */
