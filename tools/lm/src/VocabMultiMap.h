/*
 * VocabMultiMap.h --
 *	Probabilistic mappings between a vocabulary and strings from
 *	another vocabulary (as in dictionaries).
 *
 * Copyright (c) 2000 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/VocabMultiMap.h,v 1.4 2011/01/12 20:10:59 stolcke Exp $
 *
 */

#ifndef _VocabMultiMap_h_
#define _VocabMultiMap_h_

#include "Boolean.h"
#include "Prob.h"
#include "Vocab.h"
#include "Map2.h"

class VocabMultiMap
{
    friend class VocabMultiMapIter;

public:
    VocabMultiMap(Vocab &v1, Vocab &v2, Boolean logmap = false);
    virtual ~VocabMultiMap() {};
    
    Prob get(VocabIndex w1, const VocabIndex *w2);
    void put(VocabIndex w1, const VocabIndex *w2, Prob prob);
    void remove(VocabIndex w1, const VocabIndex *w2);

    virtual Boolean read(File &file, Boolean limitVocab = false);
    virtual Boolean write(File &file);
    
    Vocab &vocab1;
    Vocab &vocab2;

private:
    /*
     * The map is implemented by a two-level map where the first index is
     * from vocab1 and the second from strings over vocab2
     */
    Map2<VocabIndex,const VocabIndex *,Prob> map;

    Boolean logmap;			/* treat probabilities as log probs */
};

/*
 * Iteration over the mappings of a word
 */
class VocabMultiMapIter
{
public:
    VocabMultiMapIter(VocabMultiMap &vmap, VocabIndex w);

    void init();
    const VocabIndex *next(Prob &prob);

private:
    Map2Iter2<VocabIndex,const VocabIndex *,Prob> mapIter;
};

#endif /* _VocabMultiMap_h_ */

