/*
 * VocabDistance.h --
 *	Distance metrics over vocabularies
 *
 * Copyright (c) 2000,2004 SRI International, 2012 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/VocabDistance.h,v 1.7 2015-01-28 23:51:10 wwang Exp $
 *
 */

#ifndef _VocabDistance_h_
#define _VocabDistance_h_

#include "Vocab.h"
#include "SubVocab.h"
#include "VocabMultiMap.h"
#include "VocabMap.h"
#include "Map2.h"

class VocabDistance
{
public:
    virtual ~VocabDistance() {};     /* prevent warning about no virtual dtor */
    virtual double penalty(VocabIndex w1) = 0;	/* insertion/deletion penalty */
    virtual double distance(VocabIndex w1, VocabIndex w2) = 0; /* substitution*/
};

/*
 * Binary distance
 */
class IdentityDistance: public VocabDistance
{
public:
    IdentityDistance(Vocab &vocab) {};
    double penalty(VocabIndex w1) { return 1.0; };
    double distance(VocabIndex w1, VocabIndex w2) 
	{ return (w1 == w2) ? 0.0 : 1.0; };
};

/*
 * Distance constrained by sub-vocabulary membership
 * Return distance
 *	0 if words are identical,
 *	1 if they both belong or don't belong to SubVocab,
 *	and infinity otherwise.
 */
class SubVocabDistance: public VocabDistance
{
public:
    SubVocabDistance(Vocab &vocab, SubVocab &subVocab, double infinity = 10000)
       : subVocab(subVocab), infinity(infinity) {};

    double penalty(VocabIndex w1) { return 1.0; };
    double distance(VocabIndex w1, VocabIndex w2) {
	if (w1 == w2) return 0.0;
	else if ((subVocab.getWord(w1) != 0) == (subVocab.getWord(w2) != 0))
	    return 1.0;
	else return infinity;
    }
private:
    SubVocab &subVocab;
    double infinity;
};

/*
 * Relative phonetic distance according to dictionary
 */
class DictionaryDistance: public VocabDistance
{
public:
    DictionaryDistance(Vocab &vocab, VocabMultiMap &dictionary);
    virtual ~DictionaryDistance() {};

    double penalty(VocabIndex w1) { return 1.0; };
    double distance(VocabIndex w1, VocabIndex w2);

private:
    VocabMultiMap &dictionary;
    Map2<VocabIndex,VocabIndex,double> cache;
};

/*
 * Absolute phonetic distance
 */
class DictionaryAbsDistance: public VocabDistance
{
public:
    DictionaryAbsDistance(Vocab &vocab, VocabMultiMap &dictionary);
    virtual ~DictionaryAbsDistance() {};

    double penalty(VocabIndex w1);
    double distance(VocabIndex w1, VocabIndex w2);

private:
    VocabIndex emptyIndex;
    VocabMultiMap &dictionary;
    Map2<VocabIndex,VocabIndex,double> cache;
};

/*
 * Word distances defined by a matrix
 */
class MatrixDistance: public VocabDistance
{
public:
    MatrixDistance(Vocab &vocab, VocabMap &map);
    virtual ~MatrixDistance() {};

    double penalty(VocabIndex w1);
    double distance(VocabIndex w1, VocabIndex w2);

private:
    VocabIndex deleteIndex;
    VocabMap &map;
};

/*                                                                                                                                                                                                                       * Distance based on stem                                                                                                                                                                                                * Return distance                                                                                                                                                                                                       *      0 if words are identical or its stem are identical,                                                                                                                                                              *      1 otherwise                                                                                                                                                                                                      */
class StemDistance: public VocabDistance
{
 public:
 StemDistance(Vocab &_vocab) : vocab(_vocab) {};
  double penalty(VocabIndex w1) { return 1.0; }
  double distance(VocabIndex w1, VocabIndex w2);

 private:
  Vocab& vocab;
};

#endif /* _VocabDistance_h_ */

