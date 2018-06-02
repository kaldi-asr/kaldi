/*
 * ProductVocab.h --
 *	Interface to a product vocabulary for factored language model

 * A product vocab is a vocabulary consists of words that are the product of
 * factors. Adding a product word to a product vocab will factor the word
 * into its factors, add it to its factored vocab, and store the result.
 *
 * Copyright (c) 2003 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/ProductVocab.h,v 1.9 2006/08/11 20:47:15 stolcke Exp $
 *
 */

#ifndef _ProductVocab_h_
#define _ProductVocab_h_

#include "Vocab.h"

#include "FactoredVocab.h"
#include "LHash.h"
#include "Array.h"

class ProductNgram;

class ProductVocab: public Vocab
{
    friend class ProductNgram;

public:
    ProductVocab(VocabIndex start = 0, VocabIndex end = Vocab_None-1);

    // tie parameters to fvocab 
    virtual VocabIndex &unkIndex() { return fvocab.unkIndex(); };
    virtual VocabIndex &ssIndex() { return fvocab.ssIndex(); };
    virtual VocabIndex &seIndex() { return fvocab.seIndex(); };
    virtual VocabIndex &pauseIndex() { return fvocab.pauseIndex(); };
    virtual Boolean &unkIsWord() { return fvocab.unkIsWord(); };
    virtual Boolean &toLower() { return fvocab.toLower(); };
    virtual VocabString &metaTag() { return fvocab.metaTag(); };
    virtual Boolean &nullIsWord() { return fvocab.nullIsWord(); };

    /*
     * Modified Vocab methods
     */
    virtual VocabIndex addWord(VocabString name);
    virtual VocabString getWord(VocabIndex index);
    virtual VocabIndex getIndex(VocabString name,
				VocabIndex unkIndex = Vocab_None);

    virtual inline Boolean isNonEvent(VocabIndex word) const
	{ return Vocab::isNonEvent(word); }

    virtual void memStats(MemStats &stats) const;

    Boolean loadWidFactors(VocabIndex word, VocabIndex *factors);

protected:
    FactoredVocab fvocab;

    // a map from the ID of a product we have encountered (i.e., in Vocab)
    // to a vector of ids for each corresponding factor.
    LHash<VocabIndex, Array<VocabIndex> > productIdToFactorIds;
};

#endif /* _ProductVocab_h_ */
