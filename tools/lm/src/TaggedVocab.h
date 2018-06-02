/*
 * TaggedVocab.h --
 *	Interface to a tagged vocabulary.
 *
 * A tagged vocabulary consists of word that have been labelled with class
 * labels, e.g., "boat/N" .
 * This class provides support for encoding/decoding tagged vocab items.
 *
 * Copyright (c) 1995-2010 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/TaggedVocab.h,v 1.7 2010/06/02 05:49:58 stolcke Exp $
 *
 */

#ifndef _TaggedVocab_h_
#define _TaggedVocab_h_

#include "Vocab.h"
#include "LHash.h"

/*
 * TaggedVocab extends the standard Vocab functionality by supporting
 * but vocab words alone and word/tag combinations.
 * It keeps an extra Vocab object on the side to manage the tag set.
 * Word/tag combinations are encoded as integers by or-ing the
 * index for the word and the shifted tag index.
 */

const unsigned TagShift = 5 * sizeof(VocabIndex);
				/* 5/8 of the bits are used to encode words;
				 * 3/8 are used to encode tags */
const unsigned maxTaggedIndex = (1U<<TagShift)-2;
const unsigned maxTagIndex = (1U<<(sizeof(VocabIndex)*8 - TagShift))-1;

const unsigned Tagged_None = (1U<<TagShift)-1;
const unsigned Tag_None = 0;		/* the tag of an untagged word */

class TaggedVocab: public Vocab
{
public:
    TaggedVocab(VocabIndex start = 0, VocabIndex end = maxTaggedIndex);
    ~TaggedVocab();

    /*
     * Modified Vocab methods
     */
    virtual VocabIndex addWord(VocabString name);
    virtual VocabString getWord(VocabIndex index);
    virtual VocabIndex getIndex(VocabString name,
				    VocabIndex unkIndex = Vocab_None);
    virtual void remove(VocabString name);
    virtual void remove(VocabIndex index);

    virtual void write(File &file, Boolean sorted = true) const;

    /*
     * Tagged index accessors/constructors
     */
    static inline VocabIndex getTag(VocabIndex index) {
	return index >> TagShift;
    };
    static inline VocabIndex unTag(VocabIndex index) {
	return index & ((1U<<TagShift) - 1);
    };
    static VocabIndex tagWord(VocabIndex word, VocabIndex tag) {
	return (tag << TagShift) | (word & ((1U<<TagShift) - 1));
    };
    static Boolean isTag(VocabIndex index) {
	return unTag(index) == Tagged_None;
    }

    virtual inline Boolean isNonEvent(VocabIndex word) const {
	return isTag(word) || Vocab::isNonEvent(word);
    }

    virtual void memStats(MemStats &stats) const;

    /*
     * Access to the tag set
     */
    inline Vocab &tags() { return _tags; };

private:
    Vocab _tags;				/* the tag vocabulary */

    LHash<VocabIndex, VocabString> taggedWords;	/* cache of word/tag strings */
};

#endif /* _TaggedVocab_h_ */
