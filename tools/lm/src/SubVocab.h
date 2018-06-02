/*
 * SubVocab.h --
 *	Vocabulary subset class
 *
 * Copyright (c) 1996,1999,2003 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/SubVocab.h,v 1.7 2014-04-22 06:57:46 stolcke Exp $
 *
 */

#ifndef _SubVocab_h_
#define _SubVocab_h_

#include "Vocab.h"

/*
 * SubVocab is a version of Vocab that only contains words also in a base
 * vocabulary.  The indices used by the SubVocab are the same as those in the
 * base vocabulary.
 */
class SubVocab: public Vocab
{
public:
    SubVocab(Vocab &baseVocab, Boolean keepNonwords = false);
    ~SubVocab() { };			/* works around g++ 2.7.2 bug */

    virtual VocabIndex addWord(VocabString name);
    virtual VocabIndex addWord(VocabIndex wid);

    // parameters tied to the base vocabulary 
    virtual Boolean &unkIsWord() { return _baseVocab.unkIsWord(); };
    virtual Boolean &toLower() { return _baseVocab.toLower(); };
    virtual VocabString &metaTag() { return _baseVocab.metaTag(); };
    virtual Boolean isMetaTag(VocabIndex word) const
	{ return _baseVocab.isMetaTag(word); };
    virtual unsigned typeOfMetaTag(VocabIndex word) const
 	{ return _baseVocab.typeOfMetaTag(word); };
    virtual VocabIndex metaTagOfType(unsigned type)
	{ return _baseVocab.metaTagOfType(type); };

    // a non-event in the base vocab that is also in the SubVocab
    // must be a non-event for the SubVocab
    virtual Boolean isNonEvent(VocabIndex word) const
	{ return Vocab::isNonEvent(word) ||
	         _baseVocab.isNonEvent(word); };

    inline Vocab &baseVocab() { return _baseVocab; };

protected:
    Vocab &_baseVocab;
};

#endif /* _SubVocab_h_ */
