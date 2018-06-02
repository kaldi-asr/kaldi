/*
 * FLMThreads.cc
 *
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#include "FLMThreads.h"
#include "LMThreads.h"
#include "FNgramStats.h"
#include "FactoredVocab.h"
#include "ProductNgram.h"

#if !defined(NO_TLS)
void 
FLMThreads::freeThread() {
    // Call freeThread on the various classes that utilize TLS/static storage.
    FNgram::freeThread();
    FactoredVocab::freeThread();
    ProductNgram::freeThread();

    // FLM module depends on LM, so call its freeThread method.
    LMThreads::freeThread();

    // Free the template TLS variables
    TLSW_FREE(countSentenceWordMatrix);
    TLSW_FREE(countSentenceWidMatrix);
    TLSW_FREE(countSentenceWids);
    TLSW_FREE(readWords);
    TLSW_FREE(readWids);
    TLSW_FREE(readTagsFound);
    TLSW_FREE(writeSpecBuffer);
}
#endif

/* These TLS variables are used by templates in the FLM module. They are 
   defined here so that there is only one instance of each regardless of the
   number of template instantiations, thus allowing us to know them by name 
   and free them. */
TLSW_DEFC(WordMatrix, countSentenceWordMatrix);
TLSW_DEFC(WidMatrix, countSentenceWidMatrix);
TLSW_DEF_ARRAY(VocabIndex, countSentenceWids, maxNumParentsPerChild + 2);
TLSW_DEF_ARRAY(VocabString, readWords, maxNumParentsPerChild+1);
TLSW_DEF_ARRAY(VocabIndex, readWids,  maxNumParentsPerChild+1);
TLSW_DEF_ARRAY(Boolean, readTagsFound, maxNumParentsPerChild+1);
TLSW_DEF_ARRAY(char, writeSpecBuffer, maxLineLength);
