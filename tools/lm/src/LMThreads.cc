/* 
 * LMThreads.cc
 *
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#include "LMThreads.h"

#include "LM.h"
#include "LMStats.h"
#include "NgramStats.h"
#include "NBest.h"
#include "DStructThreads.h"
#include "RefList.h"
#include "Vocab.h"
#include "WordAlign.h"
#include "WordMesh.h"
#include "XCount.h"

void 
LMThreads::freeThread() {
    // Call freeThread on the various classes that utilize TLS/static storage.
    LM::freeThread();
    LMStats::freeThread(); 
    NBestHyp::freeThread();
    Vocab::freeThread();
    WordMesh::freeThread();
    XCount::freeThread();

    RefList_freeThread();
    wordError_freeThread();

    // Call freeThread() on this module's dependency
    DStructThreads::freeThread();

    TLSW_FREE(countSentenceWidsTLS);
    TLSW_FREE(writeBufferTLS);
}

/* These TLS variables are used by templates in the LM module. They are 
   defined here so that there is only one instance of each regardless of the
   number of template instantiations, thus allowing us to know them by name 
   and free them. */
TLSW_DEF_ARRAY(VocabIndex, countSentenceWidsTLS, maxWordsPerLine+3);
TLSW_DEF_ARRAY(char, writeBufferTLS, maxLineLength);
