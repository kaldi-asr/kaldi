/*
 * LMClient.h
 *	Client-side for network-based LM
 *
 * Copyright (c) 2007 SRI International, 2012 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/LMClient.h,v 1.8 2012/07/06 01:02:11 stolcke Exp $
 *
 */

#ifndef _LMClient_h_
#define _LMClient_h_

#include <stdio.h>

#if defined(_MSC_VER) || defined(WIN32)
#include <winsock.h>
#else
typedef int	SOCKET;		// for MS compatibility
#endif

#include "LM.h"
#include "Ngram.h"
#include "Array.h"

class LMClient: public LM
{
public:
    LMClient(Vocab &vocab, const char *server, unsigned order = 0, unsigned cacheOrder = 0);
    ~LMClient();


    LogP wordProb(VocabIndex word, const VocabIndex *context);
    void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    LogP contextBOW(const VocabIndex *context, unsigned length);

    Boolean addUnkWords() { return true; };	/* Words are implicitly added
						 * to vocab so we can transmit
						 * them over the network */

protected:
    unsigned order;		/* maximum N-gram length */
    char serverHost[256];
    unsigned serverPort;
    SOCKET serverSocket;

    unsigned cacheOrder;	/* max N-gram length to cache */
    Ngram probCache;		/* cache for wordProb() results  */
    struct _CIC {
	VocabIndex word;
	Array<VocabIndex> context;
	void *id;
	unsigned length;
    } contextIDCache;		/* single-result cache for contextID() */
    struct _CBC {
	Array<VocabIndex> context;
	unsigned length;
	LogP bow;
    } contextBOWCache;		/* single-result cache for contextBOW() */
};

#endif /* _LMClient_h_ */
