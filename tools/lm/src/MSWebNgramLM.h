/*
 * MSWebNgramLM.h
 *	Client-side for Microsoft Web Ngram LM
 *	(see http://web-ngram.research.microsoft.com/info/ for details)
 *
 * Copyright (c) 2012 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/MSWebNgramLM.h,v 1.5 2012/08/17 18:14:05 stolcke Exp $
 *
 */

#ifndef _MSWebNgramLM_h_
#define _MSWebNgramLM_h_

#include <stdio.h>

#if defined(_MSC_VER) || defined(WIN32)
#include <winsock.h>
#else
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
typedef int	SOCKET;		// for MS compatibility
#endif

#include "LM.h"
#include "Boolean.h"
#include "Ngram.h"
#include "NgramStats.h"
#include "Array.h"

class MSWebNgramLM: public LM
{
public:
    MSWebNgramLM(Vocab &vocab, unsigned order = 3, unsigned cacheOrder = 0);
    ~MSWebNgramLM();

    Boolean read(File &file, Boolean limitVocab = false);

    LogP wordProb(VocabIndex word, const VocabIndex *context);
    void *contextID(VocabIndex word, const VocabIndex *context, unsigned &length);

    template <class CountT>
    LogP countsProb(NgramCounts<CountT> &counts, TextStats &stats,
				    unsigned order, Boolean entropy = false);
						/* probability from counts */

    Boolean addUnkWords() { return true; };	/* Words are implicitly added
						 * to vocab so we can transmit
						 * them over the network */

    /*
     * Enable prefetching of ngrams in batches if caching is enabled
     */
    unsigned prefetchingNgrams() { return cacheOrder; };
    Boolean prefetchNgrams(NgramCounts<Count> &counts)
				{ return cacheNgrams(counts); };
    Boolean prefetchNgrams(NgramCounts<XCount> &counts)
				{ return cacheNgrams(counts); };
    Boolean prefetchNgrams(NgramCounts<FloatCount> &counts)
				{ return cacheNgrams(counts); };

protected:
    unsigned order;		/* maximum N-gram length */
    char serverHost[256];	/* "web-ngram.research.microsoft.com" */
    unsigned serverPort;	/* should be 80 (HTTP) */
    char urlPrefix[256];	/* "/rest/lookup.svc" */

    char catalogName[40];	/* e.g., "bing-anchor" */
    char catalogVersion[40];	/* e.g., "jun09" */
    unsigned modelOrder;	/* 1 ... 5 */
    char userToken[100];	/* users's unique token */

    unsigned maxRetries;	/* how often to retry failed calls */

    unsigned tracing;		/* trace server interaction */

private:
    Boolean connectToServer();
    char *callServer(const char *request, const char *data = 0, unsigned responseLength = 1000, unsigned retries = 1);
    Boolean getModelNames();

    template <class CountT>
    Boolean cacheNgrams(NgramCounts<CountT> &counts);

    struct sockaddr_in sockName;
    SOCKET serverSocket;

    char getprobRequest[200];	/* POST /rest/lookup.svc/bing-anchor/jun09/4/... */

    Array<char *> modelNames;
    unsigned numModels;

    unsigned cacheOrder;	/* max N-gram length to cache */
    Ngram probCache;		/* cache for wordProb() results  */
};

#endif /* _MSWebNgramLM_h_ */
