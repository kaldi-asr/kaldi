/*
 * MSWebNgramLM.cc --
 *	Client-side for Microsoft Web Ngram LM
 *	(see http://web-ngram.research.microsoft.com/info/ for details)
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2012 Microsoft Corp. All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/MSWebNgramLM.cc,v 1.11 2014-08-30 03:54:40 stolcke Exp $";
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <string>

#if !defined(_MSC_VER) && !defined(WIN32)
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>

#include "tserror.h"

#define SOCKET_ERROR_STRING	srilm_ts_strerror(errno)

#define closesocket(s)	close(s)	// MS compatibility
#define INVALID_SOCKET	-1
#define SOCKET_ERROR	-1

#if __INTEL_COMPILER == 700
// old Intel compiler cannot deal with optimized byteswapping functions
#undef htons
#undef ntohs
#endif

#else /* native MS Windows */

#include <winsock.h>

/* defined in LM.cc */
extern WSADATA wsaData;
extern int wsaInitialized;
extern const char *WSA_strerror(int errCode);

#define SOCKET_ERROR_STRING	WSA_strerror(WSAGetLastError())

#define sleep(n)		Sleep((n)*1000)
#define snprintf		_snprintf

#endif  /* !_MSC_VER && !WIN32 */

#include "MSWebNgramLM.h"
#include "RemoteLM.h"
#include "Array.cc"
#include "MStringTokUtil.h"

#define DEFAULT_SERVER_NAME	"web-ngram.research.microsoft.com"
#define DEFAULT_SERVER_PORT	80
#define DEFAULT_LOOKUP_PREFIX	"/rest/lookup.svc"
#define DEFAULT_CATALOG_NAME		"bing-body"
#define DEFAULT_CATALOG_VERSION		"jun09"
#define DEFAULT_MAX_RETRIES	2

#define HTTP_OK_RESPONSE	"HTTP/1.1 200 OK"
#define HTTP_TOOFAST_RESPONSE	"HTTP/1.1 500 "
#define HTTP_OTHER_RESPONSE	"HTTP/1.1 "
#define HTTP_EMPTY_LINE		"\r\n\r\n"
#define HTTP_RESPONSE_HEADER_SIZE	500
#define RESPONSE_SIZE_PER_NGRAM		20

//#define SLOWDOWN_WAIT		120

MSWebNgramLM::MSWebNgramLM(Vocab &vocab, unsigned order, unsigned cacheOrder)
    : LM(vocab), order(order), tracing(0), serverSocket(INVALID_SOCKET), numModels(0),
      cacheOrder(cacheOrder), probCache(vocab, order)
{
#ifndef SOCK_STREAM
    cerr << "MSWebNgramLM not supported\n";
    exit(1);
#endif /* SOCK_STREAM */
}

MSWebNgramLM::~MSWebNgramLM()
{
#ifdef SOCK_STREAM
    for (unsigned i = 0; i < numModels; i ++) {
	free(modelNames[i]);
    }

    if (serverSocket != INVALID_SOCKET) {
    	closesocket(serverSocket);
    }
#endif /* SOCK_STREAM */
}

Boolean
MSWebNgramLM::connectToServer()
{
    /*
     * Create socket, then (re-)connect socket to the server
     */
    if (serverSocket != INVALID_SOCKET) {
	closesocket(serverSocket);
    }

    serverSocket = socket(PF_INET, SOCK_STREAM, 0);

    if (serverSocket == INVALID_SOCKET) {
    	cerr << "socket: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	exit(1);
    }

    if (connect(serverSocket, (struct sockaddr *)&sockName, sizeof(sockName)) == SOCKET_ERROR) {
    	cerr << "connect: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	return false;
    } else if (debug(1)) {
    	dout() << "connected to server " << serverPort << "@" << serverHost << endl;
    }

    return true;
}

char *
MSWebNgramLM::callServer(const char *request, const char *data, unsigned responseSize, unsigned retries)
{
    static Array<char> requestBuffer;

    if (data == 0) {
        requestBuffer[strlen(request) + 18 + strlen(serverHost) + 4] = '\0';
	sprintf(requestBuffer, "%s HTTP/1.1\r\nHost: %s%s",
			request, serverHost, HTTP_EMPTY_LINE);
    } else {
        requestBuffer[strlen(request) + 18 + strlen(serverHost) + 30 + strlen(data)] = '\0';
	sprintf(requestBuffer, "%s HTTP/1.1\r\nHost: %s\r\nContent-Length: %u%s%s",
			request, serverHost, (unsigned)strlen(data), HTTP_EMPTY_LINE, data);
    }

    if (tracing) {
    	cerr << endl << "REQUEST (retries = " << retries
             << "): " << requestBuffer;
    }

    unsigned numTries = 0;

retry:
    if (send(serverSocket, requestBuffer, strlen(requestBuffer), 0) == SOCKET_ERROR) {
        if (numTries++ < retries) {
	    if (connectToServer()) goto retry;
	}

	cerr << "send: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	return 0;
    }

    static Array<char> responseBuffer;
    responseBuffer[HTTP_RESPONSE_HEADER_SIZE+responseSize+1] = '\0';

    int responseLength = recv(serverSocket, responseBuffer, HTTP_RESPONSE_HEADER_SIZE+responseSize, 0);

    if (responseLength == SOCKET_ERROR) {
        if (numTries++ < retries) {
	    if (connectToServer()) goto retry;
	}

	cerr << "recv: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	return 0;
    } else if (responseLength == 0) {
        if (numTries++ < retries) {
	    if (connectToServer()) goto retry;
	}

	cerr << "recv: server " << serverPort << "@" << serverHost
	     << ": empty result\n";
	return 0;
    } else {
	responseBuffer[responseLength] = '\0';

	/*
	 * Find the end of the response header (empty line)
	 */
	char *headerEnd = strstr(responseBuffer, HTTP_EMPTY_LINE);

        assert(headerEnd - responseBuffer <= HTTP_RESPONSE_HEADER_SIZE);

	if (tracing) {
	    cerr << "\nRESPONSE (max = " << (HTTP_RESPONSE_HEADER_SIZE+responseSize)
	         << "): " << responseBuffer <<endl;
	}

	if (strncmp(responseBuffer, HTTP_OK_RESPONSE, sizeof(HTTP_OK_RESPONSE)-1) == 0 &&
	    headerEnd != 0)
	{
	    /*
	     * Return pointer to the beginning of the respone body
	     */
	    return headerEnd + sizeof(HTTP_EMPTY_LINE) - 1;
#ifdef SLOWDOWN_WAIT
	} else if (strncmp(responseBuffer, HTTP_TOOFAST_RESPONSE,
						sizeof(HTTP_TOOFAST_RESPONSE)-1) == 0)
	{
	    cerr << "recv: server " << serverPort << "@" << serverHost
		 << ": going too fast -- waiting for " << SLOWDOWN_WAIT << " seconds\n";

	    sleep(SLOWDOWN_WAIT); 

	    if (connectToServer()) goto retry;

	    return 0;
#endif /* SLOWDOWN_WAIT */
	} else {
	    if (numTries++ < retries) {
		if (connectToServer()) goto retry;
	    }

	    cerr << "recv: server " << serverPort << "@" << serverHost
		 << ": request failed";

	    char *endofline = strchr(responseBuffer, '\n');

	    if (strncmp(responseBuffer, HTTP_OTHER_RESPONSE,
					sizeof(HTTP_OTHER_RESPONSE)-1) == 0) {
		if (endofline) *endofline = '\0';

		cerr << " with code " << &responseBuffer[sizeof(HTTP_OTHER_RESPONSE)-1];
	    }
	    cerr << endl;
	    

	    return 0;
	}
    }
}

Boolean
MSWebNgramLM::getModelNames()
{
    for (unsigned i = 0; i < numModels; i ++) {
	free(modelNames[i]);
    }

    /* 
     * Assume we're connected to server
     */

    makeArray(char, command, strlen(urlPrefix) + 20);

    /* the complete URL is
	http://web-ngram.research.microsoft.com/rest/lookup.svc/{catalog}/{version}/{order}/{operation}?{parameters}
     */
    sprintf(command, "GET %s/?format=text", urlPrefix);

    char *result = callServer(command, 0, 1000, maxRetries);

    if (result == 0) {
	return false;
    } else {
	char *model;
	unsigned i;
	char *strtok_ptr = NULL;

	for (i = 0, model = MStringTokUtil::strtok_r(result, "\r\n", &strtok_ptr);
	     model != 0;
	     i++, model = MStringTokUtil::strtok_r(0, "\r\n", &strtok_ptr))
	{
	    modelNames[i] = strdup(model);
	    assert(modelNames[i] != 0);
	}
	numModels = i;
    }
    return true;
}

Boolean
MSWebNgramLM::read(File &file, Boolean limitVocab /* ignored */)
{
#ifdef SOCK_STREAM
    /*
     * forget about current server
     */
    if (serverSocket != INVALID_SOCKET) {
    	closesocket(serverSocket);
    }

    /*
     * Restore default values
     */
    strcpy(serverHost, DEFAULT_SERVER_NAME);
    serverPort = DEFAULT_SERVER_PORT;
    maxRetries = DEFAULT_MAX_RETRIES;
    strcpy(urlPrefix, DEFAULT_LOOKUP_PREFIX);
    strcpy(catalogName, DEFAULT_CATALOG_NAME);
    strcpy(catalogVersion, DEFAULT_CATALOG_VERSION);
    modelOrder = order;

    /*
     * Parse the config file
     */
    char *line;

    while ((line = file.getline())) {
	char arg1[256];
	unsigned arg2;

	if (sscanf(line, "modelorder %u", &arg2) == 1) {
	    modelOrder = arg2;
	} else if (sscanf(line, "servername %255s", arg1) == 1) {
	    // @kw false positive: SV.STRBO.BOUND_COPY.UNTERM (serverHost)
	    strncpy(serverHost, arg1, sizeof(serverHost));
	    serverHost[sizeof(serverHost) - 1] = 0;
	} else if (sscanf(line, "serverport %u", &arg2) == 1) {
	    serverPort = arg2;
	} else if (sscanf(line, "urlprefix %255s", arg1) == 1) {
	    strncpy(urlPrefix, arg1, sizeof(urlPrefix));
	    urlPrefix[sizeof(urlPrefix) - 1] = 0;
	} else if (sscanf(line, "catalog %39s", arg1) == 1) {
	    strncpy(catalogName, arg1, sizeof(catalogName));
	    catalogName[sizeof(catalogName) - 1] = 0;
	} else if (sscanf(line, "version %39s", arg1) == 1) {
	    strncpy(catalogVersion, arg1, sizeof(catalogVersion));
	    catalogVersion[sizeof(catalogVersion) - 1] = 0;
	} else if (sscanf(line, "usertoken %99s", arg1) == 1) {
	    strncpy(userToken, arg1, sizeof(userToken));
	    userToken[sizeof(userToken) - 1] = 0;
	} else if (sscanf(line, "tracing %u", &arg2) == 1) {
	    tracing = arg2;
	} else if (sscanf(line, "cacheorder %u", &arg2) == 1) {
	    cacheOrder = arg2;
	} else if (sscanf(line, "maxretries %u", &arg2) == 1) {
	    maxRetries = arg2;
	} else {
	    file.position() << "unknown keyword or bad value\n";
	    return false;
	}
    }

    if (cacheOrder > order) {
	cacheOrder = order;
    }

#if defined(_MSC_VER) || defined(WIN32)
    if (!wsaInitialized) {
	int result = WSAStartup(MAKEWORD(2,2), &wsaData);
	if (result != 0) {
	    cerr << "could not initialize winsocket: " << SOCKET_ERROR_STRING << endl;
	    return false;
	}
	wsaInitialized = 1;
    }
#endif /* _MSC_VER || WIN32 */

    /*
     * Get server address either by IP number or by name
     */
    struct in_addr serverAddr;

    if (isdigit(*serverHost)) {
        serverAddr.s_addr = inet_addr(serverHost);
    } else {
        struct hostent *host;

	host = gethostbyname(serverHost);

	if (host == 0) {
	    cerr << "server host " << serverHost << " not found\n";
	    return false;
	}

	assert((unsigned)host->h_length <= sizeof(serverAddr));
	memcpy(&serverAddr, host->h_addr, host->h_length);
    }

    memset(&sockName, 0, sizeof(sockName));

    sockName.sin_family = AF_INET;
    sockName.sin_addr = serverAddr;
    sockName.sin_port = htons(serverPort);

    /* construct the prefix of the lookup command */
    /* the complete URL is
	http://web-ngram.research.microsoft.com/rest/lookup.svc/{catalog}/{version}/{order}/{operation}?{parameters}
     */
    int nneeded = snprintf(getprobRequest, sizeof(getprobRequest), "POST %s/%s/%s/%u/cp?u=%s", urlPrefix,
				catalogName, catalogVersion, modelOrder, userToken);

    // make sure we didn't truncate the path
    assert((unsigned)nneeded < sizeof(getprobRequest));

    /*
     * Establish first server connection
     */
    if (!connectToServer()) {
	return false;
    }

    if (!getModelNames()) {
	return false;
    }

    /*
     * Check that chosen model exists
     */
    char model[100];
    sprintf(model, "%s/%s/%u", catalogName, catalogVersion, modelOrder);
    assert(strlen(model) < sizeof(model)-1);

    unsigned i;
    for (i = 0; i < numModels; i ++) {
	if (strcmp(model, modelNames[i]) == 0) break;
    }
    if (i == numModels) {
	file.position() << "model " << model << " not found on server\n";
	if (debug(1)) {
	    dout() << "available models are:";
	    for (i = 0; i < numModels; i ++) {
		dout() << " " << modelNames[i];
	    }
	    dout() << endl;
	}
	return false;
    } else {
	if (debug(1)) {
	    dout() << "using web-ngram model "<< model << endl;
	}
    }

    return true;
#endif /* SOCK_STREAM */
}

LogP
MSWebNgramLM::wordProb(VocabIndex word, const VocabIndex *context)
{
#ifdef SOCK_STREAM
    if (serverSocket == INVALID_SOCKET) {
    	exit(1);
    }

    unsigned clen = Vocab::length(context);

    /*
     * Limit context length as requested
     */
    if (order > 0 && clen > order - 1) {
    	clen = order - 1;
    }
    if (modelOrder > 0 && clen > modelOrder - 1) {
    	clen = modelOrder - 1;
    }

    LogP *cachedProb = 0;

    /*
     * If this n-gram is cacheable, see if we already have it
     */
    if (clen < cacheOrder) {
	TruncatedContext usedContext(context, clen);
	cachedProb = probCache.insertProb(word, usedContext);

	if (*cachedProb != 0.0) {
	    return *cachedProb;
	}
    }

    string ngramBuffer = "";

    for (int i = clen - 1; i >= 0; i --) {
    	ngramBuffer += vocab.getWord(context[i]);
	ngramBuffer += " ";
    }
    ngramBuffer += vocab.getWord(word);

    char *result = callServer(getprobRequest, ngramBuffer.c_str(), RESPONSE_SIZE_PER_NGRAM, maxRetries);

    if (result == 0) {
	closesocket(serverSocket);
	exit(1);
    } else {
	LogP lprob;

	if (parseLogP(result, lprob))
	{
	    /*
	     * Save new probability in cache
	     */
	    if (cachedProb != 0) {
		*cachedProb = lprob;
	    }

	    return lprob;
	} else {
	    cerr << "server " << serverPort << "@" << serverHost
		 << ": unexpected response: " << result << endl;
	    closesocket(serverSocket);
	    exit(1);
	}
    }
#else 
    exit(1);
#endif /* SOCK_STREAM */
}

void *
MSWebNgramLM::contextID(VocabIndex word, const VocabIndex *context, unsigned &length)
{
#ifdef SOCK_STREAM
    if (modelOrder == 0) {
	length = 0;
    } else {
	length = Vocab::length(context);
        if (length > modelOrder - 1) {
	    length = modelOrder - 1;
	}
    }
    return 0;
#else 
    exit(1);
#endif /* SOCK_STREAM */
}

/*
 * Helper function that looks up and caches all ngrams
 * in the given counts trie
 */
template <class CountT>
Boolean
MSWebNgramLM::cacheNgrams(NgramCounts<CountT> &counts)
{
    if (cacheOrder == 0) {
	return true;
    }

    string ngramBuffer = "";
    unsigned numNgrams = 0;

    unsigned countorder = cacheOrder;

    makeArray(VocabIndex, ngram, countorder + 1);

    /*
     * Enumerate all ngrams and assemble the web query
     */
    for (unsigned i = 1; i <= countorder; i++ ) {
	NgramCountsIter<CountT> ngramIter(counts, ngram, i);

	CountT *count;

	/*
	 * This enumerates all ngrams of the given order
	 */
	while ((count = ngramIter.next())) {
	    /*
	     * Skip zero counts
	     */
	    if (*count == 0) {
		continue;
	    }

	    Vocab::reverse(ngram);

	    VocabIndex *context = &ngram[1];
	    unsigned clen = i - 1;

	    /*
	     * Limit context length as requested
	     */
	    if (order > 0 && clen > order - 1) {
		clen = order - 1;
	    }
	    if (modelOrder > 0 && clen > modelOrder - 1) {
		clen = modelOrder - 1;
	    }

	    /*
	     * If this n-gram is cacheable, see if we already have it
	     */
	    if (clen < cacheOrder) {
		TruncatedContext usedContext(context, clen);
		LogP *cachedProb = probCache.findProb(ngram[0], usedContext);

		/*
		 * If not, add it to the web query
		 */
		if (cachedProb == 0) {
		    if (numNgrams > 0) {
			ngramBuffer += "\n";
		    }
		    for (int i = clen - 1; i >= 0; i --) {
			ngramBuffer += vocab.getWord(context[i]);
			ngramBuffer += " ";
		    }
		    ngramBuffer += vocab.getWord(ngram[0]);

		    numNgrams ++;

		    /*
		     * Allocate the cache entry and fill in a dummy
		     * value to prevent repeated caching of the same ngram
		     */
		    *probCache.insertProb(ngram[0], usedContext) = LogP_Inf;
		}
	    }

	    Vocab::reverse(ngram);
	}
    }

    if (numNgrams == 0) {
	return true;
    }

    char *result = callServer(getprobRequest, ngramBuffer.c_str(),
				RESPONSE_SIZE_PER_NGRAM * numNgrams, maxRetries);

    /*
     * Enumerate the same ngrams again and save the retrieved probs
     */
    char *nextResult = result;

    for (unsigned i = 1; i <= countorder; i++ ) {
	NgramCountsIter<CountT> ngramIter(counts, ngram, i);

	CountT *count;

	while ((count = ngramIter.next())) {
	    /*
	     * Skip zero counts
	     */
	    if (*count == 0) {
		continue;
	    }

	    Vocab::reverse(ngram);

	    VocabIndex *context = &ngram[1];
	    unsigned clen = i - 1;

	    /*
	     * Limit context length as requested
	     */
	    if (order > 0 && clen > order - 1) {
		clen = order - 1;
	    }
	    if (modelOrder > 0 && clen > modelOrder - 1) {
		clen = modelOrder - 1;
	    }

	    if (clen < cacheOrder) {
		TruncatedContext usedContext(context, clen);
		LogP *cachedProb = probCache.findProb(ngram[0], usedContext);

		assert(cachedProb != 0);

		if (*cachedProb == LogP_Inf) {
		    if (nextResult == 0) {
			/*
			 * Failed call or too few results:
			 * remove the dummy cache entry
			 */
			probCache.removeProb(ngram[0], usedContext);
		    } else {
		        LogP lprob;

			if (parseLogP(nextResult, lprob)) {
			    /*
			     * Save probability in cache
			     */
			    *cachedProb = lprob;
			} else {
			    cerr << "server " << serverPort << "@" << serverHost
				 << ": unexpected response: " << nextResult << endl;
			}

			/* 
			 * Skip to next result line
			 */
			nextResult = strchr(nextResult, '\n');
			if (nextResult) nextResult ++;
		    }
		}
	    }

	    Vocab::reverse(ngram);
	}
    }

    return true;
}

/*
 * Version of countsProb() that obtains all ngram probs in one call,
 * then caches them
 */
template <class CountT>
LogP
MSWebNgramLM::countsProb(NgramCounts<CountT> &counts, TextStats &stats,
					unsigned order, Boolean entropy)
{
    cacheNgrams(counts);

    return LM::countsProb(counts, stats, order, entropy);
}

