/*
 * LMClient.cc --
 *	Client-side for network-based LM
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2007-2012 SRI International, 2012 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/LMClient.cc,v 1.23 2016/07/27 06:55:01 stolcke Exp $";
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

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

#endif  /* !_MSC_VER && !WIN32 */

#include "LMClient.h"
#include "RemoteLM.h"
#include "Array.cc"

LMClient::LMClient(Vocab &vocab, const char *server,
		   unsigned order, unsigned cacheOrder)
    : LM(vocab), order(order), serverSocket(INVALID_SOCKET),
      cacheOrder(cacheOrder), probCache(vocab, cacheOrder)
{
#ifdef SOCK_STREAM
    if (server == 0) {
	strcpy(serverHost, "localhost");
    	serverPort = SRILM_DEFAULT_PORT;
    } else {
	unsigned i = sscanf(server, "%u@%255s", &serverPort, serverHost);

	if (i == 2) {
	    // we have server port and hostname
	    ;
	} else if (i == 1) {
	    // we use localhost as the hostname
	    strcpy(serverHost, "localhost");
	} else if (sscanf(server, "%64s", serverHost) == 1) {
	    // we use a default port number
	    serverPort = SRILM_DEFAULT_PORT;
	} else {
	    strcpy(serverHost, "localhost");
	    serverPort = SRILM_DEFAULT_PORT;
	}
    }

#if defined(_MSC_VER) || defined(WIN32)
    if (!wsaInitialized) {
	int result = WSAStartup(MAKEWORD(2,2), &wsaData);
	if (result != 0) {
	    cerr << "could not initialize winsocket: " << SOCKET_ERROR_STRING << endl;
	    return;
	}
	wsaInitialized = 1;
    }
#endif /* _MSC_VER || WIN32 */

    struct hostent *host;
    struct in_addr serverAddr;
    struct sockaddr_in sockName;

    /*
     * Get server address either by ip number or by name
     */
    if (isdigit(*serverHost)) {
        serverAddr.s_addr = inet_addr(serverHost);
    } else {
	host = gethostbyname(serverHost);

	if (host == 0) {
	    cerr << "server host " << serverHost << " not found\n";
	    return;
	}

	assert((unsigned)host->h_length <= sizeof(serverAddr));
	memcpy(&serverAddr, host->h_addr, host->h_length);
    }

    memset(&sockName, 0, sizeof(sockName));

    sockName.sin_family = AF_INET;
    sockName.sin_addr = serverAddr;
    sockName.sin_port = htons(serverPort);

    /*
     * Create, then connect socket to the server
     */
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
	exit(1);
    }

    /*
     * Read the banner line from the server
     */
    char buffer[REMOTELM_MAXRESULTLEN];

    int msglen = recv(serverSocket, buffer, sizeof(buffer)-1, 0);
    if (msglen == SOCKET_ERROR) {
	cerr << "server " << serverPort << "@" << serverHost
	     << ": could not read banner\n";
	closesocket(serverSocket);
	serverSocket = INVALID_SOCKET;
	exit(1);
    } else if (debug(1)) {
	buffer[msglen] = '\0';
	cerr << "server " << serverPort << "@" << serverHost
	     << ": " << buffer;
    }

    /*
     * Switch to version 2 protocol
     */
    
    char msg[REMOTELM_MAXREQUESTLEN];
    sprintf(msg, "%s\n", REMOTELM_VERSION2);

    if (send(serverSocket, msg, strlen(msg), 0) == SOCKET_ERROR) {
	cerr << "send: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	serverSocket = INVALID_SOCKET;
	exit(1);
    }
     
    msglen = recv(serverSocket, msg, sizeof(msg)-1, 0);
    if (msglen == SOCKET_ERROR || strncmp(msg, REMOTELM_OK, sizeof(REMOTELM_OK)-1) != 0) {
	cerr << "server " << serverPort << "@" << serverHost
	     << ": protocol version 2 not supported\n";
	
	closesocket(serverSocket);
	serverSocket = INVALID_SOCKET;
	exit(1);
    }

    /*
     * Initialize contextID() cache
     */
    contextIDCache.word = Vocab_None;
    contextIDCache.context[0] = Vocab_None;
    contextIDCache.id = 0;
    contextIDCache.length = 0;

    /*
     * Initialize contextBOW() cache
     */
    contextBOWCache.context[0] = Vocab_None;
    contextBOWCache.length = 0;
    contextBOWCache.bow = LogP_One;
#else
    cerr << "LMClient not supported\n";
    exit(1);
#endif /* SOCK_STREAM */
}

LMClient::~LMClient()
{
    if (serverSocket != INVALID_SOCKET) {
    	closesocket(serverSocket);
    }
}

LogP
LMClient::wordProb(VocabIndex word, const VocabIndex *context)
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

    char msg[REMOTELM_MAXREQUESTLEN], *msgEnd;

    sprintf(msg, "%s ", REMOTELM_WORDPROB);
    msgEnd = msg + strlen(msg);
    for (int i = clen - 1; i >= 0; i --) {
    	sprintf(msgEnd, "%s ", vocab.getWord(context[i]));
	msgEnd += strlen(msgEnd);
    }
    sprintf(msgEnd, "%s\n", vocab.getWord(word));
    msgEnd += strlen(msgEnd);

    assert(msgEnd - msg < (int)sizeof(msg));

    if (send(serverSocket, msg, msgEnd - msg, 0) == SOCKET_ERROR) {
	cerr << "send: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	serverSocket = INVALID_SOCKET;
	exit(1);
    }

    char buffer[REMOTELM_MAXRESULTLEN];

    int msglen = recv(serverSocket, buffer, sizeof(buffer)-1, 0);

    if (msglen == SOCKET_ERROR) {
	cerr << "recv: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	exit(1);
    } else {
	buffer[msglen] = '\0';

	LogP lprob;

	if (strncmp(buffer, REMOTELM_OK, sizeof(REMOTELM_OK)-1) == 0 &&
	    parseLogP(buffer + sizeof(REMOTELM_OK), lprob))
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
		 << ": unexpected return: " << buffer;
	    closesocket(serverSocket);
	    exit(1);
	}
    }
#else 
    exit(1);
#endif /* SOCK_STREAM */
}

void *
LMClient::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
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

    TruncatedContext usedContext(context, clen);

    /*
     * If this context is cacheable, see if we already have it from the
     * immediately prior call
     */

    if (clen < cacheOrder) {
	Vocab::setCompareVocab(0);		// force index-based compare

	if (word == contextIDCache.word &&
	    Vocab::compare(usedContext, contextIDCache.context) == 0)
	{
	    length = contextIDCache.length;
	    return contextIDCache.id;
	}
    }

    char msg[REMOTELM_MAXREQUESTLEN], *msgEnd;

    sprintf(msg, "%s ", word == Vocab_None ? REMOTELM_CONTEXTID1 : REMOTELM_CONTEXTID2);
    msgEnd = msg + strlen(msg);

    for (int i = clen - 1; i >= 0; i --) {
    	sprintf(msgEnd, "%s ", vocab.getWord(context[i]));
	msgEnd += strlen(msgEnd);
    }

    if (word == Vocab_None) {
	sprintf(msgEnd, "\n");
    } else {
	sprintf(msgEnd, "%s\n", vocab.getWord(word));
    }
    msgEnd += strlen(msgEnd);

    assert(msgEnd - msg < (int)sizeof(msg));

    if (send(serverSocket, msg, msgEnd - msg, 0) == SOCKET_ERROR) {
	cerr << "send: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	serverSocket = INVALID_SOCKET;
	exit(1);
    }

    char buffer[REMOTELM_MAXRESULTLEN];

    int msglen = recv(serverSocket, buffer, sizeof(buffer)-1, 0);

    if (msglen < 0) {
	cerr << "recv: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	exit(1);
    } else {
	buffer[msglen] = '\0';

	unsigned long long cid;

	if (strncmp(buffer, REMOTELM_OK, sizeof(REMOTELM_OK)-1) == 0 &&
	    sscanf(buffer + sizeof(REMOTELM_OK), "%llu %u", &cid, &length) == 2)
    	 {
	    if (clen < cacheOrder) {
	    	// cache results
		contextIDCache.word = word;
		contextIDCache.context[clen] = Vocab_None;
		Vocab::copy(contextIDCache.context, usedContext);
		contextIDCache.id = (void *)cid;
		contextIDCache.length = length;
	    }
	    return (void *)cid;
	} else {
	    cerr << "server " << serverPort << "@" << serverHost
		 << ": unexpected return: " << buffer;
	    closesocket(serverSocket);
	    exit(1);
	}
    }
#else 
    exit(1);
#endif /* SOCK_STREAM */
}

LogP
LMClient::contextBOW(const VocabIndex *context, unsigned length)
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

    TruncatedContext usedContext(context, clen);

    /*
     * If this context is cacheable, see if we already have it from the
     * immediately prior call
     */

    if (clen < cacheOrder) {
	Vocab::setCompareVocab(0);		// force index-based compare

	if (length == contextBOWCache.length &&
	    Vocab::compare(usedContext, contextBOWCache.context) == 0)
	{
	    return contextBOWCache.bow;
	}
    }

    char msg[REMOTELM_MAXREQUESTLEN], *msgEnd;

    sprintf(msg, "%s ", REMOTELM_CONTEXTBOW);
    msgEnd = msg + strlen(msg);

    for (int i = clen - 1; i >= 0; i --) {
    	sprintf(msgEnd, "%s ", vocab.getWord(context[i]));
	msgEnd += strlen(msgEnd);
    }
    sprintf(msgEnd, "%u\n", length);
    msgEnd += strlen(msgEnd);
    
    assert(msgEnd - msg < (int)sizeof(msg));

    if (send(serverSocket, msg, msgEnd - msg, 0) == SOCKET_ERROR) {
	cerr << "server " << serverPort << "@" << serverHost
	     << ": send " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	serverSocket = INVALID_SOCKET;
	exit(1);
    }

    char buffer[REMOTELM_MAXRESULTLEN];

    int msglen = recv(serverSocket, buffer, sizeof(buffer)-1, 0);

    if (msglen < 0) {
	cerr << "recv: server " << serverPort << "@" << serverHost
	     << ": " << SOCKET_ERROR_STRING << endl;
	closesocket(serverSocket);
	exit(1);
    } else {
	buffer[msglen] = '\0';

	LogP bow;

	if (strncmp(buffer, REMOTELM_OK, sizeof(REMOTELM_OK)-1) == 0 &&
	    parseLogP(buffer + sizeof(REMOTELM_OK), bow))
    	{
	    if (clen < cacheOrder) {
	    	// cache results
		contextBOWCache.context[clen] = Vocab_None;
		Vocab::copy(contextBOWCache.context, usedContext);
		contextBOWCache.length = length;
		contextBOWCache.bow = bow;
	    }
	    return bow;
	} else {
	    cerr << "server " << serverPort << "@" << serverHost
		 << ": unexpected return: " << buffer;
	    closesocket(serverSocket);
	    exit(1);
	}
    }
#else 
    exit(1);
#endif /* SOCK_STREAM */
}


