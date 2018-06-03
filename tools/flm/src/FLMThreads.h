/*
 * FLMThreads.h
 *
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#ifndef FLMThreads_h
#define FLMThreads_h

#ifndef NO_TLS
class FLMThreads {
public:
  static void freeThread();
};
#endif /* NO_TLS */

#endif /* FLMThreads_h */

