/*
 * LMThreads.h
 * 
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#ifndef LMThreads_h
#define LMThreads_h

class LMThreads {
public:
  static void freeThread();
};

#endif /* LMThreads_h */

