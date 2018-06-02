/*
 * LatticeThreads.h
 *
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#ifndef LatticeThreads_h
#define LatticeThreads_h

class LatticeThreads {
public:
  static void freeThread();
};

#endif /* LatticeThreads_h */

