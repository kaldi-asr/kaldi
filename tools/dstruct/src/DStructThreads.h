/*
 * DStructThreads.h
 *
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#ifndef DStructThreads_h
#define DStructThreads_h

class DStructThreads {
public:
  static void freeThread();
};

#endif /* DStructThreads_h */

