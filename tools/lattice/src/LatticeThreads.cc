/*
 * LatticeThreads.cc
 *
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#include "LatticeThreads.h"

#include "Lattice.h"
#include "LMThreads.h"

void 
LatticeThreads::freeThread() {
    Lattice::freeThread();

    // Lattice module depends on LM, so call its freeThread method.
    LMThreads::freeThread();
}
