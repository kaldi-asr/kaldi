/*
 * IntTrellis.cc --
 *	Finite-state trellis dynamic programming with integer states
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Attic/IntTrellis.cc,v 1.2 1996-09-08 21:06:34 stolcke Exp $";
#endif

#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "IntTrellis.h"

#define ZERO_INITIALIZE
#include "Array.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_ARRAY(TrellisNode);
#endif

IntTrellis::IntTrellis(unsigned size)
    : trellisSize(size)
{
    assert(size > 0);

    trellis = new TrellisSlice[size];
    assert(trellis != 0);

    init();
}

IntTrellis::~IntTrellis(unsigned size)
{
    delete [] trellis;
}

void
IntTrellis::init(unsigned time)
{
    assert(time < trellisSize);

    currTime = time;
    initSlice(trellis[currTime]);
}

void
IntTrellis::step()
{
    currTime ++;
    assert(currTime < trellisSize);

    initSlice(trellis[currTime]);
}

void
IntTrellis::setProb(TrellisState state, LogP prob)
{
    IntTrellisSlice &currSlice = trellis[currTime];

    TrellisNode &node = currSlice[state];

    node.prob = LogPtoProb(prob);

    if (!foundP) {
	node->max = prob;
	node->prev = TrellisState_None;
    } else {
	if (prob > node->max) {
	    node->max = prob;
	}
    }
}

Prob
Trellis::getProb(TrellisState state, unsigned time)
{
    assert(time <= currTime);
    TrellisSlice &currSlice = trellis[time];

    TrellisNode *node = currSlice.find(state);

    if (node) {
	return node->prob;
    } else {
	return 0.0;
    }
}

void
Trellis::update(TrellisState oldState, TrellisState newState, LogP trans)
{
    assert(currTime > 0);
    TrellisSlice &lastSlice = trellis[currTime - 1];
    TrellisSlice &currSlice = trellis[currTime];

    TrellisNode *oldNode = lastSlice.find(oldState);
    if (!oldNode) {
	/*
	 * If the predecessor state doesn't exist its probability is
	 * implicitly zero and we have nothing to do!
	 */
	return;
    } else {
	Boolean foundP;
	TrellisNode *newNode = currSlice.insert(newState, foundP);

	/*
	 * Accumulate total forward prob
	 */
	Prob newProb = oldNode->prob *  LogPtoProb(trans);
	if (!foundP) {
	    newNode->prob = newProb;
	} else {
	    newNode->prob += newProb;
	}

	/*
	 * Update maximal state prob and Viterbi links
	 */
	LogP totalProb = oldNode->max + trans;
	if (!foundP || totalProb > newNode->max) {
	    newNode->max = totalProb;
	    newNode->prev = oldState;
	}
    }
}

void
Trellis::initSlice(TrellisSlice &slice)
{
    LHashIter<TrellisState,TrellisNode> iter(slice);
    TrellisNode *node;
    TrellisState state;

    while (node = iter.next(state)) {
	node->prob = 0.0;
	node->max = LogP_Zero;
	node->prev = TrellisState_None;
    }
}

Prob
Trellis::sum(unsigned time)
{
    assert(time <= currTime);

    return sumSlice(trellis[time]);
}

Prob
Trellis::sumSlice(TrellisSlice &slice)
{
    LHashIter<TrellisState,TrellisNode> iter(slice);
    TrellisNode *node;
    TrellisState state;

    Prob sum = 0.0;
    while (node = iter.next(state)) {
	sum += node->prob;
    }

    return sum;
}

TrellisState
Trellis::max(unsigned time)
{
    assert(time <= currTime);

    return maxSlice(trellis[time]);
}

TrellisState
Trellis::maxSlice(TrellisSlice &slice)
{
    LHashIter<TrellisState,TrellisNode> iter(slice);
    TrellisNode *node;
    TrellisState state;

    TrellisState maxState = TrellisState_None;
    LogP maxProb = LogP_Zero;

    while (node = iter.next(state)) {
	if (maxState == TrellisState_None || node->max > maxProb) {
	    maxProb = node->max;
	    maxState = state;
	}
    }

    return maxState;
}

unsigned
Trellis::viterbi(TrellisState *path, unsigned length, TrellisState lastState)
{
    if (length > currTime + 1) {
	length = currTime + 1;
    }

    assert(length > 0);

    TrellisState currState;

    /*
     * Backtrace from the last state with maximum score, unless the caller
     * has given us a specific one to start with.
     */
    if (lastState == TrellisState_None) {
    	currState = maxSlice(trellis[length - 1]);
    } else {
	currState = lastState;
    }

    unsigned pos = length;
    while (currState != TrellisState_None) {
	assert(pos > 0);
	pos --;
	path[pos] = currState;

	TrellisSlice &currSlice = trellis[pos];
	TrellisNode *currNode = currSlice.find(currState);
	assert(currNode != 0);
	
	currState = currNode->prev;
    }
    if (pos != 0) {
	/*
	 * Viterbi backtrace failed before reaching start of sentence,
	 */
	return 0;
    } else {
	return length;
    }
}

