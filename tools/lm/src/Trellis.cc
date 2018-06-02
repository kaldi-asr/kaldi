/*
 * Trellis.cc --
 *	Finite-state trellis dynamic programming.  This file contains functions for
 * Trellis and its associated classes: TrellisNode, TrellisSlice, TrellisNBest,
 * and TrellisIter.
 */

#ifndef _Trellis_cc_
#define _Trellis_cc_

#ifndef lint
static char Trellis_Copyright[] = "Copyright (c) 1995-2010 SRI International, 2013 Microsoft Corp.  All Rights Reserved.";
static char Trellis_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Trellis.cc,v 1.26 2015-03-05 07:45:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "Trellis.h"

#include "LHash.cc"
#include "SArray.h"

#define INSTANTIATE_TRELLIS(StateT) \
    INSTANTIATE_LHASH(StateT,TrellisNode<StateT>); \
    template class Trellis<StateT>

template <class StateT>
Trellis<StateT>::Trellis(unsigned len, unsigned numNbest)
  : trellisSize(len), numNbest(numNbest)
{
    assert(len > 0);

    trellis = new TrellisSlice<StateT> [len];
    assert(trellis != 0);

    init(0);
}

template <class StateT>
Trellis<StateT>::~Trellis()
{
    delete [] trellis;
}

template <class StateT>
void
Trellis<StateT>::clear()
{
    /*
     * This function used to clear the entire trellis, which is wasteful
     * since we typically only ever use a small fraction of its full length.
     * Clearing of old entries is now done incrementally, on-demand in
     * TrellisSlice::init().
     */
    init();
}

template <class StateT>
void
Trellis<StateT>::init(unsigned t)
{
    assert(t < trellisSize);
    currTime = t;
    trellis[t].init();			// Initialize the time slice t.
}

template <class StateT>
void
Trellis<StateT>::step()
{
    currTime ++;
    assert(currTime < trellisSize);
    trellis[currTime].init();
}

/*
 * Explicitly set the total and max probability of a path ending at the given
 * state
 */
template <class StateT>
void
Trellis<StateT>::setProb(const StateT &state, LogP prob)
{
    TrellisSlice<StateT>& currSlice = trellis[currTime];
    Boolean foundP;
    TrellisNode<StateT> *node = currSlice.insert(state, foundP);

    node->lprob = prob;
    if (foundP) {
	if (node->nbestSize() > 0 && prob > node->nbest[0].score) {
	    node->nbest[0].score = prob;
	}
	return;
    }

    /*
     * Not found.  Create a new entry.
     */
    node->backlpr = LogP_Zero;
    node->backmax = LogP_Zero;
    node->nbest.init(numNbest);
    if (node->nbestSize() > 0) {
	node->nbest[0].score = prob;
    }
}

template <class StateT>
LogP
Trellis<StateT>::getLogP(const StateT &state, unsigned t)
{
    assert(t <= currTime);
    TrellisNode<StateT> *node = trellis[t].find(state);

    return (node? node->lprob : LogP_Zero);
}

template <class StateT>
LogP
Trellis<StateT>::getMax(const StateT &state, unsigned t, LogP &backmax)
{
    assert(t <= currTime);
    TrellisNode<StateT> *node = trellis[t].find(state);

    if (node && node->nbestSize() > 0) {
	backmax = node->backmax;
	return node->nbest[0].score;
    }
    backmax = LogP_Zero;
    return LogP_Zero;
}

template<class StateT>
void
Trellis<StateT>::update(const StateT &oldState, const StateT &newState, LogP trans)
{
    assert(currTime > 0 && currTime < trellisSize);
    TrellisSlice<StateT>& lastSlice = trellis[currTime-1];
    TrellisSlice<StateT>& currSlice = trellis[currTime];

    TrellisNode<StateT> *oldNode = lastSlice.find(oldState);

    /*
     * If the predecessor state doesn't exist its probability is
     * implicitly zero and we have nothing to do!
     */
    if (!oldNode) {
	return;
    }

    Boolean foundP;
    TrellisNode<StateT> *newNode = currSlice.insert(newState, foundP);

    LogP2 newProb = oldNode->lprob + trans;	// Accumulate total FW prob.
    if (!foundP) {
	newNode->lprob = newProb;
	newNode->backlpr = LogP_Zero;
	newNode->backmax = LogP_Zero;
	newNode->nbest.init(numNbest);
    } else {
	newNode->lprob = AddLogP(newNode->lprob, newProb);
    }

    /*
     * Update Viterbi related info.
     */
    for (unsigned i = 0; i < oldNode->nbestSize(); i++) {
	LogP totalProb = oldNode->nbest[i].score + trans;
	newNode->nbest.insert(Hyp<StateT>(totalProb, oldState, i));
    }
}

template <class StateT>
LogP
Trellis<StateT>::sumLogP(unsigned t)
{
    assert(t <= currTime);
    return trellis[t].sum();
}

template <class StateT>
StateT
Trellis<StateT>::max(unsigned t)
{
    assert(t <= currTime);
    return trellis[t].max();
}

template <class StateT>
unsigned
Trellis<StateT>::prune(LogP p, unsigned t)
{
    assert(t <= currTime);
    return trellis[t].prune(p);
}

template <class StateT>
void
Trellis<StateT>::setBackProb(const StateT &state, LogP prob)
{
    TrellisNode<StateT> *node = trellis[backTime].find(state);

    if (!node) {
	cerr << "trying to set backward prob for nonexistent node " << state
	     << " at time " << backTime << endl;
	return;
    }

    node->backlpr = prob;
    if (prob > node->backmax) {
	node->backmax = prob;
    }
}

template <class StateT>
LogP
Trellis<StateT>::getBackLogP(const StateT &state, unsigned t)
{
    assert(t <= currTime);
    TrellisNode<StateT> *node = trellis[t].find(state);

    return (node? node->backlpr : LogP_Zero);
}

template <class StateT>
void
Trellis<StateT>::initBack(unsigned t)
{
    assert(t <= currTime);

    backTime = t;
}

template <class StateT>
void
Trellis<StateT>::stepBack()
{
    assert(backTime > 0);
    backTime --;
}

template <class StateT>
void
Trellis<StateT>::updateBack(const StateT &oldState, const StateT &newState, LogP trans)
{
    assert(backTime != (unsigned)-1);	/* check for underflow */

    TrellisSlice<StateT>& currSlice = trellis[backTime];
    TrellisSlice<StateT>& nextSlice = trellis[backTime + 1];
    TrellisNode<StateT> *nextNode = nextSlice.find(newState);

    /*
     * If the successor state doesn't exist its probability is
     * implicitly zero and we have nothing to do!
     */
    if (!nextNode) {
	return;
    }

    TrellisNode<StateT> *thisNode = currSlice.find(oldState);

    if (!thisNode) {
	cerr << "trying to update backward prob for nonexistent node "
	     << oldState << " at time " << backTime << endl;
	return;
    }

    /* Accumulate total backward prob
    */
    LogP2 thisProb = nextNode->backlpr + trans;
    thisNode->backlpr = AddLogP(thisNode->backlpr, thisProb);

    LogP totalMax = nextNode->backmax + trans;
    if (totalMax > thisNode->backmax) {
	thisNode->backmax = totalMax;
    }
}

//-------------Viterbi backtrace algorithms-------------------------------

/*
 * Returns in "path" the most likely partial path of the given length, len.
 * We obtain this by calling the overloaded viterbi() with an unmapped
 * lastState, which causes it to default to the most likely last state.
 */
template <class StateT>
unsigned
Trellis<StateT>::viterbi(StateT *path, unsigned len)
{
    LogP dummy;
    StateT lastState;
    Map_noKey(lastState);
    return nbest_viterbi(path, len, 0, dummy, lastState);
}

/* Same as viterbi(), but instead returns the nth best partial path
 */
template <class StateT>
unsigned
Trellis<StateT>::nbest_viterbi(StateT *path, unsigned len, unsigned nth, LogP& score)
{
    StateT lastState;
    Map_noKey(lastState);
    return nbest_viterbi(path, len, nth, score, lastState);
}

/*
 * If lastState is unmapped, this returns in "path" the Viterbi backtrace
 * of the nth best partial path of the given length from the n-best of all
 * the nbest lists in the required timeslice.  Alternately, lastState may be
 * mapped, in which case, the returned path is just the nth best partial
 * path of the given length that ends at the given state.
 */
template <class StateT>
unsigned
Trellis<StateT>::nbest_viterbi(StateT *path, unsigned len, unsigned n,
					    LogP &score, const StateT &lastState)
{
    if (len > currTime + 1) {		  // Sanity check
	len = currTime + 1;
    }
    assert(len > 0 && len <= trellisSize);

    if (n >= numNbest) {
	return 0;
    }

    StateT currState;
    int currWhichbest;
    
    /*
     * Suppose lastState is explicitly given. i.e., mapped.  Then we
     * backtrace from this state's nth best hyp.  Otherwise, we
     * construct the nbest from the required time slice, determine
     * which state actually ends the nth overall-best hyp and
     * backtrace from that state.
     */
    if (Map_noKeyP(lastState)) {
	TrellisNBestList<StateT>& nblist = trellis[len-1].nbest(numNbest);
	currState = nblist[n].prev;
	currWhichbest = nblist[n].whichbest;
	score = nblist[n].score;
    } else {
	currState = lastState;
	currWhichbest = n;

	TrellisNode<StateT> *node = trellis[len-1].find(currState);
	if (!node) {
	    return 0;
	}
	score = node->nbest[n].score;
    }

    unsigned pos = len;
    while (!Map_noKeyP(currState)) {
	assert(pos > 0);
	pos --;
	path[pos] = currState;

	TrellisNode<StateT> *currNode = trellis[pos].find(currState);
	assert(currNode);

	currState = currNode->nbest[currWhichbest].prev;
	currWhichbest = currNode->nbest[currWhichbest].whichbest;
    }

    if (pos != 0) {		// Backtrace failed before reaching start
	  return 0;
    }
    return len;
}

//------------------ Slice related functions -----------------------------------

template<class StateT>
ostream&
operator<<(ostream& os, const TrellisSlice<StateT>& slice)
{
    LHashIter<StateT, TrellisNode<StateT> > iter(slice.nodes);
    TrellisNode<StateT>* node;
    StateT state;

    while ((node = iter.next(state))) {
	os << "  State: [" << state << "],\t" << node->nbestSize() << "-Best = "
	   << *node << endl;
    }
    return os;
}

template <class StateT>
TrellisSlice<StateT>::~TrellisSlice()
{
    /*
     * Destroy node structures and associated n-best lists
     */
    init();
}

/*
 * Initialization of a time slice.
 */
template <class StateT>
void
TrellisSlice<StateT>::init()
{
    LHashIter<StateT, TrellisNode<StateT> > iter(nodes);
    TrellisNode<StateT> *node;
    StateT state;

    /*
     * XXX: We need to explicitly destroy the nodes in the hash table,
     * due to lossage in LHash, to cause n-best lists to be freed.
     * Unfortunately gcc 2.8.1 has a bug that prevents us from calling 
     * ~TrellisNode(), so we make do with clear().
     */
    while ((node = iter.next(state))) {
#if __GNUC__ == 2 && __GNUC_MINOR__ <= 8
	node->clear();
#else
	node->~TrellisNode<StateT>();
#endif
    }
    nodes.clear(0);

    /*
     * The globalNbest list is cleared and left unexpanded.
     * We only fill it in when asked for.
     */
    globalNbest.init(0);
}

/*
 * Returns the log of the sum of the probabilities of paths that end at
 * the current time slice.
 */
template <class StateT>
LogP
TrellisSlice<StateT>::sum()
{
    LHashIter<StateT, TrellisNode<StateT> > iter(nodes);
    TrellisNode<StateT> *node;
    StateT state;
    LogP2 logSum = LogP_Zero;

    while ((node = iter.next(state))) {
	logSum = AddLogP(logSum, node->lprob);
    }
    return logSum;
}

/*
 * Returns the state that ends the highest probability path at the current
 * time slice.
 */
template <class StateT>
StateT
TrellisSlice<StateT>::max()
{
    LHashIter<StateT, TrellisNode<StateT> > iter(nodes);
    TrellisNode<StateT> *node;
    StateT state, maxState;
    LogP maxProb = LogP_Zero;

    Map_noKey(maxState);
    while ((node = iter.next(state))) {
	if (Map_noKeyP(maxState) ||
	    (node->nbestSize() > 0 && node->nbest[0].score > maxProb))
	{
	    maxProb = node->nbest[0].score;
	    maxState = state;
	}
    }
    return maxState;
}

/*
 * Remove states with forward log prob less than the max forward probs minus p.
 * Returns number of pruned states.
 */
template <class StateT>
unsigned
TrellisSlice<StateT>::prune(LogP p)
{
    LogP maxProb = LogP_Zero;

    LHashIter<StateT, TrellisNode<StateT> > iter(nodes);
    TrellisNode<StateT> *node;
    StateT state;

    /*
     * Find the largest forward probability
     * Note: this is different from max(), which looks at the probability of
     * single path leading into a node.
     */
    while ((node = iter.next(state))) {
	if (node->lprob > maxProb) {
	    maxProb = node->lprob;
	}
    }

    unsigned pruned = 0;

    iter.init();
    while ((node = iter.next(state))) {
	if (node->lprob < maxProb - p) {
	    nodes.remove(state);
	    pruned += 1;
	}
    }
    return pruned;
}

/*
 * Calculates the nbest list of paths ending at the current time slice.
 * Once this is calculated, it is stored in the globalNbest member to avoid
 * recomputation. The n-best list thus computed is the n-best of the union
 * of all the n-best hyps belonging to each state in this time-slice.
 *
 * To get the n-best paths over the entire trellis, we must first call this
 * function on the last time slice.  The nbest list thus obtained can then
 * be back-traced to obtain n-best of the best paths.
 */
template<class StateT>
TrellisNBestList<StateT>&
TrellisSlice<StateT>::nbest(unsigned numNbest)
{
    if (globalNbest.size() >= numNbest) {
	return globalNbest;
    }

    globalNbest.init(numNbest);

    LHashIter<StateT, TrellisNode<StateT> > iter(nodes);
    TrellisNode<StateT> *node;
    StateT state;

    while ((node = iter.next(state))) {
	for (unsigned n = 0; n < node->nbestSize(); n++) {
	    globalNbest.insert(Hyp<StateT>(node->nbest[n].score, state, n));
	}
    }

    return globalNbest;
}

//-------------------TrellisNBestList functions--------------------------------

template<class StateT>
TrellisNBestList<StateT>::TrellisNBestList(unsigned num)
  : numNbest(0), nblist(0)
{
    init(num);
}

template<class StateT>
TrellisNBestList<StateT>::~TrellisNBestList()
{
    delete [] nblist;
}

/*
 * allocate or clear an N-best list
 */
template<class StateT>
void
TrellisNBestList<StateT>::init(unsigned newSize)
{
    StateT s;
    Map_noKey(s);
    Hyp<StateT> h(LogP_Zero, s, 0);

    if (newSize == 0) {
	delete [] nblist;
	nblist = 0;
    } else if (newSize > numNbest) {
	delete [] nblist;
	nblist = new Hyp<StateT> [newSize];
	assert(nblist != 0);
    }
    numNbest = newSize;

    /*
     * clear entries
     */
    for (unsigned i = 0; i < numNbest; i++) {
	nblist[i] = h;
    }
}

/*
 * Moves n bytes from src to dst starting at the end.  This is useful
 * to "shift down" part of an array.
 */
template<class T>
inline void rmemmove(T *dst, T *src, unsigned n)
{
    T *d = dst + n;
    T *s = src + n;

    while (n--) {
	*(--d) = *(--s);
    }
}

/* Returns the position where hyp would be inserted into the nbest
 * list.  This may be numNbest if the hyp is worse than the worst
 * hyp already in the list.
 */
template<class StateT>
inline unsigned
TrellisNBestList<StateT>::findrank(const Hyp<StateT>& hyp) const
{
    unsigned low = 0, high = numNbest - 1;

    while (low+1 < high) {
	unsigned m = (high+low)/2;
	if (nblist[m].score > hyp.score ||
	    (nblist[m].score == hyp.score && SArray_compareKey(nblist[m].prev, hyp.prev) > 0))
	{
	    low = m;
	} else {
	    high = m;
	}
    }

    /*
     * low+1 == high at this point, but it may be that low == n-1
     * where n is the correct insertion point, e.g., when inserting
     * 2.5 in (...,3,2,...).
     */
    while (low < numNbest &&
	   (nblist[low].score > hyp.score ||
	    (nblist[low].score == hyp.score && SArray_compareKey(nblist[low].prev, hyp.prev) > 0)))
    {
	low ++;
    }
    return low;
}

/*
 * insert(hyp) inserts the given hyp into the current nBestList if the score
 * of hyp is better (greater) than the score of the worst hyp in the list.  The
 * hyp is inserted before the very first hyp in the list that has a score
 * *worse* than it.
 */
template<class StateT>
void 
TrellisNBestList<StateT>::insert(const Hyp<StateT>& hyp)
{
    unsigned i = findrank(hyp);
    if (i < numNbest) {
	rmemmove<Hyp<StateT> >(&nblist[i+1], &nblist[i], numNbest-i-1);
	nblist[i] = hyp;
    }
}

//------------------ Iteration over states in a trellis slice -----------------

template <class StateT>
TrellisIter<StateT>::TrellisIter(Trellis<StateT> &trellis, unsigned t)
 : sliceIter(trellis.trellis[t].nodes)
{
    assert(t <= trellis.currTime);
}

template <class StateT>
void
TrellisIter<StateT>::init()
{
    sliceIter.init();
}

template <class StateT>
Boolean
TrellisIter<StateT>::next(StateT &state, LogP &prob)
{
    TrellisNode<StateT> *node = sliceIter.next(state);

    if (!node) {
	return false;
    }

    prob = node->lprob;
    return true;
}

#endif /* _Trellis_cc_ */
