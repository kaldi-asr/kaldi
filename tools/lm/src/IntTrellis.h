/*
 * Trellis.h --
 *	Trellises for dynamic programming finite state models
 *
 * Copyright (c) 1995, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/Attic/IntTrellis.h,v 1.1 1996-05-31 19:19:24 stolcke Exp $
 *
 */

#ifndef _Trellis_h_
#define _Trellis_h_

#include "Boolean.h"
#include "Prob.h"
#include "LHash.h"
#include "MemStats.h"

typedef unsigned TrellisState;

const TrellisState TrellisState_None = (TrellisState)-1;
/*
 * A node in the trellis
 */
typedef struct {
    Prob prob;		/* total forward probability */
    LogP max;		/* maximum forward probability */
    TrellisState prev;	/* Viterbi backpointer */
} TrellisNode;

typedef LHash<TrellisState,TrellisNode> TrellisSlice;

class Trellis
{
public:
    Trellis(unsigned size);
    ~Trellis();

    unsigned where() { return currTime; };	/* current time index */

    void init(unsigned time = 0);	/* start DP for time index 0 */
    void step();	/* step and initialize next time index */

    void setProb(TrellisState state, LogP prob);
    Prob getProb(TrellisState state) { return getProb(state, currTime); };
    Prob getProb(TrellisState state, unsigned time);

    void update(TrellisState oldState, TrellisState newState, LogP trans);
			/* update DP with a transition */

    Prob sum(unsigned time);		/* sum of all state probs */
    TrellisState max(unsigned time);	/* maximum prob state */

    unsigned viterbi(TrellisState *path, unsigned length,
			TrellisState lastState = TrellisState_None);

    void memStats(MemStats &stats);


private:
    TrellisSlice *trellis;
    unsigned trellisSize;	/* maximum time index */

    unsigned currTime;		/* current time index */

    void initSlice(TrellisSlice &slice);
    Prob sumSlice(TrellisSlice &slice);
    TrellisState maxSlice(TrellisSlice &slice);
};

#endif /* _Trellis_h_ */
