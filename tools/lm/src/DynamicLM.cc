/*
 * DynamicLM.cc --
 *	Dynamically loaded language model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/DynamicLM.cc,v 1.4 1997/08/16 06:57:31 stolcke Exp $";
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "File.h"
#include "DynamicLM.h"
#include "Ngram.h"

/*
 * Debug levels used
 */
#define DEBUG_STATE_CHANGES	1

DynamicLM::DynamicLM(Vocab &vocab)
    : LM(vocab)
{
   myLM = 0;
   currentState = 0;
}

/*
 * Dispatch to the current language model
 */
LogP
DynamicLM::wordProb(VocabIndex word, const VocabIndex *context)
{
    if (myLM == 0) {
	if (running() && debug(DEBUG_STATE_CHANGES)) {
	    dout() << "[nocache]";
	}
	return LogP_Zero;
    } else {
	return myLM->wordProb(word, context);
    }
}

/*
 * Load new LM from file
 */
void
DynamicLM::setState(const char *state)
{
    /*
     * Avoid redundant model reloading
     */
    if (currentState && strcmp(state, currentState) == 0) {
	return;
    }

    if (myLM != 0) {
	free((void *)currentState);
	currentState = 0;

	delete myLM;
	myLM = 0;
    }

    if (debug(DEBUG_STATE_CHANGES)) {
	dout() << "DynamicLM: changing to state " << state;
    }

    char fileName[201];
    if (sscanf(state, " %200s ", fileName) != 1) {
	cerr << "no filename found in state info\n";
    } else {
	File lmFile(fileName, "r", false);

	if (lmFile.error()) {
	    cerr << "error opening LM file " << fileName << endl;
	    return;
	}

	/*
	 * generalize this to allow the order to be specified
	 */
	myLM = new Ngram(vocab, 3);
	assert(myLM);

	/*
	 * propagate local state to new model
	 */
	myLM->debugme(debuglevel());
	myLM->dout(dout());
	myLM->running(running());

	if (!myLM->read(lmFile)) {
	    cerr << "format error in LM file\n";
	    delete myLM;
	    myLM = 0;
	} else {
	    currentState = strdup(state);
	}
    }
}

