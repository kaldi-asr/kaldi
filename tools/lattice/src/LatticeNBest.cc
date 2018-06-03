/*
 * LatticeNBest.cc --
 *	N-best generation from lattices
 *
 * 	(Originally contributed by Dustin Hillard, University of Washington,
 *	Viterbi N-best added by Jing Zheng, SRI International.)
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2004-2015 SRI International, 2013-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeNBest.cc,v 1.36 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

#include "Lattice.h"

#include "Array.cc"
#include "LHash.cc"
#include "IntervalHeap.cc"
#include "File.h"
#include "mkdir.h"
#include "LatticeNBest.h"

#define DebugPrintFatalMessages         0 
#define DebugPrintFunctionality         1 
// for large functionality listed in options of the program
#define DebugPrintOutLoop               2
// for out loop of the large functions or small functions
#define DebugPrintInnerLoop             3
// for inner loop of the large functions or outloop of small functions
#define DebugPrintInnerLoop             3
#define DebugAStar             		4


static void
printDebugHyp(Lattice &lat, unsigned numOutput, unsigned numHyps,
						     LatticeNBestHyp &hyp)
{
  lat.dout() << "Lattice::computeNBest: "
  	     << "Hyp " << numOutput
	     << " " << numHyps
	     << " : " << hyp.score << " ";

  Array<NodeIndex> path;
  hyp.nbestPath->getPath(path);

  for (unsigned n = 0; n < path.size(); n++) {
    LatticeNode *thisNode = lat.findNode(path[n]);
    assert(thisNode != 0);
    if (thisNode->word != Vocab_None) {
      lat.dout() << lat.getWord(thisNode->word)
      		 << "(" << path[n] << ")"; 
      if (thisNode->htkinfo) {
	lat.dout() << "{" << thisNode->htkinfo->acoustic
		   << "}{" << thisNode->htkinfo->language << "} ";
	}
    }
  }
  lat.dout() << "\n";
}


/* *************************
 * A-star N-best generation
 * ************************* */

struct nbestLess {  
  // custom sort for nbest scores
  bool operator() (const LatticeNBestHyp *first, const LatticeNBestHyp *second)     { return (first->score < second->score) || 
											     (first->score == second->score && first->wordCnt < second->wordCnt); }
};
struct nbestGreater {  
  // custom sort for nbest scores
  bool operator() (const LatticeNBestHyp *first, const LatticeNBestHyp *second)     { return (first->score > second->score) || 
											     (first->score == second->score && first->wordCnt > second->wordCnt); }
};
struct nbestEqual {  
  // custom sort for nbest scores
  bool operator() (const LatticeNBestHyp *first, const LatticeNBestHyp *second)     { return first->score == second->score && first->wordCnt == second->wordCnt; }
};



struct SuccInfo {
    NodeIndex to;
    LogP bwScore;    
};

static int
compareSucc(const void *p1, const void *p2)
{
    LogP s1 = ((const SuccInfo *) p1)->bwScore;
    LogP s2 = ((const SuccInfo *) p2)->bwScore;

    if (s1 == s2) return 0;
    else if (s1 < s2) return 1;
    else return -1;
}

struct NodeInfo {
    int numSuccs;
    SuccInfo *succs;
    
    NodeInfo() { numSuccs = 0; succs = 0; };
    ~NodeInfo() { if (numSuccs) delete [] succs; succs = 0; };

    void sortSuccs() {
    	if (numSuccs) qsort(succs, numSuccs, sizeof(SuccInfo), compareSucc);
    };
};

/*
 * Compute top N word sequences with highest probability paths through latttice
 */
Boolean 
Lattice::computeNBest(unsigned N, NBestOptions &nbestOut, SubVocab &ignoreWords,
			    const char *multiwordSeparator, unsigned maxHyps,
			    unsigned nbestDuplicates)
{
  /*
   * Find the top N hyps in the lattice using A* search
   *   First compute the forward and backward max prob paths for each node
   *   Then select the top N paths with A* search (sorting with the computed
   *   max prob paths)
   */

  /*
   * topological sort
   */
  unsigned numNodes = getNumNodes(); 

  NodeIndex *sortedNodes = new NodeIndex[numNodes];
  assert(sortedNodes != 0);
  unsigned numReachable = sortNodes(sortedNodes);
  
  if (numReachable != numNodes) {
    dout() << "Lattice::computeNBest: warning: called with unreachable nodes\n";
  }

  if (sortedNodes[0] != initial) {
    dout() << "Lattice::computeNBest: initial node is not first\n";
    delete [] sortedNodes;
    return LogP_Inf;
  }
  
  unsigned finalPosition = 0;
  for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
    if (sortedNodes[finalPosition] == final) break;
  }
  if (finalPosition == numReachable) {
    dout() << "Lattice::computeNBest: final node is not reachable\n";
    delete [] sortedNodes;
    return LogP_Inf;
  }

  // Done with sortedNodes
  delete [] sortedNodes;
  sortedNodes = NULL;
  
  /*
   * compute fb viterbi probabilities
   */
  LogP *viterbiBackwardProbs = new LogP[maxIndex];
  LogP *viterbiForwardProbs  = new LogP[maxIndex];
  assert(viterbiBackwardProbs != 0 && viterbiForwardProbs != 0);

  NodeInfo *nodeInfos = new NodeInfo[maxIndex];
  assert(nodeInfos != 0);

  LogP bestProb =
      computeForwardBackwardViterbi(viterbiForwardProbs, viterbiBackwardProbs);
  
  for (unsigned i = 0; i < maxIndex; i++) {
      LatticeNode *node = nodes.find(i);      
    
      if (!node) continue;
      NodeInfo *ni = nodeInfos + i;
    
      ni->numSuccs = node->outTransitions.numEntries();
      ni->succs = new SuccInfo[ni->numSuccs];
      assert(ni->succs != 0);
      
      TRANSITER_T<NodeIndex, LatticeTransition> transIter(node->outTransitions);
      
      NodeIndex toNode;
      int j = 0;
      while (LatticeTransition *trans = transIter.next(toNode)) {
          LogP bwScore = trans->weight + viterbiBackwardProbs[toNode];
          
          ni->succs[j].to = toNode;
          ni->succs[j].bwScore = bwScore;
          j ++;
      }

      ni->sortSuccs();
    }
  
  if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::computeNBest: best FB prob: " << bestProb << endl;
  }
  
  /*
   * select top nbest hyps from lattice with A* search
   */
  
  if (debug(DebugPrintFunctionality)) {
    dout() << "Lattice::computeNBest: writing nbest list\n";
  }

  IntervalHeap<LatticeNBestHyp*, nbestLess, nbestGreater, nbestEqual>
				    hyps(maxHyps > 0 ? maxHyps : (2 * N + 1));

  LHash<VocabString, unsigned> hypsPrinted;	// for duplicate removal

  /*
   * 
   * Implement A* 
   *
   * 1 - Initialize priority queue with a null theory
   * 2 - Pop the highest score hyp
   * 3 - If end-of-sentence, output hyp (return to 2)
   * 4 - Create new hyps for all outgoing word transitions
   * 5 - Score each new extended hyp, re-insert to queue (possibly marking end-of-sentence)
   * 6 - Go to 2 
   *
   */

  // start queue with hyp containing only the initial node
  LatticeNBestPath *initialPath = new LatticeNBestPath(initial, 0);
  assert(initialPath != 0);

  LatticeNBestHyp *initialHyp =
  		new LatticeNBestHyp(LogP_One, LogP_One, initial, -1,
                                    false, initialPath, 0, 
				    LogP_One, LogP_One, LogP_One, LogP_One,
				    LogP_One, LogP_One, LogP_One, LogP_One,
				    LogP_One, LogP_One, LogP_One, LogP_One,
				    LogP_One, LogP_One);
  assert(initialHyp != 0);

  hyps.push(initialHyp);

  unsigned outputHyps = 0;
  Boolean firstPruned = true;

  while (outputHyps < N && !hyps.empty()) {
    LatticeNBestHyp *topHyp = hyps.top_max(); // get hyp
    hyps.pop_max();
    if (topHyp->endOfSent) {
      // check to see if this hyp was already printed
      char *feature =
		topHyp->getHypFeature(ignoreWords, *this, multiwordSeparator);
      Boolean isDuplicate;
      unsigned *timesPrinted = hypsPrinted.insert(feature, isDuplicate);
      if (isDuplicate) {
	*timesPrinted += 1;
      } else {
	*timesPrinted = 1;
      }

      if (isDuplicate && *timesPrinted > nbestDuplicates) {
        if (debug(DebugPrintOutLoop)) {
	  dout() << "Lattice::computeNBest: not outputting hypothesis because it matches previously printed one\n";
	}
	
	// debugging output
	if (debug(DebugPrintInnerLoop)) {
	  printDebugHyp(*this, outputHyps, hyps.size(), *topHyp);
	}

        free(feature);
      } else {
        // output hyp
	outputHyps++;
	if (!topHyp->writeHyp(outputHyps, *this, nbestOut)) {
	  cerr << "could not write hyp " << outputHyps << " for lattice " << this->getName() << endl;
	}

	// debugging output
	if (debug(DebugPrintOutLoop)) {
	  printDebugHyp(*this, outputHyps, hyps.size(), *topHyp);
	}
      }
    } else {
      // top hyp is not an end of sentence, so extend it
      // expand all outgoing paths from current node

      LatticeNode *node = findNode(topHyp->nbestPath->node); 
      assert(node != 0);
      NodeIndex nodeIndex = topHyp->nodeIndex;
      int succIndex = topHyp->succIndex + 1;
      NodeInfo *nodeInfo = &(nodeInfos[nodeIndex]);
	
      if (succIndex < nodeInfo->numSuccs) {
        NodeIndex toNodeIndex = nodeInfo->succs[succIndex].to;
	// compute accumulated scores
        double score = topHyp->forwardProb + nodeInfo->succs[succIndex].bwScore;

        // compute the forward part of the score
	LogP forwardProb = score - viterbiBackwardProbs[toNodeIndex];

	unsigned cnt = topHyp->wordCnt;		// word count (ignore non-words)
	LogP acoustic = topHyp->acoustic;	// acoustic model log score
	LogP ngram = topHyp->ngram;		// ngram model log score
	LogP language = topHyp->language;	// language model log score
	LogP pron = topHyp->pron;		// pronunciation log score
	LogP duration = topHyp->duration;	// duration log score
	LogP xscore1 = topHyp->xscore1; 	// extra score #1
	LogP xscore2 = topHyp->xscore2; 	// extra score #2
	LogP xscore3 = topHyp->xscore3; 	// extra score #3
	LogP xscore4 = topHyp->xscore4; 	// extra score #4
	LogP xscore5 = topHyp->xscore5; 	// extra score #5
	LogP xscore6 = topHyp->xscore6; 	// extra score #6
	LogP xscore7 = topHyp->xscore7; 	// extra score #7
	LogP xscore8 = topHyp->xscore8; 	// extra score #8
	LogP xscore9 = topHyp->xscore9; 	// extra score #9

	if (node->htkinfo) {
	  if (!ignoreWord(node->word) &&	// NULL and pause nodes
	      !ignoreWords.getWord(node->word) &&
	      !vocab.isNonEvent(node->word) &&	// <s> and other non-events
	      node->word != vocab.seIndex())
	    cnt      += 1;

	  if (node->htkinfo->acoustic != HTK_undef_float) 
	    acoustic += node->htkinfo->acoustic;
	  if (node->htkinfo->ngram != HTK_undef_float) 
	    ngram    += node->htkinfo->ngram;
	  if (node->htkinfo->language != HTK_undef_float) 
	    language += node->htkinfo->language;
	  if (node->htkinfo->pron != HTK_undef_float)
	    pron     += node->htkinfo->pron;
	  if (node->htkinfo->duration != HTK_undef_float) 
	    duration += node->htkinfo->duration;
	  if (node->htkinfo->xscore1 != HTK_undef_float) 
	    xscore1  += node->htkinfo->xscore1;
	  if (node->htkinfo->xscore2 != HTK_undef_float) 
	    xscore2  += node->htkinfo->xscore2;
	  if (node->htkinfo->xscore3 != HTK_undef_float) 
	    xscore3  += node->htkinfo->xscore3;
	  if (node->htkinfo->xscore4 != HTK_undef_float) 
	    xscore4  += node->htkinfo->xscore4;
	  if (node->htkinfo->xscore5 != HTK_undef_float) 
	    xscore5  += node->htkinfo->xscore5;
	  if (node->htkinfo->xscore6 != HTK_undef_float) 
	    xscore6  += node->htkinfo->xscore6;
	  if (node->htkinfo->xscore7 != HTK_undef_float) 
	    xscore7  += node->htkinfo->xscore7;
	  if (node->htkinfo->xscore8 != HTK_undef_float) 
	    xscore8  += node->htkinfo->xscore8;
	  if (node->htkinfo->xscore9 != HTK_undef_float) 
	    xscore9  += node->htkinfo->xscore9;
	}
						    // add this node to path
	LatticeNBestPath *thisPath =
			new LatticeNBestPath(toNodeIndex, topHyp->nbestPath);
	assert(thisPath != 0);
    
        Boolean isFinal = (toNodeIndex == final);

	LatticeNBestHyp *expandedHyp =
	    new LatticeNBestHyp(score, forwardProb, toNodeIndex, -1,
				isFinal, thisPath, cnt, 
				    acoustic, ngram, language, pron, duration,
				xscore1, xscore2, xscore3, xscore4, xscore5,
				xscore6, xscore7, xscore8, xscore9);
	assert(expandedHyp != 0);
	if (maxHyps > 0 && hyps.size() >= maxHyps) {
	  LatticeNBestHyp *pruneHyp = hyps.top_min(); // get hyp
	  hyps.pop_min();
	  delete pruneHyp;
	  if (debug(DebugPrintOutLoop) ||
	      (firstPruned && debug(DebugPrintFunctionality)))
	  {
	    dout() << "Lattice::computeNBest: max number of hyps reached, pruning lowest score hyp\n";
	    firstPruned = false;
	  }
	}
        hyps.push(expandedHyp);
      }
      
      if (succIndex < nodeInfo->numSuccs - 1) {
        double score = topHyp->forwardProb +
				nodeInfo->succs[succIndex + 1].bwScore;
        
        LatticeNBestHyp *expandedHyp =
            new LatticeNBestHyp(score, topHyp->forwardProb, topHyp->nodeIndex,
	    			succIndex, false, topHyp->nbestPath,
				topHyp->wordCnt, topHyp->acoustic,
				topHyp->ngram, topHyp->language,
                                topHyp->pron, topHyp->duration,
				topHyp->xscore1, topHyp->xscore2,
				topHyp->xscore3, topHyp->xscore4,
				topHyp->xscore5, topHyp->xscore6,
				topHyp->xscore7, topHyp->xscore8,
				topHyp->xscore9);
        
        assert(expandedHyp != 0);
	if (maxHyps > 0 && hyps.size() >= maxHyps) {
	  LatticeNBestHyp *pruneHyp = hyps.top_min(); // get hyp
	  hyps.pop_min();
	  delete pruneHyp;
	  if (debug(DebugPrintOutLoop) ||
	      (firstPruned && debug(DebugPrintFunctionality)))
	  {
	    dout() << "Lattice::computeNBest: max number of hyps reached, pruning lowest score hyp\n";
	    firstPruned = false;
	  }
	}
	hyps.push(expandedHyp);
      }
    }
    delete topHyp;
  }

  // pop remaining hyps to free memory
  while (!hyps.empty()) {
    LatticeNBestHyp *topHyp = hyps.top_max();
    hyps.pop_max();
    delete topHyp;
  }
  delete [] viterbiForwardProbs;
  delete [] viterbiBackwardProbs;
  delete [] nodeInfos;

  return true;
}


/* *************************
 * Viterbi N-best generation
 * ************************* */

typedef LHash<const char *, LatticeNBestHyp *> ENTRYHASH_T;
typedef LHashIter<const char *, LatticeNBestHyp *> ENTRYITER_T;

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(const char *, LatticeNBestHyp*);
#endif

struct HypEntry { const char * key; LatticeNBestHyp * hyp; };
static int compEntry(const void *p1, const void *p2) 
{
    LogP pr1 = ((const HypEntry *) p1)->hyp->forwardProb;
    LogP pr2 = ((const HypEntry *) p2)->hyp->forwardProb;

    if (pr1 == pr2) return 0;
    else if (pr1 < pr2) return 1;
    else return -1;
}

Boolean
Lattice::computeNBestViterbi(unsigned N, NBestOptions &nbestOut,
						SubVocab &ignoreWords,
			     			const char *multiwordSeparator)
{
    /*
     * topological sort
     */
    unsigned numNodes = getNumNodes(); 
    
    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);
    
    if (numReachable != numNodes) {
        dout() << "Lattice::computeNBestViterbi: warning: called with unreachable nodes\n";
    }
    
    if (sortedNodes[0] != initial) {
        dout() << "Lattice::computeNBestViterbi: initial node is not first\n";
        delete [] sortedNodes;
        return LogP_Inf;
    }
    
    unsigned finalPosition = 0;
    for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
        if (sortedNodes[finalPosition] == final) break;
    }

    if (finalPosition == numReachable) {
        dout() << "Lattice::computeNBestViterbi: final node is not reachable\n";
        delete [] sortedNodes;
        return LogP_Inf;
    }

    ENTRYHASH_T ht(2 * N);

    unsigned i;
    Array<HypEntry> **hyps = new Array<HypEntry> * [numReachable];
    assert(hyps != 0);
    memset(hyps, 0, sizeof(void *) * numReachable);

    unsigned *map = new unsigned[maxIndex];
    assert(map != 0);
    memset(map, 0, sizeof(unsigned) * maxIndex);

    for (i = 0; i <= finalPosition; i++) {
        map[sortedNodes[i]] = i;
    }

    Array<int> *freeList = new Array<int> [numReachable];
    assert(freeList != 0);

    for (i = 0; i < finalPosition; i++) {
        NodeIndex nodeIndex = sortedNodes[i];
        LatticeNode *node = nodes.find(nodeIndex);
        assert(node != 0);

        TRANSITER_T<NodeIndex, LatticeTransition> transIter(node->outTransitions);
        NodeIndex toNodeIndex;
        NodeIndex maxTo = 0;
        while (transIter.next(toNodeIndex)) {
            if (map[toNodeIndex] > maxTo) {
                maxTo = map[toNodeIndex];
            }
        }
        
        Array<int> &fl = freeList[maxTo];
        fl[fl.size()] = i;
    }

    /* initialize initial node */
    LatticeNBestPath *initialPath = new LatticeNBestPath(initial, 0);
    assert(initialPath != 0);

    LatticeNBestHyp *initialHyp =
        new LatticeNBestHyp(LogP_One, LogP_One, initial, -1,
                            false, initialPath, 0, 
                            LogP_One, LogP_One, LogP_One, LogP_One,
			    LogP_One, LogP_One, LogP_One, LogP_One,
			    LogP_One, LogP_One, LogP_One, LogP_One,
			    LogP_One, LogP_One);

    
    char *initialKey = (char *)"";
    hyps[0] = new Array<HypEntry> (0, 1);
    assert(hyps[0] != 0);
    HypEntry *e = hyps[0]->data();
    e->key = strdup(initialKey);
    e->hyp = initialHyp;
    
    for (i = 1; i <= finalPosition; i++) {
        NodeIndex nodeIndex = sortedNodes[i];
        LatticeNode *node = nodes.find(nodeIndex);
        assert(node != 0);

        Boolean isFinal = (nodeIndex == final);

        const char *word = "";
        if (!ignoreWord(node->word) &&
            !ignoreWords.getWord(node->word) &&
            !vocab.isNonEvent(node->word) &&
            node->word != vocab.seIndex()) 
          word = getWord(node->word);
        
        int len = strlen(word);

        unsigned wcnt = (len > 0 ? 1 : 0); 	// word count (ignore non-words)
	LogP acoustic = LogP_One;	// acoustic model log score
	LogP ngram    = LogP_One;	// ngram model log score
	LogP language = LogP_One;	// language model log score
	LogP pron     = LogP_One;	// pronunciation log score
	LogP duration = LogP_One;	// duration log score
	LogP xscore1  = LogP_One; 	// extra score #1
	LogP xscore2  = LogP_One; 	// extra score #2
	LogP xscore3  = LogP_One; 	// extra score #3
	LogP xscore4  = LogP_One; 	// extra score #4
	LogP xscore5  = LogP_One; 	// extra score #5
	LogP xscore6  = LogP_One; 	// extra score #6
	LogP xscore7  = LogP_One; 	// extra score #7
	LogP xscore8  = LogP_One; 	// extra score #8
	LogP xscore9  = LogP_One; 	// extra score #9

	if (node->htkinfo) {
            if (node->htkinfo->acoustic != HTK_undef_float) 
              acoustic = node->htkinfo->acoustic;
            if (node->htkinfo->ngram != HTK_undef_float) 
              ngram    = node->htkinfo->ngram;
            if (node->htkinfo->language != HTK_undef_float) 
              language = node->htkinfo->language;
            if (node->htkinfo->pron != HTK_undef_float)
              pron     = node->htkinfo->pron;
            if (node->htkinfo->duration != HTK_undef_float) 
              duration = node->htkinfo->duration;
            if (node->htkinfo->xscore1 != HTK_undef_float) 
              xscore1  = node->htkinfo->xscore1;
            if (node->htkinfo->xscore2 != HTK_undef_float) 
              xscore2  = node->htkinfo->xscore2;
            if (node->htkinfo->xscore3 != HTK_undef_float) 
              xscore3  = node->htkinfo->xscore3;
            if (node->htkinfo->xscore4 != HTK_undef_float) 
              xscore4  = node->htkinfo->xscore4;
            if (node->htkinfo->xscore5 != HTK_undef_float) 
              xscore5  = node->htkinfo->xscore5;
            if (node->htkinfo->xscore6 != HTK_undef_float) 
              xscore6  = node->htkinfo->xscore6;
            if (node->htkinfo->xscore7 != HTK_undef_float) 
              xscore7  = node->htkinfo->xscore7;
            if (node->htkinfo->xscore8 != HTK_undef_float) 
              xscore8  = node->htkinfo->xscore8;
            if (node->htkinfo->xscore9 != HTK_undef_float) 
              xscore9  = node->htkinfo->xscore9;
	}

        unsigned num;

        // propogate to successors
        TRANSITER_T<NodeIndex, LatticeTransition>
					transIter(node->inTransitions);
        transIter.init();
        LatticeTransition *inTrans;
        NodeIndex fromNodeIndex;
        while ((inTrans = transIter.next(fromNodeIndex))) {

            unsigned from = map[fromNodeIndex];
            
            Array<HypEntry> *fromArray = hyps[from];
            if (fromArray == 0) continue;
            num = fromArray->size();
            HypEntry *entries = fromArray->data();
                
            for (unsigned j = 0; j < num; j++) {
                LatticeNBestHyp *hyp = entries[j].hyp;
                LogP forwardProb = hyp->forwardProb + inTrans->weight;

                char *newkey = new char [strlen(entries[j].key) + len + 2];
		assert(newkey != 0);
            
                if (len) {
                  sprintf(newkey, "%s%c%s", entries[j].key, 
				multiwordSeparator ? *multiwordSeparator : ' ',
				word);
                } else {
                  strcpy(newkey, entries[j].key);
		}
                
                Boolean foundP;
                LatticeNBestHyp **p = ht.insert(newkey, foundP);

                if (foundP) {
                    // already exists
                    LatticeNBestHyp *oldHyp = *p;

                    if (oldHyp->score < forwardProb) {
                        oldHyp->score = forwardProb;
                        oldHyp->wordCnt  = wcnt + hyp->wordCnt;

                        oldHyp->nbestPath->pred->release();
                        oldHyp->nbestPath->pred = hyp->nbestPath;
                        hyp->nbestPath->linkto();
                    
                        oldHyp->forwardProb = forwardProb;
                        oldHyp->acoustic    = acoustic + hyp->acoustic;
                        oldHyp->ngram       = ngram + hyp->ngram;
                        oldHyp->language    = language + hyp->language;
                        oldHyp->pron        = pron + hyp->pron;
                        oldHyp->duration    = duration + hyp->duration;
                        oldHyp->xscore1     = xscore1 + hyp->xscore1;
                        oldHyp->xscore2     = xscore2 + hyp->xscore2;
                        oldHyp->xscore3     = xscore3 + hyp->xscore3;
                        oldHyp->xscore4     = xscore4 + hyp->xscore4;
                        oldHyp->xscore5     = xscore5 + hyp->xscore5;
                        oldHyp->xscore6     = xscore6 + hyp->xscore6;
                        oldHyp->xscore7     = xscore7 + hyp->xscore7;
                        oldHyp->xscore8     = xscore8 + hyp->xscore8;
                        oldHyp->xscore9     = xscore9 + hyp->xscore9;
                    }

                } else {
                    // new one
                    LatticeNBestPath *path =
				new LatticeNBestPath(nodeIndex, hyp->nbestPath);

                    LatticeNBestHyp *newHyp =
				new LatticeNBestHyp(forwardProb, forwardProb,
						     nodeIndex, -1, isFinal,
						     path, wcnt + hyp->wordCnt,
                                                     acoustic + hyp->acoustic,
						     ngram + hyp->ngram,
						     language + hyp->language,
						     pron + hyp->pron,
						     duration + hyp->duration,
						     xscore1 + hyp->xscore1,
						     xscore2 + hyp->xscore2,
						     xscore3 + hyp->xscore3,
						     xscore4 + hyp->xscore4,
						     xscore5 + hyp->xscore5,
						     xscore6 + hyp->xscore6,
						     xscore7 + hyp->xscore7,
						     xscore8 + hyp->xscore8,
						     xscore9 + hyp->xscore9);

                    *p = newHyp;
                }

                delete [] newkey;
            }
        }

        // copy entries to array
        num = ht.numEntries();

        if (num > 0) { 
            HypEntry *entries = new HypEntry [ num ];
	    assert(entries != 0);
            num = 0;

            ENTRYITER_T iter(ht);
            LatticeNBestHyp **phyp;
            const char *key;
            while ((phyp = iter.next(key))) {
                entries[num].key = key;
                entries[num].hyp = *phyp;
                num++;
            }
            
            qsort(entries, num, sizeof(HypEntry), compEntry);
            if (num > N) num = N;
            
            hyps[i] = new Array<HypEntry> (0, num);
	    assert(hyps[i] != 0);
            HypEntry *dst = hyps[i]->data();
            HypEntry *src = entries;

	    unsigned j;
            for (j = 0; j < num; j++, src++, dst++) {
                dst->key = strdup(src->key);
                dst->hyp = src->hyp;
            }
            
            for (j = num; j < ht.numEntries(); j++) {
                delete entries[j].hyp;
            }
            
            ht.clear();
            
            delete [] entries;
        }

        // free hyps in freeList
        Array<int> &fl = freeList[i];
        
        for (unsigned j = 0; j < fl.size(); j++) {
            int index = fl[j];
            
            if (hyps[index] == 0) continue;
            HypEntry *entries = hyps[index]->data();
            
            for (unsigned k = hyps[index]->size()-1; (int)k >= 0; k--) {
                delete entries[k].hyp;
                free ((char *) entries[k].key);
            }
            
            delete hyps[index];
            hyps[index] = 0;
        }
    }

    // output reasults
    if (hyps[finalPosition] && hyps[finalPosition]->size()) {
        Array<HypEntry> *results = hyps[finalPosition];
        
        unsigned num = results->size();

        HypEntry *entries = results->data();

        qsort(entries, num, sizeof(HypEntry), compEntry);
        
        for (unsigned i = 0; i < num && i < N; i++) {
            
            LatticeNBestHyp *hyp = entries[i].hyp;
            if (!hyp->writeHyp(i, *this, nbestOut)) {
	      cerr << "could not write hyp " << i << " for lattice " << this->getName() << endl;
	    }

            delete hyp;
            free((void *)entries[i].key);
        }

        delete results;
        hyps[finalPosition] = 0;
    } else {        
        dout() << "Lattice::computeNBestViterbi: "
	       << "no hyp reached final!" << endl;
    }

    // free hyps and keys -- should not be necessary, but just in case
    for (i = 0; i <= finalPosition; i++) {
        if (hyps[i] == 0) continue;

        unsigned num = hyps[i]->size();
        HypEntry *entries = hyps[i]->data();

        for (unsigned j = 0; j < num; j++) {
            delete entries[j].hyp;
            free((void *)entries[j].key);
        }

        delete hyps[i];
        hyps[i] = 0;
    }

    delete [] freeList;
    delete [] map;
    delete [] sortedNodes;
    delete [] hyps;

    return true;
}

NBestOptions::NBestOptions(char *myNbestOutDir,
			   char *myNbestOutDirNgram,
			   char *myNbestOutDirPron,
			   char *myNbestOutDirDur,
			   char *myNbestOutDirXscore1,
			   char *myNbestOutDirXscore2,
			   char *myNbestOutDirXscore3,
			   char *myNbestOutDirXscore4,
			   char *myNbestOutDirXscore5,
			   char *myNbestOutDirXscore6,
			   char *myNbestOutDirXscore7,
			   char *myNbestOutDirXscore8,
			   char *myNbestOutDirXscore9,
			   char *myNbestOutDirRttm,
			   char *myNbestOutDirRttm2)
  : nbestOutDir(myNbestOutDir), nbestOutDirNgram(myNbestOutDirNgram),
    nbestOutDirPron(myNbestOutDirPron), nbestOutDirDur(myNbestOutDirDur),
    nbestOutDirXscore1(myNbestOutDirXscore1),
    nbestOutDirXscore2(myNbestOutDirXscore2),
    nbestOutDirXscore3(myNbestOutDirXscore3),
    nbestOutDirXscore4(myNbestOutDirXscore4),
    nbestOutDirXscore5(myNbestOutDirXscore5),
    nbestOutDirXscore6(myNbestOutDirXscore6),
    nbestOutDirXscore7(myNbestOutDirXscore7),
    nbestOutDirXscore8(myNbestOutDirXscore8),
    nbestOutDirXscore9(myNbestOutDirXscore9),
    nbestOutDirRttm(myNbestOutDirRttm),
    nbestOutDirRttm2(myNbestOutDirRttm2),
    writingFiles(false), nbest(0), nbestNgram(0), nbestPron(0), nbestDur(0),
    nbestXscore1(0), nbestXscore2(0), nbestXscore3(0),
    nbestXscore4(0), nbestXscore5(0), nbestXscore6(0),
    nbestXscore7(0), nbestXscore8(0), nbestXscore9(0),
    nbestRttm(0),nbestRttm2(0)
{
}

NBestOptions::~NBestOptions()
{
    if (writingFiles) {
	closeFiles();
    }
}

static void
makeDir(const char *dir, Boolean overwrite)
{
    if (MKDIR(dir) < 0) {
      if (errno == EEXIST) {
	if (!overwrite) {
	  cerr << "Dir " << dir << " already exists, please give another one\n";
	  exit(2);
	}
      } else {
	perror(dir);
	exit(1);
      }
    }
}

Boolean
NBestOptions::makeDirs(Boolean overwrite)
{
    if (nbestOutDir) {
	makeDir(nbestOutDir, overwrite);
	if (nbestOutDirNgram) makeDir(nbestOutDirNgram, overwrite);
	if (nbestOutDirPron) makeDir(nbestOutDirPron, overwrite);
	if (nbestOutDirDur) makeDir(nbestOutDirDur, overwrite);
	if (nbestOutDirXscore1) makeDir(nbestOutDirXscore1, overwrite);
	if (nbestOutDirXscore2) makeDir(nbestOutDirXscore2, overwrite);
	if (nbestOutDirXscore3) makeDir(nbestOutDirXscore3, overwrite);
	if (nbestOutDirXscore4) makeDir(nbestOutDirXscore4, overwrite);
	if (nbestOutDirXscore5) makeDir(nbestOutDirXscore5, overwrite);
	if (nbestOutDirXscore6) makeDir(nbestOutDirXscore6, overwrite);
	if (nbestOutDirXscore7) makeDir(nbestOutDirXscore7, overwrite);
	if (nbestOutDirXscore8) makeDir(nbestOutDirXscore8, overwrite);
	if (nbestOutDirXscore9) makeDir(nbestOutDirXscore9, overwrite);
	if (nbestOutDirRttm) makeDir(nbestOutDirRttm, overwrite);
	if (nbestOutDirRttm2) makeDir(nbestOutDirRttm2, overwrite);
	return true;
    } else {
	cerr << "Warning: no nbest output directory specified\n";
	return false;
    }
}

Boolean
NBestOptions::openFiles(const char *name)
{
  if (!writingFiles) {
    // make sure all the File pointers are empty
    assert(nbest == 0 &&
	   nbestNgram   == 0 &&
	   nbestPron    == 0 &&
	   nbestDur     == 0 &&
	   nbestXscore1 == 0 &&
	   nbestXscore2 == 0 &&
	   nbestXscore3 == 0 &&
	   nbestXscore4 == 0 &&
	   nbestXscore5 == 0 &&
	   nbestXscore6 == 0 &&
	   nbestXscore7 == 0 &&
	   nbestXscore8 == 0 &&
	   nbestXscore9 == 0 &&
	   nbestRttm    == 0 &&
	   nbestRttm2    == 0);

    writingFiles = true;

    unsigned basenameLen = 1 + strlen(name) + sizeof(GZIP_SUFFIX);
    
    if (nbestOutDir) {
      makeArray(char, outfile, strlen(nbestOutDir) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDir, name, GZIP_SUFFIX);
      nbest = new File(outfile, "w");
    }
    if (nbestOutDirNgram) {
      makeArray(char, outfile, strlen(nbestOutDirNgram) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirNgram, name, GZIP_SUFFIX);
      nbestNgram = new File(outfile, "w");
    }
    if (nbestOutDirPron) {
      makeArray(char, outfile, strlen(nbestOutDirPron) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirPron, name, GZIP_SUFFIX);
      nbestPron = new File(outfile, "w");
    }
    if (nbestOutDirDur) {
      makeArray(char, outfile, strlen(nbestOutDirDur) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirDur, name, GZIP_SUFFIX);
      nbestDur = new File(outfile, "w");
    }
    if (nbestOutDirXscore1) {
      makeArray(char, outfile, strlen(nbestOutDirXscore1) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore1, name, GZIP_SUFFIX);
      nbestXscore1 = new File(outfile, "w");
    }
    if (nbestOutDirXscore2) {
      makeArray(char, outfile, strlen(nbestOutDirXscore2) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore2, name, GZIP_SUFFIX);
      nbestXscore2 = new File(outfile, "w");
    }
    if (nbestOutDirXscore3) {
      makeArray(char, outfile, strlen(nbestOutDirXscore3) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore3, name, GZIP_SUFFIX);
      nbestXscore3 = new File(outfile, "w");
    }
    if (nbestOutDirXscore4) {
      makeArray(char, outfile, strlen(nbestOutDirXscore4) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore4, name, GZIP_SUFFIX);
      nbestXscore4 = new File(outfile, "w");
    }
    if (nbestOutDirXscore5) {
      makeArray(char, outfile, strlen(nbestOutDirXscore5) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore5, name, GZIP_SUFFIX);
      nbestXscore5 = new File(outfile, "w");
    }
    if (nbestOutDirXscore6) {
      makeArray(char, outfile, strlen(nbestOutDirXscore6) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore6, name, GZIP_SUFFIX);
      nbestXscore6 = new File(outfile, "w");
    }
    if (nbestOutDirXscore7) {
      makeArray(char, outfile, strlen(nbestOutDirXscore7) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore7, name, GZIP_SUFFIX);
      nbestXscore7 = new File(outfile, "w");
    }
    if (nbestOutDirXscore8) {
      makeArray(char, outfile, strlen(nbestOutDirXscore8) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore8, name, GZIP_SUFFIX);
      nbestXscore8 = new File(outfile, "w");
    }
    if (nbestOutDirXscore9) {
      makeArray(char, outfile, strlen(nbestOutDirXscore9) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirXscore9, name, GZIP_SUFFIX);
      nbestXscore9 = new File(outfile, "w");
    }
    if (nbestOutDirRttm) {
      makeArray(char, outfile, strlen(nbestOutDirRttm) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirRttm, name, GZIP_SUFFIX);
      nbestRttm = new File(outfile, "w");
    }
    if (nbestOutDirRttm2) {
      makeArray(char, outfile, strlen(nbestOutDirRttm2) + basenameLen);
      sprintf(outfile, "%s/%s%s", nbestOutDirRttm2, name, GZIP_SUFFIX);
      nbestRttm2 = new File(outfile, "w");
    }
    return true;
  } else {
    cerr << "Already have file open for lattice " << name
         << ", not opening another set\n";
    return false;
  }
}

Boolean
NBestOptions::closeFiles()
{
  if (writingFiles) {
    delete nbest;
    delete nbestNgram;
    delete nbestPron;
    delete nbestDur;
    delete nbestXscore1;
    delete nbestXscore2;
    delete nbestXscore3;
    delete nbestXscore4;
    delete nbestXscore5;
    delete nbestXscore6;
    delete nbestXscore7;
    delete nbestXscore8;
    delete nbestXscore9;
    delete nbestRttm;
    delete nbestRttm2;

    nbest        = 0;
    nbestNgram   = 0;
    nbestPron    = 0;
    nbestDur     = 0;
    nbestXscore1 = 0;
    nbestXscore2 = 0;
    nbestXscore3 = 0;
    nbestXscore4 = 0;
    nbestXscore5 = 0;
    nbestXscore6 = 0;
    nbestXscore7 = 0;
    nbestXscore8 = 0;
    nbestXscore9 = 0;
    nbestRttm    = 0;
    nbestRttm2   = 0;

    writingFiles = false;
    return true;
  } else {
    cerr << "Warning: File close method called when files are not opened\n";
    return false;
  }
}

LatticeNBestPath::LatticeNBestPath(NodeIndex node, LatticeNBestPath *pred)
  : node(node), pred(pred), numReferences(0)
{
    if (pred != 0) {
    	pred->linkto();
    }
}

LatticeNBestPath::~LatticeNBestPath()
{
    if (pred != 0) {
	pred->release();
    }
}

void
LatticeNBestPath::linkto()
{
    numReferences += 1;
}

void
LatticeNBestPath::release()
{
    assert(numReferences > 0);
    numReferences -= 1;
    if (numReferences == 0) {
	delete this;
    }
}

unsigned
LatticeNBestPath::getPath(Array<NodeIndex> &result)
{
    unsigned startIndex = result.size();

    // extract path in reverse
    for (LatticeNBestPath *current = this;
         current != 0;
	 current = current->pred)
    {
    	result[result.size()] = current->node;
    }

    // reverse order
    for (unsigned i = startIndex, j = result.size() - 1; i < j; i ++, j --) {
    	NodeIndex h = result[i];
	result[i] = result[j];
	result[j] = h;
    }

    return result.size() - startIndex;
}

LatticeNBestHyp::LatticeNBestHyp(double myScore, LogP myForwardProb, 
                                 NodeIndex myNodeIndex, int mySuccIndex,
				 Boolean myEndOfSent,
				 LatticeNBestPath *myNBestPath,
				 unsigned myWordCnt,
				 LogP myAcoustic, LogP myNgram, LogP myLanguage,
				 LogP myPron, LogP myDuration, 
				 LogP myXscore1, LogP myXscore2, LogP myXscore3,
				 LogP myXscore4, LogP myXscore5, LogP myXscore6,
				 LogP myXscore7, LogP myXscore8, LogP myXscore9)
  :  score(myScore), forwardProb(myForwardProb), 
     endOfSent(myEndOfSent),
     nbestPath(myNBestPath),
     nodeIndex(myNodeIndex), succIndex(mySuccIndex),
  wordCnt(myWordCnt), acoustic(myAcoustic),
     ngram(myNgram), language(myLanguage), pron(myPron), duration(myDuration), 
     xscore1(myXscore1), xscore2(myXscore2), xscore3(myXscore3),
     xscore4(myXscore4), xscore5(myXscore5), xscore6(myXscore6),
     xscore7(myXscore7), xscore8(myXscore8), xscore9(myXscore9)
{
    if (nbestPath != 0) {
	nbestPath->linkto();
    }
}

LatticeNBestHyp::~LatticeNBestHyp()
{
    if (nbestPath != 0) {
        nbestPath->release();
    }
}

// convert floats to string in a way that produces standard results for MSVC
static char *
float2string(char *buffer, double score)
{
    if (isnan(score)) {
	strcpy(buffer, "nan");
    } else if (!isfinite(score)) {
	strcpy(buffer, score < 0.0 ? "-inf" : "inf");
    } else {
	sprintf(buffer, "%.*lg", Prob_Precision, score);
    }	    
    return buffer;
}

// Use "stdio" functions in File() object to allow writing in-memory to File() string object.
Boolean
LatticeNBestHyp::writeHyp(int hypNum, Lattice &lat, NBestOptions &nbestOut)
{

  VocabIndex HTK_SU = lat.vocab.getIndex("<su>");
  if (HTK_SU == Vocab_None) {
    // try capitalized
    HTK_SU =lat.vocab.getIndex("<SU>");
    if (HTK_SU == Vocab_None) {
      //cerr << "Could not find SU word, index will be Vocab_None\n";
    }
  }

  assert(nbestOut.writingFiles);
  // Check both because nbestOut.nbestOutDir can be NULL when writing NBest list to memory
  if (nbestOut.nbestOutDir || nbestOut.nbest) {
    char *speaker, channel, *time, *session;
    float start_time = 0;

    if (nbestOut.nbestRttm) {
      speaker = strdup(lat.getName());
      if (!speaker) {
	// Allocation failed
	return false;
      }
      char *ptr  = strchr(speaker, '_');
      char *ptr2 = (ptr == NULL) ? NULL : strchr(++ptr, '_');
      char *ptr3 = (ptr2 == NULL) ? NULL : strchr(++ptr2, '_');
      
      if (ptr3 == NULL) {
	free(speaker);
	return false;
      }

      ptr3[0] = '\0'; // end string so that new string is 'corpus'_'ses'_'spkr'
      time = strdup(++ptr3);
      if (!time) {
	// Allocation failed
	free(speaker);
	return false;
      }
      char *end  = strchr(time, '_');
      if (!end) {
	// Invalid input
	free(time);
	free(speaker);
	return false;
      }
      end[0] = '\0';
      start_time = atof(time) / 1000;
      float end_time  = atof(++end) / 1000;

      channel = ptr2[0];
      session = strdup(speaker);
      char *ptr4 = strrchr(session, '_');
      if (ptr4 != NULL); {
	ptr4[0] = '\0'; // end string so that new string is 'corpus'_'ses'
      }
    }

    nbestOut.nbest->fprintf("%.*lg %.*lg %u ",
				LogP_Precision, (double)acoustic,
				LogP_Precision, (double)language,
				wordCnt);
    if (nbestOut.nbestRttm2) {
	nbestOut.nbestRttm2->fprintf("%.*lg %.*lg %u ",
				LogP_Precision, (double)acoustic,
				LogP_Precision, (double)language,
				wordCnt);
    }

    Array<NodeIndex> path;
    nbestPath->getPath(path);
    LatticeNode *prevRealNode = NULL;

    for (unsigned n = 0; n < path.size(); n++) {
      LatticeNode *thisNode = lat.findNode(path[n]);
      LatticeNode *prevNode = lat.findNode(path[(n>0 ? n-1 : 0)]);
      if (n == 0) prevRealNode = thisNode; // should be <s>...
      assert(thisNode != 0 && prevNode != 0);

      if (thisNode->word != Vocab_None) {
	nbestOut.nbest->fprintf("%s ", lat.getWord(thisNode->word));
	
	//LEXEME sw_47411 2 51.670 0.470 Yeah lex       SW_47411_B <NA>
	//SU     sw_47411 2 52.140 1.930 <NA> statement SW_47411_B <NA>
	if (nbestOut.nbestRttm) {
	  if (thisNode->word == HTK_SU) {
	    if (thisNode->htkinfo) {
	      nbestOut.nbestRttm->fprintf(
		    "%d SU     %s %c %.2f X <NA> statement %s <NA>\n",
		    hypNum, session, channel,
		    start_time+thisNode->htkinfo->time, speaker); 	
	    }
	  } else {
	    if (thisNode->htkinfo && prevNode->htkinfo) {
	      nbestOut.nbestRttm->fprintf(
	            "%d LEXEME %s %c %.2f %.2f %s lex %s <NA>\n",
		    hypNum, session, channel,
		    start_time+prevNode->htkinfo->time,
		    thisNode->htkinfo->time-prevNode->htkinfo->time,
		    lat.getWord(thisNode->word), speaker);
	    }
	  }
	}

	// If set, extract detailed NBest word information in (an
	// approximation of) the NBestList2.0 format
	if (nbestOut.nbestRttm2 && thisNode && thisNode->word != HTK_SU && thisNode->htkinfo && prevRealNode && prevRealNode->htkinfo) {

	  /* printf("RTTM2: w=%s%s t=%f, pw=%s%s, pt=%f, , prw=%s%s, prt=%f\n", 
		 lat.getWord(thisNode->word), (thisNode->word == HTK_SU ? "(SU)" : ""), thisNode->htkinfo->time,
		 lat.getWord(prevNode->word), (prevNode->word == HTK_SU ? "(SU)" : ""), prevNode->htkinfo->time,
		 lat.getWord(prevRealNode->word), (prevRealNode->word == HTK_SU ? "(SU)" : ""), prevRealNode->htkinfo->time); */

	  // Victor: We need to add or subtract 0.01 somewhere or else the segments will abut each
	  // other exactly.  This is how it appears to work in SRILM, so for now I won't do that....
	  // 
	  // If we uncomment the print statement above, and compare the actual data in the 1st and
	  // 2nd pass lattices and the sausage, We can see that the time marks for the most likely
	  // elements in the lattices are not always the best/correct times in the waveform, or even
	  // the best times in the lattice, but there's not much we can do about that.  It may be
	  // related to weight pushing in the FST...  we would need to compare the time marks when
	  // using less optimized recognizer search graphs to be really sure.
	  //
	  // NOTE: The </s> item goes to the end of the utterance, there is not necessarily a -pau-
	  // or other "real" word item that goes to this end time.  Therefore, it may be necessary
	  // to "extend" the last real word ending to the </s> ending time...
	  float st	 = prevRealNode->htkinfo->time;
	  // add 1 frame to start time except at beginning of utterance....
	  // if (st > 0) st += 0.01;
	  float et	 = thisNode->htkinfo->time;
	  float dur	 = thisNode->htkinfo->time - prevRealNode->htkinfo->time;
	  LogP acoustic	 = thisNode->htkinfo->acoustic;
	  LogP language	 = thisNode->htkinfo->language;
	  LogP post	 = thisNode->htkinfo->xscore2;
	  LogP conf	 = thisNode->htkinfo->xscore1;
	  char sst[50], set[50], sdur[50];
	  char spost[50], sconf[50], sacoustic[50], slanguage[50];

	  // horrible code because msvc is not c99 compliant!
	  float2string(sst, st);
	  float2string(set, et);
	  float2string(sdur, dur);
	  float2string(sacoustic, acoustic);
	  float2string(slanguage, language);
	  float2string(spost, post);
	  float2string(sconf, conf);
	  
	  nbestOut.nbestRttm2->fprintf("%s (st=%s,et=%s,dur=%s,a=%s,g=%s,p=%s,c=%s) ",
				       lat.getWord(thisNode->word), 
				       sst, set, sdur, sacoustic, slanguage, spost, sconf);

	  // remember last node for correct time information, prevNode
	  // can be a HTK_SU node and timing is sometimes right,
	  // sometimes wronge...
	  prevRealNode = thisNode; 
	}
      }
    }
    nbestOut.nbest->fprintf("\n");
    if (nbestOut.nbestRttm2) nbestOut.nbestRttm2->fprintf("\n");

    if (nbestOut.nbestRttm) {
      free(speaker);    
      free(session);
      free(time);
    }

    if (nbestOut.nbestOutDirNgram) {
      nbestOut.nbestNgram->fprintf("%.*lg\n", LogP_Precision, (double)ngram);
    }
    if (nbestOut.nbestOutDirPron) {
      nbestOut.nbestPron->fprintf("%.*lg\n", LogP_Precision, (double)pron);
    }
    if (nbestOut.nbestOutDirDur) {
      nbestOut.nbestDur->fprintf("%.*lg\n", LogP_Precision, (double)duration);
    }
    if (nbestOut.nbestOutDirXscore1) {
      nbestOut.nbestXscore1->fprintf("%.*lg\n", LogP_Precision, (double)xscore1);
    }
    if (nbestOut.nbestOutDirXscore2) {
      nbestOut.nbestXscore2->fprintf("%.*lg\n", LogP_Precision, (double)xscore2);
    }
    if (nbestOut.nbestOutDirXscore3) {
      nbestOut.nbestXscore3->fprintf("%.*lg\n", LogP_Precision, (double)xscore3);
    }
    if (nbestOut.nbestOutDirXscore4) {
      nbestOut.nbestXscore4->fprintf("%.*lg\n", LogP_Precision, (double)xscore4);
    }
    if (nbestOut.nbestOutDirXscore5) {
      nbestOut.nbestXscore5->fprintf("%.*lg\n", LogP_Precision, (double)xscore5);
    }
    if (nbestOut.nbestOutDirXscore6) {
      nbestOut.nbestXscore6->fprintf("%.*lg\n", LogP_Precision, (double)xscore6);
    }
    if (nbestOut.nbestOutDirXscore7) {
      nbestOut.nbestXscore7->fprintf("%.*lg\n", LogP_Precision, (double)xscore7);
    }
    if (nbestOut.nbestOutDirXscore8) {
      nbestOut.nbestXscore8->fprintf("%.*lg\n", LogP_Precision, (double)xscore8);
    }
    if (nbestOut.nbestOutDirXscore9) {
      nbestOut.nbestXscore9->fprintf("%.*lg\n", LogP_Precision, (double)xscore9);
    }

    return true;
  } else {
    cerr << "Not writing nbest lists because no out dir is specified\n";
    return false;
  }
}

// return a string consisting of all hypotheses words
char *
LatticeNBestHyp::getHypFeature(SubVocab &ignoreWords, Lattice &lat,
						const char *multiwordSeparator)
{
    unsigned featureLen = 0;
    LatticeNBestPath *prev;

    for (prev = nbestPath; prev != 0; prev = prev->pred) {
        LatticeNode *node = lat.findNode(prev->node);
	assert(node != 0);

        if (!lat.ignoreWord(node->word) && !ignoreWords.getWord(node->word)) {
	    featureLen += strlen(lat.getWord(node->word)) + 1;
        }           
    }

    char *feature = (char *)malloc(featureLen + 1);
    assert(feature != 0);
    feature[featureLen] = '\0';

    for (prev = nbestPath; prev != 0; prev = prev->pred) {
        LatticeNode *node = lat.findNode(prev->node);

        if (!lat.ignoreWord(node->word) && !ignoreWords.getWord(node->word)) {
	    unsigned wordLen = strlen(lat.getWord(node->word));

	    featureLen -= wordLen + 1;

	    strcpy(&feature[featureLen], lat.getWord(node->word));
	    feature[featureLen + wordLen] =
				multiwordSeparator ? *multiwordSeparator : ' ';
        }           
    }

    return feature;
}

