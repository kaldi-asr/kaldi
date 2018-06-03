/*
 * LatticeNBest.h --
 *	Extract NBest lists from Lattice
 *
 * Copyright (c) 1997-2008, SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeNBest.h,v 1.1 2008/04/30 01:55:47 stolcke Exp $
 *
 */

#ifndef _LatticeNBest_h_
#define _LatticeNBest_h_

/* *************************
 * A path through the lattice (stored as reversed linked list)
 * ************************* */
class LatticeNBestPath
{
 friend class LatticeNBestHyp;

 public:
  LatticeNBestPath(NodeIndex node, LatticeNBestPath *predecessor);
  ~LatticeNBestPath();
  void linkto();	// add a reference to this path
  void release();	// remove a reference to this path
  unsigned getPath(Array<NodeIndex> &path);

  NodeIndex node;
  LatticeNBestPath *pred;

 private:
  unsigned numReferences;
};

class LatticeNBestHyp
{
 public:
  LatticeNBestHyp(double score, LogP myForwardProb, 
                  NodeIndex myNodeIndex, int mySuccIndex, Boolean endOfSent,
		  LatticeNBestPath *nbestPath, unsigned myWordCnt,
		  LogP myAcoustic, LogP myNgram, LogP myLanguage,
		  LogP myPron, LogP myDuration,
		  LogP myXscore1, LogP myXscore2, LogP myXscore3,
		  LogP myXscore4, LogP myXscore5, LogP myXscore6,
		  LogP myXscore7, LogP myXscore8, LogP myXscore9);
  ~LatticeNBestHyp();
  
  double score; // score for path (forward prob plus total backward prob)
  LogP forwardProb; // forward prob so far on this path
  Boolean endOfSent;
  LatticeNBestPath *nbestPath;		// linked list of nodes to the start

  NodeIndex nodeIndex;  
  int succIndex;

  // Accumulated HTK scores over path
  unsigned wordCnt;                     // number of words (ignore non-words)
  LogP acoustic;			// acoustic model log score
  LogP ngram;				// ngram model log score
  LogP language;			// language model log score
  LogP pron;				// pronunciation log score
  LogP duration;			// duration log score
  LogP xscore1;			        // extra score #1
  LogP xscore2;			        // extra score #2
  LogP xscore3;			        // extra score #3
  LogP xscore4;			        // extra score #4
  LogP xscore5;			        // extra score #5
  LogP xscore6;			        // extra score #6
  LogP xscore7;			        // extra score #7
  LogP xscore8;			        // extra score #8
  LogP xscore9;			        // extra score #9

  Boolean writeHyp(int hypNum, Lattice &lat, NBestOptions &nbestOut);
  char *getHypFeature(SubVocab &ignoreWords, Lattice &lat,
						const char *multiwordSeparator);
};

#endif /* _LatticeNBest_h_ */
