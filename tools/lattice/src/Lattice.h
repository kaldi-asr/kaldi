/*
 * Lattice.h --
 *	Word lattices
 *
 * Copyright (c) 1997-2010 SRI International, 2013 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lattice/src/Lattice.h,v 1.108 2015-06-25 07:19:39 stolcke Exp $
 *
 */

#ifndef _Lattice_h_
#define _Lattice_h_

/* ******************************************************************
   header files 
   ****************************************************************** */
#include <math.h>

#include "Prob.h"
#include "Boolean.h"
#include "Array.h"
#include "LHash.h"
#include "SArray.h"
#include "Map2.h"
#include "Vocab.h"
#include "LM.h"
#include "SubVocab.h"
#include "File.h"
#include "Debug.h"
#include "Ngram.h"
#include "VocabMultiMap.h"
#include "MultiwordVocab.h"
#include "WordMesh.h"
#include "NBest.h"

#include "HTKLattice.h"

class Lattice;               /* forward declaration */

typedef const VocabIndex *VocabContext;

typedef unsigned NodeIndex;

const NodeIndex NoNode = (NodeIndex)(-1);

/* ******************************************************************
   flags for node
   ****************************************************************** */
const unsigned markedFlag = 8;     //to indicate that this node is processed

/* ******************************************************************
   flags for transition
   ****************************************************************** */
const unsigned pauseTFlag = 2;     //there was a pause node on the link
const unsigned directTFlag = 4;    //this is a non-pause link between two nodes
const unsigned markedTFlag = 8;    //to indicate that this trans has been 
                                   //processed
const unsigned reservedTFlag = 16; //to indicate that this trans needs to be 
                                   //  reserved for bigram;

/* ******************************************************************
   other constants
   ****************************************************************** */

const int minIntlog = -250000;	   // minumum intlog values used when
				   // printing PFSGs.  Value is chosen such
				   // that PFSG probs can be safely converted
				   // to bytelogs in the recognizer.

extern const char *LATTICE_OR;		   // lattice disjunction
extern const char *LATTICE_CONCATE;	   // lattice concatenation

extern const char *LATTICE_NONAME;	   // default internal lattice name

/* ******************************************************************
   structure
   ****************************************************************** */


inline LogP combWeights(LogP weight1, LogP weight2)
{
  return (weight1+weight2); 
}

inline LogP 
unionWeights(LogP weight1, LogP weight2)
{
  return (weight1 > weight2 ? weight1 : weight2); 
}

inline LogP unit()
{
  return 0;
}    

inline int
nodeSort(NodeIndex n1, NodeIndex n2)
{
  return (n1 - n2);
}

class NodeQueue;

typedef struct SelfLoopDB {

  // initA
  NodeIndex preNodeIndex; 
  NodeIndex postNodeIndex2;
  NodeIndex postNodeIndex3;
  NodeIndex nodeIndex;
  VocabIndex wordName; 
  unsigned selfTransFlags; 
  LogP loopProb;

  // initB
  NodeIndex fromNodeIndex;
  VocabIndex fromWordName; 
  LogP fromPreProb; 
  LogP prePostProb; 
  unsigned fromSelfTransFlags; 

  // initC
  NodeIndex toNodeIndex;
  unsigned selfToTransFlags;

} SelfLoopDB;

#ifdef USE_SARRAY
#define TRANS_T		SArray
#define TRANSITER_T	SArrayIter
#else
#define TRANS_T		LHash
#define TRANSITER_T	LHashIter
#endif

/* *************************
 * A transition in a lattice
 * ************************* */
class LatticeTransition 
{ 
  public: 
    LatticeTransition() : weight(0), flags(0) {};
    LatticeTransition(LogP weight, unsigned flags)
	: weight(weight), flags(flags) {};
    
    void markTrans(unsigned flag) { flags |= flag; }; 

    void setWeight(LogP givenWeight) { weight = givenWeight; }; 

    Boolean getFlag(unsigned flag) { return (flags & flag); }; 

    LogP weight;		// weight (e.g., probability) of transition
    unsigned flags;		// miscellaneous flags;
}; 

/*
 * Compare two transition lists for equality
 */
Boolean
compareTransitions(const TRANS_T<NodeIndex,LatticeTransition> &transList1,
		   const TRANS_T<NodeIndex,LatticeTransition> &transList2);

/* *************************
 * A node in a lattice
 ************************* */
class LatticeNode
{
    friend class LatticeTransition;

public:
    LatticeNode();     // initializing lattice node;

    unsigned flags; 
    VocabIndex word;		// word associated with this node
    LogP2 posterior;		// node posterior (unnormalized)
    HTKWordInfo *htkinfo;	// HTK lattice info

    TRANS_T<NodeIndex,LatticeTransition> outTransitions;// outgoing transitions
    TRANS_T<NodeIndex,LatticeTransition> inTransitions; // incoming transitions

    void
      markNode(unsigned flag) { flags |= flag; }; 
    // set to one the bits indicated by flag;
 
    void
      unmarkNode(unsigned flag) { flags &= ~flag; }; 
    // set to zero the bits indicated by flag;
 
    Boolean
      getFlag(unsigned flag) { return (flags & flag); };
};

/* *************************
 * Output file arguments for Lattice::computeNBest()
 ************************* */
class NBestOptions
{
 public:
  NBestOptions(char *nbestOutDir,        char *nbestOutDirNgram,
	       char *nbestOutDirPron,    char *nbestOutDirDur,
	       char *nbestOutDirXscore1, char *nbestOutDirXscore2,
	       char *nbestOutDirXscore3, char *nbestOutDirXscore4,
	       char *nbestOutDirXscore5, char *nbestOutDirXscore6,
	       char *nbestOutDirXscore7, char *nbestOutDirXscore8,
	       char *nbestOutDirXscore9,
	       char *nbestOutDirRttm, char *nbestOutDirRttm2);
  ~NBestOptions();

  char *nbestOutDir;
  char *nbestOutDirNgram;
  char *nbestOutDirPron;
  char *nbestOutDirDur;
  char *nbestOutDirXscore1;
  char *nbestOutDirXscore2;
  char *nbestOutDirXscore3;
  char *nbestOutDirXscore4;
  char *nbestOutDirXscore5;
  char *nbestOutDirXscore6;
  char *nbestOutDirXscore7;
  char *nbestOutDirXscore8;
  char *nbestOutDirXscore9;
  char *nbestOutDirRttm;
  char *nbestOutDirRttm2;

  Boolean writingFiles;

  File *nbest;
  File *nbestNgram;
  File *nbestPron;
  File *nbestDur;
  File *nbestXscore1;
  File *nbestXscore2;
  File *nbestXscore3;
  File *nbestXscore4;
  File *nbestXscore5;
  File *nbestXscore6;
  File *nbestXscore7;
  File *nbestXscore8;
  File *nbestXscore9;
  File *nbestRttm;
  // If set, extract detailed NBest word information in (an
  // approximation of) the NBestList2.0 format
  File *nbestRttm2;

  Boolean makeDirs(Boolean overwrite);
  Boolean openFiles(const char *name);
  Boolean closeFiles();
};

class PackedNodeList; 
class NodePathInfo;

/* *************************
 * A lattice 
 ************************* */

class Lattice: public Debug
{
    friend class NodeQueue;
    friend class PackedNodeList; 
    friend class LatticeTransition;
    friend class LatticeNode;

public:

    /* *************************************************
       within single lattice operations
       ************************************************* */
    Lattice(Vocab &vocab, const char *name = LATTICE_NONAME);
    Lattice(Vocab &vocab, const char *name, SubVocab &ignoreVocab);
    virtual ~Lattice();

    Boolean computeNodeEntropy(); 
    LogP detectSelfLoop(NodeIndex nodeIndex);
    Boolean recoverPauses(Boolean loop = true, Boolean all = false); 
    Boolean recoverCompactPauses(Boolean loop = true, Boolean all = false); 
    Boolean removeAllXNodes(VocabIndex xWord);
    Boolean replaceWeights(LM &lm); 
    Boolean simplePackBigramLattice(unsigned iters = 0, Boolean maxAdd = false);
    Boolean approxRedBigramLattice(unsigned iters, int base, double ratio);
    Boolean expandToTrigram(LM &lm, unsigned maxNodes = 0);
    Boolean expandToCompactTrigram(Ngram &ngram, unsigned maxNodes = 0);
    Boolean expandToLM(LM &lm, unsigned maxNodes = 0, Boolean compact = false);
    Boolean noBackoffWeights;	// hack to suppress backoff weights in expansion
    Boolean collapseSameWordNodes(SubVocab &exceptions);
    void splitMultiwordNodes(MultiwordVocab &vocab, LM &lm);
    void splitHTKMultiwordNodes(MultiwordVocab &vocab,
		LHash<const char *, Array< Array<char *> * > > &multiwordDict);
    static Boolean readMultiwordDict(File &file, 
		LHash<const char *, Array< Array<char *> * > > &multiwordDict);
    Boolean scorePronunciations(VocabMultiMap &dictionary,
						Boolean intlogs = false);
    void alignLattice(WordMesh &sausage, double posteriorScale = 1.0)
	{ alignLattice(sausage, ignoreVocab, posteriorScale); }
    void alignLattice(WordMesh &sausage, 
                      SubVocab &ignoreWords,
                      double posteriorScale = 1.0,
                      Boolean acousticInfo = false);
    void alignLattice(WordMesh &sausage, 
                      SubVocab &ignoreWords,
                      LHash<NodeIndex,unsigned>& latticeNodeToUnsortedMeshMap,
                      LHash<NodeIndex,VocabIndex>& latticeNodeToVocabMap,
                      double posteriorScale = 1.0,
                      Boolean acousticInfo = false);

    void addWords(const VocabIndex *words, Prob prob, Boolean pauses = false);

    /* *************************************************
        operations with two lattices
       ************************************************* */
    Boolean implantLattice(NodeIndex nodeIndex, Lattice &lat,
							float addTime = 0.0);
    Boolean implantLatticeXCopies(Lattice &lat);
    Boolean latticeCat(Lattice &lat1, Lattice &lat2,
						float interSegmentTime = 0.0);
    Boolean latticeOr(Lattice &lat1, Lattice &lat2);

    /* ********************************************************* 
       lattice input and output 
       ********************************************************* */

    Boolean readPFSG(File &file);
    Boolean readPFSGs(File &file);
    Boolean readPFSGFile(File &file);
    Boolean readRecPFSGs(File &file);
    Boolean readHTK(File &file, HTKHeader *header = 0,
				Boolean useNullNodes = false);
    Boolean readMesh(File &file);
    Boolean createFromMesh(WordMesh &inputMesh);

    Boolean writePFSG(File &file);
    Boolean writeCompactPFSG(File &file);
    Boolean writePFSGFile(File &file);
    Boolean writeHTK(File &file, HTKScoreMapping scoreMapping = mapHTKnone,
					    Boolean writePosteriors = false);

    void setHTKHeader(HTKHeader &header);

    Boolean useUnk;		// map unknown words to <unk>
    Boolean keepUnk;		// keep unknown word labels
    Boolean limitIntlogs;	// whether output probs should fit in bytelogs
    Boolean printSentTags;	// whether sentence start/end tags are printed

    /* ********************************************************* 
       nodes and transitions
       ********************************************************* */
    Boolean insertNode(const char *word, NodeIndex nodeIndex); 
    Boolean insertNode(const char *word) {
	return insertNode(word, maxIndex++); };
    // duplicate a node with the same word name; 
    NodeIndex dupNode(VocabIndex windex, unsigned markedFlag = 0,
						    HTKWordInfo *htkinfo = 0);
    Boolean removeNode(NodeIndex nodeIndex);  
    // all the edges connected with this node will be removed;  
    LatticeNode *findNode(NodeIndex nodeIndex) {
        return nodes.find(nodeIndex); };
    void removeAll();		// remove all nodes
    unsigned removeUselessNodes();		

    Boolean insertTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex, 
			const LatticeTransition &trans, Boolean maxAdd = false);
    LatticeTransition *findTrans(NodeIndex fromNodeIndex,
						    NodeIndex toNodeIndex);
    Boolean setWeightTrans(NodeIndex fromNodeIndex, 
			   NodeIndex toNodeIndex, LogP weight); 
    void markTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex, 
		   unsigned flag);
    Boolean removeTrans(NodeIndex fromNodeIndex, NodeIndex toNodeIndex);

    void dumpFlags();
    void clearMarkOnAllNodes(unsigned flag = ~0); 
    void clearMarkOnAllTrans(unsigned flag = ~0);

    // topological sorting
    unsigned sortNodes(NodeIndex *sortedNodes, Boolean reversed = false); 
    
    // detect words ignored by lattice LM
    inline Boolean ignoreWord(VocabIndex word)
	{ return word == Vocab_None || ignoreVocab.getWord(word) != 0; };

    /* ********************************************************* 
       get protected class values
       ********************************************************* */
    NodeIndex getInitial() { return initial; }
    NodeIndex getFinal() { return final; }
    const char *getName() { return name; }
    const char *setName(const char *newname);
    NodeIndex getMaxIndex() { return maxIndex; }
    VocabString getWord(VocabIndex word);
    unsigned getNumNodes() { return nodes.numEntries(); }
    unsigned getNumTransitions();

    NodeIndex setInitial(NodeIndex index) { return (initial=index); }
    NodeIndex setFinal(NodeIndex index) { return (final=index); }

    /* ********************************************************* 
       diagnostic tools
       ********************************************************* */
    Boolean printNodeIndexNamePair(File &file); 
    Boolean areConnected(NodeIndex fromNodeIndex, NodeIndex toNodeIndex,
			 unsigned direction = 0); 
    unsigned latticeWER(const VocabIndex *words,
			unsigned &sub, unsigned &ins, unsigned &del)
        { return latticeWER(words, sub, ins, del, ignoreVocab); };
    unsigned latticeWER(const VocabIndex *words,
			unsigned &sub, unsigned &ins, unsigned &del,
			SubVocab &ignoreWords); 

    LogP2 computePosteriors(double posteriorScale = 1.0,
						Boolean normalize = false);
    Boolean writePosteriors(File &file, double posteriorScale = 1.0);
    Boolean prunePosteriors(Prob threshold, double posteriorScale = 1.0,
				double maxDensity = 0.0, unsigned maxNodes = 0,
				Boolean fast = false);

    LogP bestWords(VocabIndex *words, unsigned maxLength)
	{ return bestWords(words, maxLength, ignoreVocab); }
    LogP bestWords(VocabIndex *words, unsigned maxLength,
						    SubVocab &ignoreWords);
    LogP bestWords(NBestWordInfo *words, unsigned maxLength,
						    SubVocab &ignoreWords);
    Boolean computeNBest(unsigned N, NBestOptions &nbestOut,
					SubVocab &ignoreWords,
					const char *multiwordSeparator = 0,
					unsigned maxHyps = 0,
					unsigned nbestDuplicates = 0);
    Boolean computeNBestViterbi(unsigned N, NBestOptions &nbestOut,
					SubVocab &ignoreWords,
					const char *multiwordSeparator = 0);

    LogP decode1Best(VocabIndex *words, unsigned maxWords,
    		     SubVocab &ignoreWords, LM *lm,
		     unsigned contextLen, double beamwidth,
		     LogP logP_floor = LogP_Zero, unsigned maxPaths = 0);

    LogP decode1Best(NBestWordInfo *winfo, unsigned maxWords,
    		     SubVocab &ignoreWords, LM *lm,
		     unsigned contextLen, double beamwidth,
		     LogP logP_floor = LogP_Zero, unsigned maxPaths = 0);
  
    Boolean decodeNBest(unsigned N, NBestOptions &nbestOut,
    			SubVocab &ignoreWords, LM *lm,
			unsigned contextLen, unsigned maxFanIn,
			double beamwidth, const char *multiwordSeparator,
			LogP logP_floor = LogP_Zero, unsigned maxPaths = 0);

    unsigned findBestPath(unsigned n, VocabString *words, NodeIndex *path,
			  unsigned maxNodes, LogP &prob);
    unsigned findBestPath(VocabIndex *words, NodeIndex *path,
			  unsigned maxNodes, LogP & prob);

    FloatCount countNgrams(unsigned order, NgramCounts<FloatCount> &counts,
						double posteriorScale = 1.0);
    FloatCount indexNgrams(unsigned order, File &file,
    					Prob minCount = 0.0,
					NBestTimestamp maxPause = 0.0,
					NBestTimestamp timeTolerance = 0.0,
					double posteriorScale = 1.0, Vocab *keywords = NULL);

    double getDuration();
    double computeDensity(double duration = 0.0)
	{ unsigned numNodes; return computeDensity(numNodes, duration); }
    double computeDensity(unsigned &numNodes, double duration = 0.0);

    /* ********************************************************* 
       self-loop processing
       ********************************************************* */
    static void initASelfLoopDB(SelfLoopDB &selfLoopDB, LM &lm,
				NodeIndex nodeIndex, LatticeNode *node,
				LatticeTransition *trans);
    static void initBSelfLoopDB(SelfLoopDB &selfLoopDB, LM &lm, 
				NodeIndex fromNodeIndex, LatticeNode *fromNode,
				LatticeTransition *fromTrans);
    static void initCSelfLoopDB(SelfLoopDB &selfLoopDB, NodeIndex toNodeIndex,
				LatticeTransition *toTrans);

    Boolean expandSelfLoop(LM &lm, SelfLoopDB &selfLoopDB, 
				PackedNodeList &packedSelfLoopNodeList);

    /* ********************************************************* 
       internal data structure
       ********************************************************* */
    Vocab &vocab;		// vocabulary used for words
    LHash<NodeIndex,LatticeNode> nodes;	// node list; 

    SubVocab ignoreVocab;	// words to ignore in lattice operations

    HTKHeader htkheader;	// HTK header information

    static void freeThread();

protected: 
    Array<HTKWordInfo *> htkinfos;
				// HTK link information (to avoid duplication
				// inside node structures)
    LHash<VocabIndex, Lattice *> subPFSGs;  // for processing subPFSGs

    NodeIndex maxIndex;		// the current largest node index plus one
				// (i.e., the next index we can allocate)
    const char *name;		// name string for lattice
    double duration;		// waveform duration

    NodeIndex initial;		// start node index
    NodeIndex final;		// final node index;

    Boolean top;                // an indicator for whether two null nodes 
				// (initial and final) have been converted
				// to <s> and </s>.

    Lattice *getNonRecPFSG(VocabIndex nodeVocab);
    Boolean recoverPause(NodeIndex nodeIndex, Boolean loop = true,
							Boolean all = false);
    Boolean recoverCompactPause(NodeIndex nodeIndex, Boolean loop = true,
							Boolean all = false); 

    void sortNodesRecursive(NodeIndex nodeIndex, unsigned &numVisited,
			    NodeIndex *sortedNodes, Boolean *visitedNodes);

    LogP2 computeForwardBackward(LogP2 forwardProbs[], LogP2 backwardProbs[],
							double posteriorScale);
    LogP computeForwardBackwardViterbi(LogP forwardProbs[],
							LogP backwardProbs[]);

    LogP computeViterbi(NodeIndex forwardPreds[], NodeIndex *backwardPreds = 0);

    typedef void NgramAccumulatorFunction(Lattice *lat,
						const NBestWordInfo *ngram,
						Prob count,
						void *clientData);
    Prob countNgramsAtNode(VocabIndex oldIndex, unsigned order,
		 LogP2 backwardProbs[], double posteriorScale,
		 Map2<NodeIndex, const NBestWordInfo *, LogP2> &forwardProbMap,
		 Lattice::NgramAccumulatorFunction *accumulator,
		 void *clientData, Boolean acousticInfo = false);

    Boolean
      expandNodeToTrigram(NodeIndex nodeIndex, LM &lm, unsigned maxNodes = 0);

    Boolean
      expandNodeToCompactTrigram(NodeIndex nodeIndex, Ngram &ngram,
							unsigned maxNodes = 0);

    Boolean
      expandAddTransition(VocabIndex *usedContext, unsigned usedLength,
		  VocabIndex word, LogP wordProb, LM &lm,
		  NodeIndex oldIndex2, NodeIndex newIndex,
		  LatticeTransition *oldtrans, unsigned maxNodes,
		  Map2<NodeIndex, VocabContext, NodeIndex> &expandMap);

    Boolean
      expandNodeToLM(VocabIndex node, LM &ngram, unsigned maxNodes, 
			Map2<NodeIndex, VocabContext, NodeIndex> &expandMap);

    Boolean
      expandNodeToCompactLM(VocabIndex node, LM &ngram, unsigned maxNodes,
			Map2<NodeIndex, VocabContext, NodeIndex> &expandMap);

    void
      mergeNodes(NodeIndex nodeIndex1, NodeIndex nodeIndex2,
			LatticeNode *node1 = 0, LatticeNode *node2 = 0,
			Boolean maxAdd = false);

    Boolean 
      approxMatchInTrans(NodeIndex nodeIndexI, NodeIndex nodeIndexJ,
			 unsigned overlap); 

    Boolean 
      approxMatchOutTrans(NodeIndex nodeIndexI, NodeIndex nodeIndexJ,
			  unsigned overlap); 

    void
      packNodeF(NodeIndex nodeIndex, Boolean maxAdd = false);

    void
      packNodeB(NodeIndex nodeIndex, Boolean maxAdd = false);

    Boolean 
      approxRedNodeF(NodeIndex nodeIndex, NodeQueue &nodeQueue, 
		     int base, double ratio);

    Boolean 
      approxRedNodeB(NodeIndex nodeIndex, NodeQueue &nodeQueue, 
		     int base, double ratio);

    // helpers to alignLattice()
    NodeIndex findMaxPosteriorNode(LHash<NodeIndex,LogP2> &nodeSet, LogP2 &max);
    unsigned findFirstAligned(NodeIndex from, NodeIndex predecessors[],
				LHash<NodeIndex, unsigned> &nodeSet,
				NodeIndex pathNodes[]);


    NodePathInfo **decode(unsigned contextLen, LM *lm, unsigned finalPosition,
    			  NodeIndex *sortedNodes, double beamwidth,
			  float lmscale, int nbest, int maxFanIn,
			  LogP logP_floor = LogP_Zero, unsigned maxPaths = 0);

    // helper for findBestPath(...)
    void pathFinder(NodeIndex nodeIndex, LatticeNode *node, unsigned depth,
		    VocabIndex *wids, unsigned numMatched, NodeIndex *path,
		    LogP prob, NodeIndex *bestPath, unsigned maxNodes,
		    LogP &bestProb, unsigned &bestLength,
		    LHash<long, LogP> &records);
};

/*
 * Iterators to enumerate non-null successor nodes
 */
class LatticeFollowIter
{
public:
    LatticeFollowIter(Lattice &lat, LatticeNode &node,
		      LHash<NodeIndex, LogP> *useVisitedNodes = 0,
		      LogP totalWeight = LogP_One);
    ~LatticeFollowIter();

    void init();
    LatticeNode *next(NodeIndex &followIndex, LogP &weight);

private:
    Lattice &lat;
    TRANSITER_T<NodeIndex,LatticeTransition> transIter;
    LatticeFollowIter *subFollowIter;
    LogP startWeight;		// accumulated weight from start node
    LHash<NodeIndex, LogP> *visitedNodes;
    Boolean freeVisitedNodes;
};

class QueueItem: public Debug
{
  // for template class, we need to add;
  // friend class Queue<Type>; 

friend class NodeQueue; 
  
public:
    QueueItem(NodeIndex nodeIndex, unsigned clevel = 0, 
	      LogP cweight = 0.0); 
    
    NodeIndex QueueGetItem() { return item; }
    unsigned QueueGetLevel() { return level; }
    LogP QueueGetWeight() { return weight; }

private:
    NodeIndex item;
    unsigned level; 
    LogP weight; 
    QueueItem *next; 
};

class NodeQueue: public Debug
{
public:
    NodeQueue() { queueHead = queueTail = 0; }
    virtual ~NodeQueue(); 

    NodeIndex popNodeQueue(); 
    QueueItem *popQueueItem(); 

    Boolean is_empty() {
      return queueHead == 0 ? true : false; }

    // Boolean is_full();
      
    Boolean addToNodeQueue(NodeIndex nodeIndex, unsigned level = 0, 
			   LogP weight = 0.0); 

    Boolean addToNodeQueueNoSamePrev(NodeIndex nodeIndex, unsigned level = 0, 
			   LogP weight = 0.0); 

    // Boolean pushQueueItem(QueueItem queueItem); 

    Boolean inNodeQueue(NodeIndex nodeIndex);

private:

    QueueItem *queueHead;      // the head of the node queue;
    QueueItem *queueTail;      // the tail of the node queue;
    
};


typedef struct PackedNode {
  // currently, this information is not used.
  // In a later version, I will test this List first, and if the wordName
  //   is not found, then compute trigram prob. This way, I can gain
  //   some efficiency.
  // if there is a trigramProg, both bigramProg and backoffWeight are 0;
  // if there is a bigramProg,  trigramProg will be 0;

  // In the latest version, no distinction made for the order of ngram.
  // All the weights are treated as same, and they are based only their
  // positions.
  // LogP inWeight;
  LogP outWeight;
  unsigned toNodeId; 
  NodeIndex toNode; 
  NodeIndex midNodeIndex; 
} PackedNode; 

// this is an input structure for PackedNodeList::packingNodes
typedef struct PackInput {

  VocabIndex fromWordName;  // starting node name
  VocabIndex wordName;      // current node name
  VocabIndex toWordName;    // ending node name
  LM *lm;		    // LM to compute outWeight on demand

  unsigned toNodeId; 
  NodeIndex fromNodeIndex;  // starting node
  NodeIndex toNodeIndex;    // ending node, 
		  // a new node will be created in between these two nodes.  
  NodeIndex nodeIndex;      // for debugging purpose

  LogP inWeight;            // weight for an in-coming trans to the new node
  LogP outWeight;           // weight for an out-going trans from the new node
  unsigned inFlag; 
  unsigned outFlag; 

} PackInput; 

class PackedNodeList: public Debug 
{
public: 
  PackedNodeList() : lastPackedNode(0) {};
  virtual ~PackedNodeList() {};
  Boolean packNodes(Lattice &lat, PackInput &packInput);

private:
  LHash<VocabIndex,PackedNode> packedNodesByFromNode;
  PackedNode *lastPackedNode;
}; 


#endif /* _Lattice_h_ */

