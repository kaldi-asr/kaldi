/*
 * LatticeDecode.cc --
 *	Viterbi decoding of lattices without expansion
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2004-2012 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeDecode.cc,v 1.21 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <time.h>
#include <string>
#include <assert.h>

#include <vector>
#include <algorithm>
using namespace std;

#include "Lattice.h"
#include "LatticeNBest.h"
#include "HTKLattice.h"

#include "Array.cc"
#include "IntervalHeap.cc"
#include "LHash.cc"
#include "CachedMem.cc"

const unsigned MAX_LM_ORDER = 10;

struct LatticeDecodePath;
typedef int (* sorter) (const void *, const void *);

struct PATH_PTR { 
  LatticeDecodePath * p; 
  PATH_PTR () { p = 0; }
  PATH_PTR(LatticeDecodePath * path ) { p = path; }
  operator LatticeDecodePath * () { return p; }
};

typedef LHash<PATH_PTR, LatticeDecodePath *> PathHash;
typedef LHashIter<PATH_PTR, LatticeDecodePath *> PathHashIter;

struct LatticeDecodePath;
struct PathLink : public CachedMem<PathLink> { 
  LatticeDecodePath * prev;
  LogP diff;
  PathLink * next;
  
  PathLink(LatticeDecodePath * pa, LogP pd, PathLink * pn) 
  { prev = pa; diff = pd; next = pn;  }
  PathLink() { prev = 0; diff = 0; next = 0; }
};

// sort in descending order 
static int cmpLinkPtr(const PathLink ** l1, const PathLink ** l2);

class NodePathInfo;  
struct LatticeDecodePath : CachedMem<LatticeDecodePath>
{
  LatticeNode * m_Node;
  VocabIndex *	m_History;

  LogP  m_Prob, m_GProb;
  LatticeDecodePath * m_Prev;
  
  // for nbest
  unsigned m_NumPreds;
  PathLink *  m_Preds;
  LHash<const char *, LogP> m_Seen;

  // for LM
  VocabIndex m_Context[MAX_LM_ORDER];

  LatticeDecodePath() {
    m_Node = 0;
    m_Prob = LogP_One;
    m_GProb = LogP_One;
    m_Prev = 0;
    m_NumPreds = 0;

    // for nbest
    m_Preds = 0;

  }

  ~LatticeDecodePath() {
    for (PathLink * p = m_Preds; p;) {
      m_Preds = p->next;
      delete p;
      p = m_Preds;
    }
  }
  
  void addLink(LatticeDecodePath * path, LogP diff) {
    PathLink * l = new PathLink(path, diff, m_Preds);
    assert(l != 0);

    m_Preds = l;
    m_NumPreds ++;
  }

  void merge(LatticeDecodePath * p, int nbest) {
    
    if (m_Prob < p->m_Prob) {
      m_Prob = p->m_Prob;
      m_GProb = p->m_GProb;
      m_Prev = p->m_Prev;
    }
    
    if (nbest) {	
      LogP diff = p->m_Prob;
      if (p->m_Prev)
	diff -= p->m_Prev->m_Prob;

      addLink(p->m_Prev, diff);
    }
    
    delete p;  
  }

  int truncLinks(unsigned maxDegree) {

    if (maxDegree == 0 || m_NumPreds <= maxDegree) 
      return m_NumPreds;

    unsigned i = 0;
    PathLink * p;
    Array<PathLink *> array(0, m_NumPreds);
    p = m_Preds;
    for (i = 0; i < m_NumPreds; i++, p = p->next) {
      array[i] = p;
    }
    qsort(array.data(), m_NumPreds, sizeof(PathLink *), (sorter) cmpLinkPtr);
    
    // re-organize link list, always keep one at least
    p = m_Preds = array[0];    
    for (i = 1; i < maxDegree ; i++) {
      p->next = array[i];
      p = array[i];
    }

    p->next = 0;

    for (; i < m_NumPreds; i++)
      delete array[i];
    
    return (m_NumPreds = maxDegree);
  }
};

static int 
cmpLinkPtr(const PathLink ** l1, const PathLink ** l2) {
  
  LogP p1 = (*l1)->prev->m_Prob + (*l1)->diff;
  LogP p2 = (*l2)->prev->m_Prob + (*l2)->diff;
  
  if (p1 < p2) 
    return 1;
  else if (p1 > p2)
    return -1;
  else
    return 0;
}

// sort in descending order
int 
cmpPath(const LatticeDecodePath ** p1, const LatticeDecodePath ** p2)
{
  float pr1 = (*p1)->m_Prob;
  float pr2 = (*p2)->m_Prob;

  if (pr1 > pr2)
    return -1;
  else if (pr1 < pr2)
    return 1;
  else 
    return 0;

}

class NodePathInfo : public CachedMem <NodePathInfo> {

public:

  NodePathInfo():m_PHash(0) { m_OldLMScore = 0; m_ProbBwd = LogP_Zero; m_PList = 0; m_NumPaths = 0;  }
  ~NodePathInfo() { delete [] m_PList; }

  LogP	m_OldLMScore; // m_OldLMScore
  unsigned m_NumPaths; // number of paths
  LogP	m_ProbBwd; // backward prob
  PathHash	m_PHash;
  LatticeDecodePath **	m_PList;  
};


/***************************
 * A-star N-best generation
 **************************/

class NBestPath : public CachedMem<NBestPath> {

public:
  NBestPath() {
    m_LatPath = 0; m_Prob = m_ProbBwd = LogP_One; m_Prev = 0;
    m_Depth = 0;
    m_Feature = 0;
  }
  ~NBestPath() { if (m_Feature) free(m_Feature); }

  Boolean writeHyp(int hypNum, Lattice &lat, SubVocab & ignoreWords, NBestOptions &nbestOut, LM * lm);
  char *  getHypFeature(SubVocab & ignoreWords, Lattice & lat, const char * multiwordSeparator);

  LatticeDecodePath * m_LatPath;  
  LogP m_Prob, m_ProbBwd;
  int m_Depth;
  NBestPath * m_Prev;
  NBestPath * m_PrevAlloc;
  char * m_Feature;

};

/***************************
 * IntervalHeap support
 ***************************/
struct NPLess {
  bool operator() (const NBestPath * p1, const NBestPath * p2) {
    return (p1->m_Prob < p2->m_Prob) || (p1->m_Prob == p2->m_Prob && p1->m_Depth < p2->m_Depth);        
  }
};

struct NPGreater {
  bool operator() (const NBestPath * p1, const NBestPath * p2) {
    return (p1->m_Prob > p2->m_Prob) || (p1->m_Prob == p2->m_Prob && p1->m_Depth > p2->m_Depth);
  }
};

struct NPEqual {
  bool operator() (const NBestPath * p1, const NBestPath * p2) {
    return (p1->m_Prob == p2->m_Prob) && (p1->m_Depth == p2->m_Depth);
  }
};

/*************************************
 *          for LHash
 *************************************/
inline void Map_noKey(PATH_PTR & key) { key.p = 0; }
inline bool Map_noKeyP(PATH_PTR key) { return (key.p == 0); }
inline ostream & operator << (ostream & os, const PATH_PTR & p) { return os; }

inline size_t
LHash_hashKey(const PATH_PTR & key, unsigned maxBits) {

  register VocabIndex * p = key.p->m_Context;
  register unsigned i = 0;
  while (*p != Vocab_None) {
    i += (i<<12) + *p++;
  }
  return LHash_hashKey(i, maxBits);
}

inline Boolean
LHash_equalKey(const PATH_PTR & key1, const PATH_PTR & key2)
{
  register VocabIndex * p1 = key1.p->m_Context;
  register VocabIndex * p2 = key2.p->m_Context;

  while (*p1 != Vocab_None && *p2 != Vocab_None) {
    if (*p1++ != *p2++)
      return false;
  }
  return true;
}

static void
freePaths(NodePathInfo ** nodeinfo, unsigned finalPosition)
{

  for (unsigned i = 0; i <= finalPosition; i++) {
    NodePathInfo & info = *nodeinfo[i];
     for (unsigned j = 0; j < info.m_NumPaths; j++)
       delete info.m_PList[j];
    delete nodeinfo[i];
  }
}

NodePathInfo **
Lattice::decode(unsigned contextLen, LM * lm, unsigned finalPosition, NodeIndex * sortedNodes, 
		double beamwidth, float lmscale, int nbest, int maxFanIn, LogP logP_floor, 
		unsigned maxPaths)
{
  unsigned i;  

  unsigned nn = getNumNodes();
  LHash<int,int> nodeMap(nn);

  assert(finalPosition < nn);

  NodePathInfo ** nodeinfo = new NodePathInfo *[ nn ];
  assert(nodeinfo != 0);

  for (i = 0; i < nn; i++) {
    nodeinfo[i] = new NodePathInfo;
    assert(nodeinfo[i] != 0);
    *nodeMap.insert(sortedNodes[i]) = i;
  }

  unsigned numPaths = 0;

  LogP thresh = LogP_Zero;
  
  if (beamwidth != 0) {
    // run a backward pass
    VocabIndex dummyContext = Vocab_None;
    nodeinfo[finalPosition]->m_ProbBwd = LogP_One;

    for (i = finalPosition; (int)i >= 0; i --) {
      LatticeNode * node = nodes.find(sortedNodes[i]);
      assert (node != 0);

      // when old lattice does not have language model score, use unigram score 
      // to estimate backward probability
      LogP unigramScore = LogP_One;
      if (lm && !ignoreWord(node->word) && node->htkinfo && node->htkinfo->language == HTK_undef_float)
	unigramScore = lm->wordProb(node->word, &dummyContext) * lmscale;
            
      LogP prob = nodeinfo[i]->m_ProbBwd;
      TRANSITER_T<NodeIndex, LatticeTransition> inTransIter(node->inTransitions);
      NodeIndex fromNodeIndex;
      
      while (LatticeTransition * inTrans = inTransIter.next(fromNodeIndex)) {
	
	LogP prob2 = inTrans->weight + prob + unigramScore;
	unsigned mappedIndex = *nodeMap.insert(fromNodeIndex);
	NodePathInfo & fromInfo = *nodeinfo[mappedIndex];
	if (prob2 > fromInfo.m_ProbBwd) {
	  fromInfo.m_ProbBwd = prob2;
	}
      }
    } 
    
    thresh = nodeinfo[0]->m_ProbBwd - beamwidth;
  } else {
    for (i = 0; i <= finalPosition; i++)
      nodeinfo[i]->m_ProbBwd = LogP_One;
  }

  // get old LM scores to correctly compute new transition weight
  if (lm) {
    for (i = 0; i <= finalPosition; i++) {
      LatticeNode * node = nodes.find(sortedNodes[i]);
      if (node->htkinfo && node->htkinfo->language != HTK_undef_float) 
	nodeinfo[i]->m_OldLMScore = node->htkinfo->language * lmscale;
      else 
	nodeinfo[i]->m_OldLMScore = 0;
    }
  }

  // for the initial node
  LatticeDecodePath * path = new LatticeDecodePath; // this is dummy initial one
  assert(path != 0);
  numPaths = 1;
  path->m_Node = nodes.find(initial);
  if (contextLen > 1) {
    path->m_Context[0] = path->m_Node->word;
  }
  
  for (i = 1; i < contextLen - 1; i++)  
    path->m_Context[i] = Vocab_None;

  path->m_Context[contextLen - 1] = Vocab_None; 

  nodeinfo[0]->m_PList = new LatticeDecodePath * [1];
  assert(nodeinfo[0]->m_PList != 0);

  nodeinfo[0]->m_PList[0] = path;
  nodeinfo[0]->m_NumPaths = 1;

  // go through all the nodes, make all transitions
  NodeIndex n;
  for (n = 1 ; n <= finalPosition; n++) {
    
    LatticeNode * node = nodes.find(sortedNodes[n]);
    VocabIndex word = node->word;
    NodePathInfo & info = *nodeinfo[n];
    LogP oldlmscore = info.m_OldLMScore;
    Boolean nolmword = ignoreWord(word);
    Boolean uselm = (lm != NULL) && (!nolmword);
    LogP probBwd = info.m_ProbBwd;
    
    // go through all in-transitions
    TRANSITER_T<NodeIndex,LatticeTransition> inTransIter(node->inTransitions);
    NodeIndex fromNodeIndex;

    while (LatticeTransition * inTrans = inTransIter.next(fromNodeIndex)) {

      unsigned mappedIndex = *nodeMap.insert(fromNodeIndex);
      NodePathInfo & fromInfo = *nodeinfo[mappedIndex];
      for (i = 0; i < fromInfo.m_NumPaths; i++) {
	LatticeDecodePath * path = fromInfo.m_PList[i];
	
	LogP prob = path->m_Prob + inTrans->weight - oldlmscore;
	LogP lmscore = LogP_One;
	LogP gprob = LogP_One;

	if (uselm) {
	  gprob = lm->wordProb(word, path->m_Context);
	  if (gprob < logP_floor) gprob = logP_floor;
	  lmscore = gprob * lmscale;
	  prob += lmscore;
	} else {
	  if (node->htkinfo) 
	    gprob = node->htkinfo->language;
	}

	if ((prob + probBwd) >= thresh) {
	
	  LatticeDecodePath * newpath = new LatticeDecodePath;
	  assert(newpath != 0);
	  numPaths ++;
	  if (maxPaths && numPaths > maxPaths)
	    goto FINAL;
	  newpath->m_Prob = prob;
	  newpath->m_GProb = path->m_GProb + gprob;
	  newpath->m_Node = node;
	  newpath->m_Prev = path;
	  if (nbest) {
	    newpath->addLink(path, prob - path->m_Prob);
	  }
	  
	  if (nolmword) {
	    memcpy(newpath->m_Context, path->m_Context, sizeof(VocabIndex) * contextLen);
	  } else {
	    newpath->m_Context[0] = word;
	    if (contextLen > 2)
	      memcpy(newpath->m_Context + 1, path->m_Context, sizeof(VocabIndex) * (contextLen - 2));
	    newpath->m_Context[contextLen - 1] = Vocab_None;
	  }

	  LatticeDecodePath ** p;
	  if ((p = info.m_PHash.find(PATH_PTR(newpath)))) {
	    LatticeDecodePath * op = *p;
	    op->merge(newpath, nbest);
	  } else {
	    *info.m_PHash.insert(PATH_PTR(newpath)) = newpath;
	  }
	} 
      }
    }


    // sort the list
    int num = info.m_PHash.numEntries();     
    info.m_PList = new LatticeDecodePath * [ num ];
    assert(info.m_PList != 0);
    
    PATH_PTR ptr;
    PathHashIter piter(info.m_PHash);
    i = 0;
    while (piter.next(ptr)) {
      info.m_PList[i++] = ptr.p;
      if (nbest && maxFanIn) {
	ptr.p->truncLinks(maxFanIn);
      }
    }

    info.m_PHash.clear(); 
    info.m_NumPaths = num;
  } 

 FINAL:
  
  NodePathInfo * pfinal = nodeinfo[finalPosition];
 
  if (pfinal->m_NumPaths == 0) {
    freePaths(nodeinfo, finalPosition);
    delete [] nodeinfo;
    return NULL;
  }
  qsort(pfinal->m_PList, pfinal->m_NumPaths, sizeof(LatticeDecodePath *), (sorter)cmpPath);  

  return nodeinfo;
}

void 
Lattice::freeThread() {
    PathLink::freeThread();
    LatticeDecodePath::freeThread();
}

void
swap(NBestWordInfo & w1, NBestWordInfo & w2)
{
  if (&w1 == &w2) {
    return;
  }

  VocabIndex tw = w1.word;
  w1.word = w2.word;
  w2.word = tw;

  NBestTimestamp tt = w1.start;
  w1.start = w2.start;
  w2.start = tt;

  tt = w1.duration;
  w1.duration = w2.duration;
  w2.duration = tt;

  LogP tp;
  tp = w1.acousticScore;
  w1.acousticScore = w2.acousticScore;
  w2.acousticScore = tp;

  tp = w1.languageScore;
  w1.languageScore = w2.languageScore;
  w2.languageScore = tp;

  char * ts = w1.phones;
  w1.phones = w2.phones;
  w2.phones = ts;

  ts = w1.phoneDurs;
  w1.phoneDurs = w2.phoneDurs;
  w2.phoneDurs = ts;

  Prob tpr = w1.wordPosterior;
  w1.wordPosterior = w2.wordPosterior;
  w2.wordPosterior = tpr;

  tpr = w1.transPosterior;
  w1.transPosterior = w2.transPosterior;
  w2.transPosterior = tpr;
  
}

LogP
Lattice::decode1Best(VocabIndex *words, unsigned maxWords, SubVocab &ignoreWords,
		     LM * lm , unsigned contextLen, double beamwidth, LogP logP_floor, 
		     unsigned maxPaths)
{
  NBestWordInfo * winfo = new NBestWordInfo [ maxWords + 1];
  assert(winfo != 0);

  LogP result = decode1Best(winfo, maxWords, ignoreWords, lm, contextLen, beamwidth, logP_floor, maxPaths);

  if (result != LogP_Zero) {
    for (unsigned i = 0; i < maxWords; i++) {
      words[i] = winfo[i].word;
      if (words[i] == Vocab_None)
	break;
    }
  } else {
    words[0] = Vocab_None;
  }

  delete [] winfo;
  return result;

}

LogP
Lattice::decode1Best(NBestWordInfo *winfo, unsigned maxWords, SubVocab &ignoreWords,
		     LM * lm, unsigned contextLen, double beamwidth, LogP logP_floor,
		     unsigned maxPaths)
{

  // check if lmscale is available.
  float lmscale = htkheader.lmscale;
  if (lm) {
    if (lmscale == HTK_undef_float || lmscale <= 0) {
      dout() << "Lattice::decode1Best: error: lmscale not specified!\n";
      return LogP_Zero;
    }
  } else {
    lmscale = 1.0f;
  }

  if (contextLen <= 1 || !lm) {
    // use at least 1  
    contextLen = 1;
  }

  if (contextLen > MAX_LM_ORDER) {
    dout() << "Lattice::decode1Best: error: lm order too high, can support up to " << MAX_LM_ORDER << endl;
    return false;
  }

  unsigned numNodes = getNumNodes();  
  NodeIndex * sortedNodes = new NodeIndex [ numNodes ];
  assert(sortedNodes != 0);

  unsigned numReachable = sortNodes(sortedNodes);
  
  if (numReachable != numNodes) {
    dout() << "Lattice::decode1Best: warning: called with unreachable nodes\n";
  }

  if (sortedNodes[0] != initial) {
    dout() << "Lattice::decode1Best: warning: initial node is not the first\n";
    delete [] sortedNodes;
    return LogP_Zero;
  }

  unsigned finalPosition = 0;
  for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
    if (sortedNodes[finalPosition] == final) break;
  }

  if (finalPosition == numReachable) {
    dout() << "Lattice::decode1Best: final node is not reachable\n";
    delete [] sortedNodes;
    return LogP_Zero;
  }

  NodePathInfo ** nodeinfo = decode(contextLen, lm, finalPosition, sortedNodes,
				    beamwidth, lmscale, 0, 0, logP_floor, maxPaths);

  if (!nodeinfo) {
    delete [] sortedNodes;
    return LogP_Zero;
  }

  LatticeDecodePath * path = nodeinfo[finalPosition]->m_PList[0];
  LogP result = path->m_Prob;
  unsigned num = 0;
  LogP gprob = LogP_One;

  while (path && num < maxWords) {
    LatticeNode * node = path->m_Node;
    if (!ignoreWord(node->word) && !ignoreWords.getWord(node->word)) {
      
      NBestWordInfo & wi = winfo[num++];

      if (node->htkinfo == 0 && node->word == Vocab_None) {
	wi.invalidate();
      } if (node->htkinfo == 0) {
	// only word label is available, e.g., when processing PFSGs
	wi.word = node->word;
	wi.languageScore = path->m_GProb; // this is cumulative score
      } else {
	// find the time of the predecessor node
	float startTime = 0.f;
	float lastGProb = LogP_One;
	LatticeNode * prevNode;
	
	if (path->m_Prev && (prevNode = path->m_Prev->m_Node) && 
	    prevNode->htkinfo) {	  
	  startTime = prevNode->htkinfo->time;	  
	}
	
	wi.word = node->word;
	wi.start = startTime;
	wi.duration = node->htkinfo->time - startTime;
	wi.acousticScore = node->htkinfo->acoustic;	
	wi.languageScore = path->m_GProb; // this is cumulative score
      }
      
      wi.wordPosterior = wi.transPosterior = 1.0;
    }
        
    path = path->m_Prev;
  }

  freePaths(nodeinfo, finalPosition);
  delete [] nodeinfo;
  delete [] sortedNodes;

  if (path) {

    dout() << "Lattice::decode1Best: " << "word string longer than " << maxWords << endl;
    winfo[0].word = Vocab_None;
    return LogP_Zero;

  } else {
    // reverse the sequence
    NBestWordInfo * p1 = winfo;
    NBestWordInfo * p2 = winfo + num - 1;
    while (p1 < p2) {

      // swap content
      swap(*p1++, *p2--);
    }

    winfo[num].word = Vocab_None;
    for (int i = num - 1; i > 0; i--) {
      winfo[i].languageScore -= winfo[i-1].languageScore;
    }
  }

  return result;
}

char * 
NBestPath::getHypFeature(SubVocab &ignoreWords, Lattice & lat, const char * multiwordSeparator)
{
  string feature;
  const char * sep = (multiwordSeparator ? multiwordSeparator : " ");
  
  NBestPath * p = this;

  while (p) {
    if (p->m_LatPath) {
      LatticeNode * node = p->m_LatPath->m_Node;
      if (node) {
	VocabIndex word = node->word;
	if (!lat.ignoreWord(word) &&
	    !ignoreWords.getWord(node->word)) {
	  feature += lat.getWord(word);
	  feature += sep;
	}
      }
    }
    p = p->m_Prev;
  }

  return strdup(feature.c_str());
}

Boolean
NBestPath::writeHyp(int hypNum, Lattice & lat, SubVocab & ignoreWords, NBestOptions &nbestOut, LM * lm)
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
  if (nbestOut.nbestOutDir) {

    char *speaker = NULL, channel, *time = NULL, *session = NULL;
    float start_time;

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

    unsigned wordCnt = 0;
    Array<VocabIndex> sentence(0, 128);
    LogP acoustic = 0, ngram = 0, language = 0, pron = 0, duration = 0;
    LogP xscore1 = 0, xscore2 = 0, xscore3 = 0, xscore4 = 0, xscore5 = 0;
    LogP xscore6 = 0, xscore7 = 0, xscore8 = 0, xscore9 = 0;
    Boolean haveHTKscores = false;
    
    NBestPath * p = this;
    string text;

    LatticeNode * prevNode = p->m_LatPath->m_Node;
    while(p) {
      
      LatticeNode * node = p->m_LatPath->m_Node;

      if (node) {
	if (!lat.ignoreWord(node->word) && // NULL & -pau-
	    !ignoreWords.getWord(node->word) &&
	    !lat.vocab.isNonEvent(node->word) && // <s> and other non-events
	    node->word != lat.vocab.seIndex()) {
	  sentence[wordCnt ++] = node->word;
	}
	if (node->word != Vocab_None) {
	  text += lat.getWord(node->word);
	  text += " ";
	}
	
	if (nbestOut.nbestRttm) {
	  if (node->word == HTK_SU) {
	    if (node->htkinfo) {
	      nbestOut.nbestRttm->fprintf(
		      "%d SU     %s %c %.2f X <NA> statement %s <NA>\n",    
		      hypNum, session, channel,
		      start_time + node->htkinfo->time, speaker);
	    }
	  } else {
	    if (node->htkinfo && prevNode && prevNode->htkinfo) {
	      nbestOut.nbestRttm->fprintf(
		      "%d LEXEME %s %c %.2f %.2f %s lex %s <NA>\n",
		      hypNum, session, channel,
		      start_time+prevNode->htkinfo->time,
		      node->htkinfo->time - prevNode->htkinfo->time,
		      lat.getWord(node->word), speaker);
	    }
	  }
	}

	if (node->htkinfo) {
	  haveHTKscores = true;

	  if (node->htkinfo->acoustic != HTK_undef_float)
	    acoustic += node->htkinfo->acoustic;
	  if (node->htkinfo->language != HTK_undef_float)
	    language += node->htkinfo->language;
	  if (node->htkinfo->ngram != HTK_undef_float) 
	    ngram    += node->htkinfo->ngram;
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
      }

      prevNode = node;
          
      p = p->m_Prev;
    }

    if (lm) {
      TextStats ts;
      sentence[wordCnt] = lat.vocab.seIndex();
      sentence[wordCnt + 1] = Vocab_None;
      language = lm->sentenceProb(sentence.data(), ts);
    }

    if (!haveHTKscores) {
      float lmScale = lat.htkheader.lmscale;
      if (lmScale == HTK_undef_float) lmScale = 1.0;

      acoustic = this->m_Prob - language * lmScale;
    }

    nbestOut.nbest->fprintf("%.*lg %.*lg %u %s\n",
					LogP_Precision, (double)acoustic,
					LogP_Precision, (double)language,
					wordCnt, text.c_str());
    
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

    if (session) {
      free(session);
    }

    if (time) {
      free(time);
    }

    if (speaker) {
      free(speaker);
    }

    return true;
  } else {
    cerr << "Not writing nbest lists because no out dir is specified\n";
    return false;
  }
    
}


Boolean
Lattice::decodeNBest(unsigned N, NBestOptions &nbestOut, SubVocab &ignoreWords, 	     
		     LM * lm, unsigned contextLen, unsigned maxDegree, double beamwidth,
		     const char *multiwordSeparator, LogP logP_floor, unsigned maxPaths)
{
  if (beamwidth == 0) 
    beamwidth = 1e30;
   
  const char * mwsep = (multiwordSeparator ? multiwordSeparator : " ");

  // check if lmscale is available.
  float lmscale = htkheader.lmscale;
  if (lmscale == HTK_undef_float || lmscale <= 0) {
    dout() << "Lattice::decodeNBest: error: lmscale not specified!\n";
    return false;
  }

  if (contextLen <= 1 || !lm) {
    // use at least 1  
    contextLen = 1;
  }

  if (contextLen > MAX_LM_ORDER) {
    dout() << "Lattice::decodeNBest: error: lm order too high, can support up to " << MAX_LM_ORDER << endl;
    return false;
  }
  
  unsigned numNodes = getNumNodes();  
  NodeIndex * sortedNodes = new NodeIndex [ numNodes ];
  assert(sortedNodes != 0);

  unsigned numReachable = sortNodes(sortedNodes);
  
  if (numReachable != numNodes) {
    dout() << "Lattice::decodeNBest: warning: called with unreachable nodes\n";
  }
  
  if (sortedNodes[0] != initial) {
    dout() << "Lattice::decodeNBest: warning: initial node is not the first\n";
    delete [] sortedNodes;
    return false;
  }
  
  unsigned finalPosition = 0;
  for (finalPosition = 1; finalPosition < numReachable; finalPosition ++) {
    if (sortedNodes[finalPosition] == final) break;
  }
  
  if (finalPosition == numReachable) {
    dout() << "Lattice::decodeNBest: final node is not reachable\n";
    delete [] sortedNodes;
    return false;
  }

  NodePathInfo ** nodeinfo = decode(contextLen, lm, finalPosition, sortedNodes, 
                                    beamwidth, lmscale, 1, maxDegree, LogP_Zero, maxPaths);
  
  if (!nodeinfo) {
    delete [] sortedNodes;
    return false;
  }

  LHash<const char *, unsigned> hypsPrinted;

  /* A* searh */
  IntervalHeap<NBestPath*, NPLess, NPGreater, NPEqual> pathQueue (2 * N + 1);
  
  NodePathInfo & info = *nodeinfo[finalPosition];
  NBestPath * allocList = 0;

  LogP thresh = info.m_PList[0]->m_Prob - beamwidth;
  
  // add all final hyps
  unsigned i;
  for (i = 0; i < info.m_NumPaths; i++) {
    LatticeDecodePath * lpath = info.m_PList[i];
    NBestPath * npath = new NBestPath;
    npath->m_LatPath = lpath;
    npath->m_Prob = lpath->m_Prob;
    npath->m_ProbBwd = LogP_One;
    npath->m_Prev = 0;
    npath->m_Feature = strdup("</s>");
    npath->m_PrevAlloc = allocList;
    allocList = npath;
    
    pathQueue.push(npath); 
  }

  unsigned outputHyps = 0;

  while(outputHyps < N && !pathQueue.empty()) {
    
    NBestPath * topPath = pathQueue.top_max();
    pathQueue.pop_max();

    if (topPath->m_LatPath->m_Prev == 0) {

      // the algorithm guarantee uniqueness

      outputHyps++;
      if (!topPath->writeHyp(outputHyps, *this, ignoreWords, nbestOut, lm)) {
	cerr << "could not write hyp " << outputHyps << " for lattice " << this->getName() << endl;
      }
    } else {
      // extend the path

      unsigned n = topPath->m_LatPath->m_NumPreds;
      PathLink * lptr = topPath->m_LatPath->m_Preds;
      for (i = 0; i < n; i++, lptr = lptr->next) {
	
	LatticeDecodePath * prev = lptr->prev;
	  
	LogP prob = prev->m_Prob + lptr->diff + topPath->m_ProbBwd;
	if (prob > thresh) {

	  string feature = topPath->m_Feature;
	  VocabIndex w = prev->m_Node->word;
	  if (!ignoreWord(w) && !ignoreWords.getWord(w)) {
	    feature += mwsep;
	    feature += vocab.getWord(w);
	  }

	  LogP * pp;
	  if ((pp = prev->m_Seen.find(feature.c_str()))) {
	    if (prob <= *pp) 
	      continue; // this string has been seen, no improvement, do not try
	  } 

	  *prev->m_Seen.insert(feature.c_str()) = prob;
	  
	  NBestPath * npath = new NBestPath;
	  assert(npath != 0);

	  npath->m_LatPath = prev;
	  npath->m_Prob = prob;
	  npath->m_ProbBwd = topPath->m_ProbBwd + lptr->diff;
	  npath->m_Prev = topPath;
	  npath->m_Depth = topPath->m_Depth + 1;
	  npath->m_Feature = strdup(feature.c_str());
	  npath->m_PrevAlloc = allocList;
	  allocList = npath;
	  
	  pathQueue.push(npath);
	}
      }      
    }
  }

  while(allocList) {
    NBestPath * next = allocList->m_PrevAlloc;
    delete allocList;
    allocList = next;
  }

  freePaths(nodeinfo, finalPosition);
  delete [] nodeinfo;
  delete [] sortedNodes;

  return true;
  
}

unsigned
Lattice::findBestPath(unsigned n, VocabString *words, NodeIndex *path, unsigned maxNodes, LogP & prob)
{
  makeArray(VocabIndex, wids, n + 1);
  unsigned i;
  for (i = 0; i < n; i++) {
    if ((wids[i] = vocab.getIndex(words[i])) == Vocab_None)
      break;
    if (ignoreWord(wids[i]))
      n --, i --;
  }
  
  if (i < n || i == 0) {
    // there is OOV or empty sentence
    return 0;
  } else {
    wids[n] = Vocab_None;
  }

  return findBestPath(wids, path, maxNodes, prob);

}

inline bool mySortFunc(pair<NodeIndex, LogP> p1, pair<NodeIndex, LogP> p2)
{
  return (p2.second < p1.second);
}

void
Lattice::pathFinder(NodeIndex nodeIndex, LatticeNode * node, unsigned depth, VocabIndex * wids, 
		    unsigned numMatched, NodeIndex * path, LogP prob, NodeIndex *bestPath, 
		    unsigned maxNodes, LogP & bestProb, unsigned & bestLength, LHash<long, LogP> & records)
{
  unsigned i;

  if (depth >= maxNodes)
    return;

  // Since numMatched <= maxNodes is always true, so the pair 
  // (numMatched, nodeIndex) has a one-to-one map to the key defined below
  long key = (long) numMatched + (long) maxNodes * (long) nodeIndex;

  Boolean foundP;
  LogP & oldProb = *records.insert(key, foundP);
  if (foundP) {
    if (prob <= oldProb) {
      // visited and not better, give up
      return;
    }
  } else {
    oldProb = prob;
  }
  
  if (nodeIndex == final) { 
    if (wids[numMatched] != Vocab_None)
      return;
      
    if (prob <= bestProb)
      return;

    path[depth++] = nodeIndex;

    bestLength = depth;
    bestProb = prob;
    memcpy(bestPath, path, sizeof(NodeIndex) * depth);

    return;
  }

  if (wids[numMatched] == node->word && node->word != Vocab_None) {
    numMatched++;
  }

  path[depth++] = nodeIndex;

  TRANSITER_T<NodeIndex, LatticeTransition> outTransIter(node->outTransitions);

  NodeIndex toIndex;
  vector<pair<NodeIndex, LogP> > trans;
  
  while (LatticeTransition * outTrans = outTransIter.next(toIndex)) {

    LatticeNode * to = findNode(toIndex);
    VocabIndex w = to->word;
    if (w != wids[numMatched] && !ignoreWord(w) && w != vocab.ssIndex() && w != vocab.seIndex())
      continue;
    
    LogP newProb = prob + outTrans->weight;

    trans.push_back(pair<NodeIndex, LogP>(toIndex, newProb));
  }

  sort(trans.begin(), trans.end(), mySortFunc);
  
  for (vector<pair<NodeIndex,LogP> >::iterator it = trans.begin(); it != trans.end(); it++) {

    NodeIndex toIndex = it->first;
    LatticeNode * to = findNode(toIndex);
    pathFinder(it->first, to, depth, wids, numMatched, path, it->second, bestPath, maxNodes, 
	       bestProb, bestLength, records);
  }
}

unsigned
Lattice::findBestPath(VocabIndex *wids, NodeIndex *path, unsigned maxNodes, LogP & prob)
{
  makeArray(NodeIndex, work, maxNodes);
  
  LHash<long, LogP> records (maxNodes);

  unsigned bestLength = 0;
  LogP bestProb = LogP_Zero;

  LatticeNode * iNode = findNode(initial);

  pathFinder(initial, iNode, 0, wids, 0, work, LogP_One, path, maxNodes, 
	     bestProb, bestLength, records);

  prob = bestProb;
 
  return bestLength;
}
