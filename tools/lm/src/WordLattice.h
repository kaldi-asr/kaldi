/*
 * WordLattice.h --
 *	Word lattices
 *
 * Copyright (c) 1995-1998 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/WordLattice.h,v 1.14 2003/11/01 07:07:21 stolcke Exp $
 *
 */

#ifndef _WordLattice_h_
#define _WordLattice_h_

#include "MultiAlign.h"

#include "Array.h"

/*
 * A node in a word lattice
 */
class WordLatticeNode
{
public:
    WordLatticeNode();

    VocabIndex word;		// word associated with this node
    Prob score;			// posterior probability for this node
    unsigned align;		// node equiv class from alignment
#define NO_ALIGN	((unsigned)-1)

    Array<unsigned> succs;	// indices to successor nodes
    Array<Prob> probs;		// transition probabilities
    unsigned numSuccs;		// number of successor nodes
};


/*
 * A lattice of words
 */
class WordLattice: public MultiAlign
{
public:
    WordLattice(Vocab &vocab, const char *myname = 0);
    ~WordLattice();

    Boolean read(File &file);
    Boolean write(File &file);
    Boolean read1(File &file);	// read old version lattice format
    Boolean write1(File &file);	// write old version lattice format

    unsigned sortNodes(unsigned *sortedNodes);		// topological sort
    unsigned sortAlignedNodes(unsigned *sortedNodes);	// sort with alignments

    // hypID parameter is currently ignored
    void addWords(const VocabIndex *words, Prob score, const HypID *hypID = 0);
    void alignWords(const VocabIndex *words, Prob score, Prob *wordScores = 0,
							const HypID *hypID = 0);

    unsigned wordError(const VocabIndex *words,
				unsigned &sub, unsigned &ins, unsigned &del);

    double minimizeWordError(VocabIndex *words, unsigned length,
				double &sub, double &ins, double &del,
				unsigned flags = 0, double delBias = 1.0);
#define WORDLATTICE_NOVITERBI	0x01		/* flag value */
    
    Boolean isEmpty();
    Boolean hasArc(unsigned from, unsigned to)
	{ Prob prob; return hasArc(from, to, prob); };
    Boolean hasArc(unsigned from, unsigned to, Prob &prob);
    void addArc(unsigned from, unsigned to, Prob prob = 1.0);

    unsigned numNodes;		// number of nodes

private:
    Array<WordLatticeNode> nodes;	// node list

    unsigned numAligns;		// number of alignment equivalences

    unsigned initial;		// start node index
    unsigned final;		// final node index

    void sortNodesRecursive(unsigned index, unsigned &numVisited,
			unsigned *sortedNodes, Boolean *visitedNodes);
};

#endif /* _WordLattice_h_ */

