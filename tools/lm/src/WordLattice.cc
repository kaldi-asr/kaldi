/*
 * WordLattice.cc --
 *	Word lattices
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/WordLattice.cc,v 1.42 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "WordLattice.h"

#include "WordAlign.h"			/* for *_COST constants */

#include "Array.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_ARRAY(WordLatticeNode);
#endif

/*
 * LHash over lattice edges (pairs of nodes)
 */

class NodePair
{
public:
    NodePair(unsigned n1 = 0, unsigned n2 = 0) : node1(n1), node2(n2) {} ;

    int operator==(NodePair &np)
	{ return node1 == np.node1 && node2 == np.node2; };
    unsigned node1, node2;
};

#include "LHash.cc"

static ostream &
operator<<(ostream &str, NodePair &key)
{
    return str << "(" << key.node1 << "->" << key.node2 << ")";
}

static inline size_t
LHash_hashKey(NodePair &key, unsigned maxBits)
{
    return LHash_hashKey(key.node1 + 10 * key.node2, maxBits);
}

static inline void
Map_noKey(NodePair &key)
{
    key.node1 = key.node2 = (unsigned)(-1);
}

inline Boolean
Map_noKeyP(const NodePair &key)
{
    return key.node1 == (unsigned)(-1) &&
           key.node2 == (unsigned)(-1);
}

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(NodePair, Boolean);
#endif

/*
 * Generic array reversal
 */
template <class T>
void reverseArray(T *array, unsigned length)
{
    int i, j;	/* j can get negative ! */

    for (i = 0, j = length - 1; i < j; i++, j--) {
	T x = array[i];
	array[i] = array[j];
	array[j] = x;
    }
}

WordLattice::WordLattice(Vocab &vocab, const char *myname)
    : MultiAlign(vocab, myname), numNodes(0), numAligns(0)
{
    initial = numNodes ++;
    final = numNodes ++;

    nodes[initial].numSuccs = 0;
    nodes[final].numSuccs = 0;

    /*
     * Initialize alignment classes so that 0 and 1 correspond to initial and
     * final. This makes the job easier in sortAlignedNodes().
     */
    nodes[initial].align = numAligns ++;
    nodes[final].align = numAligns ++;
}

WordLattice::~WordLattice()
{
    if (name != 0) {
	free(name);
    }
}

WordLatticeNode::WordLatticeNode()
    : word(Vocab_None), score(0.0), align(NO_ALIGN), numSuccs(0)
{
}

Boolean
WordLattice::isEmpty()
{
    return numNodes == 2 && nodes[initial].numSuccs == 0;
}

Boolean
WordLattice::hasArc(unsigned from, unsigned to, Prob &prob)
{
    if (from < numNodes) {
	WordLatticeNode &lnode = nodes[from];

	for (unsigned i = 0; i < lnode.numSuccs; i++) {
	    if (lnode.succs[i] == to) {
		prob = lnode.probs[i];
		return true;
	    }
	}
    }
    return false;
}

void
WordLattice::addArc(unsigned from, unsigned to, Prob prob)
{
    if (from >= numNodes) {
	numNodes = from + 1;
    }

    WordLatticeNode &lnode = nodes[from];

    /*
     * See if the arc is already there.  If so, do nothing.
     */
    for (unsigned i = 0; i < lnode.numSuccs; i++) {
	if (lnode.succs[i] == to) {
	    lnode.probs[i] += prob;
	    return;
	}
    }

    /*
     * Add the arc
     */
    lnode.succs[lnode.numSuccs] = to;
    lnode.probs[lnode.numSuccs] = prob;
    lnode.numSuccs += 1;
    return;
}

Boolean
WordLattice::write1(File &file)
{
    file.fprintf("initial %u\n", initial);
    file.fprintf("final %u\n", final);

    for (unsigned i = 0; i < numNodes; i ++) {
	file.fprintf("node %u %s %.*lg", i,
		nodes[i].word == Vocab_None ?
			"NULL" : vocab.getWord(nodes[i].word),
		Prob_Precision, (double)nodes[i].score);

	for (unsigned j = 0; j < nodes[i].numSuccs; j ++) {
	    file.fprintf(" %u", nodes[i].succs[j]);
	}
	file.fprintf("\n");
    }
    return true;
}

Boolean
WordLattice::write(File &file)
{
    file.fprintf("version 2\n");
    if (name != 0) {
	file.fprintf("name %s\n", name);
    }
    file.fprintf("initial %u\n", initial);
    file.fprintf("final %u\n", final);

    for (unsigned i = 0; i < numNodes; i ++) {
	file.fprintf("node %u %s %u %.*lg", i,
		nodes[i].word == Vocab_None ?
			"NULL" : vocab.getWord(nodes[i].word),
		nodes[i].align,
		Prob_Precision, (double)nodes[i].score);

	for (unsigned j = 0; j < nodes[i].numSuccs; j ++) {
	    file.fprintf(" %u %.*lg",
		    nodes[i].succs[j],
		    Prob_Precision, (double)nodes[i].probs[j]);
	}
	file.fprintf("\n");
    }
    return true;
}

Boolean
WordLattice::read1(File &file)
{
    char *line;

    while ((line = file.getline())) {
	unsigned arg1;
	char arg2[100];
	double arg3;
	int parsed;
	unsigned version;

	if (sscanf(line, "version %u", &version) == 1) {
	    if (version == 1) {
		continue;
	    } else if (version == 2) {
		return read(file);
	    } else {
		file.position() << "unknown version\n";
		return false;
	    }
	} else if (sscanf(line, "initial %u", &initial) == 1) {
	    continue;
	} else if (sscanf(line, "final %u", &final) == 1) {
	    continue;
	} else if (sscanf(line, "node %u %100s %lg%n",
				&arg1, arg2, &arg3, &parsed) == 3)
	{
	    
	    // @kw false positive: SV.TAINTED.ALLOC_SIZE (arg1)
	    WordLatticeNode &node = nodes[arg1];
	    if (arg1 >= numNodes) {
		numNodes = arg1 + 1;
	    }

	    node.word = (strcmp(arg2, "NULL") == 0) ?
				Vocab_None : vocab.addWord(arg2);
	    node.score = arg3;
	    node.align = NO_ALIGN;
	    node.numSuccs = 0;

	    // @kw false positive: SV.TAINTED.INDEX_ACCESS (parsed)
	    char *cp = line + parsed;
	    while (sscanf(cp, "%u%n", &arg1, &parsed) == 1) {
		node.succs[node.numSuccs++] = arg1;
		cp += parsed;
	    }
	} else {
	    file.position() << "unknown keyword\n";
	    return false;
	}
    }
    return true;
}

Boolean
WordLattice::read(File &file)
{
    char *line;

    while ((line = file.getline())) {
	unsigned arg1;
	char arg2[100];
	double arg3;
	unsigned arg4;
	int parsed;
	unsigned version;

	if (sscanf(line, "version %u", &version) == 1) {
	    if (version == 1) {
		return read1(file);
	    } else if (version == 2) {
		continue;
	    } else {
		file.position() << "unknown version\n";
		return false;
	    }
	} else if (sscanf(line, "name %100s", arg2) == 1) {
	    if (name != 0) {
		free(name);
	    }
	    name = strdup(arg2);
	    assert(name != 0);
	} else if (sscanf(line, "initial %u", &initial) == 1) {
	    continue;
	} else if (sscanf(line, "final %u", &final) == 1) {
	    continue;
	} else if (sscanf(line, "node %u %100s %u %lg%n",
				&arg1, arg2, &arg4, &arg3, &parsed) == 4)
	{
	    // @kw false positive: SV.TAINTED.ALLOC_SIZE (arg1)
	    WordLatticeNode &node = nodes[arg1];
	    if (arg1 >= numNodes) {
		numNodes = arg1 + 1;
	    }

	    node.word = (strcmp(arg2, "NULL") == 0) ?
				Vocab_None : vocab.addWord(arg2);
	    node.score = arg3;
	    node.align = arg4;
	    node.numSuccs = 0;

	    // @kw false positive: SV.TAINTED.INDEX_ACCESS (parsed)
	    char *cp = line + parsed;
	    double prob;
	    while (sscanf(cp, "%u %lg%n", &arg1, &prob, &parsed) == 2) {
		node.succs[node.numSuccs] = arg1;
		node.probs[node.numSuccs] = prob;
		node.numSuccs += 1;
		cp += parsed;
	    }
	} else {
	    file.position() << "unknown keyword\n";
	    return false;
	}
    }
    return true;
}

/*
 * sortNodes --
 *	Sort node indices topologically
 *
 * Result:
 *	The number of reachable nodes.
 *
 * Side effects:
 *	sortedNodes is filled with the sorted node indices.
 */
unsigned
WordLattice::sortNodes(unsigned *sortedNodes)
{
    makeArray(Boolean, visitedNodes, numNodes);

    for (unsigned i = 0; i < numNodes; i ++) {
	visitedNodes[i] = false;
    }

    unsigned numVisited = 0;

    sortNodesRecursive(initial, numVisited, sortedNodes, visitedNodes);
    
    /*
     * reverse the node order from the way we generated it
     */
    reverseArray(sortedNodes, numVisited);

    return numVisited;
}

void
WordLattice::sortNodesRecursive(unsigned index, unsigned &numVisited,
			unsigned *sortedNodes, Boolean *visitedNodes)
{
    if (visitedNodes[index]) {
	return;
    }

    visitedNodes[index] = true;

    WordLatticeNode &lNode = nodes[index];

    for (unsigned i = 0; i < lNode.numSuccs; i ++) {
	sortNodesRecursive(lNode.succs[i], numVisited, 
						sortedNodes, visitedNodes);
    }

    sortedNodes[numVisited++] = index;
}

/*
 * sortAlignedNodes --
 *	Sort node indices topologically, keeping nodes with same alignment
 *	class adjacent.
 *
 * Result:
 *	The number of reachable nodes.
 *
 * Side effects:
 *	sortedNodes is filled with the sorted node indices.
 *	Adjecent alignment classes that are ordered (have no arcs between
 *	them) are merged.
 */
unsigned
WordLattice::sortAlignedNodes(unsigned *sortedNodes)
{
    /*
     * Create a lattice of alignment classes and build a
     * a homomorphic image of the word lattice
     */
    WordLattice alat(vocab);
    unsigned i;

    assert(nodes[initial].align == initial);
    assert(nodes[final].align == final);

    for (i = 0; i < numNodes; i ++) {
	WordLatticeNode &from = nodes[i];

	for (unsigned j = 0; j < from.numSuccs; j ++) {
	    assert(from.align != NO_ALIGN);
	    assert(nodes[from.succs[j]].align != NO_ALIGN);

	    alat.addArc(from.align,
			nodes[from.succs[j]].align,
			from.probs[j]);
	}
    }

    /*
     * Construct alignment class to node maps
     */
    Array< Array<unsigned> > alignMap(0, numAligns);

    for (i = 0; i < numNodes; i ++) {
	unsigned align = nodes[i].align;
	unsigned n = alignMap[align].size();
	alignMap[align][n] = i;
    }

    /*
     * Sort alignment classes topologically
     */
    Array<unsigned> sortedAligns(0, numAligns);
    unsigned numSortedAligns = alat.sortNodes(sortedAligns.data());

    /*
     * Expand sorted alignment classes back into nodes
     */
    i = 0;
    unsigned lastAlign;
    for (unsigned k = 0; k < numSortedAligns; k ++) {
	/*
	 * If current and previous alignment class are not ordere w.r.t.
	 * each other, then merge them.
	 */
	Prob prob;
	Boolean mergeAligns = 
		    k > 0 &&
			!alat.hasArc(sortedAligns[k-1], sortedAligns[k], prob);

	for (unsigned j = 0; j < alignMap[sortedAligns[k]].size(); j ++, i ++) {
	    sortedNodes[i] = alignMap[sortedAligns[k]][j];
	    if (mergeAligns) {
		nodes[sortedNodes[i]].align = lastAlign;
	    }
	}

	if (!mergeAligns) {
	    lastAlign = sortedAligns[k];
	}
    }
    
    return i;
}

/*
 * addWords --
 *	Add new nodes representing word string
 */
void
WordLattice::addWords(const VocabIndex *words, Prob score, const HypID *hypID)
{
    unsigned prevNode = initial;

    nodes[initial].score += score;

    for (unsigned i = 0; words[i] != Vocab_None; i ++) {
	unsigned newNode = numNodes ++;

	nodes[newNode].word = words[i];
	nodes[newNode].score = score;
	nodes[newNode].align = numAligns++;
	addArc(prevNode, newNode, score);

	prevNode = newNode;
    }

    addArc(prevNode, final, score);
    nodes[final].score += score;
}

/*
 * NOTE: MAX_COST is small enough that we can add any of the *_COST constants
 * without overflow.
 */
const unsigned MAX_COST = (unsigned)(-10);	// an impossible path

/*
 * error type used in tracing back word/lattice alignments
 */
typedef enum {
	ERR_NONE, ERR_SUB, ERR_INS, ERR_DEL, ERR_ARC
} ErrorType;

/*
 * costs for local lattice alignments
 * (note these are different from word alignment costs)
 */
const unsigned LAT_SUB_COST = 2;
const unsigned LAT_DEL_COST = 3;
const unsigned LAT_INS_COST = 3;
const unsigned LAT_ARC_COST = 1;

/*
 * alignWords --
 *	Add a word string to lattice by finding best alignment
 */
void
WordLattice::alignWords(const VocabIndex *words, Prob score, Prob *wordScores,
							const HypID *hypID)
{
    unsigned numWords = Vocab::length(words);

    /*
     * The states indexing the DP chart correspond to lattice nodes.
     */
    const unsigned NO_PRED = (unsigned)(-1);	// default for pred link

    makeArray(unsigned, sortedNodes, numNodes);

    unsigned numReachable = sortAlignedNodes(sortedNodes);
    if (numReachable != numNodes) {
	cerr << "WARNING: " << getName()
	     << ": alignWords called with unreachable nodes\n";
    }

    /*
     * Build a mapping of words to lattice nodes to locate matching
     * nodes quickly.  Note we store the nodes by their index in
     * the topological sort since we have to compare order later.
     */
    LHash<VocabIndex, Array<unsigned> > nodeWordMap;

    unsigned j;
    for (j = 0; j < numReachable; j ++) {
	VocabIndex word = nodes[sortedNodes[j]].word;

	if (word != Vocab_None) {
	    Array<unsigned> *nodeList = nodeWordMap.insert(word);

	    (*nodeList)[nodeList->size()] = j;
	}
    }

    /*
     * Allocate the DP chart.
     * chartEntries are indexed by [word_position][lattice_node],
     * where word_position = 0 is the  left string margin,
     * word_position = numWords + 1 is the right string margin.
     */
    typedef struct {
	unsigned cost;		// minimal error to this state
	unsigned predNode;	// predecessor state used in getting there
	ErrorType errType;	// error type
    } ChartEntry;

    ChartEntry **chart = new ChartEntry *[numWords + 2];
    assert(chart != 0);

    unsigned i;
    for (i = 0; i <= numWords + 1; i ++) {
	chart[i] = new ChartEntry[numNodes];
	assert(chart[i] != 0);
	for (j = 0; j < numNodes; j ++) {
	    chart[i][j].cost = MAX_COST;
	    chart[i][j].predNode = NO_PRED;
	    chart[i][j].errType = ERR_NONE;
	}
    }

    /*
     * Prime the chart by anchoring the alignment at the left edge
     */
    chart[0][initial].cost = 0;
    chart[0][initial].predNode = initial;
    chart[0][initial].errType = ERR_NONE;

    /*
     * Insertions before the first word
     * NOTE: since we process nodes in topological order this
     * will allow chains of multiple insertions.
     */
    for (j = 0; j < numReachable; j ++) {
	unsigned curr = sortedNodes[j];
	WordLatticeNode &node = nodes[curr];
	unsigned insCost = chart[0][curr].cost + LAT_INS_COST;

	if (insCost >= MAX_COST) continue;

	for (unsigned s = 0; s < node.numSuccs; s ++) {
	    unsigned next = node.succs[s];

	    if (insCost < chart[0][next].cost) {
		chart[0][next].cost = insCost;
		chart[0][next].predNode = curr;
		chart[0][next].errType = ERR_INS;
	    }
	}
    }

    /*
     * For all word positions, compute minimal alignment werr for each
     * state.
     */
    for (i = 1; i <= numWords + 1; i ++) {
	/*
	 * Compute partial alignment cost for all lattice nodes
	 */
	unsigned j;
	for (j = 0; j < numReachable; j ++) {
	    unsigned curr = sortedNodes[j];
	    WordLatticeNode &node = nodes[curr];
	    unsigned cost = chart[i - 1][curr].cost;

	    if (cost >= MAX_COST) continue;

	    /*
	     * Deletion error: current word not matched by lattice
	     * To align the word string we need to create a new node
	     * as well as arc from/to previous/next nodes.
	     */
	    {
		unsigned delCost = cost + LAT_DEL_COST;

		if (delCost < chart[i][curr].cost) {
		    chart[i][curr].cost = delCost;
		    chart[i][curr].predNode = curr;
		    chart[i][curr].errType = ERR_DEL;
		}
	    }

	    /*
	     * Substitution errors:
	     * To align the word string we need to create a new node
	     * and arc from/to it.
	     */
	    for (unsigned s = 0; s < node.numSuccs; s ++) {
		unsigned next = node.succs[s];
		unsigned haveSub =
			(nodes[next].word == words[i - 1]) ? 0 : 1;
		unsigned subCost = cost + haveSub * LAT_SUB_COST;

		if (subCost < chart[i][next].cost) {
		    chart[i][next].cost = subCost;
		    chart[i][next].predNode = curr;
		    chart[i][next].errType = haveSub ? ERR_SUB : ERR_NONE;
		}
	    }

	    /*
	     * Arc deletion:
	     * To align the word string we need to create only a new
	     * arc between existing nodes.
	     * For efficiency we look up the nodes that could possibly
	     * match the next word, instead of checking all nodes that
	     * come after the current one in topological order.
	     */
	    Array<unsigned> *nodeList =
				(words[i - 1] == Vocab_None) ? 0 :
					nodeWordMap.find(words[i - 1]);

	    for (unsigned k = 0; nodeList && k < nodeList->size(); k ++) {
		unsigned nextIndex = (*nodeList)[k];

		/*
		 * Check the the next node comes after the current one
		 * in the topological sort.
		 */
		if (nextIndex > j) {
		    unsigned next = sortedNodes[nextIndex];
		    unsigned newCost = cost + LAT_ARC_COST;

		    if (newCost < chart[i][next].cost &&
			(nodes[curr].align == NO_ALIGN ||
			 nodes[curr].align != nodes[next].align))
		    {
			chart[i][next].cost = newCost;
			chart[i][next].predNode = curr;
			chart[i][next].errType = ERR_ARC;
		    }
		}
	    }
	}

	for (j = 0; j < numReachable; j ++) {
	    unsigned curr = sortedNodes[j];
	    WordLatticeNode &node = nodes[curr];
	    unsigned insCost = chart[i][curr].cost + LAT_INS_COST;

	    if (insCost >= MAX_COST) continue;

	    /*
	     * Insertion errors: lattice node not matched by word
	     * NOTE: since we process nodes in topological order this
	     * will allow chains of multiple insertions.
	     */
	    for (unsigned s = 0; s < node.numSuccs; s ++) {
		unsigned next = node.succs[s];

		if (insCost < chart[i][next].cost) {
		    chart[i][next].cost = insCost;
		    chart[i][next].predNode = curr;
		    chart[i][next].errType = ERR_INS;
		}
	    }
	}
    }

    /*
     * Viterbi backtrace to find best alignment and add new nodes/arcs
     */
    {
	unsigned bestPred = final;
	unsigned lastState = NO_PRED;

	for (i = numWords + 1; ; i --) {
	    assert(chart[i][bestPred].predNode != NO_PRED);

	    unsigned thisState;

	    /*
	     * Skip over any "inserted" lattice nodes
	     */
	    while (bestPred != NO_PRED && chart[i][bestPred].errType == ERR_INS)
	    {
		bestPred = chart[i][bestPred].predNode;
	    }

	    if (chart[i][bestPred].errType == ERR_DEL ||
		chart[i][bestPred].errType == ERR_SUB)
	    {
		/*
		 * Create a new node
		 */
		thisState = numNodes ++;
		nodes[thisState].word = words[i - 1];
		nodes[thisState].score = score;
		if (chart[i][bestPred].errType == ERR_SUB) {
		    nodes[thisState].align = nodes[bestPred].align;
		} else {
		    nodes[thisState].align = numAligns++;
		}

		if (wordScores && i > 0 && i <= numWords) {
		    wordScores[i - 1] = score;
		}
	    } else {
		/*
		 * Add the score to an existing node
		 */
		thisState = bestPred;
		nodes[thisState].score += score;

		if (wordScores && i > 0 && i <= numWords) {
		    wordScores[i - 1] = nodes[thisState].score;
		}
	    }

	    /*
	     * Add arc between this and the successor state on the path,
	     * (unless we're at the final state).
	     * If the arc already exists then this just adds the score
	     * to the existing score.
	     */
	    if (thisState != final && thisState != lastState) {
		addArc(thisState, lastState, score);
	    }

	    bestPred = chart[i][bestPred].predNode;
	    lastState = thisState;

	    if (i == 0) break;
	}
	assert(lastState == initial);
    }


    /*
     * XXX: LHash lossage.
     */
    LHashIter<VocabIndex, Array<unsigned> > nodeWordIter(nodeWordMap);
    VocabIndex word;
    Array<unsigned> *nodeList;
    while ((nodeList = nodeWordIter.next(word))) {
	nodeList->~Array();
    }

    for (i = 0; i <= numWords + 1; i ++) {
	delete [] chart[i];
    }
    delete [] chart;
}

/*
 * Search for path with lowest expected word error.
 */
class AlignErrors
{
public:
    AlignErrors() : subs(0.0), inss(0.0), dels(0.0) {};
    double subs, inss, dels;
};

typedef struct {
    double errs;		// total minimal error to this state
    unsigned pred;		// predecessor state used in getting there
    unsigned nnodes;
    double subs, inss, dels;// error counts by type
} ChartEntry;

double
WordLattice::minimizeWordError(VocabIndex *words, unsigned length,
				    double &sub, double &ins, double &del,
				    unsigned flags, double delBias)
				// delBias is ignored, unimplemented
{
    const unsigned NO_PRED = (unsigned)(-1);	// default for pred link
    double expectedError;

    /*
     * Sort nodes topologically respecting alignments
     */
    Array<unsigned> sortedNodes(0, numNodes);
    unsigned numReachable = sortAlignedNodes(sortedNodes.data());

    if (numReachable != numNodes) {
	cerr << "WARNING: " << getName()
	     << ": minimizeWordError called with unreachable nodes\n";
    }
    assert(sortedNodes[numReachable - 1] == final);

    /*
     * Create arrays to keep track of error counts by node and by edge
     */
    Array<AlignErrors> nodeErrors(0, numNodes);
    Array< Array<AlignErrors> > edgeErrors(0, numNodes);

    /*
     * Create the set of pending edges, initially empty
     */
    LHash<NodePair,Boolean> pendingEdges;

    unsigned numWords = 0;	/* result word count */
    double totalSubs = 0.0;
    double totalInss = 0.0;
    double totalDels = 0.0;

    for (unsigned i = 0; i < numReachable; ) {
	/*
	 * Find the end of the current alignment class.
	 * Also total the posterior probs for this class.
	 */
	Prob totalAlignPost = nodes[sortedNodes[i]].score;
	unsigned endOfAlign = i + 1;

	while (endOfAlign < numReachable &&
	       nodes[sortedNodes[i]].align ==
			nodes[sortedNodes[endOfAlign]].align)
	{
	    totalAlignPost += nodes[sortedNodes[endOfAlign]].score;
	    endOfAlign ++;
	}

	/*
	 * Compute insertion/deletion errors:
	 *	assign deletion errors to edges straddling the alignment class
	 *	assign insertion errors to nodes in the alignment class
	 *	compute total posterior for all unfinished edges
	 */
	Prob totalEdgePost = 0.0;
	LHashIter<NodePair,Boolean> edgeIter(pendingEdges);
        NodePair currEdge;
	while (edgeIter.next(currEdge)) {
	    /*
	     * Check if edge terminates with current alignment class
	     */
	    Boolean edgeEnds = false;

	    for (unsigned k = i; k < endOfAlign; k ++) {
		if (currEdge.node2 == sortedNodes[k]) {
		    edgeEnds = true;
		    break;
		}
	    }

	    if (edgeEnds) {
		//cerr << "finishing edge " << currEdge
		//     << " dels = " 
		//     << edgeErrors[currEdge.node1][currEdge.node2].dels
		//     << endl;
		pendingEdges.remove(currEdge);
	    } else {
		Prob edgePost;
		hasArc(currEdge.node1, currEdge.node2, edgePost);

		totalEdgePost += edgePost;

		edgeErrors[currEdge.node1][currEdge.node2].dels +=
								totalAlignPost;

		for (unsigned k = i; k < endOfAlign; k ++) {
		    nodeErrors[sortedNodes[k]].inss += edgePost;
		}
	    }
	}

	/*
	 * Compute substitution errors
	 * and start edges emanating from this alignment class.
	 * Also compute word with highest posterior.
	 */
	VocabIndex bestWord = Vocab_None;
	Prob bestPosterior = totalEdgePost;

	for (unsigned k = i; k < endOfAlign; k ++) {
	    unsigned currNode = sortedNodes[k];

	    if (nodes[currNode].score > bestPosterior) {
		bestWord = nodes[currNode].word;
		bestPosterior = nodes[currNode].score;
	    }

	    nodeErrors[currNode].subs += totalAlignPost - nodes[currNode].score;
	    //cerr << "node " << currNode 
	    //	 << " subs = " << nodeErrors[currNode].subs
	    //   << " inss = " << nodeErrors[currNode].inss << endl;

	    for (unsigned j = 0; j < nodes[currNode].numSuccs; j ++) {
		NodePair newEdge(currNode, nodes[currNode].succs[j]);
		pendingEdges.insert(newEdge);

		//cerr << "starting edge " << newEdge << endl;
	    }
	}

	/*
	 * Save best word for this position
	 */
	if (sortedNodes[i] == initial || sortedNodes[i] == final) {
	    ;
	} else if (bestWord == Vocab_None) {
	    totalDels += totalAlignPost;
	} else {
	    if (numWords < length) {
		words[numWords ++] = bestWord;
	    }

	    totalInss += totalEdgePost;
	    totalSubs += totalAlignPost - bestPosterior;
	}

	/*
	 * Advance to next alignment class
	 */
	i = endOfAlign;
    }

    /*
     * With the "no viterbi" option we return the sequence of most probable
     * words.
     */
    if (flags|WORDLATTICE_NOVITERBI) {
	if (numWords < length) {
	    words[numWords] = Vocab_None;
	}

	sub = totalSubs;
	ins = totalInss;
	del = totalDels;
	return totalSubs + totalInss + totalDels;
    }
	    
    /*
     * Viterbi search for path with lowest total error
     */
    Array<ChartEntry> chart(0, numNodes);

    unsigned j;
    for (j = 0; j < numNodes; j ++) {
	chart[j].errs = 0.0;
	chart[j].pred = NO_PRED;
	chart[j].nnodes = 0;
	chart[j].subs = chart[j].inss = chart[j].dels = 0.0;
    }
    
    for (j = 0; j < numReachable; j ++) {
	unsigned curr = sortedNodes[j];
	WordLatticeNode &node = nodes[curr];

	if (curr != initial && chart[curr].pred == NO_PRED) continue;

	for (unsigned s = 0; s < node.numSuccs; s ++) {
	    unsigned next = node.succs[s];

	    double newSubs = chart[curr].subs +
				  edgeErrors[curr][next].subs +
				  nodeErrors[next].subs;
	    double newInss = chart[curr].inss +
				  edgeErrors[curr][next].inss +
				  nodeErrors[next].inss;
	    double newDels = chart[curr].dels +
				  edgeErrors[curr][next].dels +
				  nodeErrors[next].dels;
	    double newErrs = newSubs + newInss + newDels;

	    if (chart[next].pred == NO_PRED || newErrs < chart[next].errs) {
		chart[next].pred = curr;
		chart[next].errs = newErrs;
		chart[next].subs = newSubs;
		chart[next].inss = newInss;
		chart[next].dels = newDels;
		chart[next].nnodes = chart[curr].nnodes + 1;
	    }
	}
    }

    /*
     * Trace back the best path
     */

    if (chart[final].pred == NO_PRED) {
	cerr << "WARNING: " << getName()
	     << ": minimizeWordError: final node not reachable\n";

	if (length > 0) {
	    words[0] = Vocab_None;
	}

	expectedError = 0.0;
    } else {

	sub = chart[final].subs;
	ins = chart[final].inss;
	del = chart[final].dels;

	unsigned curr = final;
	numWords = 0;
	while (numWords < length && curr != NO_PRED) {
	    if (nodes[curr].word != Vocab_None) {
		words[numWords ++] = nodes[curr].word;
	    }
	    curr = chart[curr].pred;
	}
	if (numWords < length) {
	    words[numWords] = Vocab_None;
	}

	if (curr != NO_PRED) {
	    cerr << "WARNING: " << getName()
		 << ": minimizeWordError: word buffer too short\n";
	}

	Vocab::reverse(words);

	expectedError = chart[final].errs;
    }
    return expectedError;
}

/*
 * wordError --
 *	compute minimal word error of path through lattice
 */
unsigned
WordLattice::wordError(const VocabIndex *words,
				    unsigned &sub, unsigned &ins, unsigned &del)
{
    unsigned numWords = Vocab::length(words);

    /*
     * The states indexing the DP chart correspond to lattice nodes.
     */
    const unsigned NO_PRED = (unsigned)(-1);	// default for pred link

    makeArray(unsigned, sortedNodes, numNodes);

    unsigned numReachable = sortNodes(sortedNodes);
    if (numReachable != numNodes) {
	cerr << "WARNING: " << getName()
	     << ": alignWords called with unreachable nodes\n";
    }

    /*
     * Allocate the DP chart.
     * chartEntries are indexed by [word_position][lattice_node],
     * where word_position = 0 is the  left string margin,
     * word_position = numWords + 1 is the right string margin.
     */
    typedef struct {
	unsigned cost;		// minimal path cost to this state
	unsigned ins, del, sub; // error counts by type
	unsigned predNode;	// predecessor state used in getting there
	ErrorType errType;	// error type
    } ChartEntry;

    ChartEntry **chart = new ChartEntry *[numWords + 2];
    assert(chart != 0);

    unsigned i;
    for (i = 0; i <= numWords + 1; i ++) {
	chart[i] = new ChartEntry[numNodes];
	assert(chart[i] != 0);
	for (unsigned j = 0; j < numNodes; j ++) {
	    chart[i][j].cost = MAX_COST;
	    chart[i][j].sub = chart[i][j].ins = chart[i][j].del = 0;
	    chart[i][j].predNode = NO_PRED;
	    chart[i][j].errType = ERR_NONE;
	}
    }

    /*
     * Prime the chart by anchoring the alignment at the left edge
     */
    chart[0][initial].cost = 0;
    chart[0][initial].ins = chart[0][initial].del = chart[0][initial].sub = 0;
    chart[0][initial].predNode = initial;
    chart[0][initial].errType = ERR_NONE;

    /*
     * Insertions before the first word
     * NOTE: since we process nodes in topological order this
     * will allow chains of multiple insertions.
     */
    for (unsigned j = 0; j < numReachable; j ++) {
	unsigned curr = sortedNodes[j];
	WordLatticeNode &node = nodes[curr];
	unsigned insCost = chart[0][curr].cost + INS_COST;

	if (insCost >= MAX_COST) continue;

	for (unsigned s = 0; s < node.numSuccs; s ++) {
	    unsigned next = node.succs[s];

	    if (insCost < chart[0][next].cost) {
		chart[0][next].cost = insCost;
		chart[0][next].ins = chart[0][curr].ins + 1;
		chart[0][next].del = chart[0][curr].del;
		chart[0][next].sub = chart[0][curr].sub;
		chart[0][next].predNode = curr;
		chart[0][next].errType = ERR_INS;
	    }
	}
    }

    /*
     * For all word positions, compute minimal cost alignment for each
     * state.
     */
    for (i = 1; i <= numWords + 1; i ++) {
	/*
	 * Compute partial alignment cost for all lattice nodes
	 */
	unsigned j;
	for (j = 0; j < numReachable; j ++) {
	    unsigned curr = sortedNodes[j];
	    WordLatticeNode &node = nodes[curr];
	    unsigned cost = chart[i - 1][curr].cost;

	    if (cost >= MAX_COST) continue;

	    /*
	     * Deletion error: current word not matched by lattice
	     */
	    {
		unsigned delCost = cost + DEL_COST;

		if (delCost < chart[i][curr].cost) {
		    chart[i][curr].cost = delCost;
		    chart[i][curr].del = chart[i - 1][curr].del + 1;
		    chart[i][curr].ins = chart[i - 1][curr].ins;
		    chart[i][curr].sub = chart[i - 1][curr].sub;
		    chart[i][curr].predNode = curr;
		    chart[i][curr].errType = ERR_DEL;
		}
	    }

	    /*
	     * Substitution errors
	     */
	    for (unsigned s = 0; s < node.numSuccs; s ++) {
		unsigned next = node.succs[s];
		unsigned haveSub =
			(nodes[next].word == words[i - 1]) ? 0 : 1;
		unsigned subCost = cost + haveSub * SUB_COST;

		if (subCost < chart[i][next].cost) {
		    chart[i][next].cost = subCost;
		    chart[i][next].sub = chart[i - 1][curr].sub + haveSub;
		    chart[i][next].ins = chart[i - 1][curr].ins;
		    chart[i][next].del = chart[i - 1][curr].del;
		    chart[i][next].predNode = curr;
		    chart[i][next].errType = haveSub ? ERR_SUB : ERR_NONE;
		}
	    }
	}

	for (j = 0; j < numReachable; j ++) {
	    unsigned curr = sortedNodes[j];
	    WordLatticeNode &node = nodes[curr];
	    unsigned insCost = chart[i][curr].cost + INS_COST;

	    if (insCost >= MAX_COST) continue;

	    /*
	     * Insertion errors: lattice node not matched by word
	     * NOTE: since we process nodes in topological order this
	     * will allow chains of multiple insertions.
	     */
	    for (unsigned s = 0; s < node.numSuccs; s ++) {
		unsigned next = node.succs[s];

		if (insCost < chart[i][next].cost) {
		    chart[i][next].cost = insCost;
		    chart[i][next].ins = chart[i][curr].ins + 1;
		    chart[i][next].del = chart[i][curr].del;
		    chart[i][next].sub = chart[i][curr].sub;
		    chart[i][next].predNode = curr;
		    chart[i][next].errType = ERR_INS;
		}
	    }
	}
    }

    if (chart[numWords + 1][final].predNode == NO_PRED) {
	/*
	 * Final node is unreachable
	 */
	sub = ins = 0;
	del = numWords;
    } else {
	sub = chart[numWords + 1][final].sub;
	ins = chart[numWords + 1][final].ins;
	del = chart[numWords + 1][final].del;
    }

    for (i = 0; i <= numWords + 1; i ++) {
	delete [] chart[i];
    }
    delete [] chart;

    return sub + ins + del;
}

