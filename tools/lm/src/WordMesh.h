/*
 * WordMesh.h --
 *	Word Meshes (a simple type of word lattice with transitions between
 *	any two adjacent words).
 *
 * Copyright (c) 1998-2012 SRI International, 2012 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/WordMesh.h,v 1.29 2014-10-15 21:09:21 bartels Exp $
 *
 */

#ifndef _WordMesh_h_
#define _WordMesh_h_

#include <vector>

#include "MultiAlign.h"
#include "VocabDistance.h"

#include "Array.h"
#include "LHash.h"
#include "SArray.h"

class WordMeshIter;

class WordMesh: public MultiAlign
{
    friend class WordMeshIter;

public:
    WordMesh(Vocab &vocab, const char *myname = 0, VocabDistance *distance = 0);
    ~WordMesh();

    Boolean read(File &file);
    Boolean write(File &file);

    void alignWords(const VocabIndex *words, Prob score,
				Prob *wordScores = 0, const HypID *hypID = 0);
    void alignWords(const NBestWordInfo *winfo, Prob score,
				Prob *wordScores = 0, const HypID *hypID = 0)
	{ alignWords(winfo, score, wordScores, hypID,
		     numAligns, numAligns, (unsigned *)0);
	};
    // Note: returns success/failure
    Boolean alignWords(const NBestWordInfo *winfo, Prob score,
				Prob *wordScores, const HypID *hypID,
				unsigned from, unsigned to,
				unsigned *wordAlignment);
		    
    void alignAlignment(MultiAlign &alignment, Prob score,
							Prob *alignScores = 0);

    // only obtain slot-slot alignment (structure of WCN is unchanged)
    void alignAlignment(MultiAlign &other_alignment, std::vector<int>& src2other_col_map);

    void normalizeDeletes();

    unsigned wordError(const VocabIndex *words,
				unsigned &sub, unsigned &ins, unsigned &del);

    double minimizeWordError(VocabIndex *words, unsigned length,
				double &sub, double &ins, double &del,
				unsigned flags = 0, double delBias = 1.0);
    double minimizeWordError(NBestWordInfo *winfo, unsigned length,
				double &sub, double &ins, double &del,
				unsigned flags = 0, double delBias = 1.0);

    Boolean isEmpty();
    unsigned length() { return numAligns; };
    LHash<VocabIndex,Prob> *wordColumn(unsigned columnNumber);
    LHash<VocabIndex,NBestWordInfo> *wordinfoColumn(unsigned columnNumber);
    NBestWordInfo* wordInfoFromUnsortedColumn(unsigned unsortedColumnIndex, VocabIndex word);  
    Prob wordProbFromUnsortedColumn(unsigned unsortedColumnIndex, VocabIndex word);

    static void freeThread();
    
    Prob totalPosterior;		// accumulated sample scores

    VocabIndex deleteIndex;		// pseudo-word representing deletions

    double alignError(const LHash<VocabIndex,Prob> *column,
		      Prob columnPosterior,
		      VocabIndex word);
					// error from aligning word to column
    double alignError(const LHash<VocabIndex,Prob> *column1,
		      Prob columnPosterior,
		      const LHash<VocabIndex,Prob> *column2,
		      Prob columnPosterior2 = 1.0);
					// error from aligning two columns

private:

    Array< LHash<VocabIndex,Prob>* > aligns;	// alignment columns
    Array< LHash<VocabIndex,NBestWordInfo>* > wordInfo;
					// word backtrace info
    Array< LHash<VocabIndex,Array<HypID> >* > hypMap;
					// pointers from word hyps
					//       to sentence hyps
    Array< Prob > columnPosteriors;	// sum of posteriors by column
    Array< Prob > transPosteriors;	// sum of posteriors from column to next
    SArray<HypID,HypID> allHyps;	// list of all aligned hyp IDs
					// 	(Note: only keys are used)
    unsigned numAligns;			// number of alignment columns
    Array<unsigned> sortedAligns;	// topoligical order of alignment

    VocabDistance *distance;		// word distance (or null)
};

/*
 * Enumeration of words in alignment and their associated hypMaps
 */
class WordMeshIter
{
public:
    WordMeshIter(WordMesh &mesh, unsigned position)
       : myIter(*mesh.hypMap[mesh.sortedAligns[position]]) {};

    void init()
	{ myIter.init(); };
    Array<HypID> *next(VocabIndex &word)
	{ return myIter.next(word); };

private:
    LHashIter<VocabIndex, Array<HypID> > myIter;
};

#endif /* _WordMesh_h_ */

