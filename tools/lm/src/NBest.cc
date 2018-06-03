/*
 * NBest.cc --
 *	N-best hypotheses and lists
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2012 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NBest.cc,v 1.95 2016/06/17 00:11:06 victor Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "NBest.h"
#include "WordAlign.h"
#include "Bleu.h"
#include "MultiwordVocab.h"	// for MultiwordSeparator
#include "TLSWrapper.h"

#include "Array.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_ARRAY(NBestHyp);
#endif

#define DEBUG_PRINT_RANK	1

const char *phoneSeparator = ":";	 // used for phones & phoneDurs strings
const NBestTimestamp frameLength = 0.01f; // quantization unit of word timemarks

/*
 * N-best word backtrace information
 */

const unsigned phoneStringLength = 100;

NBestWordInfo::NBestWordInfo()
    : word(Vocab_None), confidenceScore(LogP_Inf), confidenceScore2(LogP_Inf), confidenceScore3(LogP_Inf), phones(0), phoneDurs(0),
      wordPosterior(0.0), transPosterior(0.0)
{
    // make acoustic info invalid by default
    invalidate();
}

NBestWordInfo::~NBestWordInfo()
{
    if (phones) free(phones);
    if (phoneDurs) free(phoneDurs);
}

NBestWordInfo &
NBestWordInfo::operator= (const NBestWordInfo &other)
{
    if (&other == this) {
	return *this;
    }

    if (phones) free(phones);
    if (phoneDurs) free(phoneDurs);

    word = other.word;
    start = other.start;
    duration = other.duration;
    acousticScore = other.acousticScore;
    languageScore = other.languageScore;

    if (!other.phones) {
	phones = 0;
    } else {
	phones = strdup(other.phones);
	assert(phones != 0);
    }

    if (!other.phoneDurs) {
	phoneDurs = 0;
    } else {
	phoneDurs = strdup(other.phoneDurs);
	assert(phoneDurs != 0);
    }

    wordPosterior = other.wordPosterior;
    transPosterior = other.transPosterior;

    return *this;
}

void
NBestWordInfo::write(File &file)
{
    file.fprintf("%lg %lg %.*lg %.*lg %s %s",
			(double)start, (double)duration,
			LogP_Precision, (double)acousticScore,
			LogP_Precision, (float)languageScore,
			phones ? phones : phoneSeparator,
			phoneDurs ? phoneDurs : phoneSeparator);

    if (confidenceScore != LogP_Inf) {
        file.fprintf(" %.*lg", LogP_Precision, (double)confidenceScore);
    }
    if (confidenceScore2 != LogP_Inf) {
        file.fprintf(" %.*lg", LogP_Precision, (double)confidenceScore2);
    }
    if (confidenceScore3 != LogP_Inf) {
        file.fprintf(" %.*lg", LogP_Precision, (double)confidenceScore3);
    }
}

Boolean
NBestWordInfo::parse(const char *s)
{
    double sTime, dur, aScore, lScore, cScore, cScore2, cScore3;
    char phs[phoneStringLength], phDurs[phoneStringLength];
    int nRet = 0;

    // Handle differently depending on how many confidence values can be found
    // confidence fields are optional
    nRet = sscanf(s, "%lg %lg %lg %lg %100s %100s %lg %lg %lg",
		  &sTime, &dur, &aScore, &lScore, phs, phDurs, &cScore, &cScore2, &cScore3);
    if (nRet == 9) 
    {
        confidenceScore3 = cScore3;
        confidenceScore2 = cScore2;
        confidenceScore = cScore;
    }
    else if (nRet == 8)
    {
        confidenceScore2 = cScore2;
        confidenceScore = cScore;
    }
    else if (nRet == 7)
    {
        confidenceScore = cScore;
    }
    else 
    {
        // required fields
        if (sscanf(s, "%lg %lg %lg %lg %100s %100s",
		   &sTime, &dur, &aScore, &lScore, phs, phDurs) != 6)
        {
            return false;
        } 
    }

    start = sTime;
    duration = dur;
    acousticScore = aScore;
    languageScore = lScore;

    if (strcmp(phs, phoneSeparator) == 0) {
        phones = 0;
    } else {
        phones = strdup(phs);
        assert(phones != 0);
    }

    if (strcmp(phDurs, phoneSeparator) == 0) {
        phoneDurs = 0;
    } else {
        phoneDurs = strdup(phDurs);
        assert(phoneDurs != 0);
    }

    return true;
}

void
NBestWordInfo::invalidate()
{
    duration = (NBestTimestamp)HUGE_VAL;
}

Boolean
NBestWordInfo::valid() const
{
    return (duration != (NBestTimestamp)HUGE_VAL);
}

void
NBestWordInfo::merge(const NBestWordInfo &other, Prob otherPosterior)
{
    /* 
     * Optional argument lets caller override word posterior
     */
    if (otherPosterior == 0.0) {
	otherPosterior = other.wordPosterior;
    }

    /*
     * let the "other" word information supercede our own if it has
     * higher word-level posterior probability.
     */
    if (otherPosterior > wordPosterior)
    {
	*this = other;
	wordPosterior = otherPosterior;
    }
}

/*
 * Utility functions for arrays of NBestWordInfo
 *	(analogous to VocabIndex * functions)
 */

unsigned 
NBestWordInfo::length(const NBestWordInfo *words)
{
    unsigned len = 0;

    while (words[len].word != Vocab_None) len++;
    return len;
}

NBestWordInfo *
NBestWordInfo::copy(NBestWordInfo *to, const NBestWordInfo *from)
{
    unsigned i;
    for (i = 0; from[i].word != Vocab_None; i ++) {
        to[i] = from[i];
    }
    to[i] = from[i];

    return to;
}

/*
 * Extract VocabIndex string from NBestWordInfo array
 */
VocabIndex *
NBestWordInfo::copy(VocabIndex *to, const NBestWordInfo *from)
{
    unsigned i;
    for (i = 0; from[i].word != Vocab_None; i ++) {
	to[i] = from[i].word;
    }
    to[i] = Vocab_None;

    return to;
}

size_t
LHash_hashKey(const NBestWordInfo *key, unsigned maxBits)
{
    unsigned i = 0;

    /*
     * Incorporate start time into hash key
     */
    if (key[0].valid()) {
    	i = (unsigned)key[0].start;
    }

    /*
     * The rationale here is similar to LHash_hashKey(unsigned),
     * except that we shift more to preserve more of the typical number of
     * bits in a VocabIndex.  The value was optimized to encoding 3 words
     * at a time (trigrams).
     */
    for (; key->word != Vocab_None; key ++) {
	i += (i << 12) + key->word;
    }
    return LHash_hashKey(i, maxBits);
}

const NBestWordInfo *
Map_copyKey(const NBestWordInfo *key)
{
    NBestWordInfo *copy = new NBestWordInfo[NBestWordInfo::length(key) + 1];
    assert(copy != 0);

    unsigned i;
    for (i = 0; key[i].word != Vocab_None; i ++) {
	copy[i] = key[i];
    }
    copy[i] = key[i];

    return copy;
}

void
Map_freeKey(const NBestWordInfo *key)
{
    delete [] (NBestWordInfo *)key;
}

/*
 * Two ngram keys are different if
 *	- their lengths differ, or
 *	- their start or end times (if defined) differ, or
 *	- their words differ
 */
Boolean
LHash_equalKey(const NBestWordInfo *key1, const NBestWordInfo *key2)
{
    unsigned len1 = NBestWordInfo::length(key1);
    unsigned len2 = NBestWordInfo::length(key2);

    if (len1 > 0 && len2 > 0) {
	if (len1 != len2 ||
	    key1[0].valid() != key2[0].valid() ||
	    (key1[0].valid() && key1[0].start != key2[0].start) ||
	    key1[len1-1].valid() != key2[len1-1].valid() ||
	    (key1[len1-1].valid() && key1[len1-1].start+key1[len1-1].duration !=
				     key2[len1-1].start+key2[len1-1].duration))
	{
	    return false;
	}
    }

    for (unsigned i = 0; i < len1; i ++) {
	if (key1[i].word != key2[i].word) {
	    return false;
	}
    }

    return true;
}

static inline int
sign(NBestTimestamp x)
{
    if (x < 0.0) {
    	return -1;
    } else if (x > 0) {
    	return 1;
    } else {
    	return 0;
    }
}

/*
 * Two ngram keys are sorted
 *	- first, by their start times
 *	- second, by their end times
 *	- last, by their words
 */
int
SArray_compareKey(const NBestWordInfo *key1, const NBestWordInfo *key2)
{
    unsigned len1 = NBestWordInfo::length(key1);
    unsigned len2 = NBestWordInfo::length(key2);

    if (len1 > 0 && len2 > 0) {
	if (key1[0].valid() != key2[0].valid()) {
	    return key1[0].valid() ? -1 : 1;
	}

	if (key1[len1-1].valid() != key2[len2-1].valid()) {
	    return key1[len1-1].valid() ? -1 : 1;
	}

	if (key1[0].valid()) {
	    NBestTimestamp diff = key1[0].start - key2[0].start;

	    if (diff != 0) {
		return sign(diff);	/* start times differ */
	    }
	}

	if (key1[len1-1].valid()) {
	    NBestTimestamp diff = (key1[len1-1].start + key1[len1-1].duration) -
	    			  (key2[len2-1].start + key2[len2-1].duration);
   
	    if (diff != 0) {
		return sign(diff);	/* end times differ */
	    }
	}
    }

    for (unsigned i = 0; ; i++) {
	if (key1[i].word == Vocab_None) {
	    if (key2[i].word == Vocab_None) {
		return 0;
	    } else {
		return -1;      /* key1 is shorter */
	    }
	} else {
	    if (key2[i].word == Vocab_None) {
		return 1;       /* key2 is shorter */
	    } else {
		int comp = SArray_compareKey(key1[i].word, key2[i].word);
		if (comp != 0) {
		    return comp;        /* words differ at pos i */
		}
	    }
	}
    }
    /*NOTREACHED*/
}


/*
 * N-Best hypotheses
 */

NBestHyp::NBestHyp()
{
    words = 0;
    wordInfo = 0;
    acousticScore = languageScore = totalScore = 0.0;
    posterior = 0.0;
    numWords = 0;
    numErrors = 0.0;
    rank = 0;
    bleuCount = 0;
    closestRefLeng = 0;
}

NBestHyp::~NBestHyp()
{
    delete [] words;
    delete [] wordInfo;
    delete bleuCount;
}

NBestHyp &
NBestHyp::operator= (const NBestHyp &other)
{
    // cerr << "warning: NBestHyp::operator= called\n";

    if (&other == this) {
	return *this;
    }

    delete [] words;
    delete [] wordInfo;
    delete bleuCount;

    acousticScore = other.acousticScore;
    languageScore = other.languageScore;
    totalScore = other.totalScore;

    numWords = other.numWords;
    posterior = other.posterior;
    numErrors = other.numErrors;
    rank = other.rank;
    
    if (other.bleuCount) {      
        bleuCount = new BleuCount;
        *bleuCount = *other.bleuCount;
    } else
        bleuCount = 0;

    closestRefLeng = other.closestRefLeng;

    if (other.words) {
	unsigned actualNumWords = Vocab::length(other.words) + 1;

	words = new VocabIndex[actualNumWords];
	assert(words != 0);

	for (unsigned i = 0; i < actualNumWords; i++) {
	    words[i] = other.words[i];
	}

	if (other.wordInfo) {
	    wordInfo = new NBestWordInfo[actualNumWords];
	    assert(wordInfo != 0);

	    for (unsigned i = 0; i < actualNumWords; i++) {
		wordInfo[i] = other.wordInfo[i];
	    }
	} else {
	    wordInfo = 0;
	}

    } else {
	words = 0;
	wordInfo = 0;
    }

    return *this;
}

/*
 * N-Best Hypotheses
 */

static Boolean
addPhones(char *old, const char *ph, Boolean reversed = false) 
{
    unsigned oldLen = strlen(old);
    unsigned newLen = strlen(ph);

    if (oldLen + 1 + newLen + 1 > phoneStringLength) {
	return false;
    } else if (reversed) {
	if (oldLen > 0) {
	    memmove(&old[newLen + 1], old, oldLen + 1);
	}
	strcpy(old, ph);
	if (oldLen > 0) {
	    old[newLen] = phoneSeparator[0];
	}
    } else {
	if (oldLen > 0) {
	    old[oldLen ++] = phoneSeparator[0];
	}
	strcpy(&old[oldLen], ph);
    }

    return true;
}

/* NBestList2.0 format uses 11 fields per word */
static const unsigned maxFieldsPerLine = 11 * maxWordsPerLine + 4;
static TLSW_ARRAY(VocabString, wstringsTLS, maxFieldsPerLine);
static TLSW_ARRAY(VocabString, justWordsTLS, maxFieldsPerLine + 1);

Boolean
NBestHyp::parse(char *line, Vocab &vocab, unsigned decipherFormat,
		LogP2 acousticOffset, const char *multiChar, Boolean backtrace)
{
    VocabString *wstrings  = TLSW_GET_ARRAY(wstringsTLS);
    VocabString *justWords = TLSW_GET_ARRAY(justWordsTLS);

    Array<NBestWordInfo> backtraceInfo;

    unsigned actualNumWords =
		Vocab::parseWords(line, wstrings, maxFieldsPerLine);

    if (actualNumWords == maxFieldsPerLine) {
	cerr << "more than " << maxFieldsPerLine << " fields per line\n";
	return false;
    }

    /*
     * We don't do multiword splitting with backtraces -- that would require
     * a dictionary (see external split-nbest-words script).
     */
    if (backtrace) {
	multiChar = 0;
    }

    /*
     * We accept three formats for N-best hyps.
     * - The Decipher NBestList1.0 format, which has one combined bytelog score
     *	 in parens preceding the hyp.
     * - The Decipher NBestList2.0 format, where each word is followed by
     *	  ( st: <starttime> et: <endtime> g: <grammar_score> a: <ac_score> )
     * - Our own format, which has acoustic score, LM score, and number of
     *   words followed by the hyp.
     * If (decipherFormat > 0) only the specified Decipher format is accepted.
     */

    if (decipherFormat == 1 || 
	(decipherFormat == 0 && wstrings[0][0] == '('))
    {
	/*
	 * These formats don't support backtrace info
	 */
	backtrace = false;

	actualNumWords --;

	if (actualNumWords > maxWordsPerLine) {
	    cerr << "more than " << maxWordsPerLine << " words in hyp\n";
	    return false;
	}

	/*
	 * Parse the first word as a score (in parens)
	 */
	double score;

	if (sscanf(wstrings[0], "(%lf)", &score) != 1)
	{
	    cerr << "bad Decipher score: " << wstrings[0] << endl;
	    return false;
	}

	/*
	 * Save score
	 */
	totalScore = acousticScore = BytelogToLogP(score);
	languageScore = 0.0;

	/* 
	 * Note: numWords includes pauses, consistent with the way the 
	 * recognizer applies word transition weights.  Elimination of pauses
	 * is the job of LM rescoring.
	 */
	numWords = actualNumWords;

	Vocab::copy(justWords, &wstrings[1]);

    } else if (decipherFormat == 2) {
	if ((actualNumWords - 1) % 11) {
	    cerr << "badly formatted hyp\n";
	    return false;
	}

	unsigned numTokens = (actualNumWords - 1)/11;

	if (numTokens > maxWordsPerLine) {
	    cerr << "more than " << maxWordsPerLine << " tokens in hyp\n";
	    return false;
	}

	/*
	 * Parse the first word as a score (in parens)
	 */
	double score;

	if (sscanf(wstrings[0], "(%lf)", &score) != 1)
	{
	    cerr << "bad Decipher score: " << wstrings[0] << endl;
	    return false;
	}

	/*
	 * Parse remaining line into words and scores
	 *	skip over phone and state backtrace tokens, which can be
	 *	identified by noting that their times are contained within
	 *	the word duration.
	 */
	Bytelog acScore = 0;
	Bytelog lmScore = 0;

	NBestTimestamp prevEndTime = -1.0;	/* end time of last token */
	NBestTimestamp prevPhoneStart = 0.0;
	NBestWordInfo *prevWordInfo = 0;

	char phones[phoneStringLength];
	char phoneDurs[phoneStringLength];

	actualNumWords = 0;
	for (unsigned i = 0; i < numTokens; i ++) {

	    const char *token = wstrings[1 + 11 * i];
	    NBestTimestamp startTime = atof(wstrings[1 + 11 * i + 3]);
	    NBestTimestamp endTime = atof(wstrings[1 + 11 * i + 5]);

	    /*
	     * Check if this token refers to an HMM state, i.e., if
	     * it matches the pattern /-[0-9]$/.
	     * XXX: because of a bug in Decipher we need to perform this
	     * check even if we're scanning for word tokens.
	     */
	    const char *hyphen = strrchr(token, '-');
	    Boolean isStateToken = hyphen != 0 &&
				hyphen[1] >= '0' && hyphen[1] <= '9' &&
				hyphen[2] == '\0';

	    if (startTime > prevEndTime && !isStateToken) {
		int acWordScore = atol(wstrings[1 + 11 * i + 9]);
		int lmWordScore = atol(wstrings[1 + 11 * i + 7]);

		justWords[actualNumWords] = token;

		if (backtrace) {
		    /*
		     * save pronunciation info for previous word
		     */
		    if (prevWordInfo && phones[0] != '\0') {
			prevWordInfo->phones = strdup(phones);
			assert(prevWordInfo->phones != 0);

			prevWordInfo->phoneDurs = strdup(phoneDurs);
			assert(prevWordInfo->phoneDurs != 0);
		    }

		    NBestWordInfo winfo;
		    winfo.word = Vocab_None;
		    winfo.start = startTime;
		    /*
		     * NB: "et" in nbest backtrace is actually the START time
		     * of the last frame
		     */
		    winfo.duration = endTime - startTime + frameLength;
		    winfo.acousticScore = BytelogToLogP(acWordScore);
		    winfo.languageScore = BytelogToLogP(lmWordScore);
		    winfo.phones = winfo.phoneDurs = 0;

		    backtraceInfo[actualNumWords] = winfo;

		    /*
		     * prepare for collecting phone backtrace info
		     */
		    prevWordInfo = &backtraceInfo[actualNumWords];
		    phones[0] = phoneDurs[0] = '\0';
		}

		acScore += acWordScore;
		lmScore += lmWordScore;

		actualNumWords ++;

		prevEndTime = endTime;
	    } else {
		/*
		 * check if this token refers to an HMM state, i.e., if
		 * if matches the pattern /-[0-9]$/
		 */
		if (isStateToken) {
		    continue;
		}

		/*
		 * A Decipher phone token: if we're extracting backtrace
		 * information, get phone identity and duration and store
		 * in word Info.
		 * The format of Decipher context-dep phone token is:
		 *    leftcontext '[' phonelabel '_' diacritic ']' rightcontext
		 * Everything except the phonelabel is optional.
		 */
		if (prevWordInfo) {
		    const char *lbracket = strchr(token, '[');
		    const char *phone = lbracket ? lbracket + 1 : token;
		    char *rbracket = (char *)strrchr(phone, ']');
		    if (rbracket) *rbracket = '\0';
		    char *subscript = (char *)strrchr(phone, '_');
		    if (subscript) *subscript = '\0';
		    addPhones(phones, phone, startTime < prevPhoneStart);

		    char phoneDur[20];
		    sprintf(phoneDur, "%d",
			    (int)((endTime - startTime)/frameLength + 0.5) + 1);
		    addPhones(phoneDurs, phoneDur, startTime < prevPhoneStart);

		    prevPhoneStart = startTime;
		}
	    }
	}

	if (backtrace) {
	    /*
	     * save pronunciation info for last word
	     */
	    if (prevWordInfo && phones[0] != '\0') {
		prevWordInfo->phones = strdup(phones);
		assert(prevWordInfo->phones != 0);

		prevWordInfo->phoneDurs = strdup(phoneDurs);
		assert(prevWordInfo->phoneDurs != 0);
	    }
	}

	justWords[actualNumWords] = 0;

	/*
	 * Save scores
	 */
	totalScore = BytelogToLogP(score);
	acousticScore = BytelogToLogP(acScore);
	languageScore = BytelogToLogP(lmScore);
	numWords = actualNumWords;

	/*
	if (score != acScore + lmScore) {
	    cerr << "acoustic and language model scores don't add up ("
		 << acScore << " + " << lmScore << " != " << score << ")\n";
	}
	*/

    } else {
	actualNumWords -= 3;

	if (actualNumWords > maxWordsPerLine) {
	    cerr << "more than " << maxWordsPerLine << " words in hyp\n";
	    return false;
	}

	/*
	 * Parse the first three columns as numbers
	 */
	if (!parseLogP(wstrings[0], acousticScore)) {
	    cerr << "bad acoustic score: " << wstrings[0] << endl;
	    return false;
	}
	if (!parseLogP(wstrings[1], languageScore)) {
	    cerr << "bad LM score: " << wstrings[1] << endl;
	    return false;
	}
	if (!stringToCount(wstrings[2], numWords)) {
	    cerr << "bad word count: " << wstrings[2] << endl;
	    return false;
	}

	/*
	 * Set the total score to the acoustic score so 
	 * decipherFix() with a null language model leaves everything
	 * unchanged.
	 */
	totalScore = acousticScore;

	Vocab::copy(justWords, &wstrings[3]);
    }

    /*
     * Apply acoustic normalization in effect
     */
    acousticScore -= acousticOffset;
    totalScore -= acousticOffset;

    /*
     * Adjust number of words for multiwords if appropriate
     */
    if (multiChar) {
	for (unsigned j = 0; justWords[j] != 0; j ++) {
	    const char *cp = justWords[j];

	    for (cp = strchr(cp, *multiChar);
		 cp != 0;
		 cp = strchr(cp + 1, *multiChar))
	    {
		actualNumWords ++;
	    }
	}
    }

    /*
     * Copy words to nbest list
     */
    delete [] words;
    words = new VocabIndex[actualNumWords + 1];
    assert(words != 0);

    Boolean unkIsWord = vocab.unkIsWord();

    /*
     * Map word strings to indices
     */
    if (!multiChar) {
	if (unkIsWord) {
	    vocab.getIndices(justWords, words, actualNumWords + 1,
							    vocab.unkIndex());
	} else {
	    vocab.addWords(justWords, words, actualNumWords + 1);
	}

	if (decipherFormat == 2 && backtrace) {
	    delete [] wordInfo;
	    wordInfo = new NBestWordInfo[actualNumWords + 1];

	    for (unsigned j = 0; j < actualNumWords; j ++) {
		wordInfo[j] = backtraceInfo[j];
		wordInfo[j].word = words[j];
	    }
	    wordInfo[actualNumWords].word = Vocab_None;
	} else {
	    wordInfo = 0;
	}
    } else {
	unsigned i = 0;
	for (unsigned j = 0; justWords[j] != 0; j ++) {
	    char *start = (char *)justWords[j];
	    char *cp;

	    while ((cp = strchr(start, *multiChar))) {
		*cp = '\0';
		words[i++] =
		    unkIsWord ? vocab.getIndex(start, vocab.unkIndex())
			      : vocab.addWord(start);
		*cp = *multiChar;
		start = cp + 1;
	    }

	    words[i++] =
		unkIsWord ? vocab.getIndex(start, vocab.unkIndex())
			  : vocab.addWord(start);
	}
	words[i] = Vocab_None;
    }

    return true;
}

void
NBestHyp::write(File &file, Vocab &vocab, Boolean decipherFormat,
						LogP2 acousticOffset)
{
    if (decipherFormat) {
	file.fprintf("(%d)", (int)LogPtoBytelog(totalScore + acousticOffset));
    } else {
	file.fprintf("%.15lg %.15lg %lu", (double)(acousticScore + acousticOffset),
				      (double)languageScore, (unsigned long)numWords);
    }

    for (unsigned i = 0; words[i] != Vocab_None; i++) {
	/*
	 * Write backtrace information if known and Decipher format is desired
	 */
	if (decipherFormat && wordInfo) {
	    file.fprintf(" %s ( st: %.2f et: %.2f g: %d a: %d )", 
			   vocab.getWord(wordInfo[i].word),
			   wordInfo[i].start,
			   wordInfo[i].start+wordInfo[i].duration - frameLength,
			   (int)LogPtoBytelog(wordInfo[i].languageScore),
			   (int)LogPtoBytelog(wordInfo[i].acousticScore));
	} else {
	    file.fprintf(" %s", vocab.getWord(words[i]));
	}
    }

    file.fprintf("\n");
}

// SRInterp format has scores and words in the same line. Scores are in the form of 
// key=val. The first field is always "pr=xxx". Some scores might have non-digit values. 
// This function extracts pre-registered scores and put it in the hash table.
static TLSW(Boolean, firstTimeSentStartFlagTLS);
Boolean
NBestHyp::parseSRInterpFormat(char * line, Vocab &vocab, LHash<VocabString, LogP>& scores)
{
  const char * location = "NBestHyp::parseSRInterpFormat";
  
  bool &firstTimeSentStartFlag = TLSW_GET(firstTimeSentStartFlagTLS);

  const unsigned maxFieldLength = 1024;
  const unsigned maxNameLength = 10;
  const unsigned minNameLength = 1;

  unsigned numScores = 0;

  char field[maxFieldLength];
  char *p;
  unsigned pos = 0;
  int newPos;
  // @kw false positive: SV.TAINTED.INDEX_ACCESS (pos)
  while(sscanf(line + pos, "%1023s%n", field, &newPos) == 1) {
    char * eqSign = strchr(field, '=');
    if (!eqSign) {
      
        if (strcmp(field, Vocab_SentStart) == 0) {
	    if (firstTimeSentStartFlag) {
	        firstTimeSentStartFlag = false;
		cerr << location << ": will strip <s> and </s> from hyps" << endl;
	    }
	
	    pos += newPos;
	    p = strstr(line + pos, Vocab_SentEnd);
	    if (p) {
	        *p = '\0';
	    } else {
	        cerr << location << ": has <s> but not </s> in hyp" << endl;
	    }
	}
	
	break;
    }
    
    if (eqSign > field + maxNameLength || // key is too long
	eqSign < field + minNameLength || // key is too short
	isspace(eqSign[1])) { // val is empty
      
      cerr << location << ": warning: \"" << field << "\" is treated as word instead of field" << endl;
      break;
    } 

    *eqSign = '\0';
    Boolean foundP;
    LogP *pScore = scores.find(field, foundP);
    if (foundP) {
      *pScore = (LogP) atof(eqSign + 1);
      numScores++;
    }

    pos += newPos;
  }
  
  if (numScores < scores.numEntries()) {

    cerr << "read " << numScores << " scores ; fewer than expected (" 
	 << scores.numEntries() << ")" << endl;

    return false;
  }

  assert(isspace(line[pos]) && pos > 3);
  
  // use fake decipher format   
  line[pos - 3] = '(';
  line[pos - 2] = '0';
  line[pos - 1] = ')';

  return parse(line + pos - 3, vocab, 1); 
  
}

void 
NBestHyp::freeThread() {
    TLSW_FREE(wstringsTLS);
    TLSW_FREE(justWordsTLS);
    TLSW_FREE(firstTimeSentStartFlagTLS);  
}

void
NBestHyp::rescore(LM &lm, double lmScale, double wtScale)
{
    TextStats stats;

    /*
     * LM score is recomputed,
     * numWords is set to take non-word tokens into account
     */
    languageScore = weightLogP(lmScale, lm.sentenceProb(words, stats));
    numWords = (Count)stats.numWords;

    /*
     * Note: In the face of zero probaility words we do NOT
     * set the LM probability to zero.  These cases typically
     * reflect a vocabulary mismatch between the rescoring LM
     * and the recognizer, and it is more useful to rescore based on
     * the known words alone.  The warning hopefull will cause
     * someone to asssess the problem.
     */
    if (stats.zeroProbs > 0) {
	cerr << "warning: hyp contains zero prob words: "
	     << (lm.vocab.use(), words) << endl;
    }

    if (stats.numOOVs > 0) {
	cerr << "warning: hyp contains OOV words: "
	     << (lm.vocab.use(), words) << endl;
    }

    totalScore = acousticScore +
			    languageScore +
			    wtScale * numWords;
}

void
NBestHyp::reweight(double lmScale, double wtScale, double amScale)
{
    totalScore = weightLogP(amScale, acousticScore) +
			    weightLogP(lmScale, languageScore) +
			    wtScale * numWords;
}

void
NBestHyp::decipherFix(LM &lm, double lmScale, double wtScale)
{
    TextStats stats;

    /*
     * LM score is recomputed,
     * numWords is set to take non-word tokens into account
     */
    languageScore = weightLogP(lmScale, lm.sentenceProb(words, stats));
    numWords = (Count)stats.numWords;

    /*
     * Arguably a bug, but Decipher actually applies WTW to pauses.
     * So we have to do the same when subtracting the non-acoustic
     * scores below.
     */
    unsigned numAllWords = Vocab::length(words);

    if (stats.zeroProbs > 0) {
	cerr << "warning: hyp contains zero prob words: "
	     << (lm.vocab.use(), words) << endl;
	languageScore = LogP_Zero;
    }

    if (stats.numOOVs > 0) {
	cerr << "warning: hyp contains OOV words: "
	     << (lm.vocab.use(), words) << endl;
	languageScore = LogP_Zero;
    }

    acousticScore = totalScore -
			    languageScore -
			    wtScale * numAllWords;
}


/*
 * N-Best lists
 */

const unsigned NBestList::initialSize = 100;

NBestList::NBestList(Vocab &vocab, unsigned maxSize,
				    Boolean multiwords, Boolean backtrace)
    : vocab(vocab), acousticOffset(0.0),
      hypList(0, initialSize), _numHyps(0), maxSize(maxSize),
      multiChar(0), backtrace(backtrace)
{
    if (multiwords) {
    	multiChar = MultiwordSeparator;
    }
}

// enable multiwords if multiChar != 0
NBestList::NBestList(Vocab &vocab, unsigned maxSize,
				const char *multiChar, Boolean backtrace)
    : vocab(vocab), acousticOffset(0.0),
      hypList(0, initialSize), _numHyps(0), maxSize(maxSize),
      multiChar(multiChar), backtrace(backtrace)
{
}

/* 
 * Compute memory usage
 */
void
NBestList::memStats(MemStats &stats)
{
    stats.total += sizeof(*this) - sizeof(hypList);
    hypList.memStats(stats);

    /*
     * Add space taken up by hyp strings
     */
    for (unsigned h = 0; h < _numHyps; h++) {
	unsigned numWords = Vocab::length(hypList[h].words);
	stats.total += (numWords + 1) * sizeof(VocabIndex);
	if (hypList[h].wordInfo) {
	    stats.total += (numWords + 1) * sizeof(NBestWordInfo);
	}
    }
}

static int
compareHyps(const void *h1, const void *h2)
{
    LogP score1 = ((NBestHyp *)h1)->totalScore;
    LogP score2 = ((NBestHyp *)h2)->totalScore;
    
    return score1 > score2 ? -1 :
		score1 < score2 ? 1 : 0;
}

void
NBestList::sortHyps()
{
    /*
     * Sort the underlying array in place, in order of descending scores
     */
    qsort(hypList.data(), _numHyps, sizeof(NBestHyp), compareHyps);
}

void
NBestList::sortHypsBySentenceBleu(unsigned order)
{
    // compute sentence bleu
    unsigned total [MAX_BLEU_NGRAM];
    unsigned correct [MAX_BLEU_NGRAM];
    
    for (unsigned i = 0; i < _numHyps; i++) {
        NBestHyp & h = hypList[i];

        int t = h.numWords;
        for (unsigned k = 0; k < order; k++) {
            total[k] = t;
            correct[k] = h.bleuCount->correct[k];
            if (t > 0) t --;
        }

        double bleu = computeBleu(order, correct, total, h.numWords, h.closestRefLeng);
        h.totalScore = bleu;
    }
    
    sortHyps();
}

float
NBestList::sortHypsByErrorRate()
{
    if (_numHyps == 0) return 0;
  
    for (unsigned i = 0; i < _numHyps; i++) {
        NBestHyp & h = hypList[i];
        h.totalScore = -h.numErrors;
    }
    
    sortHyps();
    return hypList[0].numErrors;  
}


Boolean
NBestList::read(File &file)
{
    char *line = file.getline();
    unsigned decipherFormat = 0;

    /*
     * If the first line contains the Decipher magic string
     * we enforce Decipher format for the entire N-best list.
     */
    if (line != 0) {
	if (strncmp(line, nbest1Magic, sizeof(nbest1Magic) - 1) == 0) {
	    decipherFormat = 1;
	    line = file.getline();
	} else if (strncmp(line, nbest2Magic, sizeof(nbest2Magic) - 1) == 0) {
	    decipherFormat = 2;
	    line = file.getline();
	}
    }

    unsigned int howmany = 0;

    while (line && (maxSize == 0 || howmany < maxSize)) {
	if (! hypList[howmany].parse(line, vocab, decipherFormat,
					acousticOffset, multiChar, backtrace))
	{
	    file.position() << "bad n-best hyp\n";
	    return false;
	}

	hypList[howmany].rank = howmany;

	howmany ++;

	line = file.getline();
    }

    _numHyps = howmany;

    return true;
}

Boolean
NBestList::readSRInterpFormat(File &file, LHash<VocabString, Array<LogP>* > & nbestScores)
{
  
    unsigned int howmany = 0;
    const char * start = "pr=";
    char * line;
    
    // first strip the possible headers
    while ((line = file.getline()) != NULL) {
        if (strncmp(line, start, strlen(start)) == 0) 
	  break;
    }

    if (!line) {
        file.position() << "empty n-best list" << endl;
        return false;
    }

    LHashIter<VocabString, Array<NBestScore>* > iter(nbestScores);
    VocabString key;
    LHash<VocabString, NBestScore> scores;
    while(iter.next(key)) {
        *scores.insert(key) = 0;
    }    

    while (line && (maxSize == 0 || howmany < maxSize)) {
        if (! hypList[howmany].parseSRInterpFormat(line, vocab, scores))
	{
	    file.position() << "bad n-best hyp\n";
	    return false;
	}

	// copy scores
	iter.init();
	while(Array<NBestScore> ** ppa = iter.next(key)) {
	    Array<NBestScore> & array = **ppa;
	    array[howmany] = *scores.find(key);
	}

	hypList[howmany].rank = howmany;

	howmany ++;

	line = file.getline();
    }

    _numHyps = howmany;

    return true;
}


Boolean
NBestList::write(File &file, Boolean decipherFormat, unsigned numHyps)
{
    if (decipherFormat) {
	file.fprintf("%s\n", backtrace ? nbest2Magic : nbest1Magic);
    }

    for (unsigned h = 0;
	 h < _numHyps && (numHyps == 0 || h < numHyps);
	 h++)
    {
	hypList[h].write(file, vocab, decipherFormat, acousticOffset);
    }

    return true;
}

/*
 * Recompute total scores by recomputing LM scores and adding them to the
 * acoustic scores including a word transition penalty.
 */
void
NBestList::rescoreHyps(LM &lm, double lmScale, double wtScale)
{
    for (unsigned h = 0; h < _numHyps; h++) {
	hypList[h].rescore(lm, lmScale, wtScale);
    }
}

/*
 * Recompute total hyp scores using new scaling constants.
 */
void
NBestList::reweightHyps(double lmScale, double wtScale, double amScale)
{
    for (unsigned h = 0; h < _numHyps; h++) {
	hypList[h].reweight(lmScale, wtScale, amScale);
    }
}

/*
 * Compute posterior probabilities
 */
void
NBestList::computePosteriors(double lmScale, double wtScale,
					    double postScale, double amScale)
{
    /*
     * First compute the numerators for the posteriors
     */
    LogP2 totalNumerator = LogP_Zero;
    LogP scoreOffset;

    unsigned h;
    for (h = 0; h < _numHyps; h++) {
	NBestHyp &hyp = hypList[h];

	/*
	 * This way of computing the total score differs from 
	 * hyp.reweight() in that we're scaling back the acoustic
	 * scores, rather than scaling up the LM scores.
	 *
	 * Store the score back into the nbest list so we can
	 * sort on it later.
	 *
	 * The posterior weight is a parameter that controls the
	 * peakedness of the posterior distribution.
	 *
	 * As a special case, if all weights are zero, we compute the
	 * posterios directly from the stored aggregate scores.
	 */
	LogP totalScore;
	
	if (amScale == 0.0 && lmScale == 0.0 && wtScale == 0.0) {
	    totalScore = hyp.totalScore / postScale;
	} else {
	    totalScore = (weightLogP(amScale, hyp.acousticScore) +
				weightLogP(lmScale, hyp.languageScore) +
				wtScale * hyp.numWords) /
			     postScale;
	}

	/*
	 * To prevent underflow when converting LogP's to Prob's, we 
	 * subtract off the LogP of the first hyp.
	 * This is equivalent to a constant factor on all Prob's, which
	 * cancels in the normalization.
	 */
	if (h == 0) {
	    scoreOffset = totalScore;
	    totalScore = 0.0;
	} else {
	    totalScore -= scoreOffset;
	}

	/*
	 * temporarily store unnormalized log posterior in hyp
	 */
	hyp.posterior = totalScore;

	totalNumerator = AddLogP(totalNumerator, hyp.posterior);
    }

    /*
     * Normalize posteriors
     */
    for (h = 0; h < _numHyps; h++) {
	NBestHyp &hyp = hypList[h];

	hyp.posterior = LogPtoProb(hyp.posterior - totalNumerator);
    }
}

/*
 * Recompute acoustic scores by subtracting recognizer LM scores
 * from totals.
 */
void
NBestList::decipherFix(LM &lm, double lmScale, double wtScale)
{
    for (unsigned h = 0; h < _numHyps; h++) {
	hypList[h].decipherFix(lm, lmScale, wtScale);
    }
}

/*
 * Remove noise and pause words from hyps
 */
void
NBestList::removeNoise(LM &lm)
{
    NBestWordInfo endOfHyp;
    endOfHyp.word = Vocab_None;

    for (unsigned h = 0; h < _numHyps; h++) {
	lm.removeNoise(hypList[h].words);

	NBestWordInfo *wordInfo = hypList[h].wordInfo;

	// remove corresponding tokens from wordInfo array
	if (wordInfo) {
	    unsigned from, to;

	    for (from = 0, to = 0; wordInfo[from].word != Vocab_None; from ++) {
		if (wordInfo[from].word != vocab.pauseIndex() &&
		    !lm.noiseVocab.getWord(wordInfo[from].word))
		{
		    wordInfo[to++] = wordInfo[from];
		}
	    }
	    wordInfo[to] = endOfHyp;
	}
    }
}

/*
 * Normalize acoustic scores so that maximum is 0
 */
void
NBestList::acousticNorm()
{
    unsigned h;
    LogP maxScore = 0.0;

    /*
     * Find maximum acoustic score
     */
    for (h = 0; h < _numHyps; h++) {
	if (h == 0 || hypList[h].acousticScore > maxScore) {
	    maxScore = hypList[h].acousticScore;
	}
    }

    /* 
     * Normalize all scores
     */
    for (h = 0; h < _numHyps; h++) {
	hypList[h].acousticScore -= maxScore;
	hypList[h].totalScore -= maxScore;
    }

    acousticOffset = maxScore;
}

/*
 * Restore acoustic scores to their un-normalized values
 */
void
NBestList::acousticDenorm()
{
    for (unsigned h = 0; h < _numHyps; h++) {
	hypList[h].acousticScore += acousticOffset;
	hypList[h].totalScore -= acousticOffset;
    }

    acousticOffset = 0.0;
}

/*
 * Compute minimal word error of all hyps in the list.
 * As a side-effect the error counts on all nbest hyps are updated.
 * If weight != 0, the error counts are set to the old counts plus the 
 * computed new counts times the given weight.
 */
unsigned
NBestList::wordError(const VocabIndex *words,
		     unsigned &sub, unsigned &ins, unsigned &del, float weight)
{
    unsigned minErr = (unsigned)(-1);

    for (unsigned h = 0; h < _numHyps; h++) {
	unsigned s, i, d;
	unsigned werr = ::wordError(hypList[h].words, words, s, i, d);

	if (h == 0 || werr < minErr) {
	    minErr = werr;
	    sub = s;
	    ins = i;
	    del = d;
	}

	if (weight == 0.0) {
	    hypList[h].numErrors = (float)werr;
	} else {
	    hypList[h].numErrors += weight * werr;
	}
    }

    if (_numHyps == 0) {
	/* 
	 * If the n-best lists is empty we count all reference words as deleted.
	 */
	minErr = del = Vocab::length(words);
	sub = 0;
	ins = 0;
    }

    return minErr;
}

/*
 * Return hyp with minimum expected word error
 */
double
NBestList::minimizeWordError(VocabIndex *words, unsigned length,
				double &subs, double &inss, double &dels,
				unsigned maxRescore, Prob postPrune)
{
    /*
     * Compute expected word errors
     */
    double bestError = 0.0;
    unsigned bestHyp = 0;

    unsigned howmany = (maxRescore > 0) ? maxRescore : _numHyps;
    if (howmany > _numHyps) {
	howmany = _numHyps;
    }

    for (unsigned i = 0; i < howmany; i ++) {
	NBestHyp &hyp = getHyp(i);

	double totalErrors = 0.0;
	double totalSubs = 0.0;
	double totalInss = 0.0;
	double totalDels = 0.0;
	Prob totalPost = 0.0;

	for (unsigned j = 0; j < _numHyps; j ++) {
	    NBestHyp &otherHyp = getHyp(j);

	    if (i != j) {
		unsigned sub, ins, del;
		totalErrors += otherHyp.posterior *
			::wordError(hyp.words, otherHyp.words, sub, ins, del);
		totalSubs += otherHyp.posterior * sub;
		totalInss += otherHyp.posterior * ins;
		totalDels += otherHyp.posterior * del;
	    }

	    /*
	     * Optimization: if the partial accumulated error exceeds the
	     * current best error then this cannot be a new best.
	     */
	    if (i > 0 && totalErrors > bestError) {
		break;
	    }

	    /*
	     * Ignore hyps whose cummulative posterior mass is below threshold
	     */
	    totalPost += otherHyp.posterior;
	    if (postPrune > 0.0 && totalPost > 1.0 - postPrune) {
		break;
	    }
	}

	if (i == 0 || totalErrors < bestError) {
	    bestHyp = i;
	    bestError = totalErrors;
	    subs = totalSubs;
	    inss = totalInss;
	    dels = totalDels;
	}
    }

    if (debug(DEBUG_PRINT_RANK)) {
	cerr << "best hyp = " << bestHyp
	     << " post = " << getHyp(bestHyp).posterior
	     << " wer = " << bestError << endl;
    }

    if (howmany > 0) {
	for (unsigned j = 0; j < length; j ++) {
	    words[j] = getHyp(bestHyp).words[j];

	    if (words[j] == Vocab_None) break;
	}

	return bestError;
    } else {
	if (length > 0) {
	    words[0] = Vocab_None;
	}

	return 0.0;
    }
}

