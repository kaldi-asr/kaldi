/*
 * HTKLattice.cc --
 *	HTK Standard Lattice Format support for SRILM lattices
 *
 *	Note: there is no separate HTKLattice class, only I/O methods!
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2003-2011 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/HTKLattice.cc,v 1.69 2016/04/09 06:53:00 stolcke Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>

#include <string>
#include <limits>

#include "Prob.h"
#include "Array.cc"
#include "LHash.cc"
#include "Lattice.h"
#include "MultiwordVocab.h"
#include "NBest.h"		// for phoneSeparator and frameLength defn
#include "RefList.h"
#include "MStringTokUtil.h"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_ARRAY(HTKWordInfo);
#endif

#if defined(_MSC_VER) || defined(WIN32)
#define snprintf		_snprintf
#endif

/* from Lattice.cc */
#define DebugPrintFatalMessages         1 
#define DebugPrintFunctionality         1 

const char *HTKLattice_Version = "1.1";

const char *HTK_null_word = "!NULL";

const char HTK_single_quote = '\'';
const char HTK_double_quote = '\"';
const char HTK_escape_quote = '\\';

const float HTK_def_tscale = 1.0;
const float HTK_def_acscale = 1.0;
const float HTK_def_lmscale = 1.0;
const float HTK_def_ngscale = 1.0;
const float HTK_def_wdpenalty = 0.0;
const float HTK_def_prscale = 1.0;
const float HTK_def_duscale = 0.0;
const float HTK_def_xscale = 0.0;

double HTK_LogP_Zero = LogP_Zero;

const unsigned dbl_prec = numeric_limits<double>::digits10 + 1;

HTKHeader::HTKHeader()
    : logbase(10), tscale(HTK_def_tscale), acscale(HTK_def_acscale),
      ngscale(HTK_def_ngscale), lmscale(HTK_def_lmscale),
      wdpenalty(HTK_def_wdpenalty), prscale(HTK_def_prscale),
      duscale(HTK_def_duscale), amscale(HTK_undef_float),
      x1scale(HTK_def_xscale), x2scale(HTK_def_xscale), x3scale(HTK_def_xscale),
      x4scale(HTK_def_xscale), x5scale(HTK_def_xscale), x6scale(HTK_def_xscale),
      x7scale(HTK_def_xscale), x8scale(HTK_def_xscale), x9scale(HTK_def_xscale),
      vocab(0), lmname(0), ngname(0), hmms(0),
      wordsOnNodes(false), scoresOnNodes(false), useQuotes(false)
{
};

HTKHeader::HTKHeader(double acscale, double lmscale, double ngscale,
			double prscale, double duscale, double wdpenalty,
			double x1scale, double x2scale, double x3scale,
			double x4scale, double x5scale, double x6scale,
			double x7scale, double x8scale, double x9scale)
    : logbase(10), tscale(HTK_def_tscale), acscale(acscale),
      ngscale(ngscale), lmscale(lmscale),
      wdpenalty(wdpenalty), prscale(prscale),
      duscale(duscale), amscale(HTK_undef_float),
      x1scale(x2scale), x2scale(x2scale), x3scale(x3scale),
      x4scale(x4scale), x5scale(x5scale), x6scale(x6scale),
      x7scale(x7scale), x8scale(x8scale), x9scale(x9scale),
      vocab(0), lmname(0), ngname(0), hmms(0),
      wordsOnNodes(false), scoresOnNodes(false), useQuotes(false)
{
};

HTKHeader::~HTKHeader()
{
    if (vocab) free(vocab);
    if (lmname) free(lmname);
    if (ngname) free(ngname);
    if (hmms) free(hmms);
}

HTKHeader &
HTKHeader::operator= (const HTKHeader &other)
{
    if (&other == this) {
	return *this;
    }

    if (vocab) free(vocab);
    if (lmname) free(lmname);
    if (ngname) free(ngname);
    if (hmms) free(hmms);

    tscale = other.tscale;
    acscale = other.acscale;
    ngscale = other.ngscale;
    lmscale = other.lmscale;
    wdpenalty = other.wdpenalty;
    prscale = other.prscale;
    duscale = other.duscale;
    x1scale = other.x1scale;
    x2scale = other.x2scale;
    x3scale = other.x3scale;
    x4scale = other.x4scale;
    x5scale = other.x5scale;
    x6scale = other.x6scale;
    x7scale = other.x7scale;
    x8scale = other.x8scale;
    x9scale = other.x9scale;
    amscale = other.amscale;
    if (other.vocab == 0) {
	vocab = 0;
    } else {
	vocab = strdup(other.vocab);
	assert(vocab != 0);
    }
    if (other.lmname == 0) {
	lmname = 0;
    } else {
	lmname = strdup(other.lmname);
	assert(lmname != 0);
    }
    if (other.ngname == 0) {
	ngname = 0;
    } else {
	ngname = strdup(other.ngname);
	assert(ngname != 0);
    }
    if (other.hmms == 0) {
	hmms = 0;
    } else {
	hmms = strdup(other.hmms);
	assert(hmms != 0);
    }

    return *this;
}


HTKWordInfo::HTKWordInfo()
    : time(HTK_undef_float), word(Vocab_None), wordLabel(0),
      var(HTK_undef_uint), div(0), states(0),
      acoustic(HTK_undef_float), ngram(HTK_undef_float),
      language(HTK_undef_float), pron(HTK_undef_float),
      duration(HTK_undef_float), xscore1(HTK_undef_float),
      xscore2(HTK_undef_float), xscore3(HTK_undef_float),
      xscore4(HTK_undef_float), xscore5(HTK_undef_float),
      xscore6(HTK_undef_float), xscore7(HTK_undef_float),
      xscore8(HTK_undef_float), xscore9(HTK_undef_float),
      posterior(HTK_undef_float)
{
}

HTKWordInfo::HTKWordInfo(const HTKWordInfo &other)
    : wordLabel(0), div(0), states(0)
{
    *this = other;
}

HTKWordInfo::~HTKWordInfo()
{
    if (wordLabel) free((char *)wordLabel);
    if (div) free(div);
    if (states) free(states);
}

HTKWordInfo &
HTKWordInfo::operator= (const HTKWordInfo &other)
{
    if (&other == this) {
	return *this;
    }

    if (wordLabel) free((char *)wordLabel);
    if (div) free(div);
    if (states) free(states);

    time = other.time;
    word = other.word;
    if (other.wordLabel == 0) {
	wordLabel = 0;
    } else {
	wordLabel = strdup(other.wordLabel);
	assert(wordLabel != 0);
    }
    var = other.var;
    if (other.div == 0) {
	div = 0;
    } else {
	div = strdup(other.div);
	assert(div != 0);
    }
    if (other.states == 0) {
	states = 0;
    } else {
	states = strdup(other.states);
	assert(states != 0);
    }
    acoustic = other.acoustic;
    ngram = other.ngram;
    language = other.language;
    pron = other.pron;
    duration = other.duration;
    xscore1 = other.xscore1;
    xscore2 = other.xscore2;
    xscore3 = other.xscore3;
    xscore4 = other.xscore4;
    xscore5 = other.xscore5;
    xscore6 = other.xscore6;
    xscore7 = other.xscore7;
    xscore8 = other.xscore8;
    xscore9 = other.xscore9;
    posterior = other.posterior;
    return *this;
}

/* 
 * Format HTKWordInfo (for debugging)
 */
ostream &
operator<< (ostream &stream, HTKWordInfo &link)
{
    stream << "[HTKWordInfo";

    if (link.word != Vocab_None) {
	stream << " WORD=" << link.word;
    }
    if (link.time != HTK_undef_float) {
	stream << " time=" << link.time;
    }
    if (link.var != HTK_undef_uint) {
	stream << " var=" << link.var;
    }
    if (link.div != 0) {
	stream << " div=" << link.div;
    }
    if (link.states != 0) {
	stream << " s=" << link.states;
    }
    if (link.acoustic != HTK_undef_float) {
	stream << " a=" << link.acoustic;
    }
    if (link.ngram != HTK_undef_float) {
	stream << " n=" << link.ngram;
    }
    if (link.language != HTK_undef_float) {
	stream << " l=" << link.language;
    }
    if (link.pron != HTK_undef_float) {
	stream << " r=" << link.pron;
    }
    if (link.duration != HTK_undef_float) {
	stream << " ds=" << link.duration;
    }
    if (link.xscore1 != HTK_undef_float) {
	stream << " x1=" << link.xscore1;
    }
    if (link.xscore2 != HTK_undef_float) {
	stream << " x2=" << link.xscore2;
    }
    if (link.xscore3 != HTK_undef_float) {
	stream << " x3=" << link.xscore3;
    }
    if (link.xscore4 != HTK_undef_float) {
	stream << " x4=" << link.xscore4;
    }
    if (link.xscore5 != HTK_undef_float) {
	stream << " x5=" << link.xscore5;
    }
    if (link.xscore6 != HTK_undef_float) {
	stream << " x6=" << link.xscore6;
    }
    if (link.xscore7 != HTK_undef_float) {
	stream << " x7=" << link.xscore7;
    }
    if (link.xscore8 != HTK_undef_float) {
	stream << " x8=" << link.xscore8;
    }
    if (link.xscore9 != HTK_undef_float) {
	stream << " x9=" << link.xscore9;
    }
    if (link.posterior != HTK_undef_float) {
	stream << " p=" << link.posterior;
    }
    stream << "]";
    return stream;
}


/*
 * Find the next key=value pair in line, return string value, nad 
 * advance line pointer past it.
 * The string pointed to by line is modified in the process.
 */
static char *
getHTKField(char *&line, char *&value, Boolean useQuotes)
{
    char *cp = line;
    char *key;

    do {
	switch (*cp) {
	case '\0':
	case '#':
		return 0;
		break;
	case ' ':
	case '\t':
	case '\n':
	case '\r':
		cp ++;
		break;
	default:
		key = cp;

		while (*cp != '\0' && !isspace(*cp) && *cp != '=') cp++;

		if (*cp == '=') {
		    *(cp++) = '\0';	// terminate key string
		    value = cp;		// beginning of value string
		    char *cpv = cp;	// target location for copying value

		    char inquote = '\0';

		    /*
		     * Quotes are only treated specially if they 
		     * occur in first position
		     */
		    if (useQuotes &&
			(*cp == HTK_single_quote || *cp == HTK_double_quote))
		    {
			inquote = *(cp++);
		    }

		    while (*cp != '\0') {
		  	/*
			 * Backslash quotes not inside single quotes
			 */
			if (useQuotes && *cp == HTK_escape_quote &&
			    (inquote != HTK_single_quote || *(cp+1) == HTK_single_quote))
			{
			    /*
			     * Backslash quote processing
			     */
			    cp ++;
			    if (*cp == '\0') {
				/*
				 * Shouldn't happen, we just ignore it
				 */
				break;
			    } else if (*cp == '0') {
				/*
				 * Octal char code
				 */
				unsigned charcode;
				int charlen;
				sscanf(cp, "%o%n", &charcode, &charlen);
				*(cpv++) = charcode;
				cp += charlen;
			    } else {
				/*
				 * Other quoted character
				 */
				*(cpv++) = *(cp++);
			    }
			} else if (!inquote && isspace(*cp)) {
			    /*
			     * String delimited by white-space
			     */
			    cp ++;
			    break;
			} else if (inquote && *cp == inquote) {
			    /*
			     * String delimited by end quote
			     */
			    cp ++;
			    break;
			} else {
			    /* 
			     * Character in string
			     */
			    *(cpv++) = *(cp++);
			}
		    }
		    *cpv = '\0';	// terminate value string
		} else {
		    value = cp;		// beginning of value string
		    if (*cp != '\0') {
			*(cp++) = '\0';	// terminate value string
		    }
		}

		line = cp;
		return key;
	}
    } while (1);
}

/*
 * Convert string to log score 
 */
static inline LogP
getHTKscore(const char *value, double logbase, File &file)
{
    if (logbase > 0.0) {
	LogP score;
	if (parseLogP(value, score)) {
	    if (score == LogP_Zero) {
		return HTK_LogP_Zero;
	    } else {
		return score * ProbToLogP(logbase);
	    }
	} else {
	    file.position() << "warning: malformed HTK log score "
			    << value << endl;
	    return HTK_LogP_Zero;
	}
    } else {
	Prob score = atof(value);
	if (score == 0.0) {
	    return HTK_LogP_Zero;
	} else {
	    return ProbToLogP(score);
	}
    }
}

/*
 * Output quoted version of string
 */
// Use "stdio" functions in File() object to allow writing in-memory to File() string object.
static void
printQuoted(File &file, const char *name, Boolean useQuotes)
{
    Boolean octalPrinted = false;

    if (!useQuotes) {
	file.fputs(name);
    } else {
	for (const char *cp = name; *cp != '\0'; cp ++) {
	    if (*cp == ' ' || *cp == HTK_escape_quote ||
		(cp == name &&
		    (*cp == HTK_single_quote || *cp == HTK_double_quote)) ||
		(octalPrinted && isdigit(*cp)))
	    {
		/*
		 * This character needs to be quoted
		 */
		file.fputc(HTK_escape_quote);
		file.fputc(*cp);
		octalPrinted = false;
	    } else if (!isprint(*cp) || isspace(*cp)) {
		/*
		 * Print as octal char code
		 */
		file.fprintf("%c0%o", HTK_escape_quote, *cp);
		octalPrinted = true;
	    } else {
		/*
		 * Print as plain character
		 */
		file.fputc(*cp);
		octalPrinted = false;
	    }
	}
    }
}

/*
 * Set user-specified parameters in the HTK lattice header structure
 */
void
Lattice::setHTKHeader(HTKHeader &header)
{
    if (header.logbase != HTK_undef_float) {
	htkheader.logbase = header.logbase;
    }
    if (header.acscale != HTK_undef_float) {
	htkheader.acscale = header.acscale;
    }
    if (header.lmscale != HTK_undef_float) {
	htkheader.lmscale = header.lmscale;
    }
    if (header.ngscale != HTK_undef_float) {
	htkheader.ngscale = header.ngscale;
    }
    if (header.prscale != HTK_undef_float) {
	htkheader.prscale = header.prscale;
    }
    if (header.duscale != HTK_undef_float) {
	htkheader.duscale = header.duscale;
    }
    if (header.wdpenalty != HTK_undef_float) {
	// scale user-specific wdpenalty from user-specified/default logbase
	if (htkheader.logbase > 0.0) {
	    htkheader.wdpenalty =
			header.wdpenalty * ProbToLogP(htkheader.logbase);
	} else {
	    htkheader.wdpenalty = ProbToLogP(header.wdpenalty);
	}
    }
    if (header.x1scale != HTK_undef_float) {
	htkheader.x1scale = header.x1scale;
    }
    if (header.x2scale != HTK_undef_float) {
	htkheader.x2scale = header.x2scale;
    }
    if (header.x3scale != HTK_undef_float) {
	htkheader.x3scale = header.x3scale;
    }
    if (header.x4scale != HTK_undef_float) {
	htkheader.x4scale = header.x4scale;
    }
    if (header.x5scale != HTK_undef_float) {
	htkheader.x5scale = header.x5scale;
    }
    if (header.x6scale != HTK_undef_float) {
	htkheader.x6scale = header.x6scale;
    }
    if (header.x7scale != HTK_undef_float) {
	htkheader.x7scale = header.x7scale;
    }
    if (header.x8scale != HTK_undef_float) {
	htkheader.x8scale = header.x8scale;
    }
    if (header.x9scale != HTK_undef_float) {
	htkheader.x9scale = header.x9scale;
    }
    if (header.amscale != HTK_undef_float) {
	htkheader.amscale = header.amscale;
    }
    htkheader.wordsOnNodes = header.wordsOnNodes;
    htkheader.scoresOnNodes = header.scoresOnNodes;
    htkheader.useQuotes = header.useQuotes;
}


/*
 * Input lattice in HTK format
 *	Algorithm:
 *	- each HTK node becomes a null node.
 *	- each HTK link becomes a non-null node.
 *	- word and other link information is added to the non-null nodes.
 *	- link information attached to HTK nodes is added to non-null nodes.
 *	- lattice transition weights are computed as a log-linear combination
 *	  of HTK scores.
 * Arguments:
 *	- if header != 0, supplied scaling parameters override information
 *	  from lattice header
 *	- if useNullNodes == false null nodes corresponding to original
 *	  HTK nodes are eliminated
 */
Boolean
Lattice::readHTK(File &file, HTKHeader *header, Boolean useNullNodes)
{
    removeAll();

    unsigned HTKnumnodes = 0;
    float HTKlogbase = (float) M_E;
    unsigned HTKfinal = HTK_undef_uint;
    unsigned HTKinitial = HTK_undef_uint;
    char HTKdirection = 'f';
    char HTKwdpenalty[100];
    HTKwdpenalty[0] = HTKwdpenalty[sizeof(HTKwdpenalty)-1] = '\0';

    LHash<unsigned, NodeIndex> nodeMap;		// maps HTK nodes->lattice nodes
    Array<HTKWordInfo> nodeInfoMap;		// node-based link information

    // dummy word used temporarily to represent HTK nodes
    // (could have used null nodes, but this way we preserve null nodes in
    // the input lattice)
    const char *HTKNodeWord = "***HTK_Node***";
    VocabIndex HTKNodeDummy = useNullNodes ? Vocab_None :
					     vocab.addWord(HTKNodeWord);

    /*
     * Override supplied header parameters
     */
    if (header != 0) {
	setHTKHeader(*header);
    }

    /*
     * Parse HTK lattice file
     */
    while (char *line = file.getline()) {
	char *key;
	char *value;
	string savedLine = line;

	/*
	 * Parse key=value pairs
	 * (we test for frequent fields first to save time)
	 * We assume that header information comes before node information,
	 * which comes before link information.  However, this is is not
	 * enforced, and incomplete lattices may result if the input file
	 * contains things out of order.
	 */
	while ((key = getHTKField(line, value, htkheader.useQuotes))) {
#define keyis(x)	(strcmp(key, (x)) == 0)
	    /*
	     * Link fields
	     */
	    if (keyis("J")) {
		unsigned HTKlinkno = atoi(value);

		/*
		 * parse link fields
		 */
		HTKWordInfo *linkinfo = new HTKWordInfo;
		assert(linkinfo != 0);
				// allocates new HTKWordInfo pointer in lattice
		htkinfos[htkinfos.size()] = linkinfo;

		unsigned HTKstartnode, HTKendnode;
		NodeIndex startIndex = NoNode, endIndex = NoNode;

		while ((key = getHTKField(line, value, htkheader.useQuotes))) {
		    if (keyis("S") || keyis("START")) {
			HTKstartnode = atoi(value);
			Boolean found;
			NodeIndex *startIndexPtr =
				nodeMap.insert(HTKstartnode, found);
			if (!found) {
			    // node index not seen before; create it
			    *startIndexPtr = dupNode(Vocab_None);
			}
			startIndex = *startIndexPtr;

		    } else if (keyis("E") || keyis("END")) {
			HTKendnode = atoi(value);
			Boolean found;
			NodeIndex *endIndexPtr =
				nodeMap.insert(HTKendnode, found);
			if (!found) {
			    // node index not seen before; create it
			    *endIndexPtr = dupNode(Vocab_None);
			}
			endIndex = *endIndexPtr;

		    } else if (keyis("W") || keyis("WORD")) {
			if (strcmp(value, HTK_null_word) == 0) {
			    linkinfo->word = Vocab_None;
			} else if (useUnk || keepUnk) {
			    linkinfo->word =
					vocab.getIndex(value, vocab.unkIndex());
			    if (keepUnk && linkinfo->word == vocab.unkIndex()) {
				linkinfo->wordLabel = strdup(value);
				assert(linkinfo->wordLabel != 0);
			    }
			} else {
			    linkinfo->word = vocab.addWord(value);
			}
			if (linkinfo->word == vocab.ssIndex()) {
			    if (debug(DebugPrintFunctionality)) {
				dout()  << "Lattice::readHTK: discarding explicit start-of-sentence tag\n";
			    }
			    linkinfo->word = Vocab_None;
			}
			if (linkinfo->word == vocab.seIndex()) {
			    if (debug(DebugPrintFunctionality)) {
				dout()  << "Lattice::readHTK: discarding explicit end-of-sentence tag\n";
			    }
			    linkinfo->word = Vocab_None;
			}
		    } else if (keyis("v") || keyis("var")) {
			linkinfo->var = atoi(value);
		    } else if (keyis("d") || keyis("div")) {
			linkinfo->div = strdup(value);
			assert(linkinfo->div != 0);
		    } else if (keyis("s") || keyis("states")) {
			linkinfo->states = strdup(value);
			assert(linkinfo->states != 0);
		    } else if (keyis("a") || keyis("acoustic")) {
			linkinfo->acoustic = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("n") || keyis("ngram")) {
			linkinfo->ngram = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("l") || keyis("language")) {
			linkinfo->language = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("r")) {
			linkinfo->pron = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("ds")) {
			linkinfo->duration = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x1")) {
			linkinfo->xscore1 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x2")) {
			linkinfo->xscore2 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x3")) {
			linkinfo->xscore3 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x4")) {
			linkinfo->xscore4 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x5")) {
			linkinfo->xscore5 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x6")) {
			linkinfo->xscore6 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x7")) {
			linkinfo->xscore7 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x8")) {
			linkinfo->xscore8 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x9")) {
			linkinfo->xscore9 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("p")) {
			linkinfo->posterior = atof(value);
		    } else {
			file.position() << "unexpected link field name "
					<< key << endl;
			if (!useNullNodes) vocab.remove(HTKNodeDummy);
			return false;
		    }
		}

		if (startIndex == NoNode) {
		    file.position() << "missing start node spec\n";
		    if (!useNullNodes) vocab.remove(HTKNodeDummy);
		    return false;
		}

		if (endIndex == NoNode) {
		    file.position() << "missing end node spec\n";
		    if (!useNullNodes) vocab.remove(HTKNodeDummy);
		    return false;
		}

		/*
		 * fill in unspecified link info from associated node info
		 * 'forward' lattices use end-node information.
		 * 'backward' lattices use start-node information.
		 */
		HTKWordInfo *nodeinfo = 0;
		if (HTKdirection == 'f') {
		    nodeinfo = &nodeInfoMap[HTKendnode];
		} else if (HTKdirection == 'b') {
		    nodeinfo = &nodeInfoMap[HTKstartnode];
		}

		if (nodeinfo != 0) {
		    linkinfo->time = nodeinfo->time;

		    if (linkinfo->word == Vocab_None) {
			linkinfo->word = nodeinfo->word;
		    }
		    if (linkinfo->wordLabel == 0 && nodeinfo->wordLabel != 0) {
			linkinfo->wordLabel = strdup(nodeinfo->wordLabel);
			assert(linkinfo->wordLabel != 0);
		    }
		    if (linkinfo->var == HTK_undef_uint) {
			linkinfo->var = nodeinfo->var;
		    }
		    if (linkinfo->div == 0 && nodeinfo->div != 0) {
			linkinfo->div = strdup(nodeinfo->div);
			assert(linkinfo->div != 0);
		    }
		    if (linkinfo->states == 0 && nodeinfo->states != 0) {
			linkinfo->states = strdup(nodeinfo->states);
			assert(linkinfo->states != 0);
		    }
		    if (linkinfo->acoustic == HTK_undef_float) {
			linkinfo->acoustic = nodeinfo->acoustic;
		    }
		    if (linkinfo->pron == HTK_undef_float) {
			linkinfo->pron = nodeinfo->pron;
		    }
		    if (linkinfo->duration == HTK_undef_float) {
			linkinfo->duration = nodeinfo->duration;
		    }
		    if (linkinfo->xscore1 == HTK_undef_float) {
			linkinfo->xscore1 = nodeinfo->xscore1;
		    }
		    if (linkinfo->xscore2 == HTK_undef_float) {
			linkinfo->xscore2 = nodeinfo->xscore2;
		    }
		    if (linkinfo->xscore3 == HTK_undef_float) {
			linkinfo->xscore3 = nodeinfo->xscore3;
		    }
		    if (linkinfo->xscore4 == HTK_undef_float) {
			linkinfo->xscore4 = nodeinfo->xscore4;
		    }
		    if (linkinfo->xscore5 == HTK_undef_float) {
			linkinfo->xscore5 = nodeinfo->xscore5;
		    }
		    if (linkinfo->xscore6 == HTK_undef_float) {
			linkinfo->xscore6 = nodeinfo->xscore6;
		    }
		    if (linkinfo->xscore7 == HTK_undef_float) {
			linkinfo->xscore7 = nodeinfo->xscore7;
		    }
		    if (linkinfo->xscore8 == HTK_undef_float) {
			linkinfo->xscore8 = nodeinfo->xscore8;
		    }
		    if (linkinfo->xscore9 == HTK_undef_float) {
			linkinfo->xscore9 = nodeinfo->xscore9;
		    }
		}

		/*
		 * Create lattice node
		 */
		NodeIndex newNode = dupNode(linkinfo->word, 0, linkinfo);

		/*
		 * Compute lattice transition weight as a weighted combination
		 * of HTK lattice scores
		 */
		LogP weight = LogP_One;

		if (linkinfo->acoustic != HTK_undef_float) {
		    if (htkheader.acscale != 0.0) {
			weight += htkheader.acscale * linkinfo->acoustic;
		    }
		}
		if (linkinfo->ngram != HTK_undef_float) {
		    if (htkheader.ngscale != 0.0) {
			weight += htkheader.ngscale * linkinfo->ngram;
		    }
		}
		if (linkinfo->language != HTK_undef_float) {
		    // if lmscale == 0 we ignore even -infinity lm scores
		    if (htkheader.lmscale != 0.0) {
			weight += htkheader.lmscale * linkinfo->language;
		    }
		}
		if (linkinfo->pron != HTK_undef_float) {
		    if (htkheader.prscale != 0.0) {
			weight += htkheader.prscale * linkinfo->pron;
		    }
		}
		if (linkinfo->duration != HTK_undef_float) {
		    if (htkheader.duscale != 0.0) {
			weight += htkheader.duscale * linkinfo->duration;
		    }
		}
		if (!ignoreWord(linkinfo->word)) {
		    weight += htkheader.wdpenalty;
		}
		if (linkinfo->xscore1 != HTK_undef_float) {
		    if (htkheader.x1scale != 0.0) {
			weight += htkheader.x1scale * linkinfo->xscore1;
		    }
		}
		if (linkinfo->xscore2 != HTK_undef_float) {
		    if (htkheader.x2scale != 0.0) {
			weight += htkheader.x2scale * linkinfo->xscore2;
		    }
		}
		if (linkinfo->xscore3 != HTK_undef_float) {
		    if (htkheader.x3scale != 0.0) {
			weight += htkheader.x3scale * linkinfo->xscore3;
		    }
		}
		if (linkinfo->xscore4 != HTK_undef_float) {
		    if (htkheader.x4scale != 0.0) {
			weight += htkheader.x4scale * linkinfo->xscore4;
		    }
		}
		if (linkinfo->xscore5 != HTK_undef_float) {
		    if (htkheader.x5scale != 0.0) {
			weight += htkheader.x5scale * linkinfo->xscore5;
		    }
		}
		if (linkinfo->xscore6 != HTK_undef_float) {
		    if (htkheader.x6scale != 0.0) {
			weight += htkheader.x6scale * linkinfo->xscore6;
		    }
		}
		if (linkinfo->xscore7 != HTK_undef_float) {
		    if (htkheader.x7scale != 0.0) {
			weight += htkheader.x7scale * linkinfo->xscore7;
		    }
		}
		if (linkinfo->xscore8 != HTK_undef_float) {
		    if (htkheader.x8scale != 0.0) {
			weight += htkheader.x8scale * linkinfo->xscore8;
		    }
		}
		if (linkinfo->xscore9 != HTK_undef_float) {
		    if (htkheader.x9scale != 0.0) {
			weight += htkheader.x9scale * linkinfo->xscore9;
		    }
		}

		if (isnan(weight)) {
		    file.position() << "link " << HTKlinkno << " has NaN weight\n";
		    if (!useNullNodes) vocab.remove(HTKNodeDummy);
		    return false;
		}

		/*
		 * Add transitions from start node, and to end node
		 */
		LatticeTransition trans1(weight, 0);
		insertTrans(startIndex, newNode, trans1);

		LatticeTransition trans2(LogP_One, 0);
		insertTrans(newNode, endIndex, trans2);

		continue;

	    /*
	     * Node fields
	     */
	    } else if (keyis("I")) {
		unsigned HTKnodeno = atoi(value);

		/*
		 * create a null node for this HTK node,
		 * and record node-related info.
		 */
		NodeIndex nullNodeIndex = dupNode(HTKNodeDummy);

		*nodeMap.insert(HTKnodeno) = nullNodeIndex;
		HTKWordInfo &nodeinfo = nodeInfoMap[HTKnodeno];

		/*
		 * parse node fields
		 */
		while ((key = getHTKField(line, value, htkheader.useQuotes))) {
		    if (keyis("t") || keyis("time")) {
			nodeinfo.time = atof(value);
		    } else if (keyis("W") || keyis("WORD")) {
			if (strcmp(value, HTK_null_word) == 0) {
			    nodeinfo.word = Vocab_None;
			} else if (useUnk || keepUnk) {
			    nodeinfo.word =
					vocab.getIndex(value, vocab.unkIndex());
			    if (keepUnk && nodeinfo.word == vocab.unkIndex()) {
				nodeinfo.wordLabel = strdup(value);
				assert(nodeinfo.wordLabel != 0);
			    }
			} else {
			    nodeinfo.word = vocab.addWord(value);
			}
			if (nodeinfo.word == vocab.ssIndex()) {
			    if (debug(DebugPrintFunctionality)) {
				dout()  << "Lattice::readHTK: discarding explicit start-of-sentence tag\n";
			    }
			    nodeinfo.word = Vocab_None;
			}
			if (nodeinfo.word == vocab.seIndex()) {
			    if (debug(DebugPrintFunctionality)) {
				dout()  << "Lattice::readHTK: discarding explicit end-of-sentence tag\n";
			    }
			    nodeinfo.word = Vocab_None;
			}
		    } else if (keyis("v") || keyis("var")) {
			nodeinfo.var = atoi(value);
		    } else if (keyis("d") || keyis("div")) {
			nodeinfo.div = strdup(value);
			assert(nodeinfo.div != 0);
		    } else if (keyis("s") || keyis("states")) {
			nodeinfo.states = strdup(value);
			assert(nodeinfo.states != 0);
		    } else if (keyis("a") || keyis("acoustic")) {
			nodeinfo.acoustic = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("r")) {
			nodeinfo.pron = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("ds")) {
			nodeinfo.duration = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x1")) {
			nodeinfo.xscore1 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x2")) {
			nodeinfo.xscore2 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x3")) {
			nodeinfo.xscore3 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x4")) {
			nodeinfo.xscore4 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x5")) {
			nodeinfo.xscore5 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x6")) {
			nodeinfo.xscore6 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x7")) {
			nodeinfo.xscore7 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x8")) {
			nodeinfo.xscore8 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("x9")) {
			nodeinfo.xscore9 = getHTKscore(value, HTKlogbase, file);
		    } else if (keyis("p")) {
			nodeinfo.posterior = atof(value);
		    } else {
			file.position() << "unexpected node field name "
					<< key << endl;
			if (!useNullNodes) vocab.remove(HTKNodeDummy);
			return false;
		    }
		}

		if (nodeinfo.time != HTK_undef_float) {
		    // record node time, but no word-related info
		    LatticeNode *nullNode = findNode(nullNodeIndex);
		    assert(nullNode != 0);

		    HTKWordInfo *nullInfo = new HTKWordInfo;
		    assert(nullInfo != 0);
		    htkinfos[htkinfos.size()] = nullInfo;

		    nullNode->htkinfo = nullInfo;
		    nullInfo->time = nodeinfo.time;
		}

		continue;

	    /*
	     * Header fields
	     */
	    } else if (keyis("V") || keyis("VERSION")) {
		; 		// ignore
	    } else if ( keyis("U") || keyis("UTTERANCE")) {
		if (name) free((void *)name);

		// HACK: strip duration spec (which shouldn't be there)
		char *p = strstr(value, "(duration=");
		if (p != 0) *p = '\0';
		    
		unsigned uttlen = strlen(value);

		// remove HTK double quotes
		if (value[0] == '"' && uttlen > 1 && value[uttlen-1] == '"') {
		   value[uttlen-1] = '\0';
		   value = &value[1];
		   uttlen -= 2;
		}

		// remove HTK alias= specification
		p = strstr(value, "=");
		if (p != 0) {
		    *p = '\0';
		    uttlen = strlen(value);
		}

		name = strdup(idFromFilename(value));
		assert(name != 0);
	    } else if (keyis("base")) {
		HTKlogbase = atof(value);

		if (HTKwdpenalty[0] &&
		    (header == 0 || header->wdpenalty == HTK_undef_float))
		{
		    // recompute wdpenalty with new logbase
		    htkheader.wdpenalty =
				    getHTKscore(HTKwdpenalty, HTKlogbase, file);
		}
	    } else if (keyis("start")) {
		HTKinitial = atoi(value);
	    } else if (keyis("end")) {
		HTKfinal = atoi(value);
	    } else if (keyis("dir")) {
		HTKdirection = value[0];
	    } else if (keyis("tscale")) {
		htkheader.tscale = atof(value);
	    } else if (keyis("hmms")) {
		htkheader.hmms = strdup(value);
		assert(htkheader.hmms != 0);
	    } else if (keyis("ngname")) {
		htkheader.ngname = strdup(value);
		assert(htkheader.ngname != 0);
	    } else if (keyis("lmname")) {
		htkheader.lmname = strdup(value);
		assert(htkheader.lmname != 0);
	    } else if (keyis("vocab")) {
		htkheader.vocab = strdup(value);
		assert(htkheader.vocab != 0);
	    } else if (keyis("acscale")) {
		if (header == 0 || header->acscale == HTK_undef_float) {
		    htkheader.acscale = atof(value);
		}
	    } else if (keyis("ngscale")) {
		if (header == 0 || header->ngscale == HTK_undef_float) {
		    htkheader.ngscale = atof(value);
		}
	    } else if (keyis("lmscale")) {
		if (header == 0 || header->lmscale == HTK_undef_float) {
		    htkheader.lmscale = atof(value);
		}
	    } else if (keyis("prscale")) {
		if (header == 0 || header->prscale == HTK_undef_float) {
		    htkheader.prscale = atof(value);
		}
	    } else if (keyis("duscale")) {
		if (header == 0 || header->duscale == HTK_undef_float) {
		    htkheader.duscale = atof(value);
		}
	    } else if (keyis("wdpenalty")) {
		if (header == 0 || header->wdpenalty == HTK_undef_float) {
		    htkheader.wdpenalty = getHTKscore(value, HTKlogbase, file);
		    strncpy(HTKwdpenalty, value, sizeof(HTKwdpenalty)-1);
		}
	    } else if (keyis("x1scale")) {
		if (header == 0 || header->x1scale == HTK_undef_float) {
		    htkheader.x1scale = atof(value);
		}
	    } else if (keyis("x2scale")) {
		if (header == 0 || header->x2scale == HTK_undef_float) {
		    htkheader.x2scale = atof(value);
		}
	    } else if (keyis("x3scale")) {
		if (header == 0 || header->x3scale == HTK_undef_float) {
		    htkheader.x3scale = atof(value);
		}
	    } else if (keyis("x4scale")) {
		if (header == 0 || header->x4scale == HTK_undef_float) {
		    htkheader.x4scale = atof(value);
		}
	    } else if (keyis("x5scale")) {
		if (header == 0 || header->x5scale == HTK_undef_float) {
		    htkheader.x5scale = atof(value);
		}
	    } else if (keyis("x6scale")) {
		if (header == 0 || header->x6scale == HTK_undef_float) {
		    htkheader.x6scale = atof(value);
		}
	    } else if (keyis("x7scale")) {
		if (header == 0 || header->x7scale == HTK_undef_float) {
		    htkheader.x7scale = atof(value);
		}
	    } else if (keyis("x8scale")) {
		if (header == 0 || header->x8scale == HTK_undef_float) {
		    htkheader.x8scale = atof(value);
		}
	    } else if (keyis("x9scale")) {
		if (header == 0 || header->x9scale == HTK_undef_float) {
		    htkheader.x9scale = atof(value);
		}
	    } else if (keyis("amscale")) {
		if (header == 0 || header->amscale == HTK_undef_float) {
		    htkheader.amscale = atof(value);
		}
	    } else if (keyis("NODES") || keyis("N")) {
		HTKnumnodes = atoi(value);
	    } else if (keyis("LINKS") || keyis("L")) {
		/* not used */
		/*HTKnumlinks = atoi(value)*/;
	    } else if (keyis("engineconfig")) {
		/* silently ignored */;
	    } else {
		file.position() << "ignoring field name \"" << key << "\"" << endl;
	    }
#undef keyis
	}
    }

    if (HTKnumnodes == 0) {
	file.position() << "lattice has no nodes\n";
	if (!useNullNodes) vocab.remove(HTKNodeDummy);
	return false;
    }

    /*
     * Set up initial node
     */
    HTKWordInfo *initialinfo;
    LatticeNode *initialNode;

    if (HTKinitial != HTK_undef_uint) {
	initialinfo = &nodeInfoMap[HTKinitial];
	NodeIndex *initialPtr = nodeMap.find(HTKinitial);
	if (initialPtr) {
	    initial = *initialPtr;
	    initialNode = findNode(initial);
	} else {
	    file.position() << "undefined start node " << HTKinitial << endl;
	    if (!useNullNodes) vocab.remove(HTKNodeDummy);
	    return false;
	}
    } else {
	// search for start node: the one without incoming transitions
	LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
	NodeIndex nodeIndex;

	initialNode = 0;
	while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	    if (node->inTransitions.numEntries() == 0) {
		initial = nodeIndex;
		initialNode = node;
		break;
	    }
	}

	if (!initialNode) {
	    file.position() << "could not find start node\n";
	    if (!useNullNodes) vocab.remove(HTKNodeDummy);
	    return false;
	}

	// now find the HTK node info associated with first node
	LHashIter<unsigned, NodeIndex> nodeMapIter(nodeMap);
	unsigned htkNode;
	while (NodeIndex *pfsgNode = nodeMapIter.next(htkNode)) {
	    if (*pfsgNode == initial) {
		HTKinitial = htkNode;
		initialinfo = &nodeInfoMap[HTKinitial];
		break;
	    }
	}
    }
    initialNode->word = vocab.ssIndex();

    // attach HTK initial node info to lattice initial node
    if (initialinfo) {
	initialNode->htkinfo = new HTKWordInfo(*initialinfo);
	assert(initialNode->htkinfo != 0);
	htkinfos[htkinfos.size()] = initialNode->htkinfo;
    }

    /*
     * Set up final node
     */
    HTKWordInfo *finalinfo;
    LatticeNode *finalNode;

    if (HTKfinal != HTK_undef_uint) {
	finalinfo = &nodeInfoMap[HTKfinal];
	NodeIndex *finalPtr = nodeMap.find(HTKfinal);
	if (finalPtr) {
	    final = *finalPtr;
	    finalNode = findNode(final);
	} else {
	    file.position() << "undefined end node " << HTKfinal << endl;
	    if (!useNullNodes) vocab.remove(HTKNodeDummy);
	    return false;
	}
    } else {
	// search for end node: the one without outgoing transitions
	LHashIter<NodeIndex, LatticeNode> nodeIter(nodes);
	NodeIndex nodeIndex;

	finalNode = 0;
	while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	    if (node->outTransitions.numEntries() == 0) {
		final = nodeIndex;
		finalNode = node;
		break;
	    }
	}

	if (!finalNode) {
	    file.position() << "could not find final node\n";
	    if (!useNullNodes) vocab.remove(HTKNodeDummy);
	    return false;
	}

	// now find the HTK node info associated with final node
	LHashIter<unsigned, NodeIndex> nodeMapIter(nodeMap);
	unsigned htkNode;
	while (NodeIndex *pfsgNode = nodeMapIter.next(htkNode)) {
	    if (*pfsgNode == final) {
		HTKfinal = htkNode;
		finalinfo = &nodeInfoMap[HTKfinal];
		break;
	    }
	}
    }
    finalNode->word = vocab.seIndex();

    // attach HTK final node info to lattice final node
    if (finalinfo) {
	finalNode->htkinfo = new HTKWordInfo(*finalinfo);
	assert(finalNode->htkinfo != 0);
	htkinfos[htkinfos.size()] = finalNode->htkinfo;
    }

    // eliminate dummy nodes 
    if (!useNullNodes) {
	removeAllXNodes(HTKNodeDummy);
	vocab.remove(HTKNodeDummy);
    }

    return true;
}

/*
 * allowAsTrans()
 *  Determine if this node has the appropriate properties to be printed 
 *  as just an HTK transition.  The purpose is to allow a more compact printing.
 *  If an internal lattice node meets these characteristics then it need not 
 *  be printed as an HTK node, and it's transitions can also be saved from
 *  printing because the node itself will be printed as the transition.
 * 
 *  Check for these properties:
 *    - must have one outgoing transition
 *    - must have one incoming transition
 *    - nodes on each side must be HTK null nodes
 *    - don't allow as a transition if an adjacent node could also be allowed
 *      (this last case should allow for the collapsing of these two nodes,
 *	but it's not impl.)
 */
static Boolean
allowAsTrans(Lattice &lat, NodeIndex nodeIndex)
{
    LatticeNode *node = lat.findNode(nodeIndex);
    assert(node != 0);
  
    if (node->inTransitions.numEntries() == 1 &&
	node->outTransitions.numEntries() == 1)
    {
	TRANSITER_T<NodeIndex,LatticeTransition>
	    outTransIter(node->outTransitions);
	NodeIndex next;
	while (outTransIter.next(next)) {
	    LatticeNode *nextNode = lat.findNode(next); 
	    assert(nextNode != 0);
	    // check if next node is a NULL (the final node also acts as one)
	    if (nextNode->word == Vocab_None || next == lat.getFinal()) {
		TRANSITER_T<NodeIndex,LatticeTransition> 
		    inTransIter(node->inTransitions);
		NodeIndex prev;
		while (inTransIter.next(prev)) {
		    LatticeNode *prevNode = lat.findNode(prev); 
		    assert(prevNode != 0);

		    // check if prev node is a NULL
		    // (the inital node also acts as one)
		    if (prevNode->word == Vocab_None ||
			prev == lat.getInitial())
		    {
			// check if next node would be allowed as a transition
			if (allowAsTrans(lat, next)) {
			    // if we have a string of two nodes that could be
			    // transitions, that should be redundant and allow
			    // them to be combined, for now we'll just treat
			    // it as a case that returns false because two
			    // adjacent transitions is a problem (a node
			    // between them is required) so the first node
			    // will not be allowed as a transition
			    return false;
			} else {
			    return true;
			}
		    }
		}
	    }
	}
    }
    return false;
}

static double
scaleHTKScore(double score, double logscale)
{
    if (logscale == 0.0) {
	return LogPtoProb(score);
    } else {
	return score * logscale;
    }
}

// Use "stdio" functions in File() object to allow writing in-memory to File() string object.
static void
writeScoreInfo(File &file, HTKWordInfo &htkinfo, HTKScoreMapping scoreMapping,
								double logscale)
{
    if (scoreMapping != mapHTKacoustic &&
	  htkinfo.acoustic != HTK_undef_float)
    {
	file.fprintf("\ta=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.acoustic, logscale));
    }
    if (htkinfo.pron != HTK_undef_float)
    {
	file.fprintf("\tr=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.pron, logscale));
    }
    if (htkinfo.duration != HTK_undef_float)
    {
	file.fprintf("\tds=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.duration, logscale));
    }
    if (htkinfo.xscore1 != HTK_undef_float)
    {
	file.fprintf("\tx1=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore1, logscale));
    }
    if (htkinfo.xscore2 != HTK_undef_float)
    {
	file.fprintf("\tx2=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore2, logscale));
    }
    if (htkinfo.xscore3 != HTK_undef_float)
    {
	file.fprintf("\tx3=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore3, logscale));
    }
    if (htkinfo.xscore4 != HTK_undef_float)
    {
	file.fprintf("\tx4=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore4, logscale));
    }
    if (htkinfo.xscore5 != HTK_undef_float)
    {
	file.fprintf("\tx5=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore5, logscale));
    }
    if (htkinfo.xscore6 != HTK_undef_float)
    {
	file.fprintf("\tx6=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore6, logscale));
    }
    if (htkinfo.xscore7 != HTK_undef_float)
    {
	file.fprintf("\tx7=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore7, logscale));
    }
    if (htkinfo.xscore8 != HTK_undef_float)
    {
	file.fprintf("\tx8=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore8, logscale));
    }
    if (htkinfo.xscore9 != HTK_undef_float)
    {
	file.fprintf("\tx9=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.xscore9, logscale));
    }
}

// Use "stdio" functions in File() object to allow writing in-memory to File() string object.
static void
writeWordInfo(File &file, HTKWordInfo &htkinfo)
{
    if (htkinfo.var != HTK_undef_uint)
    {
	file.fprintf("\tv=%u", htkinfo.var);
    }
    if (htkinfo.div != 0)
    {
	file.fprintf("\td=%s", htkinfo.div);
    }
    if (htkinfo.states != 0)
    {
	file.fprintf("\ts=%s", htkinfo.states);
    }
}

/*
 * Output lattice in HTK format
 *	Algorithm:
 *	- each lattice node becomes an HTK node
 *        (unless it has only one incoming and one outgoing transition,
 *	  both to HTK null nodes)
 *	- each lattice transitions becomes an HTK link.
 *        (unless it exists on a node which has been flagged to print as a
 *	  transition, then the links are ignored)
 *	- word information is added to the HTK nodes.
 *        (all words and scores must be printed on the HTK transitions,
 *	  not on the words)
 *	- link information attached to each node is added to the HTK link
 *	  leading into the node.
 *	- lattice transition weights are mapped to one of the
 *	  HTK score fields as indicated by the second argument.
 */
// Use "stdio" functions in File() object to allow writing in-memory to File() string object.
Boolean
Lattice::writeHTK(File &file, HTKScoreMapping scoreMapping,
                  Boolean printPosteriors)
{
    if (debug(DebugPrintFunctionality)) {
	dout()  << "Lattice::writeHTK: writing ";
    }

    file.fprintf("# Header (generated by SRILM)\n");
    file.fprintf("VERSION=%s\n", HTKLattice_Version);
    file.fprintf("UTTERANCE="); printQuoted(file, name, htkheader.useQuotes);
    file.fputc('\n');
    file.fprintf("base=%.*lg\n", dbl_prec, htkheader.logbase);
    file.fprintf("dir=%s\n", "f");		// forward lattice

    double logscale = 1.0 / ProbToLogP(htkheader.logbase);

    /* 
     * Ancillary header information preserved from readHTK()
     */
    if (htkheader.tscale != HTK_def_tscale) {
	file.fprintf("tscale=%.*lg\n", dbl_prec, htkheader.tscale);
    }
    if (htkheader.acscale != HTK_def_acscale) {
	file.fprintf("acscale=%.*lg\n", dbl_prec, htkheader.acscale);
    }
    if (htkheader.lmscale != HTK_def_lmscale) {
	file.fprintf("lmscale=%.*lg\n", dbl_prec, htkheader.lmscale);
    }
    if (htkheader.ngscale != HTK_def_ngscale) {
	file.fprintf("ngscale=%.*lg\n", dbl_prec, htkheader.ngscale);
    }
    if (htkheader.prscale != HTK_def_prscale) {
	file.fprintf("prscale=%.*lg\n", dbl_prec, htkheader.prscale);
    }
    if (htkheader.wdpenalty != HTK_def_wdpenalty) {
	file.fprintf("wdpenalty=%.*lg\n", LogP_Precision, scaleHTKScore(htkheader.wdpenalty, logscale));
    }
    if (htkheader.duscale != HTK_def_duscale) {
	file.fprintf("duscale=%.*lg\n", dbl_prec, htkheader.duscale);
    }
    if (htkheader.x1scale != HTK_def_xscale) {
	file.fprintf("x1scale=%.*lg\n", dbl_prec, htkheader.x1scale);
    }
    if (htkheader.x2scale != HTK_def_xscale) {
	file.fprintf("x2scale=%.*lg\n", dbl_prec, htkheader.x2scale);
    }
    if (htkheader.x3scale != HTK_def_xscale) {
	file.fprintf("x3scale=%.*lg\n", dbl_prec, htkheader.x3scale);
    }
    if (htkheader.x4scale != HTK_def_xscale) {
	file.fprintf("x4scale=%.*lg\n", dbl_prec, htkheader.x4scale);
    }
    if (htkheader.x5scale != HTK_def_xscale) {
	file.fprintf("x5scale=%.*lg\n", dbl_prec, htkheader.x5scale);
    }
    if (htkheader.x6scale != HTK_def_xscale) {
	file.fprintf("x6scale=%.*lg\n", dbl_prec, htkheader.x6scale);
    }
    if (htkheader.x7scale != HTK_def_xscale) {
	file.fprintf("x7scale=%.*lg\n", dbl_prec, htkheader.x7scale);
    }
    if (htkheader.x8scale != HTK_def_xscale) {
	file.fprintf("x8scale=%.*lg\n", dbl_prec, htkheader.x8scale);
    }
    if (htkheader.x9scale != HTK_def_xscale) {
	file.fprintf("x9scale=%.*lg\n", dbl_prec, htkheader.x9scale);
    }
    if (htkheader.amscale != HTK_undef_float && printPosteriors) {
	file.fprintf("amscale=%.*lg\n", dbl_prec, htkheader.amscale);
    }
    if (htkheader.hmms != 0) {
	file.fprintf("hmms=");
	printQuoted(file, htkheader.hmms, htkheader.useQuotes);
	file.fputc('\n');
    }
    if (htkheader.lmname != 0) {
	file.fprintf("lmname=");
	printQuoted(file, htkheader.lmname, htkheader.useQuotes);
	file.fputc('\n');
    }
    if (htkheader.ngname != 0) {
	file.fprintf("ngname=");
	printQuoted(file, htkheader.ngname, htkheader.useQuotes);
	file.fputc('\n');
    }
    if (htkheader.vocab != 0) {
	file.fprintf("vocab=");
	printQuoted(file, htkheader.vocab, htkheader.useQuotes);
	file.fputc('\n');
    }
	
    /*
     * We remap the internal node indices to consecutive unsigned integers
     * to allow a compact output representation.
     * We iterate over all nodes, renumbering them, and also counting the
     * number of transitions overall.
     * (Nodes which can be treated as transitions are not added as nodes.)
     */
    LHash<NodeIndex,unsigned> nodeMap;		// map nodeIndex to unsigned
    LHash<NodeIndex,Boolean> treatNodeAsTrans;	// keep a hash of nodes to be 
                                                //  printed only as transitions
    unsigned numNodes = 0;
    unsigned numTransitions = 0;

    LHashIter<NodeIndex, LatticeNode> nodeIter(nodes, nodeSort);
    NodeIndex nodeIndex;

    if (!htkheader.wordsOnNodes && !htkheader.scoresOnNodes) {
	// store nodes that can be treated as transitions for future reference
	while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	    if (allowAsTrans(*this, nodeIndex)) {
		*treatNodeAsTrans.insert(nodeIndex) = true;
	    }
	}
    }

    // count number of nodes and transitions
    nodeIter.init();
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
	if (treatNodeAsTrans.find(nodeIndex)) {
	    numTransitions ++;
	} else {
	    *nodeMap.insert(nodeIndex) = numNodes ++;
	
	    NodeIndex toNodeIndex;
	    TRANSITER_T<NodeIndex,LatticeTransition>
		transIter(node->outTransitions);
	    while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
		// only count transitions here when the destination node
		// is not being treated as a transition
		if (!treatNodeAsTrans.find(toNodeIndex)) {
		    numTransitions ++;
		}
	    }
	}
    }

    if (initial != NoNode) {
	unsigned int *initialNodePtr = nodeMap.find(initial);
	file.fprintf("start=%u\n", initialNodePtr?*initialNodePtr:0);
    }
    if (final != NoNode) {
	unsigned int *nodePtr = nodeMap.find(final);
	file.fprintf("end=%u\n", nodePtr?*nodePtr:0);
    }
    file.fprintf("NODES=%u LINKS=%u\n", numNodes, numTransitions);

    if (debug(DebugPrintFunctionality)) {
      dout()  << numNodes << " nodes, "
	      << numTransitions << " transitions\n";
    }

    file.fprintf("# Nodes\n");

    nodeIter.init(); 
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
        // skip printing this node if it can be treated as just a transition
        if (treatNodeAsTrans.find(nodeIndex)) continue;

	unsigned int *nodePtr = nodeMap.find(nodeIndex);
	file.fprintf("I=%u", nodePtr?*nodePtr:0);

 	if (htkheader.wordsOnNodes) {
	    file.fprintf("\tW=");
	    printQuoted(file, ((!printSentTags &&
				    node->word == vocab.ssIndex()) ||
			       (!printSentTags &&
				    node->word == vocab.seIndex()) ||
			       node->word == Vocab_None) ?
				    HTK_null_word :
				    (node->htkinfo && node->htkinfo->wordLabel ?
					node->htkinfo->wordLabel :
					vocab.getWord(node->word)),
			htkheader.useQuotes);
	}

	if (node->htkinfo != 0) {
	    HTKWordInfo &htkinfo = *node->htkinfo;

	    if (htkinfo.time != HTK_undef_float) {
		file.fprintf("\tt=%g", htkinfo.time);
	    }
	    if (htkheader.scoresOnNodes) {
		writeScoreInfo(file, htkinfo, scoreMapping, logscale);
	    }
	    if (htkheader.wordsOnNodes) {
		writeWordInfo(file, htkinfo);
	    }
	}
	if (printPosteriors) {
	    file.fprintf("\tp=%.*lg", Prob_Precision, (double)LogPtoProb(node->posterior));
	}
	file.fprintf("\n");
    }

    file.fprintf("# Links\n");

    unsigned linkNumber = 0;
    nodeIter.init(); 
    while (LatticeNode *node = nodeIter.next(nodeIndex)) {
      // if this node can be treated as a transition, print it as one and 
      // don't print it's own transitions as HTK transitions
      if (treatNodeAsTrans.find(nodeIndex)) {
	
	// get this node's neighboring nodes
	TRANSITER_T<NodeIndex,LatticeTransition> 
	    outTransIter(node->outTransitions);
	NodeIndex nextIndex;
	outTransIter.next(nextIndex);
	LatticeNode *nextNode = findNode(nextIndex); 
	assert(nextNode != 0);
	
	TRANSITER_T<NodeIndex,LatticeTransition> 
	    inTransIter(node->inTransitions);
	NodeIndex prevIndex;
	inTransIter.next(prevIndex);
	LatticeNode *prevNode = findNode(prevIndex); 
	assert(prevNode != 0);
	
	unsigned *toNodeId = nodeMap.find(nextIndex); 
	assert(toNodeId != 0);

	unsigned *fromNodeId = nodeMap.find(prevIndex); 
	assert(fromNodeId != 0);
	
	file.fprintf("J=%u\tS=%u\tE=%u", linkNumber++, *fromNodeId, *toNodeId);
	
	if (!htkheader.wordsOnNodes) {
	    file.fprintf("\tW=");
	    printQuoted(file, ((!printSentTags &&
				    node->word == vocab.ssIndex()) ||
			       (!printSentTags &&
				    node->word == vocab.seIndex()) ||
			       node->word == Vocab_None) ?
			        HTK_null_word :
				(node->htkinfo && node->htkinfo->wordLabel ?
				     node->htkinfo->wordLabel :
				     vocab.getWord(node->word)),
			htkheader.useQuotes);
	}
	
	if (node->htkinfo != 0) {
	    HTKWordInfo &htkinfo = *node->htkinfo;

	    writeScoreInfo(file, htkinfo, scoreMapping, logscale);
	    writeWordInfo(file, htkinfo);

	    if (scoreMapping != mapHTKngram &&
		htkinfo.ngram != HTK_undef_float)
	    {
		file.fprintf("\tn=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.ngram, logscale));
	    }
	    if (scoreMapping != mapHTKlanguage &&
		htkinfo.language != HTK_undef_float)
	    {
		file.fprintf("\tl=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.language, logscale));
	    }
	}
	
	/*
	 * map transition weight to one of the standard HTK scores
	 */
	if (scoreMapping != mapHTKnone) {
	    LatticeTransition *thisTrans =
				node->outTransitions.find(nextIndex);
	    assert(thisTrans != 0);
	    LatticeTransition *prevTrans =
				prevNode->outTransitions.find(nodeIndex);
	    assert(prevTrans != 0);

	    LogP combinedWeight = thisTrans->weight + 
				  prevTrans->weight;

	    file.fprintf("\t%c=%.*lg",
		    (scoreMapping == mapHTKacoustic ? 'a' :
		     (scoreMapping == mapHTKngram ? 'n' :
		      (scoreMapping == mapHTKlanguage ? 'l' : '?'))),
		    LogP_Precision, scaleHTKScore(combinedWeight, logscale));
	}

	if (printPosteriors) {
	    file.fprintf("\tp=%.*lg",
				Prob_Precision, (double)LogPtoProb(node->posterior));
	}
	
	file.fprintf("\n");
      } else {
        // treat this node in the normal sense if it can't be treated solely
	// as a trans (but we have to ignore transitions to nodes that were
	// printed only as transitions)
	unsigned *fromNodeId = nodeMap.find(nodeIndex);

 	NodeIndex toNodeIndex;

	TRANSITER_T<NodeIndex,LatticeTransition>
					transIter(node->outTransitions);
	while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
	    // skip printing this transition if the destination node is being
	    // treated as a transition (the transition weight is taken care of
	    // in printing of transition node case)
	    if (treatNodeAsTrans.find(toNodeIndex)) continue;

	    LatticeNode *toNode = findNode(toNodeIndex);
	    assert(toNode != 0);

	    unsigned *toNodeId = nodeMap.find(toNodeIndex); 
	    assert(toNodeId != 0);

	    file.fprintf("J=%u\tS=%u\tE=%u",
			 linkNumber++, fromNodeId?*fromNodeId:0, *toNodeId);

	    if (!htkheader.wordsOnNodes) {
		file.fprintf("\tW=");
		printQuoted(file, ((!printSentTags &&
					toNode->word == vocab.ssIndex()) ||
				   (!printSentTags &&
					toNode->word == vocab.seIndex()) ||
				   toNode->word == Vocab_None) ?
				    HTK_null_word :
				    (toNode->htkinfo && toNode->htkinfo->wordLabel ?
					toNode->htkinfo->wordLabel :
					vocab.getWord(toNode->word)),
			    htkheader.useQuotes);
	    }

	    if (toNode->htkinfo != 0) {
		HTKWordInfo &htkinfo = *toNode->htkinfo;

		if (!htkheader.scoresOnNodes) {
		    writeScoreInfo(file, htkinfo, scoreMapping, logscale);
		}
		if (!htkheader.wordsOnNodes){
		    writeWordInfo(file, htkinfo);
		}

		if (scoreMapping != mapHTKngram &&
		    htkinfo.ngram != HTK_undef_float)
		{
		    file.fprintf("\tn=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.ngram, logscale));
		}
		if (scoreMapping != mapHTKlanguage &&
		    htkinfo.language != HTK_undef_float)
		{
		    file.fprintf("\tl=%.*lg", LogP_Precision, scaleHTKScore(htkinfo.language, logscale));
		}
	    }

	    /*
	     * map transition weight to one of the standard HTK scores
	     */
	    if (scoreMapping != mapHTKnone) {
		file.fprintf("\t%c=%.*lg",
			    (scoreMapping == mapHTKacoustic ? 'a' :
			     (scoreMapping == mapHTKngram ? 'n' :
			      (scoreMapping == mapHTKlanguage ? 'l' : '?'))),
			    LogP_Precision, scaleHTKScore(trans->weight, logscale));
	    }

	    file.fprintf("\n");
	  }
      }
    }

    return true;
}


/* 
 * Compute pronunciation scores
 * 	(for nodes with HTKWordInfo that have phone backtraces)
 */
Boolean
Lattice::scorePronunciations(VocabMultiMap &dictionary, Boolean intlogs)
{
    if (debug(DebugPrintFunctionality)) {
      dout() << "Lattice::scorePronunciations: starting\n";
    }

    Vocab &phoneVocab = dictionary.vocab2;

    /*
     * Go through all HTKWordInfo structures, extract the phone sequences,
     * and look up their probabilities in the dictionary
     */
    for (unsigned i = 0; i < htkinfos.size(); i ++) {
	HTKWordInfo *info = htkinfos[i];

	/*
	 * only rescore words that have pronunciations and a non-NULL
	 */
	if (info->div != 0) {
	    if (info->word == Vocab_None) {
		dout() << "Lattice::scorePronunciations: warning: " << name
		       << " has pronunciation on " << HTK_null_word << " node"
		       << " (time = " << info->time << ")\n";

		info->pron = LogP_Zero;
	    } else {
		/*
		 * parse the phone sequence from the string
		 * example:
		 * d=:#[s]t,0.12:s[t]r,0.03:t[r]ay,0.05:r[ay]k,0.09:ay[k]#,0.09:
		 * and convert into an index string
		 */
		makeArray(char, phoneString, strlen(info->div) + 1);
		strcpy(phoneString, info->div);

		Array<VocabIndex> phones;
		unsigned numPhones = 0;
		char *strtok_ptr = NULL;
    
		for (char *s = MStringTokUtil::strtok_r(phoneString, phoneSeparator, &strtok_ptr);
		     s != 0;
		     s = MStringTokUtil::strtok_r(NULL, phoneSeparator, &strtok_ptr))
		{
		    // skip empty components (at beginning and end)
		    if (s[0] == '\0') continue;

		    // strip duration part
		    char *e = strchr(s, ',');
		    if (e != 0) *e = '\0';

		    // strip context from triphone labels
		    e = strchr(s, '[');
		    if (e != 0) s = e + 1;

		    e = strrchr(s, ']');
		    if (e != 0) *e = '\0';

		    phones[numPhones ++] = phoneVocab.addWord(s);
		}
		phones[numPhones] = Vocab_None;

		// find pronunciation prob
		Prob p = dictionary.get(info->word, phones.data());

		if (p == 0.0) {
		    // missing pronunciation get score 0
		    info->pron = LogP_One;
		} else {
		    if (intlogs) {
			info->pron = IntlogToLogP(p);
		    } else {
			info->pron = ProbToLogP(p);
		    }
		}
	    }
	}
    }

    return true;
}

/*
 * Take recognizer phones and split into two arrays
 *  one has each phone as an element, the other
 *  has each phone duration as an element
 */
static int
splitPhones(const char *phoneString, Array<char *> &phones,
					Array<NBestTimestamp> &phoneDurs)
{
    unsigned numPhones = HTK_undef_uint;

    if (phoneString != 0) {
	const char *index = phoneString;
	unsigned phoneLen = strlen(phoneString);

	while (index[0] != '\0') {
	    makeArray(char, tmp, phoneLen+1);
	    unsigned tmpIndex;

	    if (index[0] == *phoneSeparator) {
		// phone divider
		if (numPhones != HTK_undef_uint) {
		    numPhones++;
		} else {
		    numPhones = 0;
		}
		index += 1;
	    } else if (index[0] == ',') {
		// phone duration
		index += 1; // skip the ','
		tmpIndex = 0;
		while (index[0] != *phoneSeparator) { // read in phone duration
		    tmp[tmpIndex++] = index[0];
		    index += 1;
		}
		tmp[tmpIndex] = '\0';
		phoneDurs[numPhones] = atof(tmp) / frameLength;
	    } else {
		// should be start of phone
		if (index[0] == '-' && index[1] == ',') {
		    // process '-'
		    // (know it's a pause if comma directly follows dash)
		    tmpIndex = 0;
		    tmp[tmpIndex++] = index[0];
		    tmp[tmpIndex] = '\0';
		    index += 1;
		} else {
		    // process other phones
		    const char *startPhone = index;
		    while (index[0] != '[' && index[0] != ',') {
			index += 1;
		    }
		    if (index[0] == ',') {
			// didn't find the phone in brackets like expected 
			tmpIndex = 0;
			for (const char *i = startPhone; i < index; i ++) {
			    tmp[tmpIndex++] = *i;
			}
			tmp[tmpIndex] = '\0';
			if (strcmp(tmp,"rej") != 0) {
			    // Force phones that are not 'rej' to be treated
			    // like a rej if their format is bad
			    // change this to just a warning
			    cerr << "splitPhones: Unexpected phone format: " 
				 << tmp << endl;
			}
		    } else {
			// normal case 
			index += 1; // skip the '['
			tmpIndex = 0;
			while (index[0] != ']') { // read in phone
			    tmp[tmpIndex++] = index[0];
			    index += 1;
			}
			tmp[tmpIndex] = '\0';
		    }
		}
		phones[numPhones] = strdup(tmp);
		assert(phones[numPhones] != 0);
		// advance to phone duration
		while (index[0] != ',') {
		    index += 1;
		}
	    }
	}
    }
    return (numPhones);
}

/*
 * Split multiwords
 * (different than normal because phones are distributed across new nodes
 */
void
Lattice::splitHTKMultiwordNodes(MultiwordVocab &vocab,
		LHash<const char *, Array< Array<char *> * > > &multiwordDict)
{
    if (debug(DebugPrintFunctionality)) {
	dout() << "Lattice::splitHTKMultiwordNodes:"
	       << " splitting multiword nodes\n";
    }

    unsigned numNodes = getNumNodes(); 

    NodeIndex *sortedNodes = new NodeIndex[numNodes];
    assert(sortedNodes != 0);
    unsigned numReachable = sortNodes(sortedNodes);

    for (unsigned i = 0; i < numReachable; i++) {
	NodeIndex nodeIndex = sortedNodes[i];
	LatticeNode *node = findNode(nodeIndex); 
	assert(node != 0);

	Boolean foundSplit = true;

	VocabIndex oneWord[2];
	oneWord[0] = node->word;
	oneWord[1] = Vocab_None;

	NBestTimestamp origTime = HTK_undef_float;
	char *origPron = 0;
	VocabIndex origWord = node->word;
	VocabIndex expanded[maxWordsPerLine + 1];

	unsigned expandedLength =
		  vocab.expandMultiwords(oneWord, expanded, maxWordsPerLine);

	if (expandedLength > 1) {
	    NodeIndex prevNodeIndex = nodeIndex;
	    NodeIndex firstNewIndex = HTK_undef_uint;
	    NodeIndex firstNullIndex = HTK_undef_uint;

	    TRANSITER_T<NodeIndex,LatticeTransition>
	    				transIterIn(node->inTransitions);

	    NodeIndex fromNodeIndex;
	    if (transIterIn.next(fromNodeIndex) == 0) {
		dout() << "Lattice::splitHTKMultiwordNodes:"
		       << " no predecessor for multiword node "
		       << getWord(origWord) << endl;
		continue;
	    }
	    	
	    LatticeNode *fromNode = nodes.find(fromNodeIndex);
	    assert(fromNode != 0);

	    // we just take the first node, because all incoming nodes have
	    // the same end time, which is what we are looking to grab
	    // (there should only be one, the !NULL node that we came from)

	    if (fromNode->htkinfo == 0) {
		dout() << "Lattice::splitHTKMultiwordNodes:"
		       << " no HTK info on multiword node "
		       << getWord(origWord) << endl;
		continue;
	    }

	    HTKWordInfo *myHtk = node->htkinfo;
	    origTime = myHtk->time;
	    origPron = myHtk->div;

	    if (origPron == 0) {
		dout() << "Lattice::splitHTKMultiwordNodes:"
		       << " no pronunciation on multiword node "
		       << getWord(origWord) << endl;
		continue;
	    }

	    NBestTimestamp multiStart = fromNode->htkinfo->time;

	    unsigned divLen = strlen(myHtk->div);
	    makeArray(char, prePhone, divLen+1);
	    makeArray(char, postPhone, divLen+1);

	    Array<char *> phones;
	    Array<NBestTimestamp> phoneDurs;
	    splitPhones(myHtk->div, phones, phoneDurs);

	    // split the multiword into individual words
	    
	    Array< Array<char *> * > *pronunciations =
				    multiwordDict.find(getWord(origWord));
	    
	    if (pronunciations == 0) {
		dout() << "Lattice::splitHTKMultiwordNodes:"
		       << " no multiword pronunciations for "
		       << getWord(origWord) << endl;
		continue;
	    }

	    Array<NBestTimestamp> *thisPhonePtr;
	    
	    // go through the possible pronunciations of this word
	    for (unsigned n = 0; n < pronunciations->size(); n ++) {
		Array<char *> *pronunciation = (*pronunciations)[n];
		Array<unsigned> wordBoundaries;
		Array<NBestTimestamp> wordTimes;

		unsigned l = 0;
		for (unsigned m = 0; m < phones.size(); m ++) {
		    char *thisPhone = phones[m];

		    if (l < pronunciation->size() &&
			strcmp((*pronunciation)[l], thisPhone) == 0)
		    {
			// right phone
			l ++;
			
			// look for multiword boundary
			if (l < pronunciation->size() &&
			    strcmp((*pronunciation)[l], "|") == 0)
			{
			    wordBoundaries[wordBoundaries.size()] = m;
			    l ++;		    
			}
			
			// quit if we've found a complete match
			if (m + 1 == phones.size()) {
			    // add final boundary
			    wordBoundaries[wordBoundaries.size()] = m;
			    n = pronunciations->size();		

			    if (wordBoundaries.size() < expandedLength) {
				cerr << "Lattice::splitHTKMultiwordNodes: "
				     << " found more words than existed in"
				     << " multiword split, dropping word "
				     << wordBoundaries.size()
				     << " " << expandedLength << endl;
				expandedLength = wordBoundaries.size();
			    }

			    // change info and add nodes
			    unsigned i;
			    for (i = 0; i < expandedLength; i ++) {
				if (i == 0) {
				    // don't need a new node,
				    // just change info
				    node->word = myHtk->word = expanded[i];

				    // get the preceding phone
				    int n = 0;
				    char *index = myHtk->div;
				    Array<char> tmp;
				    unsigned tmpIndex;
				    while (index[0] != '\0' && index[0] !='[')
				    {
					prePhone[n++] = index[0];
					index++;
				    }
				    prePhone[n] = '\0';

				    // get the transition phone
				    // to next word
				    n = 0;
				    index = strrchr(myHtk->div,']') + 1;
				    if (index) {
				        while ((index[0] != ',') && (index[0] != 0)) {
					    postPhone[n++] = index[0];
					    index++;
					}
				    }
				    postPhone[n] = '\0';

				    assert(i < wordBoundaries.size());

				    // build new phone string for
				    // split multiword
				    char tmpDiv[512];	
				    unsigned c;

				    for (unsigned k = 0;
					 k <= wordBoundaries[i];
					 k ++)
				    {
					if (k == 0) {
					    // use prePhone to start
					    c = sprintf(tmpDiv,
						  "%s[%s]%s,%01.2f:",
						  (char *)prePhone,
						  (char *)phones[k],
						  (char *)phones[k+1],
						  phoneDurs[k] * frameLength);
					    assert(c < sizeof(tmpDiv));
					    wordTimes[i] =
						  multiStart +
						  phoneDurs[k] * frameLength;
					}
					else if (k > 0) {
					    c = snprintf(tmpDiv,
						  sizeof(tmpDiv),
						  "%s%s[%s]%s,%01.2f:",
						  tmpDiv,
						  (char *)phones[k-1],
						  (char *)phones[k],
						  (char *)phones[k+1],
						  phoneDurs[k] * frameLength);
					    assert(c < sizeof(tmpDiv));
					    wordTimes[i] +=
						  phoneDurs[k] * frameLength;
					}
				    }

				    // @kw false positive: MLK.MIGHT (myHtk->div)
				    myHtk->div = strdup(tmpDiv);
				    assert(myHtk->div != 0);
				    myHtk->time = wordTimes[i];
				} else {
				    // add node and attach info
			    
				    // create new nodes for all
				    // subsequent word components, and
				    // string them together with zero
				    // weight transitions
				    HTKWordInfo *newinfo = 0;	
				    newinfo = new HTKWordInfo;
				    assert(newinfo != 0);
				      
				    htkinfos[htkinfos.size()] = newinfo;
				    newinfo->word = expanded[i];	

				    // build new phone string for
				    // split multiword
				    char tmpDiv[512];	
				    unsigned c;
				    sprintf(tmpDiv,":");

				    for (unsigned k =
						wordBoundaries[i-1]+1;
					 k <= wordBoundaries[i];
					 k ++)
				    {
					if (k == phones.size()-1) {
					    // use post phone to finish
					    c = sprintf(tmpDiv,
						  "%s%s[%s]%s,%01.2f:",
						  tmpDiv,
						  (char *)phones[k-1],
						  (char *)phones[k],
						  (char *)postPhone,
						  phoneDurs[k] * frameLength);
					    assert(c < sizeof(tmpDiv));
					} else {
					    c = sprintf(tmpDiv,
						  "%s%s[%s]%s,%01.2f:",
						  tmpDiv,
						  (char *)phones[k-1],
						  (char *)phones[k],
						  (char *)phones[k+1],
						  phoneDurs[k] * frameLength);
					    assert(c < sizeof(tmpDiv));
					}

					if (k == wordBoundaries[i-1]+1) {
					    wordTimes[i] =
						wordTimes[i-1] +
						phoneDurs[k] * frameLength;
					} else {
					    wordTimes[i] +=
						phoneDurs[k] * frameLength;
					}
				    }
				    newinfo->div = strdup(tmpDiv);
				    assert(newinfo->div != 0);
				    
				    // update the time with start
				    // of intermediate word
				    newinfo->time = wordTimes[i];
						    // need to add difference

				    // scores of all subsequent
				    // components are 0 since the first
				    // component carries the full scores
				    if (myHtk->acoustic != HTK_undef_float) {
					newinfo->acoustic = LogP_One;
				    }
				    if (myHtk->language != HTK_undef_float) {
					newinfo->language = LogP_One;
				    }
				    if (myHtk->ngram != HTK_undef_float) {
					newinfo->ngram = LogP_One;
				    }
				    if (myHtk->pron != HTK_undef_float) {
					newinfo->pron = LogP_One;
				    }
				   
				    NodeIndex newNodeIndex =
					      dupNode(expanded[i], 0, newinfo);

				    // We have null nodes between each
				    // word with times on them, so
				    // insert a new null node here
				    NodeIndex NullIndex = HTK_undef_uint;
				    HTKWordInfo *NullNode = new HTKWordInfo;
				    htkinfos[htkinfos.size()] = NullNode;

				    // NullNode should have the time
				    // from the previous node
				    assert(i > 0);
				    NullNode->time = wordTimes[i-1];
				    NullIndex = dupNode(Vocab_None,
							0, NullNode);

				    // delay inserting the first new
				    // transition to not interfere
				    // with removal of old links below
				    if (prevNodeIndex == nodeIndex) {
					firstNewIndex = NullIndex;
					LatticeTransition trans;
					insertTrans(NullIndex,
						    newNodeIndex,
						    trans);
				    } else {
					LatticeTransition trans;
					insertTrans(prevNodeIndex,
						    NullIndex,
						    trans);
					insertTrans(NullIndex,
						    newNodeIndex,
						    trans);
				    }
				    prevNodeIndex = newNodeIndex;
				}
			    }
			    if (i != wordBoundaries.size()) {
				if (debug(DebugPrintFunctionality)) {
				    dout() << "Lattice::splitHTKMultiwordNodes:"
				    	   << " found different number of"
				    	   << " boundaries than expected words"
					   << endl;
				}
			    }
			}
		    } else {
			// phone doesn't match
			if (n + 1 == pronunciations->size()) {
			    dout() << "Lattice::splitHTKMultiwordNodes:"
				   << " no multiword pronunciation for "
				   << getWord(origWord) << " " << origPron
				   << endl;
			    foundSplit = false;
			}
			// don't explore this path because it already
			// doesn't match
			m = phones.size(); 
		    }
		}
	    }

	    for (unsigned i = 0; i < phones.size(); i ++) {
		free(phones[i]);
	    }

	    // Don't change anything if we didn't find a split
	    if (foundSplit == false) {
	    	// restore node information already changed
	    	node->word = myHtk->word = origWord;
	    	myHtk->time = origTime;
	    	myHtk->div = origPron;
	    	continue;
	    }

	    free(origPron);

	    // node may have moved since others were added!!!
	    node = findNode(nodeIndex); 
	    assert(node != 0);
	    
	    // copy original outgoing transitions onto final new node
	    TRANSITER_T<NodeIndex,LatticeTransition>
	    				transIter(node->outTransitions);
	    NodeIndex toNodeIndex;
	    while (LatticeTransition *trans = transIter.next(toNodeIndex)) {
		// prevNodeIndex still has the last of the newly created nodes
		insertTrans(prevNodeIndex, toNodeIndex, *trans);
	    }
	    
	    // now insert new transition out of original node
	    LatticeTransition trans;
	    assert(firstNewIndex != HTK_undef_uint);
	    insertTrans(nodeIndex, firstNewIndex, trans);
	    removeTrans(nodeIndex,toNodeIndex);
	}
    }
    
    delete [] sortedNodes;
}

Boolean
Lattice::readMultiwordDict(File &file,
		LHash<const char *, Array< Array<char *> * > > &multiwordDict)
{
    while (char *line = file.getline()) {
	char *index = line;

	// first read in the phone name
	Array<char> tmp;
	unsigned tmpIndex = 0;
	while (*index != '\0' && !isspace(*index)) {
	    tmp[tmpIndex++] = *index;
	    index++;
	}
	tmp[tmpIndex] = '\0';	// tmp now hold current word string

	Array<char *> *thisWordPtr = new Array<char *>;
						// array to store pron. data
	Array< Array<char *> *> *thisWordProns = multiwordDict.insert(tmp);

	if (thisWordProns != 0) {
	    // already exists, so add this pron. to the next slot in the array
	    (*thisWordProns)[thisWordProns->size()] = thisWordPtr;
	} else {
	    // doesn't exist, so put this pron. in the first slot of the array
	    (*thisWordProns)[0] = thisWordPtr;
	}

	while (isspace(*index)) index++; // skip spaces

	// read in the following pronunciation data
	while (*index != '\0') {
	    tmpIndex = 0;
	    while (*index != '\0' && !isspace(*index)) {
		tmp[tmpIndex++] = *index;
		index++;
	    }
	    tmp[tmpIndex] = '\0';
	    char *phone = strdup(tmp);
	    assert(phone != 0);

	    (*thisWordPtr)[thisWordPtr->size()] = phone;

	    while (isspace(*index)) index++; // skip spaces
	}
    }

    return true;
}

