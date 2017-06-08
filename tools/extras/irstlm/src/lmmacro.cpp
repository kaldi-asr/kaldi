// $Id: lmmacro.cpp 3631 2010-10-07 12:04:12Z bertoldi $

/******************************************************************************
 IrstLM: IRST Language Model Toolkit
 Copyright (C) 2006 Marcello Federico, ITC-irst Trento, Italy
 
 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.
 
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "math.h"
#include "mempool.h"
#include "htable.h"
#include "ngramcache.h"
#include "dictionary.h"
#include "n_gram.h"
#include "lmtable.h"
#include "lmmacro.h"
#include "util.h"

using namespace std;

// local utilities: start

inline void error(const char* message)
{
  cerr << message << "\n";
  throw runtime_error(message);
}

// local utilities: end


namespace irstlm {
	
	lmmacro::lmmacro(float nlf, float dlfi):lmtable(nlf,dlfi)
	{
		dict = new dictionary((char *)NULL,1000000); // dict of micro tags
		getDict()->incflag(1);
	};
	
	lmmacro::~lmmacro()
	{
		if (mapFlag) unloadmap();
	}
	
	
	void lmmacro::load(const std::string &filename,int memmap)
	{
		VERBOSE(2,"lmmacro::load(const std::string &filename,int memmap)" << std::endl);
		
		//get info from the configuration file
		fstream inp(filename.c_str(),ios::in|ios::binary);
		
		char line[MAX_LINE];
		const char* words[MAX_TOKEN_N_MAP];
		int tokenN;
		inp.getline(line,MAX_LINE,'\n');
		tokenN = parseWords(line,words,MAX_TOKEN_N_MAP);
		
		if (tokenN != 4 || ((strcmp(words[0],"LMMACRO") != 0) && (strcmp(words[0],"lmmacro")!=0)))
			error((char*)"ERROR: wrong header format of configuration file\ncorrect format: LMMACRO lmsize field [true|false]\nfilename_of_LM\nfilename_of_map (optional)");
		maxlev = atoi(words[1]);
		selectedField = atoi(words[2]);
		
		if ((strcmp(words[3],"TRUE") == 0) || (strcmp(words[3],"true") == 0))
			collapseFlag = true;
		else if ((strcmp(words[3],"FALSE") == 0) || (strcmp(words[3],"false") == 0))
			collapseFlag = false;
		else
			error((char*)"ERROR: wrong header format of configuration file\ncorrect format: LMMACRO lmsize field [true|false]\nfilename_of_LM\nfilename_of_map (optional)");
		
#ifdef DLEXICALLM
		selectedFieldForLexicon = atoi(words[3]);
		collapseFlag = atoi(words[4]);
#endif
		
		if (selectedField == -1)
			cerr << "no selected field: the whole string is used" << std::endl;
		else
			cerr << "selected field n. " << selectedField << std::endl;
		if (collapseFlag)
			cerr << "collapse is enabled" << std::endl;
		else
			cerr << "collapse is disabled" << std::endl;
		
		
		std::string lmfilename;
		if (inp.getline(line,MAX_LINE,'\n')) {
			tokenN = parseWords(line,words,MAX_TOKEN_N_MAP);
			lmfilename = words[0];
		} else
			error((char*)"ERROR: wrong format of configuration file\ncorrect format: LMMACRO lmsize field [true|false]\nfilename_of_LM\nfilename_of_map (optional)");
		
		std::string mapfilename = "";
		if (inp.getline(line,MAX_LINE,'\n')) {
			tokenN = parseWords(line,words,MAX_TOKEN_N_MAP);
			mapfilename = words[0];
			mapFlag = true;
		} else {
			mapFlag = false;
		}
		
		inp.close();
		
		
		std::cerr << "lmfilename:" << lmfilename << std::endl;
		if (mapfilename != "") {
			std::cerr << "mapfilename:" << mapfilename << std::endl;
		} else {
			std::cerr << "no mapfilename" << std::endl;
			mapFlag = false;
		}
		
		//allow the dictionary to add new words
		getDict()->incflag(1);
		
		
		if ((!mapFlag) && (collapseFlag)) {
			error((char*)"ERROR: you must specify a map if you want to collapse a specific field!");
		}
#ifdef DLEXICALLM
		
		std::string lexicalclassesfilename = words[2];
		if (lexicalclassesfilename != "NULL" && lexicalclassesfilename != "null") lexicalclassesfilename = "";
		
		if (lexicalclassesfilename != "") std::cerr << "lexicalclassesfilename:" << lexicalclassesfilename << std::endl;
		else std::cerr << "no lexicalclassesfilename" << std::endl;
		
		// Load the classes of lexicalization tokens:
		if (lexicalclassesfilename != "")    loadLexicalClasses(lexicalclassesfilename.c_str());
#endif
		
		// Load the (possibly binary) LM
		lmtable::load(lmfilename,memmap);
		
		getDict()->incflag(1);
		
		if (mapFlag)
			loadmap(mapfilename);
		getDict()->genoovcode();
		
	};
	
	void lmmacro::unloadmap()
	{
		delete dict;
		free(microMacroMap);
		if (collapseFlag) {
			free(collapsableMap);
			free(collapsatorMap);
		}
#ifdef DLEXICALLM
		free(lexicaltoken2classMap);
#endif
	}
	
	void lmmacro::loadmap(const std::string mapfilename)
	{
		microMacroMapN = 0;
		microMacroMap = NULL;
		collapsableMap = NULL;
		collapsatorMap = NULL;
		
#ifdef DLEXICALLM
		lexicaltoken2classMap = NULL;
		lexicaltoken2classMapN = 0;
#endif
		
		microMacroMap = (int *)calloc(BUFSIZ, sizeof(int));
		if (collapseFlag) {
			collapsableMap = (bool *)calloc(BUFSIZ, sizeof(bool));
			collapsatorMap = (bool *)calloc(BUFSIZ, sizeof(bool));
		}
		
		
		getDict()->genoovcode();
		microMacroMap[microMacroMapN] = lmtable::getDict()->oovcode();
		MY_ASSERT(microMacroMapN == getDict()->oovcode());
		microMacroMapN++;
		
		
		if (lmtable::getDict()->getcode(BOS_)==-1) {
			lmtable::getDict()->incflag(1);
			lmtable::getDict()->encode(BOS_);
			lmtable::getDict()->incflag(0);
		}
		
		if (lmtable::getDict()->getcode(EOS_)==-1) {
			lmtable::getDict()->incflag(1);
			lmtable::getDict()->encode(EOS_);
			lmtable::getDict()->incflag(0);
		}
		
		char line[MAX_LINE];
		const char* words[MAX_TOKEN_N_MAP];
		const char *macroW;
		const char *microW;
		int tokenN;
		bool bos=false,eos=false;
		
		// Load the dictionary of micro tags (to be put in "dict" of lmmacro class):
		inputfilestream inpMap(mapfilename.c_str());
		std::cerr << "Reading map " << mapfilename << "..." << std::endl;
		while (inpMap.getline(line,MAX_LINE,'\n')) {
			tokenN = parseWords(line,words,MAX_TOKEN_N_MAP);
			if (tokenN != 2)
				error((char*)"ERROR: wrong format of map file\n");
			microW = words[0];
			macroW = words[1];
			int microW_c=getDict()->encode(microW);
			VERBOSE(4, "microW gets the code:" << microW_c << std::endl);
			
			if (microMacroMapN>0 && !(microMacroMapN % BUFSIZ)) {
				microMacroMap = (int *)reallocf(microMacroMap, sizeof(int)*(BUFSIZ*(1+microMacroMapN/BUFSIZ)));
				if (collapseFlag) {
					//create supporting info for collapse
					
					collapsableMap = (bool *)reallocf(collapsableMap, sizeof(bool)*(BUFSIZ*(1+microMacroMapN/BUFSIZ)));
					collapsatorMap = (bool *)reallocf(collapsatorMap, sizeof(bool)*(BUFSIZ*(1+microMacroMapN/BUFSIZ)));
				}
			}
			microMacroMap[microMacroMapN] = lmtable::getDict()->getcode(macroW);
			
			if (collapseFlag) {
				
				int len = strlen(microW)-1;
				if (microW[len] == '(') {
					collapsableMap[microMacroMapN] = false;
					collapsatorMap[microMacroMapN] = true;
				} else if (microW[len] == ')') {
					collapsableMap[microMacroMapN] = true;
					collapsatorMap[microMacroMapN] = false;
				} else if (microW[len] == '+') {
					collapsableMap[microMacroMapN] = true;
					collapsatorMap[microMacroMapN] = true;
				} else {
					collapsableMap[microMacroMapN] = false;
					collapsatorMap[microMacroMapN] = false;
				}
			}
			
			if (!bos && !strcmp(microW,BOS_)) bos=true;
			if (!eos && !strcmp(microW,EOS_)) eos=true;
			
			VERBOSE(2,"\nmicroW = " << microW << "\n"
							<< "macroW = " << macroW << "\n"
							<< "microMacroMapN = " << microMacroMapN << "\n"
							<< "code of micro = " <<  getDict()->getcode(microW) << "\n"
							<< "code of macro = " <<  lmtable::getDict()->getcode(macroW) << "\n");
			
			microMacroMapN++;
		}
		
		if ((microMacroMapN == 0) && (selectedField == -1))
			error((char*)"ERROR: with no field selection, a map for the whole string is mandatory\n");
		
		if (microMacroMapN>0) {
			// Add <s>-><s> to map if missing
			if (!bos) {
				getDict()->encode(BOS_);
				if (microMacroMapN && !(microMacroMapN%BUFSIZ))
					microMacroMap = (int *)reallocf(microMacroMap, sizeof(int)*(microMacroMapN+BUFSIZ));
				microMacroMap[microMacroMapN++] = lmtable::getDict()->getcode(BOS_);
			}
			
			// Add </s>-></s> to map if missing
			if (!eos) {
				getDict()->encode(EOS_);
				if (microMacroMapN && !(microMacroMapN%BUFSIZ))
					microMacroMap = (int *)reallocf(microMacroMap, sizeof(int)*(microMacroMapN+BUFSIZ));
				microMacroMap[microMacroMapN++] = lmtable::getDict()->getcode(EOS_);
			}
		}
		//  getDict()->incflag(0);
		
		VERBOSE(2,"oovcode(micro)=" <<  getDict()->oovcode() << "\n"
						<< "oovcode(macro)=" <<  lmtable::getDict()->oovcode() << "\n"
						<< "microMacroMapN = " << microMacroMapN << "\n"
						<< "macrodictsize  = " << getDict()->size() << "\n"
						<< "microdictsize  = " << lmtable::getDict()->size() << "\n");
		
		IFVERBOSE(2) {
			for (int i=0; i<microMacroMapN; i++) {
				VERBOSE(2,"micro[" << getDict()->decode(i) << "] {"<< i << "} -> " << lmtable::getDict()->decode(microMacroMap[i]) << " {" << microMacroMap[i]<< "}" << "\n");
			}
		}
		std::cerr << "...done\n";
	}
	
	
//	double lmmacro::lprob(ngram micro_ng)
	double lmmacro::lprob(ngram micro_ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow)
	{
		VERBOSE(2,"lmmacro::lprob, parameter = <" <<  micro_ng << ">\n");
		
		ngram macro_ng(lmtable::getDict());
		
		if (micro_ng.dict ==  macro_ng.dict)
			macro_ng.trans(micro_ng);  // micro to macro mapping already done
		else
			map(&micro_ng, &macro_ng); // mapping required
		
		VERBOSE(3,"lmmacro::lprob: micro_ng = " << micro_ng << "\n"
						<<  "lmmacro::lprob: macro_ng = " << macro_ng << "\n");
		
		// ask LM with macro
		double prob;
		prob = lmtable::lprob(macro_ng, bow, bol, maxsuffidx, maxsuffptr, statesize, extendible, lastbow);
		VERBOSE(3,"prob = " << prob << "\n");
		
		return prob;
	};

	double lmmacro::clprob(ngram micro_ng, double* bow, int* bol, ngram_state_t* ngramstate, char** state,unsigned int* statesize,bool* extendible, double* lastbow)
	{
		
		VERBOSE(3," lmmacro::clprob(ngram), parameter = <" <<  micro_ng << ">\n");
		
		ngram transformed_ng(lmtable::getDict());
		bool collapsed = transform(micro_ng, transformed_ng);
		VERBOSE(3,"lmmacro::clprob(ngram), transformed_ng = <" <<  transformed_ng << ">\n");
		
		double logpr;
		
		if (collapsed) {
			// the last token of the ngram continues an already open "chunk"
			// the probability at chunk-level is not computed because it has been already computed when the actual"chunk" opens
			VERBOSE(3,"  SKIPPED call to lmtable::clprob because of collapse; logpr: 0.0\n");
			logpr = 0.0;
		} else {
			VERBOSE(3,"  QUERY MACRO LM on (after transformation and size reduction) " << transformed_ng << "\n");
			//    logpr = lmtable::clprob(transformed_ng, bow, bol, state, statesize, extendible);
			logpr = lmtable::clprob(transformed_ng, bow, bol, ngramstate, state, statesize, extendible, lastbow);
		}
		VERBOSE(3,"  GET logpr: " << logpr << "\n");
		
		return logpr;
	}
	
	bool lmmacro::transform(ngram &in, ngram &out)
	{
		VERBOSE(3,"lmmacro::transform(ngram &in, ngram &out), in = <" <<  in << ">\n");
		
		//step 1: selection of the correct field
		ngram field_ng(getDict());
		if (selectedField >= 0)
			field_selection(in, field_ng);
		else
			field_ng = in;
		
		//step 2: collapsing
		ngram collapsed_ng(getDict());
		bool collapsed = false;
		if (collapseFlag)
			collapsed = collapse(field_ng, collapsed_ng);
		else
			collapsed_ng = field_ng;
		
		//step 3: mapping using the loaded map
		if (mapFlag)
			mapping(collapsed_ng, out);
		else
			out.trans(collapsed_ng);
		
		if (out.size>lmtable::maxlevel()) out.size=lmtable::maxlevel();
		
		VERBOSE(3,"lmmacro::transform(ngram &in, ngram &out), out = <" <<  out << ">\n");
		return collapsed;
	}
	
	
	
	void lmmacro::field_selection(ngram &in, ngram &out)
	{
		VERBOSE(3,"In lmmacro::field_selection(ngram &in, ngram &out) in    = " <<  in  << "\n");
		
		int microsize = in.size;
		
		for (int i=microsize; i>0; i--) {
			
			char curr_token[BUFSIZ];
			strcpy(curr_token, getDict()->decode(*in.wordp(i)));
			char *field;
			if (strcmp(curr_token,"<s>") &&
					strcmp(curr_token,"</s>") &&
					strcmp(curr_token,"_unk_")) {
				field = strtok(curr_token, "#");
				int j=0;
				while (j<selectedField && field != NULL) {
					field = strtok(0, "#");
					j++;
				}
			} else {
				field = curr_token;
			}
			
			
			if (field) {
				out.pushw(field);
			} else {
				
				out.pushw((char*)"_unk_");
				
				// cerr << *in << "\n";
				// error((char*)"ERROR: Malformed input: selected field does not exist in token\n");
				
				/**
				 We can be here in 2 cases:
				 
				 a. effectively when the token is malformed, that is the selected
				 field does not exist
				 
				 b. in case of verbatim translation, that is the source word is
				 not known to the phrase table and moses transfers it as it is
				 to the target side: in this case, no assumption can be made on its
				 format, which means that the selected field can not exist
				 
				 The possibility of case (b) makes incorrect the error exit from
				 the code at this point: correct, on the contrary, push the _unk_ string
				 **/
			}
		}
		VERBOSE(3,"In lmmacro::field_selection(ngram &in, ngram &out) out    = " <<  out  << "\n");
		return;
	}
	
	bool lmmacro::collapse(ngram &in, ngram &out)
	{
		VERBOSE(3,"In lmmacro::collapse(ngram &in, ngram &out) in    = " <<  in  << "\n")
		
		// fill the ngram out with the collapsed tokens
		//return true if collapse happens for the most recent token
		//return false if collapse does not happen for the most recent token
		int microsize = in.size;
		out.size = 0;
		
		if (microsize == 1) {
			out.pushc(*in.wordp(1));
			return false;
		}
		
		int curr_code = *in.wordp(1);
		int prev_code = *in.wordp(2);
		
		if (microMacroMap[curr_code] == microMacroMap[prev_code]) {
			if (collapsableMap[curr_code] && collapsatorMap[prev_code]) {
				return true;
			}
		}
		
		//collapse does not happen for the most recent token
		// collapse all previous tokens, but the last
		
		prev_code = *in.wordp(microsize);
		out.pushc(prev_code);
		
		for (int i=microsize-1; i>1; i--) {
			
			curr_code = *in.wordp(i);
			
			if (microMacroMap[curr_code] != microMacroMap[prev_code]) {
				out.pushc(curr_code);
			} else {
				if (!(collapsableMap[curr_code] && collapsatorMap[prev_code])) {
					out.pushc(prev_code);
				}
			}
			prev_code = curr_code;
		}
		// and insert the most recent token
		out.pushc(*in.wordp(1));
		VERBOSE(3,"In lmmacro::collapse(ngram &in, ngram &out) out    = " <<  out  << "\n");
		return false;
	}
	
	void lmmacro::mapping(ngram &in, ngram &out)
	{
		VERBOSE(3,"In lmmacro::mapping(ngram &in, ngram &out) in    = " <<  in  << "\n");
		
		int microsize = in.size;
		
		// map microtag sequence (in) into the corresponding sequence of macrotags (possibly shorter) (out)
		
		for (int i=microsize; i>0; i--) {
			
			int in_code = *in.wordp(i);
			int out_code;
			if (in_code < microMacroMapN)
				out_code = microMacroMap[in_code];
			else
				out_code = lmtable::getDict()->oovcode();
			
			out.pushc(out_code);
		}
		VERBOSE(3,"In lmmacro::mapping(ngram &in, ngram &out) out    = " <<  out  << "\n");
		return;
	}
	
	
	//maxsuffptr returns the largest suffix of an n-gram that is contained
	//in the LM table. This can be used as a compact representation of the
	//(n-1)-gram state of a n-gram LM. if the input k-gram has k>=n then it
	//is trimmed to its n-1 suffix.
	
	const char *lmmacro::maxsuffptr(ngram micro_ng, unsigned int* size)
	{
		ngram macro_ng(lmtable::getDict());
		
		if (micro_ng.dict ==  macro_ng.dict)
			macro_ng.trans(micro_ng);  // micro to macro mapping already done
		else
			map(&micro_ng, &macro_ng); // mapping required
		
		VERBOSE(2,"lmmacro::lprob: micro_ng = " << micro_ng << "\n"
						<< "lmmacro::lprob: macro_ng = " << macro_ng << "\n");
		
		return lmtable::maxsuffptr(macro_ng,size);
	}
	
	/*
	 const char *lmmacro::cmaxsuffptr(ngram micro_ng, unsigned int* size)
	 {
	 //cerr << "lmmacro::CMAXsuffptr\n";
	 //cerr << "micro_ng: " << micro_ng
	 //	<< " -> micro_ng.size: " << micro_ng.size << "\n";
	 
	 //the LM working on the selected field = 0
	 //contributes to the LM state
	 //  if (selectedField>0)    return NULL;
	 
	 ngram macro_ng(lmtable::getDict());
	 
	 if (micro_ng.dict ==  macro_ng.dict)
	 macro_ng.trans(micro_ng);  // micro to macro mapping already done
	 else
	 map(&micro_ng, &macro_ng); // mapping required
	 
	 VERBOSE(2,"lmmacro::lprob: micro_ng = " << micro_ng << "\n"
	 <<  "lmmacro::lprob: macro_ng = " << macro_ng << "\n")
	 
	 return lmtable::cmaxsuffptr(macro_ng,size);
	 
	 }
	 */
	ngram_state_t lmmacro::maxsuffidx(ngram micro_ng, unsigned int* size)
	{
		//cerr << "lmmacro::CMAXsuffptr\n";
		//cerr << "micro_ng: " << micro_ng
		//	<< " -> micro_ng.size: " << micro_ng.size << "\n";
		
		//the LM working on the selected field = 0
		//contributes to the LM state
		//  if (selectedField>0)    return NULL;
		
		ngram macro_ng(lmtable::getDict());
		
		if (micro_ng.dict ==  macro_ng.dict)
			macro_ng.trans(micro_ng);  // micro to macro mapping already done
		else
			map(&micro_ng, &macro_ng); // mapping required
		
		VERBOSE(2,"lmmacro::lprob: micro_ng = " << micro_ng << "\n"
						<<  "lmmacro::lprob: macro_ng = " << macro_ng << "\n")
		
		return lmtable::cmaxsuffidx(macro_ng,size);
		
	}
	
	/*
	 ngram_state_t lmmacro::cmaxsuffidx(ngram micro_ng, unsigned int* size)
	 {
	 //cerr << "lmmacro::CMAXsuffptr\n";
	 //cerr << "micro_ng: " << micro_ng
	 //	<< " -> micro_ng.size: " << micro_ng.size << "\n";
	 
	 //the LM working on the selected field = 0
	 //contributes to the LM state
	 //  if (selectedField>0)    return NULL;
	 
	 ngram macro_ng(lmtable::getDict());
	 
	 if (micro_ng.dict ==  macro_ng.dict)
	 macro_ng.trans(micro_ng);  // micro to macro mapping already done
	 else
	 map(&micro_ng, &macro_ng); // mapping required
	 
	 VERBOSE(2,"lmmacro::lprob: micro_ng = " << micro_ng << "\n"
	 <<  "lmmacro::lprob: macro_ng = " << macro_ng << "\n")
	 
	 return lmtable::cmaxsuffidx(macro_ng,size);
	 
	 }
	 */
	
	void lmmacro::map(ngram *in, ngram *out)
	{
		
		VERBOSE(2,"In lmmacro::map, in = " << *in << endl
						<< " (selectedField = " << selectedField << " )\n");
		
		if (selectedField==-2) // the whole token is compatible with the LM words
			One2OneMapping(in, out);
		
		else if (selectedField==-1) // the whole token has to be mapped before querying the LM
			Micro2MacroMapping(in, out);
		
		else if (selectedField<10) { // select the field "selectedField" from tokens (separator is assumed to be "#")
			ngram field_ng(((lmmacro *)this)->getDict());
			int microsize = in->size;
			
			for (int i=microsize; i>0; i--) {
				
				char curr_token[BUFSIZ];
				strcpy(curr_token, ((lmmacro *)this)->getDict()->decode(*(in->wordp(i))));
				char *field;
				if (strcmp(curr_token,"<s>") &&
						strcmp(curr_token,"</s>") &&
						strcmp(curr_token,"_unk_")) {
					field = strtok(curr_token, "#");
					int j=0;
					while (j<selectedField && field != NULL) {
						field = strtok(0, "#");
						j++;
					}
				} else {
					field = curr_token;
				}
				
				if (field)
					field_ng.pushw(field);
				else {
					
					field_ng.pushw((char*)"_unk_");
					
					// cerr << *in << "\n";
					// error((char*)"ERROR: Malformed input: selected field does not exist in token\n");
					
					/**
					 We can be here in 2 cases:
					 
					 a. effectively when the token is malformed, that is the selected
					 field does not exist
					 
					 b. in case of verbatim translation, that is the source word is
					 not known to the phrase table and moses transfers it as it is
					 to the target side: in this case, no assumption can be made on its
					 format, which means that the selected field can not exist
					 
					 The possibility of case (b) makes incorrect the error exit from
					 the code at this point: correct, on the contrary, push the _unk_ string
					 **/
				}
			}
			if (microMacroMapN>0)
				Micro2MacroMapping(&field_ng, out);
			else
				out->trans(field_ng);
		} else {
			
#ifdef DLEXICALLM
			// selectedField>=10: tens=idx of micro tag (possibly to be mapped to
			// macro tag), unidx=idx of lemma to be concatenated by "_" to the
			// (mapped) tag
			
			int tagIdx = selectedField/10;
			int lemmaIdx = selectedField%10;
			
			// micro (or mapped to macro) sequence construction:
			ngram tag_ng(getDict());
			char *lemmas[BUFSIZ];
			
			int microsize = in->size;
			for (int i=microsize; i>0; i--) {
				char curr_token[BUFSIZ];
				strcpy(curr_token, getDict()->decode(*(in->wordp(i))));
				char *tag = NULL, *lemma = NULL;
				
				if (strcmp(curr_token,"<s>") &&
						strcmp(curr_token,"</s>") &&
						strcmp(curr_token,"_unk_")) {
					
					if (tagIdx<lemmaIdx) {
						tag = strtok(curr_token, "#");
						for (int j=0; j<tagIdx; j++)
							tag = strtok(0, "#");
						for (int j=tagIdx; j<lemmaIdx; j++)
							lemma = strtok(0, "#");
					} else {
						lemma = strtok(curr_token, "#");
						for (int j=0; j<lemmaIdx; j++)
							lemma = strtok(0, "#");
						for (int j=lemmaIdx; j<tagIdx; j++)
							tag = strtok(0, "#");
					}
					
					VERBOSE(3,"(tag,lemma) = " << tag << " " << lemma << "\n");
				} else {
					tag = curr_token;
					lemma = curr_token;
					VERBOSE(3,"(tag=lemma) = " << tag << " " << lemma << "\n");
				}
				if (tag) {
					tag_ng.pushw(tag);
					lemmas[i] = strdup(lemma);
				} else {
					tag_ng.pushw((char*)"_unk_");
					lemmas[i] = strdup("_unk_");
				}
			}
			
			if (microMacroMapN>0)
				Micro2MacroMapping(&tag_ng, out, lemmas);
			else
				out->trans(tag_ng); // qui si dovrebbero sostituire i tag con tag_lemma, senza mappatura!
			
#endif
			
		}
		
		VERBOSE(2,"In lmmacro::map, FINAL out = " << *out << endl);
	}
	
	void lmmacro::One2OneMapping(ngram *in, ngram *out)
	{
		int insize = in->size;
		
		// map each token of the sequence "in" into the same-length sequence "out" through the map
		
		for (int i=insize; i>0; i--) {
			
			int curr_code = *(in->wordp(i));
			const char *outtoken =
      lmtable::getDict()->decode((curr_code<microMacroMapN)?microMacroMap[curr_code]:lmtable::getDict()->oovcode());
			out->pushw(outtoken);
		}
		return;
	}
	
	
	void lmmacro::Micro2MacroMapping(ngram *in, ngram *out)
	{
		
		int microsize = in->size;
		
		VERBOSE(2,"In Micro2MacroMapping, in    = " <<  *in  << "\n");
		
		// map microtag sequence (in) into the corresponding sequence of macrotags (possibly shorter) (out)
		
		for (int i=microsize; i>0; i--) {
			
			int curr_code = *(in->wordp(i));
			const char *curr_macrotag = lmtable::getDict()->decode((curr_code<microMacroMapN)?microMacroMap[curr_code]:lmtable::getDict()->oovcode());
			
			if (i==microsize) {
				out->pushw(curr_macrotag);
				
			} else {
				int prev_code = *(in->wordp(i+1));
				
				const char *prev_microtag = getDict()->decode(prev_code);
				const char *curr_microtag = getDict()->decode(curr_code);
				const char *prev_macrotag = lmtable::getDict()->decode((prev_code<microMacroMapN)?microMacroMap[prev_code]:lmtable::getDict()->oovcode());
				
				
				int prev_len = strlen(prev_microtag)-1;
				int curr_len = strlen(curr_microtag)-1;
				
				if (strcmp(curr_macrotag,prev_macrotag) != 0 ||
						!(
							(( prev_microtag[prev_len]== '(' || ( prev_microtag[0]== '(' && prev_microtag[prev_len]!= ')' )) &&  ( curr_microtag[curr_len]==')' && curr_microtag[0]!='(')) ||
							(( prev_microtag[prev_len]== '(' || ( prev_microtag[0]== '(' && prev_microtag[prev_len]!= ')' )) &&  curr_microtag[curr_len]=='+' ) ||
							(prev_microtag[prev_len]== '+' &&  curr_microtag[curr_len]=='+' ) ||
							(prev_microtag[prev_len]== '+' &&  ( curr_microtag[curr_len]==')' && curr_microtag[0]!='(' ))))
					out->pushw(curr_macrotag);
			}
		}
		return;
	}
	
	
	
	// DISMITTED ON FEB 2011 BECAUSE TOO MUCH PROBLEMATIC FROM A THEORETICAL POINT OF VIEW
	
#ifdef DLEXICALLM
	
	void lmmacro::Micro2MacroMapping(ngram *in, ngram *out, char **lemmas)
	{
		VERBOSE(2,"In Micro2MacroMapping, in    = " <<  *in  << "\n")
		
		int microsize = in->size;
		
		IFVERBOSE(3) {
			VERBOSE(3,"In Micro2MacroMapping, lemmas:\n");
			if (lexicaltoken2classMap)
				for (int i=microsize; i>0; i--)
					VERBOSE(3,"lemmas[" << i << "]=" << lemmas[i] << " -> class -> " << lexicaltoken2classMap[lmtable::getDict()->encode(lemmas[i])] << endl);
			else
				for (int i=microsize; i>0; i--)
					VERBOSE(3,"lemmas[" << i << "]=" << lemmas[i] << endl);
		}
		
		// map microtag sequence (in) into the corresponding sequence of macrotags (possibly shorter) (out)
		
		char tag_lemma[BUFSIZ];
		
		for (int i=microsize; i>0; i--) {
			
			int curr_code = *(in->wordp(i));
			
			const char *curr_microtag = getDict()->decode(curr_code);
			const char *curr_lemma    = lemmas[i];
			const char *curr_macrotag = lmtable::getDict()->decode((curr_code<microMacroMapN)?microMacroMap[curr_code]:lmtable::getDict()->oovcode());
			int curr_len = strlen(curr_microtag)-1;
			
			if (i==microsize) {
				if (( curr_microtag[curr_len]=='(' ) || ( curr_microtag[0]=='(' && curr_microtag[curr_len]!=')' ) || ( curr_microtag[curr_len]=='+' ))
					sprintf(tag_lemma, "%s", curr_macrotag); // non lessicalizzo il macrotag se sono ancora all''interno del chunk
				else if (lexicaltoken2classMap)
					sprintf(tag_lemma, "%s_class%d", curr_macrotag, lexicaltoken2classMap[lmtable::getDict()->encode(curr_lemma)]);
				else
					sprintf(tag_lemma, "%s_%s", curr_macrotag, lemmas[microsize]);
				
				VERBOSE(2,"In Micro2MacroMapping, starting tag_lemma = >" <<  tag_lemma   << "<\n");
				
				out->pushw(tag_lemma);
				free(lemmas[microsize]);
				
				
			} else {
				
				int prev_code = *(in->wordp(i+1));
				const char *prev_microtag = getDict()->decode(prev_code);
				const char *prev_macrotag = lmtable::getDict()->decode((prev_code<microMacroMapN)?microMacroMap[prev_code]:lmtable::getDict()->oovcode());
				
				
				int prev_len = strlen(prev_microtag)-1;
				
				if (( curr_microtag[curr_len]=='(' ) || ( curr_microtag[0]=='(' && curr_microtag[curr_len]!=')' ) || ( curr_microtag[curr_len]=='+' ))
					sprintf(tag_lemma, "%s", curr_macrotag); // non lessicalizzo il macrotag se sono ancora all''interno del chunk
				else if (lexicaltoken2classMap)
					sprintf(tag_lemma, "%s_class%d", curr_macrotag, lexicaltoken2classMap[lmtable::getDict()->encode(curr_lemma)]);
				else
					sprintf(tag_lemma, "%s_%s", curr_macrotag, curr_lemma);
				
				VERBOSE(2,"In Micro2MacroMapping, tag_lemma = >" <<  tag_lemma   << "<\n");
				
				if (strcmp(curr_macrotag,prev_macrotag) != 0 ||
						!(
							(( prev_microtag[prev_len]== '(' || ( prev_microtag[0]== '(' && prev_microtag[prev_len]!=')' )) && curr_microtag[curr_len]==')' && curr_microtag[0]!='(') ||
							(( prev_microtag[prev_len]== '(' || ( prev_microtag[0]== '(' && prev_microtag[prev_len]!= ')')) && curr_microtag[curr_len]=='+' ) ||
							(prev_microtag[prev_len]== '+' &&  curr_microtag[curr_len]=='+' ) ||
							(prev_microtag[prev_len]== '+' &&  curr_microtag[curr_len]==')' && curr_microtag[0]!='(' ))) {
							
							VERBOSE(2,"In Micro2MacroMapping, before pushw, out = " <<  *out << endl);
							out->pushw(tag_lemma);
							VERBOSE(2,"In Micro2MacroMapping, after pushw, out = " <<  *out << endl);
						} else {
							VERBOSE(2,"In Micro2MacroMapping, before shift, out = " <<  *out << endl);
							out->shift();
							VERBOSE(2,"In Micro2MacroMapping, after shift, out = " <<  *out << endl);
							out->pushw(tag_lemma);
							VERBOSE(2,"In Micro2MacroMapping, after push, out = " <<  *out << endl);
						}
				free(lemmas[i]);
			}
		}
		return;
	}
	
	void lmmacro::loadLexicalClasses(const char *fn)
	{
		char line[MAX_LINE];
		const char* words[MAX_TOKEN_N_MAP];
		int tokenN;
		
		lexicaltoken2classMap = (int *)calloc(BUFSIZ, sizeof(int));
		lexicaltoken2classMapN = BUFSIZ;
		
		lmtable::getDict()->incflag(1);
		
		inputfilestream inp(fn);
		while (inp.getline(line,MAX_LINE,'\n')) {
			tokenN = parseWords(line,words,MAX_TOKEN_N_MAP);
			if (tokenN != 2)
				error((char*)"ERROR: wrong format of lexical classes file\n");
			else {
				int classIdx = atoi(words[1]);
				int wordCode = lmtable::getDict()->encode(words[0]);
				
				if (wordCode>=lexicaltoken2classMapN) {
					int r = (wordCode-lexicaltoken2classMapN)/BUFSIZ;
					lexicaltoken2classMapN += (r+1)*BUFSIZ;
					lexicaltoken2classMap = (int *)reallocf(lexicaltoken2classMap, sizeof(int)*lexicaltoken2classMapN);
				}
				lexicaltoken2classMap[wordCode] = classIdx;
			}
		}
		
		lmtable::getDict()->incflag(0);
		
		IFVERBOSE(3) {
			for (int x=0; x<lmtable::getDict()->size(); x++)
				VERBOSE(3,"class of <" << lmtable::getDict()->decode(x) << "> (code=" << x << ") = " << lexicaltoken2classMap[x] << endl);
		}
		
		return;
	}
	
	
	void lmmacro::cutLex(ngram *in, ngram *out)
	{
		*out=*in;
		
		const char *curr_macro = out->dict->decode(*(out->wordp(1)));
		out->shift();
		const char *p = strrchr(curr_macro, '_');
		int lexLen;
		if (p)
			lexLen=strlen(p);
		else
			lexLen=0;
		char curr_NoLexMacro[BUFSIZ];
		memset(&curr_NoLexMacro,0,BUFSIZ);
		strncpy(curr_NoLexMacro,curr_macro,strlen(curr_macro)-lexLen);
		out->pushw(curr_NoLexMacro);
		return;
	}
#endif
	
}//namespace irstlm
