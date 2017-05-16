// $Id: lmmacro.h 3461 2010-08-27 10:17:34Z bertoldi $

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


#ifndef MF_LMMACRO_H
#define MF_LMMACRO_H

#ifndef WIN32
#include <sys/types.h>
#include <sys/mman.h>
#endif

#include "util.h"
#include "ngramcache.h"
#include "dictionary.h"
#include "n_gram.h"
#include "lmtable.h"

#define MAX_TOKEN_N_MAP 5

namespace irstlm {
	
	class lmmacro: public lmtable
	{
		
		dictionary     *dict;
		int             maxlev; //max level of table
		int             selectedField;
		
		bool            collapseFlag; //flag for the presence of collapse
		bool            mapFlag; //flag for the presence of map
		
		int             microMacroMapN;
		int            *microMacroMap;
		bool           *collapsableMap;
		bool           *collapsatorMap;
		
#ifdef DLEXICALLM
		int             selectedFieldForLexicon;
		int            *lexicaltoken2classMap;
		int             lexicaltoken2classMapN;
#endif
		
		
		void loadmap(const std::string mapfilename);
		void unloadmap();
		
		bool transform(ngram &in, ngram &out);
		void field_selection(ngram &in, ngram &out);
		bool collapse(ngram &in, ngram &out);
		void mapping(ngram &in, ngram &out);
		
	public:
		
		lmmacro(float nlf=0.0, float dlfi=0.0);
		~lmmacro();
		
		virtual void load(const std::string &filename,int mmap=0);
		
		virtual double lprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow);
		virtual double clprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow);
		
		virtual const char *maxsuffptr(ngram ong, unsigned int* size=NULL);
		virtual ngram_state_t maxsuffidx(ngram ong, unsigned int* size=NULL);

		void map(ngram *in, ngram *out);
		void One2OneMapping(ngram *in, ngram *out);
		void Micro2MacroMapping(ngram *in, ngram *out);
#ifdef DLEXICALLM
		void Micro2MacroMapping(ngram *in, ngram *out, char **lemma);
		void loadLexicalClasses(const char *fn);
		void cutLex(ngram *in, ngram *out);
#endif
		
		inline bool is_OOV(int code) {
			ngram word_ng(getDict());
			ngram field_ng(getDict());
			word_ng.pushc(code); 
			if (selectedField >= 0)
				field_selection(word_ng, field_ng);
			else
				field_ng = word_ng;
			int field_code=*field_ng.wordp(1);
			VERBOSE(2,"inline virtual bool lmmacro::is_OOV(int code) word_ng:" << word_ng << " field_ng:" << field_ng << std::endl);
			//the selected field(s) of a token is considered OOV 
			//either if unknown by the microMacroMap
			//or if its mapped macroW is OOV
			if (field_code >= microMacroMapN) return true;
			VERBOSE(2,"inline virtual bool lmmacro::is_OOV(int code)*field_code:" << field_code << "  microMacroMap[field_code]:" << microMacroMap[field_code] << " lmtable::dict->oovcode():" << lmtable::dict->oovcode() << std::endl);
			return (microMacroMap[field_code] == lmtable::dict->oovcode());
		};
		inline dictionary* getDict() const {
			return dict;
		}
		inline int maxlevel() const {
			return maxlev;
		};
		
		inline virtual void dictionary_incflag(const bool flag) {
			dict->incflag(flag);
		};

		using lmtable::filter;
		inline virtual bool filter(const string sfilter, lmmacro* sublmt, const string skeepunigrams) {
			UNUSED(sfilter);
			UNUSED(sublmt);
			UNUSED(skeepunigrams);
			return false;
		}
	};
	
}//namespace irstlm
#endif

