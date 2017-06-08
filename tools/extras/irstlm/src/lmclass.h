// $Id: lmclass.h 3461 2010-08-27 10:17:34Z bertoldi $

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


#ifndef MF_LMCLASS_H
#define MF_LMCLASS_H

#ifndef WIN32
#include <sys/types.h>
#include <sys/mman.h>
#endif

#include "util.h"
#include "ngramcache.h"
#include "dictionary.h"
#include "n_gram.h"
#include "lmtable.h"

#define LMCLASS_MAX_TOKEN 2

namespace irstlm {
	class lmclass: public lmtable
	{
		dictionary     *dict; // dictionary (words - macro tags)
		double *MapScore;
		int MapScoreN;
		int MaxMapSize;
		
	protected:
		void loadMap(std::istream& inp);
		void loadMapElement(const char* in, const char* out, double sc);
		void mapping(ngram &in, ngram &out);
		
		inline double getMapScore(int wcode) {
			//the input word is un-known by the map, so I "transform" this word into the oov (of the words)
			if (wcode >= MapScoreN) {
				wcode = getDict()->oovcode();
			}
			return MapScore[wcode];
		};
		
		inline size_t getMap(int wcode) {
			//the input word is un-known by the map, so I "transform" this word into the oov (of the words)
			if (wcode >= MapScoreN) {
				wcode = getDict()->oovcode();
			}
			return dict->freq(wcode);
		};
		
		void checkMap();
		
	public:
		lmclass(float nlf=0.0, float dlfi=0.0);
		
		~lmclass();
		
		virtual void load(const std::string &filename,int mmap=0);
		
		virtual double lprob(ngram ng, double* bow,int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow);
		virtual double clprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow) { return lprob(ng, bow, bol, maxsuffidx, maxsuffptr, statesize, extendible, lastbow); };
		
		inline bool is_OOV(int code) {
			//a word is consisdered OOV if its mapped value is OOV
			return lmtable::is_OOV(getMap(code));
		};
		
		inline dictionary* getDict() const {
			return dict;
		}
		inline virtual void dictionary_incflag(const bool flag) {
			dict->incflag(flag);
		};
	};
	
}//namespace irstlm

#endif

