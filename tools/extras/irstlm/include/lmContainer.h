// $Id: lmContainer.h 3686 2010-10-15 11:55:32Z bertoldi $

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

#ifndef MF_LMCONTAINER_H
#define MF_LMCONTAINER_H

#define _IRSTLM_LMUNKNOWN 0
#define _IRSTLM_LMTABLE 1
#define _IRSTLM_LMMACRO 2
#define _IRSTLM_LMCLASS 3
#define _IRSTLM_LMINTERPOLATION 4


#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include "util.h"
#include "n_gram.h"
#include "dictionary.h"

typedef enum {BINARY,TEXT,YRANIB,NONE} OUTFILE_TYPE;

typedef enum {LMT_FIND,    //!< search: find an entry
	LMT_ENTER,   //!< search: enter an entry
	LMT_INIT,    //!< scan: start scan
	LMT_CONT     //!< scan: continue scan
} LMT_ACTION;

namespace irstlm {
	class lmContainer
	{
		static const bool debug=true;
		static bool ps_cache_enabled;
		static bool lmt_cache_enabled;
		
	protected:
		int          lmtype; //auto reference to its own type
		int          maxlev; //maximun order of sub LMs;
		int  requiredMaxlev; //max loaded level, i.e. load up to requiredMaxlev levels
		
	public:
		
		lmContainer();
		virtual ~lmContainer() {};
		
		
		virtual void load(const std::string &filename, int mmap=0) {
			UNUSED(filename);
			UNUSED(mmap);
		};
		
		virtual void savetxt(const char *filename) {
			UNUSED(filename);
		};
		virtual void savebin(const char *filename) {
			UNUSED(filename);
		};
		
		virtual double getlogOOVpenalty() const {
			return 0.0;
		};
		virtual double setlogOOVpenalty(int dub) {
			UNUSED(dub);
			return 0.0;
		};
		virtual double setlogOOVpenalty(double oovp) {
			UNUSED(oovp);
			return 0.0;
		};
		
		inline virtual dictionary* getDict() const {
			return NULL;
		};
		inline virtual void maxlevel(int lev) {
			maxlev = lev;
		};
		inline virtual int maxlevel() const {
			return maxlev;
		};
		inline virtual void stat(int lev=0) {
			UNUSED(lev);
		};
		
		inline virtual void setMaxLoadedLevel(int lev) {
			requiredMaxlev=lev;
		};
		inline virtual int getMaxLoadedLevel() {
			return requiredMaxlev;
		};
		
		virtual bool is_inverted(const bool flag) {
			UNUSED(flag);
			return false;
		};
		virtual bool is_inverted() const {
			return false;
		};	
		
		virtual double clprob(ngram ng) { return clprob(ng, NULL, NULL, NULL, NULL, NULL, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow) { return clprob(ng, bow, NULL, NULL, NULL, NULL, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow, int* bol) { return clprob(ng, bow, bol, NULL, NULL, NULL, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow, int* bol, char** maxsuffptr) { return clprob(ng, bow, bol, NULL, maxsuffptr, NULL, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow, int* bol, char** maxsuffptr, unsigned int* statesize) { return clprob(ng, bow, bol, NULL, maxsuffptr, statesize, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow, int* bol, char** maxsuffptr, unsigned int* statesize, bool* extendible) { return clprob(ng, bow, bol, NULL, maxsuffptr, statesize, extendible, NULL); };
		virtual double clprob(ngram ng, double* bow, int* bol, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow) { return clprob(ng, bow, bol, NULL, maxsuffptr, statesize, extendible, lastbow); }
		
		virtual double clprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx) { return clprob(ng, bow, bol, maxsuffidx, NULL, NULL, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr) { return clprob(ng, bow, bol, maxsuffidx, maxsuffptr, NULL, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize) { return clprob(ng, bow, bol, maxsuffidx, maxsuffptr, statesize, NULL, NULL); }
		virtual double clprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible) { return clprob(ng, bow, bol, maxsuffidx, maxsuffptr, statesize, extendible, NULL); };

		
		virtual double clprob(int* ng, int ngsize){ return clprob(ng, ngsize, NULL, NULL, NULL, NULL, NULL, NULL, NULL); }
		virtual double clprob(int* ng, int ngsize, double* bow){ return clprob(ng, ngsize, bow, NULL, NULL, NULL, NULL, NULL, NULL); }
		virtual double clprob(int* ng, int ngsize, double* bow, int* bol){ return clprob(ng, ngsize, bow, bol, NULL, NULL, NULL, NULL, NULL); }
		virtual double clprob(int* ng, int ngsize, double* bow, int* bol, char** maxsuffptr, unsigned int* statesize=NULL, bool* extendible=NULL, double* lastbow=NULL){ return clprob(ng, ngsize, bow, bol, NULL, maxsuffptr, statesize, extendible, lastbow); }
		virtual double clprob(int* ng, int ngsize, double* bow, int* bol, ngram_state_t* maxsuffidx){ return clprob(ng, ngsize, bow, bol, maxsuffidx, NULL, NULL, NULL, NULL); }
		virtual double clprob(int* ng, int ngsize, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize=NULL, bool* extendible=NULL, double* lastbow=NULL)
		{
                        //create the actual ngram
                        ngram ong(getDict());
                        ong.pushc(ng,ngsize);
                        MY_ASSERT (ong.size == ngsize);

			return clprob(ong, bow, bol, maxsuffidx, maxsuffptr, statesize, extendible, lastbow);
		}

		virtual double clprob(int* ng, int ngsize, topic_map_t& topic_weights, double* bow=NULL, int* bol=NULL, ngram_state_t* maxsuffidx=NULL, char** maxsuffptr=NULL, unsigned int* statesize=NULL,bool* extendible=NULL, double* lastbow=NULL)
		{
                        //create the actual ngram
                        ngram ong(getDict());
                        ong.pushc(ng,ngsize);
                        MY_ASSERT (ong.size == ngsize);

			return clprob(ong, topic_weights, bow, bol, maxsuffidx, maxsuffptr, statesize, extendible, lastbow);
		}

		virtual double clprob(ngram ng, double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow)
                {
                        UNUSED(ng);
                        UNUSED(bow);
                        UNUSED(bol);
                        UNUSED(maxsuffidx);
                        UNUSED(maxsuffptr);
                        UNUSED(statesize);
                        UNUSED(extendible);
                        UNUSED(lastbow);

                        return 0.0;
                }

	//this is a function which could be overwritten	
                virtual double clprob(ngram ng, topic_map_t& topic_weights, double* bow=NULL, int* bol=NULL, ngram_state_t* maxsuffidx=NULL, char** maxsuffptr=NULL, unsigned int* statesize=NULL,bool* extendible=NULL, double* lastbow=NULL)
                {
                        UNUSED(topic_weights);
                        UNUSED(ng);
                        UNUSED(bow);
                        UNUSED(bol);
                        UNUSED(maxsuffidx);
                        UNUSED(maxsuffptr);
                        UNUSED(statesize);
                        UNUSED(extendible);
                        UNUSED(lastbow);
                        
                        return 0.0;
		}                

		virtual const char *cmaxsuffptr(ngram ng, unsigned int* statesize=NULL)
		{
			UNUSED(ng);
			UNUSED(statesize);
			return NULL;
		}
		
		virtual const char *cmaxsuffptr(int* ng, int ngsize, unsigned int* statesize=NULL)
		{
			//create the actual ngram 
			ngram ong(getDict());
			ong.pushc(ng,ngsize);
			MY_ASSERT (ong.size == ngsize);
			return cmaxsuffptr(ong, statesize);
		}
		
		virtual ngram_state_t cmaxsuffidx(ngram ng, unsigned int* statesize=NULL)
		{
			UNUSED(ng);
			UNUSED(statesize);
			return 0;
		}
		
		virtual ngram_state_t cmaxsuffidx(int* ng, int ngsize, unsigned int* statesize=NULL)
		{
			//create the actual ngram 
			ngram ong(getDict());                
			ong.pushc(ng,ngsize);
			MY_ASSERT (ong.size == ngsize); 
			return cmaxsuffidx(ong,statesize);
		}
		
		virtual inline int get(ngram& ng) {
			UNUSED(ng);
			return 0;
		}
		
		virtual int get(ngram& ng,int n,int lev){
			UNUSED(ng);
			UNUSED(n);
			UNUSED(lev);
			return 0;
		}
		
		virtual int succscan(ngram& h,ngram& ng,LMT_ACTION action,int lev){
			UNUSED(ng);
			UNUSED(h);
			UNUSED(action);
			UNUSED(lev);
			return 0;     
		}
		
		
		virtual void used_caches() const {};
		virtual void init_caches(int uptolev) {
			UNUSED(uptolev);
		};
		virtual void check_caches_levels() const {};
		virtual void reset_caches() {};
		
		virtual void  reset_mmap() {};
		
		void inline setLanguageModelType(int type) {
			lmtype=type;
		};
		int getLanguageModelType() const {
			return lmtype;
		};
		static int getLanguageModelType(std::string filename);
		
		inline virtual void dictionary_incflag(const bool flag) {
			UNUSED(flag);
		};

		virtual bool filter(const string sfilter, lmContainer*& sublmt, const string skeepunigrams);

		static lmContainer* CreateLanguageModel(const std::string infile, float nlf=0.0, float dlf=0.0);
		static lmContainer* CreateLanguageModel(int type, float nlf=0.0, float dlf=0.0);
		
		inline virtual bool is_OOV(int code) {
			UNUSED(code);
			return false;
		};
		
		
		inline static bool is_lmt_cache_enabled(){
			VERBOSE(3,"inline static bool is_lmt_cache_enabled() " << lmt_cache_enabled << std::endl);
			return lmt_cache_enabled;
		}
		
		inline static bool is_ps_cache_enabled(){
			VERBOSE(3,"inline static bool is_ps_cache_enabled() " << ps_cache_enabled << std::endl);
			return ps_cache_enabled;
		}
		
		inline static bool is_cache_enabled(){
			return is_lmt_cache_enabled() && is_ps_cache_enabled();
		}
		
		virtual int addWord(const char *w){
			getDict()->incflag(1);
			int c=getDict()->encode(w);
			getDict()->incflag(0);
			return c;
		}
		
		virtual void print_table_stat(){
			VERBOSE(3,"virtual void lmContainer::print_table_stat() "<< std::endl);
		};
		
	};
	
}//namespace irstlm

#endif

