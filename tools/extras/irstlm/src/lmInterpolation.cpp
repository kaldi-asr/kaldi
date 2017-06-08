// $Id: lmInterpolation.cpp 3686 2010-10-15 11:55:32Z bertoldi $

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
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include "lmContainer.h"
#include "lmInterpolation.h"
#include "util.h"

using namespace std;

namespace irstlm {
	lmInterpolation::lmInterpolation(float nlf, float dlf)
	{
		ngramcache_load_factor = nlf;
		dictionary_load_factor = dlf;
		
		order=0;
		memmap=0;
		isInverted=false;
	}
	
	void lmInterpolation::load(const std::string &filename,int mmap)
	{
		VERBOSE(2,"lmInterpolation::load(const std::string &filename,int memmap)" << std::endl);
		VERBOSE(2," filename:|" << filename << "|" << std::endl);
		
		
		dictionary_upperbound=1000000;
		int memmap=mmap;
		
		
		dict=new dictionary((char *)NULL,1000000,dictionary_load_factor);
		
		//get info from the configuration file
		fstream inp(filename.c_str(),ios::in|ios::binary);
		
		char line[MAX_LINE];
		const char* words[LMINTERPOLATION_MAX_TOKEN];
		int tokenN;
		inp.getline(line,MAX_LINE,'\n');
		tokenN = parseWords(line,words,LMINTERPOLATION_MAX_TOKEN);
		
		if (tokenN != 2 || ((strcmp(words[0],"LMINTERPOLATION") != 0) && (strcmp(words[0],"lminterpolation")!=0))){
			exit_error(IRSTLM_ERROR_DATA, "ERROR: wrong header format of configuration file\ncorrect format: LMINTERPOLATION number_of_models\nweight_of_LM_1 filename_of_LM_1\nweight_of_LM_2 filename_of_LM_2");
		}
		m_number_lm = atoi(words[1]);
		
		m_weight.resize(m_number_lm);
		m_file.resize(m_number_lm);
		m_isinverted.resize(m_number_lm);
		m_lm.resize(m_number_lm);
		
		VERBOSE(2,"lmInterpolation::load(const std::string &filename,int mmap) m_number_lm:"<< m_number_lm << std::endl);
		
		dict->incflag(1);
		for (size_t i=0; i<m_number_lm; i++) {
			inp.getline(line,BUFSIZ,'\n');
			tokenN = parseWords(line,words,3);
			
			if(tokenN < 2 || tokenN >3) {
				exit_error(IRSTLM_ERROR_DATA, "ERROR: wrong header format of configuration file\ncorrect format: LMINTERPOLATION number_of_models\nweight_of_LM_1 filename_of_LM_1\nweight_of_LM_2 filename_of_LM_2");
			}
			
			//check whether the (textual) LM has to be loaded as inverted
			m_isinverted[i] = false;
			if(tokenN == 3) {
				if (strcmp(words[2],"inverted") == 0)
					m_isinverted[i] = true;
			}
			VERBOSE(2,"i:" << i << " m_isinverted[i]:" << m_isinverted[i] << endl);
			
			m_weight[i] = (float) atof(words[0]);
			m_file[i] = words[1];
			VERBOSE(2,"lmInterpolation::load(const std::string &filename,int mmap) m_file:"<< words[1] << std::endl);
			
			m_lm[i] = load_lm(i,memmap,ngramcache_load_factor,dictionary_load_factor);
			//set the actual value for inverted flag, which is known only after loading the lM
			m_isinverted[i] = m_lm[i]->is_inverted();
			
			dictionary *_dict=m_lm[i]->getDict();
			for (int j=0; j<_dict->size(); j++) {
				dict->encode(_dict->decode(j));
			}
		}
		dict->genoovcode();
		inp.close();
		
		int maxorder = 0;
		for (size_t i=0; i<m_number_lm; i++) {
			maxorder = (maxorder > m_lm[i]->maxlevel())?maxorder:m_lm[i]->maxlevel();
		}
		
		if (order == 0) {
			order = maxorder;
			VERBOSE(3, "order is not set; reset to the maximum order of LMs: " << order << std::endl);
		} else if (order > maxorder) {
			order = maxorder;
			VERBOSE(3, "order is too high; reset to the maximum order of LMs: " << order << std::endl);
		}
		maxlev=order;
	}
	
	lmContainer* lmInterpolation::load_lm(int i,int memmap, float nlf, float dlf)
	{
		//checking the language model type
		lmContainer* lmt=lmContainer::CreateLanguageModel(m_file[i],nlf,dlf);
		
		//let know that table has inverted n-grams
		lmt->is_inverted(m_isinverted[i]);  //set inverted flag for each LM
		
		lmt->setMaxLoadedLevel(requiredMaxlev);
		
		lmt->load(m_file[i], memmap);
		
		lmt->init_caches(lmt->maxlevel());
		return lmt;
	}
	
	//return log10 prob of an ngram
	double lmInterpolation::clprob(ngram ng, double* bow,int* bol,ngram_state_t* maxsuffidx, char** maxsuffptr,unsigned int* statesize,bool* extendible, double* lastbow)
	{
		
		double pr=0.0;
		double _logpr;
		
		char* _maxsuffptr=NULL,*actualmaxsuffptr=NULL;
		ngram_state_t _maxsuffidx=0,actualmaxsuffidx=0;
		unsigned int _statesize=0,actualstatesize=0;
		int _bol=0,actualbol=MAX_NGRAM;
		double _bow=0.0,actualbow=0.0; 
		double _lastbow=0.0,actuallastbow=0.0; 
		bool _extendible=false,actualextendible=false;
		
		for (size_t i=0; i<m_number_lm; i++) {
			
			if (m_weight[i]>0.0){
				ngram _ng(m_lm[i]->getDict());
				_ng.trans(ng);
				//				_logpr=m_lm[i]->clprob(_ng,&_bow,&_bol,&_maxsuffptr,&_statesize,&_extendible);				
				_logpr=m_lm[i]->clprob(_ng,&_bow,&_bol,&_maxsuffidx,&_maxsuffptr,&_statesize,&_extendible, lastbow);
				
				IFVERBOSE(3){
					//cerr.precision(10);
					VERBOSE(3," LM " << i << " weight:" << m_weight[i] << std::endl);
					VERBOSE(3," LM " << i << " log10 logpr:" << _logpr<< std::endl);
					VERBOSE(3," LM " << i << " pr:" << pow(10.0,_logpr) << std::endl);
					VERBOSE(3," _statesize:" << _statesize << std::endl);
					VERBOSE(3," _bow:" << _bow << std::endl);
					VERBOSE(3," _bol:" << _bol << std::endl);
					VERBOSE(3," _lastbow:" << _lastbow << std::endl);
				}
				
				/*
				 //TO CHECK the following claims
				 //What is the statesize of a LM interpolation? The largest _statesize among the submodels
				 //What is the maxsuffptr of a LM interpolation? The _maxsuffptr of the submodel with the largest _statesize
				 //What is the bol of a LM interpolation? The smallest _bol among the submodels
				 //What is the bow of a LM interpolation? The weighted sum of the bow of the submodels
				 //What is the prob of a LM interpolation? The weighted sum of the prob of the submodels
				 //What is the extendible flag of a LM interpolation? true if the extendible flag is one for any LM
				 //What is the lastbow of a LM interpolation? The weighted sum of the lastbow of the submodels
				 */
				
				pr+=m_weight[i]*pow(10.0,_logpr);
				actualbow+=m_weight[i]*pow(10.0,_bow);
				
				if(_statesize > actualstatesize || i == 0) {
					actualmaxsuffptr = _maxsuffptr;
					actualmaxsuffidx = _maxsuffidx;
					actualstatesize = _statesize;
				}
				if (_bol < actualbol) {
					actualbol=_bol; //backoff limit of LM[i]
				}
				if (_extendible) {
					actualextendible=true; //set extendible flag to true if the ngram is extendible for any LM
				}
				if (_lastbow < actuallastbow) {
					actuallastbow=_lastbow; //backoff limit of LM[i]
				}
			}
		}
		if (bol) *bol=actualbol;
		if (bow) *bow=log(actualbow);
		if (maxsuffptr) *maxsuffptr=actualmaxsuffptr;
		if (maxsuffidx) *maxsuffidx=actualmaxsuffidx;
		if (statesize) *statesize=actualstatesize;
		if (extendible) *extendible=actualextendible;
		if (lastbow) *bol=actuallastbow;
		
		if (statesize) VERBOSE(3, " statesize:" << *statesize << std::endl);
		if (bow) VERBOSE(3, " bow:" << *bow << std::endl);
		if (bol) VERBOSE(3, " bol:" << *bol << std::endl);
		if (lastbow) VERBOSE(3, " lastbow:" << *lastbow << std::endl);
		
		return log10(pr);
	}
	
	const char *lmInterpolation::cmaxsuffptr(ngram ng, unsigned int* statesize)
	{
		
		char *maxsuffptr=NULL;
		unsigned int _statesize=0,actualstatesize=0;
		
		for (size_t i=0; i<m_number_lm; i++) {
			
			if (m_weight[i]>0.0){
				ngram _ng(m_lm[i]->getDict());
				_ng.trans(ng);
				
				const char* _maxsuffptr = m_lm[i]->cmaxsuffptr(_ng,&_statesize);
				
				IFVERBOSE(3){
					//cerr.precision(10);
					VERBOSE(3," LM " << i << " weight:" << m_weight[i] << std::endl);
					VERBOSE(3," _statesize:" << _statesize << std::endl);
				}
				
				/*
				 //TO CHECK the following claims
				 //What is the statesize of a LM interpolation? The largest _statesize among the submodels
				 //What is the maxsuffptr of a LM interpolation? The _maxsuffptr of the submodel with the largest _statesize
				 */
				
				if(_statesize > actualstatesize || i == 0) {
					maxsuffptr = (char*) _maxsuffptr;
					actualstatesize = _statesize;
				}
			}
		}
		if (statesize) *statesize=actualstatesize;
		
		if (statesize) VERBOSE(3, " statesize:" << *statesize << std::endl);
		
		return maxsuffptr;
	}

	ngram_state_t lmInterpolation::cmaxsuffidx(ngram ng, unsigned int* statesize)
	{
		ngram_state_t maxsuffidx=0;
		unsigned int _statesize=0,actualstatesize=0;
		
		for (size_t i=0; i<m_number_lm; i++) {
			
			if (m_weight[i]>0.0){
				ngram _ng(m_lm[i]->getDict());
				_ng.trans(ng);
				
				ngram_state_t _maxsuffidx = m_lm[i]->cmaxsuffidx(_ng,&_statesize);
				
				IFVERBOSE(3){
					//cerr.precision(10);
					VERBOSE(3," LM " << i << " weight:" << m_weight[i] << std::endl);
					VERBOSE(3," _statesize:" << _statesize << std::endl);
				}
				
				/*
				 //TO CHECK the following claims
				 //What is the statesize of a LM interpolation? The largest _statesize among the submodels
				 //What is the maxsuffptr of a LM interpolation? The _maxsuffptr of the submodel with the largest _statesize
				 */
				
				if(_statesize > actualstatesize || i == 0) {
					maxsuffidx = _maxsuffidx;
					actualstatesize = _statesize;
				}
			}
		}
		
	  if (statesize) *statesize=actualstatesize;
		
		if (statesize) VERBOSE(3, " statesize:" << *statesize << std::endl);
		
		return maxsuffidx;
	}
	
	double lmInterpolation::setlogOOVpenalty(int dub)
	{
		MY_ASSERT(dub > dict->size());
		double _logpr;
		double OOVpenalty=0.0;
		for (size_t i=0; i<m_number_lm; i++) {
			if (m_weight[i]>0.0){
				m_lm[i]->setlogOOVpenalty(dub);  //set OOV Penalty for each LM
				_logpr=m_lm[i]->getlogOOVpenalty(); // logOOV penalty is in log10
				//    OOVpenalty+=m_weight[i]*exp(_logpr);
				OOVpenalty+=m_weight[i]*exp(_logpr*M_LN10);  // logOOV penalty is in log10
			}
		}
		//  logOOVpenalty=log(OOVpenalty);
		logOOVpenalty=log10(OOVpenalty);
		return logOOVpenalty;
	}
}//namespace irstlm

