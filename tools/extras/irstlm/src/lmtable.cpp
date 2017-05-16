// $Id: lmtable.cpp 3686 2010-10-15 11:55:32Z bertoldi $

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
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <set>
#include "math.h"
#include "mempool.h"
#include "htable.h"
#include "ngramcache.h"
#include "dictionary.h"
#include "n_gram.h"
#include "lmContainer.h"
#include "lmtable.h"
#include "util.h"

//special value for pruned iprobs
#define NOPROB ((float)-1.329227995784915872903807060280344576e36)

using namespace std;

inline void error(const char* message)
{
	VERBOSE(2,message << std::endl);
	throw std::runtime_error(message);
}

void print(prob_and_state_t* pst, std::ostream& out)
{
	if (pst != NULL) {
		out << "PST [";
		out << "logpr:" << pst->logpr;
		out << ",state:" << (void*) pst->state;
		out << ",statesize:" << pst->statesize;
		out << ",bow:" << pst->bow;
		out << ",bol:" << pst->bol;
		out << "]";
		out << std::endl;
	} else {
		out << "PST [NULL]" << std::endl;
	}
}

namespace irstlm {
	
	//instantiate an empty lm table
	lmtable::lmtable(float nlf, float dlf):lmContainer()
	{
		ngramcache_load_factor = nlf;
		dictionary_load_factor = dlf;
		isInverted=false;
		configure(1,false);
		
		dict=new dictionary((char *)NULL,1000000,dictionary_load_factor);
		delete_dict=true;
		
		memset(table, 0, sizeof(table));
		memset(tableGaps, 0, sizeof(tableGaps));
		memset(cursize, 0, sizeof(cursize));
		memset(tbltype, 0, sizeof(tbltype));
		memset(maxsize, 0, sizeof(maxsize));
		memset(tb_offset, 0, sizeof(maxsize));
		memset(info, 0, sizeof(info));
		memset(NumCenters, 0, sizeof(NumCenters));  
		
		max_cache_lev=0;
		for (int i=0; i<LMTMAXLEV+1; i++) lmtcache[i]=NULL;
		for (int i=0; i<LMTMAXLEV+1; i++) prob_and_state_cache[i]=NULL;
		//		prob_and_state_cache=NULL;
		
#ifdef TRACE_CACHELM
		//cacheout=new std::fstream(get_temp_folder()++"tracecache",std::ios::out);
		cacheout=new std::fstream("/tmp/tracecache",std::ios::out);
		sentence_id=0;
#endif
		
		memmap=0;
		requiredMaxlev=IRSTLM_REQUIREDMAXLEV_DEFAULT;
		
		isPruned=false;
		isInverted=false;
		
		//statistics
		for (int i=0; i<LMTMAXLEV+1; i++) totget[i]=totbsearch[i]=0;
		
		logOOVpenalty=0.0; //penalty for OOV words (default 0)
		
		// by default, it is a standard LM, i.e. queried for score
		setOrderQuery(false);
	};
	
	lmtable::~lmtable()
	{
		delete_caches();
		
#ifdef TRACE_CACHELM
		cacheout->close();
		delete cacheout;
#endif
		
		for (int l=1; l<=maxlev; l++) {
			if (table[l]) {
				if (memmap > 0 && l >= memmap)
					Munmap(table[l]-tableGaps[l],cursize[l]*nodesize(tbltype[l])+tableGaps[l],0);
				else
					delete [] table[l];
			}
			if (isQtable) {
				if (Pcenters[l]) delete [] Pcenters[l];
				if (l<maxlev)
					if (Bcenters[l]) delete [] Bcenters[l];
			}
		}
		
		if (delete_dict) delete dict;
	};
	
	void lmtable::init_prob_and_state_cache()
	{
#ifdef PS_CACHE_ENABLE
		for (int i=1; i<=max_cache_lev; i++)
		{
			MY_ASSERT(prob_and_state_cache[i]==NULL);
			prob_and_state_cache[i]=new NGRAMCACHE_t(i,sizeof(prob_and_state_t),400000,ngramcache_load_factor); // initial number of entries is 400000
			VERBOSE(2,"creating cache for storing prob, state and statesize of size " << i << std::endl);
		}
#endif
	}
	
	//	void lmtable::init_lmtcaches(int uptolev)
	void lmtable::init_lmtcaches()
	{
#ifdef LMT_CACHE_ENABLE
		for (int i=2; i<=max_cache_lev; i++)
		{
			MY_ASSERT(lmtcache[i]==NULL);
			lmtcache[i]=new NGRAMCACHE_t(i,sizeof(char*),200000,ngramcache_load_factor); // initial number of entries is 200000
		}
#endif
	}
	
	void lmtable::init_caches(int uptolev)
	{
		max_cache_lev=uptolev;
#ifdef PS_CACHE_ENABLE
		init_prob_and_state_cache();
#endif
#ifdef LMT_CACHE_ENABLE
		init_lmtcaches();
#endif
	}
	
	void lmtable::delete_prob_and_state_cache()
	{
#ifdef PS_CACHE_ENABLE
		for (int i=1; i<=max_cache_lev; i++)
		{
			if (prob_and_state_cache[i])
			{
				delete prob_and_state_cache[i];
			}
			prob_and_state_cache[i]=NULL;
		}
#endif
	}
	
	void lmtable::delete_lmtcaches()
	{
#ifdef LMT_CACHE_ENABLE
		for (int i=2; i<=max_cache_lev; i++)
		{
			if (lmtcache[i])
			{
				delete lmtcache[i];
			}
			lmtcache[i]=NULL;
		}
#endif
	}
	
	void lmtable::delete_caches()
	{
#ifdef PS_CACHE_ENABLE
		delete_prob_and_state_cache();
#endif		
#ifdef LMT_CACHE_ENABLE
		delete_lmtcaches();
#endif
	}
	
	void lmtable::stat_prob_and_state_cache()
	{
#ifdef PS_CACHE_ENABLE
		for (int i=1; i<=max_cache_lev; i++)
		{
			std::cout << "void lmtable::stat_prob_and_state_cache() level:" << i << std::endl;
			if (prob_and_state_cache[i])
			{
				prob_and_state_cache[i]->stat();
			}
		}
#endif
	}
	void lmtable::stat_lmtcaches()
	{
#ifdef PS_CACHE_ENABLE
		for (int i=2; i<=max_cache_lev; i++)
		{
			std::cout << "void lmtable::stat_lmtcaches() level:" << i << std::endl;
			if (lmtcache[i])
			{
				lmtcache[i]->stat();
			}
		}
#endif
	}
	
	void lmtable::stat_caches()
	{
#ifdef PS_CACHE_ENABLE
		stat_prob_and_state_cache();
#endif
#ifdef LMT_CACHE_ENABLE
		stat_lmtcaches();
#endif
	}
	
	
	void lmtable::used_prob_and_state_cache() const
	{
#ifdef PS_CACHE_ENABLE
		for (int i=1; i<=max_cache_lev; i++)
		{
			if (prob_and_state_cache[i])
			{
				prob_and_state_cache[i]->used();
			}
		}
#endif
	}
	
	void lmtable::used_lmtcaches() const
	{
#ifdef LMT_CACHE_ENABLE
		for (int i=2; i<=max_cache_lev; i++)
		{
			if (lmtcache[i])
			{
				lmtcache[i]->used();
			}
		}
#endif
	}
	
	void lmtable::used_caches() const
	{		
#ifdef PS_CACHE_ENABLE
		used_prob_and_state_cache();
#endif		
#ifdef LMT_CACHE_ENABLE
		used_lmtcaches();
#endif
	}
	
	
	void lmtable::check_prob_and_state_cache_levels() const
	{
#ifdef PS_CACHE_ENABLE
		for (int i=1; i<=max_cache_lev; i++)
		{
			if (prob_and_state_cache[i] && prob_and_state_cache[i]->isfull())
			{
				prob_and_state_cache[i]->reset(prob_and_state_cache[i]->cursize());
			}
		}
#endif
	}
	
	void lmtable::check_lmtcaches_levels() const
	{
#ifdef LMT_CACHE_ENABLE
		for (int i=2; i<=max_cache_lev; i++)
		{
			if (lmtcache[i] && lmtcache[i]->isfull())
			{
				lmtcache[i]->reset(lmtcache[i]->cursize());
			}
		}
#endif
	}
	
	void lmtable::check_caches_levels() const
	{
#ifdef PS_CACHE_ENABLE
		check_prob_and_state_cache_levels();
#endif
#ifdef LMT_CACHE_ENABLE
		check_lmtcaches_levels();
#endif
	}
	
	void lmtable::reset_prob_and_state_cache() 
	{
#ifdef PS_CACHE_ENABLE		
		for (int i=1; i<=max_cache_lev; i++)
		{
			if (prob_and_state_cache[i])
			{
				prob_and_state_cache[i]->reset(MAX(prob_and_state_cache[i]->cursize(),prob_and_state_cache[i]->maxsize()));
			}
		}
#endif
	}
	
	void lmtable::reset_lmtcaches()
	{
#ifdef LMT_CACHE_ENABLE
		for (int i=2; i<=max_cache_lev; i++)
		{
			if (lmtcache[i])
			{
				lmtcache[i]->reset(MAX(lmtcache[i]->cursize(),lmtcache[i]->maxsize()));
			}
		}
#endif
	}
	
	void lmtable::reset_caches()
	{
		VERBOSE(2,"void lmtable::reset_caches()" << std::endl);
#ifdef PS_CACHE_ENABLE
		reset_prob_and_state_cache();
#endif
#ifdef LMT_CACHE_ENABLE
		reset_lmtcaches();
#endif
	}
	
	bool lmtable::are_prob_and_state_cache_active() const
	{
#ifdef PS_CACHE_ENABLE
		if (max_cache_lev < 1)
		{
			return false;
		}
		for (int i=1; i<=max_cache_lev; i++)
		{
			if (prob_and_state_cache[i]==NULL)
			{
				return false;
			}
		}
		return true;
		//		return prob_and_state_cache!=NULL;
#else
		return false;
#endif
	}
	
	bool lmtable::are_lmtcaches_active() const
	{
#ifdef LMT_CACHE_ENABLE
		if (max_cache_lev < 2)
		{
			return false;
		}
		for (int i=2; i<=max_cache_lev; i++)
		{
			if (lmtcache[i]==NULL)
			{
				return false;
			}
		}
		return true;
#else
		return false;
#endif
	}
	
	bool lmtable::are_caches_active() const
	{
		return (are_prob_and_state_cache_active() && are_lmtcaches_active());
	}
	
	void lmtable::configure(int n,bool quantized)
	{
		VERBOSE(2,"void lmtable::configure(int n,bool quantized) with n:" << n << std::endl);
		maxlev=n;
		VERBOSE(2,"   maxlev:" << maxlev << " maxlevel():" << maxlevel() << " this->maxlevel():" << this->maxlevel() << std::endl);
		
		//The value for index 0 is never used
		for (int i=0; i<n; i++)
		{
			tbltype[i]=(quantized?QINTERNAL:INTERNAL);
		}
		tbltype[n]=(quantized?QLEAF:LEAF);
	}
	
	
	void lmtable::load(const std::string &infile, int mmap)
	{
		VERBOSE(2,"lmtable::load(const std::string &filename, int mmap)" << std::endl);
		VERBOSE(2,"Reading " << infile << "..." << std::endl);
		inputfilestream inp(infile.c_str());
		
		if (!inp.good()) {
			VERBOSE(2, "Failed to open " << infile << "!" << std::endl);
			exit_error(IRSTLM_ERROR_IO, "Failed to open "+infile);
		}
		setMaxLoadedLevel(requiredMaxlev);
		
		//check whether memory mapping is required
		if (infile.compare(infile.size()-3,3,".mm")==0) {
			mmap=1;
		}
		
		if (mmap>0) { //check whether memory mapping can be used
#ifdef WIN32
			mmap=0; //don't use memory map
#endif
		}
		
		load(inp,infile.c_str(),NULL,mmap);
		getDict()->incflag(0);
	}
	
	void lmtable::load(istream& inp,const char* filename,const char* outfilename,int keep_on_disk)
	{
		VERBOSE(2,"lmtable::load(istream& inp,...)" << std::endl);
		
#ifdef WIN32
		if (keep_on_disk>0) {
			VERBOSE(2, "lmtable::load memory mapping not yet available under WIN32" << std::endl);
			keep_on_disk = 0;
		}
#endif
		
		//give a look at the header to select loading method
		char header[MAX_LINE];
		inp >> header;
		VERBOSE(2, header << std::endl);
		
		if (strncmp(header,"Qblmt",5)==0 || strncmp(header,"blmt",4)==0) {
			loadbin(inp,header,filename,keep_on_disk);
		} else { //input is in textual form
			
			if (keep_on_disk && outfilename==NULL) {
				VERBOSE(2, "Load Error: inconsistent setting. Passed input file: textual. Memory map: yes. Outfilename: not specified." << std::endl);
				exit(0);
			}
			
			loadtxt(inp,header,outfilename,keep_on_disk);
		}
		
		VERBOSE(2, "OOV code is " << lmtable::getDict()->oovcode() << std::endl);
	}
	
	
	//load language model on demand through a word-list file
	
	int lmtable::reload(std::set<string> words)
	{
		//build dictionary
		dictionary dict(NULL,(int)words.size());
		dict.incflag(1);
		
		std::set<string>::iterator w;
		for (w = words.begin(); w != words.end(); ++w)
			dict.encode((*w).c_str());
		
		return 1;
	}
	
	
	
	void lmtable::load_centers(istream& inp,int Order)
	{
		char line[MAX_LINE];
		
		//first read the coodebook
		VERBOSE(2, Order << " read code book " << std::endl);
		inp >> NumCenters[Order];
		Pcenters[Order]=new float[NumCenters[Order]];
		Bcenters[Order]=(Order<maxlev?new float[NumCenters[Order]]:NULL);
		
		for (int c=0; c<NumCenters[Order]; c++) {
			inp >> Pcenters[Order][c];
			if (Order<maxlev) inp >> Bcenters[Order][c];
		};
		//empty the last line
		inp.getline((char*)line,MAX_LINE);
	}
	
	void lmtable::loadtxt(istream& inp,const char* header,const char* outfilename,int mmap)
	{
		if (mmap>0)
			loadtxt_mmap(inp,header,outfilename);
		else {
			loadtxt_ram(inp,header);
			lmtable::getDict()->genoovcode();
		}
	}
	
	void lmtable::loadtxt_mmap(istream& inp,const char* header,const char* outfilename)
	{
		
		char nameNgrams[BUFSIZ];
		char nameHeader[BUFSIZ];
		
		FILE *fd = NULL;
		table_pos_t filesize=0;
		
		int Order,n;
		
		//char *SepString = " \t\n"; unused
		
		//open input stream and prepare an input string
		char line[MAX_LINE];
		
		//prepare word dictionary
		//dict=(dictionary*) new dictionary(NULL,1000000,NULL,NULL);
		lmtable::getDict()->incflag(1);
		
		//check the header to decide if the LM is quantized or not
		isQtable=(strncmp(header,"qARPA",5)==0?true:false);
		
		//check the header to decide if the LM table is incomplete
		isItable=(strncmp(header,"iARPA",5)==0?true:false);
		
		if (isQtable) {
			int maxlevel_h;
			//check if header contains other infos
			inp >> line;
			if (!(maxlevel_h=atoi(line))) {
				VERBOSE(2, "loadtxt with mmap requires new qARPA header. Please regenerate the file." << std::endl);
				exit(1);
			}
			
			for (n=1; n<=maxlevel_h; n++) {
				inp >> line;
				if (!(NumCenters[n]=atoi(line))) {
					VERBOSE(2, "loadtxt with mmap requires new qARPA header. Please regenerate the file." << std::endl);
					exit(0);
				}
			}
		}
		
		//we will configure the table later we we know the maxlev;
		bool yetconfigured=false;
		
		VERBOSE(2,"loadtxtmmap()" << std::endl);
		
		// READ ARPA Header
		
		while (inp.getline(line,MAX_LINE)) {
			
			if (strlen(line)==MAX_LINE-1) {
				VERBOSE(2,"lmtable::loadtxt_mmap: input line exceed MAXLINE (" << MAX_LINE << ") chars " << line << std::endl);
				exit(1);
			}
			
			bool backslash = (line[0] == '\\');
			
			if (sscanf(line, "ngram %d=%d", &Order, &n) == 2) {
				maxsize[Order] = n;
				maxlev=Order; //upadte Order
				VERBOSE(2,"size[" << Order << "]=" << maxsize[Order] << std::endl);
			}
			
			VERBOSE(2,"maxlev" << maxlev << std::endl);
			if (maxlev>requiredMaxlev) maxlev=requiredMaxlev;
			VERBOSE(2,"maxlev" << maxlev << std::endl);
			VERBOSE(2,"lmtable:requiredMaxlev" << requiredMaxlev << std::endl);
			
			if (backslash && sscanf(line, "\\%d-grams", &Order) == 1) {
				
				//at this point we are sure about the size of the LM
				if (!yetconfigured) {
					configure(maxlev,isQtable);
					yetconfigured=true;
					
					//opening output file
					strcpy(nameNgrams,outfilename);
					strcat(nameNgrams, "-ngrams");
					
					fd = fopen(nameNgrams, "w+");
					
					// compute the size of file (only for tables and - possibly - centroids; no header nor dictionary)
					for (int l=1; l<=maxlev; l++) {
						if (l<maxlev)
							filesize +=  (table_pos_t) maxsize[l] * nodesize(tbltype[l]) + 2 * NumCenters[l] * sizeof(float);
						else
							filesize +=  (table_pos_t) maxsize[l] * nodesize(tbltype[l]) + NumCenters[l] * sizeof(float);
					}
					
					// set the file to the proper size:
					ftruncate(fileno(fd),filesize);
					table[0]=(char *)(MMap(fileno(fd),PROT_READ|PROT_WRITE,0,filesize,&tableGaps[0]));
					
					//allocate space for tables into the file through mmap:
					/*
					 if (maxlev>1)
					 table[1]=table[0] + (table_pos_t) (2 * NumCenters[1] * sizeof(float));
					 else
					 table[1]=table[0] + (table_pos_t) (NumCenters[1] * sizeof(float));
					 */
					
					for (int l=1; l<=maxlev; l++) {
						if (l<maxlev)
							table[l]=(char *)(table[l-1] + (table_pos_t) maxsize[l-1]*nodesize(tbltype[l-1]) +
																2 * NumCenters[l] * sizeof(float));
						else
							table[l]=(char *)(table[l-1] + (table_pos_t) maxsize[l-1]*nodesize(tbltype[l-1]) +
																NumCenters[l] * sizeof(float));
						
						VERBOSE(2,"table[" << l << "]-table[" << l-1 << "]=" << (table_pos_t) table[l]-(table_pos_t) table[l-1] << " (nodesize=" << nodesize(tbltype[l-1]) << std::endl);
					}
				}
				
				loadtxt_level(inp,Order);
				
				if (isQtable) {
					// writing centroids on disk
					if (Order<maxlev) {
						memcpy(table[Order] - 2 * NumCenters[Order] * sizeof(float),
									 Pcenters[Order],
									 NumCenters[Order] * sizeof(float));
						memcpy(table[Order] - NumCenters[Order] * sizeof(float),
									 Bcenters[Order],
									 NumCenters[Order] * sizeof(float));
					} else {
						memcpy(table[Order] - NumCenters[Order] * sizeof(float),
									 Pcenters[Order],
									 NumCenters[Order] * sizeof(float));
					}
				}
				// To avoid huge memory write concentrated at the end of the program
				msync(table[0],filesize,MS_SYNC);
				
				// now we can fix table at level Order -1
				// (not required if the input LM is in lexicographical order)
				if (maxlev>1 && Order>1) {
					checkbounds(Order-1);
					delete startpos[Order-1];
				}
			}
		}
		
		VERBOSE(2,"closing output file: " << nameNgrams << std::endl);
		for (int i=1; i<=maxlev; i++) {
			if (maxsize[i] != cursize[i]) {
				for (int l=1; l<=maxlev; l++)
					VERBOSE(2,"Level " << l << ": starting ngrams=" << maxsize[l] << " - actual stored ngrams=" << cursize[l] << std::endl);
				break;
			}
		}
		
		Munmap(table[0],filesize,MS_SYNC);
		for (int l=1; l<=maxlev; l++)
			table[l]=0; // to avoid wrong free in ~lmtable()
		VERBOSE(2,"running fclose..." << std::endl);
		fclose(fd);
		VERBOSE(2,"done" << std::endl);
		
		lmtable::getDict()->incflag(0);
		lmtable::getDict()->genoovcode();
		
		// saving header + dictionary
		
		strcpy(nameHeader,outfilename);
		strcat(nameHeader, "-header");
		VERBOSE(2,"saving header+dictionary in " << nameHeader << "\n");
		fstream out(nameHeader,ios::out);
		
		// print header
		if (isQtable) {
			out << "Qblmt" << (isInverted?"I ":" ") << maxlev;
			for (int i=1; i<=maxlev; i++) out << " " << maxsize[i]; // not cursize[i] because the file was already allocated
			out << "\nNumCenters";
			for (int i=1; i<=maxlev; i++)  out << " " << NumCenters[i];
			out << "\n";
			
		} else {
			out << "blmt" << (isInverted?"I ":" ") << maxlev;
			for (int i=1; i<=maxlev; i++) out << " " << maxsize[i]; // not cursize[i] because the file was already allocated
			out << "\n";
		}
		
		lmtable::getDict()->save(out);
		
		out.close();
		VERBOSE(2,"done" << std::endl);
		
		// cat header+dictionary and n-grams files:
		
		char cmd[BUFSIZ];
		sprintf(cmd,"cat %s >> %s", nameNgrams, nameHeader);
		VERBOSE(2,"run cmd <" << cmd << std::endl);
		system(cmd);
		
		sprintf(cmd,"mv %s %s", nameHeader, outfilename);
		VERBOSE(2,"run cmd <" << cmd << std::endl);
		system(cmd);
		
		removefile(nameNgrams);
		
		//no more operations are available, the file must be saved!
		exit(0);
		return;
	}
	
	
	void lmtable::loadtxt_ram(istream& inp,const char* header)
	{
		//open input stream and prepare an input string
		char line[MAX_LINE];
		
		//prepare word dictionary
		lmtable::getDict()->incflag(1);
		
		//check the header to decide if the LM is quantized or not
		isQtable=(strncmp(header,"qARPA",5)==0?true:false);
		
		//check the header to decide if the LM table is incomplete
		isItable=(strncmp(header,"iARPA",5)==0?true:false);
		
		//we will configure the table later when we will know the maxlev;
		bool yetconfigured=false;
		
		VERBOSE(2,"loadtxt_ram()" << std::endl);
		
		// READ ARPA Header
		int Order;
		unsigned int n;
		
		while (inp.getline(line,MAX_LINE)) {
			if (strlen(line)==MAX_LINE-1) {
				VERBOSE(2,"lmtable::loadtxt_ram: input line exceed MAXLINE (" << MAX_LINE << ") chars " << line << std::endl);
				exit(1);
			}
			
			bool backslash = (line[0] == '\\');
			
			if (sscanf(line, "ngram %d=%u", &Order, &n) == 2) {
				maxsize[Order] = n;
				maxlev=Order; //update Order
			}
			
			if (maxlev>requiredMaxlev) maxlev=requiredMaxlev;
			
			if (backslash && sscanf(line, "\\%d-grams", &Order) == 1) {
				
				//at this point we are sure about the size of the LM
				if (!yetconfigured) {
					configure(maxlev,isQtable);
					yetconfigured=true;
					//allocate space for loading the table of this level
					for (int i=1; i<=maxlev; i++)
						table[i] = new char[(table_pos_t) maxsize[i] * nodesize(tbltype[i])];
				}
				
				loadtxt_level(inp,Order);
				
				// now we can fix table at level Order - 1
				if (maxlev>1 && Order>1) {
					checkbounds(Order-1);
				}
			}
		}
		
		lmtable::getDict()->incflag(0);
		VERBOSE(2,"done" << std::endl);
	}
	
	void lmtable::loadtxt_level(istream& inp, int level)
	{
		VERBOSE(2, level << "-grams: reading " << std::endl);
		
		if (isQtable) {
			load_centers(inp,level);
		}
		
		//allocate support vector to manage badly ordered n-grams
		if (maxlev>1 && level<maxlev) {
			startpos[level]=new table_entry_pos_t[maxsize[level]];
			for (table_entry_pos_t c=0; c<maxsize[level]; c++) {
				startpos[level][c]=BOUND_EMPTY1;
			}
		}
		
		//prepare to read the n-grams entries
		VERBOSE(2, maxsize[level] << " entries" << std::endl);
		
		float prob,bow;
		
		//put here ngrams, log10 probabilities or their codes
		ngram ng(lmtable::getDict());
		ngram ing(lmtable::getDict()); //support n-gram
		
		//WE ASSUME A WELL STRUCTURED FILE!!!
		for (table_entry_pos_t c=0; c<maxsize[level]; c++) {
			
			if (parseline(inp,level,ng,prob,bow)) {
				
				// if table is inverted then revert n-gram
				if (isInverted && (level>1)) {
					ing.invert(ng);
					ng=ing;
				}
				
				//if table is in incomplete ARPA format prob is just the
				//discounted frequency, so we need to add bow * Pr(n-1 gram)
				if (isItable && (level>1)) {
					//get bow of lower context
					get(ng,ng.size,ng.size-1);
					float rbow=0.0;
					if (ng.lev==ng.size-1) { //found context
						rbow=ng.bow;
					}
					
					int tmp=maxlev;
					maxlev=level-1;
					prob= log(exp((double)prob * M_LN10) +  exp(((double)rbow + lprob(ng)) * M_LN10))/M_LN10;
					maxlev=tmp;
				}
				
				//insert an n-gram into the TRIE table
				if (isQtable) add(ng, (qfloat_t)prob, (qfloat_t)bow);
				else add(ng, prob, bow);
			}
		}
		VERBOSE(2, "done level " << level << std::endl);
	}
	
	
	void lmtable::expand_level(int level, table_entry_pos_t size, const char* outfilename, int mmap)
	{
		if (mmap>0)
			expand_level_mmap(level, size, outfilename);
		else {
			expand_level_nommap(level, size);
		}
	}
	
	void lmtable::expand_level_mmap(int level, table_entry_pos_t size, const char* outfilename)
	{
		maxsize[level]=size;
		
		//getting the level-dependent filename
		char nameNgrams[BUFSIZ];
		sprintf(nameNgrams,"%s-%dgrams",outfilename,level);
		
		//opening output file
		FILE *fd = NULL;
		fd = fopen(nameNgrams, "w+");
		if (fd == NULL) {
			perror("Error opening file for writing");
			exit_error(IRSTLM_ERROR_IO, "Error opening file for writing");
		}
		table_pos_t filesize=(table_pos_t) maxsize[level] * nodesize(tbltype[level]);
		// set the file to the proper size:
		ftruncate(fileno(fd),filesize);
		
		/* Now the file is ready to be mmapped.
		 */
		table[level]=(char *)(MMap(fileno(fd),PROT_READ|PROT_WRITE,0,filesize,&tableGaps[level]));
		if (table[level] == MAP_FAILED) {
			fclose(fd);
			perror("Error mmapping the file");
			exit_error(IRSTLM_ERROR_IO, "Error mmapping the file");
		}
		
		if (maxlev>1 && level<maxlev) {
			startpos[level]=new table_entry_pos_t[maxsize[level]];
			/*
			 LMT_TYPE ndt=tbltype[level];
			 TOCHECK XXXXXXXXX
			 int ndsz=nodesize(ndt);
			 char *found = table[level];
			 */
			for (table_entry_pos_t c=0; c<maxsize[level]; c++) {
				startpos[level][c]=BOUND_EMPTY1;
				/*
				 TOCHECK XXXXXXXXX
				 found += ndsz;
				 bound(found,ndt,BOUND_EMPTY2);
				 */
			}
		}
	}
	
	void lmtable::expand_level_nommap(int level, table_entry_pos_t size)
	{
		VERBOSE(2,"lmtable::expand_level_nommap START level:" << level << " size:" << size << endl);
		maxsize[level]=size;
		table[level] = new char[(table_pos_t) maxsize[level] * nodesize(tbltype[level])];
		if (maxlev>1 && level<maxlev) {
			startpos[level]=new table_entry_pos_t[maxsize[level]];
			/*
			 TOCHECK XXXXXXXXX
			 LMT_TYPE ndt=tbltype[level];
			 int ndsz=nodesize(ndt);
			 char *found = table[level];
			 */
			LMT_TYPE ndt=tbltype[level];
			int ndsz=nodesize(ndt);
			char *found = table[level];
			
			for (table_entry_pos_t c=0; c<maxsize[level]; c++) {
				startpos[level][c]=BOUND_EMPTY1;
				/*
				 TOCHECK XXXXXXXXX
				 found += ndsz;
				 bound(found,ndt,BOUND_EMPTY2);
				 */
				found += ndsz;
			}
		}
		VERBOSE(2,"lmtable::expand_level_nommap END level:" << level << endl);
	}
	
	void lmtable::printTable(int level)
	{
		char*  tbl=table[level];
		LMT_TYPE ndt=tbltype[level];
		int ndsz=nodesize(ndt);
		table_entry_pos_t printEntryN=getCurrentSize(level);
		//  if (cursize[level]>0)
		//    printEntryN=(printEntryN<cursize[level])?printEntryN:cursize[level];
		
		cout << "level = " << level << " of size:" << printEntryN <<" ndsz:" << ndsz << " \n";
		
		//TOCHECK: Nicola, 18 dicembre 2009
		
		if (level<maxlev){
			float p;
			float bw;
			table_entry_pos_t bnd;		
			table_entry_pos_t start;
			for (table_entry_pos_t c=0; c<printEntryN; c++) {
				p=prob(tbl,ndt);
				bw=bow(tbl,ndt);
				bnd=bound(tbl,ndt);
				start=startpos[level][c];	
				VERBOSE(2, p << " " << word(tbl) << " -> " << dict->decode(word(tbl)) << " bw:" << bw << " bnd:" << bnd << " " << start << " tb_offset:" << tb_offset[level+1] << std::endl);
				tbl+=ndsz;
			}
		}else{
			float p;
			for (table_entry_pos_t c=0; c<printEntryN; c++) {
				p=prob(tbl,ndt);
				VERBOSE(2, p << " " << word(tbl) << " -> " << dict->decode(word(tbl)) << std::endl);
				tbl+=ndsz;
			}
		}
		return;
	}
	
	//Checkbound with sorting of n-gram table on disk
	void lmtable::checkbounds(int level)
	{
		VERBOSE(2,"lmtable::checkbounds START Level:" << level << endl);
		
		if (getCurrentSize(level) > 0 ){
			
			char*  tbl=table[level];
			char*  succtbl=table[level+1];
			
			LMT_TYPE ndt=tbltype[level];
			LMT_TYPE succndt=tbltype[level+1];
			int ndsz=nodesize(ndt);
			int succndsz=nodesize(succndt);
			
			//re-order table at level+1 on disk
			//generate random filename to avoid collisions
			
			std::string filePath;
			//  ofstream out;
			mfstream out;
			createtempfile(out, filePath, ios::out|ios::binary);
			
			if (out.fail())
			{
				perror("checkbound creating out on filePath");
				exit(4);
			}
			
			table_entry_pos_t start,end,newend;
			table_entry_pos_t succ;
			
			//re-order table at level l+1
			char* found;
			for (table_entry_pos_t c=0; c<cursize[level]; c++) {
				found=tbl+(table_pos_t) c*ndsz;
				start=startpos[level][c];
				end=boundwithoffset(found,ndt,level);
				
				if (c>0) newend=boundwithoffset(found-ndsz,ndt,level);
				else 		newend=0;
				
				//if start==BOUND_EMPTY1 there are no successors for this entry
				if (start==BOUND_EMPTY1){
					succ=0;
				}
				else{
					MY_ASSERT(end>start);
					succ=end-start;
				}
				
				startpos[level][c]=newend;
				newend += succ;
				
				MY_ASSERT(newend<=cursize[level+1]);
				
				if (succ>0) {
					out.write((char*)(succtbl + (table_pos_t) start * succndsz),(table_pos_t) succ * succndsz);
					if (!out.good()) {
						VERBOSE(2," Something went wrong while writing temporary file " << filePath << " Maybe there is not enough space on this filesystem" << endl);
						
						out.close();
						exit(2);
						removefile(filePath);
					}
				}
				
				boundwithoffset(found,ndt,newend,level);
			}
			out.close();	
			if (out.fail())
			{
				perror("error closing out");
				exit(4);
			}
			
			fstream inp(filePath.c_str(),ios::in|ios::binary);
			if (inp.fail())
			{
				perror("error opening inp");
				exit(4);
			}
			
			inp.read(succtbl,(table_pos_t) cursize[level+1]*succndsz);
			inp.close();
			if (inp.fail())
			{
				perror("error closing inp");
				exit(4);
			}
			
			removefile(filePath);
		}
		VERBOSE(2,"lmtable::checkbounds END Level:" << level << endl);
	}
	
	//Add method inserts n-grams in the table structure. It is ONLY used during
	//loading of LMs in text format. It searches for the prefix, then it adds the
	//suffix to the last level and updates the start-end positions.
	int lmtable::addwithoffset(ngram& ng, float iprob, float ibow)
	{
		char *found;
		LMT_TYPE ndt=tbltype[1]; //default initialization
		int ndsz=nodesize(ndt); //default initialization
		static int no_more_msg = 0;
		
		if (ng.size>1) {
			
			// find the prefix starting from the first level
			table_entry_pos_t start=0;
			table_entry_pos_t end=cursize[1];		
			table_entry_pos_t position;
			
			for (int l=1; l<ng.size; l++) {
				
				ndt=tbltype[l];
				ndsz=nodesize(ndt);
				
				if (search(l,start,(end-start),ndsz, ng.wordp(ng.size-l+1),LMT_FIND, &found)) {
					
					//update start and end positions for next step
					if (l < (ng.size-1)) {
						//set start position
						if (found==table[l]){
							start=0; //first pos in table
						}
						else {
							position=(table_entry_pos_t) (((table_pos_t) (found)-(table_pos_t) table[l])/ndsz);
							start=startpos[l][position];
						}
						
						end=boundwithoffset(found,ndt,l);
					}
				} else {
					if (!no_more_msg)
					{
						VERBOSE(2, "warning: missing back-off (at level " << l << ") for ngram " << ng << " (and possibly for others)" << std::endl);
					}
					no_more_msg++;
					if (!(no_more_msg % 5000000))
					{
						VERBOSE(2, "!" << std::endl);
					}
					return 0;
				}
			}
			
			// update book keeping information about level ng-size -1.
			position=(table_entry_pos_t) (((table_pos_t) found-(table_pos_t) table[ng.size-1])/ndsz);
			
			// if this is the first successor update start position in the previous level
			if (startpos[ng.size-1][position]==BOUND_EMPTY1)
				startpos[ng.size-1][position]=cursize[ng.size];
			
			//always update ending position
			boundwithoffset(found,ndt,cursize[ng.size]+1,ng.size-1);
		}
		
		// just add at the end of table[ng.size]
		
		MY_ASSERT(cursize[ng.size]< maxsize[ng.size]); // is there enough space?
		ndt=tbltype[ng.size];
		ndsz=nodesize(ndt);
		
		found=table[ng.size] + ((table_pos_t) cursize[ng.size] * ndsz);
		word(found,*ng.wordp(1));
		prob(found,ndt,iprob);
		if (ng.size<maxlev) {
			//find the bound of the previous entry
			table_entry_pos_t newend;
			if (found==table[ng.size])			newend=0; //first pos in table
			else 			newend=boundwithoffset(found - ndsz,ndt,ng.size);
			
			bow(found,ndt,ibow);
			boundwithoffset(found,ndt,newend,ng.size);
		}
		cursize[ng.size]++;
		
		if (!(cursize[ng.size]%5000000))
		{			
			VERBOSE(1, "." << std::endl);
		}
		return 1;
		
	};
	
	
	//template<typename TA, typename TB>
	//int lmtable::add(ngram& ng, TA iprob,TB ibow)
	
	int lmtable::add(ngram& ng, float iprob, float ibow)
	{
		char *found;
		LMT_TYPE ndt=tbltype[1]; //default initialization
		int ndsz=nodesize(ndt); //default initialization
		static int no_more_msg = 0;
		
		if (ng.size>1) {
			
			// find the prefix starting from the first level
			table_entry_pos_t start=0;
			table_entry_pos_t end=cursize[1];		
			table_entry_pos_t position;
			
			for (int l=1; l<ng.size; l++) {
				
				ndt=tbltype[l];
				ndsz=nodesize(ndt);
				
				if (search(l,start,(end-start),ndsz, ng.wordp(ng.size-l+1),LMT_FIND, &found)) {
					
					//update start and end positions for next step
					if (l < (ng.size-1)) {
						//set start position
						if (found==table[l]){
							start=0; //first pos in table
						}
						else {
							position=(table_entry_pos_t) (((table_pos_t) (found)-(table_pos_t) table[l])/ndsz);
							start=startpos[l][position];
						}
						
						end=bound(found,ndt);
					}
				}
				else {
					if (!no_more_msg)
					{
						VERBOSE(2, "warning: missing back-off (at level " << l << ") for ngram " << ng << " (and possibly for others)" << std::endl);
					}
					no_more_msg++;
					if (!(no_more_msg % 5000000))
					{
						VERBOSE(2, "!" << std::endl);						
					}
					return 0;
				}
			}
			
			// update book keeping information about level ng-size -1.
			position=(table_entry_pos_t) (((table_pos_t) found-(table_pos_t) table[ng.size-1])/ndsz);
			
			// if this is the first successor update start position in the previous level
			if (startpos[ng.size-1][position]==BOUND_EMPTY1)
				startpos[ng.size-1][position]=cursize[ng.size];
			
			//always update ending position
			bound(found,ndt,cursize[ng.size]+1);
		}
		
		// just add at the end of table[ng.size]
		
		MY_ASSERT(cursize[ng.size]< maxsize[ng.size]); // is there enough space?
		ndt=tbltype[ng.size];
		ndsz=nodesize(ndt);
		
		found=table[ng.size] + ((table_pos_t) cursize[ng.size] * ndsz);
		word(found,*ng.wordp(1));
		prob(found,ndt,iprob);
		if (ng.size<maxlev) {
			//find the bound of the previous entry
			table_entry_pos_t newend;
			if (found==table[ng.size])			newend=0; //first pos in table
			else 		newend=bound(found - ndsz,ndt);
			
			bow(found,ndt,ibow);
			bound(found,ndt,newend);
		}
		
		cursize[ng.size]++;
		
		if (!(cursize[ng.size]%5000000))
		{
			VERBOSE(1, "." << std::endl);
		}
		return 1;
		
	};
	
	
	void *lmtable::search(int lev,
												table_entry_pos_t offs,
												table_entry_pos_t n,
												int sz,
												int *ngp,
												LMT_ACTION action,
												char **found)
	{
		
		/***
		 if (n >=2)
		 cout << "searching entry for codeword: " << ngp[0] << "...";
		 ***/
		
		//assume 1-grams is a 1-1 map of the vocabulary
		//CHECK: explicit cast of n into float because table_pos_t could be unsigned and larger than MAXINT
		if (lev==1) return *found=(*ngp < (float) n ? table[1] + (table_pos_t)*ngp * sz:NULL);
		
		
		//prepare table to be searched with mybsearch
		char* tb;
		tb=table[lev] + (table_pos_t) offs * sz;
		//prepare search pattern
		char w[LMTCODESIZE];
		putmem(w,ngp[0],0,LMTCODESIZE);
		
		table_entry_pos_t idx=0; // index returned by mybsearch
		*found=NULL;	//initialize output variable
		
		totbsearch[lev]++;
		switch(action) {
			case LMT_FIND:
				//    if (!tb || !mybsearch(tb,n,sz,(unsigned char *)w,&idx)) return NULL;
				
				if (!tb || !mybsearch(tb,n,sz,w,&idx)) {
					return NULL;
				} else {
					//      return *found=tb + (idx * sz);
					return *found=tb + ((table_pos_t)idx * sz);
				}
			default:
				error((char*)"lmtable::search: this option is available");
		};
		return NULL;
	}
	
	
	/* returns idx with the first position in ar with entry >= key */
	
	int lmtable::mybsearch(char *ar, table_entry_pos_t n, int size, char *key, table_entry_pos_t *idx)
	{
		if (n==0) return 0;
		
		*idx=0;
		table_entry_pos_t low=0, high=n;
		unsigned char *p;
		int result;
		
#ifdef INTERP_SEARCH
		
		char *lp=NULL;
		char  *hp=NULL;
		
#endif
		
		while (low < high) {
			
#ifdef INTERP_SEARCH
			//use interpolation search only for intervals with at least 4096 entries
			
			if ((high-low)>=10000) {
				
				lp=(char *) (ar + (low * size));
				if (codecmp((char *)key,lp)<0) {
					*idx=low;
					return 0;
				}
				
				hp=(char *) (ar + ((high-1) * size));
				if (codecmp((char *)key,hp)>0) {
					*idx=high;
					return 0;
				}
				
				*idx= low + ((high-1)-low) * codediff((char *)key,lp)/codediff(hp,(char *)lp);
			} else
#endif
				*idx = (low + high) / 2;
			
			//after redefining the interval there is no guarantee
			//that wlp <= wkey <= whigh
			
			p = (unsigned char *) (ar + (*idx * size));
			result=codecmp((char *)key,(char *)p);
			
			if (result < 0)
				high = *idx;
			
			else if (result > 0)
				low = ++(*idx);
			else
				return 1;
		}
		
		*idx=low;
		
		return 0;
		
	}
	
	
	// generates a LM copy for a smaller dictionary
	
	void lmtable::cpsublm(lmtable* slmt, dictionary* subdict,bool keepunigr)
	{
		
		//keepunigr=false;
		//let slmt inherit all features of this lmtable
		
		slmt->configure(maxlev,isQtable);
		slmt->dict=new dictionary((keepunigr?dict:subdict),false);
		
		if (isQtable) {
			for (int i=1; i<=maxlev; i++)  {
				slmt->NumCenters[i]=NumCenters[i];
				slmt->Pcenters[i]=new float [NumCenters[i]];
				memcpy(slmt->Pcenters[i],Pcenters[i],NumCenters[i] * sizeof(float));
				
				if (i<maxlev) {
					slmt->Bcenters[i]=new float [NumCenters[i]];
					memcpy(slmt->Bcenters[i],Bcenters[i],NumCenters[i] * sizeof(float));
				}
			}
		}
		
		//manage dictionary information
		
		//generate OOV codes and build dictionary lookup table
		dict->genoovcode();
		slmt->dict->genoovcode();
		subdict->genoovcode();
		
		int* lookup=new int [dict->size()];
		
		for (int c=0; c<dict->size(); c++) {
			lookup[c]=subdict->encode(dict->decode(c));
			if (c != dict->oovcode() && lookup[c] == subdict->oovcode())
				lookup[c]=-1; // words of this->dict that are not in slmt->dict
		}
		
		//variables useful to navigate in the lmtable structure
		LMT_TYPE ndt,pndt;
		int ndsz,pndsz;
		char *entry, *newentry;
		table_entry_pos_t start, end, origin;
		
		for (int l=1; l<=maxlev; l++) {
			
			slmt->cursize[l]=0;
			slmt->table[l]=NULL;
			
			if (l==1) { //1-gram level
				
				ndt=tbltype[l];
				ndsz=nodesize(ndt);
				
				for (table_entry_pos_t p=0; p<cursize[l]; p++) {
					
					entry=table[l] + (table_pos_t) p * ndsz;
					if (lookup[word(entry)]!=-1 || keepunigr) {
						
						if ((slmt->cursize[l] % slmt->dict->size()) ==0)
							slmt->table[l]=(char *)reallocf(slmt->table[l],((table_pos_t) slmt->cursize[l] + (table_pos_t) slmt->dict->size()) * ndsz);
						
						newentry=slmt->table[l] + (table_pos_t) slmt->cursize[l] * ndsz;
						memcpy(newentry,entry,ndsz);
						if (!keepunigr) //do not change encoding if keepunigr is true
							slmt->word(newentry,lookup[word(entry)]);
						
						if (l<maxlev)
							slmt->bound(newentry,ndt,p); //store in bound the entry itself (**) !!!!
						slmt->cursize[l]++;
					}
				}
			}
			
			else { //n-grams n>1: scan lower order table
				
				pndt=tbltype[l-1];
				pndsz=nodesize(pndt);
				ndt=tbltype[l];
				ndsz=nodesize(ndt);
				
				for (table_entry_pos_t p=0; p<slmt->cursize[l-1]; p++) {
					
					//determine start and end of successors of this entry
					origin=slmt->bound(slmt->table[l-1] + (table_pos_t)p * pndsz,pndt); //position of n-1 gram in this table (**)
					if (origin == 0) start=0;                              //succ start at first pos in table[l]
					else start=bound(table[l-1] + (table_pos_t)(origin-1) * pndsz,pndt);//succ start after end of previous entry
					end=bound(table[l-1] + (table_pos_t)origin * pndsz,pndt);           //succ end where indicated
					
					if (!keepunigr || lookup[word(table[l-1] + (table_pos_t)origin * pndsz)]!=-1) {
						while (start < end) {
							
							entry=table[l] + (table_pos_t) start * ndsz;
							
							if (lookup[word(entry)]!=-1) {
								
								if ((slmt->cursize[l] % slmt->dict->size()) ==0)
									slmt->table[l]=(char *)reallocf(slmt->table[l],(table_pos_t) (slmt->cursize[l]+slmt->dict->size()) * ndsz);
								
								newentry=slmt->table[l] + (table_pos_t) slmt->cursize[l] * ndsz;
								memcpy(newentry,entry,ndsz);
								if (!keepunigr) //do not change encoding if keepunigr is true
									slmt->word(newentry,lookup[word(entry)]);
								
								if (l<maxlev)
									slmt->bound(newentry,ndt,start); //store in bound the entry itself!!!!
								slmt->cursize[l]++;
							}
							start++;
						}
					}
					
					//updated bound information of incoming entry
					slmt->bound(slmt->table[l-1] + (table_pos_t) p * pndsz, pndt,slmt->cursize[l]);
				}
			}
		}
		
		return;
	}
	
	
	
	// saves a LM table in text format
	
	void lmtable::savetxt(const char *filename)
	{
		
		fstream out(filename,ios::out);
		table_entry_pos_t cnt[1+MAX_NGRAM];
		int l;
		
		//	out.precision(7);
		out.precision(6);
		
		if (isQtable) {
			out << "qARPA " << maxlev;
			for (l=1; l<=maxlev; l++)
				out << " " << NumCenters[l];
			out << endl;
		}
		
		ngram ng(lmtable::getDict(),0);
		
		VERBOSE(2, "savetxt: " << filename << std::endl);
		
		if (isPruned) ngcnt(cnt); //check size of table by considering pruned n-grams
		
		out << "\n\\data\\\n";
		char buff[100];
		for (l=1; l<=maxlev; l++) {
			sprintf(buff,"ngram %2d=%10d\n",l,(isPruned?cnt[l]:cursize[l]));
			out << buff;
		}
		out << "\n";
		
		for (l=1; l<=maxlev; l++) {
			
			out << "\n\\" << l << "-grams:\n";
			VERBOSE(2, "save: " << (isPruned?cnt[l]:cursize[l]) << " " << l << "-grams" << std::endl);
			if (isQtable) {
				out << NumCenters[l] << "\n";
				for (int c=0; c<NumCenters[l]; c++) {
					out << Pcenters[l][c];
					if (l<maxlev) out << " " << Bcenters[l][c];
					out << "\n";
				}
			}
			
			ng.size=0;
			dumplm(out,ng,1,l,0,cursize[1]);
			
		}
		
		out << "\\end\\\n";
		VERBOSE(2, "done" << std::endl);
	}
	
	
	
	void lmtable::savebin(const char *filename)
	{
		VERBOSE(2,"lmtable::savebin START " << filename << "\n");
		
		if (isPruned) {
			VERBOSE(2,"lmtable::savebin: pruned LM cannot be saved in binary form\n");
			exit(0);
		}
		
		
		fstream out(filename,ios::out);
		
		// print header
		if (isQtable) {
			out << "Qblmt" << (isInverted?"I":"") << " " << maxlev;
			for (int i=1; i<=maxlev; i++) out << " " << cursize[i];
			out << "\nNumCenters";
			for (int i=1; i<=maxlev; i++)  out << " " << NumCenters[i];
			out << "\n";
			
		} else {
			out << "blmt" << (isInverted?"I":"") << " " << maxlev;
			char buff[100];
			for (int i=1; i<=maxlev; i++){
				sprintf(buff," %10d",cursize[i]);
				out << buff;
			}
			out << "\n";
		}
		
		lmtable::getDict()->save(out);
		
		for (int i=1; i<=maxlev; i++) {
			if (isQtable) {
				out.write((char*)Pcenters[i],NumCenters[i] * sizeof(float));
				if (i<maxlev)
					out.write((char *)Bcenters[i],NumCenters[i] * sizeof(float));
			}
			out.write(table[i],(table_pos_t) cursize[i]*nodesize(tbltype[i]));
		}
		
		VERBOSE(2,"lmtable::savebin: END\n");
	}
	
	void lmtable::savebin_dict(std::fstream& out)
	{
		/*
		 if (isPruned)
		 {
		 VERBOSE(2,"savebin_dict: pruned LM cannot be saved in binary form\n");
		 exit(0);
		 }
		 */
		
		VERBOSE(2,"savebin_dict ...\n");
		getDict()->save(out);
	}
	
	
	
	void lmtable::appendbin_level(int level, fstream &out, int mmap)
	{
		if (getCurrentSize(level) > 0 ){
			if (mmap>0)
				appendbin_level_mmap(level, out);
			else {
				appendbin_level_nommap(level, out);
			}
		}
	}
	
	void lmtable::appendbin_level_nommap(int level, fstream &out)
	{
		VERBOSE(2,"lmtable:appendbin_level_nommap START Level:" << level << std::endl);
		
		/*
		 if (isPruned){
		 VERBOSE(2,"savebin_level (level " << level << "):  pruned LM cannot be saved in binary form" << std::endl);
		 exit(0);
		 }
		 */
		
		MY_ASSERT(level<=maxlev);	
		
		// print header
		if (isQtable) {
			//NOT IMPLEMENTED
		} else {
			//do nothing
		}
		
		VERBOSE(3,"appending " << cursize[level] << " (maxsize:" << maxsize[level] << ") " << level << "-grams" << "   table " << (void*) table << "  table[level] " << (void*) table[level] << endl);
		
		if (isQtable) {
			//NOT IMPLEMENTED
		}
		
		out.write(table[level],(table_pos_t) cursize[level]*nodesize(tbltype[level]));
		
		if (!out.good()) {
			perror("Something went wrong while writing");
			out.close();
			exit(2);
		}
		
		VERBOSE(2,"lmtable:appendbin_level_nommap END Level:" << level << std::endl);
	}
	
	
	void lmtable::appendbin_level_mmap(int level, fstream &out)
	{
		UNUSED(out);
		VERBOSE(2,"appending " << level << " (Actually do nothing)" << std::endl);
	}
	
	void lmtable::savebin_level(int level, const char* outfilename, int mmap)
	{
		if (mmap>0)
			savebin_level_mmap(level, outfilename);
		else {
			savebin_level_nommap(level, outfilename);
		}
	}
	
	void lmtable::savebin_level_nommap(int level, const char* outfilename)
	{
		VERBOSE(2,"lmtable:savebin_level_nommap START" << requiredMaxlev << std::endl);
		
		/*
		 if (isPruned){
		 cerr << "savebin_level (level " << level << "):  pruned LM cannot be saved in binary form\n";
		 exit(0);
		 }
		 */
		
		MY_ASSERT(level<=maxlev);
		
		char nameNgrams[BUFSIZ];
		sprintf(nameNgrams,"%s-%dgrams",outfilename,level);
		
		fstream out(nameNgrams, ios::out|ios::binary);
		
		if (out.fail())
		{
			perror("cannot be opened");
			exit(3);
		}
		
		// print header
		if (isQtable) {
			//NOT IMPLEMENTED
		} else {
			//do nothing
		}
		
		VERBOSE(3,"saving " << cursize[level] << "(maxsize:" << maxsize[level] << ") " << level << "-grams in " << nameNgrams << "   table " << (void*) table << "  table[level] " << (void*) table[level] << endl);
		if (isQtable) {
			//NOT IMPLEMENTED
		}
		
		out.write(table[level],(table_pos_t) cursize[level]*nodesize(tbltype[level]));
		
		if (!out.good()) {
			VERBOSE(2," Something went wrong while writing temporary file " << nameNgrams << endl);
			out.close();
			removefile(nameNgrams);
			exit(2);
		}
		out.close();
		if (out.fail())
		{
			perror("cannot be closed");
			exit(3);
		}
		
		VERBOSE(2,"lmtable:savebin_level_nommap END" << requiredMaxlev << std::endl);
	}
	
	void lmtable::savebin_level_mmap(int level, const char* outfilename)
	{
		char nameNgrams[BUFSIZ];
		sprintf(nameNgrams,"%s-%dgrams",outfilename,level);
		VERBOSE(2,"saving " << level << "-grams probs in " << nameNgrams << " (Actually do nothing)" <<std::endl);
	}
	
	
	
	void lmtable::print_table_stat()
	{
		VERBOSE(2,"printing statistics of tables" << std::endl);
		for (int i=1; i<=maxlev; i++)
			print_table_stat(i);
	}
	
	void lmtable::print_table_stat(int level)
	{
		VERBOSE(2," level: " << level << std::endl);
		VERBOSE(2," maxsize[level]:" << maxsize[level] << std::endl);
		VERBOSE(2," cursize[level]:" << cursize[level] << std::endl);
		VERBOSE(2," tb_offset[level]:" << tb_offset[level] << std::endl);
		VERBOSE(2," table:" << (void*) table << std::endl);
		VERBOSE(2," table[level]:" << (void*) table[level] << std::endl);
		VERBOSE(2," table[level]-table:" << ((char*) table[level]-(char*) table) << std::endl);
		VERBOSE(2," tableGaps[level]:" << (void*) tableGaps[level] << std::endl);
	}
	
	//concatenate corresponding single level files of two different tables for each level
	void lmtable::concatenate_all_levels(const char* fromfilename, const char* tofilename){
		//single level files should have a name derived from "filename"
		//there no control that the tables have the same size
		for (int i=1; i<=maxlevel(); i++) {
			concatenate_single_level(i, fromfilename, tofilename);
		}
	}
	
	//concatenate corresponding single level files of two different tables
	void lmtable::concatenate_single_level(int level, const char* fromfilename, const char* tofilename){
		//single level files should have a name derived from "fromfilename" and "tofilename"
		char fromnameNgrams[BUFSIZ];
		char tonameNgrams[BUFSIZ];
		sprintf(fromnameNgrams,"%s-%dgrams",fromfilename,level);
		sprintf(tonameNgrams,"%s-%dgrams",tofilename,level);
		
		VERBOSE(2,"concatenating " << level << "-grams probs from " << fromnameNgrams << " to " << tonameNgrams<< std::endl);
		
		
		//concatenating of new table to the existing data
		char cmd[BUFSIZ];
		sprintf(cmd,"cat %s >> %s", fromnameNgrams, tonameNgrams);
		system(cmd);
	}
	
	//remove all single level files
	void lmtable::remove_all_levels(const char* filename){
		//single level files should have a name derived from "filename"
		for (int i=1; i<=maxlevel(); i++) {
			remove_single_level(i,filename);
		}
	}
	
	//remove a single level file
	void lmtable::remove_single_level(int level, const char* filename){
		//single level files should have a name derived from "filename"
		char nameNgrams[BUFSIZ];
		sprintf(nameNgrams,"%s-%dgrams",filename,level);
		
		//removing temporary files
		removefile(nameNgrams);
	}
	
	
	
	//delete the table of a single level
	void lmtable::delete_level(int level, const char* outfilename, int mmap){
		if (mmap>0)
			delete_level_mmap(level, outfilename);
		else {
			delete_level_nommap(level);
		}
	}
	
	void lmtable::delete_level_mmap(int level, const char* outfilename)
	{
		//getting the level-dependent filename
		char nameNgrams[BUFSIZ];
		sprintf(nameNgrams,"%s-%dgrams",outfilename,level);
		
		//compute exact filesize
		table_pos_t filesize=(table_pos_t) cursize[level] * nodesize(tbltype[level]);
		
		// set the file to the proper size:
		Munmap(table[level]-tableGaps[level],(table_pos_t) filesize+tableGaps[level],0);
		
		maxsize[level]=cursize[level]=0;
	}
	
	void lmtable::delete_level_nommap(int level)
	{
		delete table[level];
		maxsize[level]=cursize[level]=0;
	}
	
	void lmtable::compact_all_levels(const char* filename){
		//single level files should have a name derived from "filename"
		for (int i=1; i<=maxlevel(); i++) {
			compact_single_level(i,filename);
		}
	}
	
	void lmtable::compact_single_level(int level, const char* filename)
	{
		char nameNgrams[BUFSIZ];
		sprintf(nameNgrams,"%s-%dgrams",filename,level);
		
		VERBOSE(2,"concatenating " << level << "-grams probs from " << nameNgrams << " to " << filename<< std::endl);
		
		
		//concatenating of new table to the existing data
		char cmd[BUFSIZ];
		sprintf(cmd,"cat %s >> %s", nameNgrams, filename);
		system(cmd);
		
		//removing temporary files
		removefile(nameNgrams);
	}
	
	void lmtable::resize_level(int level, const char* outfilename, int mmap)
	{
		if (getCurrentSize(level) > 0 ){
			if (mmap>0)
				resize_level_mmap(level, outfilename);
			else {
				if (level<maxlev) // (apart from last level maxlev, because is useless), resizing is done when saving
					resize_level_nommap(level);
			}
		}
	}
	
	void lmtable::resize_level_mmap(int level, const char* outfilename)
	{
		//getting the level-dependent filename
		char nameNgrams[BUFSIZ];
		sprintf(nameNgrams,"%s-%dgrams",outfilename,level);
		
		//recompute exact filesize
		table_pos_t filesize=(table_pos_t) cursize[level] * nodesize(tbltype[level]);
		
		//opening output file
		FILE *fd = NULL;
		fd = fopen(nameNgrams, "r+");
		
		// set the file to the proper size:
		Munmap(table[level]-tableGaps[level],(table_pos_t) filesize+tableGaps[level],0);
		ftruncate(fileno(fd),filesize);
		table[level]=(char *)(MMap(fileno(fd),PROT_READ|PROT_WRITE,0,filesize,&tableGaps[level]));
		maxsize[level]=cursize[level];
	}
	
	void lmtable::resize_level_nommap(int level)
	{
		VERBOSE(2,"lmtable::resize_level_nommap START Level " << level << "\n");
		
		//recompute exact filesize
		table_pos_t filesize=(table_pos_t) cursize[level] * nodesize(tbltype[level]);
		
		char* ptr = new char[filesize];
		memcpy(ptr,table[level],filesize);
		delete table[level];
		table[level]=ptr;
		maxsize[level]=cursize[level];
		
		VERBOSE(2,"lmtable::resize_level_nommap END Level " << level << "\n");
	}
	
	
	//manages the long header of a bin file
	//and allocates table for each n-gram level
	
	void lmtable::loadbin_header(istream& inp,const char* header)
	{
		
		// read rest of header
		inp >> maxlev;
		
		//set the inverted falg to false, in order to rely on the header only
		isInverted=false;
		
		if (strncmp(header,"Qblmt",5)==0) {
			isQtable=true;
			if (strncmp(header,"QblmtI",6)==0)
				isInverted=true;
		} else if(strncmp(header,"blmt",4)==0) {
			isQtable=false;
			if (strncmp(header,"blmtI",5)==0)
				isInverted=true;
		} else error((char*)"loadbin: LM file is not in binary format");
		
		configure(maxlev,isQtable);
		
		for (int l=1; l<=maxlev; l++) {
			inp >> cursize[l];
			maxsize[l]=cursize[l];
		}
		
		//update table offsets
		for (int l=2; l<=maxlev; l++) update_offset(l,tb_offset[l-1]+maxsize[l-1]);
		
		char header2[MAX_LINE];
		if (isQtable) {
			inp >> header2;
			for (int i=1; i<=maxlev; i++) {
				inp >> NumCenters[i];
				VERBOSE(2,"reading  " << NumCenters[i] << " centers" << "\n");
			}
		}
		inp.getline(header2, MAX_LINE);
	}
	
	//load codebook of level l
	void lmtable::loadbin_codebook(istream& inp,int l)
	{
		Pcenters[l]=new float [NumCenters[l]];
		inp.read((char*)Pcenters[l],NumCenters[l] * sizeof(float));
		if (l<maxlev) {
			Bcenters[l]=new float [NumCenters[l]];
			inp.read((char *)Bcenters[l],NumCenters[l]*sizeof(float));
		}
	}
	
	
	//load a binary lmfile
	
	void lmtable::loadbin(istream& inp, const char* header, const char* filename,int mmap)
	{
		VERBOSE(2,"loadbin()" << "\n");
		loadbin_header(inp,header);
		loadbin_dict(inp);
		
		VERBOSE(3,"lmtable::maxlev" << maxlev << std::endl);
		if (maxlev>requiredMaxlev) maxlev=requiredMaxlev;
		VERBOSE(3,"lmtable::maxlev:" << maxlev << std::endl);
		VERBOSE(3,"lmtable::requiredMaxlev" << requiredMaxlev << std::endl);
		
		//if MMAP is used, then open the file
		if (filename && mmap>0) {
			
#ifdef WIN32
			error("lmtable::loadbin mmap facility not yet supported under WIN32\n");
#else
			
			if (mmap <= maxlev) memmap=mmap;
			else error((char*)"keep_on_disk value is out of range\n");
			
			if ((diskid=open(filename, O_RDONLY))<0) {
				VERBOSE(2,"cannot open " << filename << std::endl);
				error((char*)"dying");
			}
			
			//check that the LM is uncompressed
			char miniheader[4];
			read(diskid,miniheader,4);
			if (strncmp(miniheader,"Qblm",4) && strncmp(miniheader,"blmt",4))
				error((char*)"mmap functionality does not work with compressed binary LMs\n");
#endif
		}
		
		for (int l=1; l<=maxlev; l++) {
			loadbin_level(inp,l);
		}
		VERBOSE(2,"done" << std::endl);
	}
	
	
	//load only the dictionary of a binary lmfile
	void lmtable::loadbin_dict(istream& inp)
	{
		VERBOSE(2,"lmtable::loadbin_dict()" << std::endl);
		lmtable::getDict()->load(inp);		
		VERBOSE(2,"dict->size(): " << lmtable::getDict()->size() << std::endl);
	}
	
	//load ONE level of a binary lmfile
	void lmtable::loadbin_level(istream& inp, int level)
	{
		VERBOSE(2,"loadbin_level (level " << level << std::endl);
		
		if (isQtable)
		{
			loadbin_codebook(inp,level);
		}
		if ((memmap == 0) || (level < memmap))
		{
			VERBOSE(2,"loading " << cursize[level] << " " << level << "-grams" << std::endl);
			table[level]=new char[(table_pos_t) cursize[level] * nodesize(tbltype[level])];
			inp.read(table[level],(table_pos_t) cursize[level] * nodesize(tbltype[level]));
		} else {
			
#ifdef WIN32
			error((char*)"mmap not available under WIN32\n");
#else
			VERBOSE(2,"mapping " << cursize[level] << " " << level << "-grams" << std::endl);
			tableOffs[level]=inp.tellg();
			table[level]=(char *)MMap(diskid,PROT_READ,
																tableOffs[level], (table_pos_t) cursize[level]*nodesize(tbltype[level]),
																&tableGaps[level]);
			table[level]+=(table_pos_t) tableGaps[level];
			VERBOSE(2,"tableOffs " << tableOffs[level] << " tableGaps" << tableGaps[level] << "-grams" << std::endl);
			inp.seekg((table_pos_t) cursize[level]*nodesize(tbltype[level]),ios_base::cur);
#endif
		}
		VERBOSE(2,"done (level " << level << std::endl);
	}
	
	int lmtable::get(ngram& ng,int n,int lev)
	{
		totget[lev]++;
		
		if (lev > maxlev) error((char*)"get: lev exceeds maxlevel");
		if (n < lev) error((char*)"get: ngram is too small");
		
		//set boudaries for 1-gram
		table_entry_pos_t offset=0,limit=cursize[1];
		
		//information of table entries
		char* found;
		LMT_TYPE ndt;
		ng.link=NULL;
		ng.lev=0;
		
		for (int l=1; l<=lev; l++) {
			
			//initialize entry information
			found = NULL;
			ndt=tbltype[l];
			
#ifdef LMT_CACHE_ENABLE
			bool hit = false;
			if (lmtcache[l] && lmtcache[l]->get(ng.wordp(n),found)) {
				hit=true;
			} else {
				search(l,
							 offset,
							 (limit-offset),
							 nodesize(ndt),
							 ng.wordp(n-l+1),
							 LMT_FIND,
							 &found);
			}
			
			
			
			//insert both found and not found items!!!
			//			if (lmtcache[l] && hit==true) {
			
			//insert only not found items!!!
			if (lmtcache[l] && hit==false) {
				const char* found2=found;
				lmtcache[l]->add(ng.wordp(n),found2);
			}
#else
			search(l,
						 offset,
						 (limit-offset),
						 nodesize(ndt),
						 ng.wordp(n-l+1),
						 LMT_FIND,
						 &found);
#endif
			
			if (!found) return 0;
			
			float pr = prob(found,ndt);
			if (pr==NOPROB) return 0; //pruned n-gram
			
			ng.path[l]=found; //store path of found entries
			ng.bow=(l<maxlev?bow(found,ndt):0);
			ng.prob=pr;
			ng.link=found;
			ng.info=ndt;
			ng.lev=l;
			
			if (l<maxlev) { //set start/end point for next search
				
				//if current offset is at the bottom also that of successors will be
				if (offset+1==cursize[l]) limit=cursize[l+1];
				else limit=bound(found,ndt);
				
				//if current start is at the begin, then also that of successors will be
				if (found==table[l]) offset=0;
				else offset=bound((found - nodesize(ndt)),ndt);
				
				MY_ASSERT(offset!=BOUND_EMPTY1);
				MY_ASSERT(limit!=BOUND_EMPTY1);
			}
		}
		
		
		//put information inside ng
		ng.size=n;
		ng.freq=0;
		ng.succ=(lev<maxlev?limit-offset:0);
		
#ifdef TRACE_CACHELM
		if (ng.size==maxlev && sentence_id>0) {
			*cacheout << sentence_id << " miss " << ng << " " << ng.link << "\n";
		}
#endif
		return 1;
	}
	
	
	//recursively prints the language model table
	
	void lmtable::dumplm(fstream& out,ngram ng, int ilev, int elev, table_entry_pos_t ipos,table_entry_pos_t epos)
	{
		
		LMT_TYPE ndt=tbltype[ilev];
		ngram ing(ng.dict);
		int ndsz=nodesize(ndt);
		
		MY_ASSERT(ng.size==ilev-1);
		
		//Note that ipos and epos are always larger than or equal to 0 because they are unsigned int
		MY_ASSERT(epos<=cursize[ilev]);
		MY_ASSERT(ipos<epos);
		ng.pushc(0);
		
		for (table_entry_pos_t i=ipos; i<epos; i++) {
			char* found=table[ilev]+ (table_pos_t) i * ndsz;
			*ng.wordp(1)=word(found);
			
			float ipr=prob(found,ndt);
			
			//skip pruned n-grams
			if(isPruned && ipr==NOPROB) continue;
			
			if (ilev<elev) {
				//get first and last successor position
				table_entry_pos_t isucc=(i>0?bound(table[ilev]+ (table_pos_t) (i-1) * ndsz,ndt):0);
				table_entry_pos_t esucc=bound(found,ndt);
				
				if (isucc < esucc) //there are successors!
					dumplm(out,ng,ilev+1,elev,isucc,esucc);
			} else {
				out << ipr <<"\t";
				
				// if table is inverted then revert n-gram
				if (isInverted && (ng.size>1)) {
					ing.invert(ng);
					for (int k=ing.size; k>=1; k--) {
						if (k<ing.size) out << " ";
						out << lmtable::getDict()->decode(*ing.wordp(k));
					}
				} else {
					for (int k=ng.size; k>=1; k--) {
						if (k<ng.size) out << " ";
						out << lmtable::getDict()->decode(*ng.wordp(k));
					}
				}
				
				if (ilev<maxlev) {
					float ibo=bow(table[ilev]+ (table_pos_t)i * ndsz,ndt);
					if (isQtable){
						out << "\t" << ibo;
					}
					else{
						if ((ibo>UPPER_SINGLE_PRECISION_OF_0 || ibo<-UPPER_SINGLE_PRECISION_OF_0)) out << "\t" << ibo;
					}
				}
				out << "\n";
			}
		}
	}
	
	//succscan iteratively returns all successors of an ngram h for which
	//get(h,h.size,h.size) returned true.
	
	int lmtable::succscan(ngram& h,ngram& ng,LMT_ACTION action,int lev)
	{
		MY_ASSERT(lev==h.lev+1 && h.size==lev && lev<=maxlev);
		
		LMT_TYPE ndt=tbltype[h.lev];
		int ndsz=nodesize(ndt);
		
		table_entry_pos_t offset;
		switch (action) {
				
			case LMT_INIT:
				//reset ngram local indexes
				
				ng.size=lev;
				ng.trans(h);
				//get number of successors of h
				ng.midx[lev]=0;
				offset=(h.link>table[h.lev]?bound(h.link-ndsz,ndt):0);
				h.succ=bound(h.link,ndt)-offset;
				h.succlink=table[lev]+(table_pos_t) offset * nodesize(tbltype[lev]);
				return 1;
				
			case LMT_CONT:
				if (ng.midx[lev] < h.succ) {
					//put current word into ng
					*ng.wordp(1)=word(h.succlink+(table_pos_t) ng.midx[lev]*nodesize(tbltype[lev]));
					ng.midx[lev]++;
					return 1;
				} else
					return 0;
				
			default:
				exit_error(IRSTLM_ERROR_MODEL, "succscan: only permitted options are LMT_INIT and LMT_CONT");
		}
		return 0;
	}
	
	ngram_state_t lmtable::convert(const char* suffptr, size_t lev){
		int ndsz=nodesize(tbltype[lev]);
		ngram_state_t suffidx=0;
		if (suffptr){
			suffidx = (ngram_state_t) ( ((table_pos_t) suffptr - (table_pos_t) table[lev]) / ndsz ) + tb_offset[lev] + 1; //added 1 to distinguish from zero-ngram
		}
		return suffidx;
	}
	
	
	//maxsuffptr returns the largest suffix of an n-gram that is contained
	//in the LM table. This can be used as a compact representation of the
	//(n-1)-gram state of a n-gram LM. If the input k-gram has k>=n then it
	//is trimmed to its n-1 suffix.
	
	//non recursive version
	const char *lmtable::maxsuffptr(ngram ong, unsigned int* size)
	{
		VERBOSE(3,"const char *lmtable::maxsuffptr(ngram ong, unsigned int* size) ong:|" << ong <<"|\n");
		
		if (ong.size==0) {
			if (size!=NULL) *size=0;
			return (char*) NULL;
		}
		
		
		if (isInverted) {
			if (ong.size>maxlev) ong.size=maxlev; //if larger than maxlen reduce size
			ngram ing=ong; //inverted ngram
			
			ing.invert(ong);
			
			get(ing,ing.size,ing.size); // dig in the trie
			if (ing.lev > 0) { //found something?
				unsigned int isize = MIN(ing.lev,(ing.size-1)); //find largest n-1 gram suffix
				if (size!=NULL)  *size=isize;
				return ing.path[isize];
			} else { // means a real unknown word!
				if (size!=NULL)  *size=0;     //default statesize for zero-gram!
				return NULL; //default stateptr for zero-gram!
			}
		} else {
			if (ong.size>0) ong.size--; //always reduced by 1 word
			
			if (ong.size>=maxlev) ong.size=maxlev-1; //if still larger or equals to maxlen reduce again
			
			if (size!=NULL) *size=ong.size; //will return the largest found ong.size
			for (ngram ng=ong; ng.size>0; ng.size--) {
				if (get(ng,ng.size,ng.size)) {
					//					if (ng.succ==0) (*size)--;
					//					if (size!=NULL) *size=ng.size;
					if (size!=NULL)
					{
						if (ng.succ==0) *size=ng.size-1;
						else *size=ng.size;
					}
					return ng.link;
				}
			}
			if (size!=NULL) *size=0;
			return NULL;
		}
	}
	
	const char *lmtable::cmaxsuffptr(ngram ong, unsigned int* size)
	{
		VERBOSE(3,"const char *lmtable::cmaxsuffptr(ngram ong, unsigned int* size) ong:|" << ong  << "|\n");
		
		if (ong.size==0) {
			if (size!=NULL) *size=0;
			return (char*) NULL;
		}
		
		if (size!=NULL) *size=ong.size; //will return the largest found ong.size
		
#ifdef PS_CACHE_ENABLE
		prob_and_state_t pst;
		
		size_t orisize=ong.size;
		if (ong.size>=maxlev) ong.size=maxlev;
		
		//cache hit
		//		if (prob_and_state_cache && ong.size==maxlev && prob_and_state_cache->get(ong.wordp(maxlev),pst)) {
		if (prob_and_state_cache[ong.size] && prob_and_state_cache[ong.size]->get(ong.wordp(ong.size),pst)) {
			*size=pst.statesize;
			return pst.state;
		}
		ong.size = orisize;
		
		//cache miss
		unsigned int isize; //internal state size variable
		char* found=(char *)maxsuffptr(ong,&isize);
		ngram_state_t msidx = convert(found,isize);
		
		//cache insert
		//IMPORTANT: this function updates only two fields (state, statesize) of the entry of the cache; the reminaing fields (logpr, bow, bol, extendible) are undefined; hence, it should not be used before the corresponding clprob()
		
		if (ong.size>=maxlev) ong.size=maxlev;
		//		if (prob_and_state_cache && ong.size==maxlev) {
		if (prob_and_state_cache[ong.size]) {
			pst.state=found;
			pst.ngramstate=msidx;
			pst.statesize=isize;
			//			prob_and_state_cache->add(ong.wordp(maxlev),pst);
			prob_and_state_cache[ong.size]->add(ong.wordp(ong.size),pst);
		}
		if (size!=NULL) *size=isize;
		return found;
#else
		return (char *)maxsuffptr(ong,size);
#endif
	}

	
	//maxsuffidx returns an index of the largest of an n-gram that is contained
	//in the LM table. This can be used as a compact representation of the
	//(n-1)-gram state of a n-gram LM. If the input k-gram has k>=n then it
	//is trimmed to its n-1 suffix.
	//non recursive version
	//It relies on the computation of maxsuffptr
	ngram_state_t lmtable::maxsuffidx(ngram ong, unsigned int* size)
	{
		VERBOSE(3,"ngram_state_t lmtable::maxsuffidx(ngram ong, unsigned int* size) ong:|" << ong  << "|\n");
		unsigned int isize;
		const char* suffptr = cmaxsuffptr(ong,&isize);
		if (size) *size=isize;
		return convert(suffptr,isize);
	}
	
	ngram_state_t lmtable::cmaxsuffidx(ngram ong, unsigned int* size)
	{
		VERBOSE(3,"ngram_state_t lmtable::cmaxsuffidx(ngram ong, unsigned int* size) ong:|" << ong  << "|\n");

		if (ong.size==0) {
			if (size!=NULL) *size=0;
			return 0;
		}
		
		if (size!=NULL) *size=ong.size; //will return the largest found ong.size
		
#ifdef PS_CACHE_ENABLE
		prob_and_state_t pst;
		
		size_t orisize=ong.size;
		if (ong.size>=maxlev) ong.size=maxlev;
		
		//cache hit
		//		if (prob_and_state_cache && ong.size==maxlev && prob_and_state_cache->get(ong.wordp(maxlev),pst)) {
		if (prob_and_state_cache[ong.size] && prob_and_state_cache[ong.size]->get(ong.wordp(ong.size),pst)) {
			*size=pst.statesize;
			return pst.ngramstate;
		}
		ong.size = orisize;
		
		//cache miss
		unsigned int isize; //internal state size variable
		char* msptr = cmaxsuffptr(ong,&isize);
		ngram_state_t msidx = convert(suffptr,isize);
		
		//cache insert
		//IMPORTANT: this function updates only two fields (ngramstate, statesize) of the entry of the cache; the reminaing fields (logpr, bow, bol, extendible) are undefined; hence, it should not be used before the corresponding clprob()
		
		if (ong.size>=maxlev) ong.size=maxlev;
		//		if (prob_and_state_cache && ong.size==maxlev) {
		if (prob_and_state_cache[ong.size]) {
			pst.state=found;
			pst.ngramstate=msidx;
			pst.statesize=isize;
			//			prob_and_state_cache->add(ong.wordp(maxlev),pst);
			prob_and_state_cache[ong.size]->add(ong.wordp(ong.size),pst);
		}
		if (size!=NULL) *size=isize;
		return msidx;
#else
		return maxsuffidx(ong,size);
#endif
	}
	
	//returns log10prob of n-gram
	//bow: backoff weight
	//bol: backoff level
	
	//additional infos related to use in Moses:
	//maxsuffptr: recombination state after the LM call
	//statesize: lenght of the recombination state
	//extensible: true if the deepest found ngram has successors
	//lastbow: bow of the deepest found ngram
	
	//non recursive version, also includes maxsuffptr and maxsuffidx
	double lmtable::lprob(ngram ong,double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr,unsigned int* statesize,bool* extendible, double *lastbow)
	{
		VERBOSE(3," lmtable::lprob(ngram) ong |" << ong  << "|\n" << std::endl);
		
		if (ong.size==0){ //sanity check
			if (maxsuffptr) *maxsuffptr=NULL;
			if (maxsuffidx) *maxsuffidx=0;
			return 0.0;
		}
		if (ong.size>maxlev) ong.size=maxlev; //adjust n-gram level to table size
		
		if (bow) *bow=0; //initialize back-off weight
		if (bol) *bol=0; //initialize bock-off level
		if (lastbow) *lastbow=0; //initialize back-off weight of the deepest found ngram

		double rbow=0,lpr=0; //output back-off weight and logprob
		float ibow,iprob;    //internal back-off weight and logprob
		
		if (isInverted) {
			ngram ing=ong; //Inverted ngram TRIE
			
			ing.invert(ong);
			get(ing,ing.size,ing.size); // dig in the trie
			if (ing.lev >0) { //found something?
				iprob=ing.prob;
				lpr = (double)(isQtable?Pcenters[ing.lev][(qfloat_t)iprob]:iprob);
				if (*ong.wordp(1)==dict->oovcode()) lpr-=logOOVpenalty; //add OOV penalty
				size_t isize=MIN(ing.lev,(ing.size-1));
				if (statesize)  *statesize=isize; //find largest n-1 gram suffix

				char* suffptr=ing.path[isize];
				
				if (maxsuffptr) *maxsuffptr=suffptr;				
				if (maxsuffidx)	*maxsuffidx = convert(suffptr,isize);

				if (extendible) *extendible=succrange(ing.path[ing.lev],ing.lev)>0;
				if (lastbow) *lastbow=(double) (isQtable?Bcenters[ing.lev][(qfloat_t)ing.bow]:ing.bow);
			} else { // means a real unknown word!
				lpr=-log(UNIGRAM_RESOLUTION)/M_LN10;
				if (statesize)  *statesize=0;     //default statesize for zero-gram!
				if (maxsuffptr) *maxsuffptr=NULL; //default stateptr for zero-gram!
				if (maxsuffidx) *maxsuffidx=0; //default state-value for zero-gram!
			}
			
			if (ing.lev < ing.size) { //compute backoff weight
				int depth=(ing.lev>0?ing.lev:1); //ing.lev=0 (real unknown word) is still a 1-gram
				if (bol) *bol=ing.size-depth;
				ing.size--; //get n-gram context
				get(ing,ing.size,ing.size); // dig in the trie
				if (ing.lev>0) { //found something?
					//collect back-off weights
					for (int l=depth; l<=ing.lev; l++) {
						//start from first back-off level
						MY_ASSERT(ing.path[l]!=NULL); //check consistency of table
						ibow=this->bow(ing.path[l],tbltype[l]);
						rbow+= (double) (isQtable?Bcenters[l][(qfloat_t)ibow]:ibow);
						//avoids bad quantization of bow of <unk>
					  //if (isQtable && (*ing.wordp(1)==dict->oovcode())) {
						if (isQtable && (*ing.wordp(ing.size)==dict->oovcode())) {
							rbow-=(double)Bcenters[l][(qfloat_t)ibow];
						}
					}
				}
			}
			
			if (bow) (*bow)=rbow;
			return rbow + lpr;
		} //Direct ngram TRIE
		else {
			MY_ASSERT((extendible == NULL) || (extendible && *extendible==false));
			//		MY_ASSERT(lastbow==NULL);
			for (ngram ng=ong; ng.size>0; ng.size--) {
				if (get(ng,ng.size,ng.size)) {
					iprob=ng.prob;
					lpr = (double)(isQtable?Pcenters[ng.size][(qfloat_t)iprob]:iprob);
					if (*ng.wordp(1)==dict->oovcode()) lpr-=logOOVpenalty; //add OOV penalty
					if (maxsuffptr || maxsuffidx || statesize) { //one extra step is needed if ng.size=ong.size
						if (ong.size==ng.size) {
							ng.size--;
							get(ng,ng.size,ng.size);
						}
						if (statesize)	*statesize=ng.size;

						char* suffptr=ng.link; //we should check ng.link != NULL
						size_t isize=ng.size;					
						
						if (maxsuffptr) *maxsuffptr=suffptr;	
						if (maxsuffidx)	*maxsuffidx = convert(suffptr,isize);
					}
					return rbow+lpr;
				} else {
					if (ng.size==1) { //means a real unknow word!
						if (statesize)  *statesize=0;
						if (maxsuffptr) *maxsuffptr=NULL; //default stateptr for zero-gram!
						if (maxsuffidx) *maxsuffidx=0; //default state-value for zero-gram!
						return rbow -log(UNIGRAM_RESOLUTION)/M_LN10;
					} else { //compute backoff
						if (bol) (*bol)++; //increase backoff level
						if (ng.lev==(ng.size-1)) { //if search stopped at previous level
							ibow=ng.bow;
							rbow+= (double) (isQtable?Bcenters[ng.lev][(qfloat_t)ibow]:ibow);
							//avoids bad quantization of bow of <unk>
							if (isQtable && (*ng.wordp(2)==dict->oovcode())) {
								rbow-=(double)Bcenters[ng.lev][(qfloat_t)ibow];
							}
						}
						if (bow) (*bow)=rbow;
					}
					
				}
			}
		}
		MY_ASSERT(0); //never pass here!!!
		return 1.0;
	}
	
	//return log10 probsL use cache memory
	double lmtable::clprob(ngram ong,double* bow, int* bol, ngram_state_t* ngramstate, char** state, unsigned int* statesize, bool* extendible, double* lastbow)
	{
		VERBOSE(3,"double lmtable::clprob(ngram ong,double* bow, int* bol, ngram_state_t* ngramstate, char** state, unsigned int* statesize, bool* extendible, double* lastbow) ong:|" << ong  << "|\n");
		
#ifdef TRACE_CACHELM
		//		if (probcache && ong.size==maxlev && sentence_id>0) {
		if (probcache && sentence_id>0) {
			*cacheout << sentence_id << " " << ong << "\n";
		}
#endif
		
		if (ong.size==0) {
			if (statesize!=NULL) *statesize=0;
			if (state!=NULL) *state=NULL;
			if (ngramstate!=NULL) *ngramstate=0;
			if (extendible!=NULL) *extendible=false;
			if (lastbow!=NULL) *lastbow=false;
			return 0.0;
		}
		
		if (ong.size>maxlev) ong.size=maxlev; //adjust n-gram level to table size
		
#ifdef PS_CACHE_ENABLE
		double logpr = 0.0;
		//cache hit
		prob_and_state_t pst_get;
		
		//		if (prob_and_state_cache && ong.size==maxlev && prob_and_state_cache->get(ong.wordp(maxlev),pst_get)) {
		if (prob_and_state_cache[ong.size] && prob_and_state_cache[ong.size]->get(ong.wordp(ong.size),pst_get)) {
			logpr=pst_get.logpr;
			if (bow) *bow = pst_get.bow;
			if (bol) *bol = pst_get.bol;
			if (state) *state = pst_get.state;
			if (ngramstate) *ngramstate = pst_get.ngramstate;
			if (statesize) *statesize = pst_get.statesize;
			if (extendible) *extendible = pst_get.extendible;
			if (lastbow) *lastbow = pst_get.lastbow;
			
			return logpr;
		}
		
		//cache miss
		
		prob_and_state_t pst_add;
		logpr = pst_add.logpr = lmtable::lprob(ong, &(pst_add.bow), &(pst_add.bol), &(pst_add.ngramstate), &(pst_add.state), &(pst_add.statesize), &(pst_add.extendible), &(pst_add.lastbow));
		
		
		if (bow) *bow = pst_add.bow;
		if (bol) *bol = pst_add.bol;
		if (state) *state = pst_add.state;
		if (ngramstate) *ngramstate = pst_add.ngramstate;
		if (statesize) *statesize = pst_add.statesize;
		if (extendible) *extendible = pst_add.extendible;
		if (lastbow) *lastbow = pst_add.lastbow;
		
		
		//		if (prob_and_state_cache && ong.size==maxlev) {
		//			prob_and_state_cache->add(ong.wordp(maxlev),pst_add);
		//    }
		if (prob_and_state_cache[ong.size]) {
			prob_and_state_cache[ong.size]->add(ong.wordp(ong.size),pst_add);
		}
		return logpr;
#else
		return lmtable::lprob(ong, bow, bol, ngramstate, state, statesize, extendible, lastbow);
#endif
	};
	
	int lmtable::succrange(node ndp,int level,table_entry_pos_t* isucc,table_entry_pos_t* esucc)
	{
		table_entry_pos_t first,last;
		LMT_TYPE ndt=tbltype[level];
		
		//get table boundaries for next level
		if (level<maxlev) {
			first = ndp>table[level]? bound(ndp-nodesize(ndt), ndt) : 0;
			last  = bound(ndp, ndt);
		} else {
			first=last=0;
		}
		if (isucc) *isucc=first;
		if (esucc) *esucc=last;
		
		return last-first;
	}
	
	
	void lmtable::stat(int level)
	{
		table_pos_t totmem=0,memory;
		float mega=1024 * 1024;
		
		cout.precision(2);
		
		cout << "lmtable class statistics\n";
		
		cout << "levels " << maxlev << "\n";
		for (int l=1; l<=maxlev; l++) {
			memory=(table_pos_t) cursize[l] * nodesize(tbltype[l]);
			cout << "lev " << l
			<< " entries "<< cursize[l]
			<< " used mem " << memory/mega << "Mb\n";
			totmem+=memory;
		}
		
		cout << "total allocated mem " << totmem/mega << "Mb\n";
		
		cout << "total number of get and binary search calls\n";
		for (int l=1; l<=maxlev; l++) {
			cout << "level " << l << " get: " << totget[l] << " bsearch: " << totbsearch[l] << "\n";
		}
		
		if (level >1 ) lmtable::getDict()->stat();
		
		stat_caches();
		
	}
	
	void lmtable::reset_mmap()
	{
#ifndef WIN32
		if (memmap>0 and memmap<=maxlev)
			for (int l=memmap; l<=maxlev; l++) {
				VERBOSE(2,"resetting mmap at level:" << l << std::endl);
				Munmap(table[l]-tableGaps[l],(table_pos_t) cursize[l]*nodesize(tbltype[l])+tableGaps[l],0);
				table[l]=(char *)MMap(diskid,PROT_READ,
															tableOffs[l], (table_pos_t)cursize[l]*nodesize(tbltype[l]),
															&tableGaps[l]);
				table[l]+=(table_pos_t)tableGaps[l];
			}
#endif
	}
	
	// ng: input n-gram
	
	// *lk: prob of n-(*bol) gram
	// *boff: backoff weight vector
	// *bol:  backoff level
	
	double lmtable::lprobx(ngram	ong,
												 double	*lkp,
												 double	*bop,
												 int	*bol)
	{
		double bo, lbo, pr;
		float		ipr;
		//int		ipr;
		ngram		ng(dict), ctx(dict);
		
		if(bol) *bol=0;
		if(ong.size==0) {
			if(lkp) *lkp=0;
			return 0;	// lprob ritorna 0, prima lprobx usava LOGZERO
		}
		if(ong.size>maxlev) ong.size=maxlev;
		ctx = ng = ong;
		bo=0;
		ctx.shift();
		while(!get(ng)) { // back-off
			
			//OOV not included in dictionary
			if(ng.size==1) {
				pr = -log(UNIGRAM_RESOLUTION)/M_LN10;
				if(lkp) *lkp=pr; // this is the innermost probability
				pr += bo; //add all the accumulated back-off probability
				return pr;
			}
			// backoff-probability
			lbo = 0.0; //local back-off: default is logprob 0
			if(get(ctx)) { //this can be replaced with (ng.lev==(ng.size-1))
				ipr = ctx.bow;
				lbo = isQtable?Bcenters[ng.size][(qfloat_t)ipr]:ipr;
				//lbo = isQtable?Bcenters[ng.size][ipr]:*(float*)&ipr;
			}
			if(bop) *bop++=lbo;
			if(bol) ++*bol;
			bo += lbo;
			ng.size--;
			ctx.size--;
		}
		ipr = ng.prob;
		pr = isQtable?Pcenters[ng.size][(qfloat_t)ipr]:ipr;
		//pr = isQtable?Pcenters[ng.size][ipr]:*((float*)&ipr);
		if(lkp) *lkp=pr;
		pr += bo;
		return pr;
	}
	
	
	// FABIO
	table_entry_pos_t lmtable::wdprune(float *thr, int aflag)
	{
		//this function implements a method similar to the "Weighted Difference Method"
		//described in "Scalable Backoff Language Models"  by Kristie Seymore	and Ronald Rosenfeld
		int	l;
		ngram	ng(lmtable::getDict(),0);
		
		isPruned=true;  //the table now might contain pruned n-grams
		
		ng.size=0;
		
		for(l=2; l<=maxlev; l++) wdprune(thr, aflag, ng, 1, l, 0, cursize[1]);
		return 0;
	}
	
	// FABIO: LM pruning method
	
	table_entry_pos_t lmtable::wdprune(float *thr, int aflag, ngram ng, int ilev, int elev, table_entry_pos_t ipos, table_entry_pos_t epos, double tlk, double bo, double *ts, double *tbs)
	{
		LMT_TYPE	ndt=tbltype[ilev];
		int		   ndsz=nodesize(ndt);
		char		 *ndp;
		float		 lk;
		float ipr, ibo;
		//int ipr, ibo;
		table_entry_pos_t i, k, nk;
		
		MY_ASSERT(ng.size==ilev-1);
		//Note that ipos and epos are always larger than or equal to 0 because they are unsigned int
		MY_ASSERT(epos<=cursize[ilev] && ipos<epos);
		
		ng.pushc(0); //increase size of n-gram
		
		for(i=ipos, nk=0; i<epos; i++) {
			
			//scan table at next level ilev from position ipos
			ndp = table[ilev]+(table_pos_t)i*ndsz;
			*ng.wordp(1) = word(ndp);
			
			//get probability
			ipr = prob(ndp, ndt);
			if(ipr==NOPROB) continue;	// Has it been already pruned ??
			
			if ((ilev == 1) && (*ng.wordp(ng.size) == getDict()->getcode(BOS_))) {
				//the n-gram starts with the sentence start symbol
				//do not consider is actual probability because it is not reliable (its frequency is manually set)
				ipr = 0.0;
			}
			lk = ipr;
			
			if(ilev<elev) { //there is an higher order
				
				//get backoff-weight for next level
				ibo = bow(ndp, ndt);
				bo = ibo;
				
				//get table boundaries for next level
				table_entry_pos_t isucc,esucc;
				succrange(ndp,ilev,&isucc,&esucc);
				
				//table_entry_pos_t isucc = i>0 ? bound(ndp-ndsz, ndt) : 0;
				//table_entry_pos_t  esucc = bound(ndp, ndt);
				if(isucc>=esucc) continue; // no successors
				
				//look for n-grams to be pruned with this context (see
				//back-off weight)
			prune:
				double nextlevel_ts=0, nextlevel_tbs=0;
				k = wdprune(thr, aflag, ng, ilev+1, elev, isucc, esucc, tlk+lk, bo, &nextlevel_ts, &nextlevel_tbs);
				//k  is the number of pruned n-grams with this context
				if(ilev!=elev-1) continue;
				if(nextlevel_ts>=1 || nextlevel_tbs>=1) {
					VERBOSE(2, "ng: " << ng <<" nextlevel_ts=" << nextlevel_ts <<" nextlevel_tbs=" << nextlevel_tbs 					<<" k=" << k <<" ns=" << esucc-isucc << "\n");
					if(nextlevel_ts>=1) {
						pscale(ilev+1, isucc, esucc, 0.999999/nextlevel_ts);
						goto prune;
					}
				}
				// adjusts backoff:
				// 1-sum_succ(pr(w|ng)) / 1-sum_succ(pr(w|bng))
				bo = log((1-nextlevel_ts)/(1-nextlevel_tbs))/M_LN10;
				ibo=(float)bo;
				bow(ndp, ndt, ibo);
			} else { //we are at the highest level
				
				//get probability of lower order n-gram
				ngram bng = ng;
				bng.size--;
				double blk = lprob(bng);
				
				double wd = pow(10., tlk+lk) * (lk-bo-blk);
				if(aflag&&wd<0) wd=-wd;
				if(wd > thr[elev-1]) {	// kept
					*ts += pow(10., lk);
					*tbs += pow(10., blk);
				} else {		// discarded
					++nk;
					prob(ndp, ndt, NOPROB);
				}
			}
		}
		return nk;
	}
	
	int lmtable::pscale(int lev, table_entry_pos_t ipos, table_entry_pos_t epos, double s)
	{
		LMT_TYPE        ndt=tbltype[lev];
		int             ndsz=nodesize(ndt);
		char            *ndp;
		float             ipr;
		
		s=log(s)/M_LN10;
		ndp = table[lev]+ (table_pos_t) ipos*ndsz;
		for(table_entry_pos_t i=ipos; i<epos; ndp+=ndsz,i++) {
			ipr = prob(ndp, ndt);
			if(ipr==NOPROB) continue;
			ipr+=(float) s;
			prob(ndp, ndt, ipr);
		}
		return 0;
	}
	
	//recompute table size by excluding pruned n-grams
	table_entry_pos_t lmtable::ngcnt(table_entry_pos_t	*cnt)
	{
		ngram	ng(lmtable::getDict(),0);
		memset(cnt, 0, (maxlev+1)*sizeof(*cnt));
		ngcnt(cnt, ng, 1, 0, cursize[1]);
		return 0;
	}
	
	//recursively compute size
	table_entry_pos_t lmtable::ngcnt(table_entry_pos_t *cnt, ngram	ng, int	l, table_entry_pos_t ipos, table_entry_pos_t	epos)
	{
		
		table_entry_pos_t	i, isucc, esucc;
		float ipr;
		char		*ndp;
		LMT_TYPE	ndt=tbltype[l];
		int		ndsz=nodesize(ndt);
		
		ng.pushc(0);
		for(i=ipos; i<epos; i++) {
			ndp = table[l]+(table_pos_t) i*ndsz;
			*ng.wordp(1)=word(ndp);
			ipr=prob(ndp, ndt);
			if(ipr==NOPROB) continue;
			++cnt[l];
			if(l==maxlev) continue;
			succrange(ndp,l,&isucc,&esucc);
			if(isucc < esucc) ngcnt(cnt, ng, l+1, isucc, esucc);
		}
		return 0;
	}
}//namespace irstlm

