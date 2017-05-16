// $Id: lmclass.cpp 3631 2010-10-07 12:04:12Z bertoldi $

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
#include "lmclass.h"
#include "util.h"

using namespace std;

// local utilities: start

int parseWords(char *sentence, const char **words, int max);

inline void error(const char* message)
{
  cerr << message << "\n";
  throw runtime_error(message);
}

// local utilities: end

namespace irstlm {
	
	lmclass::lmclass(float nlf, float dlfi):lmtable(nlf,dlfi)
	{
		MaxMapSize=1000000;
		MapScore= (double *)malloc(MaxMapSize*sizeof(double));// //array of probabilities
		memset(MapScore,0,MaxMapSize*sizeof(double));
		MapScoreN=0;
		dict = new dictionary((char *)NULL,MaxMapSize); //word to cluster dictionary
	};
	
	lmclass::~lmclass()
	{
		free (MapScore);
		delete dict;
	}
	
	void lmclass::load(const std::string &filename,int memmap)
	{
		VERBOSE(2,"lmclass::load(const std::string &filename,int memmap)" << std::endl);
		
		//get info from the configuration file
		fstream inp(filename.c_str(),ios::in|ios::binary);
		
		char line[MAX_LINE];
		const char* words[LMCLASS_MAX_TOKEN];
		int tokenN;
		inp.getline(line,MAX_LINE,'\n');
		tokenN = parseWords(line,words,LMCLASS_MAX_TOKEN);
		
		if (tokenN != 2 || ((strcmp(words[0],"LMCLASS") != 0) && (strcmp(words[0],"lmclass")!=0)))
			error((char*)"ERROR: wrong header format of configuration file\ncorrect format: LMCLASS LM_order\nfilename_of_LM\nfilename_of_map");
		
		maxlev = atoi(words[1]);
		std::string lmfilename;
		if (inp.getline(line,MAX_LINE,'\n')) {
			tokenN = parseWords(line,words,LMCLASS_MAX_TOKEN);
			lmfilename = words[0];
		} else {
			error((char*)"ERROR: wrong header format of configuration file\ncorrect format: LMCLASS LM_order\nfilename_of_LM\nfilename_of_map");
		}
		
		std::string W2Cdict = "";
		if (inp.getline(line,MAX_LINE,'\n')) {
			tokenN = parseWords(line,words,LMCLASS_MAX_TOKEN);
			W2Cdict = words[0];
		} else {
			error((char*)"ERROR: wrong header format of configuration file\ncorrect format: LMCLASS LM_order\nfilename_of_LM\nfilename_of_map");
		}
		inp.close();
		
		std::cerr << "lmfilename:" << lmfilename << std::endl;
		if (W2Cdict != "") {
			std::cerr << "mapfilename:" << W2Cdict << std::endl;
		} else {
			error((char*)"ERROR: you must specify a map!");
		}
		
		
		// Load the (possibly binary) LM
		inputfilestream inpLM(lmfilename.c_str());
		if (!inpLM.good()) {
			std::cerr << "Failed to open " << lmfilename << "!" << std::endl;
			exit(1);
		}
		lmtable::load(inpLM,lmfilename.c_str(),NULL,memmap);
		
		inputfilestream inW2C(W2Cdict);
		if (!inW2C.good()) {
			std::cerr << "Failed to open " << W2Cdict << "!" << std::endl;
			exit(1);
		}
		loadMap(inW2C);
		getDict()->genoovcode();
		
		VERBOSE(2,"OOV code of lmclass is " << getDict()->oovcode() << " mapped into " << getMap(getDict()->oovcode())<< "\n");
		getDict()->incflag(1);
	}
	
	void lmclass::loadMap(istream& inW2C)
	{
		
		double lprob=0.0;
		int howmany=0;
		
		const char* words[1 + LMTMAXLEV + 1 + 1];
		
		//open input stream and prepare an input string
		char line[MAX_LINE];
		
		dict->incflag(1); //can add to the map dictionary
		
		cerr<<"loadW2Cdict()...\n";
		//save freq of EOS and BOS
		
		loadMapElement(dict->BoS(),lmtable::dict->BoS(),0.0);
		loadMapElement(dict->EoS(),lmtable::dict->EoS(),0.0);
		
		//should i add <unk> to the dict or just let the trans_freq handle <unk>
		loadMapElement(dict->OOV(),lmtable::dict->OOV(),0.0);
		
		while (inW2C.getline(line,MAX_LINE)) {
			if (strlen(line)==MAX_LINE-1) {
				cerr << "lmtable::loadW2Cdict: input line exceed MAXLINE ("
				<< MAX_LINE << ") chars " << line << "\n";
				exit(1);
			}
			
			howmany = parseWords(line, words, 4); //3
			
			if(howmany == 3) {
				MY_ASSERT(sscanf(words[2], "%lf", &lprob));
				lprob=(double)log10(lprob);
			} else if(howmany==2) {
				
				VERBOSE(3,"No score for the pair (" << words[0] << "," << words[1] << "); set to default 1.0\n");
				
				lprob=0.0;
			} else {
				cerr << "parseline: not enough entries" << line << "\n";
				exit(1);
			}
			loadMapElement(words[0],words[1],lprob);
			
			//check if the are available position in MapScore
			checkMap();
		}
		
		VERBOSE(2,"There are " << MapScoreN << " entries in the map\n");
		
		dict->incflag(0); //can NOT add to the dictionary of lmclass
	}
	
	void lmclass::checkMap()
	{
		if (MapScoreN > MaxMapSize) {
			MaxMapSize=2*MapScoreN;
			MapScore = (double*) reallocf(MapScore, sizeof(double)*(MaxMapSize));
			VERBOSE(2,"In lmclass::checkMap(...) MaxMapSize=" <<  MaxMapSize  << " MapScoreN=" <<  MapScoreN  << "\n");
		}
	}
	
	void lmclass::loadMapElement(const char* in, const char* out, double sc)
	{
		//freq of word (in) encodes the ID of the class (out)
		//save the probability associated with the pair (in,out)
		int wcode=dict->encode(in);
		dict->freq(wcode,lmtable::dict->encode(out));
		MapScore[wcode]=sc;
		VERBOSE(3,"In lmclass::loadMapElement(...) in=" << in  << " wcode=" <<  wcode << " out=" << out << " ccode=" << lmtable::dict->encode(out) << " MapScoreN=" << MapScoreN  << "\n");
		
		if (wcode >= MapScoreN) MapScoreN++; //increment size of the array MapScore if the element is new
	}
	
	//double lmclass::lprob(ngram ong,double* bow, int* bol, char** maxsuffptr,unsigned int* statesize,bool* extendible)
	double lmclass::lprob(ngram ong,double* bow, int* bol, ngram_state_t* maxsuffidx, char** maxsuffptr, unsigned int* statesize, bool* extendible, double* lastbow)
	{
		double lpr=getMapScore(*ong.wordp(1));
		
		VERBOSE(3,"In lmclass::lprob(...) Mapscore    = " <<  lpr  << "\n");
		
		//convert ong to it's clustered encoding
		ngram mapped_ng(lmtable::getDict());
		//  mapped_ng.trans_freq(ong);
		mapping(ong,mapped_ng);
		
		//  lpr+=lmtable::clprob(mapped_ng,bow,bol,maxsuffptr,statesize, extendible);
		lpr+=lmtable::clprob(mapped_ng, bow, bol, maxsuffidx, maxsuffptr, statesize, extendible, lastbow);
		
		VERBOSE(3,"In lmclass::lprob(...) global prob  = " <<  lpr  << "\n");
		return lpr;
	}
	
	void lmclass::mapping(ngram &in, ngram &out)
	{
		int insize = in.size;
		VERBOSE(3,"In lmclass::mapping(ngram &in, ngram &out) in    = " <<  in  << "\n");
		
		// map the input sequence (in) into the corresponding output sequence (out), by applying the provided map
		for (int i=insize; i>0; i--) {
			out.pushc(getMap(*in.wordp(i)));
		}
		
		VERBOSE(3,"In lmclass::mapping(ngram &in, ngram &out) out    = " <<  out  << "\n");
		return;
	}
}//namespace irstlm

