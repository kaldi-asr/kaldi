/******************************************************************************
 IrstLM: IRST Language Model Toolkit, compile LM
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

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <stdlib.h>
#include "cmd.h"
#include "util.h"
#include "math.h"
#include "lmContainer.h"

#define MAX_N   100
/********************************/
using namespace std;
using namespace irstlm;

inline void error(const char* message)
{
  std::cerr << message << "\n";
  throw std::runtime_error(message);
}

lmContainer* load_lm(std::string file,int requiredMaxlev,int dub,int memmap, float nlf, float dlf);

void print_help(int TypeFlag=0){
  std::cerr << std::endl << "interpolate-lm - interpolates language models" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl;
	std::cerr << "       interpolate-lm [options] <lm-list-file> [lm-list-file.out]" << std::endl;
	
	std::cerr << std::endl << "DESCRIPTION:" << std::endl;
	std::cerr << "       interpolate-lm reads a LM list file including interpolation weights " << std::endl;
	std::cerr << "       with the format: N\\n w1 lm1 \\n w2 lm2 ...\\n wN lmN\n" << std::endl;
	std::cerr << "       It estimates new weights on a development text, " << std::endl;
	std::cerr << "       computes the perplexity on an evaluation text, " << std::endl;
	std::cerr << "       computes probabilities of n-grams read from stdin." << std::endl;
	std::cerr << "       It reads LMs in ARPA and IRSTLM binary format." << std::endl;
	
  std::cerr << std::endl << "OPTIONS:" << std::endl;
	FullPrintParams(TypeFlag, 0, 1, stderr);
	
}

void usage(const char *msg = 0)
{
  if (msg){
    std::cerr << msg << std::endl;
	}
  else{
		print_help();
	}
}

int main(int argc, char **argv)
{		
	char *slearn = NULL;
	char *seval = NULL;
	bool learn=false;
	bool score=false;
	bool sent_PP_flag = false;
	
	int order = 0;
	int debug = 0;
  int memmap = 0;
  int requiredMaxlev = IRSTLM_REQUIREDMAXLEV_DEFAULT;
  int dub = IRSTLM_DUB_DEFAULT;
  float ngramcache_load_factor = 0.0;
  float dictionary_load_factor = 0.0;
	
	bool help=false;
  std::vector<std::string> files;
	
	DeclareParams((char*)
					
		"learn", CMDSTRINGTYPE|CMDMSG, &slearn, "learn optimal interpolation for text-file; default is false",
		"l", CMDSTRINGTYPE|CMDMSG, &slearn, "learn optimal interpolation for text-file; default is false",
		"order", CMDINTTYPE|CMDMSG, &order, "order of n-grams used in --learn (optional)",
		"o", CMDINTTYPE|CMDMSG, &order, "order of n-grams used in --learn (optional)",						
                "eval", CMDSTRINGTYPE|CMDMSG, &seval, "computes perplexity of the specified text file",
		"e", CMDSTRINGTYPE|CMDMSG, &seval, "computes perplexity of the specified text file",
								
                "DictionaryUpperBound", CMDINTTYPE|CMDMSG, &dub, "dictionary upperbound to compute OOV word penalty: default 10^7",
                "dub", CMDINTTYPE|CMDMSG, &dub, "dictionary upperbound to compute OOV word penalty: default 10^7",
                "score", CMDBOOLTYPE|CMDMSG, &score, "computes log-prob scores of n-grams from standard input",
		"s", CMDBOOLTYPE|CMDMSG, &score, "computes log-prob scores of n-grams from standard input",
								
                "debug", CMDINTTYPE|CMDMSG, &debug, "verbose output for --eval option; default is 0",
		"d", CMDINTTYPE|CMDMSG, &debug, "verbose output for --eval option; default is 0",
                "memmap", CMDINTTYPE|CMDMSG, &memmap, "uses memory map to read a binary LM",
		"mm", CMDINTTYPE|CMDMSG, &memmap, "uses memory map to read a binary LM",
		"sentence", CMDBOOLTYPE|CMDMSG, &sent_PP_flag, "computes perplexity at sentence level (identified through the end symbol)",
                "dict_load_factor", CMDFLOATTYPE|CMDMSG, &dictionary_load_factor, "sets the load factor for ngram cache; it should be a positive real value; default is 0",
                "ngram_load_factor", CMDFLOATTYPE|CMDMSG, &ngramcache_load_factor, "sets the load factor for ngram cache; it should be a positive real value; default is false",
                "level", CMDINTTYPE|CMDMSG, &requiredMaxlev, "maximum level to load from the LM; if value is larger than the actual LM order, the latter is taken",
		"lev", CMDINTTYPE|CMDMSG, &requiredMaxlev, "maximum level to load from the LM; if value is larger than the actual LM order, the latter is taken",
								
		"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
		"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								
								(char *)NULL
								);
	
	if (argc == 1){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
	for(int i=1; i < argc; i++) {
		if(argv[i][0] != '-') files.push_back(argv[i]);
	}
	
  GetParams(&argc, &argv, (char*) NULL);
	
	if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}

  if (files.size() > 2) {
    usage();
		exit_error(IRSTLM_ERROR_DATA,"Too many arguments");
  }
	
  if (files.size() < 1) {
    usage();
		exit_error(IRSTLM_ERROR_DATA,"Must pecify a LM list file to read from");
  }

  std::string infile = files[0];
  std::string outfile="";

  if (files.size() == 1) {
    outfile=infile;
    //remove path information
    std::string::size_type p = outfile.rfind('/');
    if (p != std::string::npos && ((p+1) < outfile.size()))
      outfile.erase(0,p+1);
    outfile+=".out";
  } else
    outfile = files[1];

  std::cerr << "inpfile: " << infile << std::endl;
  learn = ((slearn != NULL)? true : false);
	
  if (learn) std::cerr << "outfile: " << outfile << std::endl;
  if (score) std::cerr << "interactive: " << score << std::endl;
  if (memmap) std::cerr << "memory mapping: " << memmap << std::endl;
  std::cerr << "loading up to the LM level " << requiredMaxlev << " (if any)" << std::endl;
  std::cerr << "order: " << order << std::endl;
  if (requiredMaxlev > 0) std::cerr << "loading up to the LM level " << requiredMaxlev << " (if any)" << std::endl;

  std::cerr << "dub: " << dub<< std::endl;

  lmContainer *lmt[MAX_N], *start_lmt[MAX_N]; //interpolated language models
  std::string lmf[MAX_N]; //lm filenames

  float w[MAX_N]; //interpolation weights
  int N;


  //Loading Language Models`
  std::cerr << "Reading " << infile << "..." << std::endl;
  std::fstream inptxt(infile.c_str(),std::ios::in);

  //std::string line;
  char line[BUFSIZ];
  const char* words[3];
  int tokenN;

  inptxt.getline(line,BUFSIZ,'\n');
  tokenN = parseWords(line,words,3);

  if (tokenN != 2 || ((strcmp(words[0],"LMINTERPOLATION") != 0) && (strcmp(words[0],"lminterpolation")!=0)))
    error((char*)"ERROR: wrong header format of configuration file\ncorrect format: LMINTERPOLATION number_of_models\nweight_of_LM_1 filename_of_LM_1\nweight_of_LM_2 filename_of_LM_2");

  N=atoi(words[1]);
  std::cerr << "Number of LMs: " << N << "..." << std::endl;
  if(N > MAX_N) {
		exit_error(IRSTLM_ERROR_DATA,"Can't interpolate more than MAX_N language models");
		
  }

  for (int i=0; i<N; i++) {
    inptxt.getline(line,BUFSIZ,'\n');
    tokenN = parseWords(line,words,3);
    if(tokenN != 2) {
			exit_error(IRSTLM_ERROR_DATA,"Wrong input format");
    }
    w[i] = (float) atof(words[0]);
    lmf[i] = words[1];

    std::cerr << "i:" << i << " w[i]:" << w[i] << " lmf[i]:" << lmf[i] << std::endl;
    start_lmt[i] = lmt[i] = load_lm(lmf[i],requiredMaxlev,dub,memmap,ngramcache_load_factor,dictionary_load_factor);
  }

  inptxt.close();

  int maxorder = 0;
  for (int i=0; i<N; i++) {
    maxorder = (maxorder > lmt[i]->maxlevel())?maxorder:lmt[i]->maxlevel();
  }

  if (order <= 0) {
    order = maxorder;
    std::cerr << "order is not set or wrongly set to a non positive value; reset to the maximum order of LMs: " << order << std::endl;
  } else if (order > maxorder) {
    order = maxorder;
    std::cerr << "order is too high; reset to the maximum order of LMs" << order << std::endl;
  }

  //Learning mixture weights
  if (learn) {
    std::vector<float> *p = new std::vector<float>[N]; //LM probabilities
    float c[N]; //expected counts
    float den,norm; //inner denominator, normalization term
    float variation=1.0; // global variation between new old params

    dictionary* dict=new dictionary(slearn,1000000,dictionary_load_factor);
    ngram ng(dict);
    int bos=ng.dict->encode(ng.dict->BoS());
    std::ifstream dev(slearn,std::ios::in);

    for(;;) {
      std::string line;
      getline(dev, line);
      if(dev.eof())
        break;
      if(dev.fail()) {
				exit_error(IRSTLM_ERROR_IO,"Problem reading input file");
      }
      std::istringstream lstream(line);
      if(line.substr(0, 29) == "###interpolate-lm:replace-lm ") {
        std::string token, newlm;
        int id;
        lstream >> token >> id >> newlm;
        if(id <= 0 || id > N) {
          std::cerr << "LM id out of range." << std::endl;
	  delete[] p;
          return 1;
        }
        id--; // count from 0 now
        if(lmt[id] != start_lmt[id])
          delete lmt[id];
        lmt[id] = load_lm(newlm,requiredMaxlev,dub,memmap,ngramcache_load_factor,dictionary_load_factor);
        continue;
      }
      while(lstream >> ng) {

        // reset ngram at begin of sentence
        if (*ng.wordp(1)==bos) {
          ng.size=1;
          continue;
        }
        if (order > 0 && ng.size > order) ng.size=order;
        for (int i=0; i<N; i++) {
          ngram ong(lmt[i]->getDict());
          ong.trans(ng);
          double logpr;
          logpr = lmt[i]->clprob(ong); //LM log-prob (using caches if available)
          p[i].push_back(pow(10.0,logpr));
        }
      }

      for (int i=0; i<N; i++) lmt[i]->check_caches_levels();
    }
    dev.close();

    while( variation > 0.01 ) {

      for (int i=0; i<N; i++) c[i]=0;	//reset counters

      for(unsigned i = 0; i < p[0].size(); i++) {
        den=0.0;
        for(int j = 0; j < N; j++)
          den += w[j] * p[j][i]; //denominator of EM formula
        //update expected counts
        for(int j = 0; j < N; j++)
          c[j] += w[j] * p[j][i] / den;
      }

      norm=0.0;
      for (int i=0; i<N; i++) norm+=c[i];

      //update weights and compute distance
      variation=0.0;
      for (int i=0; i<N; i++) {
        c[i]/=norm; //c[i] is now the new weight
        variation+=(w[i]>c[i]?(w[i]-c[i]):(c[i]-w[i]));
        w[i]=c[i]; //update weights
      }
      std::cerr << "Variation " << variation << std::endl;
    }

    //Saving results
    std::cerr << "Saving in " << outfile << "..." << std::endl;
    std::fstream outtxt(outfile.c_str(),std::ios::out);
    outtxt << "LMINTERPOLATION " << N << "\n";
    for (int i=0; i<N; i++) outtxt << w[i] << " " << lmf[i] << "\n";
    outtxt.close();
    delete[] p;
  }

  for(int i = 0; i < N; i++)
    if(lmt[i] != start_lmt[i]) {
      delete lmt[i];
      lmt[i] = start_lmt[i];
    }

  if (seval != NULL) {
    std::cerr << "Start Eval" << std::endl;

    std::cout.setf(ios::fixed);
    std::cout.precision(2);
    int i;
    int Nw=0,Noov_all=0, Noov_any=0, Nbo=0;
    double Pr,lPr;
    double logPr=0,PP=0;

    // variables for storing sentence-based Perplexity
    int sent_Nw=0, sent_Noov_all=0, sent_Noov_any=0, sent_Nbo=0;
    double sent_logPr=0,sent_PP=0;

    //normalize weights
    for (i=0,Pr=0; i<N; i++) Pr+=w[i];
    for (i=0; i<N; i++) w[i]/=Pr;

    dictionary* dict=new dictionary(NULL,1000000,dictionary_load_factor);
    dict->incflag(1);
    ngram ng(dict);
    int bos=ng.dict->encode(ng.dict->BoS());
    int eos=ng.dict->encode(ng.dict->EoS());

    std::fstream inptxt(seval,std::ios::in);

    for(;;) {
      std::string line;
      getline(inptxt, line);
      if(inptxt.eof())
        break;
      if(inptxt.fail()) {
        std::cerr << "Problem reading input file " << seval << std::endl;
        return 1;
      }
      std::istringstream lstream(line);
      if(line.substr(0, 26) == "###interpolate-lm:weights ") {
        std::string token;
        lstream >> token;
        for(int i = 0; i < N; i++) {
          if(lstream.eof()) {
            std::cerr << "Not enough weights!" << std::endl;
            return 1;
          }
          lstream >> w[i];
        }
        continue;
      }
      if(line.substr(0, 29) == "###interpolate-lm:replace-lm ") {
        std::string token, newlm;
        int id;
        lstream >> token >> id >> newlm;
        if(id <= 0 || id > N) {
          std::cerr << "LM id out of range." << std::endl;
          return 1;
        }
        id--; // count from 0 now
        delete lmt[id];
        lmt[id] = load_lm(newlm,requiredMaxlev,dub,memmap,ngramcache_load_factor,dictionary_load_factor);
        continue;
      }

      double bow;
      int bol=0;
      ngram_state_t msidx;
      char *msp;
      unsigned int statesize;

      while(lstream >> ng) {

        // reset ngram at begin of sentence
        if (*ng.wordp(1)==bos) {
          ng.size=1;
          continue;
        }
        if (order > 0 && ng.size > order) ng.size=order;


        if (ng.size>=1) {

          int  minbol=MAX_NGRAM; //minimum backoff level of the mixture
          bool OOV_all_flag=true;  //OOV flag wrt all LM[i]
          bool OOV_any_flag=false; //OOV flag wrt any LM[i]
          float logpr;

          Pr = 0.0;
          for (i=0; i<N; i++) {

            ngram ong(lmt[i]->getDict());
            ong.trans(ng);
//            logpr = lmt[i]->clprob(ong,&bow,&bol,&msp,&statesize); //actual prob of the interpolation
            logpr = lmt[i]->clprob(ong,&bow,&bol,&msidx,&msp,&statesize); //actual prob of the interpolation
            //logpr = lmt[i]->clprob(ong,&bow,&bol); //LM log-prob

            Pr+=w[i] * pow(10.0,logpr); //actual prob of the interpolation
            if (bol < minbol) minbol=bol; //backoff of LM[i]

            if (*ong.wordp(1) != lmt[i]->getDict()->oovcode()) OOV_all_flag=false; //OOV wrt LM[i]
            if (*ong.wordp(1) == lmt[i]->getDict()->oovcode()) OOV_any_flag=true; //OOV wrt LM[i]
          }

          lPr=log(Pr)/M_LN10;
          logPr+=lPr;
          sent_logPr+=lPr;

          if (debug==1) {
            std::cout << ng.dict->decode(*ng.wordp(1)) << " [" << ng.size-minbol << "]" << " ";
            if (*ng.wordp(1)==eos) std::cout << std::endl;
          }
          if (debug==2)
            std::cout << ng << " [" << ng.size-minbol << "-gram]" << " " << log(Pr) << std::endl;

          if (minbol) {
            Nbo++;  //all LMs have back-offed by at least one
            sent_Nbo++;
          }

          if (OOV_all_flag) {
            Noov_all++;  //word is OOV wrt all LM
            sent_Noov_all++;
          }
          if (OOV_any_flag) {
            Noov_any++;  //word is OOV wrt any LM
            sent_Noov_any++;
          }

          Nw++;
          sent_Nw++;

          if (*ng.wordp(1)==eos && sent_PP_flag) {
            sent_PP=exp((-sent_logPr * log(10.0)) /sent_Nw);
            std::cout << "%% sent_Nw=" << sent_Nw
                      << " sent_PP=" << sent_PP
                      << " sent_Nbo=" << sent_Nbo
                      << " sent_Noov=" << sent_Noov_all
                      << " sent_OOV=" << (float)sent_Noov_all/sent_Nw * 100.0 << "%"
                      << " sent_Noov_any=" << sent_Noov_any
                      << " sent_OOV_any=" << (float)sent_Noov_any/sent_Nw * 100.0 << "%" << std::endl;
            //reset statistics for sentence based Perplexity
            sent_Nw=sent_Noov_any=sent_Noov_all=sent_Nbo=0;
            sent_logPr=0.0;
          }


          if ((Nw % 10000)==0) std::cerr << ".";
        }
      }
    }

    PP=exp((-logPr * M_LN10) /Nw);

    std::cout << "%% Nw=" << Nw
              << " PP=" << PP
              << " Nbo=" << Nbo
              << " Noov=" << Noov_all
              << " OOV=" << (float)Noov_all/Nw * 100.0 << "%"
              << " Noov_any=" << Noov_any
              << " OOV_any=" << (float)Noov_any/Nw * 100.0 << "%" << std::endl;

  };


  if (score == true) {


    dictionary* dict=new dictionary(NULL,1000000,dictionary_load_factor);
    dict->incflag(1); // start generating the dictionary;
    ngram ng(dict);
    int bos=ng.dict->encode(ng.dict->BoS());

    double Pr,logpr;

    double bow;
    int bol=0, maxbol=0;
    unsigned int maxstatesize, statesize;
    int i,n=0;
    std::cout << "> ";
    while(std::cin >> ng) {

      // reset ngram at begin of sentence
      if (*ng.wordp(1)==bos) {
        ng.size=1;
        continue;
      }

      if (ng.size>=maxorder) {

        if (order > 0 && ng.size > order) ng.size=order;
        n++;
        maxstatesize=0;
        maxbol=0;
        Pr=0.0;
        for (i=0; i<N; i++) {
          ngram ong(lmt[i]->getDict());
          ong.trans(ng);
//          logpr = lmt[i]->clprob(ong,&bow,&bol,NULL,&statesize); //LM log-prob (using caches if available)
          logpr = lmt[i]->clprob(ong,&bow,&bol,NULL,NULL,&statesize); //LM log-prob (using caches if available)
					
          Pr+=w[i] * pow(10.0,logpr); //actual prob of the interpolation
          std::cout << "lm " << i << ":" << " logpr: " << logpr << " weight: " << w[i] << std::endl;
          if (maxbol<bol) maxbol=bol;
          if (maxstatesize<statesize) maxstatesize=statesize;
        }

        std::cout << ng << " p= " << log(Pr) << " bo= " << maxbol << " recombine= " << maxstatesize << std::endl;

        if ((n % 10000000)==0) {
          std::cerr << "." << std::endl;
          for (i=0; i<N; i++) lmt[i]->check_caches_levels();
        }

      } else {
        std::cout << ng << " p= NULL" << std::endl;
      }
      std::cout << "> ";
    }


  }

  for (int i=0; i<N; i++) delete lmt[i];

  return 0;
}

lmContainer* load_lm(std::string file,int requiredMaxlev,int dub,int memmap, float nlf, float dlf)
{
  lmContainer* lmt = lmContainer::CreateLanguageModel(file,nlf,dlf);
	
  lmt->setMaxLoadedLevel(requiredMaxlev);

  lmt->load(file,memmap);

  if (dub) lmt->setlogOOVpenalty((int)dub);

  //use caches to save time (only if PS_CACHE_ENABLE is defined through compilation flags)
  lmt->init_caches(lmt->maxlevel());
  return lmt;
}
