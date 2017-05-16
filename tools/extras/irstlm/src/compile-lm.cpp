// $Id: compile-lm.cpp 3677 2010-10-13 09:06:51Z bertoldi $

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
#include <vector>
#include <string>
#include <stdlib.h>
#include "cmd.h"
#include "util.h"
#include "math.h"
#include "lmContainer.h"

using namespace std;
using namespace irstlm;

/********************************/
void print_help(int TypeFlag=0){
  std::cerr << std::endl << "compile-lm - compiles an ARPA format LM into an IRSTLM format one" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl;
	std::cerr << "       compile-lm [options] <input-file.lm> [output-file.blm]" << std::endl;
	std::cerr << std::endl << "DESCRIPTION:" << std::endl;
	std::cerr << "       compile-lm reads a standard LM file in ARPA format and produces" << std::endl;
	std::cerr << "       a compiled representation that the IRST LM toolkit can quickly" << std::endl;
	std::cerr << "       read and process. LM file can be compressed." << std::endl;
	std::cerr << std::endl << "OPTIONS:" << std::endl;
	
	FullPrintParams(TypeFlag, 0, 1, stderr);
}

void usage(const char *msg = 0)
{
  if (msg) {
    std::cerr << msg << std::endl;
  }
	if (!msg){
		print_help();
	}
}

int main(int argc, char **argv)
{	
  char *seval=NULL;
	char *tmpdir=NULL;
	char *sfilter=NULL;
	
	bool textoutput = false;
	bool sent_PP_flag = false;
	bool invert = false;
	bool sscore = false;
	bool ngramscore = false;
	bool skeepunigrams = false;
	
	int debug = 0;
  bool memmap = false;
  bool keep_start_symbols = false; //flag to keep (or not) multiple contiguous start symbols in the n-gram; false means that just one start symbol is kept, true means that all start symbols are kept
  int requiredMaxlev = IRSTLM_REQUIREDMAXLEV_DEFAULT;
  int dub = IRSTLM_DUB_DEFAULT;
  int randcalls = 0;
  float ngramcache_load_factor = 0.0;
  float dictionary_load_factor = 0.0;
	
	bool help=false;
  std::vector<std::string> files;
	
	DeclareParams((char*)
                "text", CMDBOOLTYPE|CMDMSG, &textoutput, "output is again in text format; default is false",
                "t", CMDBOOLTYPE|CMDMSG, &textoutput, "output is again in text format; default is false",
                "filter", CMDSTRINGTYPE|CMDMSG, &sfilter, "filter a binary language model with a word list",
                "f", CMDSTRINGTYPE|CMDMSG, &sfilter, "filter a binary language model with a word list",
                "keepunigrams", CMDBOOLTYPE|CMDMSG, &skeepunigrams, "filter by keeping all unigrams in the table, default  is true",
                "ku", CMDBOOLTYPE|CMDMSG, &skeepunigrams, "filter by keeping all unigrams in the table, default  is true",
                "eval", CMDSTRINGTYPE|CMDMSG, &seval, "computes perplexity of the specified text file",
		"e", CMDSTRINGTYPE|CMDMSG, &seval, "computes perplexity of the specified text file",
                "randcalls", CMDINTTYPE|CMDMSG, &randcalls, "computes N random calls on the specified text file",
		"r", CMDINTTYPE|CMDMSG, &randcalls, "computes N random calls on the specified text file",
                "score", CMDBOOLTYPE|CMDMSG, &sscore, "computes log-prob scores of n-grams from standard input",
		"s", CMDBOOLTYPE|CMDMSG, &sscore, "computes log-prob scores of n-grams from standard input",
                "ngramscore", CMDBOOLTYPE|CMDMSG, &ngramscore, "computes log-prob scores of the last n-gram  before an _END_NGRAM_ symbol from standard input",
                "ns", CMDBOOLTYPE|CMDMSG, &ngramscore, "computes log-prob scores of the last n-gram  before an _END_NGRAM_ symbol from standard input",
		"debug", CMDINTTYPE|CMDMSG, &debug, "verbose output for --eval option; default is 0",
		"d", CMDINTTYPE|CMDMSG, &debug, "verbose output for --eval option; default is 0",
                "level", CMDINTTYPE|CMDMSG, &requiredMaxlev, "maximum level to load from the LM; if value is larger than the actual LM order, the latter is taken",
		"l", CMDINTTYPE|CMDMSG, &requiredMaxlev, "maximum level to load from the LM; if value is larger than the actual LM order, the latter is taken",
                "memmap", CMDBOOLTYPE|CMDMSG, &memmap, "uses memory map to read a binary LM",
		"mm", CMDBOOLTYPE|CMDMSG, &memmap, "uses memory map to read a binary LM",
                "keep-start-symbols", CMDBOOLTYPE|CMDMSG, &keep_start_symbols, "keeps (or not) multiple contiguous start symbols in the n-grams; false means that just one start symbol is kept, true means that all start symbols are kept",
                "dub", CMDINTTYPE|CMDMSG, &dub, "dictionary upperbound to compute OOV word penalty: default 10^7",
                "tmpdir", CMDSTRINGTYPE|CMDMSG, &tmpdir, "directory for temporary computation, default is either the environment variable TMP if defined or \"/tmp\")",
                "invert", CMDBOOLTYPE|CMDMSG, &invert, "builds an inverted n-gram binary table for fast access; default if false",
		"i", CMDBOOLTYPE|CMDMSG, &invert, "builds an inverted n-gram binary table for fast access; default if false",
                "sentence", CMDBOOLTYPE|CMDMSG, &sent_PP_flag, "computes perplexity at sentence level (identified through the end symbol)",
                "dict_load_factor", CMDFLOATTYPE|CMDMSG, &dictionary_load_factor, "sets the load factor for ngram cache; it should be a positive real value; default is 0",
                "ngram_load_factor", CMDFLOATTYPE|CMDMSG, &ngramcache_load_factor, "sets the load factor for ngram cache; it should be a positive real value; default is false",

		"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
		"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								
                (char*)NULL
		);
	
	if (argc == 1){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
	for(int i=1; i < argc; i++) {
		if(argv[i][0] != '-'){
			files.push_back(argv[i]);
		}
	}
	
	
	GetParams(&argc, &argv, (char*) NULL);
	
	if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}	

  if (files.size() > 2) {
    usage();
		exit_error(IRSTLM_ERROR_DATA,"Warning: Too many arguments");
  }

  if (files.size() < 1) {
    usage();
		exit_error(IRSTLM_ERROR_DATA,"Warning: Please specify a LM file to read from");
  }

  std::string infile = files[0];
  std::string outfile = "";

  if (files.size() == 1) {
    outfile=infile;

    //remove path information
    std::string::size_type p = outfile.rfind('/');
    if (p != std::string::npos && ((p+1) < outfile.size()))
      outfile.erase(0,p+1);

    //eventually strip .gz
    if (outfile.compare(outfile.size()-3,3,".gz")==0)
      outfile.erase(outfile.size()-3,3);

    outfile+=(textoutput?".lm":".blm");
  } else{
    outfile = files[1];
  }
	
  std::cerr << "inpfile: " << infile << std::endl;
	std::cerr << "outfile: " << outfile << std::endl;
  if (seval!=NULL) std::cerr << "evalfile: " << seval << std::endl;
  if (sscore==true) std::cerr << "interactive: " << sscore << std::endl;
  if (ngramscore==true) std::cerr << "interactive for ngrams only: " << ngramscore << std::endl;
  if (memmap) std::cerr << "memory mapping: " << memmap << std::endl;
  std::cerr << "loading up to the LM level " << requiredMaxlev << " (if any)" << std::endl;
  std::cerr << "dub: " << dub<< std::endl;
  if (tmpdir != NULL) {
    if (setenv("TMP",tmpdir,1))
      std::cerr << "temporary directory has not been set" << std::endl;
    std::cerr << "tmpdir: " << tmpdir << std::endl;
  }


  //checking the language model type
  lmContainer* lmt = lmContainer::CreateLanguageModel(infile,ngramcache_load_factor,dictionary_load_factor);
	
  //let know that table has inverted n-grams
  if (invert) lmt->is_inverted(invert);

  lmt->setMaxLoadedLevel(requiredMaxlev);

  lmt->load(infile);

	lmt->print_table_stat();
	
  //CHECK this part for sfilter to make it possible only for LMTABLE
  if (sfilter != NULL) {
    lmContainer* filtered_lmt = NULL;
    std::cerr << "BEFORE sublmC (" << (void*) filtered_lmt <<  ") (" << (void*) &filtered_lmt << ")\n";

    // the function filter performs the filtering and returns true, only for specific lm type
    if (((lmContainer*) lmt)->filter(sfilter,filtered_lmt,skeepunigrams?"yes":"no")) {
      std::cerr << "BFR filtered_lmt (" << (void*) filtered_lmt << ") (" << (void*) &filtered_lmt << ")\n";
      filtered_lmt->stat();
      delete lmt;
      lmt=filtered_lmt;
      std::cerr << "AFTER filtered_lmt (" << (void*) filtered_lmt << ")\n";
      filtered_lmt->stat();
      std::cerr << "AFTER lmt (" << (void*) lmt << ")\n";
      lmt->stat();
    }
  }

  if (dub) lmt->setlogOOVpenalty((int)dub);

  //use caches to save time (only if PS_CACHE_ENABLE is defined through compilation flags)
  lmt->init_caches(lmt->maxlevel());

  if (seval != NULL) {
    if (randcalls>0) {

      cerr << "perform random " << randcalls << " using dictionary of test set\n";
      dictionary *dict;
      dict=new dictionary(seval);

      //build extensive histogram
      int histo[dict->totfreq()]; //total frequency
      int totfreq=0;

      for (int n=0; n<dict->size(); n++)
        for (int m=0; m<dict->freq(n); m++)
          histo[totfreq++]=n;

      ngram ng(lmt->getDict());
      srand(1234);
      double bow;
      int bol=0;

      if (debug>1) ResetUserTime();

      for (int n=0; n<randcalls; n++) {
        //extracts a random word from dict
        int w=histo[rand() % totfreq];

        ng.pushc(lmt->getDict()->encode(dict->decode(w)));

        lmt->clprob(ng,&bow,&bol);  //(using caches if available)

        if (debug==1) {
          std::cout << ng.dict->decode(*ng.wordp(1)) << " [" << lmt->maxlevel()-bol << "]" << " ";
          std::cout << std::endl;
          std::cout.flush();
        }

        if ((n % 100000)==0) {
          std::cerr << ".";
          lmt->check_caches_levels();
        }
      }
      std::cerr << "\n";
      if (debug>1) PrintUserTime("Finished in");
      if (debug>1) lmt->stat();

      delete lmt;
      return 0;

    } else {
      if (lmt->getLanguageModelType() == _IRSTLM_LMINTERPOLATION) {
        debug = (debug>4)?4:debug;
        std::cerr << "Maximum debug value for this LM type: " << debug << std::endl;
      }
      if (lmt->getLanguageModelType() == _IRSTLM_LMMACRO) {
        debug = (debug>4)?4:debug;
        std::cerr << "Maximum debug value for this LM type: " << debug << std::endl;
      }
      if (lmt->getLanguageModelType() == _IRSTLM_LMCLASS) {
        debug = (debug>4)?4:debug;
        std::cerr << "Maximum debug value for this LM type: " << debug << std::endl;
      }
      std::cerr << "Start Eval" << std::endl;
      std::cerr << "OOV code: " << lmt->getDict()->oovcode() << std::endl;
      ngram ng(lmt->getDict());
      std::cout.setf(ios::fixed);
      std::cout.precision(2);

      //			if (debug>0) std::cout.precision(8);
      std::fstream inptxt(seval,std::ios::in);
			
      int Nbo=0, Nw=0,Noov=0;
      double logPr=0,PP=0,PPwp=0,Pr;

      // variables for storing sentence-based Perplexity
      int sent_Nbo=0, sent_Nw=0,sent_Noov=0;
      double sent_logPr=0,sent_PP=0,sent_PPwp=0;

      int bos=lmt->addWord(lmt->getDict()->BoS());
      int eos=lmt->addWord(lmt->getDict()->EoS());
			
      double bow;
      int bol=0;
      ngram_state_t msidx;
      char *msp;
      unsigned int statesize;

      lmt->dictionary_incflag(1);

      while(inptxt >> ng) {
	VERBOSE(3,"read ng:|" << ng << "|" << std::endl);

        if (ng.size>lmt->maxlevel()) ng.size=lmt->maxlevel();

        // reset ngram at begin of sentence
        if (*ng.wordp(1)==bos) {
	  if (!keep_start_symbols) ng.size=1;
          continue;
        }

        if (ng.size>=1) {
  	VERBOSE(3,"computing clprob ng:|" << ng << "|" << std::endl);
//          Pr=lmt->clprob(ng,&bow,&bol,&msp,&statesize);
          Pr=lmt->clprob(ng,&bow,&bol,&msidx,&msp,&statesize);
					
          logPr+=Pr;
          sent_logPr+=Pr;

          if (debug==1) {
            std::cout << ng.dict->decode(*ng.wordp(1)) << " [" << ng.size-bol << "]" << " ";
            if (*ng.wordp(1)==eos) std::cout << std::endl;
          }
          else if (debug==2) {
            std::cout << ng << " [" << ng.size-bol << "-gram]" << " " << Pr;
            std::cout << std::endl;
            std::cout.flush();
          }
          else if (debug==3) {
            std::cout << ng << " [" << ng.size-bol << "-gram]" << " " << Pr << " bow:" << bow;
            std::cout << std::endl;
            std::cout.flush();
          }
          else if (debug==4) {
            std::cout << ng << " [" << ng.size-bol << "-gram: recombine:" << statesize << " ngramstate:" << msidx << " state:" << (void*) msp << "] [" << ng.size+1-((bol==0)?(1):bol) << "-gram: bol:" << bol << "] " << Pr << " bow:" << bow;
            std::cout << std::endl;
            std::cout.flush();
          }
          else if (debug>4) {
            std::cout << ng << " [" << ng.size-bol << "-gram: recombine:" << statesize << " ngramstate:" << msidx << " state:" << (void*) msp << "] [" << ng.size+1-((bol==0)?(1):bol) << "-gram: bol:" << bol << "] " << Pr << " bow:" << bow;
            double totp=0.0;
            int oldw=*ng.wordp(1);
            double oovp=lmt->getlogOOVpenalty();
            lmt->setlogOOVpenalty((double) 0);
            for (int c=0; c<ng.dict->size(); c++) {
              *ng.wordp(1)=c;
              totp+=pow(10.0,lmt->clprob(ng)); //using caches if available
            }
            *ng.wordp(1)=oldw;

            if ( totp < (1.0 - 1e-5) || totp > (1.0 + 1e-5))
              std::cout << "  [t=" << totp << "] POSSIBLE ERROR";
            std::cout << std::endl;
            std::cout.flush();

            lmt->setlogOOVpenalty((double)oovp);
          }


          if (lmt->is_OOV(*ng.wordp(1))) {
            Noov++;
            sent_Noov++;
          }
          if (bol) {
            Nbo++;
            sent_Nbo++;
          }
          Nw++;
          sent_Nw++;
          if (sent_PP_flag && (*ng.wordp(1)==eos)) {
            sent_PP=exp((-sent_logPr * log(10.0)) /sent_Nw);
            sent_PPwp= sent_PP * (1 - 1/exp((sent_Noov *  lmt->getlogOOVpenalty()) * log(10.0) / sent_Nw));

            std::cout << "%% sent_Nw=" << sent_Nw
                      << " sent_PP=" << sent_PP
                      << " sent_PPwp=" << sent_PPwp
                      << " sent_Nbo=" << sent_Nbo
                      << " sent_Noov=" << sent_Noov
                      << " sent_OOV=" << (float)sent_Noov/sent_Nw * 100.0 << "%" << std::endl;
            std::cout.flush();
            //reset statistics for sentence based Perplexity
            sent_Nw=sent_Noov=sent_Nbo=0;
            sent_logPr=0.0;
          }

          if ((Nw % 100000)==0) {
            std::cerr << ".";
            lmt->check_caches_levels();
          }
					
					VERBOSE(3,"computing clprob END" << std::endl);
        }
				VERBOSE(3,"read END" << std::endl);
      }

      PP=exp((-logPr * log(10.0)) /Nw);

      PPwp= PP * (1 - 1/exp((Noov *  lmt->getlogOOVpenalty()) * log(10.0) / Nw));

      std::cout << "%% Nw=" << Nw
                << " PP=" << PP
                << " PPwp=" << PPwp
                << " Nbo=" << Nbo
                << " Noov=" << Noov
                << " OOV=" << (float)Noov/Nw * 100.0 << "%";
      if (debug) std::cout << " logPr=" <<  logPr;
      std::cout << std::endl;
      std::cout.flush();

      if (debug>1) lmt->used_caches();

      if (debug>1) lmt->stat();

      delete lmt;
      return 0;
    };
  }

  if (sscore == true) {

    ngram ng(lmt->getDict());
    int bos=ng.dict->encode(ng.dict->BoS());

    int bol;
    double bow;
    unsigned int n=0;

    std::cout.setf(ios::scientific);
    std::cout.setf(ios::fixed);
    std::cout.precision(2);
    std::cout << "> ";

    lmt->dictionary_incflag(1);

    while(std::cin >> ng) {

      //std::cout << ng << std::endl;;
      // reset ngram at begin of sentence
      if (*ng.wordp(1)==bos) {
        if (!keep_start_symbols) ng.size=1;
        continue;
      }

      if (ng.size>=lmt->maxlevel()) {
        ng.size=lmt->maxlevel();
        ++n;
        if ((n % 100000)==0) {
          std::cerr << ".";
          lmt->check_caches_levels();
        }
        std::cout << ng << " p= " << lmt->clprob(ng,&bow,&bol) * M_LN10;
        std::cout << " bo= " << bol << std::endl;
      } else {
        std::cout << ng << " p= NULL" << std::endl;
      }
      std::cout << "> ";
    }
    std::cout << std::endl;
    std::cout.flush();
    if (debug>1) lmt->used_caches();

    if (debug>1) lmt->stat();

    delete lmt;
    return 0;
  }
	
	
  if (ngramscore == true) {
		
		const char* _END_NGRAM_="_END_NGRAM_";
    ngram ng(lmt->getDict());
		
		double Pr;
		double bow;
		int bol=0;
		ngram_state_t msidx;
		char *msp;
		unsigned int statesize;

		std::cout.setf(ios::fixed);
		std::cout.precision(2);
		
		ng.dict->incflag(1);
		int endngram=ng.dict->encode(_END_NGRAM_);
		ng.dict->incflag(0);
		
    while(std::cin >> ng) {
      // compute score for the last ngram when endngram symbols is found
      // and reset ngram
      if (*ng.wordp(1)==endngram) {
				ng.shift();
				if (ng.size>=lmt->maxlevel()) {
					ng.size=lmt->maxlevel();
				}
				
//				Pr=lmt->clprob(ng,&bow,&bol,&msp,&statesize);
				Pr=lmt->clprob(ng,&bow,&bol,&msidx, &msp,&statesize);
#ifndef OUTPUT_SUPPRESSED
				std::cout << ng << " [" << ng.size-bol << "-gram: recombine:" << statesize << " ngramstate:" << msidx << " state:" << (void*) msp << "] [" << ng.size+1-((bol==0)?(1):bol) << "-gram: bol:" << bol << "] " << Pr << " bow:" << bow;
				std::cout << std::endl;
    				std::cout.flush();
#endif
        ng.size=0;
      }
    }

    if (debug>1) lmt->used_caches();

    if (debug>1) lmt->stat();

    delete lmt;
    return 0;
  }

  if (textoutput == true) {
    std::cerr << "Saving in txt format to " << outfile << std::endl;
    lmt->savetxt(outfile.c_str());
  } else if (!memmap) {
    std::cerr << "Saving in bin format to " << outfile << std::endl;
    lmt->savebin(outfile.c_str());
  } else {
    std::cerr << "Impossible to save to " << outfile << std::endl;
  }
  delete lmt;
  return 0;
}

