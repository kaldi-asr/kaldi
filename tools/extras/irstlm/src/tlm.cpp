
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

#include <iostream>
#include <cmath>
#include <math.h>
#include "cmd.h"
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "mempool.h"
#include "ngramtable.h"
#include "interplm.h"
#include "normcache.h"
#include "ngramcache.h"
#include "mdiadapt.h"
#include "shiftlm.h"
#include "linearlm.h"
#include "mixture.h"
#include "lmtable.h"

/********************************/
using namespace std;
using namespace irstlm;


#define NGRAM 1
#define SEQUENCE 2
#define ADAPT 3
#define TURN 4
#define TEXT 5

static Enum_T LmTypeEnum [] = {
  {    (char*)"ImprovedKneserNey",  IMPROVED_KNESER_NEY },
  {    (char*)"ikn",                IMPROVED_KNESER_NEY },
  {    (char*)"KneserNey",          KNESER_NEY },
  {    (char*)"kn",                 KNESER_NEY },
  {    (char*)"ModifiedShiftBeta",  MOD_SHIFT_BETA },
  {    (char*)"msb",                MOD_SHIFT_BETA },
  {    (char*)"ImprovedShiftBeta",  IMPROVED_SHIFT_BETA },
  {    (char*)"isb",                IMPROVED_SHIFT_BETA },
  {    (char*)"InterpShiftBeta",    SHIFT_BETA },
  {    (char*)"ShiftBeta",          SHIFT_BETA },
  {    (char*)"sb",                 SHIFT_BETA },
  {    (char*)"InterpShiftOne",     SHIFT_ONE },
  {    (char*)"ShiftOne",           SHIFT_ONE },
  {    (char*)"s1",                 SHIFT_ONE },
  {    (char*)"LinearWittenBell",   LINEAR_WB },
  {    (char*)"wb",                 LINEAR_WB },
  {    (char*)"StupidBackoff",			LINEAR_STB },
  {    (char*)"stb",                LINEAR_STB },
  {    (char*)"LinearGoodTuring",   LINEAR_GT },
  {    (char*)"Mixture",            MIXTURE },
  {    (char*)"mix",                MIXTURE },
  END_ENUM
};

static Enum_T InteractiveModeEnum [] = {
  {    (char*)"Ngram",       NGRAM },
  {    (char*)"Sequence",    SEQUENCE },
  {    (char*)"Adapt",       ADAPT },
  {    (char*)"Turn",        TURN },
  {    (char*)"Text",        TEXT },
  {    (char*)"Yes",         NGRAM },
  END_ENUM
};

void print_help(int TypeFlag=0){
  std::cerr << std::endl << "tlm - estimates a language model" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl;
  std::cerr << "       not yet available" << std::endl;
  std::cerr << std::endl << "DESCRIPTION:" << std::endl;
  std::cerr << "       tlm is a tool for the estimation of language model" << std::endl;
  std::cerr << std::endl << "OPTIONS:" << std::endl;
  std::cerr << "       -Help|-h this help" << std::endl;
  std::cerr << std::endl;
	
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
	char *dictfile=NULL;
	char *trainfile=NULL;
	char *testfile=NULL;
	char *adaptfile=NULL;
	char *slminfo=NULL;
	
	char *imixpar=NULL;
	char *omixpar=NULL;
	
	char *BINfile=NULL;
	char *ARPAfile=NULL;
        bool SavePerLevel=true; //save-per-level or save-for-word
	
	char *ASRfile=NULL;
	
	char* scalefactorfile=NULL;
	
	bool backoff=false; //back-off or interpolation
	int lmtype=0;
	int dub=IRSTLM_DUB_DEFAULT; //dictionary upper bound
	int size=0;   //lm size
	
	int interactive=0;
	int statistics=0;
	
	int prunefreq=0;
	bool prunesingletons=true;
	bool prunetopsingletons=false;
	char *prune_thr_str=NULL;
	
	double beta=-1;
	
	bool compsize=false;
	bool checkpr=false;
	double oovrate=0;
	int max_caching_level=0;
	
	char *outpr=NULL;
	
	bool memmap = false; //write binary format with/without memory map, default is 0
	
	int adaptlevel=0;   //adaptation level
	double adaptrate=1.0;
	bool adaptoov=false; //do not increment the dictionary
	
	bool help=false;
	
	DeclareParams((char*)
		"Back-off",CMDBOOLTYPE|CMDMSG, &backoff, "boolean flag for backoff LM (default is false, i.e. interpolated LM)",
		"bo",CMDBOOLTYPE|CMDMSG, &backoff, "boolean flag for backoff LM (default is false, i.e. interpolated LM)",
		"Dictionary", CMDSTRINGTYPE|CMDMSG, &dictfile, "dictionary to filter the LM (default is NULL)",
		"d", CMDSTRINGTYPE|CMDMSG, &dictfile, "dictionary to filter the LM (default is NULL)",
								
		"DictionaryUpperBound", CMDINTTYPE|CMDMSG, &dub, "dictionary upperbound to compute OOV word penalty: default 10^7",
		"dub", CMDINTTYPE|CMDMSG, &dub, "dictionary upperbound to compute OOV word penalty: default 10^7",
								
		"NgramSize", CMDSUBRANGETYPE|CMDMSG, &size, 1, MAX_NGRAM, "order of the LM",
		"n", CMDSUBRANGETYPE|CMDMSG, &size, 1, MAX_NGRAM, "order of the LM",
								
		"Ngram", CMDSTRINGTYPE|CMDMSG, &trainfile, "training file",
		"TrainOn", CMDSTRINGTYPE|CMDMSG, &trainfile, "training file",
		"tr", CMDSTRINGTYPE|CMDMSG, &trainfile, "training file",
								
		"oASR", CMDSTRINGTYPE|CMDMSG, &ASRfile, "output file in ASR format",
		"oasr", CMDSTRINGTYPE|CMDMSG, &ASRfile, "output file in ASR format",
								
		"o", CMDSTRINGTYPE|CMDMSG, &ARPAfile, "output file in ARPA format",
		"oARPA", CMDSTRINGTYPE|CMDMSG, &ARPAfile, "output file in ARPA format",
		"oarpa", CMDSTRINGTYPE|CMDMSG, &ARPAfile, "output file in ARPA format",
								
		"oBIN", CMDSTRINGTYPE|CMDMSG, &BINfile, "output file in binary format",
		"obin", CMDSTRINGTYPE|CMDMSG, &BINfile, "output file in binary format",

		"SavePerLevel",CMDBOOLTYPE|CMDMSG, &SavePerLevel, "saving type of the LM (true: per level (default), false: per word)",
		"spl",CMDBOOLTYPE|CMDMSG, &SavePerLevel, "saving type of the LM (true: per level (default), false: per word)",
								
		"TestOn", CMDSTRINGTYPE|CMDMSG, &testfile, "file for testing",
		"te", CMDSTRINGTYPE|CMDMSG, &testfile, "file for testing",
								
		"AdaptOn", CMDSTRINGTYPE|CMDMSG, &adaptfile, "file for adaptation",
		"ad", CMDSTRINGTYPE|CMDMSG, &adaptfile, "file for adaptation",
							
		"AdaptRate",CMDDOUBLETYPE|CMDMSG , &adaptrate, "adaptation rate",
		"ar", CMDDOUBLETYPE|CMDMSG, &adaptrate, "adaptation rate",
							
		"AdaptLevel", CMDSUBRANGETYPE|CMDMSG, &adaptlevel, 1 , MAX_NGRAM, "adaptation level",
		"al",CMDSUBRANGETYPE|CMDMSG, &adaptlevel, 1, MAX_NGRAM, "adaptation level",
								
		"AdaptOOV", CMDBOOLTYPE|CMDMSG, &adaptoov, "boolean flag for increasing the dictionary during adaptation (default is false)",
		"ao", CMDBOOLTYPE|CMDMSG, &adaptoov, "boolean flag for increasing the dictionary during adaptation (default is false)",
								
		"SaveScaleFactor", CMDSTRINGTYPE|CMDMSG, &scalefactorfile, "output file for the scale factors",
		"ssf", CMDSTRINGTYPE|CMDMSG, &scalefactorfile, "output file for the scale factors",
								
		"LanguageModelType",CMDENUMTYPE|CMDMSG, &lmtype, LmTypeEnum, "type of the LM",
		"lm",CMDENUMTYPE|CMDMSG, &lmtype, LmTypeEnum, "type of the LM",
								
		"Interactive",CMDENUMTYPE|CMDMSG, &interactive, InteractiveModeEnum, "type of interaction",
		"i",CMDENUMTYPE|CMDMSG, &interactive, InteractiveModeEnum, "type of interaction",
								
		"Statistics",CMDSUBRANGETYPE|CMDMSG, &statistics, 1, 3, "output statistics of the LM of increasing detail (default is 0)",
		"s",CMDSUBRANGETYPE|CMDMSG, &statistics, 1, 3, "output statistics of the LM of increasing detail (default is 0)",
					
		"PruneThresh",CMDSUBRANGETYPE|CMDMSG, &prunefreq, 0, 1000, "threshold for pruning (default is 0)",
		"p",CMDSUBRANGETYPE|CMDMSG, &prunefreq, 0, 1000, "threshold for pruning (default is 0)",
							
		"PruneSingletons",CMDBOOLTYPE|CMDMSG, &prunesingletons, "boolean flag for pruning of singletons (default is true)",
		"ps",CMDBOOLTYPE|CMDMSG, &prunesingletons, "boolean flag for pruning of singletons (default is true)",
								
		"PruneTopSingletons",CMDBOOLTYPE|CMDMSG, &prunetopsingletons, "boolean flag for pruning of singletons at the top level (default is false)",
		"pts",CMDBOOLTYPE|CMDMSG, &prunetopsingletons, "boolean flag for pruning of singletons at the top level (default is false)",

                "PruneFrequencyThreshold",CMDSTRINGTYPE|CMDMSG, &prune_thr_str, "pruning frequency threshold for each level; comma-separated list of values; (default is \"0,0,...,0\", for all levels)",
                "pft",CMDSTRINGTYPE|CMDMSG, &prune_thr_str, "pruning frequency threshold for each level; comma-separated list of values; (default is \"0,0,...,0\", for all levels)",

		"ComputeLMSize",CMDBOOLTYPE|CMDMSG, &compsize, "boolean flag for output the LM size (default is false)",
		"sz",CMDBOOLTYPE|CMDMSG, &compsize, "boolean flag for output the LM size (default is false)",
								
		"MaximumCachingLevel", CMDINTTYPE|CMDMSG , &max_caching_level, "maximum level for caches (default is: LM order - 1)",
		"mcl", CMDINTTYPE|CMDMSG, &max_caching_level, "maximum level for caches (default is: LM order - 1)",
								
		"MemoryMap", CMDBOOLTYPE|CMDMSG, &memmap, "use memory mapping for bianry saving (default is false)",
		"memmap", CMDBOOLTYPE|CMDMSG, &memmap, "use memory mapping for bianry saving (default is false)",
		"mm", CMDBOOLTYPE|CMDMSG, &memmap, "use memory mapping for bianry saving (default is false)",
								
		"CheckProb",CMDBOOLTYPE|CMDMSG, &checkpr, "boolean flag for checking probability distribution during test (default is false)",
		"cp",CMDBOOLTYPE|CMDMSG, &checkpr, "boolean flag for checking probability distribution during test (default is false)",
								
		"OutProb",CMDSTRINGTYPE|CMDMSG, &outpr, "output file for debugging  during test (default is \"/dev/null\")",
		"op",CMDSTRINGTYPE|CMDMSG, &outpr, "output file for debugging  during test (default is \"/dev/null\")",
								
		"SubLMInfo", CMDSTRINGTYPE|CMDMSG, &slminfo, "configuration file for the mixture LM",
		"slmi", CMDSTRINGTYPE|CMDMSG, &slminfo, "configuration file for the mixture LM",
								
		"SaveMixParam", CMDSTRINGTYPE|CMDMSG, &omixpar, "output file for weights of the mixture LM",
		"smp", CMDSTRINGTYPE|CMDMSG, &omixpar, "output file for weights of the mixture LM",
								
		"LoadMixParam", CMDSTRINGTYPE|CMDMSG, &imixpar, "input file for weights of the mixture LM",
		"lmp", CMDSTRINGTYPE|CMDMSG, &imixpar, "input file for weights of the mixture LM",
								
		"SetOovRate", CMDDOUBLETYPE|CMDMSG, &oovrate, "rate for computing the OOV frequency (=oovrate*totfreq if oovrate>0) (default is 0)",
		"or", CMDDOUBLETYPE|CMDMSG, &oovrate, "rate for computing the OOV frequency (=oovrate*totfreq if oovrate>0) (default is 0)",
								
		"Beta", CMDDOUBLETYPE|CMDMSG, &beta, "beta value for Shift-Beta and Kneser-Ney LMs (default is -1, i.e. automatic estimation)",
		"beta", CMDDOUBLETYPE|CMDMSG, &beta, "beta value for Shift-Beta and Kneser-Ney LMs (default is -1, i.e. automatic estimation)",
								
		"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
		"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								
		(char *)NULL
		);
	
	if (argc == 1){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
	GetParams(&argc, &argv, (char*) NULL);
	
	if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
	if (!lmtype) {
		exit_error(IRSTLM_ERROR_DATA,"The lm type (-lm) is not specified");
	}
	
	if (!trainfile && lmtype!=MIXTURE) {
		exit_error(IRSTLM_ERROR_DATA,"The LM file (-tr) is not specified");
	}

	if (SavePerLevel == false && backoff == true){
		cerr << "WARNING: Current implementation does not support the usage of backoff (-bo=true) mixture models (-lm=mix) combined with the per-word saving (-saveperllevel=false)." << endl;
		cerr << "WARNING: The usage of backoff is disabled, i.e. -bo=no  is forced" << endl;
		
		backoff=false;
	}
	
	mdiadaptlm *lm=NULL;
	
	switch (lmtype) {
			
		case SHIFT_BETA:
			if (beta==-1 || (beta<1.0 && beta>0)){
				lm=new shiftbeta(trainfile,size,prunefreq,beta,(backoff?SHIFTBETA_B:SHIFTBETA_I));
			}	else {
				exit_error(IRSTLM_ERROR_DATA,"ShiftBeta: beta must be >0 and <1");
			}
			break;
			
		case KNESER_NEY:
			if (size>1){
				if (beta==-1 || (beta<1.0 && beta>0)){
//					lm=new kneserney(trainfile,size,prunefreq,beta,(backoff?KNESERNEY_B:KNESERNEY_I));
				}	else {
					exit_error(IRSTLM_ERROR_DATA,"Kneser-Ney: beta must be >0 and <1");
				}
			}	else {
				exit_error(IRSTLM_ERROR_DATA,"Kneser-Ney requires size >1");
			}
			break;

		case MOD_SHIFT_BETA:
			cerr << "ModifiedShiftBeta (msb) is the old name for ImprovedKneserNey (ikn); this name is not supported anymore, but it is mapped into ImprovedKneserNey for back-compatibility";
		case IMPROVED_KNESER_NEY:
			if (size>1){
				lm=new improvedkneserney(trainfile,size,prunefreq,(backoff?IMPROVEDKNESERNEY_B:IMPROVEDKNESERNEY_I));
			}	else {
				exit_error(IRSTLM_ERROR_DATA,"Improved Kneser-Ney requires size >1");
			}
			break;
			
		case IMPROVED_SHIFT_BETA:
			lm=new improvedshiftbeta(trainfile,size,prunefreq,(backoff?IMPROVEDSHIFTBETA_B:IMPROVEDSHIFTBETA_I));
			break;
	
		case SHIFT_ONE:
			lm=new shiftone(trainfile,size,prunefreq,(backoff?SIMPLE_B:SIMPLE_I));
			break;
			
		case LINEAR_STB:
			lm=new linearstb(trainfile,size,prunefreq,IMPROVEDSHIFTBETA_B);
			break;

		case LINEAR_WB:
			lm=new linearwb(trainfile,size,prunefreq,(backoff?IMPROVEDSHIFTBETA_B:IMPROVEDSHIFTBETA_I));
			break;
			
		case LINEAR_GT:
			cerr << "This LM is no more supported\n";
			break;

		case MIXTURE:
			//temporary check: so far unable to proper handle this flag in sub LMs
			//no ngramtable is created
			lm=new mixture(SavePerLevel,slminfo,size,prunefreq,imixpar,omixpar);
			break;
			
		default:
			cerr << "not implemented yet\n";
			return 1;
	};
	
	if (dub < lm->dict->size()){
		cerr << "dub (" << dub << ") is not set or too small. dub is re-set to the dictionary size (" << lm->dict->size() << ")" << endl;
		dub = lm->dict->size();
	}
	
	lm->dub(dub);
	
	lm->create_caches(max_caching_level);
	
	cerr << "eventually generate OOV code\n";
	lm->dict->genoovcode();
	
	if (oovrate) lm->dict->setoovrate(oovrate);
	
	lm->save_per_level(SavePerLevel);
	
	lm->train();
	
	//it never occurs that both prunetopsingletons and prunesingletons  are true
	if (prunetopsingletons==true) { //keep most specific
		lm->prunetopsingletons(true);
		lm->prunesingletons(false);
	} else {
		lm->prunetopsingletons(false);
		if (prunesingletons==true) {
			lm->prunesingletons(true);
		} else {
			lm->prunesingletons(false);
		}
	}
	if (prune_thr_str) lm->set_prune_ngram(prune_thr_str);
	
	if (adaptoov) lm->dict->incflag(1);
	
	if (adaptfile) lm->adapt(adaptfile,adaptlevel,adaptrate);
	
	if (adaptoov) lm->dict->incflag(0);
	
	if (scalefactorfile) lm->savescalefactor(scalefactorfile);
	
	if (backoff) lm->compute_backoff();
	
	if (size>lm->maxlevel()) {
		exit_error(IRSTLM_ERROR_DATA,"lm size is too large");
	}
	
	if (!size) size=lm->maxlevel();
	
	if (testfile) {
		cerr << "TLM: test ...";
		lm->test(testfile,size,backoff,checkpr,outpr);
		
		if (adaptfile)
			((mdiadaptlm *)lm)->get_zetacache()->stat();

		cerr << "\n";
	};
	
	if (compsize)
		cout << "LM size " << (int)lm->netsize() << "\n";
	
	if (interactive) {
		
		ngram ng(lm->dict);
		int nsize=0;
		
		cout.setf(ios::scientific);
		
		switch (interactive) {
				
			case NGRAM:
				cout << "> ";
				while(cin >> ng) {
					if (ng.wordp(size)) {
						cout << ng << " p=" << (double)log(lm->prob(ng,size)) << "\n";
						ng.size=0;
						cout << "> ";
					}
				}
				break;
				
			case SEQUENCE: {
				char c;
				double p=0;
				cout << "> ";
				
				while(cin >> ng) {
					nsize=ng.size<size?ng.size:size;
					p=log(lm->prob(ng,nsize));
					cout << ng << " p=" << p << "\n";
					
					while((c=cin.get())==' ') {
						cout << c;
					}
					cin.putback(c);
					//cout << "-" << c << "-";
					if (c=='\n') {
						ng.size=0;
						cout << "> ";
						p=0;
					}
				}
			}
				
				break;
				
			case TURN: {
				int n=0;
				double lp=0;
				double oov=0;
				
				while(cin >> ng) {
					
					if (ng.size>0) {
						nsize=ng.size<size?ng.size:size;
						lp-=log(lm->prob(ng,nsize));
						n++;
						if (*ng.wordp(1) == lm->dict->oovcode())
							oov++;
					} else {
						if (n>0) cout << n << " " << lp/(log(2.0) * n) << " " << oov/n << "\n";
						n=0;
						lp=0;
						oov=0;
					}
				}
				
				break;
			}
				
			case  TEXT: {
				int order;
				
				int n=0;
				double lp=0;
				double oov=0;
				
				while (!cin.eof()) {
					cin >> order;
					if (order>size)
						cerr << "Warning: order > lm size\n";
					
					order=order>size?size:order;
					
					while (cin >> ng) {
						if (ng.size>0) {
							nsize=ng.size<order?ng.size:order;
							lp-=log(lm->prob(ng,nsize));
							n++;
							if (*ng.wordp(1) == lm->dict->oovcode())
								oov++;
						} else {
							if (n>0) cout << n << " " << lp/(log(2.0)*n) << " " << oov/n << "\n";
							n=0;
							lp=0;
							oov=0;
							if (ng.isym>0) break;
						}
					}
				}
			}
				break;
				
			case ADAPT: {
				
				if (backoff) {
					exit_error(IRSTLM_ERROR_DATA,"This modality is not supported with backoff LMs");
				}
				
				char afile[50],tfile[50];
				while (!cin.eof()) {
					cin >> afile >> tfile;
					system("echo > .tlmlock");
					
					cerr << "interactive adaptation: "
					<< afile << " " << tfile << "\n";
					
					if (adaptoov) lm->dict->incflag(1);
					lm->adapt(afile,adaptlevel,adaptrate);
					if (adaptoov) lm->dict->incflag(0);
					if (scalefactorfile) lm->savescalefactor(scalefactorfile);
					if (ASRfile) lm->saveASR(ASRfile,backoff,dictfile);
					if (ARPAfile) lm->saveARPA(ARPAfile,backoff,dictfile);
					if (BINfile) lm->saveBIN(BINfile,backoff,dictfile,memmap);
					lm->test(tfile,size,checkpr);
					cout.flush();
					system("rm .tlmlock");
				}
			}
				break;
		}
		
		exit_error(IRSTLM_NO_ERROR);
	}
	
	if (ASRfile) {
		cerr << "TLM: save lm (ASR)...";
		lm->saveASR(ASRfile,backoff,dictfile);
		cerr << "\n";
	}
	
	if (ARPAfile) {
		cerr << "TLM: save lm (ARPA)...";
		lm->saveARPA(ARPAfile,backoff,dictfile);
		cerr << "\n";
	}
	
	if (BINfile) {
		cerr << "TLM: save lm (binary)...";
		lm->saveBIN(BINfile,backoff,dictfile,memmap);
		cerr << "\n";
	}
	
	if (statistics) {
		cerr << "TLM: lm stat ...";
		lm->lmstat(statistics);
		cerr << "\n";
	}
	
	//	lm->cache_stat();
	
	cerr << "TLM: deleting lm ...";
	delete lm;
	cerr << "\n";
	
	exit_error(IRSTLM_NO_ERROR);
}



