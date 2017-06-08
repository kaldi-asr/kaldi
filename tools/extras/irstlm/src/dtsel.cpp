// $Id: ngt.cpp 245 2009-04-02 14:05:40Z fabio_brugnara $

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

// dtsel
// by M. Federico
// Copyright Marcello Federico, Fondazione Bruno Kessler, 2012


#include <cmath>
#include "util.h"
#include <sstream>
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "ngramtable.h"
#include "cmd.h"

using namespace std;

#define YES   1
#define NO    0

void print_help(int TypeFlag=0){
		std::cerr << std::endl << "dtsel - performs data selection" << std::endl;
		std::cerr << std::endl << "USAGE:"  << std::endl
			  << "       dtsel -s=<outfile> [options]" << std::endl;
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

double prob(ngramtable* ngt,ngram ng,int size,int cv){
	MY_ASSERT(size<=ngt->maxlevel() && size<=ng.size);	
	if (size>1){				
		ngram history=ng;
		if (ngt->get(history,size,size-1) && history.freq>cv){
			double fstar=0.0;
			double lambda;
			if (ngt->get(ng,size,size)){
				cv=(cv>ng.freq)?ng.freq:cv;
				if (ng.freq>cv){
					fstar=(double)(ng.freq-cv)/(double)(history.freq -cv + history.succ);					
					lambda=(double)history.succ/(double)(history.freq -cv + history.succ);
				}else //ng.freq==cv
					lambda=(double)(history.succ-1)/(double)(history.freq -cv + history.succ-1);
			}
			else
				lambda=(double)history.succ/(double)(history.freq -cv + history.succ);			
			
			return fstar + lambda * prob(ngt,ng,size-1,cv);
		}
		else return prob(ngt,ng,size-1,cv);
		
	}else{ //unigram branch
		if (ngt->get(ng,1,1) && ng.freq>cv)
			return (double)(ng.freq-cv)/(ngt->totfreq()-1);
		else{
			//cerr << "backoff to oov unigram " << ng.freq << " " << cv << "\n";
			*ng.wordp(1)=ngt->dict->oovcode();
			if (ngt->get(ng,1,1) && ng.freq>0)
				return (double)ng.freq/ngt->totfreq();		
			else //use an automatic estimate of Pr(oov)
				return (double)ngt->dict->size()/(ngt->totfreq()+ngt->dict->size());				
		}

	}
	
}


double computePP(ngramtable* train,ngramtable* test,double oovpenalty,double& oovrate,int cv=0){
	
	
	ngram ng2(test->dict);ngram ng1(train->dict);
	int N=0; double H=0;  oovrate=0;
	
	test->scan(ng2,INIT,test->maxlevel());
 	while(test->scan(ng2,CONT,test->maxlevel())) {
		
		ng1.trans(ng2);
		H-=log(prob(train,ng1,ng1.size,cv));
		if (*ng1.wordp(1)==train->dict->oovcode()){
			H-=oovpenalty;
			oovrate++;
		}
		N++;
	}
	oovrate/=N;
	return exp(H/N);
}


int main(int argc, char **argv)
{
	char *indom=NULL;   //indomain data: one sentence per line
	char *outdom=NULL;   //domain data: one sentence per line
	char *scorefile=NULL;  //score file 
	char *evalset=NULL;    //evalset to measure performance

	int minfreq=2;          //frequency threshold for dictionary pruning (optional)
	int ngsz=0;             // n-gram size 
	int dub=IRSTLM_DUB_DEFAULT;      //upper bound of true vocabulary
	int model=2;           //data selection model: 1 only in-domain cross-entropy, 
	                       //2 cross-entropy difference. 	
	int cv=1;              //cross-validation parameter: 1 only in-domain cross-entropy, 

	int blocksize=100000; //block-size in words
	int verbose=0;
	int useindex=0; //provided score file includes and index
	double convergence_treshold=0;
	
	bool help=false;
	
	DeclareParams((char*)
				  "min-word-freq", CMDINTTYPE|CMDMSG, &minfreq, "frequency threshold for dictionary pruning, default: 2",
				  "f", CMDINTTYPE|CMDMSG, &minfreq, "frequency threshold for dictionary pruning, default: 2",
				  
		      "ngram-order", CMDSUBRANGETYPE|CMDMSG, &ngsz, 1 , MAX_NGRAM, "n-gram default size, default: 0",
				  "n", CMDSUBRANGETYPE|CMDMSG, &ngsz, 1 , MAX_NGRAM, "n-gram default size, default: 0",
				  
				  "in-domain-file", CMDSTRINGTYPE|CMDMSG, &indom, "indomain data file: one sentence per line",
				  "i", CMDSTRINGTYPE|CMDMSG, &indom, "indomain data file: one sentence per line",
				  
				  "out-domain-file", CMDSTRINGTYPE|CMDMSG, &outdom, "domain data file: one sentence per line",
				  "o", CMDSTRINGTYPE|CMDMSG, &outdom, "domain data file: one sentence per line",
				  
				  "score-file", CMDSTRINGTYPE|CMDMSG, &scorefile, "score output file",
				  "s", CMDSTRINGTYPE|CMDMSG, &scorefile, "score output file",

				  "dictionary-upper-bound", CMDINTTYPE|CMDMSG, &dub, "upper bound of true vocabulary, default: 10000000",
				  "dub", CMDINTTYPE|CMDMSG, &dub, "upper bound of true vocabulary, default: 10000000",
				  
				  "model", CMDSUBRANGETYPE|CMDMSG, &model, 1 , 2, "data selection model: 1 only in-domain cross-entropy, 2 cross-entropy difference; default: 2",
				  "m", CMDSUBRANGETYPE|CMDMSG, &model, 1 , 2, "data selection model: 1 only in-domain cross-entropy, 2 cross-entropy difference; default: 2",
				  
				  "cross-validation", CMDSUBRANGETYPE|CMDMSG, &cv, 1 , 3, "cross-validation parameter: 1 only in-domain cross-entropy; default: 1",
  				  "cv", CMDSUBRANGETYPE|CMDMSG, &cv, 1 , 3, "cross-validation parameter: 1 only in-domain cross-entropy; default: 1",
				  
				  "test", CMDSTRINGTYPE|CMDMSG, &evalset, "evaluation set file to measure performance",
				  "t", CMDSTRINGTYPE|CMDMSG, &evalset, "evaluation set file to measure performance",
				  
				  "block-size", CMDINTTYPE|CMDMSG, &blocksize, "block-size in words, default: 100000",
				  "bs", CMDINTTYPE|CMDMSG, &blocksize, "block-size in words, default: 100000",
				  
				  "convergence-threshold", CMDDOUBLETYPE|CMDMSG, &convergence_treshold, "convergence threshold, default: 0",
				  "c", CMDDOUBLETYPE|CMDMSG, &convergence_treshold, "convergence threshold, default: 0",
				  
				  "index", CMDSUBRANGETYPE|CMDMSG, &useindex,0,1, "provided score file includes and index, default: 0",
				  "x", CMDSUBRANGETYPE|CMDMSG, &useindex,0,1, "provided score file includes and index, default: 0",

				  "verbose", CMDSUBRANGETYPE|CMDMSG, &verbose,0,2, "verbose level, default: 0",
				  "v", CMDSUBRANGETYPE|CMDMSG, &verbose,0,2, "verbose level, default: 0",
								"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",

				  (char *)NULL
				  );
	
	
	
	GetParams(&argc, &argv, (char*) NULL);

	if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	if (scorefile==NULL) {
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
	if (!evalset && (!indom || !outdom)){		
    exit_error(IRSTLM_ERROR_DATA, "Must specify in-domain and out-domain data files");
	};
	
	//score file is always required: either as output or as input
	if (!scorefile){
    exit_error(IRSTLM_ERROR_DATA, "Must specify score file");
	};
	
	if (!evalset && !model){
    exit_error(IRSTLM_ERROR_DATA, "Must specify data selection model");
	}
	
	if (evalset && (convergence_treshold<0 || convergence_treshold > 0.1)){
    exit_error(IRSTLM_ERROR_DATA, "Convergence threshold must be between 0 and 0.1");
	}
	
	TABLETYPE table_type=COUNT;
		
	
	if (!evalset){
		
		//computed dictionary on indomain data
		dictionary *dict = new dictionary(indom,1000000,0);
		dictionary *pd=new dictionary(dict,true,minfreq);
		delete dict;dict=pd;		
		
		//build in-domain table restricted to the given dictionary
		ngramtable *indngt=new ngramtable(indom,ngsz,NULL,dict,NULL,0,0,NULL,0,table_type);
		double indoovpenalty=-log(dub-indngt->dict->size());
		ngram indng(indngt->dict);
		int indoovcode=indngt->dict->oovcode();
		
		//build out-domain table restricted to the in-domain dictionary
		char command[1000]="";
			
		if (useindex)
			sprintf(command,"cut -d \" \" -f 2- %s",outdom);
		else
			sprintf(command,"%s",outdom);
		
		
		ngramtable *outdngt=new ngramtable(command,ngsz,NULL,dict,NULL,0,0,NULL,0,table_type);
		double outdoovpenalty=-log(dub-outdngt->dict->size());	
		ngram outdng(outdngt->dict);
		int outdoovcode=outdngt->dict->oovcode();
		
		cerr << "dict size idom: " << indngt->dict->size() << " odom: " << outdngt->dict->size() << "\n";
		cerr << "oov penalty idom: " << indoovpenalty << " odom: " << outdoovpenalty << "\n";
		
		//go through the odomain sentences 
		int bos=dict->encode(dict->BoS());
		mfstream inp(outdom,ios::in); ngram ng(dict);
		mfstream txt(outdom,ios::in);
		mfstream output(scorefile,ios::out);
					

		int linenumber=1; string line;	
		int length=0;
		float deltaH=0;
		float deltaHoov=0;
		int words=0;string index;

		while (getline(inp,line)){

			istringstream lninp(line);
	
			linenumber++;
			
			if (useindex) lninp >> index;
			
			// reset ngram at begin of sentence
			ng.size=1; deltaH=0;deltaHoov=0; length=0;
			
			while(lninp>>ng){
			
				if (*ng.wordp(1)==bos) continue;
											
				length++; words++;
				
				if ((words % 1000000)==0) cerr << ".";				
				
				if (ng.size>ngsz) ng.size=ngsz;
				indng.trans(ng);outdng.trans(ng);
				
				if (model==1){//compute cross-entropy
					deltaH-=log(prob(indngt,indng,indng.size,0));	
					deltaHoov-=(*indng.wordp(1)==indoovcode?indoovpenalty:0);
				}
				
				if (model==2){ //compute cross-entropy difference
					deltaH+=log(prob(outdngt,outdng,outdng.size,cv))-log(prob(indngt,indng,indng.size,0));	
					deltaHoov+=(*outdng.wordp(1)==outdoovcode?outdoovpenalty:0)-(*indng.wordp(1)==indoovcode?indoovpenalty:0);
				}											
			}

			output << (deltaH + deltaHoov)/length  << " " << line << "\n";													
		}
	}
	else{
		
		//build in-domain LM from evaluation set 
		ngramtable *tstngt=new ngramtable(evalset,ngsz,NULL,NULL,NULL,0,0,NULL,0,table_type);
		
		//build empty out-domain LM
		ngramtable *outdngt=new ngramtable(NULL,ngsz,NULL,NULL,NULL,0,0,NULL,0,table_type);		
		
		//if indomain data is passed then limit comparison to its dictionary
		dictionary *dict = NULL;
		if (indom){
			cerr << "dtsel: limit evaluation dict to indomain words with freq >=" <<  minfreq << "\n";
			//computed dictionary on indomain data
			dict = new dictionary(indom,1000000,0);
			dictionary *pd=new dictionary(dict,true,minfreq);
			delete dict;dict=pd;
			outdngt->dict=dict;
		}
								
		dictionary* outddict=outdngt->dict;
		
		//get codes of <s>, </s> and UNK
		outddict->incflag(1);
		int bos=outddict->encode(outddict->BoS());
		int oov=outddict->encode(outddict->OOV());
		outddict->incflag(0);
		outddict->oovcode(oov);
		
		
		double oldPP=dub; double newPP=0; double oovrate=0;  
			
		long totwords=0; long totlines=0; long nextstep=blocksize; 

		double score; string index;
		
		mfstream outd(scorefile,ios::in); string line;
		
		//initialize n-gram	
		ngram ng(outdngt->dict); for (int i=1;i<ngsz;i++) ng.pushc(bos); ng.freq=1;

		//check if to use open or closed voabulary
		
		if (!dict) outddict->incflag(1);
		
		while (getline(outd,line)){
			
			istringstream lninp(line);
			
			//skip score and eventually the index
			lninp >> score; if (useindex) lninp >> index;

			while (lninp >> ng){
				
				if (*ng.wordp(1) == bos) continue; 
				
				if (ng.size>ngsz) ng.size=ngsz;
				
				outdngt->put(ng);
				
				totwords++;
			}

			totlines++;
			
			if (totwords>=nextstep){ //if block is complete

				if (!dict) outddict->incflag(0);
				
				newPP=computePP(outdngt,tstngt,-log(dub-outddict->size()),oovrate);
				
				if (!dict) outddict->incflag(1);
				
				cout << totwords << " " << newPP;				
				if (verbose) cout << " " << totlines << " " << oovrate;			
				cout << "\n";
				
				if (convergence_treshold && (oldPP-newPP)/oldPP < convergence_treshold) return 1; 
									
				oldPP=newPP;
				
				nextstep+=blocksize;
			}
		}
		
		if (!dict) outddict->incflag(0);
		newPP=computePP(outdngt,tstngt,-log(dub-outddict->size()),oovrate);
		cout << totwords << " " << newPP;
		if (verbose) cout << " " << totlines << " " << oovrate;			
		
	}
	
}

	
	
