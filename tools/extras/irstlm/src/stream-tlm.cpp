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

using namespace std;

#include <cmath>
#include <math.h>
#include "mfstream.h"
#include <fstream>
#include <stdio.h>
#include <iostream>
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "mempool.h"
#include "ngramcache.h"
#include "ngramtable.h"
#include "interplm.h"
#include "normcache.h"
#include "mdiadapt.h"
#include "shiftlm.h"
#include "linearlm.h"
#include "mixture.h"
#include "cmd.h"
#include "lmtable.h"


#define YES   1
#define NO    0


#define NGRAM 1
#define SEQUENCE 2
#define ADAPT 3
#define TURN 4
#define TEXT 5


#define END_ENUM    {   (char*)0,  0 }

static Enum_T BooleanEnum [] = {
  {    "Yes",    YES },
  {    "No",     NO},
  {    "yes",    YES },
  {    "no",     NO},
  {    "y",     YES },
  {    "n",     NO},
  END_ENUM
};

static Enum_T LmTypeEnum [] = {
  {    "ImprovedKneserNey",  IMPROVED_KNESER_NEY },
  {    "ikn",                IMPROVED_KNESER_NEY },
  {    "KneserNey",          KNESER_NEY },
  {    "kn",                 KNESER_NEY },
  {    "ModifiedShiftBeta",  MOD_SHIFT_BETA },
  {    "msb",                MOD_SHIFT_BETA },
  {    "ImprovedShiftBeta",  IMPROVED_SHIFT_BETA },
  {    "isb",                IMPROVED_SHIFT_BETA },
  {    "InterpShiftBeta",    SHIFT_BETA },
  {    "ShiftBeta",          SHIFT_BETA },
  {    "sb",                 SHIFT_BETA },
  {    "InterpShiftOne",     SHIFT_ONE },
  {    "ShiftOne",           SHIFT_ONE },
  {    "s1",                 SHIFT_ONE },
  {    "LinearWittenBell",   LINEAR_WB },
  {    "wb",                 LINEAR_WB },
  {    "LinearGoodTuring",   LINEAR_GT },
  {    "Mixture",            MIXTURE },
  {    "mix",                MIXTURE },
  END_ENUM
};


#define RESET 1
#define SAVE  2
#define LOAD  3
#define INIT  4
#define STOP  5

#define BIN  11
#define ARPA 12
#define ASR  13
#define TXT  14
#define NGT  15


int init(mdiadaptlm** lm, int lmtype, char *trainfile, int size, int prunefreq, double beta, int backoff, int dub, double oovrate, int mcl);
int deinit(mdiadaptlm** lm);

int main(int argc, char **argv)
{

  char *dictfile=NULL;
  char *trainfile=NULL;

  char *BINfile=NULL;
  char *ARPAfile=NULL;
  char *ASRfile=NULL;

  int backoff=0; //back-off or interpolation
  int lmtype=0;
  int dub=0; //dictionary upper bound
  int size=0;   //lm size

  int statistics=0;

  int prunefreq=NO;
  int prunesingletons=YES;
  int prunetopsingletons=NO;

  double beta=-1;

  int compsize=NO;
  int checkpr=NO;
  double oovrate=0;
  int max_caching_level=0;

  char *outpr=NULL;

  int memmap = 0; //write binary format with/without memory map, default is 0

  DeclareParams(

    "Back-off",CMDENUMTYPE, &backoff, BooleanEnum,
    "bo",CMDENUMTYPE, &backoff, BooleanEnum,

    "Dictionary", CMDSTRINGTYPE, &dictfile,
    "d", CMDSTRINGTYPE, &dictfile,

    "DictionaryUpperBound", CMDINTTYPE, &dub,
    "dub", CMDINTTYPE, &dub,

    "NgramSize", CMDSUBRANGETYPE, &size, 1 , MAX_NGRAM,
    "n", CMDSUBRANGETYPE, &size, 1 , MAX_NGRAM,

    "Ngram", CMDSTRINGTYPE, &trainfile,
    "TrainOn", CMDSTRINGTYPE, &trainfile,
    "tr", CMDSTRINGTYPE, &trainfile,

    "oASR", CMDSTRINGTYPE, &ASRfile,
    "oasr", CMDSTRINGTYPE, &ASRfile,

    "o", CMDSTRINGTYPE, &ARPAfile,
    "oARPA", CMDSTRINGTYPE, &ARPAfile,
    "oarpa", CMDSTRINGTYPE, &ARPAfile,

    "oBIN", CMDSTRINGTYPE, &BINfile,
    "obin", CMDSTRINGTYPE, &BINfile,

    "LanguageModelType",CMDENUMTYPE, &lmtype, LmTypeEnum,
    "lm",CMDENUMTYPE, &lmtype, LmTypeEnum,

    "Statistics",CMDSUBRANGETYPE, &statistics, 1 , 3,
    "s",CMDSUBRANGETYPE, &statistics, 1 , 3,

    "PruneThresh",CMDSUBRANGETYPE, &prunefreq, 1 , 1000,
    "p",CMDSUBRANGETYPE, &prunefreq, 1 , 1000,

    "PruneSingletons",CMDENUMTYPE, &prunesingletons, BooleanEnum,
    "ps",CMDENUMTYPE, &prunesingletons, BooleanEnum,

    "PruneTopSingletons",CMDENUMTYPE, &prunetopsingletons, BooleanEnum,
    "pts",CMDENUMTYPE, &prunetopsingletons, BooleanEnum,

    "ComputeLMSize",CMDENUMTYPE, &compsize, BooleanEnum,
    "sz",CMDENUMTYPE, &compsize, BooleanEnum,

    "MaximumCachingLevel", CMDINTTYPE , &max_caching_level,
    "mcl", CMDINTTYPE, &max_caching_level,

    "MemoryMap", CMDENUMTYPE, &memmap, BooleanEnum,
    "memmap", CMDENUMTYPE, &memmap, BooleanEnum,
    "mm", CMDENUMTYPE, &memmap, BooleanEnum,

    "CheckProb",CMDENUMTYPE, &checkpr, BooleanEnum,
    "cp",CMDENUMTYPE, &checkpr, BooleanEnum,

    "OutProb",CMDSTRINGTYPE, &outpr,
    "op",CMDSTRINGTYPE, &outpr,

    "SetOovRate", CMDDOUBLETYPE, &oovrate,
    "or", CMDDOUBLETYPE, &oovrate,

    "Beta", CMDDOUBLETYPE, &beta,
    "beta", CMDDOUBLETYPE, &beta,

    (char *)NULL
  );

  GetParams(&argc, &argv, (char*) NULL);

  if (!lmtype) {
    cerr <<"Missing parameters\n";
    exit(1);
  }


  cerr <<"LM size: " << size << "\n";


  char header[BUFSIZ];
  char filename[BUFSIZ];
  int cmdcounter=0;
  mdiadaptlm *lm=NULL;


  int cmdtype=INIT;
  int filetype=0;
  int BoSfreq=0;

  init(&lm, lmtype, trainfile, size, prunefreq, beta, backoff, dub, oovrate, max_caching_level);

  ngram ng(lm->dict), ng2(lm->dict);

  cerr << "filling the initial n-grams with BoS\n";
  for (int i=1; i<lm->maxlevel(); i++) {
    ng.pushw(lm->dict->BoS());
    ng.freq=1;
  }

  mfstream inp("/dev/stdin",ios::in );
  int c=0;

  while (inp >> header) {

    if (strncmp(header,"@CMD@",5)==0) {
      cmdcounter++;
      inp >> header;

      cerr << "Read |@CMD@| |" << header << "|";

      cmdtype=INIT;
      filetype=BIN;
      if (strncmp(header,"RESET",5)==0)				  cmdtype=RESET;
      else if (strncmp(header,"INIT",4)==0) 	  cmdtype=INIT;
      else if (strncmp(header,"SAVEBIN",7)==0) 	{
        cmdtype=SAVE;
        filetype=BIN;
      } else if (strncmp(header,"SAVEARPA",8)==0) {
        cmdtype=SAVE;
        filetype=ARPA;
      } else if (strncmp(header,"SAVEASR",7)==0) 	{
        cmdtype=SAVE;
        filetype=ASR;
      } else if (strncmp(header,"SAVENGT",7)==0)  {
        cmdtype=SAVE;
        filetype=NGT;
      } else if (strncmp(header,"LOADNGT",7)==0)  {
        cmdtype=LOAD;
        filetype=NGT;
      } else if (strncmp(header,"LOADTXT",7)==0)  {
        cmdtype=LOAD;
        filetype=TXT;
      } else if (strncmp(header,"STOP",4)==0) 		cmdtype=STOP;
      else {
        cerr << "CMD " << header << " is unknown\n";
        exit(1);
      }

      char** lastwords;
      char *isym;
      switch (cmdtype) {

      case STOP:
        cerr << "\n";
        exit(1);
        break;

      case SAVE:

        inp >> filename; //storing the output filename
        cerr << " |" << filename << "|\n";

        //save actual ngramtable
        char tmpngtfile[BUFSIZ];
        sprintf(tmpngtfile,"%s.ngt%d",filename,cmdcounter);
        cerr << "saving temporary ngramtable (binary)..." << tmpngtfile << "\n";
        ((ngramtable*) lm)->ngtype("ngram");
        ((ngramtable*) lm)->savetxt(tmpngtfile,size);

        //get the actual frequency of BoS symbol, because the constructor of LM will reset to 1;
        BoSfreq=lm->dict->freq(lm->dict->encode(lm->dict->BoS()));

        lm->train();

        lm->prunesingletons(prunesingletons==YES);
        lm->prunetopsingletons(prunetopsingletons==YES);

        if (prunetopsingletons==YES) //keep most specific
          lm->prunesingletons(NO);


        switch (filetype) {

        case BIN:
          cerr << "saving lm (binary) ... " << filename << "\n";
          lm->saveBIN(filename,backoff,dictfile,memmap);
          cerr << "\n";
          break;

        case ARPA:
          cerr << "save lm (ARPA)... " << filename << "\n";
          lm->saveARPA(filename,backoff,dictfile);
          cerr << "\n";
          break;

        case ASR:
          cerr << "save lm (ASR)... " << filename << "\n";
          lm->saveASR(filename,backoff,dictfile);
          cerr << "\n";
          break;

        case NGT:
          cerr << "save the ngramtable on ... " << filename << "\n";
          {
            ifstream ifs(tmpngtfile, ios::binary);
            std::ofstream ofs(filename, std::ios::binary);
            ofs << ifs.rdbuf();
          }
          cerr << "\n";
          break;

        default:
          cerr << "Saving type is unknown\n";
          exit(1);
        };

        //store last words up to the LM order (filling with BoS if needed)
        ng.size=(ng.size>lm->maxlevel())?lm->maxlevel():ng.size;
        lastwords = new char*[lm->maxlevel()];

        for (int i=1; i<lm->maxlevel(); i++) {
          lastwords[i] = new char[BUFSIZ];
          if (i<=ng.size)
            strcpy(lastwords[i],lm->dict->decode(*ng.wordp(i)));
          else
            strcpy(lastwords[i],lm->dict->BoS());
        }

        deinit(&lm);

        init(&lm, lmtype, tmpngtfile, size, prunefreq, beta, backoff, dub, oovrate, max_caching_level);
        if (remove(tmpngtfile) != 0)
          cerr << "Error deleting file " << tmpngtfile << endl;
        else
          cerr << "File " << tmpngtfile << " successfully deleted" << endl;

        //re-set the dictionaries of the working ngrams and re-encode the actual ngram
        ng.dict=ng2.dict=lm->dict;
        ng.size=lm->maxlevel();

        //restore the last words re-encoded wrt to the new dictionary
        for (int i=1; i<lm->maxlevel(); i++) {
          *ng.wordp(i)=lm->dict->encode(lastwords[i]);
          delete []lastwords[i];
        }
        delete []lastwords;


        //re-set the actual frequency of BoS symbol, because the constructor of LM deleted it;
        lm->dict->freq(lm->dict->encode(lm->dict->BoS()), BoSfreq);
        break;


      case RESET: //restart from scratch
        deinit(&lm);

        init(&lm, lmtype, NULL, size, prunefreq, beta, backoff, dub, oovrate, max_caching_level);

        ng.dict=ng2.dict=lm->dict;
        cerr << "filling the initial n-grams with BoS\n";
        for (int i=1; i<lm->maxlevel(); i++) {
          ng.pushw(lm->dict->BoS());
          ng.freq=1;
        }
        break;


      case INIT:
        cerr << "CMD " << header << " not yet implemented\n";
        exit(1);
        break;

      case LOAD:
        inp >> filename; //storing the input filename
        cerr << " |" << filename << "|\n";


        isym=new char[BUFSIZ];
        strcpy(isym,lm->dict->EoS());
        ngramtable* ngt;

        switch (filetype) {

        case NGT:
          cerr << "loading an ngramtable..." << filename << "\n";
          ngt = new ngramtable(filename,size,isym,NULL,NULL);
          ((ngramtable*) lm)->augment(ngt);
          cerr << "\n";
          break;

        case TXT:
          cerr << "loading from text..." << filename << "\n";
          ngt= new ngramtable(filename,size,isym,NULL,NULL);
          ((ngramtable*) lm)->augment(ngt);
          cerr << "\n";
          break;

        default:
          cerr << "This file type is unknown\n";
          exit(1);
        };

        break;

      default:
        cerr << "CMD " << header << " is unknown\n";
        exit(1);
      };
    } else {
      ng.pushw(header);

      // CHECK: serve questa trans()
      ng2.trans(ng); //reencode with new dictionary

      lm->check_dictsize_bound();

      //CHECK: e' corretto ng.size?  non dovrebbe essere ng2.size?
      if (ng.size) lm->dict->incfreq(*ng2.wordp(1),1);
      //CHECK: what about filtering dictionary???
      /*
       if (filterdict){
      	 int code=filterdict->encode(dict->decode(*ng2.wordp(maxlev)));
      	 if (code!=filterdict->oovcode())	put(ng2);
       }
       else put(ng2);
       */

      lm->put(ng2);

      if (!(++c % 1000000)) cerr << ".";
    }
  }

  if (statistics) {
    cerr << "TLM: lm stat ...";
    lm->lmstat(statistics);
    cerr << "\n";
  }

  cerr << "TLM: deleting lm ...";
  //delete lm;
  cerr << "\n";

  exit(0);
}

int init(mdiadaptlm** lm, int lmtype, char *trainfile, int size, int prunefreq, double beta, int backoff, int dub, double oovrate, int mcl)
{

  cerr << "initializing lm... \n";
  if (trainfile) cerr << "creating lm from " << trainfile << "\n";
  else cerr << "creating an empty lm\n";
  switch (lmtype) {

  case SHIFT_BETA:
    if (beta==-1 || (beta<1.0 && beta>0))
      *lm=new shiftbeta(trainfile,size,prunefreq,beta,(backoff?SHIFTBETA_B:SHIFTBETA_I));
    else {
      cerr << "ShiftBeta: beta must be >0 and <1\n";
      exit(1);
    }
    break;

  case KNESER_NEY:
    if (size>1){
      if (beta==-1 || (beta<1.0 && beta>0)){
//	lm=new kneserney(trainfile,size,prunefreq,beta,(backoff?KNESERNEY_B:KNESERNEY_I));
      } else {
        exit_error(IRSTLM_ERROR_DATA,"ShiftBeta: beta must be >0 and <1");
      }
    } else {
      exit_error(IRSTLM_ERROR_DATA,"Kneser-Ney requires size >1");
    }
  break;

  case MOD_SHIFT_BETA:
    cerr << "ModifiedShiftBeta (msb) is the old name for ImprovedKneserNey (ikn); this name is not supported anymore, but it is mapped into ImprovedKneserNey for back-compatibility";
  case IMPROVED_KNESER_NEY:
    if (size>1){
      lm=new improvedkneserney(trainfile,size,prunefreq,(backoff?IMPROVEDKNESERNEY_B:IMPROVEDKNESERNEY_I));
    } else {
      exit_error(IRSTLM_ERROR_DATA,"Improved Kneser-Ney requires size >1");
    }
  break;

  case IMPROVED_SHIFT_BETA:
    lm=new improvedshiftbeta(trainfile,size,prunefreq,(backoff?IMPROVEDSHIFTBETA_B:IMPROVEDSHIFTBETA_I));
  break;

  case SHIFT_ONE:
    *lm=new shiftone(trainfile,size,prunefreq,(backoff?SIMPLE_B:SIMPLE_I));
    break;

  case LINEAR_WB:
    *lm=new linearwb(trainfile,size,prunefreq,(backoff?MSHIFTBETA_B:MSHIFTBETA_I));
    break;

  case LINEAR_GT:
    cerr << "This LM is no more supported\n";
    break;

  case MIXTURE:
    cerr << "not implemented yet\n";
    break;

  default:
    cerr << "not implemented yet\n";
    exit(1);
  };

  if (dub)      (*lm)->dub(dub);
  (*lm)->create_caches(mcl);

  cerr << "eventually generate OOV code\n";
  (*lm)->dict->genoovcode();

  if (oovrate) (*lm)->dict->setoovrate(oovrate);

  (*lm)->dict->incflag(1);

  if (!trainfile) {
    cerr << "adding the initial dummy n-grams to make table consistent\n";

    ngram dummyng((*lm)->dict);
    cerr << "preparing initial dummy n-grams\n";
    for (int i=1; i<(*lm)->maxlevel(); i++) {
      dummyng.pushw((*lm)->dict->BoS());
      dummyng.freq=1;
    }
    cerr << "inside init:  dict: " << (*lm)->dict << " dictsize: " << (*lm)->dict->size() << "\n";
    cerr << "dummyng: |" << dummyng << "\n";
    (*lm)->put(dummyng);
    cerr << "inside init:  dict: " << (*lm)->dict << " dictsize: " << (*lm)->dict->size() << "\n";

  }

  cerr << "lm initialized \n";
  return 1;
}

int deinit(mdiadaptlm** lm)
{
  delete *lm;
  return 1;
}
