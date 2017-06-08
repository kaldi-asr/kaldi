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

// ngt
// by M. Federico
// Copyright Marcello Federico, ITC-irst, 1998


#include <iostream>
#include <sstream>
#include <cmath>
#include "util.h"
#include "cmd.h"
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "ngramtable.h"

using namespace std;

void print_help(int TypeFlag=0){
	std::cerr << std::endl << "ngt - collects n-grams" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl;
	std::cerr << "       ngt -i=<inputfile> [options]" << std::endl;
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
  char *inp=NULL;
  char *out=NULL;
  char *dic=NULL;       // dictionary filename
  char *subdic=NULL;    // subdictionary filename
  char *filterdict=NULL;    // subdictionary filename
  char *filtertable=NULL;   // ngramtable filename
  char *iknfile=NULL;   //  filename to save IKN statistics
  double filter_hit_rate=1.0;  // minimum hit rate of filter
  char *aug=NULL;       // augmentation data
  char *hmask=NULL;        // historymask
  bool inputgoogleformat=false;    //reads ngrams in Google format
  bool outputgoogleformat=false;    //print ngrams in Google format
  bool outputredisformat=false;    //print ngrams in Redis format
  int ngsz=0;           // n-gram default size
  int dstco=0;          // compute distance co-occurrences
  bool bin=false;
  bool ss=false;            //generate single table
  bool LMflag=false;        //work with LM table
  bool  saveeach=false;   //save all n-gram orders
  int inplen=0;         //input length for mask generation
  bool tlm=false;       //test lm table
  char* ftlm=NULL;     //file to test LM table
   
  bool memuse=false;
  bool help=false;
    

  DeclareParams((char*)
                "Dictionary", CMDSTRINGTYPE|CMDMSG, &dic, "dictionary filename",
                "d", CMDSTRINGTYPE|CMDMSG, &dic, "dictionary filename",

                "NgramSize", CMDSUBRANGETYPE|CMDMSG, &ngsz, 1, MAX_NGRAM, "n-gram default size; default is 0",
                "n", CMDSUBRANGETYPE|CMDMSG, &ngsz, 1, MAX_NGRAM, "n-gram default size; default is 0",
                "InputFile", CMDSTRINGTYPE|CMDMSG, &inp, "input file",
                "i", CMDSTRINGTYPE|CMDMSG, &inp, "input file",
                "OutputFile", CMDSTRINGTYPE|CMDMSG, &out, "output file",
                "o", CMDSTRINGTYPE|CMDMSG, &out, "output file",
                "InputGoogleFormat", CMDBOOLTYPE|CMDMSG, &inputgoogleformat, "the input file contains data in the n-gram Google format; default is false",
                "gooinp", CMDBOOLTYPE|CMDMSG, &inputgoogleformat, "the input file contains data in the n-gram Google format; default is false",
                "OutputGoogleFormat", CMDBOOLTYPE|CMDMSG, &outputgoogleformat,  "the output file contains data in the n-gram Google format; default is false",
                "gooout", CMDBOOLTYPE|CMDMSG, &outputgoogleformat,  "the output file contains data in the n-gram Google format; default is false",
                "OutputRedisFormat", CMDBOOLTYPE|CMDMSG, &outputredisformat,  "as Goolge format plus corresponding CRC.16 hash values; default is false",
                "redisout", CMDBOOLTYPE|CMDMSG, &outputredisformat,  "as Goolge format plus corresponding CRC.16 hash values; default is false",
                "SaveEach", CMDBOOLTYPE|CMDMSG, &saveeach,  "save all ngram orders; default is false",
                "saveeach", CMDBOOLTYPE|CMDMSG, &saveeach, "save all ngram orders; default is false",
                "SaveBinaryTable", CMDBOOLTYPE|CMDMSG, &bin, "saves into binary format; default is false",
                "b", CMDBOOLTYPE|CMDMSG, &bin, "saves into binary format; default is false",
                "LmTable", CMDBOOLTYPE|CMDMSG, &LMflag, "works with LM table; default is false",
                "lm", CMDBOOLTYPE|CMDMSG, &LMflag,  "works with LM table; default is false",
                "DistCo", CMDINTTYPE|CMDMSG, &dstco, "computes distance co-occurrences at the specified distance; default is 0",
                "dc", CMDINTTYPE|CMDMSG, &dstco, "computes distance co-occurrences at the specified distance; default is 0",
                "AugmentFile", CMDSTRINGTYPE|CMDMSG, &aug, "augmentation data",
                "aug", CMDSTRINGTYPE|CMDMSG, &aug, "augmentation data",
                "SaveSingle", CMDBOOLTYPE|CMDMSG, &ss, "generates single table; default is false",
                "ss", CMDBOOLTYPE|CMDMSG, &ss, "generates single table; default is false",
                "SubDict", CMDSTRINGTYPE|CMDMSG, &subdic, "subdictionary",
                "sd", CMDSTRINGTYPE|CMDMSG, &subdic, "subdictionary",
                "FilterDict", CMDSTRINGTYPE|CMDMSG, &filterdict, "filter dictionary",
                "fd", CMDSTRINGTYPE|CMDMSG, &filterdict, "filter dictionary",
                "ConvDict", CMDSTRINGTYPE|CMDMSG, &subdic, "subdictionary",
                "cd", CMDSTRINGTYPE|CMDMSG, &subdic, "subdictionary",
                "FilterTable", CMDSTRINGTYPE|CMDMSG, &filtertable, "ngramtable filename",
                "ftr", CMDDOUBLETYPE|CMDMSG, &filter_hit_rate, "ngramtable filename",
                "FilterTableRate", CMDDOUBLETYPE|CMDMSG, &filter_hit_rate, "minimum hit rate of filter; default is 1.0",
                "ft", CMDSTRINGTYPE|CMDMSG, &filtertable, "minimum hit rate of filter; default is 1.0",
                "HistoMask",CMDSTRINGTYPE|CMDMSG, &hmask, "history mask",
                "hm",CMDSTRINGTYPE|CMDMSG, &hmask, "history mask",
                "InpLen",CMDINTTYPE|CMDMSG, &inplen, "input length for mask generation; default is 0",
                "il",CMDINTTYPE|CMDMSG, &inplen, "input length for mask generation; default is 0",
                "tlm", CMDBOOLTYPE|CMDMSG, &tlm, "test LM table; default is false",
                "ftlm", CMDSTRINGTYPE|CMDMSG, &ftlm, "file to test LM table",
                "memuse", CMDBOOLTYPE|CMDMSG, &memuse, "default is false",
                "iknstat", CMDSTRINGTYPE|CMDMSG, &iknfile, "filename to save IKN statistics",

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
	
  if (inp==NULL) {
		usage();
		exit_error(IRSTLM_ERROR_DATA,"Warning: no input file specified");
  };

  if (out==NULL) {
    cerr << "Warning: no output file specified!\n";
  }

  TABLETYPE table_type=COUNT;

  if (LMflag) {
    cerr << "Working with LM table\n";
    table_type=LEAFPROB;
  }


  // check word order of subdictionary

  if (filtertable) {

    {
      ngramtable ngt(filtertable,1,NULL,NULL,NULL,0,0,NULL,0,table_type);
      mfstream inpstream(inp,ios::in); //google input table
      mfstream outstream(out,ios::out); //google output table

      cerr << "Filtering table " << inp << " assumed to be in Google Format with size " << ngsz << "\n";
      cerr << "with table " << filtertable <<  " of size " << ngt.maxlevel() << "\n";
      cerr << "with hit rate " << filter_hit_rate << "\n";

      //order of filter table must be smaller than that of input n-grams
      MY_ASSERT(ngt.maxlevel() <= ngsz);

      //read input googletable of ngrams of size ngsz
      //output entries made of at least X% n-grams contained in filtertable
      //<unk> words are not accepted

      ngram ng(ngt.dict), ng2(ng.dict);
      double hits=0;
      double maxhits=(double)(ngsz-ngt.maxlevel()+1);

      long c=0;
      while(inpstream >> ng) {

        if (ng.size>= ngt.maxlevel()) {
          //need to make a copy
          ng2=ng;
          ng2.size=ngt.maxlevel();
          //cerr << "check if " << ng2 << " is contained: ";
          hits+=(ngt.get(ng2)?1:0);
        }

        if (ng.size==ngsz) {
          if (!(++c % 1000000)) cerr << ".";
          //cerr << ng << " -> " << is_included << "\n";
          //you reached the last word before freq
          inpstream >> ng.freq;
          //consistency check of n-gram
          if (((hits/maxhits)>=filter_hit_rate) &&
              (!ng.containsWord(ngt.dict->OOV(),ng.size))
             )
            outstream << ng << "\n";
          hits=0;
          ng.size=0;
        }
      }

      outstream.flush();
      inpstream.flush();
    }

    exit_error(IRSTLM_NO_ERROR);
  }



  //ngramtable* ngt=new ngramtable(inp,ngsz,NULL,dic,dstco,hmask,inplen,table_type);
  ngramtable* ngt=new ngramtable(inp,ngsz,NULL,NULL,filterdict,inputgoogleformat,dstco,hmask,inplen,table_type);

  if (aug) {
    ngt->dict->incflag(1);
    //    ngramtable ngt2(aug,ngsz,isym,NULL,0,NULL,0,table_type);
    ngramtable ngt2(aug,ngsz,NULL,NULL,NULL,0,0,NULL,0,table_type);
    ngt->augment(&ngt2);
    ngt->dict->incflag(0);
  }


  if (subdic) {

    ngramtable *ngt2=new ngramtable(NULL,ngsz,NULL,NULL,NULL,0,0,NULL,0,table_type);

    // enforce the subdict to follow the same word order of the main dictionary
    dictionary tmpdict(subdic);
    ngt2->dict->incflag(1);
    for (int j=0; j<ngt->dict->size(); j++) {
      if (tmpdict.encode(ngt->dict->decode(j)) != tmpdict.oovcode()) {
        ngt2->dict->encode(ngt->dict->decode(j));
      }
    }
    ngt2->dict->incflag(0);

    ngt2->dict->cleanfreq();

    //possibly include standard symbols
    if (ngt->dict->encode(ngt->dict->EoS())!=ngt->dict->oovcode()) {
      ngt2->dict->incflag(1);
      ngt2->dict->encode(ngt2->dict->EoS());
      ngt2->dict->incflag(0);
    }
    if (ngt->dict->encode(ngt->dict->BoS())!=ngt->dict->oovcode()) {
      ngt2->dict->incflag(1);
      ngt2->dict->encode(ngt2->dict->BoS());
      ngt2->dict->incflag(0);
    }


    ngram ng(ngt->dict);
    ngram ng2(ngt2->dict);

    ngt->scan(ng,INIT,ngsz);
    long c=0;
    while (ngt->scan(ng,CONT,ngsz)) {
      ng2.trans(ng);
      ngt2->put(ng2);
      if (!(++c % 1000000)) cerr << ".";
    }

    //makes ngt2 aware of oov code
    int oov=ngt2->dict->getcode(ngt2->dict->OOV());
    if(oov>=0) ngt2->dict->oovcode(oov);

    for (int j=0; j<ngt->dict->size(); j++) {
      ngt2->dict->incfreq(ngt2->dict->encode(ngt->dict->decode(j)),
                          ngt->dict->freq(j));
    }

    cerr <<" oov: " << ngt2->dict->freq(ngt2->dict->oovcode()) << "\n";

    delete ngt;
    ngt=ngt2;

  }

  if (ngsz < ngt->maxlevel() && hmask) {
    cerr << "start projection of ngramtable " << inp
         << " according to hmask\n";

    int selmask[MAX_NGRAM];
    memset(selmask, 0, sizeof(int)*MAX_NGRAM);

    //parse hmask
    selmask[0]=1;
		int i=1;
    for (size_t c=0; c<strlen(hmask); c++) {
      cerr << hmask[c] << "\n";
      if (hmask[c] == '1'){
        selmask[i]=c+2;
	i++;
      }
    }

    if (i!= ngsz) {
			std::stringstream ss_msg;
			ss_msg << "wrong mask: 1 bits=" << i << " maxlev=" << ngsz;
			exit_error(IRSTLM_ERROR_DATA, ss_msg.str());
    }

    if (selmask[ngsz-1] >  ngt->maxlevel()) {
			std::stringstream ss_msg;
			ss_msg << "wrong mask: farest bits=" << selmask[ngsz-1]
           << " maxlev=" << ngt->maxlevel() << "\n";
			exit_error(IRSTLM_ERROR_DATA, ss_msg.str());
    }

    //ngramtable* ngt2=new ngramtable(NULL,ngsz,NULL,NULL,0,NULL,0,table_type);
    ngramtable* ngt2=new ngramtable(NULL,ngsz,NULL,NULL,NULL,0,0,NULL,0,table_type);

    ngt2->dict->incflag(1);

    ngram ng(ngt->dict);
    ngram png(ngt->dict,ngsz);
    ngram ng2(ngt2->dict,ngsz);

    ngt->scan(ng,INIT,ngt->maxlevel());
    long c=0;
    while (ngt->scan(ng,CONT,ngt->maxlevel())) {
      //projection
      for (int j=0; j<ngsz; j++)
        *png.wordp(j+1)=*ng.wordp(selmask[j]);
      png.freq=ng.freq;
      //transfer
      ng2.trans(png);
      ngt2->put(ng2);
      if (!(++c % 1000000)) cerr << ".";
    }

    char info[100];
    sprintf(info,"hm%s",hmask);
    ngt2->ngtype(info);

    //makes ngt2 aware of oov code
    int oov=ngt2->dict->getcode(ngt2->dict->OOV());
    if(oov>=0) ngt2->dict->oovcode(oov);

    for (int j=0; j<ngt->dict->size(); j++) {
      ngt2->dict->incfreq(ngt2->dict->encode(ngt->dict->decode(j)),
                          ngt->dict->freq(j));
    }

    cerr <<" oov: " << ngt2->dict->freq(ngt2->dict->oovcode()) << "\n";

    delete ngt;
    ngt=ngt2;
  }


  if (tlm && table_type==LEAFPROB) {
    ngram ng(ngt->dict);
    cout.setf(ios::scientific);

    cout << "> ";
    while(cin >> ng) {
      ngt->bo_state(0);
      if (ng.size>=ngsz) {
        cout << ng << " p= " << log(ngt->prob(ng));
        cout << " bo= " << ngt->bo_state() << "\n";
      } else
        cout << ng << " p= NULL\n";

      cout << "> ";
    }

  }


  if (ftlm && table_type==LEAFPROB) {

    ngram ng(ngt->dict);
    cout.setf(ios::fixed);
    cout.precision(2);

    mfstream inptxt(ftlm,ios::in);
    int Nbo=0,Nw=0,Noov=0;
    float logPr=0,PP=0,PPwp=0;

    int bos=ng.dict->encode(ng.dict->BoS());

    while(inptxt >> ng) {

      // reset ngram at begin of sentence
      if (*ng.wordp(1)==bos) {
        ng.size=1;
        continue;
      }

      ngt->bo_state(0);
      if (ng.size>=1) {
        logPr+=log(ngt->prob(ng));
        if (*ng.wordp(1) == ngt->dict->oovcode())
          Noov++;

        Nw++;
        if (ngt->bo_state()) Nbo++;
      }
    }

    PP=exp(-logPr/Nw);
    PPwp= PP * exp(Noov * log(10000000.0-ngt->dict->size())/Nw);

    cout << "%%% NGT TEST OF SMT LM\n";
    cout << "%% LM=" << inp << " SIZE="<< ngt->maxlevel();
    cout << "   TestFile="<< ftlm << "\n";
    cout << "%% OOV PENALTY = 1/" << 10000000.0-ngt->dict->size() << "\n";


    cout << "%% Nw=" << Nw << " PP=" << PP << " PPwp=" << PPwp
         << " Nbo=" << Nbo << " Noov=" << Noov
         << " OOV=" << (float)Noov/Nw * 100.0 << "%\n";

  }


  if (memuse)  ngt->stat(0);


  if (iknfile) { //compute and save statistics of Improved Kneser Ney smoothing

    ngram ng(ngt->dict);
    int n1,n2,n3,n4;
    int unover3=0;
    mfstream iknstat(iknfile,ios::out); //output of ikn statistics

    for (int l=1; l<=ngt->maxlevel(); l++) {

      cerr << "level " << l << "\n";
      iknstat << "level: " << l << " ";

      cerr << "computing statistics\n";

      n1=0;
      n2=0;
      n3=0,n4=0;

      ngt->scan(ng,INIT,l);

      while(ngt->scan(ng,CONT,l)) {

        //skip ngrams containing _OOV
        if (l>1 && ng.containsWord(ngt->dict->OOV(),l)) {
          //cerr << "skp ngram" << ng << "\n";
          continue;
        }

        //skip n-grams containing </s> in context
        if (l>1 && ng.containsWord(ngt->dict->EoS(),l-1)) {
          //cerr << "skp ngram" << ng << "\n";
          continue;
        }

        //skip 1-grams containing <s>
        if (l==1 && ng.containsWord(ngt->dict->BoS(),l)) {
          //cerr << "skp ngram" << ng << "\n";
          continue;
        }

        if (ng.freq==1) n1++;
        else if (ng.freq==2) n2++;
        else if (ng.freq==3) n3++;
        else if (ng.freq==4) n4++;
        if (l==1 && ng.freq >=3) unover3++;

      }


      cerr << " n1: " << n1 << " n2: " << n2 << " n3: " << n3 << " n4: " << n4 << "\n";
      iknstat << " n1: " << n1 << " n2: " << n2 << " n3: " << n3 << " n4: " << n4 << " unover3: " << unover3 << "\n";

    }

  }

    if (out){
    if (bin) ngt->savebin(out,ngsz);
    else if (outputredisformat) ngt->savetxt(out,ngsz,true,true,
                                             1);
    else if (outputgoogleformat) ngt->savetxt(out,ngsz,true,false);
    else ngt->savetxt(out,ngsz,false,false);
    }
}

