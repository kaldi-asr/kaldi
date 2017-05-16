// $Id: ngramtable.cpp 35 2010-07-19 14:52:11Z nicolabertoldi $

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

#include <sstream>
#include "util.h"
#include "mfstream.h"
#include "math.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "ngramtable.h"
#include "crc.h"

using namespace std;

tabletype::tabletype(TABLETYPE tt,int codesize) {

    if (codesize<=4 && codesize>0)
      CODESIZE=codesize;
    else {
			exit_error(IRSTLM_ERROR_DATA,"ngramtable wrong codesize");
    }

    code_range[1]=255;
    code_range[2]=65535;
    code_range[3]=16777214;
    code_range[4]=2147483640;
    code_range[6]=140737488360000LL; //stay below true limit
//	code_range[6]=281474977000000LL; //stay below true limit

    //information which is useful to initialize
    //LEAFPROB tables
    L_FREQ_SIZE=FREQ1;

    WORD_OFFS  =0;
    MSUCC_OFFS =CODESIZE;
    MTAB_OFFS  =MSUCC_OFFS+CODESIZE;
    FLAGS_OFFS =MTAB_OFFS+PTRSIZE;

    switch (tt) {

    case COUNT:
      SUCC1_OFFS =0;
      SUCC2_OFFS =0;
      BOFF_OFFS  =0;
      I_FREQ_OFFS=FLAGS_OFFS+CHARSIZE;
      I_FREQ_NUM=1;
      L_FREQ_NUM=1;

      ttype=tt;
      break;

    case FULL:
    case IMPROVEDKNESERNEY_B:
    case IMPROVEDSHIFTBETA_B:
      SUCC1_OFFS =FLAGS_OFFS+CHARSIZE;
      SUCC2_OFFS =SUCC1_OFFS+CODESIZE;
      BOFF_OFFS  =SUCC2_OFFS+CODESIZE;
      I_FREQ_OFFS=BOFF_OFFS+INTSIZE;
      L_FREQ_OFFS=CODESIZE;
      I_FREQ_NUM=2;
      L_FREQ_NUM=1;

      ttype=tt;
      break;

    case IMPROVEDKNESERNEY_I:
    case IMPROVEDSHIFTBETA_I:
      SUCC1_OFFS =FLAGS_OFFS+CHARSIZE;
      SUCC2_OFFS =SUCC1_OFFS+CODESIZE;
      BOFF_OFFS  =0;
      I_FREQ_OFFS=SUCC2_OFFS+CODESIZE;
      L_FREQ_OFFS=CODESIZE;
      I_FREQ_NUM=2;
      L_FREQ_NUM=1;

      ttype=tt;
      break;

    case SIMPLE_I:
      SUCC1_OFFS = 0;
      SUCC2_OFFS = 0;
      BOFF_OFFS  = 0;
      I_FREQ_OFFS= FLAGS_OFFS+CHARSIZE;
      L_FREQ_OFFS=CODESIZE;
      I_FREQ_NUM=1;
      L_FREQ_NUM=1;

      ttype=tt;
      break;

    case SIMPLE_B:
      SUCC1_OFFS  = 0;
      SUCC2_OFFS  = 0;
      BOFF_OFFS   = FLAGS_OFFS+CHARSIZE;
      I_FREQ_OFFS = BOFF_OFFS+INTSIZE;
      L_FREQ_OFFS = CODESIZE;
      I_FREQ_NUM  = 1;
      L_FREQ_NUM  = 1;

      ttype=tt;
      break;
				
		case KNESERNEY_I:
		case SHIFTBETA_I:
      SUCC1_OFFS = FLAGS_OFFS+CHARSIZE;
      SUCC2_OFFS = 0;
      BOFF_OFFS  = 0;
      I_FREQ_OFFS= SUCC1_OFFS+CODESIZE;
      L_FREQ_OFFS=CODESIZE;
      I_FREQ_NUM=1;
      L_FREQ_NUM=1;

      ttype=tt;
      break;

		case KNESERNEY_B:
    case SHIFTBETA_B:
      SUCC1_OFFS  = FLAGS_OFFS+CHARSIZE;
      SUCC2_OFFS  = 0;
      BOFF_OFFS   = SUCC1_OFFS+CODESIZE;
      I_FREQ_OFFS = BOFF_OFFS+INTSIZE;
      L_FREQ_OFFS = CODESIZE;
      I_FREQ_NUM  = 1;
      L_FREQ_NUM  = 1;

      ttype=tt;
      break;

    case LEAFPROB:
    case FLEAFPROB:
      SUCC1_OFFS  = 0;
      SUCC2_OFFS  = 0;
      BOFF_OFFS   = 0;
      I_FREQ_OFFS = FLAGS_OFFS+CHARSIZE;
      I_FREQ_NUM  = 0;
      L_FREQ_NUM  = 1;

      ttype=tt;
      break;

    case LEAFPROB2:
      SUCC1_OFFS =0;
      SUCC2_OFFS =0;
      BOFF_OFFS  =0;
      I_FREQ_OFFS=FLAGS_OFFS+CHARSIZE;
      I_FREQ_NUM=0;
      L_FREQ_NUM=2;

      ttype=LEAFPROB;
      break;

    case LEAFPROB3:
      SUCC1_OFFS =0;
      SUCC2_OFFS =0;
      BOFF_OFFS  =0;
      I_FREQ_OFFS=FLAGS_OFFS+CHARSIZE;
      I_FREQ_NUM=0;
      L_FREQ_NUM=3;

      ttype=LEAFPROB;
      break;

    case LEAFPROB4:
      SUCC1_OFFS =0;
      SUCC2_OFFS =0;
      BOFF_OFFS  =0;
      I_FREQ_OFFS=FLAGS_OFFS+CHARSIZE;
      I_FREQ_NUM=0;
      L_FREQ_NUM=4;

      ttype=LEAFPROB;
      break;

    default:
      MY_ASSERT(tt==COUNT);
    }

    L_FREQ_OFFS=CODESIZE;
};

ngramtable::ngramtable(char* filename,int maxl,char* /* unused parameter: is */, 
					   dictionary* extdict /* external dictionary */,char* filterdictfile,
					   int googletable,int dstco,char* hmask, int inplen,TABLETYPE ttype,
					   int codesize): tabletype(ttype,codesize)
{

  cerr << "[codesize " << CODESIZE << "]\n";
  char header[100];

  info[0]='\0';

  corrcounts=0;

  if (filename) {
    int n;
    mfstream inp(filename,ios::in );

    inp >> header;

    if (strncmp(header,"nGrAm",5)==0 || strncmp(header,"NgRaM",5)==0) {
      inp >> n;
      inp >> card;
      inp >> info;
      if (strcmp(info,"LM_")==0) {
        inp >> resolution;
        inp >> decay;
				char       info2[100];
        sprintf(info2,"%s %d %f",info,resolution,decay);
				strcpy(info, info2);
      } else { //default for old LM probs
        resolution=10000000;
        decay=0.9999;
      }

      maxl=n; //owerwrite maxl

      cerr << n << " " << card << " " << info << "\n";
    }

    inp.close();
  }

  if (!maxl) {
		exit_error(IRSTLM_ERROR_DATA,"ngramtable: ngram size must be specified");
  }

  //distant co-occurreces works for bigrams and trigrams
  if (dstco && (maxl!=2) && (maxl!=3)) {
		exit_error(IRSTLM_ERROR_DATA,"distant co-occurrences work with 2-gram and 3-gram");
  }

  maxlev=maxl;

  //Root not must have maximum frequency size

  treeflags=INODE | FREQ6;
  tree=(node) new char[inodesize(6)];
  memset(tree,0,inodesize(6));


  //1-gram table initial flags
  if (maxlev>1)
    mtflags(tree,INODE | FREQ4);
  else if (maxlev==1)
    mtflags(tree,LNODE | FREQ4);
  else {
		exit_error(IRSTLM_ERROR_DATA,"ngramtable: wrong level setting");
  }

  word(tree,0); // dummy variable

  if (I_FREQ_NUM)
    freq(tree,treeflags,0); // frequency of all n-grams

  msucc(tree,0);     // number of different n-grams
  mtable(tree,NULL); // table of n-gram

  mem=new storage(256,10000);

  mentr=new long long[maxlev+1];
  memory= new long long[maxlev+1];
  occupancy= new long long[maxlev+1];

//Book keeping of occupied memory
  mentr[0]=1;
  memory[0]=inodesize(6); // root is an inode with  highest frequency
  occupancy[0]=inodesize(6); // root is an inode with  highest frequency

  for (int i=1; i<=maxlev; i++)
    mentr[i]=memory[i]=occupancy[i]=0;

  dict=new dictionary(NULL,1000000);

  if (!filename) return ;

  filterdict=NULL;
  if (filterdictfile) {
    filterdict=new dictionary(filterdictfile,1000000);
    /*
     filterdict->incflag(1);
    		filterdict->encode(BOS_);
    		filterdict->encode(EOS_);
    		filterdict->incflag(0);
    */
  }

  // switch to specific loading methods

  if ((strncmp(header,"ngram",5)==0) ||
      (strncmp(header,"NGRAM",5)==0)) {
		exit_error(IRSTLM_ERROR_DATA,"this ngram file format is no more supported");
  }

  if (strncmp(header,"nGrAm",5)==0)
    loadtxt(filename);
  else if (strncmp(header,"NgRaM",5)==0)
    loadbin(filename);
  else if (dstco>0)
    generate_dstco(filename,dstco);
  else if (hmask != NULL)
    generate_hmask(filename,hmask,inplen);
  else if (googletable)
    loadtxt(filename,googletable);
  else
    generate(filename,extdict);



  if (tbtype()==LEAFPROB) {
    du_code=dict->encode(DUMMY_);
    bo_code=dict->encode(BACKOFF_);
  }
}

void ngramtable::savetxt(char *filename,int depth,bool googleformat,bool hashvalue,int startfrom)
{
    char ngstring[10000];
    
    if (depth>maxlev) {
        exit_error(IRSTLM_ERROR_DATA,"ngramtable::savetxt: wrong n-gram size");
    }
    
    if (startfrom>0 && !googleformat) {
        exit_error(IRSTLM_ERROR_DATA,
                   "ngramtable::savetxt: multilevel output only allowed in googleformat");
    }
    
    depth=(depth>0?depth:maxlev);
    
    card=mentr[depth];
    
    ngram ng(dict);
    
    if (googleformat)
    cerr << "savetxt in Google format: nGrAm " <<  depth << " " << card << " " << info << "\n";
    else
    cerr << "savetxt: nGrAm " <<  depth << " " << card << " " << info << "\n";
    
    mfstream out(filename,ios::out );
    
    if (!googleformat){
        out << "nGrAm " << depth << " " << card << " " << info << "\n";
        dict->save(out);
    }
    
    if (startfrom<=0 || startfrom > depth) startfrom=depth;
    
    for (int d=startfrom;d<=depth;d++){
        scan(ng,INIT,d);
        
        while(scan(ng,CONT,d)){
            
            if (hashvalue){
                strcpy(ngstring,ng.dict->decode(*ng.wordp(ng.size)));
                for (int i=ng.size-1; i>0; i--){
                    strcat(ngstring," ");
                    strcat(ngstring,ng.dict->decode(*ng.wordp(i)));
                }
                out << ngstring << "\t" << ng.freq << "\t" << crc16_ccitt(ngstring,strlen(ngstring)) << "\n";
            }
            else
            
            out << ng << "\n";
            
            
        }
    }
    cerr << "\n";
    
    out.close();
}


void ngramtable::loadtxt(char *filename,int googletable)
{

  ngram ng(dict);;

  cerr << "loadtxt:" << (googletable?"google format":"std table");

  mfstream inp(filename,ios::in);

  int i,c=0;

  if (googletable) {
    dict->incflag(1);
  } else {
    char header[100];
    inp.getline(header,100);
    cerr << header ;
    dict->load(inp);
  }

  while (!inp.eof()) {

    for (i=0; i<maxlev; i++) inp >> ng;

    inp >> ng.freq;

    if (ng.size==0) continue;

    //update dictionary frequency when loading from
    if (googletable) dict->incfreq(*ng.wordp(1),ng.freq);

    // if filtering dictionary exists
    // and if the first word of the ngram does not belong to it
    // do not insert the ngram

    if (filterdict) {
      int code=filterdict->encode(dict->decode(*ng.wordp(maxlev)));
      if (code!=filterdict->oovcode())	put(ng);
    } else put(ng);

    ng.size=0;

    if (!(++c % 1000000)) cerr << ".";

  }

  if (googletable) {
    dict->incflag(0);
  }

  cerr << "\n";

  inp.close();
}



void ngramtable::savebin(mfstream& out,node nd,NODETYPE ndt,int lev,int mlev)
{

  out.write(nd+WORD_OFFS,CODESIZE);

  //write frequency

  int offs=(ndt & LNODE)?L_FREQ_OFFS:I_FREQ_OFFS;

  int frnum=1;
  if (tbtype()==LEAFPROB && (ndt & LNODE))
    frnum=L_FREQ_NUM;

  if ((ndt & LNODE) || I_FREQ_NUM) { //check if to write freq
    if (ndt & FREQ1)
      out.write(nd+offs,1 * frnum);
    else if (ndt & FREQ2)
      out.write(nd+offs,2 * frnum);
    else if (ndt & FREQ3)
      out.write(nd+offs,3 * frnum);
    else
      out.write(nd+offs,INTSIZE * frnum);
  }

  if ((lev <mlev) && (ndt & INODE)) {

    unsigned char fl=mtflags(nd);
    if (lev==(mlev-1))
      //transforms flags into a leaf node
      fl=(fl & ~INODE) | LNODE;

    out.write((const char*) &fl,CHARSIZE);
    fl=mtflags(nd);

    out.write(nd+MSUCC_OFFS,CODESIZE);

    int msz=mtablesz(nd);
    int  m=msucc(nd);

    for (int i=0; i<m; i++)
      savebin(out,mtable(nd) + i * msz,fl,lev+1,mlev);
  }
}


void ngramtable::savebin(mfstream& out)
{

  int depth=maxlev;

  card=mentr[depth];

  cerr << "ngramtable::savebin ";

  out.writex((char *)&depth,INTSIZE);

  out.write((char *)&treeflags,CHARSIZE);

  savebin(out,tree,treeflags,0,depth);

  cerr << "\n";
}


void ngramtable::savebin(char *filename,int depth)
{

  if (depth > maxlev) {
		exit_error(IRSTLM_ERROR_DATA,"ngramtable::savebin: wrong n-gram size");
  }

  depth=(depth>0?depth:maxlev);

  card=mentr[depth];

  cerr << "savebin NgRaM " << depth << " " << card;

  mfstream out(filename,ios::out );

  if (dict->oovcode()!=-1) //there are OOV words
    out << "NgRaM_ " << depth << " " << card << " " << info << "\n";
  else
    out << "NgRaM " << depth << " " << card << " " << info << "\n";

  dict->save(out);

  out.writex((char *)&depth,INTSIZE);

  out.write((char *)&treeflags,CHARSIZE);

  savebin(out,tree,treeflags,0,depth);

  out.close();

  cerr << "\n";
}


void ngramtable::loadbin(mfstream& inp,node nd,NODETYPE ndt,int lev)
{
  static int c=0;

  // read code
  inp.read(nd+WORD_OFFS,CODESIZE);

  // read frequency
  int offs=(ndt & LNODE)?L_FREQ_OFFS:I_FREQ_OFFS;

  int frnum=1;
  if (tbtype()==LEAFPROB && (ndt & LNODE))
    frnum=L_FREQ_NUM;

  if ((ndt & LNODE) || I_FREQ_NUM) { //check if to read freq
    if (ndt & FREQ1)
      inp.read(nd+offs,1 * frnum);
    else if (ndt & FREQ2)
      inp.read(nd+offs,2 * frnum);
    else if (ndt & FREQ3)
      inp.read(nd+offs,3 * frnum);
    else
      inp.read(nd+offs,4 * frnum);
  }

  if (ndt & INODE) {

    //read flags
    inp.read(nd+FLAGS_OFFS,CHARSIZE);
    unsigned char fl=mtflags(nd);

    //read #of multiple entries
    inp.read(nd+MSUCC_OFFS,CODESIZE);
    int m=msucc(nd);

    if (m>0) {
      //read multiple entries
      int msz=mtablesz(nd);
      table mtb=mtable(nd);
      //table entries increase
      grow(&mtb,INODE,lev+1,m,msz);

      for (int i=0; i<m; i++)
        loadbin(inp,mtb + i * msz,fl,lev+1);

      mtable(nd,mtb);
    }

    mentr[lev+1]+=m;
    occupancy[lev+1]+=(m * mtablesz(nd));

  } else if (!(++c % 1000000)) cerr << ".";

}



void ngramtable::loadbin(mfstream& inp)
{

  cerr << "loadbin ";

  inp.readx((char *)&maxlev,INTSIZE);
  inp.read((char *)&treeflags,CHARSIZE);

  loadbin(inp,tree,treeflags,0);

  cerr << "\n";
}


void ngramtable::loadbin(const char *filename)
{

  cerr << "loadbin ";
  mfstream inp(filename,ios::in );

  //skip header
  char header[100];
  inp.getline(header,100);

  cerr << header ;

  dict->load(inp);

  inp.readx((char *)&maxlev,INTSIZE);
  inp.read((char *)&treeflags,CHARSIZE);

  loadbin(inp,tree,treeflags,0);

  inp.close();

  cerr << "\n";
}


void ngramtable::generate(char *filename, dictionary* extdict)
{
  mfstream inp(filename,ios::in);
  int i,c=0;

  if (!inp) {
		std::stringstream ss_msg;
		ss_msg << "cannot open " << filename;
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
  }

  cerr << "load:";

  ngram ng(extdict==NULL?dict:extdict); //use possible prescribed dictionary
  if (extdict) dict->genoovcode();
	
  ngram ng2(dict);
  dict->incflag(1);

  cerr << "prepare initial n-grams to make table consistent\n";
  for (i=1; i<maxlev; i++) {
    ng.pushw(dict->BoS());
    ng.freq=1;
  };

  while (inp >> ng) {
	
	if (ng.size>maxlev) ng.size=maxlev;  //speeds up 
	  
    ng2.trans(ng); //reencode with new dictionary

    check_dictsize_bound();

    if (ng2.size) dict->incfreq(*ng2.wordp(1),1);

    // if filtering dictionary exists
    // and if the first word of the ngram does not belong to it
    // do not insert the ngram
    if (filterdict) {
      int code=filterdict->encode(dict->decode(*ng2.wordp(maxlev)));
      if (code!=filterdict->oovcode())	put(ng2);
    } else put(ng2);
	  
    if (!(++c % 1000000)) cerr << ".";

  }    
	
  cerr << "adding some more n-grams to make table consistent\n";
  for (i=1; i<=maxlev; i++) {
    ng2.pushw(dict->BoS());
    ng2.freq=1;

    // if filtering dictionary exists
    // and if the first word of the ngram does not belong to it
    // do not insert the ngram
    if (filterdict) {
      int code=filterdict->encode(dict->decode(*ng2.wordp(maxlev)));
      if (code!=filterdict->oovcode())	put(ng2);
    } else put(ng2);
  };

  dict->incflag(0);
  inp.close();
  strcpy(info,"ngram");

  cerr << "\n";
}

void ngramtable::generate_hmask(char *filename,char* hmask,int inplen)
{
  mfstream inp(filename,ios::in);

  if (!inp) {
		std::stringstream ss_msg;
		ss_msg << "cannot open " << filename;
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
  }

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
  if (i!= maxlev) {
		std::stringstream ss_msg;
		ss_msg << "wrong mask: 1 bits=" << i << " maxlev=" << maxlev;
		exit_error(IRSTLM_ERROR_DATA, ss_msg.str());
  }

  cerr << "load:";

  ngram ng(dict);
  ngram ng2(dict);
  dict->incflag(1);
	long c=0;
  while (inp >> ng) {

    if (inplen && ng.size<inplen) continue;

    ng2.trans(ng); //reencode with new dictionary
    ng.size=0;    //reset  ng

    if (ng2.size >= selmask[maxlev-1]) {
      for (int j=0; j<maxlev; j++)
        *ng2.wordp(j+1)=*ng2.wordp(selmask[i]);

      //cout << ng2 << "size:" << ng2.size << "\n";
      check_dictsize_bound();

      put(ng2);
    }

    if (ng2.size) dict->incfreq(*ng2.wordp(1),1);

    if (!(++c % 1000000)) cerr << ".";
  };

  dict->incflag(0);
  inp.close();
  sprintf(info,"hm%s\n",hmask);

  cerr << "\n";
}

int cmpint(const void *a,const void *b)
{
  return (*(int *)b)-(*(int *)a);
}

void ngramtable::generate_dstco(char *filename,int dstco)
{
  mfstream inp(filename,ios::in);
  int c=0;

  if (!inp) {
		std::stringstream ss_msg;
		ss_msg << "cannot open " << filename;
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
  }

  cerr << "load distant co-occurrences:";
  if (dstco>MAX_NGRAM) {
    inp.close();
		std::stringstream ss_msg;
		ss_msg << "window size (" << dstco << ") exceeds MAXNGRAM";
		exit_error(IRSTLM_ERROR_DATA, ss_msg.str());
  }

  ngram ng(dict);
  ngram ng2(dict);
  ngram dng(dict);
  dict->incflag(1);

  while (inp >> ng) {
    if (ng.size) {

      ng2.trans(ng); //reencode with new dictionary

      if (ng2.size>dstco) ng2.size=dstco; //maximum distance

      check_dictsize_bound();

      dict->incfreq(*ng2.wordp(1),1);

      if (maxlev == 1 )
        cerr << "maxlev is wrong! (Possible values are 2 or 3)\n";

      else if (maxlev == 2 ) { //maxlev ==2
        dng.size=2;
        dng.freq=1;

        //cerr << "size=" << ng2.size << "\n";

        for (int i=2; i<=ng2.size; i++) {

          if (*ng2.wordp(1)<*ng2.wordp(i)) {
            *dng.wordp(2)=*ng2.wordp(i);
            *dng.wordp(1)=*ng2.wordp(1);
          } else {
            *dng.wordp(1)=*ng2.wordp(i);
            *dng.wordp(2)=*ng2.wordp(1);
          }
          //cerr << dng << "\n";
          put(dng);
        }
        if (!(++c % 1000000)) cerr << ".";
      } else { //maxlev ==3
        dng.size=3;
        dng.freq=1;

        //cerr << "size=" << ng2.size << "\n";
        int ar[3];

        ar[0]=*ng2.wordp(1);
        for (int i=2; i<ng2.size; i++) {
          ar[1]=*ng2.wordp(i);
          for (int j=i+1; j<=ng2.size; j++) {
            ar[2]=*ng2.wordp(j);

            //sort ar
            qsort(ar,3,sizeof(int),cmpint);

            *dng.wordp(1)=ar[0];
            *dng.wordp(2)=ar[1];
            *dng.wordp(3)=ar[2];

            //	    cerr << ng2 << "\n";
            //cerr << dng << "\n";
            //cerr << *dng.wordp(1) << " "
            //	 << *dng.wordp(2) << " "
            //	 << *dng.wordp(3) << "\n";
            put(dng);
          }
        }
      }
    }
  }
  dict->incflag(0);
  inp.close();
  sprintf(info,"co-occ%d\n",dstco);
  cerr << "\n";
}



void ngramtable::augment(ngramtable* ngt)
{

  if (ngt->maxlev != maxlev) {
		exit_error(IRSTLM_ERROR_DATA,"ngramtable::augment augmentation is not possible due to table incompatibility");
  }

  if (ngt->dict->oovcode()!=-1)
    cerr <<"oov: " << ngt->dict->freq(ngt->dict->oovcode()) << "\n";
  cerr <<"size: " << ngt->dict->size() << "\n";

  if (dict->oovcode()!=-1)
    cerr <<"oov: " << dict->freq(dict->oovcode()) << "\n";
  cerr <<"size: " << dict->size() << "\n";


  dict->incflag(1);
  cerr << "augmenting ngram table\n";
  ngram ng1(ngt->dict);
  ngram ng2(dict);
  ngt->scan(ng1,INIT);
  int c=0;
  while (ngt->scan(ng1,CONT)) {
    ng2.trans(ng1);
    put(ng2);
    if ((++c % 1000000) ==0) cerr <<".";
  }
  cerr << "\n";

  for (int i=0; i<ngt->dict->size(); i++)
    dict->incfreq(dict->encode(ngt->dict->decode(i)),
                  ngt->dict->freq(i));

  dict->incflag(0);

  int oov=dict->getcode(dict->OOV());

  if (oov>=0) {
    dict->oovcode(oov);
  }

  cerr << "oov: " << dict->freq(dict->oovcode()) << "\n";
  cerr << "size: " << dict->size() << "\n";
}

void ngramtable::show()
{

  ngram ng(dict);

  scan(ng,INIT);
  cout << "Stampo contenuto della tabella\n";
  while (scan(ng)) {
    cout << ng << "\n";
  }
}



int ngramtable::mybsearch(char *ar, int n, int size, unsigned char *key, int *idx)
{
  if (n==0) return 0;

  int low = 0, high = n;
  *idx=0;
  unsigned char *p=NULL;
  int result;

#ifdef INTERP_SEARCH
  char* lp;
  char* hp;
#endif

  /* return idx with the first
   position equal or greater than key */

  /* Warning("start bsearch \n"); */


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

    if (result < 0) {
      high = *idx;
    } else if (result > 0) {
      low = ++(*idx);
    } else
      return 1;
  }

  *idx=low;

  return 0;

}

void *ngramtable::search(table *tb,NODETYPE ndt,int lev,int n,int sz,int *ngp,
                         ACTION action,char **found)
{

  char w[CODESIZE];
  putmem(w,ngp[0],0,CODESIZE);
  int wint=ngp[0];


  // index returned by mybsearch

  if (found) *found=NULL;

  int idx=0;

  switch(action) {

  case ENTER:

    if (!*tb ||
        !mybsearch(*tb,n,sz,(unsigned char *)w,&idx)) {
      // let possibly grow the table
      grow(tb,ndt,lev,n,sz); // devo aggiungere un elemento n+1

      //shift table by one

      memmove(*tb + (idx+1) * sz,
              *tb + idx * sz,
              (n-idx) * sz);

      memset(*tb + idx * sz , 0 , sz);

      word(*tb + idx * sz, wint);

    } else if (found) *found=*tb + ( idx * sz );

    return *tb + ( idx * sz );

    break;


  case FIND:

    if (!*tb ||
        !mybsearch(*tb,n,sz,(unsigned char *)w,&idx))
      return 0;
    else if (found) *found=*tb + (idx * sz);

    return *tb + (idx * sz);

    break;

  case DELETE:

    if (*tb &&
        mybsearch(*tb,n,sz,(unsigned char *)w,&idx)) {
      //shift table down by one

      static char buffer[100];

      memcpy(buffer,*tb + idx * sz , sz);

      if (idx <(n-1))
        memmove(*tb + idx * sz,
                *tb + (idx + 1) * sz,
                (n-idx-1) * sz);

      //put the deleted item after the last item

      memcpy(*tb + (n-1) * sz , buffer , sz);

      if (found) *found=*tb + (n-1) * sz ;

      return *tb + (n-1) * sz ;

    } else

      return NULL;

    break;

  default:
    cerr << "this option is not implemented yet\n";
    break;
  }

  return NULL;

}

int ngramtable::comptbsize(int n)
{

  if (n>16384)
    return(n/16384)*16384+(n % 16384?16384:0);
  else if (n>8192) return 16384;
  else if (n>4096) return 8192;
  else if (n>2048) return 4096;
  else if (n>1024) return 2048;
  else if (n>512) return 1024;
  else if (n>256) return 512;
  else if (n>128) return 256;
  else if (n>64) return 128;
  else if (n>32) return 64;
  else if (n>16) return 32;
  else if (n>8) return 16;
  else if (n>4) return 8;
  else if (n>2) return 4;
  else if (n>1) return 2;
  else return 1;

}


char **ngramtable::grow(table *tb,NODETYPE ndt,int lev,
                        int n,int sz,NODETYPE oldndt)
{
  int inc;
  int num;

  //memory pools for inode/lnode tables

  if (oldndt==0) {

    if ((*tb==NULL) && n>0) {
      // n is the target number of entries
      //first allocation

      if (n>16384)
        inc=(n/16384)*16384+(n % 16384?16384:0);
      else if (n>8192) inc=16384;
      else if (n>4096) inc=8192;
      else if (n>2048) inc=4096;
      else if (n>1024) inc=2048;
      else if (n>512) inc=1024;
      else if (n>256) inc=512;
      else if (n>128) inc=256;
      else if (n>64) inc=128;
      else if (n>32) inc=64;
      else if (n>16) inc=32;
      else if (n>8) inc=16;
      else if (n>4) inc=8;
      else if (n>2) inc=4;
      else if (n>1) inc=2;
      else inc=1;

      n=0; //inc is the correct target size

    }

    else {
      // table will be extended on demand
      // I'm sure that one entry will be
      // added next

      // check multiples of 1024
      if ((n>=16384) && !(n % 16384)) inc=16384;
      else {
        switch (n) {
        case 0:
          inc=1;
          break;
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
          inc=n;
          break;
        default:
          return tb;
        }
      }
    }

    table ntb=(char *)mem->reallocate(*tb,n * sz,(n + inc) * sz);
	  
    memory[lev]+= (inc * sz);

    *tb=ntb;
  }

  else {
    //change frequency type of table
    //no entries will be added now

    int oldsz=0;

    // guess the current memory size !!!!
    num=comptbsize(n);

    if ((ndt & INODE) && I_FREQ_NUM) {
      if (oldndt & FREQ1)
        oldsz=inodesize(1);
      else if (oldndt & FREQ2)
        oldsz=inodesize(2);
      else if (oldndt & FREQ3)
        oldsz=inodesize(3);
      else if (oldndt & FREQ4)
        oldsz=inodesize(4);
      else {
				exit_error(IRSTLM_ERROR_DATA,"ngramtable::grow functionality not available");
      }
    } else if (ndt & LNODE) {
      if (oldndt & FREQ1)
        oldsz=lnodesize(1);
      else if (oldndt & FREQ2)
        oldsz=lnodesize(2);
      else if (oldndt & FREQ3)
        oldsz=lnodesize(3);
      else if (oldndt & FREQ4)
        oldsz=lnodesize(4);
      else {
				exit_error(IRSTLM_ERROR_DATA,"ngramtable::grow functionality not available");
      }
    }

    table ntb=(char *)mem->allocate(num * sz);
    memset((char *)ntb,0,num * sz);

    if (ndt & INODE)
      for (int i=0; i<n; i++) {
        word(ntb+i*sz,word(*tb+i*oldsz));
        msucc(ntb+i*sz,msucc(*tb+i*oldsz));
        mtflags(ntb+i*sz,mtflags(*tb+i*oldsz));
        mtable(ntb+i*sz,mtable(*tb+i*oldsz));
        for (int j=0; j<I_FREQ_NUM; j++)
          setfreq(ntb+i*sz,ndt,getfreq(*tb+i*oldsz,oldndt,j),j);
      }
    else
      for (int i=0; i<n; i++) {
        word(ntb+i*sz,word(*tb+i*oldsz));
        for (int j=0; j<L_FREQ_NUM; j++)
          setfreq(ntb+i*sz,ndt,getfreq(*tb+i*oldsz,oldndt,j),j);
      }

    mem->free(*tb,num * oldsz); //num is the correct size
    memory[lev]+=num * (sz - oldsz);
    occupancy[lev]+=n * (sz - oldsz);

    *tb=ntb;
  }

  return tb;

};


int ngramtable::put(ngram& ng)
{

  return ngramtable::put(ng,tree,treeflags,0);

}

int ngramtable::put(ngram& ng,node nd,NODETYPE ndt,int lev)
{
  char *found;
  node subnd;

  if (ng.size<maxlev) return 0;


  /*
    cerr << "l:" << lev << " put:" << ng << "\n";
  	cerr << "I_FREQ_NUM: " << I_FREQ_NUM << "\n";
  	cerr << "LNODE: " << (int) LNODE << "\n";
  	cerr << "ndt: " << (int) ndt << "\n";
  	*/

  for (int l=lev; l<maxlev; l++) {

    if (I_FREQ_NUM || (ndt & LNODE))
      freq(nd,ndt,freq(nd,ndt) + ng.freq);

    table mtb=mtable(nd);

    // it has to be added to the multiple table

    subnd=(char *)
          search(&mtb,
                 mtflags(nd),
                 l+1,
                 msucc(nd),
                 mtablesz(nd),
                 ng.wordp(maxlev-l),
                 ENTER,&found);

    if (!found) { //a new element has been added

      msucc(nd,msucc(nd)+1);

      mentr[l+1]++;
      occupancy[l+1]+=mtablesz(nd);

      unsigned char freq_flag;
      if (I_FREQ_NUM)
        //tree with internal freqs must
        //be never expanded during usage
        //of the secondary frequencies
        freq_flag=(ng.freq>65535?FREQ4:FREQ1);
      else
        //all leafprob with L_FREQ_NUM >=1
        //do NOT have INTERNAL freqs
        //will have freq size specified
        //by the resolution parameter
        //to avoid expansion
        freq_flag=L_FREQ_SIZE;

      if ((l+1)<maxlev) { //update mtable flags
        if ((l+2)<maxlev)
          mtflags(subnd,INODE | freq_flag);
        else
          mtflags(subnd,LNODE | freq_flag);

      }
    }

    // ... go on with the subtree

    // check if we must extend the subnode

    NODETYPE oldndt=mtflags(nd);

    if ((I_FREQ_NUM || (mtflags(nd) & LNODE))  &&
        (mtflags(nd) & FREQ1) &&
        ((freq(subnd,mtflags(nd))+ng.freq)>255))

      mtflags(nd,(mtflags(nd) & ~FREQ1) | FREQ2); //update flags


    if ((I_FREQ_NUM || (mtflags(nd) & LNODE))  &&
        (mtflags(nd) & FREQ2) &&
        ((freq(subnd,mtflags(nd))+ng.freq)>65535))

      mtflags(nd,(mtflags(nd) & ~FREQ2) | FREQ3); //update flags


    if ((I_FREQ_NUM || (mtflags(nd) & LNODE))  &&
        (mtflags(nd) & FREQ3) &&
        ((freq(subnd,mtflags(nd))+ng.freq)>16777215))

      mtflags(nd,(mtflags(nd) & ~FREQ3) | FREQ4); //update flags

    if ((I_FREQ_NUM || (mtflags(nd) & LNODE))  &&
        (mtflags(nd) & FREQ4) &&
        ((freq(subnd,mtflags(nd))+ng.freq)>4294967295LL))

      mtflags(nd,(mtflags(nd) & ~FREQ4) | FREQ6); //update flags

    if (mtflags(nd)!=oldndt) {
      // flags have changed, table has to be expanded
      //expand subtable
      cerr << "+"<<l+1;
      //table entries remain the same
      grow(&mtb,mtflags(nd),l+1,msucc(nd),mtablesz(nd),oldndt);
      cerr << "\b\b";
      //update subnode
      subnd=(char *)
            search(&mtb,
                   mtflags(nd),
                   l+1,
                   msucc(nd),
                   mtablesz(nd),
                   ng.wordp(maxlev-l),
                   FIND,&found);
    }


    mtable(nd,mtb);
    ndt=mtflags(nd);
    nd=subnd;
  }

  freq(nd, ndt, freq(nd,ndt) + ng.freq);

  return 1;
}



int ngramtable::get(ngram& ng,int n,int lev)
{

  node nd,subnd;
  char *found;
  NODETYPE ndt;

  MY_ASSERT(lev <= n && lev <= maxlev && ng.size >= n);

  if ((I_FREQ_NUM==0) && (lev < maxlev)) {
		exit_error(IRSTLM_ERROR_DATA,"ngramtable::get for this type of table ngram cannot be smaller than table size");
  }


  if (ng.wordp(n)) {

    nd=tree;
    ndt=treeflags;

    for (int l=0; l<lev; l++) {

      table mtb=mtable(nd);

      subnd=(char *)
            search(&mtb,
                   mtflags(nd),
                   l+1,
                   msucc(nd),
                   mtablesz(nd),
                   ng.wordp(n-l),
                   FIND,&found);

      ndt=mtflags(nd);
      nd=subnd;

      if (nd==0) return 0;
    }

    ng.size=n;
    ng.freq=freq(nd,ndt);
    ng.link=nd;
    ng.lev=lev;
    ng.pinfo=ndt; //parent node info

    if (lev<maxlev) {
      ng.succ=msucc(nd);
      ng.info=mtflags(nd);
    } else {
      ng.succ=0;
      ng.info=LNODE;
    }
    return 1;
  }
  return 0;
}


int ngramtable::scan(node nd,NODETYPE /* unused parameter: ndt */,int lev,ngram& ng,ACTION action,int maxl)
{

  MY_ASSERT(lev<=maxlev);

  if ((I_FREQ_NUM==0) && (maxl < maxlev)) {
		exit_error(IRSTLM_ERROR_MODEL,"ngramtable::scan ngram cannot be smaller than LEAFPROB table");
  }


  if (maxl==-1) maxl=maxlev;

  ng.size=maxl;

  switch (action) {


  case INIT:
    //reset ngram local indexes

    for (int l=0; l<=maxlev; l++) ng.midx[l]=0;

    return 1;

  case CONT:

    if (lev>(maxl-1)) return 0;

    if (ng.midx[lev]<msucc(nd)) {
      //put current word into ng
      *ng.wordp(maxl-lev)=
        word(mtable(nd)+ng.midx[lev] * mtablesz(nd));

      //inspect subtree
      //check if there is something left in the tree

      if (lev<(maxl-1)) {
        if (scan(mtable(nd) + ng.midx[lev] * mtablesz(nd),
                 INODE,
                 lev+1,ng,CONT,maxl))
          return 1;
        else {
          ng.midx[lev]++; //go to next
          for (int l=lev+1; l<=maxlev; l++) ng.midx[l]=0; //reset indexes

          return scan(nd,INODE,lev,ng,CONT,maxl); //restart scanning
        }
      } else {
        // put data into the n-gram

        *ng.wordp(maxl-lev)=
          word(mtable(nd)+ng.midx[lev] * mtablesz(nd));

        ng.freq=freq(mtable(nd)+ ng.midx[lev] * mtablesz(nd),mtflags(nd));
        ng.pinfo=mtflags(nd);

        if (maxl<maxlev) {
          ng.info=mtflags(mtable(nd)+ ng.midx[lev] * mtablesz(nd));
          ng.link=mtable(nd)+ng.midx[lev] * mtablesz(nd); //link to the node
          ng.succ=msucc(mtable(nd)+ ng.midx[lev] * mtablesz(nd));
        } else {
          ng.info=LNODE;
          ng.link=NULL;
          ng.succ=0;
        }

        ng.midx[lev]++;

        return 1;
      }
    } else
      return 0;

  default:
    cerr << "scan: not supported action\n";
    break;

  }
  return 0;
}


void ngramtable::freetree(node nd)
{
	int m=msucc(nd);
	int msz=mtablesz(nd);
	int truem=comptbsize(m);
	
	if (mtflags(nd) & INODE)
		for (int i=0; i<m; i++)
			freetree(mtable(nd) + i * msz);
	mem->free(mtable(nd),msz*truem);	
}


ngramtable::~ngramtable()
{
  freetree(tree);
  delete [] tree;
  delete mem;
  delete [] memory;
  delete [] occupancy;
  delete [] mentr;
  delete dict;
};

void ngramtable::stat(int level)
{
  long long totmem=0;
  long long totwaste=0;
  float mega=1024 * 1024;

  cout.precision(2);

  cout << "ngramtable class statistics\n";

  cout << "levels " << maxlev << "\n";
  for (int l=0; l<=maxlev; l++) {
    cout << "lev " << l
         << " entries "<< mentr[l]
         << " allocated mem " << memory[l]/mega << "Mb "
         << " used mem " << occupancy[l]/mega << "Mb \n";
    totmem+=memory[l];
    totwaste+=(memory[l]-occupancy[l]);
  }

  cout << "total allocated mem " << totmem/mega << "Mb ";
  cout << "wasted mem " << totwaste/mega << "Mb\n\n\n";

  if (level >1 ) dict->stat();

  cout << "\n\n";

  if (level >2) mem->stat();

}


double ngramtable::prob(ngram ong)
{

  if (ong.size==0) return 0.0;
  if (ong.size>maxlev) ong.size=maxlev;

  MY_ASSERT(tbtype()==LEAFPROB && ong.size<=maxlev);

  ngram ng(dict);
  ng.trans(ong);

  double bo;

  ng.size=maxlev;
  for (int s=ong.size+1; s<=maxlev; s++)
    *ng.wordp(s)=du_code;

  if (get(ng)) {

    if (ong.size>1 && resolution<10000000)
      return (double)pow(decay,(resolution-ng.freq));
    else
      return (double)(ng.freq+1)/10000000.0;

  } else { // backoff-probability

    bo_state(1); //set backoff state to 1

    *ng.wordp(1)=bo_code;

    if (get(ng))

      bo=resolution<10000000
         ?(double)pow(decay,(resolution-ng.freq))
         :(double)(ng.freq+1)/10000000.0;

    else
      bo=1.0;

    ong.size--;

    return bo * prob(ong);
  }
}


bool ngramtable::check_dictsize_bound()
{
  if (dict->size() >= code_range[CODESIZE]) {
		std::stringstream ss_msg;
		ss_msg << "dictionary size overflows code range " << code_range[CODESIZE];
		exit_error(IRSTLM_ERROR_MODEL, ss_msg.str());
  }
  return true;
}

int ngramtable::update(ngram ng) {

    if (!get(ng,ng.size,ng.size)) {
			std::stringstream ss_msg;
			ss_msg << "cannot find " << ng;
			exit_error(IRSTLM_ERROR_MODEL, ss_msg.str());
    }

    freq(ng.link,ng.pinfo,ng.freq);

    return 1;
}

void ngramtable::resetngramtable() {
    //clean up all memory and restart from an empty table

    freetree(); //clean memory pool
    memset(tree,0,inodesize(6)); //reset tree
    //1-gram table initial flags

    if (maxlev>1) mtflags(tree,INODE | FREQ4);
    else if (maxlev==1) mtflags(tree,LNODE | FREQ4);

    word(tree,0);      //dummy word
    msucc(tree,0);     // number of different n-grams
    mtable(tree,NULL); // table of n-gram

    for (int i=1; i<=maxlev; i++)
      mentr[i]=memory[i]=occupancy[i]=0;

}

int ngramtable::putmem(char* ptr,int value,int offs,int size) {
    MY_ASSERT(ptr!=NULL);
    for (int i=0; i<size; i++)
      ptr[offs+i]=(value >> (8 * i)) & 0xff;
    return value;
}

int ngramtable::getmem(char* ptr,int* value,int offs,int size) {
    MY_ASSERT(ptr!=NULL);
    *value=ptr[offs] & 0xff;
    for (int i=1; i<size; i++)
      *value= *value | ( ( ptr[offs+i] & 0xff ) << (8 *i));
    return *value;
}

long ngramtable::putmem(char* ptr,long long value,int offs,int size) {
    MY_ASSERT(ptr!=NULL);
    for (int i=0; i<size; i++)
      ptr[offs+i]=(value >> (8 * i)) & 0xffLL;
    return value;
}

long ngramtable::getmem(char* ptr,long long* value,int offs,int size) {
    MY_ASSERT(ptr!=NULL);
    *value=ptr[offs] & 0xff;
    for (int i=1; i<size; i++)
      *value= *value | ( ( ptr[offs+i] & 0xffLL ) << (8 *i));
    return *value;
}

void ngramtable::tb2ngcpy(int* wordp,char* tablep,int n) {
    for (int i=0; i<n; i++)
      getmem(tablep,&wordp[i],i*CODESIZE,CODESIZE);
}

void ngramtable::ng2tbcpy(char* tablep,int* wordp,int n) {
    for (int i=0; i<n; i++)
      putmem(tablep,wordp[i],i*CODESIZE,CODESIZE);
}

int ngramtable::ngtbcmp(int* wordp,char* tablep,int n) {
    int word;
    for (int i=0; i<n; i++) {
      getmem(tablep,&word,i*CODESIZE,CODESIZE);
      if (wordp[i]!=word) return 1;
    }
    return 0;
}

int ngramtable::codecmp(char * a,char *b) {
    int i,result;
    for (i=(CODESIZE-1); i>=0; i--) {
      result=(unsigned char)a[i]-(unsigned char)b[i];
      if(result) return result;
    }
    return 0;
};

long long ngramtable::freq(node nd,NODETYPE ndt,long long value) {
    int offs=(ndt & LNODE)?L_FREQ_OFFS:I_FREQ_OFFS;

    if (ndt & FREQ1)
      putmem(nd,value,offs,1);
    else if (ndt & FREQ2)
      putmem(nd,value,offs,2);
    else if (ndt & FREQ3)
      putmem(nd,value,offs,3);
    else if (ndt & FREQ4)
      putmem(nd,value,offs,4);
    else
      putmem(nd,value,offs,6);
    return value;
}

long long ngramtable::freq(node nd,NODETYPE ndt) {
    int offs=(ndt & LNODE)?L_FREQ_OFFS:I_FREQ_OFFS;
    long long value;

    if (ndt & FREQ1)
      getmem(nd,&value,offs,1);
    else if (ndt & FREQ2)
      getmem(nd,&value,offs,2);
    else if (ndt & FREQ3)
      getmem(nd,&value,offs,3);
    else if (ndt & FREQ4)
      getmem(nd,&value,offs,4);
    else
      getmem(nd,&value,offs,6);

    return value;
}


long long ngramtable::setfreq(node nd,NODETYPE ndt,long long value,int index) {
    int offs=(ndt & LNODE)?L_FREQ_OFFS:I_FREQ_OFFS;

    if (ndt & FREQ1)
      putmem(nd,value,offs+index * 1,1);
    else if (ndt & FREQ2)
      putmem(nd,value,offs+index * 2,2);
    else if (ndt & FREQ3)
      putmem(nd,value,offs+index * 3,3);
    else if (ndt & FREQ4)
      putmem(nd,value,offs+index * 4,4);
    else
      putmem(nd,value,offs+index * 6,6);

    return value;
}

long long ngramtable::getfreq(node nd,NODETYPE ndt,int index) {
    int offs=(ndt & LNODE)?L_FREQ_OFFS:I_FREQ_OFFS;

    long long value;

    if (ndt & FREQ1)
      getmem(nd,&value,offs+ index * 1,1);
    else if (ndt & FREQ2)
      getmem(nd,&value,offs+ index * 2,2);
    else if (ndt & FREQ3)
      getmem(nd,&value,offs+ index * 3,3);
    else if (ndt & FREQ4)
      getmem(nd,&value,offs+ index * 4,4);
    else
      getmem(nd,&value,offs+ index * 6,6);

    return value;
}

table ngramtable::mtable(node nd) {
    char v[PTRSIZE];;
    for (int i=0; i<PTRSIZE; i++)
      v[i]=nd[MTAB_OFFS+i];

    return *(table *)v;
}

table ngramtable::mtable(node nd,table value) {
    char *v=(char *)&value;
    for (int i=0; i<PTRSIZE; i++)
      nd[MTAB_OFFS+i]=v[i];
    return value;
}

int ngramtable::mtablesz(node nd) {
    if (mtflags(nd) & LNODE) {
      if (mtflags(nd) & FREQ1)
        return lnodesize(1);
      else if (mtflags(nd) & FREQ2)
        return lnodesize(2);
      else if (mtflags(nd) & FREQ3)
        return lnodesize(3);
      else if (mtflags(nd) & FREQ4)
        return lnodesize(4);
      else
        return lnodesize(6);
    } else if (mtflags(nd) & INODE) {
      if (mtflags(nd) & FREQ1)
        return inodesize(1);
      else if (mtflags(nd) & FREQ2)
        return inodesize(2);
      else if (mtflags(nd) & FREQ3)
        return inodesize(3);
      else if (mtflags(nd) & FREQ4)
        return inodesize(4);
      else
        return inodesize(6);
    } else {
			exit_error(IRSTLM_ERROR_DATA,"ngramtable::mtablesz node has wrong flags");
    }
	
	return lnodesize(1); //this instruction is never reached
}


/*
 main(int argc, char** argv){
         dictionary d(argv[1]);

         ngram ng(&d);

         cerr << "caricato dizionario da " << argv[1] << "\n";

         ngramtable t(&d,argv[2],1);

         t.stat(1);
         t.savetxt(argv[3]);

 }
*/





