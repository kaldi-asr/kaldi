// $Id: ngramtable.h 34 2010-06-03 09:19:34Z nicolabertoldi $

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

#ifndef MF_NGRAMTABLE_H
#define MF_NGRAMTABLE_H
	
#include "n_gram.h"

//Backoff symbol
#ifndef BACKOFF_
#define BACKOFF_ "_backoff_"
#endif

//Dummy symbol
#ifndef DUMMY_
#define DUMMY_ "_dummy_"
#endif

// internal data structure

#ifdef MYCODESIZE
#define DEFCODESIZE  MYCODESIZE
#else
#define DEFCODESIZE  (int)2
#endif

#define SHORTSIZE (int)2
#define PTRSIZE   (int)sizeof(char *)
#define INTSIZE   (int)4
#define CHARSIZE  (int)1


//Node  flags
#define FREQ1  (unsigned char)   1
#define FREQ2  (unsigned char)   2
#define FREQ4  (unsigned char)   4
#define INODE  (unsigned char)   8
#define LNODE  (unsigned char)  16
#define SNODE  (unsigned char)  32
#define FREQ6 (unsigned char)   64
#define FREQ3  (unsigned char) 128

typedef char* node;  //inodes, lnodes, snodes
typedef char* table; //inode table, lnode table, singleton table

typedef unsigned char NODETYPE;


typedef enum {FIND,    //!< search: find an entry
              ENTER,   //!< search: enter an entry
              DELETE,  //!< search: find and remove entry
              INIT,    //!< scan: start scan
              CONT     //!< scan: continue scan
             } ACTION;


typedef enum {COUNT,       //!< table: only counters
              LEAFPROB,    //!< table: only probs on leafs
              FLEAFPROB,    //!< table: only probs on leafs and FROZEN
              LEAFPROB2,   //!< table: only probs on leafs
              LEAFPROB3,   //!< table: only probs on leafs
              LEAFPROB4,   //!< table: only probs on leafs
              LEAFCODE,    //!< table: only codes on leafs
              SIMPLE_I,    //!< table: simple interpolated LM
              SIMPLE_B,    //!< table: simple backoff LM
              SHIFTBETA_I, //!< table: interpolated shiftbeta
              SHIFTBETA_B, //!< table: backoff shiftbeta
              IMPROVEDSHIFTBETA_I,//!< table: interp improved shiftbeta
              IMPROVEDSHIFTBETA_B,//!< table: interp improved shiftbeta
              KNESERNEY_I,//!< table: interp kneser-ney
              KNESERNEY_B,//!< table: backoff kneser-ney
              IMPROVEDKNESERNEY_I,//!< table: interp improved kneser-ney
              IMPROVEDKNESERNEY_B,//!< table: backoff improved kneser-ney
              FULL,        //!< table: full fledged table

             } TABLETYPE;

class tabletype
{

  TABLETYPE ttype;

public:

  int CODESIZE;                //sizeof word codes
  long long code_range[7]; //max code for each size

  //Offsets of internal node fields
  int WORD_OFFS;   //word code position
  int MSUCC_OFFS;  //number of successors
  int MTAB_OFFS;   //pointer to successors
  int FLAGS_OFFS;  //flag table
  int SUCC1_OFFS;  //number of successors with freq=1
  int SUCC2_OFFS;  //number of successors with freq=2
  int BOFF_OFFS;   //back-off probability
  int I_FREQ_OFFS; //frequency offset
  int I_FREQ_NUM;  //number of internal frequencies
  int L_FREQ_NUM;  //number of leaf frequencies
  int L_FREQ_SIZE; //minimum size for leaf frequencies

  //Offsets of leaf node fields
  int L_FREQ_OFFS; //frequency offset

  tabletype(TABLETYPE tt,int codesize=DEFCODESIZE);

  inline TABLETYPE tbtype() const {
    return ttype;
  }
  inline int inodesize(int s) const {
    return I_FREQ_OFFS + I_FREQ_NUM * s;
  }

  inline int lnodesize(int s) const {
    return L_FREQ_OFFS + L_FREQ_NUM * s;
  }

};


class ngramtable:tabletype
{
    
    node            tree; // ngram table root
	int           maxlev; // max storable n-gram
    NODETYPE   treeflags;
    char       info[100]; //information put in the header
    int       resolution; //max resolution for probabilities
    double         decay; //decay constant
    
    storage*         mem; //memory storage class
    
    long long*    memory; // memory load per level
    long long* occupancy; // memory occupied per level
    long long*     mentr; // multiple entries per level
    long long       card; //entries at maxlev
    
    int idx[MAX_NGRAM+1];
    
    int  oov_code,oov_size,du_code, bo_code; //used by prob
    
    int             backoff_state; //used by prob;
    
public:
    
    int         corrcounts; //corrected counters flag
    
    dictionary     *dict; // dictionary
    
    // filtering dictionary:
    // if the first word of the ngram does not belong to filterdict
    // do not insert the ngram
    dictionary     *filterdict;
    
    ngramtable(char* filename,int maxl,char* is,
               dictionary* extdict,
               char* filterdictfile,
               int googletable=0,
               int dstco=0,char* hmask=NULL,int inplen=0,
               TABLETYPE tt=FULL,int codesize=DEFCODESIZE);
    
    inline char* ngtype(char *str=NULL) {
        if (str!=NULL) strcpy(info,str);
        return info;
    }
    
    virtual ~ngramtable();
    
    inline void freetree() {
        freetree(tree);
    };
    
    void freetree(node nd);
    
    void resetngramtable();
    
    void stat(int level=4);
    
    inline long long totfreq(long long v=-1) {
        return (v==-1?freq(tree,INODE):freq(tree,INODE,v));
    }
    
    inline long long btotfreq(long long v=-1) {
        return (v==-1?getfreq(tree,treeflags,1):setfreq(tree,treeflags,v,1));
    }
    
    inline long long entries(int lev) const {
        return mentr[lev];
    }
    
    inline int maxlevel() const {
        return maxlev;
    }
    
    //  void savetxt(char *filename,int sz=0);
    void savetxt(char *filename,int sz=0,bool googleformat=false,bool hashvalue=false,int startfrom=0);
    void loadtxt(char *filename,int googletable=0);
    
    void savebin(char *filename,int sz=0);
    void savebin(mfstream& out);
    void savebin(mfstream& out,node nd,NODETYPE ndt,int lev,int mlev);
    
    void loadbin(const char *filename);
    void loadbin(mfstream& inp);
    void loadbin(mfstream& inp,node nd,NODETYPE ndt,int lev);
    
    void loadbinold(char *filename);
    void loadbinold(mfstream& inp,node nd,NODETYPE ndt,int lev);
    
    void generate(char *filename,dictionary *extdict=NULL);
    void generate_dstco(char *filename,int dstco);
    void generate_hmask(char *filename,char* hmask,int inplen=0);
    
    void augment(ngramtable* ngt);
    
    inline int scan(ngram& ng,ACTION action=CONT,int maxlev=-1) {
        return scan(tree,INODE,0,ng,action,maxlev);
    }
    
    inline int succscan(ngram& h,ngram& ng,ACTION action,int lev) {
        //return scan(h.link,h.info,h.lev,ng,action,lev);
        return scan(h.link,h.info,lev-1,ng,action,lev);
    }
    
    double prob(ngram ng);
    
    int scan(node nd,NODETYPE ndt,int lev,ngram& ng,ACTION action=CONT,int maxl=-1);
    
    void show();
    
    void *search(table *tb,NODETYPE ndt,int lev,int n,int sz,int *w,
                 ACTION action,char **found=(char **)NULL);
    
    int mybsearch(char *ar, int n, int size, unsigned char *key, int *idx);
    
    int put(ngram& ng);
    int put(ngram& ng,node nd,NODETYPE ndt,int lev);
    
    inline int get(ngram& ng) {
        return get(ng,maxlev,maxlev);
    }
    virtual int get(ngram& ng,int n,int lev);
    
    int comptbsize(int n);
    table *grow(table *tb,NODETYPE ndt,int lev,int n,int sz,NODETYPE oldndt=0);
    
    bool check_dictsize_bound();
    
    int putmem(char* ptr,int value,int offs,int size);
    int getmem(char* ptr,int* value,int offs,int size);
    long putmem(char* ptr,long long value,int offs,int size);
    long getmem(char* ptr,long long* value,int offs,int size);
    
    inline void tb2ngcpy(int* wordp,char* tablep,int n=1);
    inline void ng2tbcpy(char* tablep,int* wordp,int n=1);
    inline int ngtbcmp(int* wordp,char* tablep,int n=1);
    
    inline int word(node nd,int value) {
        putmem(nd,value,WORD_OFFS,CODESIZE);
        return value;
    }
    
    inline int word(node nd) {
        int v;
        getmem(nd,&v,WORD_OFFS,CODESIZE);
        return v;
    }
    
    inline unsigned char mtflags(node nd,unsigned char value) {
        return *(unsigned char *)(nd+FLAGS_OFFS)=value;
    }
    
    inline unsigned char mtflags(node nd) const {
        return *(unsigned char *)(nd+FLAGS_OFFS);
    }
    
    int codecmp(char * a,char *b);
    
    inline int codediff(node a,node b) {
        return word(a)-word(b);
    };
    
    
    int update(ngram ng);
    
    long long freq(node nd,NODETYPE ndt,long long value);
    long long freq(node nd,NODETYPE ndt);
    
    long long setfreq(node nd,NODETYPE ndt,long long value,int index=0);
    long long getfreq(node nd,NODETYPE ndt,int index=0);
    
    double boff(node nd) {
        int value=0;
        getmem(nd,&value,BOFF_OFFS,INTSIZE);
        
        return double (value/(double)1000000000.0);
    }
    
    
    double myround(double x) {
        long int i=(long int)(x);
        return (x-i)>0.500?i+1.0:(double)i;
    }
    
    int boff(node nd,double value) {
        int v=(int)myround(value * 1000000000.0);
        putmem(nd,v,BOFF_OFFS,INTSIZE);
        
        return 1;
    }
    
    int succ2(node nd,int value) {
        putmem(nd,value,SUCC2_OFFS,CODESIZE);
        return value;
    }
    
    int succ2(node nd) {
        int value=0;
        getmem(nd,&value,SUCC2_OFFS,CODESIZE);
        return value;
    }
    
    int succ1(node nd,int value) {
        putmem(nd,value,SUCC1_OFFS,CODESIZE);
        return value;
    }
    
    int succ1(node nd) {
        int value=0;
        getmem(nd,&value,SUCC1_OFFS,CODESIZE);
        return value;
    }
    
    int msucc(node nd,int value) {
        putmem(nd,value,MSUCC_OFFS,CODESIZE);
        return value;
    }
    
    int msucc(node nd) {
        int value;
        getmem(nd,&value,MSUCC_OFFS,CODESIZE);
        return value;
    }
    
    table mtable(node nd);
    table mtable(node nd,table value);
    int mtablesz(node nd);
    
    inline int bo_state() {
        return backoff_state;
    }
    inline int bo_state(int value) {
        return backoff_state=value;
    }
};

#endif




