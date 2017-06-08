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
// Basic Interpolated LM class

#ifndef MF_INTERPLM_H
#define MF_INTERPLM_H
	
#define SHIFT_BETA   1
#define SHIFT_ONE    2
#define SHIFT_ZERO   3
#define LINEAR_STB   4
#define LINEAR_WB    5
#define LINEAR_GT    6
#define MIXTURE      7
#define MOD_SHIFT_BETA         8
#define IMPROVED_SHIFT_BETA    9
#define KNESER_NEY            10
#define IMPROVED_KNESER_NEY   11

class interplm:public ngramtable
{

  int lms;

  double epsilon; //Bayes smoothing

  int unismooth; //0 Bayes, 1 Witten Bell

  int prune_singletons;
  int prune_top_singletons;
  int* prune_freq_threshold;

public:

  int backoff; //0 interpolation, 1 Back-off

  interplm(char* ngtfile,int depth=0,TABLETYPE tt=FULL);
  virtual ~interplm();

  int prunesingletons(int flag=-1) {
    return (flag==-1?prune_singletons:prune_singletons=flag);
  }

  int prunetopsingletons(int flag=-1) {
    return (flag==-1?prune_top_singletons:prune_top_singletons=flag);
  }

  inline bool prune_ngram(int lev, int freq)
  {
    return (freq > prune_freq_threshold[lev])?false:true;
  }

  void init_prune_ngram(int sz);
  void delete_prune_ngram();
  void set_prune_ngram(int lev, int val);
  void set_prune_ngram(char* values);
  void print_prune_ngram();
	
  void gencorrcounts();

  void gensuccstat();

  virtual int dub() {
    return dict->dub();
  }

  virtual int dub(int value) {
    return dict->dub(value);
  }

  int setusmooth(int v=0) {
    return unismooth=v;
  }

  double setepsilon(double v=1.0) {
    return epsilon=v;
  }

  ngramtable *unitbl;

  void trainunigr();
	
  double unigrWB(ngram ng);
  virtual double unigr(ngram ng){ return unigrWB(ng); };

  double zerofreq(int lev);

  inline int lmsize() const {
    return lms;
  }

  inline int obswrd() const {
    return dict->size();
  }

  virtual int train() {
    return 0;
  }

  virtual void adapt(char* /* unused parameter: ngtfile */, int /* unused parameter:  l */, double /* unused parameter:  w */) {}

  virtual double prob(ngram /* unused parameter: ng */,int /* unused parameter: size */) {
    return 0.0;
  }

  virtual double boprob(ngram /* unused parameter: ng */,int /* unused parameter: size */) {
    return 0.0;
  }

  void test_ngt(ngramtable& ngt,int sz=0,bool backoff=false,bool checkpr=false);

  void test_txt(char *filename,int sz=0,bool backoff=false,bool checkpr=false,char* outpr=NULL);

  void test(char* filename,int sz,bool backoff=false,bool checkpr=false,char* outpr=NULL);

  virtual int discount(ngram /* unused parameter: ng */,int /* unused parameter: size */,double& /* unused parameter: fstar */ ,double& /* unused parameter: lambda */,int /* unused parameter: cv*/=0) {
    return 0;
  }

  virtual int savebin(char* /* unused parameter: filename */,int /* unused parameter: lmsize=2 */) {
    return 0;
  }

  virtual int netsize() {
    return 0;
  }

  void lmstat(int level) {
    stat(level);
  }


};

#endif






