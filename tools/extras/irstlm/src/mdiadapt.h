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

// Adapted LM classes: extension of interp classes

#ifndef MF_MDIADAPTLM_H
#define MF_MDIADAPTLM_H

#include "ngramcache.h"
#include "normcache.h"
#include "interplm.h"

#define DONT_PRINT 1000000

namespace irstlm {
class mdiadaptlm:public interplm
{

  int adaptlev;
  interplm* forelm;
  double zeta0;
  double oovscaling;
  bool m_save_per_level;
 
  static bool mdiadaptlm_cache_enable;

protected:
  normcache *cache;

//to improve access speed
  NGRAMCACHE_t** probcache;
  NGRAMCACHE_t** backoffcache;
  int max_caching_level;
	
  int saveARPA_per_word(char *filename,int backoff=0,char* subdictfile=NULL);
  int saveARPA_per_level(char *filename,int backoff=0,char* subdictfile=NULL);
  int saveBIN_per_word(char *filename,int backoff=0,char* subdictfile=NULL,int mmap=0);
  int saveBIN_per_level(char *filename,int backoff=0,char* subdictfile=NULL,int mmap=0);
public:

  mdiadaptlm(char* ngtfile,int depth=0,TABLETYPE tt=FULL);

  inline normcache* get_zetacache() {
    return cache;
  }
  inline NGRAMCACHE_t* get_probcache(int level);
  inline NGRAMCACHE_t* get_backoffcache(int level);

  void create_caches(int mcl);
  void init_caches();
  void init_caches(int level);
  void delete_caches();
  void delete_caches(int level);

  void check_cache_levels();
  void check_cache_levels(int level);
  void reset_caches();
  void reset_caches(int level);

  void caches_stat();

  double gis_step;

  double zeta(ngram ng,int size);

  int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);

  int bodiscount(ngram ng,int size,double& fstar,double& lambda,double& bo);
	
  virtual int compute_backoff();
  virtual int compute_backoff_per_level();
  virtual int compute_backoff_per_word();

  double backunig(ngram ng);

  double foreunig(ngram ng);

  void adapt(char* ngtfile,int alev=1,double gis_step=0.4);

  int scalefact(char* ngtfile);

  int savescalefactor(char* filename);

  double scalefact(ngram ng);

  double prob(ngram ng,int size);
  double prob(ngram ng,int size,double& fstar,double& lambda, double& bo);

  double prob2(ngram ng,int size,double & fstar);

  double txclprob(ngram ng,int size);

  int saveASR(char *filename,int backoff,char* subdictfile=NULL);
  int saveMT(char *filename,int backoff,char* subdictfile=NULL,int resolution=10000000,double decay=0.999900);
	
  int saveARPA(char *filename,int backoff=0,char* subdictfile=NULL){
		if (m_save_per_level){
			cerr << " per level ...";
			return saveARPA_per_level(filename, backoff, subdictfile);
		}else{
			cerr << " per word ...";
			return saveARPA_per_word(filename, backoff, subdictfile);
		}
	}
  int saveBIN(char *filename,int backoff=0,char* subdictfile=NULL,int mmap=0){
		if (m_save_per_level){
			cerr << " per level ...";
			return saveBIN_per_level(filename, backoff, subdictfile, mmap);
		}else{
			cerr << " per word ...";
			return saveBIN_per_word(filename, backoff, subdictfile, mmap);
		}
	}
	
  inline void save_per_level(bool value){ m_save_per_level=value; }
  inline bool save_per_level() const { return m_save_per_level; }
	
  int netsize();

  ~mdiadaptlm();

  double myround(double x) {
    long int value = (long int) x;
    return (x-value)>0.500?value+1.0:(double)value;
  }

  inline static bool is_train_cache_enabled() {
    VERBOSE(3,"inline static bool is_train_cache_enabled() " << mdiadaptlm_cache_enable << std::endl);
    return mdiadaptlm_cache_enable;
  }

};

}//namespace irstlm
#endif






