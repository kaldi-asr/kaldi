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

#include <cmath>
#include <string>
#include "util.h"
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "mempool.h"
#include "ngramcache.h"
#include "ngramtable.h"
#include "normcache.h"
#include "interplm.h"
#include "mdiadapt.h"
#include "shiftlm.h"
#include "lmtable.h"

using namespace std;

namespace irstlm {

#ifdef MDIADAPTLM_CACHE_ENABLE
#if MDIADAPTLM_CACHE_ENABLE==0
#undef MDIADAPTLM_CACHE_ENABLE
#endif
#endif

#ifdef MDIADAPTLM_CACHE_ENABLE
  bool mdiadaptlm::mdiadaptlm_cache_enable=true;
#else
  bool mdiadaptlm::mdiadaptlm_cache_enable=false;
#endif

	//
	//Minimum discrimination adaptation for interplm
	//
	mdiadaptlm::mdiadaptlm(char* ngtfile,int depth,TABLETYPE tbtype):
  interplm(ngtfile,depth,tbtype)
	{
		adaptlev=0;
		forelm=NULL;
		cache=NULL;
		m_save_per_level=true;
	};
	
	mdiadaptlm::~mdiadaptlm()
	{
		if (cache) delete cache;
		delete_caches();
	};
	
	void mdiadaptlm::delete_caches(int level)
	{
		if (probcache[level]) delete probcache[level];
		if (backoffcache[level]) delete backoffcache[level];
	};
	
	void mdiadaptlm::delete_caches()
	{
#ifdef MDIADAPTLM_CACHE_ENABLE
		for (int i=0; i<=max_caching_level; i++) delete_caches(i);
		
		delete [] probcache;
		delete [] backoffcache;
#endif
	};
	
	void mdiadaptlm::caches_stat()
	{
#ifdef MDIADAPTLM_CACHE_ENABLE
		for (int i=1; i<=max_caching_level; i++) {
			if (probcache[i]) {
				cerr << "Statistics of probcache at level " << i << " (of " << maxlevel() << ") ";
				probcache[i]->stat();
			}
			if (backoffcache[i]) {
				cerr << "Statistics of backoffcache at level " << i << " (of " << maxlevel() << ") ";
				backoffcache[i]->stat();
			}
		}
#endif
	};
	
	
	void mdiadaptlm::create_caches(int mcl)
	{
		max_caching_level=(mcl>=0 && mcl<lmsize())?mcl:lmsize()-1;
		
		probcache = new NGRAMCACHE_t*[max_caching_level+1]; //index 0 will never be used, index=max_caching_level is not used
		backoffcache = new NGRAMCACHE_t*[max_caching_level+1]; //index 0 will never be used, index=max_caching_level is not used
		for (int i=0; i<=max_caching_level; i++) {
			probcache[i]=NULL;
			backoffcache[i]=NULL;
		}
		
		init_caches();
	}
	
	
	void mdiadaptlm::init_caches(int level)
	{
		MY_ASSERT(probcache[level]==NULL);
		MY_ASSERT(backoffcache[level]==NULL);
		probcache[level]=new NGRAMCACHE_t(level,sizeof(double),400000);
		backoffcache[level]=new NGRAMCACHE_t(level,sizeof(double),400000);
	};
	
	void mdiadaptlm::init_caches()
	{
#ifdef MDIADAPTLM_CACHE_ENABLE
		for (int i=1; i<=max_caching_level; i++)		init_caches(i);
#endif
	};
	
	void mdiadaptlm::check_cache_levels(int level)
	{
		if (probcache[level] && probcache[level]->isfull()) probcache[level]->reset(probcache[level]->cursize());
		if (backoffcache[level] && backoffcache[level]->isfull()) backoffcache[level]->reset(backoffcache[level]->cursize());
	};
	
	void mdiadaptlm::check_cache_levels()
	{
#ifdef MDIADAPTLM_CACHE_ENABLE
		for (int i=1; i<=max_caching_level; i++)		check_cache_levels(i);
#endif
	};
	
	void mdiadaptlm::reset_caches(int level)
	{
		if (probcache[level]) probcache[level]->reset(MAX(probcache[level]->cursize(),probcache[level]->maxsize()));
		if (backoffcache[level]) backoffcache[level]->reset(MAX(backoffcache[level]->cursize(),backoffcache[level]->maxsize()));
	};
	
	void mdiadaptlm::reset_caches()
	{
#ifdef MDIADAPTLM_CACHE_ENABLE
		for (int i=1; i<=max_caching_level; i++)		reset_caches(i);
#endif
	};
	
	
	inline NGRAMCACHE_t* mdiadaptlm::get_probcache(int level)
	{
		return probcache[level];
	}
	
	inline NGRAMCACHE_t* mdiadaptlm::get_backoffcache(int level)
	{
		return backoffcache[level];
	}
	
	int mdiadaptlm::scalefact(char *ngtfile)
	{
		if (forelm!=NULL) delete forelm;
		if (cache!=NULL) delete cache;
		cache=new normcache(dict);
		
		forelm=new shiftbeta(ngtfile,1);
		forelm->train();
		
		//compute oov scalefact term
		ngram fng(forelm->dict,1);
		ngram ng(dict,1);
		int* w=fng.wordp(1);
		
		oovscaling=1.0;
		for ((*w)=0; (*w)<forelm->dict->size(); (*w)++)
			if ((*w) != forelm->dict->oovcode()) {
				ng.trans(fng);
				if (*ng.wordp(1)==dict->oovcode()) {
					cerr << "adaptation file contains new words: use -ao=yes option\n";
					exit(1);
				}
				//forbidden situation
				oovscaling-=backunig(ng);
			}
		*w=forelm->dict->oovcode();
		oovscaling=foreunig(fng)/oovscaling;
		
		return 1;
	};
	
	int mdiadaptlm::savescalefactor(char* filename)
	{
		
		ngram ng(dict,1);
		int* w=ng.wordp(1);
		
		mfstream out(filename,ios::out);
		
		out << "\n\\data\\" << "\nngram 1=" << dict->size() << "\n\n1grams:\n";
		
		for ((*w)=0; (*w)<dict->size(); (*w)++) {
			double ratio=scalefact(ng);
			out << (float)  (ratio?log10(ratio):-99);
			if (*w==dict->oovcode())
				out << "\t" << "<unk>\n";
			else
				out << "\t" << (char *)dict->decode(*w) << "\n";
			
		}
		out << "\\end\\\n";
		
		return 1;
	}
	
	double mdiadaptlm::scalefact(ngram ng)
	{
		ngram fng(forelm->dict,1);
		fng.trans(ng);
		if (*fng.wordp(1)==forelm->dict->oovcode())
			return pow(oovscaling,gis_step);
		else {
			double prback=backunig(ng);
			double prfore=foreunig(ng);
			return pow(prfore/prback,gis_step);
		}
	}
	
	
	double mdiadaptlm::foreunig(ngram ng)
	{
		
		double fstar,lambda;
		
		forelm->discount(ng,1,fstar,lambda);
		
		return fstar;
	}
	
	double mdiadaptlm::backunig(ngram ng)
	{
		
		double fstar,lambda;
		
		discount(ng,1,fstar,lambda,0);
		
		return fstar;
	};
	
	
	
	void mdiadaptlm::adapt(char* ngtfile,int alev,double step)
	{
		
		if (alev > lmsize() || alev<=0) {
			cerr << "setting adaptation level to " << lmsize() << "\n";
			alev=lmsize();
		}
		adaptlev=alev;
		
		
		cerr << "adapt ....";
		gis_step=step;
		
		if (ngtfile==NULL) {
			cerr << "adaptation file is missing\n";
			exit(1);
		}
		
		//compute the scaling factor;
		
		scalefact(ngtfile);
		
		//compute 1-gram zeta
		ngram ng(dict,2);
		int* w=ng.wordp(1);
		
		cerr << "precomputing 1-gram normalization ...\n";
		zeta0=0;
		for ((*w)=0; (*w)<dict->size(); (*w)++)
			zeta0+=scalefact(ng) * backunig(ng);
		
		if (alev==1) return;
		
		cerr << "precomputing 2-gram normalization:\n";
		
		//precompute the bigram normalization
		w=ng.wordp(2);
		*ng.wordp(1)=0;
		
		for ((*w)=0; (*w)<dict->size(); (*w)++) {
			zeta(ng,2);
			if ((*w % 1000)==0) cerr << ".";
		}
		
		cerr << "done\n";
		
		return;
	};
	
	
	double mdiadaptlm::zeta(ngram ng,int size)
	{
		
		MY_ASSERT(size>=1);
		
		double z=0; // compute normalization term
		
		ng.size=size;
		
		if (size==1) return zeta0;
		else { //size>1
			
			//check in the 2gr and 3gr cache
			if (size <=3 && cache->get(ng,size,z)) return z;
			
			double fstar,lambda;
			ngram histo=ng;
			int succ=0;
			
			discount(ng,size,fstar,lambda,0);
			
			if ((lambda<1) && get(histo,size,size-1)) {
				;
				
				//scan all its successors
				succ=0;
				
				succscan(histo,ng,INIT,size);
				while(succscan(histo,ng,CONT,size)) {
					
					discount(ng,size,fstar,lambda,0);
					if (fstar>0) {
						z+=(scalefact(ng) * fstar);
						succ++;
						//cerr << ng << "zeta= " << z << "\n";
					}
				}
			}
			
			z+=lambda*zeta(ng,size-1);
			
			if (size<=3 && succ>1) cache->put(ng,size,z);
			
			return z;
		}
		
	}
	
	
	int mdiadaptlm::discount(ngram ng_,int size,double& fstar,double& lambda,int /* unused parameter: cv */)
	{
		VERBOSE(3,"mdiadaptlm::discount(ngram ng_,int size,double& fstar,double& lambda,int)) ng_:|" << ng_ << "| size:" << size << std::endl);
		
		ngram ng(dict);
		ng.trans(ng_);
		
		double __fstar, __lambda;
		bool lambda_cached=0;
		int size_lambda=size-1;
		
		ngram histo=ng;
		histo.shift();
		
		if (size_lambda>0 && histo.size>=size_lambda) {
#ifdef MDIADAPTLM_CACHE_ENABLE
			if (size_lambda<=max_caching_level) {
				//backoffcache hit
				if (backoffcache[size_lambda]  && backoffcache[size_lambda]->get(histo.wordp(size_lambda),__lambda))
					lambda_cached=1;
			}
#endif
		}
		
		discount(ng,size,__fstar,__lambda,0);
		
		if ((size>0) && (size<=adaptlev) && (__lambda<1)) {
			
			if (size>1) {
				double numlambda, numfstar, den;
				numfstar=scalefact(ng);
				den=zeta(ng,size);
				__fstar=__fstar * numfstar/den;
				if (!lambda_cached) {
					numlambda=zeta(ng,size-1);
					__lambda=__lambda * numlambda/den;
				}
			} else if (size==1) {
				double ratio;
				ratio=scalefact(ng)/zeta0;
				__fstar=__fstar * ratio;
				if (!lambda_cached) {
					__lambda=__lambda * ratio;
				}
			} else {
				//size==0 do nothing
			}
		}
		
#ifdef MDIADAPTLM_CACHE_ENABLE
		//backoffcache insert
		if (!lambda_cached && size_lambda>0 && size_lambda<=max_caching_level && histo.size>=size_lambda && backoffcache[size_lambda])
			backoffcache[size_lambda]->add(histo.wordp(size_lambda),__lambda);
#endif
		
		lambda=__lambda;
		fstar=__fstar;
		
		return 1;
	}
	
	int mdiadaptlm::compute_backoff()
	{
		VERBOSE(3,"mdiadaptlm::compute_backoff() ");
		if (m_save_per_level){
			VERBOSE(3," per level ...\n");
			return mdiadaptlm::compute_backoff_per_level();
		}else{
			VERBOSE(3," per word ...\n");
			return mdiadaptlm::compute_backoff_per_word();
		}	
	}
	
	int mdiadaptlm::compute_backoff_per_level()
	{
		VERBOSE(3,"mdiadaptlm::compute_backoff_per_level()\n");
		double fstar,lambda;
		
		this->backoff=1;

		for (int size=1; size<lmsize(); size++) {
			
			ngram hg(dict,size);
			
			scan(hg,INIT,size);
			
			while(scan(hg,CONT,size)) {
				ngram ng=hg;
				ng.pushc(0); //ng.size is now hg.size+1
				double pr=1.0;
				
				succscan(hg,ng,INIT,size+1);
				while(succscan(hg,ng,CONT,size+1)) {
					mdiadaptlm::discount(ng,ng.size,fstar,lambda);
					if (fstar>0){
						ng.size=ng.size-1;
						pr -= mdiadaptlm::prob(ng,size);
					}
				}
				
				MY_ASSERT(pr>=LOWER_SINGLE_PRECISION_OF_0 && pr<=UPPER_SINGLE_PRECISION_OF_1);
				
				boff(hg.link,pr);
			}
		}
		
		VERBOSE(3,"mdiadaptlm::compute_backoff_per_level() DONE\n");
		
		return 1;
	}
	
	
	int mdiadaptlm::compute_backoff_per_word()
	{	
		cerr << "Current implementation does not support the usage of backoff (-bo=yes) mixture models (-lm=mix) combined with the per-word saving (-saveperllevel=no)." << endl;
		cerr << "Please, either choose a per-level saving (-saveperllevel=yes) or do not use backoff (-bo=no) " << endl;
		
		exit(1);
	}	
	
	
	double mdiadaptlm::prob2(ngram ng,int size,double& fstar)
	{
		double lambda;
		
		mdiadaptlm::discount(ng,size,fstar,lambda);
		
		if (size>1)
			return fstar  + lambda * prob(ng,size-1);
		else
			return fstar;
	}
	
	
	//inline double mdiadaptlm::prob(ngram ng,int size){
	double mdiadaptlm::prob(ngram ng,int size)
	{
		double fstar,lambda,bo;
		return prob(ng,size,fstar,lambda,bo);
	}
	
	double mdiadaptlm::prob(ngram ng,int size,double& fstar,double& lambda, double& bo)
	{
		VERBOSE(3,"mdiadaptlm::prob(ngram ng,int size,double& fstar,double& lambda, double& bo) ng:|" << ng << "| size:" << size << std::endl);
		double pr;
		
#ifdef MDIADAPTLM_CACHE_ENABLE
		//probcache hit
		if (size<=max_caching_level && probcache[size] && ng.size>=size && probcache[size]->get(ng.wordp(size),pr))
			return pr;
#endif
		
		//probcache miss
		mdiadaptlm::bodiscount(ng,size,fstar,lambda,bo);
		VERBOSE(3,"mdiadaptlm::prob(ngram ng,int size,double& fstar,double& lambda, double& bo) after bodiscount @@@@@@@@@ ng:|" << ng << "| size:" << size << "| fstar:" << fstar << "| lambda:" << lambda << "| bo:" << bo << std::endl);
		if (fstar>UPPER_SINGLE_PRECISION_OF_1 || lambda>UPPER_SINGLE_PRECISION_OF_1) {
			cerr << "wrong probability: " << ng
			<< " , size " << size
			<< " , fstar " << fstar
			<< " , lambda " << lambda << "\n";
			fstar=(fstar>UPPER_SINGLE_PRECISION_OF_1?UPPER_SINGLE_PRECISION_OF_1:fstar);
			lambda=(lambda>UPPER_SINGLE_PRECISION_OF_1?UPPER_SINGLE_PRECISION_OF_1:lambda);
			//exit(1);
		}
		if (backoff) {
			if (size>1) {
				if (fstar>0){
					pr=fstar;
				}else {
					if (lambda<1){
						pr = lambda/bo * prob(ng,size-1);
					}else {
						MY_ASSERT(lambda<UPPER_SINGLE_PRECISION_OF_1);
						pr = prob(ng,size-1);
					}
				}
			} else
				pr = fstar;
		}
		
		else { //interpolation			
			if (size>1)
				pr = fstar  + lambda * prob(ng,size-1);
			else
				pr = fstar;
		}
		
#ifdef MDIADAPTLM_CACHE_ENABLE
		//probcache insert
		if (size<=max_caching_level && probcache[size] && ng.size>=size)
			probcache[size]->add(ng.wordp(size),pr);
#endif
		VERBOSE(3,"mdiadaptlm::prob(ngram ng,int size,double& fstar,double& lambda, double& bo) returning ng:|" << ng << "| pr:" << pr << std::endl);
		return pr;
	}
	
	
	int mdiadaptlm::bodiscount(ngram ng_,int size,double& fstar,double& lambda,double& bo)
	{
		VERBOSE(3,"mdiadaptlm::bodiscount(ngram ng_,int size,double& fstar,double& lambda,double& bo) ng_:|" << ng_ << "| size:" << size << std::endl);
		ngram ng(dict);
		ng.trans(ng_);
		
		mdiadaptlm::discount(ng,size,fstar,lambda);
		
		bo=1.0;
		
		if (backoff) { //get back-off probability
			
			if (size>1 && lambda<1) {
				
				ngram hg=ng;
				
				//			cerr<< "hg:|" << hg << "| size:|" << size << "|" <<  endl;
				if (! get(hg,size,size-1)){
					cerr << "ERROR: int mdiadaptlm::bodiscount(ngram ng_,int size,double& fstar,double& lambda,double& bo)   -> get(hg,size,size-1) returns NULL\n";
				}
				MY_ASSERT(get(hg,size,size-1));
				
				bo=boff(hg.link);
				
				//			if (lambda > bo){
				//				cerr << " mdiadaptlm::bodiscount ERROR: " << " lambda:" << lambda << " bo:" << bo << "\n";
				//				exit(1);
				//			}
			}
		}
		
		return 1;
	}
	
	
	double mdiadaptlm::txclprob(ngram ng,int size)
	{
		
		double fstar,lambda;
		
		if (size>1) {
			mdiadaptlm::discount(ng,size,fstar,lambda);
			return fstar  + lambda * txclprob(ng,size-1);
		} else {
			double freq=1;
			if ((*ng.wordp(1)!=dict->oovcode()) && get(ng,1,1))
				freq+=ng.freq;
			
			double N=totfreq()+dict->dub()-dict->size();
			return freq/N;
		}
	}
	
	
	int mdiadaptlm::netsize()
	{
		double fstar,lambda;
		int size,totsize;
		ngram ng(dict);
		
		cerr << "Computing LM size:\n";
		
		totsize=dict->size() * 2;
		
		cout << "1-gram " << totsize << "\n";
		
		for (int i=2; i<=maxlevel(); i++) {
			
			size=0;
			
			scan(ng,INIT,i);
			
			while (scan(ng,CONT,i)) {
				
				mdiadaptlm::discount(ng,i,fstar,lambda);
				
				if (fstar>0) size++;
				
			}
			
			size+=size * (i<maxlevel());
			
			totsize+=size;
			
			cout << i << "-gram " << totsize << "\n";
			
		}
		
		return totsize;
	}
	
	
	
	/*
	 * trigram file format:
	 
	 --------------------------------
	 
   <idx> dictionary length
	 
   repeat [ dictionary length ] {
	 <newline terminated string> word;
   }
	 
   while [ first word != STOP ] {
	 <idx> first word
	 <idx> number of successors
	 repeat [ number of successors ] {
	 <idx>   second word
	 <float> prob
	 }
   }
	 
   <idx> STOP
	 
   while [ first word != STOP ] {
	 <idx> first word
	 <idx> number of successor sets
	 repeat [ number of successor sets ] {
	 <idx>   second word
	 <idx>   number of successors
	 repeat [ number of successors ] {
	 <idx>   third word
	 <float> prob
	 }
	 }
   }
	 
   <idx> STOP
	 
	 */
	
	
	//void writeNull(mfbstream& out,unsigned short nullCode,float nullProb){
	//  out.writex(&nullCode,sizeof(short));
	//  out.writex(&nullProb,sizeof(float));
	//}
	
	
	int swapbytes(char *p, int sz, int n)
	{
		char c,*l,*h;
		if((n<1) ||(sz<2)) return 0;
		for(; n--; p+=sz) for(h=(l=p)+sz; --h>l; l++) {
      c=*h;
      *h=*l;
      *l=c;
    }
		return 0;
	};
	
	void fwritex(char *p,int sz,int n,FILE* f)
	{
		
		if(*(short *)"AB"==0x4241) {
			swapbytes((char*)p, sz,n);
		}
		
		fwrite((char *)p,sz,n,f);
		
		if(*(short *)"AB"==0x4241) swapbytes((char*)p, sz,n);
		
	}
	
	void ifwrite(long loc,void *ptr,int size,int /* unused parameter: n */,FILE* f)
	{
		fflush(f);
		
		long pos=ftell(f);
		
		fseek(f,loc,SEEK_SET);
		
		fwritex((char *)ptr,size,1,f);
		
		fseek(f,pos,SEEK_SET);
		
		fflush(f);
	}
	
	void writeNull(unsigned short nullCode,float nullProb,FILE* f)
	{
		fwritex((char *)&nullCode,sizeof(short),1,f);
		fwritex((char *)&nullProb,sizeof(float),1,f);
	}
	
	
	int mdiadaptlm::saveASR(char *filename,int /* unused parameter: backoff */,char* subdictfile)
	{
		int totbg,tottr;
		
		dictionary* subdict;
		
		if (subdictfile)
			subdict=new dictionary(subdictfile);
		else
			subdict=dict; // default is subdict=dict
		
		typedef unsigned short code;
		
		system("date");
		
		if (lmsize()>3 || lmsize()<1) {
			cerr << "wrong lmsize\n";
			exit(1);
		}
		
		if (dict->size()>=0xffff && subdict->size()>=0xffff) {
			cerr << "save bin requires unsigned short codes\n";
			exit(1);
		}
		
		FILE* f=fopen(filename,"w");
		
		double fstar,lambda,boff;
		float pr;
		long succ1pos,succ2pos;
		code succ1,succ2,w,h1,h2;
		code stop=0xffff;
		
		//dictionary
		//#dictsize w1\n ..wN\n NULL\n
		
		code oovcode=subdict->oovcode();
		
		//includes at least NULL
		code subdictsz=subdict->size()+1;
		
		fwritex((char *)&subdictsz,sizeof(code),1,f);
		
		subdictsz--;
		for (w=0; w<subdictsz; w++)
			fprintf(f,"%s\n",(char *)subdict->decode(w));
		
		fprintf(f,"____\n");
		
		//unigram part
		//NULL #succ w1 pr1 ..wN prN
		
		h1=subdictsz;
		fwritex((char *)&h1,sizeof(code),1,f); //NULL
		
		succ1=0;
		succ1pos=ftell(f);
		fwritex((char *)&succ1,sizeof(code),1,f);
		
		ngram ng(dict);
		ngram sng(subdict);
		
		ng.size=sng.size=1;
		
		scan(ng,INIT,1);
		while(scan(ng,CONT,1)) {
			sng.trans(ng);
			if (sng.containsWord(subdict->OOV(),1))
				continue;
			
			pr=(float)mdiadaptlm::prob(ng,1);
			if (pr>1e-50) { //do not consider too low probabilities
				succ1++;
				w=*sng.wordp(1);
				fwritex((char *)&w,sizeof(code),1,f);
				fwritex((char *)&pr,sizeof(float),1,f);
			} else {
				cerr << "small prob word " << ng << "\n";
			}
		}
		
		// update number of unigrams
		ifwrite(succ1pos,&succ1,sizeof(code),1,f);
		
		cerr << "finito unigrammi " << succ1 << "\n";
		fflush(f);
		
		if (lmsize()==1) {
			fclose(f);
			return 1;
		}
		
		// rest of bigrams
		// w1 #succ w1 pr1 .. wN prN
		
		succ1=0;
		h1=subdictsz;
		totbg=subdictsz;
		
		ngram hg1(dict,1);
		
		ng.size=sng.size=2;
		
		scan(hg1,INIT,1);
		while(scan(hg1,CONT,1)) {
			
			if (hg1.containsWord(dict->OOV(),1)) continue;
			
			MY_ASSERT((*hg1.wordp(1))<dict->size());
			
			*ng.wordp(2)=*hg1.wordp(1);
			*ng.wordp(1)=0;
			
			sng.trans(ng);
			if (sng.containsWord(dict->OOV(),1)) continue;
			
			mdiadaptlm::bodiscount(ng,2,fstar,lambda,boff);
			
			if (lambda < 1.0) {
				
				h1=*sng.wordp(2);
				
				fwritex((char *)&h1,sizeof(code),1,f);
				
				succ1=0;
				succ1pos=ftell(f);
				fwritex((char *)&succ1,sizeof(code),1,f);
				
				ngram shg=hg1;
				get(shg,1,1);
				
				succscan(shg,ng,INIT,2);
				while(succscan(shg,ng,CONT,2)) {
					
					if (*ng.wordp(1)==oovcode) continue;
					
					sng.trans(ng);
					if (sng.containsWord(dict->OOV(),2)) continue;
					
					mdiadaptlm::discount(ng,2,fstar,lambda);
					
					if (fstar>1e-50) {
						w=*sng.wordp(1);
						fwritex((char *)&w,sizeof(code),1,f);
						pr=(float)mdiadaptlm::prob(ng,2);
						//cerr << ng << " prob=" << log(pr) << "\n";
						
						fwritex((char *)&pr,sizeof(float),1,f);
						succ1++;
					}
				}
				
				if (succ1) {
					lambda/=boff; //consider backoff
					writeNull(subdictsz,(float)lambda,f);
					succ1++;
					totbg+=succ1;
					ifwrite(succ1pos,&succ1,sizeof(code),1,f);
				} else {
					//go back one word
					fseek(f,succ1pos-(streampos)sizeof(code),SEEK_SET);
				}
			}
		}
		
		fwritex((char *)&stop,sizeof(code),1,f);
		
		cerr << " finito bigrammi! " << subdictsz << "\n";
		fflush(f);
		
		system("date");
		
		if (lmsize()<3) {
			fclose(f);
			return 1;
		}
		
		//TRIGRAM PART
		
		h1=subdictsz;
		h2=subdictsz;
		tottr=0;
		succ1=0;
		succ2=0;
		
		ngram hg2(dict,2);
		
		ng.size=sng.size=3;
		
		scan(hg1,INIT,1);
		while(scan(hg1,CONT,1)) {
			
			if (*hg1.wordp(1)==oovcode) continue;
			
			*ng.wordp(3)=*hg1.wordp(1);
			
			sng.trans(ng);
			if (sng.containsWord(dict->OOV(),1)) continue;
			
			MY_ASSERT((*sng.wordp(3))<subdictsz);
			
			h1=*sng.wordp(3);
			fwritex((char *)&h1,sizeof(code),1,f);
			
			succ1=0;
			succ1pos=ftell(f);
			fwritex((char *)&succ1,sizeof(code),1,f);
			
			ngram shg1=ng;
			get(shg1,3,1);
			
			succscan(shg1,hg2,INIT,2);
			while(succscan(shg1,hg2,CONT,2)) {
				
				if (*hg2.wordp(1)==oovcode) continue;
				
				*ng.wordp(2)=*hg2.wordp(1);
				*ng.wordp(1)=0;
				
				sng.trans(ng);
				if (sng.containsWord(dict->OOV(),2)) continue;
				
				mdiadaptlm::bodiscount(ng,3,fstar,lambda,boff);
				
				if (lambda < 1.0) {
					
					h2=*sng.wordp(2);
					fwritex((char *)&h2,sizeof(code),1,f);
					
					succ2=0;
					succ2pos=ftell(f);
					fwritex((char *)&succ2,sizeof(code),1,f);
					
					ngram shg2=ng;
					get(shg2,3,2);
					
					succscan(shg2,ng,INIT,3);
					while(succscan(shg2,ng,CONT,3)) {
						
						if (*ng.wordp(1)==oovcode) continue;
						
						sng.trans(ng);
						if (sng.containsWord(dict->OOV(),3)) continue;
						
						mdiadaptlm::discount(ng,3,fstar,lambda);
						//pr=(float)mdiadaptlm::prob2(ng,3,fstar);
						
						if (fstar>1e-50) {
							
							w=*sng.wordp(1);
							fwritex((char *)&w,sizeof(code),1,f);
							
							pr=(float)mdiadaptlm::prob(ng,3);
							
							//	    cerr << ng << " prob=" << log(pr) << "\n";
							fwritex((char *)&pr,sizeof(float),1,f);
							succ2++;
						}
					}
					
					if (succ2) {
						lambda/=boff;
						writeNull(subdictsz,(float)lambda,f);
						succ2++;
						tottr+=succ2;
						ifwrite(succ2pos,&succ2,sizeof(code),1,f);
						succ1++;
					} else {
						//go back one word
						fseek(f,succ2pos-(long)sizeof(code),SEEK_SET);
					}
				}
			}
			
			if (succ1)
				ifwrite(succ1pos,&succ1,sizeof(code),1,f);
			else
				fseek(f,succ1pos-(long)sizeof(code),SEEK_SET);
		}
		
		fwritex((char *)&stop,sizeof(code),1,f);
		
		fclose(f);
		
		cerr << "Tot bg: " << totbg << " tg: " << tottr<< "\n";
		
		system("date");
		
		return 1;
	};
	
	
	///// Save in IRST MT format
	
	int mdiadaptlm::saveMT(char *filename,int backoff,
												 char* subdictfile,int resolution,double decay)
	{
		
		double logalpha=log(decay);
		dictionary* subdict;
		
		if (subdictfile)
			subdict=new dictionary(subdictfile);
		else
			subdict=dict; // default is subdict=dict
		
		ngram ng(dict,lmsize());
		ngram sng(subdict,lmsize());
		
		cerr << "Adding unigram of OOV word if missing\n";
		
		for (int i=1; i<=maxlevel(); i++)
			*ng.wordp(i)=dict->oovcode();
		
		if (!get(ng,maxlevel(),1)) {
			cerr << "oov is missing in the ngram-table\n";
			// f(oov) = dictionary size (Witten Bell)
			ng.freq=dict->freq(dict->oovcode());
			cerr << "adding oov unigram " << ng << "\n";
			put(ng);
		}
		
		cerr << "Eventually adding OOV symbol to subdictionary\n";
		subdict->encode(OOV_);
		
		system("date");
		
		mfstream out(filename,ios::out);
		
		//add special symbols
		
		subdict->incflag(1);
		int bo_code=subdict->encode(BACKOFF_);
		int du_code=subdict->encode(DUMMY_);
		subdict->incflag(0);
		
		out << "nGrAm " << lmsize() << " " << 0
		<< " " << "LM_ "
		<< resolution << " "
		<< decay << "\n";
		
		subdict->save(out);
		
		//start writing ngrams
		
		cerr << "write unigram of oov probability\n";
		ng.size=1;
		*ng.wordp(1)=dict->oovcode();
		double pr=(float)mdiadaptlm::prob(ng,1);
		sng.trans(ng);
		sng.size=lmsize();
		for (int s=2; s<=lmsize(); s++) *sng.wordp(s)=du_code;
		sng.freq=(int)ceil(pr * (double)10000000)-1;
		out << sng << "\n";
		
		for (int i=1; i<=lmsize(); i++) {
			cerr << "LEVEL " << i << "\n";
			
			double fstar,lambda,bo,dummy;
			
			scan(ng,INIT,i);
			while(scan(ng,CONT,i)) {
				
				sng.trans(ng);
				
				sng.size=lmsize();
				for (int s=i+1; s<=lmsize(); s++)
					*sng.wordp(s)=du_code;
				
				if (i>=1 && sng.containsWord(subdict->OOV(),sng.size)) {
					cerr << "skipping : " << sng << "\n";
					continue;
				}
				
				// skip also eos symbols not at the final
				//if (i>=1 && sng.containsWord(dict->EoS(),sng.size))
				//continue;
				
				mdiadaptlm::discount(ng,i,fstar,dummy);
				
				//out << sng << " fstar " << fstar << " lambda " << lambda << "\n";
				//if (i==1 && sng.containsWord(subdict->OOV(),i)){
				//	cerr << sng << " fstar " << fstar << "\n";
				//}
				
				if (fstar>0) {
					
					double pr=(float)mdiadaptlm::prob(ng,i);
					
					if (i>1 && resolution<10000000) {
						sng.freq=resolution-(int)(log(pr)/logalpha)-1;
						sng.freq=(sng.freq>=0?sng.freq:0);
					} else
						sng.freq=(int)ceil(pr * (double)10000000)-1;
					
					out << sng << "\n";
					
				}
				
				if (i<lmsize()) { /// write backoff of higher order!!
					
					ngram ng2=ng;
					ng2.pushc(0); //extend by one
					mdiadaptlm::bodiscount(ng2,i+1,dummy,lambda,bo);
					MY_ASSERT(!backoff || (lambda ==1 || bo<1 ));
					
					sng.pushc(bo_code);
					sng.size=lmsize();
					
					if (lambda<1) {
						if (resolution<10000000) {
							sng.freq=resolution-(int)((log(lambda) - log(bo))/logalpha)-1;
							sng.freq=(sng.freq>=0?sng.freq:0);
						} else
							sng.freq=(int)ceil(lambda/bo * (double)10000000)-1;
						
						out << sng << "\n";
					}
				}
			}
			cerr << "LEVEL " << i << "DONE \n";
		}
		return 1;
	};
	
	///// Save in binary format forbackoff N-gram models
	
	int mdiadaptlm::saveBIN_per_word(char *filename,int backoff,char* subdictfile,int mmap)
	{
		VERBOSE(2,"mdiadaptlm::saveBIN_per_word START\n");	
		system("date");
		
		//subdict
		dictionary* subdict;
		
		//accumulated unigram oov prob
		//CHECK why this is not used (differently from what happens in the other save functions
		//	double oovprob=0;
		
		
		if (subdictfile) subdict=new dictionary(subdictfile);
		else   subdict=dict; // default is subdict=dict
		
		if (mmap) {
			VERBOSE(2,"savebin with memory map: " << filename << "\n");
		} else {
			VERBOSE(2,"savebin: " << filename << "\n");
		}
		
		int maxlev=lmsize();
		streampos pos[LMTMAXLEV+1];
		char buff[100];
		int isQuant=0; //savebin for quantized LM is not yet implemented
		
		//temporary filename to save the LM related to a single term
		char tmpfilename[BUFSIZ];
		
		//create temporary output file stream to store single levels for all terms
		MY_ASSERT(strlen(filename)<1000);
		char tfilename[LMTMAXLEV+1][1000];
		mfstream *tout[LMTMAXLEV+1];
		
		tout[0]=NULL;
		for (int i=1; i<=maxlev; i++) {
			sprintf(tfilename[i],"%s-%dgrams",filename,i);
			tout[i]=new mfstream(tfilename[i],ios::out);
		}
		
		// print header in the main output file
		mfstream out(filename,ios::out);
		out << "blmt " << maxlev;
		
		for (int i=1; i<=maxlev; i++) { //reserve space for ngram statistics (which are not yet avalable)
			pos[i]=out.tellp();
			sprintf(buff," %10d",0);
			out << buff;
		}
		out << "\n";
		subdict->save(out);
		out.flush();
		
		ngram ng(dict,maxlev);
		ngram oldng(dict,maxlev);
		ngram locng(dict,maxlev);
		
		ngram sng(subdict,maxlev);
		
		double fstar,lambda,bo,dummy,dummy2,pr,ibow;
		
		double oovprob=0.0; //accumulated unigram oov pro
		bool _OOV_unigram=false; //flag to check whether an OOV word is present or not
		
		//n-gram counters
		table_entry_pos_t num[LMTMAXLEV+1];
		for (int i=1; i<=maxlev; i++) num[i]=0;
		
		lmtable* lmt = new lmtable();
		
		lmt->configure(maxlev,isQuant);
		lmt->setDict(subdict);
		lmt->expand_level(1,dict->size(),filename,mmap);
		
		//main loop
		for (int w=0; w<dict->size(); w++) {
			int i=1;  //set the initial value of level
			sprintf(tmpfilename,"%s_tmp_%d",filename,w);
			
			if (!w % 10000) cerr << ".";
			
			//1-gram
			ngram ung(dict,1);
			*ung.wordp(1)=w;
			sng.trans(ung);
			
			
			// frequency pruning is not applied to unigrams
			
			/*
			 //exclude words not occurring in the subdictionary
			 if (sng.containsWord(subdict->OOV(),1) && !ung.containsWord(dict->OOV(),1))	continue;
			 */
			
			pr=mdiadaptlm::prob(ung,1);			

			if (sng.containsWord(subdict->OOV(),1) || ung.containsWord(dict->OOV(),1)) {
				_OOV_unigram=true;
				oovprob+=pr; //accumulate oov probability
				continue;
			}
			pr=(pr?log10(pr):-99);
			
			if (i<maxlev)  { //compute back-off
				ung.pushc(0); //extend by one
				VERBOSE(3,"mdiadaptlm::saveBIN_per_word(char *filename,int backoff,char* subdictfile ) computing backoff for ung:|" << ung << "| size:" << i+1 << std::endl);
				mdiadaptlm::bodiscount(ung,i+1,dummy,lambda,bo);
				VERBOSE(3,"mdiadaptlm::saveBIN_per_word(char *filename,int backoff,char* subdictfile ) getting backoff for ung:|" << ung << "| lambda:" << lambda << " bo:" << bo << std::endl);
				ung.shift();//shrink by one
				
				if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
					ibow = DONT_PRINT;
				}else{
					if (backoff){
						ibow=log10(lambda) - log10(bo);
					}else{
						MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
						if (lambda<LOWER_SINGLE_PRECISION_OF_1){
							ibow = log10(lambda);
						}else { //force to be 0.0
							ibow = 0.0;
						}
					}
				}
			}
			else {
				ibow=0.0; //default value for backoff weight at the lowest level
			}
			
			if (ibow != DONT_PRINT){
				lmt->addwithoffset(ung,(float)pr,(float)ibow);
			}
			num[i]++;
			
			//manage n-grams
			if (get(ung,1,1)) {
				
				//create n-gram with history w
				*ng.wordp(lmsize())=w;
				
				//create sentinel n-gram
				for (int i=1; i<=maxlev; i++) *oldng.wordp(i)=-1;
				
				//create the table for all levels but the level 1, with the maximum number of possible entries
				for (int i=2; i<=maxlev; i++) 
					lmt->expand_level(i,entries(i),tmpfilename,mmap);
				
				scan(ung.link,ung.info,1,ng,INIT,lmsize());
				while(scan(ung.link,ung.info,1,ng,CONT,lmsize())) {
					sng.trans(ng); // convert to subdictionary
//					locng=ng;      // make a local copy
					
					//find first internal level that changed
					int f=maxlev-1; //unigrams have been already covered
					while (f>1 && (*oldng.wordp(f)==*ng.wordp(f))){ f--; }
					
					for (int l=maxlev-(f-1); l<=lmsize(); l++){
						
						locng=ng;      // make a local copy
						if (l<lmsize()) locng.shift(maxlev-l); //reduce the ngram, which has size level
						
						// frequency pruning: skip n-grams with low frequency 
						if (prune_ngram(l,sng.freq)) continue;
						
						// skip n-grams containing OOV
						if (sng.containsWord(subdict->OOV(),l)) continue;
						
						// skip also n-grams containing eos symbols not at the final
						if (sng.containsWord(dict->EoS(),l-1)) continue;
						
						VERBOSE(3,"mdiadaptlm::saveBIN_per_word(char *filename,int backoff,char* subdictfile ) computing prob for locng:|" << locng << "| size:" << l << std::endl);
						pr=mdiadaptlm::prob(locng,l,fstar,dummy,dummy2);
						VERBOSE(3,"mdiadaptlm::saveBIN_per_word(char *filename,int backoff,char* subdictfile ) getting prob locng:|" << locng << "| size:" << l << " fstar:" << fstar << " pr:" << pr << std::endl);
						
						//PATCH by Nicola (16-04-2008)
						
						if (!(pr<=1.0 && pr > 1e-10)) {
							cerr << ng << " " << pr << "\n";
							MY_ASSERT(pr<=1.0);
							cerr << "prob modified to 1e-10\n";
							pr=1e-10;
						}
						
						if (l<lmsize()) {
							
							locng.pushc(0); //extend by one
							
							VERBOSE(3,"mdiadaptlm::saveBIN_per_word(char *filename,int backoff,char* subdictfile ) computing backoff for locng:|" << locng << "| size:" << l+1 << std::endl);
							mdiadaptlm::bodiscount(locng,l+1,dummy,lambda,bo);
							VERBOSE(3,"mdiadaptlm::saveBIN_per_word(char *filename,int backoff,char* subdictfile ) getting backoff locng:|" << locng << "| lambda:" << lambda << " bo:" << bo << std::endl);
							
							locng.shift();
							
							if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
								ibow = DONT_PRINT;
							}else{								
								if (backoff){
									ibow = (float) (log10(lambda) - log10(bo));
								}else{
									MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
									if (lambda<LOWER_SINGLE_PRECISION_OF_1){
										ibow = log10(lambda);
									}else{ //no output if log10(lambda)==0
										ibow = 0.0;
									}
								}					
							}
						} else { //i==maxlev
							ibow = 0.0;
						}
						
						if (fstar>=UPPER_SINGLE_PRECISION_OF_0 || ibow!=DONT_PRINT ) {
							if (lmt->addwithoffset(locng,(float)log10(pr),(float)ibow)){
								num[l]++;
							}else{
								continue;
							}
						} else{
							continue; //skip n-grams with too small fstar
						}
					}
					oldng=ng;
				}
			}
			else{
				//create empty tables for all levels but the level 1, to keep consistency with the rest of the code
				for (int i=2; i<=maxlev; i++) 
					lmt->expand_level(i,0,tmpfilename,mmap);
			}
			
			
			//level 1 is not modified until everything is done
			//because it has to contain the full dictionary
			//which provides the direct access to the second level
			for (int i=2; i<=lmsize(); i++){
				
				if (i>2) {
					lmt->checkbounds(i-1);
					lmt->appendbin_level(i-1, *tout[i-1], mmap);
				}
				
				// now we can resize table at level i
				lmt->resize_level(i, tmpfilename, mmap);
			}
			
			// now we can save table at level maxlev, if not equal to 1
			if (maxlev>1){
				lmt->appendbin_level(maxlev, *tout[maxlev], mmap);
			}
			
			//delete levels from 2 to lmsize();
			for (int i=2; i<=maxlev; i++)			lmt->delete_level(i, tmpfilename, mmap);
			
			//update table offsets
			for (int i=2; i<=maxlev; i++) lmt->update_offset(i,num[i]);
		}
		
		if (_OOV_unigram){
			ngram ung(dict,1);
			*ung.wordp(1)=dict->oovcode();
			ibow=0.0;
			pr=oovprob;
			pr=(pr?log10(pr):-99);
			lmt->addwithoffset(ung,(float)pr,(float)ibow);
			num[1]++;
		}
		
		//close levels from 2 to lmsize()
		for (int i=2; i<=maxlev; i++) tout[i]->close();
		
		//now we can save level 1, which contains all unigrams
		//cerr << "saving level 1" << "...\n";
		lmt->savebin_level(1, filename, mmap);
		
		//update headers
		for (int i=1; i<=maxlev; i++) {
			sprintf(buff," %10d",num[i]);
			out.seekp(pos[i]);
			out << buff;
		}
		
		out.close();
		
		//concatenate files for each single level into one file
		//single level files should have a name derived from "filename"
		lmt->compact_all_levels(filename);
		
		cerr << "\n";
		system("date");
		
		VERBOSE(2,"mdiadaptlm::saveBIN_per_word END\n");	
		return 1;
	};
	
	///// Save in binary format forbackoff N-gram models
	int mdiadaptlm::saveBIN_per_level(char *filename,int backoff,char* subdictfile,int mmap)
	{
		VERBOSE(2,"mdiadaptlm::saveBIN_per_level START\n");
		system("date");
		
		//subdict
		dictionary* subdict;
		
		if (subdictfile)     subdict=new dictionary(subdictfile);
		else    subdict=dict; // default is subdict=dict
		
		if (mmap) {
			VERBOSE(2,"savebin with memory map: " << filename << "\n");
		} else {
			VERBOSE(2,"savebin: " << filename << "\n");
		}
		
		int maxlev=lmsize();
		streampos pos[LMTMAXLEV+1];
		char buff[100];
		int isQuant=0; //savebin for quantized LM is not yet implemented
		
		// print header
		fstream out(filename,ios::out);
		out << "blmt " << maxlev;
		
		for (int i=1; i<=maxlev; i++) { //reserve space for ngram statistics (which are not yet avalable)
			pos[i]=out.tellp();
			sprintf(buff," %10d",0);
			out << buff;
		}
		out << "\n";
		lmtable* lmt = new lmtable();
		
		lmt->configure(maxlev,isQuant);
		
		lmt->setDict(subdict);
		subdict->save(out);
		out.flush();
		
		
		//start adding n-grams to lmtable
		
		for (int i=1; i<=maxlev; i++) {
			cerr << "saving level " << i << "...\n";
			table_entry_pos_t numberofentries;
			if (i==1) { //unigram
				numberofentries = (table_entry_pos_t) subdict->size();
			} else {
				numberofentries = (table_entry_pos_t) entries(i);
			}
			system("date");
			lmt->expand_level(i,numberofentries,filename,mmap);
			
			double fstar,lambda,bo,dummy,dummy2,pr,ibow;
			
			ngram ng(dict,1);
			ngram ng2(dict);
			ngram sng(subdict,1);
			
			if (i==1) { //unigram case
				
				double oovprob=0.0; //accumulated unigram oov pro
				bool _OOV_unigram=false; //flag to check whether an OOV word is present or not
				
				//scan the dictionary
				for (int w=0; w<dict->size(); w++) {
					*ng.wordp(1)=w;
					
					sng.trans(ng);
					
					// frequency pruning is not applied to unigrams
					
					pr=mdiadaptlm::prob(ng,i,fstar,dummy,dummy2);
					
					if (sng.containsWord(subdict->OOV(),i) || ng.containsWord(dict->OOV(),i)) {
						_OOV_unigram=true;
						oovprob+=pr; //accumulate oov probability
						continue;
					}
					
					/*
					 if (sng.containsWord(subdict->OOV(),i) && !ng.containsWord(dict->OOV(),i)) {
					 oovprob+=pr; //accumulate oov probability
					 continue;
					 }
					 
					 if (ng.containsWord(dict->OOV(),i)) pr+=oovprob;
					 */
					
					//cerr << ng << " freq " << dict->freq(w) << " -  Pr " << pr << "\n";
					pr=(pr?log10(pr):-99);
					
					/*
					 if (w==dict->oovcode()){
					 //CHECK whether we can avoid this reassignment because dict should be lmt->getDict()   
					 *ng.wordp(1)=lmt->getDict()->oovcode();
					 ibow=0.0;
					 }
					 else {
					 //				} //do nothing
					 */
					if (i<maxlev) {
						ngram ng2=ng;
						ng2.pushc(0); //extend by one
						
						//cerr << ng2 << "\n";
												
						VERBOSE(3,"mdiadaptlm::saveBIN_per_level(char *filename,int backoff,char* subdictfile ) computing backoff for ng2:|" << ng2 << "| size:" << i+1 << std::endl);
						mdiadaptlm::bodiscount(ng2,i+1,dummy,lambda,bo);
						VERBOSE(3,"mdiadaptlm::saveBIN_per_level(char *filename,int backoff,char* subdictfile ) getting backoff for ng2:|" << ng2 << "| lambda:" << lambda << " bo:" << bo << std::endl);
						
						if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
							ibow = DONT_PRINT;
						}else{
							if (backoff){
								ibow = log10(lambda) - log10(bo);
							}
							else{
								MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
								if (lambda<LOWER_SINGLE_PRECISION_OF_1){
									ibow = log10(lambda);
								}else { //force to be 0.0
									ibow = 0.0;
								}
							}
						}
					}else { //i==maxlev
						ibow=0.0; //default value for backoff weight at the highest level
					}
					VERBOSE(3,"mdiadaptlm::saveBIN_per_level(char *filename,int backoff,char* subdictfile ) writing w:|" << (char *)dict->decode(w) << "| pr:" << pr << " ibow:" << ibow << std::endl);
					if (ibow != DONT_PRINT ) {
						lmt->add(ng,(float)pr,(float)ibow);
					}
				}
				//add unigram with OOV and its accumulate oov probability
				if (_OOV_unigram){
					*ng.wordp(1)=lmt->getDict()->oovcode();
					ibow=0.0;
					pr=oovprob;
					pr=(pr?log10(pr):-99);
					lmt->add(ng,(float)pr,(float)ibow);
				}
			}
			else { //i>1 , bigrams, trigrams, fourgrams...
				*ng.wordp(1)=0;
				get(ng,1,1); //this 
				scan(ng,INIT,i);
				while(scan(ng,CONT,i)) {
					sng.trans(ng);
					
					// frequency pruning: skip n-grams with low frequency 
					if (prune_ngram(i,sng.freq)) continue;
					
					// skip n-grams containing OOV
					if (sng.containsWord(subdict->OOV(),i)) continue;
					
					// skip also n-grams containing eos symbols not at the final
					if (sng.containsWord(dict->EoS(),i-1)) continue;
					
					//	mdiadaptlm::discount(ng,i,fstar,dummy);
					//	pr=mdiadaptlm::prob(ng,i);
					pr=mdiadaptlm::prob(ng,i,fstar,dummy,dummy2);
					
					if (!(pr<=1.0 && pr > 1e-10)) {
						cerr << ng << " " << pr << "\n";
						MY_ASSERT(pr<=1.0);
						cerr << "prob modified to 1e-10\n";
						pr=1e-10;
					}
					
					if (i<maxlev) {
						ng2=ng;
						ng2.pushc(0); //extend by one
						VERBOSE(3,"mdiadaptlm::saveBIN_per_level(char *filename,int backoff,char* subdictfile ) computing backoff for ng2:|" << ng2 << "| size:" << i+1 << std::endl);
						mdiadaptlm::bodiscount(ng2,i+1,dummy,lambda,bo);
						VERBOSE(3,"mdiadaptlm::saveBIN_per_level(char *filename,int backoff,char* subdictfile ) getting backoff for ng2:|" << ng2 << "| lambda:" << lambda << " bo:" << bo << std::endl);
						
						if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
							ibow=DONT_PRINT;
						}else{
							if (backoff){
								ibow=log10(lambda) - log10(bo);
							}else{
								MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
								if (lambda<LOWER_SINGLE_PRECISION_OF_1){
									ibow=log10(lambda);
								}else{ //force ibow to log10(lambda)==0.0
									ibow=0.0;
								}
							}
						}
					} else { //i==maxlev
						ibow=0.0; //value for backoff weight at the highest level
					}
					VERBOSE(3,"mdiadaptlm::saveBIN_per_level(char *filename,int backoff,char* subdictfile ) writing ng:|" << ng << "| pr:" << pr << " ibow:" << ibow << std::endl);
					if (ibow != DONT_PRINT ) {
						lmt->add(ng,(float)log10(pr),(float)ibow);
					}
				}
			}
			
			// now we can fix table at level i-1
			// now we can save table at level i-1
			// now we can remove table at level i-1
			if (maxlev>1 && i>1) {
				lmt->checkbounds(i-1);
				lmt->savebin_level(i-1, filename, mmap);
			}
			
			// now we can resize table at level i
			lmt->resize_level(i, filename, mmap);
			
		}
		// now we can save table at level maxlev
		lmt->savebin_level(maxlev, filename, mmap);
		
		//update headers
		for (int i=1; i<=maxlev; i++) {
			sprintf(buff," %10d",lmt->getCurrentSize(i));
			out.seekp(pos[i]);
			out << buff;
		}
		out.close();
		
		//concatenate files for each single level into one file
		//single level files should have a name derived from "filename"
		lmt->compact_all_levels(filename);
		
		VERBOSE(2,"mdiadaptlm::saveBIN_per_level END\n");
		return 1;
	}
	
	
	///// Save in format for ARPA backoff N-gram models
	int mdiadaptlm::saveARPA_per_word(char *filename,int backoff,char* subdictfile )
	{
		VERBOSE(2,"mdiadaptlm::saveARPA_per_word START\n");
		system("date");
		
		//subdict
		dictionary* subdict;
		
		
		if (subdictfile) subdict=new dictionary(subdictfile);
		else   subdict=dict; // default is subdict=dict
		
		//main output file
		mfstream out(filename,ios::out);
		
		int maxlev=lmsize();
		//create temporary output file stream
		MY_ASSERT(strlen(filename)<1000);
		char tfilename[LMTMAXLEV+1][1000];
		mfstream *tout[LMTMAXLEV+1];
		
		tout[0]=NULL;
		for (int i=1; i<=maxlev; i++) {
			sprintf(tfilename[i],"%s.%d",filename,i);
			tout[i]=new mfstream(tfilename[i],ios::out);
			*tout[i] << "\n\\" << i << "-grams:\n";
		}
		
		ngram ng(dict,lmsize());
		ngram oldng(dict,lmsize());
		ngram locng(dict,lmsize());
		
		ngram sng(subdict,lmsize());
		
		double fstar,lambda,bo,dummy,dummy2,pr,outLambda;
		
		double oovprob=0.0; //accumulated unigram oov pro
		bool _OOV_unigram=false; //flag to check whether an OOV word is present or not
		
		//n-gram counters
		table_entry_pos_t num[LMTMAXLEV+1];
		for (int i=1; i<=maxlev; i++) num[i]=0;
		
		
		//main loop
		for (int w=0; w<dict->size(); w++) {
			int i=1;  //set the initial value of level
			if (!w % 10000) cerr << ".";
			
			//1-gram
			ngram ung(dict,1);
			*ung.wordp(1)=w;
			sng.trans(ung);
			
			// frequency pruning is not applied to unigrams
			
			/*
			 //exclude words not occurring in the subdictionary
			 if (sng.containsWord(subdict->OOV(),1) && !ung.containsWord(dict->OOV(),1))	continue;
			 */
			
			pr=mdiadaptlm::prob(ung,1);
			pr=(pr?log10(pr):-99);
			
			//////CHECK
			if (sng.containsWord(subdict->OOV(),1) || ung.containsWord(dict->OOV(),1)) {
				_OOV_unigram=true;
				oovprob+=pr; //accumulate oov probability
				continue;
			}
			
			if (i<maxlev) { //print back-off
				ung.pushc(0); //extend by one
				VERBOSE(3,"mdiadaptlm::saveARPA_per_word(char *filename,int backoff,char* subdictfile ) computing backoff for ung:|" << ung << "| size:" << i+1 << std::endl);
				mdiadaptlm::bodiscount(ung,i+1,dummy,lambda,bo);
				VERBOSE(3,"mdiadaptlm::saveARPA_per_word(char *filename,int backoff,char* subdictfile ) getting backoff for ung:|" << ung << "| lambda:" << lambda << " bo:" << bo << std::endl);
				
				ung.shift();//shrink by one
				if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
					outLambda = DONT_PRINT;
				}else{
					if (backoff){
						outLambda = (float) (log10(lambda) - log10(bo));
					}
					else{
						MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
						if (lambda<LOWER_SINGLE_PRECISION_OF_1){
							outLambda = (float) log10(lambda);
						}
						else {
							outLambda = DONT_PRINT;
						}
					}
				}
			}else { //i==maxlev
				outLambda = DONT_PRINT;
			}
			
			//cerr << ng << " freq " << dict->freq(w) << " -  Pr " << pr << "\n";
			*tout[i] << (float)  (pr?log10(pr):-99);
			*tout[i] << "\t" << (char *)dict->decode(w);
			if (outLambda != DONT_PRINT){
				*tout[i] << "\t" << outLambda;
			}
			*tout[i] << "\n";
			num[i]++;
			
			//manage n-grams
			if (get(ung,1,1)) {
				
				//create n-gram with history w
				*ng.wordp(maxlev)=w;
				
				//create sentinel n-gram
				for (i=1; i<=maxlev; i++) *oldng.wordp(i)=-1;
				
				scan(ung.link,ung.info,1,ng,INIT,maxlev);
				while(scan(ung.link,ung.info,1,ng,CONT,maxlev)) {
					//cerr << ng << "\n";
					sng.trans(ng); // convert to subdictionary
					locng=ng;      // make a local copy
					
					//find first internal level that changed
					int f=maxlev-1; //unigrams have been already covered
					while (f>1 && (*oldng.wordp(f)==*ng.wordp(f))){ f--; }
					
					for (int l=maxlev; l>maxlev-f;l--){
						
						if (l<maxlev) locng.shift(); //ngram has size level
						
						// frequency pruning: skip n-grams with low frequency 
						if (prune_ngram(l,sng.freq)) continue;
						
						// skip n-grams containing OOV
						if (sng.containsWord(subdict->OOV(),l)) continue;
						
						// skip also n-grams containing eos symbols not at the final
						if (sng.containsWord(dict->EoS(),l-1)) continue;
						VERBOSE(3,"mdiadaptlm::saveARPA_per_word(char *filename,int backoff,char* subdictfile ) computing prob for locng:|" << locng << "| size:" << i << std::endl);
						pr=mdiadaptlm::prob(locng,l,fstar,dummy,dummy2);
						VERBOSE(3,"mdiadaptlm::saveARPA_per_word(char *filename,int backoff,char* subdictfile ) getting prob locng:|" << locng << "| size:" << i << " fstar:" << fstar << " pr:" << pr << std::endl);
						
						//PATCH by Nicola (16-04-2008)
						
						if (!(pr<=1.0 && pr > 1e-10)) {
							cerr << ng << " " << pr << "\n";
							MY_ASSERT(pr<=1.0);
							cerr << "prob modified to 1e-10\n";
							pr=1e-10;
						}
						
						if (l<maxlev) {
							
							locng.pushc(0); //extend by one
							VERBOSE(3,"mdiadaptlm::saveARPA_per_word(char *filename,int backoff,char* subdictfile ) computing backoff for locng:|" << locng << "| size:" << l+1 << std::endl);
							mdiadaptlm::bodiscount(locng,l+1,dummy,lambda,bo);
							VERBOSE(3,"mdiadaptlm::saveARPA_per_word(char *filename,int backoff,char* subdictfile ) getting backoff locng:|" << locng << "| lambda:" << lambda << " bo:" << bo << std::endl);
							
							locng.shift();
							if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
								outLambda = DONT_PRINT;
							}else{								
								if (backoff){
									outLambda = (float) (log10(lambda) - log10(bo));
								}else{
									MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
									if (lambda<LOWER_SINGLE_PRECISION_OF_1){
										outLambda = (float) log10(lambda);
									}else{ //no output if log10(lambda)==0
										outLambda = DONT_PRINT;
									}
								}					
							}
						} else { //i==maxlev
							outLambda = DONT_PRINT;
						}
							
						if (fstar>=UPPER_SINGLE_PRECISION_OF_0 || outLambda!=DONT_PRINT ) {
							*tout[l] << (float) log10(pr);
							*tout[l] << "\t" << (char *)dict->decode(*ng.wordp(i));
							for (int j=i-1; j>0; j--)
								*tout[l] << " " << (char *)dict->decode(*ng.wordp(j));
							if (outLambda != DONT_PRINT){
								*tout[l] << "\t" << outLambda;
							}
							*tout[l] << "\n";
							num[l]++;
						} else{
							continue; //skip n-grams with too small fstar
						}						
					}
					oldng=ng;
				}
			}
		}
		if (_OOV_unigram){
			pr=oovprob;
			num[1]++;
			out << (float)  (pr?log10(pr):-99);
			out << "\t" << "<unk>\n";
		}
		
		//print header
		out << "\n\\data\\" << "\n";
		char buff[100];
		for (int i=1; i<=maxlev; i++) {
			sprintf(buff,"ngram %2d=%10d\n",i,num[i]);
			out << buff;
		}
		out << "\n";
		
		//append and remove temporary files
		for (int i=1; i<=maxlev; i++) {
			delete tout[i];
			tout[i]=new mfstream(tfilename[i],ios::in);
			out << tout[i]->rdbuf();
			delete tout[i];
			removefile(tfilename[i]);
		}
		
		out << "\\end\\" << "\n";
		
		cerr << "\n";
		system("date");
		
		VERBOSE(2,"mdiadaptlm::saveARPA_per_word END\n");
		return 1;
	};
	
	///// Save in format for ARPA backoff N-gram models
	int mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile )
	{
		VERBOSE(2,"mdiadaptlm::saveARPA_per_level START\n");
		system("date");
		
		//subdict
		dictionary* subdict;
		
		if (subdictfile) {
			subdict=new dictionary(subdictfile);
		} else
			subdict=dict; // default is subdict=dict
		
		fstream out(filename,ios::out);
		//  out.precision(15);
		
		int maxlev = lmsize();
		streampos pos[LMTMAXLEV+1];
		table_entry_pos_t num[LMTMAXLEV+1];
		char buff[100];
		
		//print header
		out << "\n\\data\\" << "\n";
		
		for (int i=1; i<=maxlev; i++) {
			num[i]=0;
			pos[i]=out.tellp();
			sprintf(buff,"ngram %2d=%10d\n",i,num[i]);
			out << buff;
		}
		
		out << "\n";
		
		//start writing n-grams
		
		for (int i=1; i<=maxlev; i++) {
			cerr << "saving level " << i << "...\n";
			
			
			out << "\n\\" << i << "-grams:\n";
			
			double fstar,lambda,bo,dummy,dummy2,pr,outLambda;
			
			ngram ng(dict,1);
			ngram ng2(dict);
			ngram sng(subdict,1);
			
			if (i==1) { //unigram case
				
				double oovprob=0.0; //accumulated unigram oov pro
				bool _OOV_unigram=false; //flag to check whether an OOV word is present or not
				
				//scan the dictionary
				
				for (int w=0; w<dict->size(); w++) {
					*ng.wordp(1)=w;
					
					sng.trans(ng);
					
					// frequency pruning is not applied to unigrams
					VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) computing prob for ng:|" << ng << "| size:" << i << std::endl);
					pr=mdiadaptlm::prob(ng,i,fstar,dummy,dummy2);
					VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) getting prob for ng:|" << ng << "| pr:" << pr << std::endl);
					
					if (sng.containsWord(subdict->OOV(),i) || ng.containsWord(dict->OOV(),i)) {
						_OOV_unigram=true;
						oovprob+=pr; //accumulate oov probability
						continue;
					}
					
					/*
					 if (sng.containsWord(subdict->OOV(),i) && !ng.containsWord(dict->OOV(),i)) {
					 oovprob+=pr; //accumulate oov probability
					 continue;
					 }
					 
					 if (ng.containsWord(dict->OOV(),i)) pr+=oovprob;
					 */
					
					if (i<maxlev) {
						ngram ng2=ng;
						ng2.pushc(0); //extend by one
						
						VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) computing backoff for ng2:|" << ng2 << "| size:" << i+1 << std::endl);
						mdiadaptlm::bodiscount(ng2,i+1,dummy,lambda,bo);
						VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) getting backoff for ng2:|" << ng2 << "| lambda:" << lambda << " bo:" << bo << std::endl);
						if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
							outLambda = DONT_PRINT;
						}else{
							if (backoff){
								outLambda = (float) (log10(lambda) - log10(bo));
							}
							else{
								MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
								if (lambda<LOWER_SINGLE_PRECISION_OF_1){
									outLambda = (float) log10(lambda);
								}
								else { //force to be 0.0 and hence to not output lambda
									outLambda = DONT_PRINT;
								}
							}
						}
					}else { //i==maxlev
						outLambda = DONT_PRINT;
					}
					
					VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) writing w:|" << (char *)dict->decode(w) << "| pr:" << pr << " outLambda:" << outLambda << std::endl);
					//cerr << ng << " freq " << dict->freq(w) << " -  Pr " << pr << "\n";
					out << (float)  (pr?log10(pr):-99);
					out << "\t" << (char *)dict->decode(w);
					if (outLambda != DONT_PRINT){
						out << "\t" << outLambda;
					}
					out << "\n";
					
					num[i]++;
				}
				
				
				//add unigram with OOV and its accumulate oov probability
				if (_OOV_unigram){
					pr=oovprob;
					num[i]++;
					out << (float)  (pr?log10(pr):-99);
					out << "\t" << "<unk>\n";
				}
			}
			else { //i>1 , bigrams, trigrams, fourgrams...
				*ng.wordp(1)=0;
				get(ng,1,1); //this 
				scan(ng,INIT,i);
				while(scan(ng,CONT,i)) {
					
					sng.trans(ng);
					
					// frequency pruning: skip n-grams with low frequency 
					if (prune_ngram(i,sng.freq)) continue;
					
					// skip n-grams containing OOV
					if (sng.containsWord(subdict->OOV(),i)) continue;
					
					// skip also n-grams containing eos symbols not at the final
					if (sng.containsWord(dict->EoS(),i-1)) continue;
					
					VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) computing prob for ng:|" << ng << "| size:" << i << std::endl);
					pr=mdiadaptlm::prob(ng,i,fstar,dummy,dummy2);
					VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) getting prob ng:|" << ng << "| size:" << i << " fstar:" << fstar << " pr:" << pr << std::endl);
					
					//PATCH by Nicola (16-04-2008)					
					if (!(pr<=1.0 && pr > 1e-10)) {
						cerr << ng << " " << pr << "\n";
						MY_ASSERT(pr<=1.0);
						cerr << "prob modified to 1e-10\n";
						pr=1e-10;
					}
					
					if (i<maxlev) {
						ng2=ng;
						ng2.pushc(0); //extend by one
						
						VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) computing backoff for ng2:|" << ng2 << "| size:" << i+1 << std::endl);
						mdiadaptlm::bodiscount(ng2,i+1,dummy,lambda,bo);
						VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) getting backoff for ng2:|" << ng2 << "| lambda:" << lambda << " bo:" << bo << std::endl);
						
						if (fstar<UPPER_SINGLE_PRECISION_OF_0 && lambda>LOWER_SINGLE_PRECISION_OF_1){ //ngram must be skipped
							outLambda = DONT_PRINT;
						}else{
							if (backoff){
								outLambda = (float) (log10(lambda) - log10(bo));
							}else{
								MY_ASSERT((lambda<UPPER_SINGLE_PRECISION_OF_1 && lambda>LOWER_SINGLE_PRECISION_OF_1) || bo<UPPER_SINGLE_PRECISION_OF_1 );
								if (lambda<LOWER_SINGLE_PRECISION_OF_1){
									outLambda = (float) log10(lambda);
								}else{ //no output of lambda if log10(lambda)==0
									outLambda = DONT_PRINT;
								}
							}
						}			
					} else { //i==maxlev
						outLambda = DONT_PRINT;
					}
					
					VERBOSE(3,"mdiadaptlm::saveARPA_per_level(char *filename,int backoff,char* subdictfile ) writing ng:|" << ng << "| pr:" << pr << " outLambda:" << outLambda << std::endl);
					if (fstar>=UPPER_SINGLE_PRECISION_OF_0 || outLambda!=DONT_PRINT ) {
						out << (float) log10(pr);
						out << "\t" << (char *)dict->decode(*ng.wordp(i));
						for (int j=i-1; j>0; j--)
							out << " " << (char *)dict->decode(*ng.wordp(j));
						if (outLambda != DONT_PRINT){
							out << "\t" << outLambda;
						}
						out << "\n";
						num[i]++;
					}	
				}
			}
			
			cerr << i << "grams tot:" << num[i] << "\n";
		}
		
		streampos last=out.tellp();
		
		//update headers
		for (int i=1; i<=maxlev; i++) {
			sprintf(buff,"ngram %2d=%10u\n",i,num[i]);
			out.seekp(pos[i]);
			out << buff;
		}
		
		out.seekp(last);
		out << "\\end\\" << "\n";
		system("date");
		
		VERBOSE(2,"mdiadaptlm::saveARPA_per_level END\n");
		return 1;
	};
	
}//namespace irstlm

/*
 main(int argc,char** argv){
 char* dictname=argv[1];
 char* backngram=argv[2];
 int depth=atoi(argv[3]);
 char* forengram=argv[4];
 char* testngram=argv[5];
 
 dictionary dict(dictname);
 ngramtable test(&dict,testngram,depth);
 
 shiftbeta lm2(&dict,backngram,depth);
 lm2.train();
 //lm2.test(test,depth);
 
 mdi lm(&dict,backngram,depth);
 lm.train();
 for (double w=0.0;w<=1.0;w+=0.1){
 lm.getforelm(forengram);
 lm.adapt(w);
 lm.test(test,depth);
 }
 }
 */

