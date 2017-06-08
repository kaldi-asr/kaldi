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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include "util.h"
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "ngramtable.h"
#include "ngramcache.h"
#include "normcache.h"
#include "interplm.h"
#include "mdiadapt.h"
#include "shiftlm.h"
	
namespace irstlm {
//
//Shiftone interpolated language model
//

shiftone::shiftone(char* ngtfile,int depth,int prunefreq,TABLETYPE tt):
  mdiadaptlm(ngtfile,depth,tt)
{
  cerr << "Creating LM with ShiftOne smoothing\n";
  prunethresh=prunefreq;
  cerr << "PruneThresh: " << prunethresh << "\n";

  beta=1.0;

};


int shiftone::train()
{
  trainunigr();
  return 1;
}


int shiftone::discount(ngram ng_,int size,double& fstar,double& lambda, int cv)
{

  ngram ng(dict);
  ng.trans(ng_);

  //cerr << "size:" << size << " ng:|" << ng <<"|\n";

  if (size > 1) {

    ngram history=ng;

    if (ng.ckhisto(size) && get(history,size,size-1) && (history.freq>cv) &&
        ((size < 3) || ((history.freq-cv) > prunethresh))) {

      // this history is not pruned out

      get(ng,size,size);
      cv=(cv>ng.freq)?ng.freq:cv;

      if (ng.freq > cv) {

        fstar=(double)((double)(ng.freq - cv) - beta)/(double)(history.freq-cv);

        lambda=beta * ((double)history.succ/(double)(history.freq-cv));

      } else { // ng.freq == cv: do like if ng was deleted from the table

        fstar=0.0;

        lambda=beta * ((double)(history.succ-1)/ //one successor has disappeared!
                       (double)(history.freq-cv));

      }

      //cerr << "ngram :" << ng << "\n";

      //check if the last word is OOV
      if (*ng.wordp(1)==dict->oovcode()) {
        lambda+=fstar;
        fstar=0.0;
      } else { //complete lambda with oovcode probability
        *ng.wordp(1)=dict->oovcode();
        if (get(ng,size,size))
          lambda+=(double)((double)ng.freq - beta)/(double)(history.freq-cv);
      }

    } else {
      fstar=0;
      lambda=1;
    }
  } else {
    fstar=unigr(ng);
    lambda=0.0;
  }

  return 1;
}




//
//Shiftbeta interpolated language model
//

shiftbeta::shiftbeta(char* ngtfile,int depth,int prunefreq,double b,TABLETYPE tt):
  mdiadaptlm(ngtfile,depth,tt)
{
  cerr << "Creating LM with ShiftBeta smoothing\n";

  if (b==-1.0 || (b < 1.0 && b >0.0)) {
    beta=new double[lmsize()+1];
    for (int l=lmsize(); l>1; l--)
      beta[l]=b;
  } else {
		exit_error(IRSTLM_ERROR_DATA,"shiftbeta::shiftbeta beta must be < 1.0 and > 0");
  }

  prunethresh=prunefreq;
  cerr << "PruneThresh: " << prunethresh << "\n";
};



int shiftbeta::train()
{
  ngram ng(dict);
  int n1,n2;

  trainunigr();

  beta[1]=0.0;

  for (int l=2; l<=lmsize(); l++) {

    cerr << "level " << l << "\n";
    n1=0;
    n2=0;
    scan(ng,INIT,l);
    while(scan(ng,CONT,l)) {


      if (l<lmsize()) {
        //Computing succ1 statistics for this ngram
        //to correct smoothing due to singleton pruning

        ngram hg=ng;
        get(hg,l,l);
        int s1=0;
        ngram ng2=hg;
        ng2.pushc(0);

        succscan(hg,ng2,INIT,l+1);
        while(succscan(hg,ng2,CONT,l+1)) {
          if (ng2.freq==1) s1++;
        }
        succ1(hg.link,s1);
      }

      //skip ngrams containing _OOV
      if (l>1 && ng.containsWord(dict->OOV(),l)) {
        //cerr << "skp ngram" << ng << "\n";
        continue;
      }

      //skip n-grams containing </s> in context
      if (l>1 && ng.containsWord(dict->EoS(),l-1)) {
        //cerr << "skp ngram" << ng << "\n";
        continue;
      }

      //skip 1-grams containing <s>
      if (l==1 && ng.containsWord(dict->BoS(),l)) {
        //cerr << "skp ngram" << ng << "\n";
        continue;
      }

      if (ng.freq==1) n1++;
      else if (ng.freq==2) n2++;

    }
    //compute statistics of shiftbeta smoothing
    if (beta[l]==-1) {
      if (n1>0)
        beta[l]=(double)n1/(double)(n1 + 2 * n2);
      else {
        cerr << "no singletons! \n";
        beta[l]=1.0;
      }
    }
    cerr << beta[l] << "\n";
  }

  return 1;
};



int shiftbeta::discount(ngram ng_,int size,double& fstar,double& lambda, int cv)
{

  ngram ng(dict);
  ng.trans(ng_);

  if (size > 1) {

    ngram history=ng;

    if (ng.ckhisto(size) && get(history,size,size-1) && (history.freq>cv) &&

        ((size < 3) || ((history.freq-cv) > prunethresh ))) {

      // apply history pruning on trigrams only


      if (get(ng,size,size) && (!prunesingletons() || ng.freq >1 || size<3)) {
        cv=(cv>ng.freq)?ng.freq:cv;

        if (ng.freq>cv) {

          fstar=(double)((double)(ng.freq - cv) - beta[size])/(double)(history.freq-cv);

          lambda=beta[size]*((double)history.succ/(double)(history.freq-cv));

          if (size>=3 && prunesingletons())  // correction due to frequency pruning

            lambda+=(1.0-beta[size]) * (double)succ1(history.link)/(double)(history.freq-cv);

          // succ1(history.link) is not affected if ng.freq > cv

        } else { // ng.freq == cv

          fstar=0.0;

          lambda=beta[size]*((double)(history.succ-1)/ //e` sparito il successore
                             (double)(history.freq-cv));

          if (size>=3 && prunesingletons()) //take into account single event pruning
            lambda+=(1.0-beta[size]) * (double)(succ1(history.link)-(cv==1 && ng.freq==1?1:0))
                    /(double)(history.freq-cv);
        }
      } else {

        fstar=0.0;
        lambda=beta[size]*(double)history.succ/(double)history.freq;

        if (size>=3 && prunesingletons()) // correction due to frequency pruning
          lambda+=(1.0-beta[size]) * (double)succ1(history.link)/(double)history.freq;

      }

      //cerr << "ngram :" << ng << "\n";

      if (*ng.wordp(1)==dict->oovcode()) {
        lambda+=fstar;
        fstar=0.0;
      } else {
        *ng.wordp(1)=dict->oovcode();
        if (get(ng,size,size) && (!prunesingletons() || ng.freq >1 || size<3))
          lambda+=(double)((double)ng.freq - beta[size])/(double)(history.freq-cv);
      }

    } else {
      fstar=0;
      lambda=1;
    }
  } else {
    fstar=unigr(ng);
    lambda=0.0;
  }

  return 1;
}

//
//Improved Kneser-Ney language model (previously ModifiedShiftBeta)
//

improvedkneserney::improvedkneserney(char* ngtfile,int depth,int prunefreq,TABLETYPE tt):
  mdiadaptlm(ngtfile,depth,tt)
{
  cerr << "Creating LM with Improved Kneser-Ney smoothing\n";

  prunethresh=prunefreq;
  cerr << "PruneThresh: " << prunethresh << "\n";

  beta[1][0]=0.0;
  beta[1][1]=0.0;
  beta[1][2]=0.0;

};


int improvedkneserney::train()
{
	
	trainunigr();
	
	gencorrcounts();
	gensuccstat();
	
	ngram ng(dict);
	int n1,n2,n3,n4;
	int unover3=0;
	
	oovsum=0;
	
	for (int l=1; l<=lmsize(); l++) {
		
		cerr << "level " << l << "\n";
		
		cerr << "computing statistics\n";
		
		n1=0;
		n2=0;
		n3=0,n4=0;
		
		scan(ng,INIT,l);
		
		while(scan(ng,CONT,l)) {
			
			//skip ngrams containing _OOV
			if (l>1 && ng.containsWord(dict->OOV(),l)) {
				continue;
			}
			
			//skip n-grams containing </s> in context
			if (l>1 && ng.containsWord(dict->EoS(),l-1)) {
				continue;
			}
			
			//skip 1-grams containing <s>
			if (l==1 && ng.containsWord(dict->BoS(),l)) {
				continue;
			}
			
			ng.freq=mfreq(ng,l);
			
			if (ng.freq==1) n1++;
			else if (ng.freq==2) n2++;
			else if (ng.freq==3) n3++;
			else if (ng.freq==4) n4++;
			if (l==1 && ng.freq >=3) unover3++;
		}
		
		if (l==1) {
			cerr << " n1: " << n1 << " n2: " << n2 << " n3: " << n3 << " n4: " << n4 << " unover3: " << unover3 << "\n";
		} else {
			cerr << " n1: " << n1 << " n2: " << n2 << " n3: " << n3 << " n4: " << n4 << "\n";
		}
		
		if (n1 == 0 || n2 == 0 ||  n1 <= n2) {
			std::stringstream ss_msg;
			ss_msg << "Error: lower order count-of-counts cannot be estimated properly\n";
			ss_msg << "Hint: use another smoothing method with this corpus.\n";
			exit_error(IRSTLM_ERROR_DATA,ss_msg.str());
		}
		
		double Y=(double)n1/(double)(n1 + 2 * n2);
		beta[0][l] = Y; //equivalent to  1 - 2 * Y * n2 / n1
		
		if (n3 == 0 || n4 == 0 || n2 <= n3 || n3 <= n4 ){
			cerr << "Warning: higher order count-of-counts cannot be estimated properly\n";
			cerr << "Fixing this problem by resorting only on the lower order count-of-counts\n";
			
			beta[1][l] = Y;
			beta[2][l] = Y;			
		}
		else{ 	  
			beta[1][l] = 2 - 3 * Y * n3 / n2; 
			beta[2][l] = 3 - 4 * Y * n4 / n3;  
		}
		
		if (beta[1][l] < 0){
			cerr << "Warning: discount coefficient is negative \n";
			cerr << "Fixing this problem by setting beta to 0 \n";			
			beta[1][l] = 0;
			
		}		
		
		
		if (beta[2][l] < 0){
			cerr << "Warning: discount coefficient is negative \n";
			cerr << "Fixing this problem by setting beta to 0 \n";			
			beta[2][l] = 0;
			
		}
				
		
		if (l==1)
			oovsum=beta[0][l] * (double) n1 + beta[1][l] * (double)n2 + beta[2][l] * (double)unover3;
		
		cerr << beta[0][l] << " " << beta[1][l] << " " << beta[2][l] << "\n";
	}
	
	return 1;
};



int improvedkneserney::discount(ngram ng_,int size,double& fstar,double& lambda, int cv)
{
  ngram ng(dict);
  ng.trans(ng_);

  //cerr << "size:" << size << " ng:|" << ng <<"|\n";

  if (size > 1) {

    ngram history=ng;

    //singleton pruning only on real counts!!
    if (ng.ckhisto(size) && get(history,size,size-1) && (history.freq > cv) &&
        ((size < 3) || ((history.freq-cv) > prunethresh ))) { // no history pruning with corrected counts!

      int suc[3];
      suc[0]=succ1(history.link);
      suc[1]=succ2(history.link);
      suc[2]=history.succ-suc[0]-suc[1];


      if (get(ng,size,size) &&
          (!prunesingletons() || mfreq(ng,size)>1 || size<3) &&
          (!prunetopsingletons() || mfreq(ng,size)>1 || size<maxlevel())) {

        ng.freq=mfreq(ng,size);

        cv=(cv>ng.freq)?ng.freq:cv;

        if (ng.freq>cv) {

          double b=(ng.freq-cv>=3?beta[2][size]:beta[ng.freq-cv-1][size]);

          fstar=(double)((double)(ng.freq - cv) - b)/(double)(history.freq-cv);

          lambda=(beta[0][size] * suc[0] + beta[1][size] * suc[1] + beta[2][size] * suc[2])
                 /
                 (double)(history.freq-cv);

          if ((size>=3 && prunesingletons()) ||
              (size==maxlevel() && prunetopsingletons())) // correction due to frequency pruning

            lambda+=(double)(suc[0] * (1-beta[0][size])) / (double)(history.freq-cv);

        } else {
          // ng.freq==cv

          ng.freq>=3?suc[2]--:suc[ng.freq-1]--; //update successor stat

          fstar=0.0;
          lambda=(beta[0][size] * suc[0] + beta[1][size] * suc[1] + beta[2][size] * suc[2])
                 /
                 (double)(history.freq-cv);

          if ((size>=3 && prunesingletons()) ||
              (size==maxlevel() && prunetopsingletons())) // correction due to frequency pruning
            lambda+=(double)(suc[0] * (1-beta[0][size])) / (double)(history.freq-cv);

          ng.freq>=3?suc[2]++:suc[ng.freq-1]++; //resume successor stat
        }
      } else {
        fstar=0.0;
        lambda=(beta[0][size] * suc[0] + beta[1][size] * suc[1] + beta[2][size] * suc[2])
               /
               (double)(history.freq-cv);

        if ((size>=3 && prunesingletons()) ||
            (size==maxlevel() && prunetopsingletons())) // correction due to frequency pruning
          lambda+=(double)(suc[0] * (1-beta[0][size])) / (double)(history.freq-cv);

      }

      //cerr << "ngram :" << ng << "\n";


      if (*ng.wordp(1)==dict->oovcode()) {
        lambda+=fstar;
        fstar=0.0;
      } else {
        *ng.wordp(1)=dict->oovcode();
        if (get(ng,size,size)) {
          ng.freq=mfreq(ng,size);
          if ((!prunesingletons() || ng.freq>1 || size<3) &&
              (!prunetopsingletons() || ng.freq>1 || size<maxlevel())) {
            double b=(ng.freq>=3?beta[2][size]:beta[ng.freq-1][size]);
            lambda+=(double)(ng.freq - b)/(double)(history.freq-cv);
          }
        }
      }
    } else {
      fstar=0;
      lambda=1;
    }
  } else { // unigram case, no cross-validation

		fstar=unigrIKN(ng);
    lambda=0.0;
  }

  return 1;
}
	
	double improvedkneserney::unigrIKN(ngram ng)
	{ 
		int unigrtotfreq=(lmsize()>1)?btotfreq():totfreq();
		double fstar=0.0;
		if (get(ng,1,1))
			fstar=(double) mfreq(ng,1)/(double)unigrtotfreq;
		else {
			std::stringstream ss_msg;
			ss_msg << "Missing probability for word: " << dict->decode(*ng.wordp(1));
			exit_error(IRSTLM_ERROR_DATA,ss_msg.str());
		}
		return fstar;
	}
	
	//
	//Improved Shiftbeta language model (similar to Improved Kneser-Ney without corrected counts)
	//
	
	improvedshiftbeta::improvedshiftbeta(char* ngtfile,int depth,int prunefreq,TABLETYPE tt):
  mdiadaptlm(ngtfile,depth,tt)
	{
		cerr << "Creating LM with Improved ShiftBeta smoothing\n";
		
		prunethresh=prunefreq;
		cerr << "PruneThresh: " << prunethresh << "\n";
		
		beta[1][0]=0.0;
		beta[1][1]=0.0;
		beta[1][2]=0.0;
		
	};
	
	
	int improvedshiftbeta::train()
	{
		
		trainunigr();
		
		gensuccstat();
		
		ngram ng(dict);
		int n1,n2,n3,n4;
		int unover3=0;
		
		oovsum=0;
		
		for (int l=1; l<=lmsize(); l++) {
			
			cerr << "level " << l << "\n";
			
			cerr << "computing statistics\n";
			
			n1=0;
			n2=0;
			n3=0,n4=0;
			
			scan(ng,INIT,l);
			
			while(scan(ng,CONT,l)) {
				
				//skip ngrams containing _OOV
				if (l>1 && ng.containsWord(dict->OOV(),l)) {
					continue;
				}
				
				//skip n-grams containing </s> in context
				if (l>1 && ng.containsWord(dict->EoS(),l-1)) {
					continue;
				}
				
				//skip 1-grams containing <s>
				if (l==1 && ng.containsWord(dict->BoS(),l)) {
					continue;
				}
				
				ng.freq=mfreq(ng,l);
				
				if (ng.freq==1) n1++;
				else if (ng.freq==2) n2++;
				else if (ng.freq==3) n3++;
				else if (ng.freq==4) n4++;
				if (l==1 && ng.freq >=3) unover3++;
			}
			
			if (l==1) {
				cerr << " n1: " << n1 << " n2: " << n2 << " n3: " << n3 << " n4: " << n4 << " unover3: " << unover3 << "\n";
			} else {
				cerr << " n1: " << n1 << " n2: " << n2 << " n3: " << n3 << " n4: " << n4 << "\n";
			}
			
			if (n1 == 0 || n2 == 0 ||  n1 <= n2) {
				std::stringstream ss_msg;
				ss_msg << "Error: lower order count-of-counts cannot be estimated properly\n";
				ss_msg << "Hint: use another smoothing method with this corpus.\n";
				exit_error(IRSTLM_ERROR_DATA,ss_msg.str());
			}
			
			double Y=(double)n1/(double)(n1 + 2 * n2);
			beta[0][l] = Y; //equivalent to  1 - 2 * Y * n2 / n1
			
			if (n3 == 0 || n4 == 0 || n2 <= n3 || n3 <= n4 ){
				cerr << "Warning: higher order count-of-counts cannot be estimated properly\n";
				cerr << "Fixing this problem by resorting only on the lower order count-of-counts\n";
				
				beta[1][l] = Y;
				beta[2][l] = Y;			
			}
			else{ 	  
				beta[1][l] = 2 - 3 * Y * n3 / n2; 
				beta[2][l] = 3 - 4 * Y * n4 / n3;  
			}
			
			if (beta[1][l] < 0){
				cerr << "Warning: discount coefficient is negative \n";
				cerr << "Fixing this problem by setting beta to 0 \n";			
				beta[1][l] = 0;
				
			}		
			
			
			if (beta[2][l] < 0){
				cerr << "Warning: discount coefficient is negative \n";
				cerr << "Fixing this problem by setting beta to 0 \n";			
				beta[2][l] = 0;
				
			}
			
			
			if (l==1)
				oovsum=beta[0][l] * (double) n1 + beta[1][l] * (double)n2 + beta[2][l] * (double)unover3;
			
			cerr << beta[0][l] << " " << beta[1][l] << " " << beta[2][l] << "\n";
		}
		
		return 1;
	};
	
	
	
	int improvedshiftbeta::discount(ngram ng_,int size,double& fstar,double& lambda, int cv)
	{
		ngram ng(dict);
		ng.trans(ng_);
		
		//cerr << "size:" << size << " ng:|" << ng <<"|\n";
		
		if (size > 1) {
			
			ngram history=ng;
			
			//singleton pruning only on real counts!!
			if (ng.ckhisto(size) && get(history,size,size-1) && (history.freq > cv) &&
					((size < 3) || ((history.freq-cv) > prunethresh ))) { // no history pruning with corrected counts!
				
				int suc[3];
				suc[0]=succ1(history.link);
				suc[1]=succ2(history.link);
				suc[2]=history.succ-suc[0]-suc[1];
				
				
				if (get(ng,size,size) &&
						(!prunesingletons() || mfreq(ng,size)>1 || size<3) &&
						(!prunetopsingletons() || mfreq(ng,size)>1 || size<maxlevel())) {
					
					ng.freq=mfreq(ng,size);
					
					cv=(cv>ng.freq)?ng.freq:cv;
					
					if (ng.freq>cv) {
						
						double b=(ng.freq-cv>=3?beta[2][size]:beta[ng.freq-cv-1][size]);
						
						fstar=(double)((double)(ng.freq - cv) - b)/(double)(history.freq-cv);
						
						lambda=(beta[0][size] * suc[0] + beta[1][size] * suc[1] + beta[2][size] * suc[2])
						/
						(double)(history.freq-cv);
						
						if ((size>=3 && prunesingletons()) ||
								(size==maxlevel() && prunetopsingletons())) // correction due to frequency pruning
							
							lambda+=(double)(suc[0] * (1-beta[0][size])) / (double)(history.freq-cv);
						
					} else {
						// ng.freq==cv
						
						ng.freq>=3?suc[2]--:suc[ng.freq-1]--; //update successor stat
						
						fstar=0.0;
						lambda=(beta[0][size] * suc[0] + beta[1][size] * suc[1] + beta[2][size] * suc[2])
						/
						(double)(history.freq-cv);
						
						if ((size>=3 && prunesingletons()) ||
								(size==maxlevel() && prunetopsingletons())) // correction due to frequency pruning
							lambda+=(double)(suc[0] * (1-beta[0][size])) / (double)(history.freq-cv);
						
						ng.freq>=3?suc[2]++:suc[ng.freq-1]++; //resume successor stat
					}
				} else {
					fstar=0.0;
					lambda=(beta[0][size] * suc[0] + beta[1][size] * suc[1] + beta[2][size] * suc[2])
					/
					(double)(history.freq-cv);
					
					if ((size>=3 && prunesingletons()) ||
							(size==maxlevel() && prunetopsingletons())) // correction due to frequency pruning
						lambda+=(double)(suc[0] * (1-beta[0][size])) / (double)(history.freq-cv);
					
				}
				
				//cerr << "ngram :" << ng << "\n";
				
				
				if (*ng.wordp(1)==dict->oovcode()) {
					lambda+=fstar;
					fstar=0.0;
				} else {
					*ng.wordp(1)=dict->oovcode();
					if (get(ng,size,size)) {
						ng.freq=mfreq(ng,size);
						if ((!prunesingletons() || ng.freq>1 || size<3) &&
								(!prunetopsingletons() || ng.freq>1 || size<maxlevel())) {
							double b=(ng.freq>=3?beta[2][size]:beta[ng.freq-1][size]);
							lambda+=(double)(ng.freq - b)/(double)(history.freq-cv);
						}
					}
				}
			} else {
				fstar=0;
				lambda=1;
			}
		} else { // unigram case, no cross-validation
			fstar=unigr(ng);
			lambda=0;
		}
		
		return 1;
	}
	
//Symmetric Shiftbeta
int symshiftbeta::discount(ngram ng_,int size,double& fstar,double& lambda, int /* unused parameter: cv */)
{
  ngram ng(dict);
  ng.trans(ng_);

  //cerr << "size:" << size << " ng:|" << ng <<"|\n";

  // Pr(x/y)= max{(c([x,y])-beta)/(N Pr(y)),0} + lambda Pr(x)
  // lambda=#bigrams/N

  MY_ASSERT(size<=2); // only works with bigrams //

  if (size == 2) {

    //compute unigram probability of denominator
    ngram unig(dict,1);
    *unig.wordp(1)=*ng.wordp(2);
    double prunig=unigr(unig);

    //create symmetric bigram
    if (*ng.wordp(1) > *ng.wordp(2)) {
      int tmp=*ng.wordp(1);
      *ng.wordp(1)=*ng.wordp(2);
      *ng.wordp(2)=tmp;
    }

    lambda=beta[2] * (double) entries(2)/(double)totfreq();

    if (get(ng,2,2)) {
      fstar=(double)((double)ng.freq - beta[2])/
            (totfreq() * prunig);
    } else {
      fstar=0;
    }
  } else {
    fstar=unigr(ng);
    lambda=0.0;
  }
  return 1;
}

}//namespace irstlm


/*
main(int argc, char** argv){
  dictionary d(argv[1]);

  shiftbeta ilm(&d,argv[2],3);

  ngramtable test(&d,argv[2],3);
  ilm.train();
  cerr << "PP " << ilm.test(test) << "\n";

  ilm.savebin("newlm.lm",3);
}

*/
