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

namespace irstlm {

// Non linear Shift based interpolated LMs

class shiftone: public mdiadaptlm
{
protected:
  int prunethresh;
  double beta;
public:
  shiftone(char* ngtfile,int depth=0,int prunefreq=0,TABLETYPE tt=SHIFTBETA_B);
  int train();
  int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);
  ~shiftone() {}
};


class shiftbeta: public mdiadaptlm
{
protected:
  int prunethresh;
  double* beta;

public:
  shiftbeta(char* ngtfile,int depth=0,int prunefreq=0,double beta=-1,TABLETYPE tt=SHIFTBETA_B);
  int train();
  int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);
  ~shiftbeta() {
    delete [] beta;
  }

};


class symshiftbeta: public shiftbeta
{
public:
  symshiftbeta(char* ngtfile,int depth=0,int prunefreq=0,double beta=-1):
    shiftbeta(ngtfile,depth,prunefreq,beta) {}
  int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);
};

	
	class improvedkneserney: public mdiadaptlm
	{
	protected:
		int prunethresh;
		double beta[3][MAX_NGRAM];
		ngramtable* tb[MAX_NGRAM];
		
		double oovsum;
		
	public:
		improvedkneserney(char* ngtfile,int depth=0,int prunefreq=0,TABLETYPE tt=IMPROVEDKNESERNEY_B);
		int train();
		int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);
		
		~improvedkneserney() {}
		
		int mfreq(ngram& ng,int l) {
			return (l<lmsize()?getfreq(ng.link,ng.pinfo,1):ng.freq);
		}
		
		double unigrIKN(ngram ng);
		inline double unigr(ngram ng){ return unigrIKN(ng); };		
	};
	
class improvedshiftbeta: public mdiadaptlm
{
protected:
  int prunethresh;
  double beta[3][MAX_NGRAM];
  ngramtable* tb[MAX_NGRAM];

  double oovsum;

public:
  improvedshiftbeta(char* ngtfile,int depth=0,int prunefreq=0,TABLETYPE tt=IMPROVEDSHIFTBETA_B);
  int train();
  int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);

  ~improvedshiftbeta() {}

  inline int mfreq(ngram& ng,int /*NOT_USED l*/) { return ng.freq; }

};
	
}//namespace irstlm
