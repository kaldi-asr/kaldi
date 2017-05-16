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

// Mixture of linear interpolation LMs

#ifndef LM_MIXTURE
#define LM_MIXTURE

namespace irstlm {
	
class mixture: public mdiadaptlm
{
  double** l[MAX_NGRAM]; //interpolation parameters
  int* pm; //parameter mappings
  int  pmax; //#parameters
  int k1,k2; //two thresholds
  int  numslm;
  int prunethresh;
  interplm** sublm;
  char *ipfname;
  char *opfname;


  double reldist(double *l1,double *l2,int n);
  int genpmap();
  int pmap(ngram ng,int lev);
public:

  bool usefulltable;

  mixture(bool fulltable,char *sublminfo,int depth,int prunefreq=0,char* ipfile=NULL,char* opfile=NULL);

  int train();

  int savepar(char* opf);
  int loadpar(char* opf);

  inline int dub() {
    return dict->dub();
  }
	
  inline int dub(int value) {
    for (int i=0; i<numslm; i++) {
      sublm[i]->dub(value);
    }
    return dict->dub(value);
  }

  void settying(int a,int b) {
    k1=a;
    k2=b;
  }
  int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);



  ~mixture(){
	 
	  for (int i=0;i<=lmsize();i++){
		   for (int j=0; j<pmax; j++) free(l[i][j]);
  		   free(l[i]);
	  }
	
	 for (int i=0;i<numslm;i++) delete(sublm[i]);

 }

  //this extension builds a commong ngramtable on demand
  int get(ngram& ng,int n,int lev);

};
	
}//namespace irstlm
#endif




