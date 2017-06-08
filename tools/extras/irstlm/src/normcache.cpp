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


#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "ngramtable.h"
#include "normcache.h"

using namespace std;

// Normalization factors cache

normcache::normcache(dictionary* d)
{
  dict=d;
  //trigram and bigram normalization cache

  //ngt=new ngramtable(NULL,2,NULL,NULL,0,0,NULL,0,LEAFPROB);
  ngt=new ngramtable(NULL,2,NULL,NULL,NULL,0,0,NULL,0,LEAFPROB);

  maxcache[0]=d->size();//unigram cache
  maxcache[1]=d->size();//bigram cache

  cache[0]=new double[maxcache[0]];
  cache[1]=new double[maxcache[1]];

  for (int i=0; i<d->size(); i++)
    cache[0][i]=cache[1][i]=0.0;

  cachesize[0]=cachesize[1]=0;
  hit=miss=0;
}

void normcache::expand(int n)
{

  int step=100000;
  cerr << "Expanding cache ...\n";
  double *newcache=new double[maxcache[n]+step];
  memcpy(newcache,cache[n],sizeof(double)*maxcache[n]);
  delete [] cache[n];
  cache[n]=newcache;
  for (int i=0; i<step; i++)
    cache[n][maxcache[n]+i]=0;
  maxcache[n]+=step;
};


double normcache::get(ngram ng,int size,double& value)
{

  if (size==2) {
    if (*ng.wordp(2) < cachesize[0])
      return value=cache[0][*ng.wordp(2)];
    else
      return value=0;
  } else if (size==3) {
    if (ngt->get(ng,size,size-1)) {
      hit++;
      //      cerr << "hit " << ng << "\n";
      return value=cache[1][ng.freq];
    } else {
      miss++;
      return value=0;
    }
  }
  return 0;
}

double normcache::put(ngram ng,int size,double value)
{

  if (size==2) {
    if (*ng.wordp(2)>= maxcache[0]) expand(0);
    cache[0][*ng.wordp(2)]=value;
    cachesize[0]++;
    return value;
  } else if (size==3) {
    if (ngt->get(ng,size,size-1))
      return cache[1][ng.freq]=value;
    else {
      ngram histo(dict,2);
      *histo.wordp(1)=*ng.wordp(2);
      *histo.wordp(2)=*ng.wordp(3);
      histo.freq=cachesize[1]++;
      if (cachesize[1]==maxcache[1]) expand(1);
      ngt->put(histo);
      return cache[1][histo.freq]=value;
    }
  }
  return 0;
}

void normcache::stat()
{
  std::cout << "misses " << miss << ", hits " << hit << "\n";
}





