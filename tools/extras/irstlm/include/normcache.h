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

#ifndef MF_NORMCACHE_H
#define MF_NORMCACHE_H

#include "dictionary.h"
#include "ngramtable.h"

// Normalization factors cache

class normcache
{
  dictionary* dict;
  ngramtable *ngt;
  double* cache[2];
  int cachesize[2];
  int maxcache[2];
  int hit;
  int miss;

public:
  normcache(dictionary* d);
  ~normcache() {
    delete [] cache[0];
    delete [] cache[1];
    delete ngt;
  }

  void expand(int i);
  double get(ngram ng,int size,double& value);
  double put(ngram ng,int size,double value);
  void stat();
};
#endif

