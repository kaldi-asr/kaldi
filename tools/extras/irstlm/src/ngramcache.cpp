// $Id: ngramcache.cpp 3679 2010-10-13 09:10:01Z bertoldi $

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
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <sstream>
#include <string>
#include "math.h"
#include "mempool.h"
#include "htable.h"
#include "lmtable.h"
#include "util.h"

#include "ngramcache.h"

using namespace std;
	
void ngramcache::print (const int* ngp)
{
  std::cerr << "ngp: size:" << ngsize << "|";
  for (int i=0; i<ngsize; i++)
    std::cerr << " " << ngp[i];
  std::cerr << " |\n";
}

ngramcache::ngramcache(int n,int size,int maxentries,float lf)
{
  if (lf<=0.0) lf=NGRAMCACHE_LOAD_FACTOR;
  load_factor=lf;
  ngsize=n;
  infosize=size;
  maxn=maxentries;
  entries=0;
  ht=new htable<int*>((size_t) (maxn/load_factor), ngsize * sizeof(int)); //decrease the lower load factor to reduce collision
  mp=new mempool(ngsize * sizeof(int)+infosize,MP_BLOCK_SIZE);
  accesses=0;
  hits=0;
};

ngramcache::~ngramcache()
{
  delete ht;
  delete mp;
};


//resize cache to specified number of entries
void ngramcache::reset(int n)
{
  //ht->stat();
  delete ht;
  delete mp;
  if (n>0) maxn=n;
  ht=new htable<int*> ((size_t) (maxn/load_factor), ngsize * sizeof(int)); //decrease the lower load factor to reduce collision
  mp=new mempool(ngsize * sizeof(int)+infosize,MP_BLOCK_SIZE);
  entries=0;
};

char* ngramcache::get(const int* ngp,char*& info)
{
  char* found;

  accesses++;
  if ((found=(char*) ht->find((int *)ngp))) {
    memcpy(&info,found+ngsize*sizeof(int),infosize);
    hits++;
  }

  return found;
};

char* ngramcache::get(const int* ngp,double& info)
{
  char *found;

  accesses++;
  if ((found=(char*) ht->find((int *)ngp))) {
    memcpy(&info,found+ngsize*sizeof(int),infosize);
    hits++;
  };

  return found;
};

char* ngramcache::get(const int* ngp,prob_and_state_t& info)
{
  char *found;

  accesses++;
  if ((found=(char*) ht->find((int *)ngp)))
	{
    memcpy(&info,found+ngsize*sizeof(int),infosize);
    hits++;
  }
  return found;
};

int ngramcache::add(const int* ngp,const char*& info)
{
  char* entry=mp->allocate();
  memcpy(entry,(char*) ngp,sizeof(int) * ngsize);
  memcpy(entry + ngsize * sizeof(int),&info,infosize);
  char* found=(char*)ht->insert((int *)entry);
  MY_ASSERT(found == entry); //false if key is already inside
  entries++;
  return 1;
};

int ngramcache::add(const int* ngp,const double& info)
{
  char* entry=mp->allocate();
  memcpy(entry,(char*) ngp,sizeof(int) * ngsize);
  memcpy(entry + ngsize * sizeof(int),&info,infosize);
  char *found=(char*) ht->insert((int *)entry);
  MY_ASSERT(found == entry); //false if key is already inside
  entries++;
  return 1;
};

int ngramcache::add(const int* ngp,const prob_and_state_t& info)
{
  char* entry=mp->allocate();
  memcpy(entry,(char*) ngp,sizeof(int) * ngsize);
  memcpy(entry + ngsize * sizeof(int),&info,infosize);
  char *found=(char*) ht->insert((int *)entry);
  MY_ASSERT(found == entry); //false if key is already inside
  entries++;
  return 1;
};


void ngramcache::stat() const
{
  std::cout << "ngramcache stats: entries=" << entries << " acc=" << accesses << " hits=" << hits
       << " ht.used= " << ht->used() << " mp.used= " << mp->used() << " mp.wasted= " << mp->wasted() << "\n";
};

