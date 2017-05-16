// $Id: n_gram.h 3461 2010-08-27 10:17:34Z bertoldi $

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

// n-gram tables
// by M. Federico
// Copyright Marcello Federico, ITC-irst, 1998

#ifndef MF_NGRAM_H
#define MF_NGRAM_H

#include <fstream>
#include "util.h"
#include "dictionary.h"

#ifndef MYMAXNGRAM
#define MYMAXNGRAM 20
#endif
#define MAX_NGRAM MYMAXNGRAM

class dictionary;

//typedef int code;

class ngram
{
  int  word[MAX_NGRAM];  //encoded ngram
public:
  dictionary *dict;      // dictionary
  char* link;            // ngram-tree pointer
  char* succlink;        // pointer to the first successor
  int   midx[MAX_NGRAM]; // ngram-tree scan pointer
  char* path[MAX_NGRAM]; // path in the ngram-trie
  float bowv[MAX_NGRAM]; // vector of bow found in the trie

  int    lev;            // ngram-tree level
  int   size;            // ngram size
  long long   freq;      // ngram frequency or integer prob
  int   succ;            // number of successors
  float   bow;           // back-off weight
  float   prob;          // probability

  unsigned char info;    // ngram-tree info flags
  unsigned char pinfo;   // ngram-tree parent info flags
  int  isym;             // last interruption symbol

  ngram(dictionary* d,int sz=0);
  ngram(const ngram& ng);

  inline int *wordp() { // n-gram pointer
    return wordp(size);
  }
  inline int *wordp(int k) { // n-gram pointer
    return size>=k?&word[MAX_NGRAM-k]:0;
  }
  inline const int *wordp() const { // n-gram pointer
    return wordp(size);
  }
  inline const int *wordp(int k) const { // n-gram pointer
    return size>=k?&word[MAX_NGRAM-k]:0;
  }

  int containsWord(const char* s,int lev);

  void trans(const ngram& ng);
  void invert (const ngram& ng);
  void shift ();
  void shift (int sz);

  friend std::ifstream& operator>> (std::ifstream& fi,ngram& ng);
  friend std::ofstream& operator<< (std::ofstream& fi,ngram& ng);
  friend std::istream& operator>> (std::istream& fi,ngram& ng);
  friend std::ostream& operator<< (std::ostream& fi,ngram& ng);
	
	bool operator==(const ngram &compare) const;
	bool operator!=(const ngram &compare) const;
	
	/*
  friend bool operator==(const ngram &compare) const {
    if ( size != compare.size || dict != compare.dict)
      return false;
    else
      for (int i=size; i>0; i--)
        if (word[MAX_NGRAM-i] != compare.word[MAX_NGRAM-i])
          return false;
    return true;
  }

  inline bool operator!=(const ngram &compare) const {
    if ( size != compare.size || dict != compare.dict)
      return true;
    else
      for (int i=size; i>0; i--)
        if (word[MAX_NGRAM-i] != compare.word[MAX_NGRAM-i])
          return true;
    return false;
  }
*/
  int ckhisto(int sz);

  int pushc(int c);
  int pushc(int* codes, int sz);
  int pushw(const char* w);

  //~ngram();
};

#endif



