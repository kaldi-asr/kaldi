/*
 * wmatrix.cc
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2003-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/wmatrix.cc,v 1.10 2012/10/20 00:22:26 mcintyre Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <ctype.h>

#include "Trie.cc"
#include "Array.cc"
#include "FDiscount.h"
#include "FactoredVocab.h"
#include "Debug.h"
#include "hexdec.h"

#include "FNgramSpecs.h"

WidMatrix::WidMatrix()
{
  for (unsigned i = 0; i < maxNumFactors; i++) {
    wid_factors[i] = new VocabIndex[maxNumParentsPerChild+1];
  }
}

WidMatrix::~WidMatrix()
{
  for (unsigned i = 0; i < maxNumFactors; i++) {
    delete [] wid_factors[i];
  }
}

WordMatrix::WordMatrix()
{
  for (unsigned i = 0; i < maxNumFactors; i++) {
    word_factors[i] = new VocabString[maxNumParentsPerChild+1];
  }
}

WordMatrix::~WordMatrix()
{
  for (unsigned i = 0; i < maxNumFactors; i++) {
    delete [] word_factors[i];
  }
}

void
WordMatrix::print(FILE* f)
{
  for (unsigned i = 0; word_factors[i][FNGRAM_WORD_TAG_POS] != 0; i++) {
    fprintf(f,"%d ",i);
    for (unsigned j = 0; word_factors[i][j] != 0; j++) {
      fprintf(f,"%s ",word_factors[i][j]);
    }
    fprintf(f,"\n");
  }
}

