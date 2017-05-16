/******************************************************************************
 IrstLM: IRST Language Model Toolkit, compile LM
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
//class managing a collection of documents for PLSA


#define MAXDOCLEN 500
#define MAXDOCNUM 1000000000

class doc
{
  int  N;      //number of docs
  int *M;      //number of words per document
  int **V;     //words in current doc

public:
    
  doc(dictionary* d,char* docfname,bool use_null_word=false);
  ~doc();

  int numdoc();
  int doclen(int index);
  int docword(int docindex, int wordindex);
    
};


