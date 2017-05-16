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
#include <math.h>
#include "util.h"
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "doc.h"

using namespace std;

doc::doc(dictionary* d,char* docfname,bool use_null_word){
    mfstream df(docfname,ios::in);
    
    char header[100];
    df.getline(header,100);
    sscanf(header,"%d",&N);
    
    assert(N>0 && N < MAXDOCNUM);
    
    
    M=new int  [N];
    V=new int* [N];
    
    int eod=d->encode(d->EoD());
    int bod=d->encode(d->BoD());
    
    
    ngram ng(d);
    int n=0;  //track documents
    int m=0;  //track document length
    int w=0;  //track words in doc
    
    int tmp[MAXDOCLEN];
    
    while (n<N && df >> ng)
        if (ng.size>0){
            w=*ng.wordp(1);
            if (w==bod){
                if (use_null_word){
                    ng.size=1; //use <d> as NULL word
                }else{
                    ng.size=0; //skip <d>
                    continue;
                }
            }
            if (w==eod && m>0){
                M[n]=m;  //length of n-th document
                V[n]=new int[m];
                memcpy(V[n],tmp,m * sizeof(int));
                m=0;
                n++;
                continue;
            }
            
            if (m < MAXDOCLEN) tmp[m++]=w;
            if (m==MAXDOCLEN) {cerr<< "warn: clipping long document (line " << n << " )\n";exit(1);};
        }
    
    cerr << "uploaded " << n << " documents\n";
    
    
};

doc::~doc(){
    cerr << "releasing document storage\n";
    for (int i=0;i<N;i++) delete [] V[i];
    delete [] M;  delete [] V;
}


int doc::numdoc(){
    return N;
}

int doc::doclen( int index){
    assert(index>=0 && index < N);
    return M[index];
}

int doc::docword( int docindex, int wordindex){
    assert(wordindex>=0 && wordindex<doclen(docindex));
    return V[docindex][wordindex];
}








