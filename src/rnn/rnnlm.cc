// rnn/rnn-rescore-kaldi.cc

// Copyright 2009-2011  Stefan Kombrink 

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <float.h>

#include "base/kaldi-common.h"
#include "base/kaldi-error.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fst/fst.h"
#include "fst/mutable-fst.h"
#include "fst/queue.h"
#include "fst/rmepsilon.h"
#include "fst/dfs-visit.h"
#include "lat/kaldi-lattice.h"

#include "rnn/rnnlm.h"

void RnnLm::Read(std::istream& in, bool binary) {
  using namespace kaldi;

  ExpectMarker(in,binary,"<rnnlm_v2.0>");
  ExpectMarker(in,binary,"<v1>"); in >> V1_;
  ExpectMarker(in,binary,"<u1>"); in >> U1_; U1_.Transpose();
  ExpectMarker(in,binary,"<b1>"); in >> b1_;
  ExpectMarker(in,binary,"<w2>"); in >> W2_; W2_.Transpose();
  ExpectMarker(in,binary,"<b2>"); in >> b2_;
  ExpectMarker(in,binary,"<cl>"); in >> Cl_; Cl_.Transpose();
  ExpectMarker(in,binary,"<cl_b>"); in >> cl_b_;
  ExpectMarker(in,binary,"<classes>");
  ReadIntegerVector(in,binary,&int2class_);

  // determine range for classes 
  // THIS ASSUMES CLASSES ARE ENUMERABLE AND INCREASING!!
     
  int32 cl;
  for (int32 i=0;i<VocabSize();i++) {
    cl=int2class_[i];
    if (class2minint_.count(cl)==0) class2minint_[cl]=i; // mapping class -> start int
    class2maxint_[cl]=i; // mapping class -> max int
  }

  ExpectMarker(in,binary,"<words>");
  std::string wrd;

  // read vocabulary
  for (int32 i=0;i<VocabSize();i++) {
    ReadMarker(in,binary,&wrd);
    word2int_[wrd]=i;
    int2word_[i]=wrd;
  }
  
  // prepare activation layers
  h_.Resize(HiddenSize(),kSetZero);
  y_.Resize(VocabSize(),kSetZero);
  cl_.Resize(ClassSize(),kSetZero);       
};

// this is required to convert openfst symbol IDs to RNN IDs
void RnnLm::SetLatticeSymbols(const fst::SymbolTable& symtab) {
  wordsym_=&symtab;
  int32 j,ii,oov=-1;
  for (int32 i=0;i<symtab.AvailableKey();i++) {
    ii=symtab.GetNthKey(i);
    std::string w=symtab.Find(ii);
      if (w=="</s>") {intlat2intrnn[ii]=0;intrnn2intlat[0]=ii;continue;}
      if (word2int_.find(w)!=word2int_.end())
        j=word2int_[w];
      else 
        j=--oov;

      intlat2intrnn[ii]=j; intrnn2intlat[j]=ii;
    } 
}

kaldi::BaseFloat RnnLm::Propagate(int32 lastW,int32 w,KaldiVector* hOut,const KaldiVector& hIn) {
  using namespace kaldi;

  // create a local copy (for cases like aliasing!)
  KaldiVector h;
  h.Resize(hIn.Dim(),kUndefined);
  h.CopyFromVec(hIn);

  // update hidden layer
  hOut->CopyFromVec(V1_.Row(lastW));

  hOut->AddMatVec(-1.0,U1_,kNoTrans,h,0.0);       // h(t)=U1*h(t-1)
  if (IsIV(lastW)) (*hOut).AddVec(-1.0,V1_.Row(lastW)); // h(t)=h(t)+V1*w-1(t)

  // activate using sigmoid and keep as updated h(t)
  hOut->ApplyExp();hOut->Add(1.0);hOut->InvertElements();

  if (IsOOV(w)) {
    oovcnt_++;
    return OOVPenalty();
  } else {
    ivcnt_++;

    // evaluate classes: cl(t)=Cl*h(t)
    cl_.AddMatVec(1.0,Cl_,kNoTrans,*hOut,0.0);

    // activate using softmax 
    cl_.ApplySoftMax();

    int32 b=class2minint_[int2class_[w]];
    int32 n=class2maxint_[int2class_[w]]-b+1;

    // determine distribution of class of the predicted word
    // activate class part of the word layer (softmax)
    SubVector<kaldi::BaseFloat> y_part(y_,b,n);
    SubMatrix<kaldi::BaseFloat> W2_part(W2_,b,n,0,W2_.NumCols());

    y_part.AddMatVec(1.0,W2_part,kNoTrans,*hOut,0.0);
    // apply softmax
    y_part.ApplySoftMax();

    return -log(y_(w)*cl_(int2class_[w]));
  }
  return 0; 
}


void RnnLm::TreeTraverse(kaldi::CompactLattice* lat,const KaldiVector h) {
  // deal with the head of the tree explicitely, leave its scores untouched!
  for (fst::MutableArcIterator<kaldi::CompactLattice> aiter(lat,lat->Start()); !aiter.Done(); aiter.Next()) { // follow <eps>
    for (fst::MutableArcIterator<kaldi::CompactLattice> a2iter(lat,aiter.Value().nextstate); !a2iter.Done(); a2iter.Next()) { // follow <s>
      TreeTraverseRec(lat,a2iter.Value().nextstate,h,0); // call the visitor method with the given history and preceding <s> recursively
    }
  }
}

void RnnLm::TreeTraverseRec(kaldi::CompactLattice* lat, fst::StdFst::StateId i,const KaldiVector& lasth,int32 lastW) {
  KaldiVector h(HiddenSize());
  for (fst::MutableArcIterator<kaldi::CompactLattice> aiter(lat, i); !aiter.Done(); aiter.Next()) {
    int32 w=intlat2intrnn[aiter.Value().olabel];  
    kaldi::BaseFloat rnns=Propagate(lastW,w,&h,lasth);
    kaldi::BaseFloat s=aiter.Value().weight.Weight().Value1();
    kaldi::BaseFloat ams=aiter.Value().weight.Weight().Value2();
      
    kaldi::CompactLatticeArc newarc = aiter.Value();
    kaldi::CompactLatticeWeight newwgt(kaldi::LatticeWeight(s+rnns*Scale(),ams),aiter.Value().weight.String()); 
    newarc.weight=newwgt;
    aiter.SetValue(newarc);

    TreeTraverseRec(lat,aiter.Value().nextstate,h,w);
  }
}
