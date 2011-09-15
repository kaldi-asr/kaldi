// rnn/rnnlm.h

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

#ifndef RNNLM_H
#define RNNLM_H 

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

class RnnLm{
  public:
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  typedef kaldi::Matrix<kaldi::BaseFloat> KaldiMatrix;
  typedef kaldi::Vector<kaldi::BaseFloat> KaldiVector;


  private:
    KaldiMatrix V1_,U1_,W2_,Cl_; // weights
    KaldiVector b1_,b2_,cl_b_;   // biases TODO not yet used!!!
    KaldiVector h_,cl_,y_;       // activations

    std::map<int32,string> int2word_;                  // maps from rnn ints to word strings
    std::map<string,int32> word2int_;                  // maps from word strings to rnn ints
    std::map<int32,int32> intlat2intrnn;               // maps from ints in lattices to ints in RNN
    std::map<int32,int32> intrnn2intlat;               // maps from ints in RNN to ins in lattices
    std::vector<int32> int2class_;                     // mapping words (integer) to their classes
    std::map<int32,int32> class2minint_,class2maxint_; // determines the range of a class of words in the output layer

    // others...
    kaldi::BaseFloat OOVPenalty_;
    kaldi::BaseFloat Scale_;
    const fst::SymbolTable* wordsym_;

    int32 ivcnt_;
    int32 oovcnt_;

  public:
    inline int32 VocabSize() const { return V1_.NumRows(); }
    inline int32 HiddenSize() const { return V1_.NumCols(); }
    inline int32 ClassSize() const { return Cl_.NumRows(); }

    inline int32 IVProcessed() const { return ivcnt_; }
    inline int32 OOVProcessed() const { return oovcnt_; }

    inline bool IsIV(int32 w) const { return w>=0; }
    inline bool IsOOV(int32 w) const { return !IsIV(w); }

    kaldi::BaseFloat OOVPenalty() const { return OOVPenalty_; }
    void SetOOVPenalty( kaldi::BaseFloat oovp ) { OOVPenalty_=oovp; }

    kaldi::BaseFloat Scale() const { return Scale_; }
    void SetScale(kaldi::BaseFloat l) { Scale_=l; }
    void SetLatticeSymbols(const fst::SymbolTable& symtab);
    void Read(std::istream& in, bool binary);

    RnnLm() : ivcnt_(0),oovcnt_(0)  {
    }

    virtual ~RnnLm() {}

    kaldi::BaseFloat Propagate(int32 lastW,int32 w,KaldiVector* hOut,const KaldiVector& hIn); 

    void TreeTraverse(kaldi::CompactLattice* lat,const KaldiVector h);

  protected:

  void TreeTraverseRec(kaldi::CompactLattice* lat, fst::StdFst::StateId i,const KaldiVector& lasth,int32 lastW);
};

#endif
