// hmm/simple-hmm.h

// Copyright 2016   Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
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

#ifndef KALDI_HMM_SIMPLE_HMM_H_
#define KALDI_HMM_SIMPLE_HMM_H_

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "itf/context-dep-itf.h"

namespace kaldi {

/**
 * SimpleHmm is a HMM that can be directly used for decoding in the
 * place of a HCLG.fst. It is implemented as a TransitionModel object 
 * with a single "phone". It is useful to model transitions 
 * for speech activity detection, music detection etc, and can
 * either be fixed or trained.
 * The 0-indexed "pdf-class" (also equal to the pdf-id) of a SimpleHmm state
 * uniquely maps to a 1-indexed class-id = pdf-id + 1, which is the symbol
 * used for decoding.
 **/
class SimpleHmm: public TransitionModel {
 public:
  SimpleHmm(const HmmTopology &hmm_topo): 
    ctx_dep_(hmm_topo) {
      Init(ctx_dep_, hmm_topo);
      CheckSimpleHmm();
    }

  SimpleHmm(): TransitionModel() { }
  
  void Read(std::istream &is, bool binary);  // note, no symbol table: topo object always read/written w/o symbols.

 private:
  void CheckSimpleHmm() const;

  // Implements a ContextDependencyInterface that defines a 
  // mapping from transition-id to pdf-class (pdf-id)
  class FakeContextDependency: public ContextDependencyInterface {
   public:
    int ContextWidth() const { return 1; }
    int CentralPosition() const { return 0; }
  
    bool Compute(const std::vector<int32> &phoneseq, int32 pdf_class,
                 int32 *pdf_id) const {
      if (phoneseq.size() == 1 && phoneseq[0] == 1) {
        *pdf_id = pdf_class;
        return true;
      }
      return false;
    }
    
    // Stores into *pdf_info:
    // [(1, 0), (1, 1), ..., (1, num_pdf)]
    void GetPdfInfo(
        const std::vector<int32> &phones,  // [ 1 ]
        const std::vector<int32> &num_pdf_classes,  // [0, num_pdfs]
        std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const;
  
    // Stores into *pdf_info:
    // [ [ ], [ [(0, 0)], [(1, 1)], ..., [(num_pdfs-1, num_pdfs-1)] ] ]
    void GetPdfInfo(
      const std::vector<int32> &phones,   // [ 1 ]
      const std::vector<std::vector<std::pair<int32, int32> > > &pdf_class_pairs,  // [ [ ], [ (0,0), (1,1), ..., (num_pdfs-1, num_pdfs-1) ] ]
      std::vector<std::vector<std::vector<std::pair<int32, int32> > > > *pdf_info)
      const;
    
    void Init(int32 num_pdfs) { num_pdfs_ = num_pdfs; }

    int32 NumPdfs() const { return num_pdfs_; }

    FakeContextDependency(const HmmTopology &topo) {
      const std::vector<int32> &phones = topo.GetPhones();
      KALDI_ASSERT(phones.size() == 1 && phones[0] == 1);
      num_pdfs_ = topo.NumPdfClasses(1);
    }

    FakeContextDependency(): num_pdfs_(0) { }

    ContextDependencyInterface* Copy() const { 
      FakeContextDependency *copy = new FakeContextDependency();
      copy->Init(num_pdfs_); 
      return copy;
    }

   private:
    int32 num_pdfs_;
  } ctx_dep_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(SimpleHmm);
};

} // end namespace kaldi

#endif  // KALDI_HMM_SIMPLE_HMM_H_
