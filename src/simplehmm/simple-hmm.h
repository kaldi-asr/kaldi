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
    
    void GetPdfInfo(
        const std::vector<int32> &phones,  // list of phones
        const std::vector<int32> &num_pdf_classes,  // indexed by phone,
        std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const;
  
    void GetPdfInfo(
      const std::vector<int32> &phones,
      const std::vector<std::vector<std::pair<int32, int32> > > &pdf_class_pairs,
      std::vector<std::vector<std::vector<std::pair<int32, int32> > > > *pdf_info)
      const;
    
    void Init(int32 num_pdfs) { num_pdfs_ = num_pdfs; }

    int32 NumPdfs() const { return num_pdfs_; }

    FakeContextDependency(const HmmTopology &topo) {
      KALDI_ASSERT(topo.GetPhones().size() == 1);
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
