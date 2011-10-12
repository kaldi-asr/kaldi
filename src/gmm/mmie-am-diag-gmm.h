// gmm/mmie-am-diag-gmm.h

// Copyright 2009-2011  
// Author:  Petr Motlicek

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


#ifndef KALDI_GMM_MMIE_AM_DIAG_GMM_H_
#define KALDI_GMM_MMIE_AM_DIAG_GMM_H_ 1

#include <vector>

#include "gmm/am-diag-gmm.h"
#include "gmm/mmie-diag-gmm.h"
#include "gmm/mle-diag-gmm.h"


namespace kaldi {

class MmieAccumAmDiagGmm {
 public:
  MmieAccumAmDiagGmm() {}
  ~MmieAccumAmDiagGmm();

  void ReadNum(std::istream &in_stream, bool binary, bool add);
  void ReadDen(std::istream &in_stream, bool binary, bool add);
  void WriteNum(std::ostream &out_stream, bool binary) const;
  void WriteDen(std::ostream &out_stream, bool binary) const;

  /// Initializes accumulators for each GMM based on the number of components
  /// and dimension.
  void Init(const AmDiagGmm &model, GmmFlagsType flags);
  /// Initialization using different dimension than model.
  void Init(const AmDiagGmm &model, int32 dim, GmmFlagsType flags);
  void SetZero(GmmFlagsType flags);

  int32 NumAccs() { return num_accumulators_.size(); }

  int32 NumAccs() const { return num_accumulators_.size(); }

  AccumDiagGmm& GetNumAcc(int32 index) const;
  AccumDiagGmm& GetDenAcc(int32 index) const;

  void CopyToNumAcc(int32 index);
  BaseFloat TotNumCount();
  BaseFloat TotDenCount();
 private:
  /// MMIE accumulators and update methods for the GMMs
  std::vector<AccumDiagGmm*> num_accumulators_;
  std::vector<AccumDiagGmm*> den_accumulators_;


  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(MmieAccumAmDiagGmm);
};


/// for computing the maximum-likelihood estimates of the parameters of
/// an acoustic model that uses diagonal Gaussian mixture models as emission densities.
void MmieAmDiagGmmUpdate(const MmieDiagGmmOptions &config, 
                         const MmieAccumAmDiagGmm &mmieamdiaggmm_acc,
                         GmmFlagsType flags, 
                         AmDiagGmm *am_gmm,
                         BaseFloat *auxf_change_gauss,
                         BaseFloat *auxf_change_weight,
                         BaseFloat *count_out,
                         int32 *num_floored_out);

}  // End namespace kaldi


#endif  // KALDI_GMM_MMIE_AM_DIAG_GMM_H_
