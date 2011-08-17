// gmm/mmie-diag-gmm.cc

// Copyright 2009-2011  Arnab Ghoshal, Petr Motlicek

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

#include <algorithm>  // for std::max
#include <string>
#include <vector>

#include "gmm/diag-gmm.h"
#include "gmm/mmie-diag-gmm.h"


namespace kaldi {


void MmieDiagGmm::Resize(int32 num_comp, int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentFlags(flags);
  occupancy_.Resize(num_comp);
  if (flags_ & kGmmMeans)
    mean_accumulator_.Resize(num_comp, dim);
  else
    mean_accumulator_.Resize(0, 0);
  if (flags_ & kGmmVariances)
    variance_accumulator_.Resize(num_comp, dim);
  else
    variance_accumulator_.Resize(0, 0);
}


GmmFlagsType MmieDiagGmm::AugmentFlags(GmmFlagsType f) {
  KALDI_ASSERT((f & ~kGmmAll) == 0);  // make sure only valid flags are present
  if (f & kGmmVariances) f |= kGmmMeans;
  if (f & kGmmMeans) f |= kGmmWeights;
  KALDI_ASSERT(f & kGmmWeights);  // make sure zero-stats will be accumulated
  return f;
}


void MmieDiagGmm::SubtractAccumulators(const AccumDiagGmm& num_acc,
                            const AccumDiagGmm& den_acc,
                            const MmieDiagGmmOptions& opts){

  //KALDI_ASSERT(num_acc.NumGauss() == den_acc.NumGauss && num_acc.Dim() == den_acc.Dim());
  //std::cout << "NumGauss: " << num_acc.NumGauss() << " " << den_acc.NumGauss() << " " << num_comp_ << '\n';
  KALDI_ASSERT(num_acc.NumGauss() == num_comp_ && num_acc.Dim() == dim_);
  KALDI_ASSERT(den_acc.NumGauss() == num_comp_ && den_acc.Dim() == dim_);
  
  // Occupancy
  occupancy_.AddVec( 1.0, num_acc.occupancy());
  occupancy_.AddVec(-1.0, den_acc.occupancy());

  // Subtract Mean
  mean_accumulator_.AddMat( 1.0, num_acc.mean_accumulator(), kNoTrans);
  mean_accumulator_.AddMat(-1.0, den_acc.mean_accumulator(), kNoTrans);

  // Subtract variance 
  variance_accumulator_.AddMat( 1.0, num_acc.variance_accumulator(), kNoTrans);
  variance_accumulator_.AddMat(-1.0, den_acc.variance_accumulator(), kNoTrans);
}


void MmieDiagGmm::Update(const MmieDiagGmmOptions &config,
              GmmFlagsType flags,
              DiagGmm *gmm,
              BaseFloat *obj_change_out,
              BaseFloat *count_out) const {

  if (flags_ & ~flags)
    KALDI_ERR << "Flags in argument do not match the active accumulators";


  KALDI_ASSERT(gmm->NumGauss() == (num_comp_));
  if (flags_ & kGmmMeans)
    KALDI_ASSERT(dim_ == mean_accumulator_.NumCols());



std::cout << "Petr\n";
}



//BaseFloat MmieDiagGmm::MmiObjective(const DiagGmm& gmm) const {
//}

}  // End of namespace kaldi
