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
  num_occupancy_.Resize(num_comp);
  den_occupancy_.Resize(num_comp);
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


void MmieDiagGmm::SubtractAccumulatorsISmoothing(const AccumDiagGmm& num_acc,
                            const AccumDiagGmm& den_acc,
                            const MmieDiagGmmOptions& opts){

  //KALDI_ASSERT(num_acc.NumGauss() == den_acc.NumGauss && num_acc.Dim() == den_acc.Dim());
  //std::cout << "NumGauss: " << num_acc.NumGauss() << " " << den_acc.NumGauss() << " " << num_comp_ << '\n';
  KALDI_ASSERT(num_acc.NumGauss() == num_comp_ && num_acc.Dim() == dim_);
  KALDI_ASSERT(den_acc.NumGauss() == num_comp_ && den_acc.Dim() == dim_);
  
  int32 Tau = 100;

  // no subracting occs, just copy them to local vars
  num_occupancy_.CopyFromVec(num_acc.occupancy());
  den_occupancy_.CopyFromVec(den_acc.occupancy());

  // Copy nums to private vars
  mean_accumulator_.CopyFromMat(num_acc.mean_accumulator(), kNoTrans);
  variance_accumulator_.CopyFromMat(num_acc.variance_accumulator(), kNoTrans);


  /*
  std::cout << "OCC: " << num_occupancy_.Dim() << '\n';
  num_occupancy_(0) = 10;
  num_occupancy_(1) = 20;
  mean_accumulator_(0,0) = 2;
  mean_accumulator_(0,1) = 3;
  mean_accumulator_(1,0) = 4;
  mean_accumulator_(1,1) = 5;
  variance_accumulator_(0,0) = 2;
  variance_accumulator_(0,1) = 3;
  variance_accumulator_(1,0) = 4;
  variance_accumulator_(1,1) = 5;
  */

  // I- smoothing
  for (int32 g = 0; g < num_comp_; g++) {
    double occ = num_occupancy_(g);
    std::cout << "M1: " << mean_accumulator_.Row(g) << '\n';
    std::cout << "Occ: " << occ << '\n';
    mean_accumulator_.Row(g).AddVec(Tau/occ, mean_accumulator_.Row(g));
    variance_accumulator_.Row(g).AddVec(Tau/occ, variance_accumulator_.Row(g));
    std::cout << "M2: " << mean_accumulator_.Row(g) << '\n';
    std::cout << "M22: " << den_acc.mean_accumulator().Row(g) << '\n';

  }

  num_occupancy_.Add(Tau);
  occupancy_.CopyFromVec(num_occupancy_);
  occupancy_.AddVec(-1.0, den_occupancy_);

  // Subtract den from smoothed num
  mean_accumulator_.AddMat(-1.0, den_acc.mean_accumulator(), kNoTrans);
  variance_accumulator_.AddMat(-1.0, den_acc.variance_accumulator(), kNoTrans);

  for (int32 g = 0; g < num_comp_; g++) {
    std::cout << "M3: " << mean_accumulator_.Row(g) << '\n';
  }
  std::cout << "Dim: " << mean_accumulator_.NumRows() << " " << mean_accumulator_.NumCols() << '\n';
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

  
      //Vector<double> hlp;
     // hlp.CopyFromVec(num_occupancy_);
     // hlp.AddVec(-1.0, den_occupancy_);

  //occ_sum
  double occ_sum = occupancy_.Sum();
  
  int32 num_comp = num_comp_;
  int32 dim = dim_;
 
  int32 tot_floored = 0, gauss_floored = 0;
  gmm->ComputeGconsts();
  
  DiagGmmNormal *diaggmmnormal = new DiagGmmNormal();
  diaggmmnormal->CopyFromDiagGmm(*gmm);

  Vector<double> D(num_comp);
  for (int32 g = 0; g < num_comp; g++)
       D(g) = den_occupancy_(g); //* 2.0;

  std::vector<int32> removed_components;
  for (int32 g = 0; g < num_comp; g++) {
    double occ = occupancy_(g) + D(g);
    double prob;
    if (occ_sum != 0.0)
      prob = occ / occ_sum;
    else
      prob = 1.0 / num_comp;

    if (flags & kGmmWeights) diaggmmnormal->weights_(g) = prob;
    if ((flags & kGmmMeans) && !(flags & kGmmVariances)) {
        // updating means but not vars.
        Vector<double> mean(dim);
        mean.CopyFromVec(diaggmmnormal->means_.Row(g));
        std::cout << "G: " << mean << '\n';
        std::cout << "D: " << D << '\n';
        std::cout << "Occ: " << occ << '\n';

        mean.Scale(D(g));
        mean.AddVec(1.0, mean_accumulator_.Row(g));
        mean.Scale(1.0 / occ);
        diaggmmnormal->means_.CopyRowFromVec(mean, g);
        std::cout << "U: " << mean << '\n';
  
    } else if ((flags & kGmmMeans) && (flags & kGmmVariances)) {
        // updating means and vars.
        
        // updating D learning param, not ready yet
        Vector<double> mean(dim), var(dim);
        while (1){
          mean.CopyFromVec(diaggmmnormal->means_.Row(g));
          mean.Scale(D(g));
          mean.AddVec(1.0, mean_accumulator_.Row(g));
          mean.Scale(1.0 / occ);
          diaggmmnormal->means_.CopyRowFromVec(mean, g);

          //var.CopyFromVec(variance_accumulator_.Row(g));
          var.CopyFromVec(diaggmmnormal->means_.Row(g));
          var.ApplyPow(2.0);
          var.AddVec(1.0, diaggmmnormal->vars_.Row(g));
          var.Scale(D(g));
          var.AddVec(1.0, variance_accumulator_.Row(g));
          var.Scale(1.0 / occ);
          mean.ApplyPow(2.0);
          var.AddVec(-1.0, mean);
          diaggmmnormal->vars_.CopyRowFromVec(var, g);
          std::cout << "Min D: " << var.Min() << '\n';
    
          if (var.Min() > 0){
            break;
          } else {
            D(g) *= 1.1; 
          }
        }
    }

  }

std::cout << "Petr\n";
}


BaseFloat ComputeD(const DiagGmm& old_gmm, int32 mix_index, BaseFloat ebw_e){

}



//BaseFloat MmieDiagGmm::MmiObjective(const DiagGmm& gmm) const {
//}

}  // End of namespace kaldi
