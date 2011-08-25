// gmm/mmie-diag-gmm.cc

// Copyright 2009-2011 Petr Motlicek, Arnab Ghoshal

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


void MmieAccumDiagGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectMarker(in_stream, binary, "<GMMMMIACCS>");
  ExpectMarker(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dimension);
  ExpectMarker(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);
  ExpectMarker(in_stream, binary, "<FLAGS>");
  ReadBasicType(in_stream, binary, &flags);

  if (add) {
    if ((NumGauss() != 0 || Dim() != 0 || Flags() != 0)) {
      if (num_components != NumGauss() || dimension != Dim()
          || flags != Flags()) {
        KALDI_ERR << "Dimension or flags mismatch: " << NumGauss() << ", "
                  << Dim() << ", " << Flags() << " vs. " << num_components
                  << ", " << dimension << ", " << flags;
      }
    } else {
      Resize(num_components, dimension, flags);
    }
  } else {
    Resize(num_components, dimension, flags);
  }

  ReadMarker(in_stream, binary, &token);
  while (token != "</GMMMMIACCS>") {
    if (token == "<NUM_OCCUPANCY>") {
      num_occupancy_.Read(in_stream, binary, add);
    } else if (token == "<DEN_OCCUPANCY>") {
      den_occupancy_.Read(in_stream, binary, add);
    } else if (token == "<MEANACCS>") {
      mean_accumulator_.Read(in_stream, binary, add);
    } else if (token == "<DIAGVARACCS>") {
      variance_accumulator_.Read(in_stream, binary, add);
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
  /// get difference
  occupancy_.CopyFromVec(num_occupancy_);
  occupancy_.AddVec(-1.0, den_occupancy_);

}

void MmieAccumDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<GMMMMIACCS>");
  WriteMarker(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteMarker(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteMarker(out_stream, binary, "<FLAGS>");
  WriteBasicType(out_stream, binary, flags_);

  // convert into BaseFloat before writing things
  Vector<BaseFloat> num_occupancy_bf(num_occupancy_.Dim());
  Vector<BaseFloat> den_occupancy_bf(den_occupancy_.Dim());
  Matrix<BaseFloat> mean_accumulator_bf(mean_accumulator_.NumRows(),
      mean_accumulator_.NumCols());
  Matrix<BaseFloat> variance_accumulator_bf(variance_accumulator_.NumRows(),
      variance_accumulator_.NumCols());
  num_occupancy_bf.CopyFromVec(num_occupancy_);
  den_occupancy_bf.CopyFromVec(den_occupancy_);
  mean_accumulator_bf.CopyFromMat(mean_accumulator_);
  variance_accumulator_bf.CopyFromMat(variance_accumulator_);

  WriteMarker(out_stream, binary, "<NUM_OCCUPANCY>");
  num_occupancy_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DEN_OCCUPANCY>");
  den_occupancy_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<MEANACCS>");
  mean_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<DIAGVARACCS>");
  variance_accumulator_bf.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</GMMMMIACCS>");
}




void MmieAccumDiagGmm::Resize(int32 num_comp, int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentGmmFlags(flags);
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


void MmieAccumDiagGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  if (flags & kGmmWeights) {
    num_occupancy_.SetZero();
    den_occupancy_.SetZero();
    occupancy_.SetZero();
  }
  if (flags & kGmmMeans) mean_accumulator_.SetZero();
  if (flags & kGmmVariances) variance_accumulator_.SetZero();
}


void MmieAccumDiagGmm::Scale(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) {
    num_occupancy_.Scale(d);
    den_occupancy_.Scale(d);
    occupancy_.Scale(d);
  }
  if (flags & kGmmMeans) mean_accumulator_.Scale(d);
  if (flags & kGmmVariances) variance_accumulator_.Scale(d);
}


void MmieAccumDiagGmm::SubtractAccumulatorsISmoothing(const AccumDiagGmm& num_acc,
                            const AccumDiagGmm& den_acc,
                            const MmieDiagGmmOptions& opts){

  //KALDI_ASSERT(num_acc.NumGauss() == den_acc.NumGauss && num_acc.Dim() == den_acc.Dim());
  //std::cout << "NumGauss: " << num_acc.NumGauss() << " " << den_acc.NumGauss() << " " << num_comp_ << '\n';
  KALDI_ASSERT(num_acc.NumGauss() == num_comp_ && num_acc.Dim() == dim_);
  KALDI_ASSERT(den_acc.NumGauss() == num_comp_ && den_acc.Dim() == dim_);
  
  
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
  //  std::cout << "M1: " << mean_accumulator_.Row(g) << '\n';
  //  std::cout << "Occ: " << occ << '\n';
    mean_accumulator_.Row(g).AddVec(opts.i_smooth_tau/occ, mean_accumulator_.Row(g));
    variance_accumulator_.Row(g).AddVec(opts.i_smooth_tau/occ, variance_accumulator_.Row(g));
  //  std::cout << "M2: " << mean_accumulator_.Row(g) << '\n';
  //  std::cout << "M22: " << den_acc.mean_accumulator().Row(g) << '\n';

  }

  num_occupancy_.Add(opts.i_smooth_tau);
  occupancy_.CopyFromVec(num_occupancy_);
  occupancy_.AddVec(-1.0, den_occupancy_);

  // Subtract den from smoothed num
  mean_accumulator_.AddMat(-1.0, den_acc.mean_accumulator(), kNoTrans);
  variance_accumulator_.AddMat(-1.0, den_acc.variance_accumulator(), kNoTrans);

  //for (int32 g = 0; g < num_comp_; g++) {
  //  std::cout << "M3: " << mean_accumulator_.Row(g) << '\n';
  //}
  //std::cout << "Dim: " << mean_accumulator_.NumRows() << " " << mean_accumulator_.NumCols() << '\n';
}




void MmieAccumDiagGmm::Update(const MmieDiagGmmOptions &config,
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

  int32 num_comp = num_comp_;
  int32 dim = dim_;
 
  // copy DiagGMM model and transform this to the normal case
  DiagGmmNormal *diaggmmnormal = new DiagGmmNormal();
  gmm->ComputeGconsts();
  diaggmmnormal->CopyFromDiagGmm(*gmm);

  // initialize D for all components
  Vector<double> D(num_comp);
  for (int32 g = 0; g < num_comp; g++)
       D(g) = config.ebw_e * den_occupancy_(g) / 2; // E*y_den/2 where E = 2;

  // go over all components
  double occ;
  Vector<double> mean(dim), var(dim);
  for (int32 g = 0; g < num_comp; g++) {
    occ = occupancy_(g) + D(g); // update occupancy score for given component
            
    while (1){
          // update mean - MMIe
          mean.CopyFromVec(diaggmmnormal->means_.Row(g));
          //std::cout << "MEAN1: " << mean << '\n';
          //std::cout << "D: " << D << '\n';
          //std::cout << "Occ: " << occ << '\n';
          mean.Scale(D(g));
          mean.AddVec(1.0, mean_accumulator_.Row(g));
          mean.Scale(1.0 / occ);
          //std::cout << "U: " << mean << '\n';
  
          // update var - MMIe
          //var.CopyFromVec(variance_accumulator_.Row(g));
          var.CopyFromVec(diaggmmnormal->means_.Row(g));
          var.ApplyPow(2.0);
          var.AddVec(1.0, diaggmmnormal->vars_.Row(g));
          var.Scale(D(g));
          var.AddVec(1.0, variance_accumulator_.Row(g));
          var.Scale(1.0 / occ);
          var.AddVec2(-1.0, mean);
          //std::cout << "g Min D: " << g << " " << var.Min() << '\n';
          
          // check D param
          if (var.Min() > 0){
            // good, we can update the GMM 
            D(g) *= 2;
            if ((flags & kGmmMeans) && !(flags & kGmmVariances)) {
              // updating means but not vars.  
              diaggmmnormal->means_.CopyRowFromVec(mean, g);
              //std::cout << "MEAN2: " << mean << '\n';
            }
            if ((flags & kGmmMeans) && (flags & kGmmVariances)) {
              // updating means and vars
              diaggmmnormal->means_.CopyRowFromVec(mean, g);
              diaggmmnormal->vars_.CopyRowFromVec(var, g);
              //std::cout << "MEAN2: " << mean << '\n';
            } 
            // we are done
            break;

          } else {
            // small step
            D(g) *= 1.1; 
          }

    } // while  
  }   // for
  // get sum of updated occupancies
  double occ_sum = 0.0;
  for (int32 g = 0; g < num_comp; g++){
    occ = occupancy_(g) + D(g);
    occ_sum += occ; 
  }

  // update weights and check the conditions 
  double prob;
  for (int32 g = 0; g < num_comp; g++){
    occ = occupancy_(g) + D(g);
    if (occ_sum != 0.0)
     prob = occ / occ_sum;
    else
     prob = 1.0 / num_comp;

    // check other  conditions, if not fullfilled, we get back original mean and var (no update)
    if (occ > static_cast<double>(config.min_gaussian_occupancy)
        && prob > static_cast<double>(config.min_gaussian_weight)) {
          if (flags & kGmmWeights) diaggmmnormal->weights_(g) = prob;
    } else {
          mean.CopyFromVec(diaggmmnormal->means_.Row(g));
          var.CopyFromVec(diaggmmnormal->means_.Row(g));
          diaggmmnormal->means_.CopyRowFromVec(mean, g);
          diaggmmnormal->vars_.CopyRowFromVec(var, g);
    }            
  } 
 
  // copy to natural representation according to flags
  diaggmmnormal->CopyToDiagGmm(gmm, flags);
  gmm->ComputeGconsts();  // or MlObjective will fail.


  std::cout << "End MMIe update\n";
}




MmieAccumDiagGmm::MmieAccumDiagGmm(const MmieAccumDiagGmm &other)
    : dim_(other.dim_), num_comp_(other.num_comp_),
      flags_(other.flags_), num_occupancy_(other.num_occupancy_),
      den_occupancy_(other.den_occupancy_),
      mean_accumulator_(other.mean_accumulator_),
      variance_accumulator_(other.variance_accumulator_) {}



//BaseFloat ComputeD(const DiagGmm& old_gmm, int32 mix_index, BaseFloat ebw_e){
//}



//BaseFloat MmieDiagGmm::MmiObjective(const DiagGmm& gmm) const {
//}

}  // End of namespace kaldi
