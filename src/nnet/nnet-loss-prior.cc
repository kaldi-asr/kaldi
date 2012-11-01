// nnet/nnet-loss-prior.cc

// Copyright 2012  Karel Vesely

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

#include "nnet/nnet-loss-prior.h"

#include "cudamatrix/cu-math.h"
#include "util/kaldi-io.h"

#include <sstream>
#include <iterator>
#include <algorithm>

namespace kaldi {


void XentPrior::EvalVec(const CuMatrix<BaseFloat> &net_out, const std::vector<int32> &target, CuMatrix<BaseFloat> *diff) {
  //chech the prior was loaded and has same dim as MLP output
  if(inv_priors_.Dim() == 0) {
    KALDI_ERR << "Priors were not loaded!";
  }
  KALDI_ASSERT(inv_priors_.Dim() == net_out.NumCols());

  // evaluate the raw frame-level classification (binary)
  int32 correct=0;
  cu::FindRowMaxId(net_out, &max_id_);
  max_id_.CopyToVec(&max_id_host_);
  KALDI_ASSERT(max_id_host_.size() == target.size());
  for(int32 i=0; i<target.size(); i++) {
    if (target[i] == max_id_host_[i]) correct++;
  }
  
  // get the raw xentropy and global error (unscaled)
  target_device_.CopyFromVec(target);
  if(&net_out != diff) { //<allow no-copy speedup
    diff->CopyFromMat(net_out);
  }
  cu::DiffXent(target_device_, diff, &log_post_tgt_);
  log_post_tgt_.CopyToVec(&log_post_tgt_host_);
  
  // Now we have derivative of Xentropy in diff,
  // it's computed as dE/da = net_out - target_mat,
  // E ... xentropy
  // a ... activation, the input of softmax
  // note that 'target_mat' is a sparse 1-of-M matrix 
  // encoded by index vector 'target'
  //
  // The frame-level xentropy statistics are computed as:
  // log(sum_row(net_out.*target_mat)))
  // they now are stored in vector log_post_tgt_
 
  // accumulate error quantites
  // a) raw loss 
  loss_    -= log_post_tgt_host_.Sum();
  frames_  += net_out.NumRows();
  correct_ += correct;
  // b) raw loss with sil excluded
  {
    double loss_nosil = 0.0;
    int32 frames_nosil = 0;
    int32 correct_nosil = 0;
    for(int32 i=0; i<target.size(); i++) { 
      if(target[i] >= sil_pdfs_) {
        loss_nosil += log_post_tgt_host_(i);
        frames_nosil++;
        if(target[i] == max_id_host_[i]) correct_nosil++;
      }
    }
    loss_nosil_    -= loss_nosil;
    frames_nosil_  += frames_nosil;
    correct_nosil_ += correct_nosil;
  }
  // c) rescale the error by priors
  // prepare the scaling mask
  Vector<BaseFloat> mask_host(target.size());
  for(int32 i=0; i<mask_host.Dim(); i++) {
    mask_host(i) = inv_priors_(target[i]);
  }
  // rescale the error
  log_post_tgt_host_.MulElements(mask_host);
  // rescale the derivative
  CuVector<BaseFloat> mask;
  mask.CopyFromVec(mask_host);
  diff->MulRowsVec(mask);
  // accumulate error quantities
  {
    double correct_scaled = 0.0;
    for(int32 i=0; i<target.size(); i++) {
      if(target[i] == max_id_host_[i]) correct_scaled += mask_host(i);
    }
    loss_scaled_    -= log_post_tgt_host_.Sum();
    frames_scaled_  += mask_host.Sum();
    correct_scaled_ += correct_scaled;
  }
  // d) rescaled loss with sil excluded
  {
    double loss_scaled_nosil = 0.0;
    double frames_scaled_nosil = 0.0;
    double correct_scaled_nosil = 0.0;
    for(int32 i=0; i<target.size(); i++) { 
      if(target[i] >= sil_pdfs_) {
        loss_scaled_nosil += log_post_tgt_host_(i);
        frames_scaled_nosil += mask_host(i);
        if(target[i] == max_id_host_[i]) correct_scaled_nosil += mask_host(i);
      }
    }
    loss_scaled_nosil_    -= loss_scaled_nosil;
    frames_scaled_nosil_  += frames_scaled_nosil;
    correct_scaled_nosil_ += correct_scaled_nosil;
  }
}



void XentPrior::ReadPriors(std::string prior_rxfile, BaseFloat U, BaseFloat S, int32 num_sil) {
  KALDI_LOG << "Loading priors from : " << prior_rxfile;
  bool binary;
  Input in(prior_rxfile, &binary);
  Vector<BaseFloat> priors;
  priors.Read(in.Stream(), binary);
  in.Close();
  
  KALDI_ASSERT(priors.Dim()>0);

  //normalize to probs
  priors.Scale(1.0/priors.Sum());
  //get uniform probability of a class Pu
  BaseFloat P_u = 1.0/priors.Dim();

  //The inverse-prior is `softened' by uniform prior Pu;
  //
  //inv = 1.0./(P(a)+U*Pu)
  //
  inv_priors_.Resize(priors.Dim());
  if(U >= 0.0 && U < 1000) {
    for(int32 i=0; i<priors.Dim(); i++) {
      inv_priors_(i) = 1.0 / (priors(i) + U*P_u);
    }
    KALDI_LOG << "The inverse-priors softened by " << U 
              << "x uniform probability P_u " << P_u;
  } else {
    inv_priors_.Set(P_u);
    KALDI_LOG << "Using flat prior : " << P_u;
  }

  //The inv-priors of silence models are scaled down,
  //so that the effective amount is silence data is S :
  //
  //get the sil_scale from the `raw-priors'
  BaseFloat sil_scale = S * priors.Sum() / priors.Range(0, num_sil).Sum();
  //use it on silence inv-priors
  if(S > 0.0 && S < 1.0) {  //0.0 or 1.0 will bypass it
    KALDI_LOG << "Downscaling silence-data by factor : " << sil_scale;
    inv_priors_.Range(0, num_sil).Scale(sil_scale);
  } else {
    KALDI_LOG << "Silence-data downscaling is DISABLED"; 
  }

  //finally we normalize, so that the expectation 
  //of inv-priors of speech-only data is equal to original...
  //(ie. so that the effective amount of training 
  //data (speech only) is unchanged, so the learnig rates have 
  //`same meaning' for different configurations of U and S)
  float amount_of_data = priors.Range(num_sil, priors.Dim()-num_sil).Sum();
  priors.MulElements(inv_priors_);
  float amount_of_data_scaled = priors.Range(num_sil, priors.Dim()-num_sil).Sum();
  inv_priors_.Scale(amount_of_data / amount_of_data_scaled);

  //set the number of silece pdfs'
  sil_pdfs_ = num_sil;

  //use these values to scale the gradients and loss function
  //print some brief info
  KALDI_LOG << "Using inv_priors based on " << prior_rxfile 
            << " min : " << inv_priors_.Min()
            << " max : " << inv_priors_.Max();
  KALDI_LOG << "We have used this configuration : "
            << " U " << U
            << " P_u " << P_u
            << ", S " << S
            << " num_sil " << num_sil
            << " sil_scale " << sil_scale;
  //print the scaling factors
  KALDI_LOG << inv_priors_;

}


std::string XentPrior::Report() {
  std::ostringstream oss;

  oss << "XentPrior, raw-loss:" << loss_ 
      << " frames:" << frames_ 
      << " err/frm:" << loss_/frames_;
  oss << "XentPrior, raw-loss-nosil:" << loss_nosil_ 
      << " frames:" << frames_nosil_ 
      << " err/frm:" << loss_nosil_/frames_nosil_;
  oss << "XentPrior, scaled-loss:" << loss_scaled_ 
      << " frames:" << frames_scaled_ 
      << " err/frm:" << loss_scaled_/frames_scaled_;
  oss << "XentPrior, scaled-loss-nosil:" << loss_scaled_nosil_ 
      << " frames:" << frames_scaled_nosil_ 
      << " err/frm:" << loss_scaled_nosil_/frames_scaled_nosil_;

  oss << "\nFRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<";
  oss << "\nNOSIL_FRAME_ACCURACY >> " << 100.0*correct_nosil_/frames_nosil_ << "% <<";
  oss << "\nSCALED_FRAME_ACCURACY >> " << 100.0*correct_scaled_/frames_scaled_ << "% <<";
  oss << "\nSCALED_NOSIL_FRAME_ACCURACY >> " << 100.0*correct_scaled_nosil_/frames_scaled_nosil_ << "% <<";
  
  return oss.str(); 
}



} // namespace
