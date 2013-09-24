// nnet/nnet-loss.cc

// Copyright 2011  Karel Vesely

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

#include "nnet/nnet-loss.h"

#include "cudamatrix/cu-math.h"

#include <sstream>
#include <iterator>

namespace kaldi {
namespace nnet1 {


void Xent::Eval(const CuMatrix<BaseFloat> &net_out, const CuMatrix<BaseFloat> &target, CuMatrix<BaseFloat> *diff) {
  
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  diff->Resize(net_out.NumRows(), net_out.NumCols());

  // compute derivative wrt. activations of last layer of neurons
  *diff = net_out;
  diff->AddMat(-1.0, target);

  // we do not produce per-frame classification accuracy for soft labels,
  // (-1 is an indicator to Report(.) to skip prining accuracy)
  correct_ = -1;

  // TODO reimplement when needed, we compute xentropy ON CPU
  int32 num_frames = net_out.NumRows(), num_states = net_out.NumCols();
  Matrix<BaseFloat> target_host(num_frames, num_states, kUndefined),
      net_out_host(num_frames, num_states, kUndefined);
  target.CopyToMat(&target_host);
  net_out.CopyToMat(&net_out_host);

  BaseFloat val;
  double loss = 0.0;
  for(int32 r=0; r < num_frames; r++) {
    for(int32 c=0; c < num_states; c++) {
      val = -target_host(r, c)*log(net_out_host(r, c));
      if (KALDI_ISINF(val)) val = 1e10;
      loss += val;
    }
  }
  
  loss_ += loss;
  frames_ += net_out.NumRows();
}


void Xent::EvalVec(const CuMatrix<BaseFloat> &net_out, const std::vector<int32> &target, CuMatrix<BaseFloat> *diff) {
  // evaluate the frame-level classification
  int32 correct=0;
  net_out.FindRowMaxId(&max_id_);
  max_id_.CopyToVec(&max_id_host_);
  KALDI_ASSERT(max_id_host_.size() == target.size());
  for(int32 i=0; i<static_cast<int32>(target.size()); i++) {
    if (target[i] == max_id_host_[i]) correct++;
  }
  
  // get the xentropy and global error 
  target_device_.CopyFromVec(target);
  if(&net_out != diff) { //<allow no-copy speedup
    *diff = net_out;
  }
  diff->DiffXent(target_device_, &log_post_tgt_);
  //
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
  //
  log_post_tgt_host_.Resize(log_post_tgt_.Dim());
  log_post_tgt_.CopyToVec(&log_post_tgt_host_);
  loss_    -= log_post_tgt_host_.Sum();
  
  // accumulate error quantites
  frames_  += net_out.NumRows();
  correct_ += correct;
   
}


std::string Xent::Report() {
  std::ostringstream oss;
  oss << "Xent:" << loss_ << " frames:" << frames_ 
      << " err/frm:" << loss_/frames_;
  if (correct_ >= 0.0) {
    oss << "\nFRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<";
  }
  return oss.str(); 
}




void Mse::Eval(const CuMatrix<BaseFloat> &net_out, const CuMatrix<BaseFloat> &target, CuMatrix<BaseFloat> *diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());


  // compute derivative w.r.t. neural nerwork outputs
  diff->Resize(net_out.NumRows(), net_out.NumCols());
  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0, target);

  // compute the per-frame MSE stats
  diff_pow_2_.Resize(diff->NumRows(), diff->NumCols());
  diff_pow_2_.CopyFromMat(*diff);
  diff_pow_2_.MulElements(diff_pow_2_); //grid-like operation
  // at this point we have computed 'diff_pow_2'
  // now sum each row (device)
  sum_diff_pow_2_.Resize(diff_pow_2_.NumRows());
  sum_diff_pow_2_.AddColSumMat(1.0,diff_pow_2_,0.0); //tree-like reduction
  // now sum the per-frame MSE (host)
  sum_diff_pow_2_host_.Resize(sum_diff_pow_2_.Dim());
  sum_diff_pow_2_.CopyToVec(&sum_diff_pow_2_host_);
  // accumulate
  loss_ += 0.5 * sum_diff_pow_2_host_.Sum();
  frames_ += net_out.NumRows();
}


std::string Mse::Report() {
  std::ostringstream oss;
  oss << "Mse:" << loss_ << " frames:" << frames_
      << " err/frm:" << loss_/frames_ 
      << std::endl;
  return oss.str();
}




void MseProgress::Eval(const CuMatrix<BaseFloat>& net_out, const CuMatrix<BaseFloat>& target, CuMatrix<BaseFloat>* diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());

  //compute derivative w.r.t. neural nerwork outputs
  diff->Resize(net_out.NumRows(),net_out.NumCols());
  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0,target);

  // compute the per-frame MSE stats
  diff_pow_2_.Resize(diff->NumRows(), diff->NumCols());
  diff_pow_2_.CopyFromMat(*diff);
  diff_pow_2_.MulElements(diff_pow_2_); //grid-like operation
  // at this point we have computed 'diff_pow_2'
  // now sum each row (device)
  sum_diff_pow_2_.Resize(diff_pow_2_.NumRows());
  sum_diff_pow_2_.AddColSumMat(1.0,diff_pow_2_,0.0); //tree-like reduction
  // now sum the per-frame MSE (host)
  sum_diff_pow_2_host_.Resize(sum_diff_pow_2_.Dim());
  sum_diff_pow_2_.CopyToVec(&sum_diff_pow_2_host_);
  // accumulate progress statistics
  loss_progress_ += 0.5 * sum_diff_pow_2_host_.Sum();
  frames_progress_ += net_out.NumRows();

  // monitor progress per progress_step_ frames
  if(frames_progress_ > progress_step_) {
    float loss_of_step = loss_progress_/frames_progress_;
    loss_vec_.push_back(loss_of_step);
    frames_ += frames_progress_; 
    loss_ += loss_progress_;
    KALDI_LOG << "Progress chunk #" << ++progress_ctr_ << " mse:" << loss_of_step << " [last " << frames_progress_/100/3600 << "h/" << frames_/100/3600 << "h]";
    frames_progress_ = 0;
    loss_progress_ = 0;
  }
}


std::string MseProgress::Report() {
  std::ostringstream oss;
  oss << "Mse:" << loss_+loss_progress_ << " frames:" << frames_+frames_progress_
      << " err/frm:" << (loss_+loss_progress_) / (frames_+frames_progress_)
      << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
  oss << "]" << std::endl;

  return oss.str();
}


} // namespace nnet1
} // namespace kaldi
