// nnet/nnet-loss.cc

// Copyright 2011  Brno University of Technology (author: Karel Vesely)

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
#include "hmm/posterior.h"

#include <sstream>
#include <iterator>

namespace kaldi {
namespace nnet1 {


/* Xent */

void Xent::Eval(const CuMatrixBase<BaseFloat> &net_out, const CuMatrixBase<BaseFloat> &target, CuMatrix<BaseFloat> *diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  diff->Resize(net_out.NumRows(), net_out.NumCols());
  int32 num_frames = net_out.NumRows();

  // compute derivative wrt. activations of last layer of neurons
  *diff = net_out;
  diff->AddMat(-1.0, target);

  // we do not produce per-frame classification accuracy for soft labels,
  // (-1 is an indicator to Report(.) to skip prining accuracy)
  correct_ = -1;

  // calculate cross_entropy (in GPU)
  xentropy_aux_ = net_out; // y
  xentropy_aux_.ApplyLog(); // log(y)
  xentropy_aux_.MulElements(tgt_mat_device_); // t*log(y)
  log_post_tgt_.Resize(num_frames);
  log_post_tgt_.AddColSumMat(1.0,xentropy_aux_,0.0); // sum over cols (pdfs)
  log_post_tgt_host_.Resize(num_frames);
  log_post_tgt_.CopyToVec(&log_post_tgt_host_);
  double cross_entropy = -log_post_tgt_host_.Sum(); // sum over rows (frames)
  
  // caluculate entropy (in GPU)
  xentropy_aux_ = target; // t
  xentropy_aux_.Add(1e-99); // avoid log(0)
  xentropy_aux_.ApplyLog(); // log(t)
  xentropy_aux_.MulElements(tgt_mat_device_); // t*log(t)
  log_post_tgt_.Resize(num_frames);
  log_post_tgt_.AddColSumMat(1.0,xentropy_aux_,0.0); // sum over cols (pdfs)
  log_post_tgt_host_.Resize(num_frames);
  log_post_tgt_.CopyToVec(&log_post_tgt_host_);
  double entropy = -log_post_tgt_host_.Sum(); // sum over rows (frames)

  loss_ += cross_entropy;
  entropy_ += entropy;
  frames_ += num_frames;
}


void Xent::Eval(const CuMatrixBase<BaseFloat>& net_out, const Posterior& post, CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_pdf = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix
  Matrix<BaseFloat> tgt_mat_host(num_frames, num_pdf, kSetZero); // zero-filled
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      int32 pdf = post[t][i].first;
      if (pdf >= num_pdf) {
        KALDI_ERR << "Posterior pdf-id out of NN-output dimension, please check number of pdfs by 'hmm-info'."
                  << " nn-outputs : " << num_pdf << ", posterior pdf-id : " << pdf;
      }
      tgt_mat_host(t, pdf) += post[t][i].second;
    }
  }
  tgt_mat_device_ = tgt_mat_host; // -> GPU

  // compute derivaitve w.r.t. pre-softmax activation (net_out - tgt)
  *diff = net_out;
  diff->AddMat(-1.0, tgt_mat_device_);

  // evaluate the frame-level classification
  int32 correct=0;
  net_out.FindRowMaxId(&max_id_out_); // find max in nn-output
  tgt_mat_device_.FindRowMaxId(&max_id_tgt_); // find max in targets
  max_id_out_host_.resize(num_frames);
  max_id_tgt_host_.resize(num_frames);
  max_id_out_.CopyToVec(&max_id_out_host_);
  max_id_tgt_.CopyToVec(&max_id_tgt_host_);
  // count frames where maxima match
  for(int32 i=0; i<num_frames; i++) {
    if (max_id_tgt_host_[i] == max_id_out_host_[i]) correct++;
  }
  // TODO calculate phone-level accuracy,
  // need to get shuffled phone-ids externally ...

  // calculate cross_entropy (in GPU)
  xentropy_aux_ = net_out; // y
  xentropy_aux_.Add(1e-20); // avoid -inf
  xentropy_aux_.ApplyLog(); // log(y)
  xentropy_aux_.MulElements(tgt_mat_device_); // t*log(y)
  log_post_tgt_.Resize(num_frames);
  log_post_tgt_.AddColSumMat(1.0,xentropy_aux_,0.0); // sum over cols (pdfs)
  log_post_tgt_host_.Resize(num_frames);
  log_post_tgt_.CopyToVec(&log_post_tgt_host_);
  double cross_entropy = -log_post_tgt_host_.Sum(); // sum over rows (frames)

  // calculate entropy (from Posterior)
  double entropy = 0.0;
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      BaseFloat p = post[t][i].second;
      entropy += -p*log(p);
    }
  }
  
  // accumulate
  loss_ += cross_entropy;
  entropy_ += entropy;
  correct_ += correct;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += cross_entropy;
    entropy_progress_ += entropy;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[" << frames_progress_/100/3600 << "h/" << frames_/100/3600 << "h]: " 
                    << (loss_progress_-entropy_progress_)/frames_progress_ << " (Xent)";
      // store
      loss_vec_.push_back((loss_progress_-entropy_progress_)/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
      entropy_progress_ = 0.0;
    }
  }
}


void Xent::EvalVec(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &target, CuMatrix<BaseFloat> *diff) {
  // evaluate the frame-level classification
  int32 correct=0;
  net_out.FindRowMaxId(&max_id_out_);
  max_id_out_.CopyToVec(&max_id_out_host_);
  KALDI_ASSERT(max_id_out_host_.size() == target.size());
  for(int32 i=0; i<static_cast<int32>(target.size()); i++) {
    if (target[i] == max_id_out_host_[i]) correct++;
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
  oss << "AvgLoss: " << (loss_-entropy_)/frames_ << " (Xent), "
      << "[AvgXent: " << loss_/frames_ 
      << ", AvgTargetEnt: " << entropy_/frames_ << "]" << std::endl;
  if (loss_vec_.size() > 0) {
     oss << "progress: [";
     std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
     oss << "]" << std::endl;
  }
  if (correct_ >= 0.0) {
    oss << "\nFRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<";
  }
  return oss.str(); 
}


/* Mse */

void Mse::Eval(const CuMatrixBase<BaseFloat>& net_out, const CuMatrixBase<BaseFloat>& target, CuMatrix<BaseFloat>* diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  int32 num_frames = net_out.NumRows();

  //compute derivative w.r.t. neural nerwork outputs
  *diff = net_out; // y
  diff->AddMat(-1.0,target); // (y - t)

  // Compute MeanSquareError loss of mini-batch
  diff_pow_2_ = *diff;
  diff_pow_2_.MulElements(diff_pow_2_); // (y - t)^2
  sum_diff_pow_2_.Resize(num_frames);
  sum_diff_pow_2_.AddColSumMat(1.0, diff_pow_2_, 0.0); // sum over cols (pdfs)
  sum_diff_pow_2_host_.Resize(num_frames);
  sum_diff_pow_2_.CopyToVec(&sum_diff_pow_2_host_);
  double mean_square_error = 0.5 * sum_diff_pow_2_host_.Sum(); // sum over rows (frames)

  // accumulate
  loss_ += mean_square_error;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 1e6; // 2.77h
    frames_progress_ += num_frames;
    loss_progress_ += mean_square_error;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[" << frames_progress_/100/3600 << "h/" << frames_/100/3600 << "h]: " 
                    << loss_progress_/frames_progress_ << " (Mse)";
      // store
      loss_vec_.push_back(loss_progress_/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
    }
  }
}


void Mse::Eval(const CuMatrixBase<BaseFloat>& net_out, const Posterior& post, CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_pdf = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix
  Matrix<BaseFloat> tgt_mat(num_frames, num_pdf, kSetZero); // zero-filled
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      int32 pdf = post[t][i].first;
      if (pdf >= num_pdf) {
        KALDI_ERR << "Posterior pdf-id out of NN-output dimension, please check number of pdfs by 'hmm-info'."
                  << " nn-outputs : " << num_pdf << ", posterior pdf-id : " << pdf;
      }
      tgt_mat(t, pdf) += post[t][i].second;
    }
  }
  // call the other eval function
  Eval(net_out, CuMatrix<BaseFloat>(tgt_mat), diff);
}
 

std::string Mse::Report() {
  // compute root mean square
  int32 num_tgt = diff_pow_2_.NumCols();
  BaseFloat root_mean_square = sqrt(loss_/frames_/num_tgt);
  // build the mesage
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_/frames_ << " (Mse), " << "[RMS " << root_mean_square << "]" << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
  oss << "]" << std::endl;
  return oss.str();
}


} // namespace nnet1
} // namespace kaldi
