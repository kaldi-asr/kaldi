// nnet/nnet-loss.cc

// Copyright 2011-2015  Brno University of Technology (author: Karel Vesely)

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
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "hmm/posterior.h"

#include <sstream>
#include <iterator>

namespace kaldi {
namespace nnet1 {


/* Xent */

/**
 * Helper function of Xent::Eval,
 * calculates number of matching elemente in 'v1', 'v2' weighted by 'weights'.
 */
template <typename T>
inline void CountCorrectFramesWeighted(const CuArray<T> &v1, 
                                       const CuArray<T> &v2, 
                                       const VectorBase<BaseFloat> &weights, 
                                       double *correct) {
  KALDI_ASSERT(v1.Dim() == v2.Dim());
  KALDI_ASSERT(v1.Dim() == weights.Dim());
  int32 dim = v1.Dim();
  // Get GPU data to host,
  std::vector<T> v1_h(dim), v2_h(dim);
  v1.CopyToVec(&v1_h);
  v2.CopyToVec(&v2_h);
  // Get correct frame count (weighted),
  double corr = 0.0;
  for (int32 i=0; i<dim; i++) {
   corr += weights(i) * (v1_h[i] == v2_h[i] ? 1.0 : 0.0);
  }
  // Return,
  (*correct) = corr;
}


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out, 
                const CuMatrixBase<BaseFloat> &target, 
                CuMatrix<BaseFloat> *diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target.Sum()));

  double num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0)

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // compute derivative wrt. activations of last layer of neurons,
  *diff = net_out;
  diff->AddMat(-1.0, target);
  diff->MulRowsVec(frame_weights_); // weighting,

  // evaluate the frame-level classification,
  double correct; 
  net_out.FindRowMaxId(&max_id_out_); // find max in nn-output
  target.FindRowMaxId(&max_id_tgt_); // find max in targets
  CountCorrectFramesWeighted(max_id_out_, max_id_tgt_, frame_weights, &correct);

  // calculate cross_entropy (in GPU),
  xentropy_aux_ = net_out; // y
  xentropy_aux_.Add(1e-20); // avoid log(0)
  xentropy_aux_.ApplyLog(); // log(y)
  xentropy_aux_.MulElements(target); // t*log(y)
  xentropy_aux_.MulRowsVec(frame_weights_); // w*t*log(y) 
  double cross_entropy = -xentropy_aux_.Sum();
  
  // caluculate entropy (in GPU),
  entropy_aux_ = target; // t
  entropy_aux_.Add(1e-20); // avoid log(0)
  entropy_aux_.ApplyLog(); // log(t)
  entropy_aux_.MulElements(target); // t*log(t)
  entropy_aux_.MulRowsVec(frame_weights_); // w*t*log(t) 
  double entropy = -entropy_aux_.Sum();

  KALDI_ASSERT(KALDI_ISFINITE(cross_entropy));
  KALDI_ASSERT(KALDI_ISFINITE(entropy));

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
      KALDI_VLOG(1) << "ProgressLoss[last " 
                    << static_cast<int>(frames_progress_/100/3600) << "h of " 
                    << static_cast<int>(frames_/100/3600) << "h]: " 
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


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out, 
                const Posterior &post, 
                CuMatrix<BaseFloat> *diff) {
  int32 num_frames = net_out.NumRows(),
    num_pdf = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_pdf, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
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

void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out, 
               const CuMatrixBase<BaseFloat>& target, 
               CuMatrix<BaseFloat>* diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target.Sum()));

  int32 num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0)

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  //compute derivative w.r.t. neural nerwork outputs
  *diff = net_out; // y
  diff->AddMat(-1.0,target); // (y - t)
  diff->MulRowsVec(frame_weights_); // weighting,

  // Compute MeanSquareError loss of mini-batch
  diff_pow_2_ = *diff;
  diff_pow_2_.MulElements(diff_pow_2_); // (y - t)^2
  diff_pow_2_.MulRowsVec(frame_weights_); // w*(y - t)^2
  double mean_square_error = 0.5 * diff_pow_2_.Sum(); // sum the matrix,

  KALDI_ASSERT(KALDI_ISFINITE(mean_square_error));

  // accumulate
  loss_ += mean_square_error;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += mean_square_error;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last " 
                    << static_cast<int>(frames_progress_/100/3600) << "h of " 
                    << static_cast<int>(frames_/100/3600) << "h]: " 
                    << loss_progress_/frames_progress_ << " (Mse)";
      // store
      loss_vec_.push_back(loss_progress_/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
    }
  }
}


void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out, 
               const Posterior& post, 
               CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_nn_outputs = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_nn_outputs, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
}
 

std::string Mse::Report() {
  // compute root mean square,
  int32 num_tgt = diff_pow_2_.NumCols();
  BaseFloat root_mean_square = sqrt(loss_/frames_/num_tgt);
  // build the message,
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_/frames_ << " (Mse), " << "[RMS " << root_mean_square << "]" << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
  oss << "]" << std::endl;
  return oss.str();
}


} // namespace nnet1
} // namespace kaldi
