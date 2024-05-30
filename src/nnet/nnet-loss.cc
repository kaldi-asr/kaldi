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

#include <sstream>
#include <iterator>
#include <algorithm>
#include <iomanip>

#include "nnet/nnet-loss.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {


/* Xent */

/**
 * Helper function of Xent::Eval,
 * calculates number of matching elemente in 'hyp', 'ref' weighted by 'weights'.
 */
template <typename T>
inline void CountCorrectFramesWeighted(const CuArray<T> &hyp,
                                       const CuArray<T> &ref,
                                       const CuVectorBase<BaseFloat> &weights,
                                       Vector<double> *correct) {
  KALDI_ASSERT(hyp.Dim() == ref.Dim());
  KALDI_ASSERT(hyp.Dim() == weights.Dim());
  int32 dim = hyp.Dim();
  // Get GPU data to host,
  std::vector<T> hyp_h(dim), ref_h(dim);
  hyp.CopyToVec(&hyp_h);
  ref.CopyToVec(&ref_h);
  Vector<BaseFloat> w(dim);
  weights.CopyToVec(&w);
  // Accumulate weighted counts of correct frames,
  for (int32 i = 0; i < dim; i++) {
    KALDI_ASSERT(ref_h[i] < correct->Dim());
    (*correct)(ref_h[i]) += w(i) * (hyp_h[i] == ref_h[i] ? 1.0 : 0.0);
  }
}


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out,
                const CuMatrixBase<BaseFloat> &targets,
                CuMatrix<BaseFloat> *diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == targets.NumCols());
  KALDI_ASSERT(net_out.NumRows() == targets.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(targets.Sum()));

  // buffer initialization,
  int32 num_classes = targets.NumCols();
  if (frames_.Dim() == 0) {
    frames_.Resize(num_classes);
    xentropy_.Resize(num_classes);
    entropy_.Resize(num_classes);
    correct_.Resize(num_classes);
  }

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // There may be frames for which the sum of targets is zero.
  // This happens in multi-lingual training when the frame
  // has target class in the softmax of another language.
  // We 'switch-off' such frames by masking the 'frame_weights_',
  target_sum_.Resize(targets.NumRows());
  target_sum_.AddColSumMat(1.0, targets, 0.0);
  frame_weights_.MulElements(target_sum_);

  // compute derivative wrt. activations of last layer of neurons,
  *diff = net_out;
  diff->AddMat(-1.0, targets);
  diff->MulRowsVec(frame_weights_);  // weighting,

  // count frames per class,
  frames_aux_ = targets;
  frames_aux_.MulRowsVec(frame_weights_);
  frames_.AddRowSumMat(1.0, CuMatrix<double>(frames_aux_));

  // evaluate the frame-level classification,
  net_out.FindRowMaxId(&max_id_out_);  // find max in nn-output
  targets.FindRowMaxId(&max_id_tgt_);  // find max in targets
  CountCorrectFramesWeighted(max_id_out_, max_id_tgt_,
                             frame_weights_, &correct_);

  // calculate cross_entropy (in GPU),
  xentropy_aux_ = net_out;  // y
  xentropy_aux_.Add(1e-20);  // avoid log(0)
  xentropy_aux_.ApplyLog();  // log(y)
  xentropy_aux_.MulElements(targets);  // t*log(y)
  xentropy_aux_.MulRowsVec(frame_weights_);  // w*t*log(y)
  xentropy_.AddRowSumMat(-1.0, CuMatrix<double>(xentropy_aux_));

  // caluculate entropy (in GPU),
  entropy_aux_ = targets;  // t
  entropy_aux_.Add(1e-20);  // avoid log(0)
  entropy_aux_.ApplyLog();  // log(t)
  entropy_aux_.MulElements(targets);  // t*log(t)
  entropy_aux_.MulRowsVec(frame_weights_);  // w*t*log(t)
  entropy_.AddRowSumMat(-1.0, CuMatrix<double>(entropy_aux_));

  // progressive loss reporting
  if (opts_.loss_report_frames > 0) {
    frames_progress_ += frame_weights_.Sum();
    xentropy_progress_ += -xentropy_aux_.Sum();
    entropy_progress_ += -entropy_aux_.Sum();

    KALDI_ASSERT(KALDI_ISFINITE(xentropy_progress_));
    KALDI_ASSERT(KALDI_ISFINITE(entropy_progress_));

    if (frames_progress_ > opts_.loss_report_frames) {
      // loss value,
      double progress_value =
        (xentropy_progress_ - entropy_progress_) / frames_progress_;

      // time-related info (fps is weighted),
      double time_now = timer_.Elapsed();
      double fps = frames_progress_ / (time_now - elapsed_seconds_);
      double elapsed_hours = time_now / 3600;
      elapsed_seconds_ = time_now; // store,

      // print,
      KALDI_LOG << "ProgressLoss[last "
                << static_cast<int>(frames_progress_/100/3600) << "h of "
                << static_cast<int>(frames_.Sum()/100/3600) << "h]: "
                << progress_value << " (Xent)"
                << ", fps=" << fps
                << std::setprecision(3)
                << ", elapsed " << elapsed_hours << "h";
      // store,
      loss_vec_.push_back(progress_value);
      // reset,
      frames_progress_ = 0;
      xentropy_progress_ = 0.0;
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
  double loss_value =
    (xentropy_.Sum() - entropy_.Sum()) / frames_.Sum();
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_value << " (Xent), "
      << "[AvgXent: " << xentropy_.Sum() / frames_.Sum()
      << ", AvgTargetEnt: " << entropy_.Sum() / frames_.Sum()
      << "]" << std::endl;

  oss << "progress: [";
  std::copy(loss_vec_.begin(), loss_vec_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;

  double frame_accuracy = 100.0 * correct_.Sum() / frames_.Sum();
  oss << "FRAME_ACCURACY >> " << frame_accuracy << "% <<" << std::endl;

  return oss.str();
}


std::string Xent::ReportPerClass() {
  std::ostringstream oss;
  oss << "PER-CLASS PERFORMANCE:" << std::endl;
  oss << "@@@ Frames per-class:" << frames_;
  // get inverted counts,
  CuVector<double> inv_frames(frames_);
  inv_frames.Add(0.5);  // avoid 0-frames,
  inv_frames.ApplyPow(-1.0);
  // loss, kl = xentropy-entropy,
  CuVector<double> loss(xentropy_);
  loss.AddVec(-1.0, entropy_);
  loss.MulElements(inv_frames);
  oss << "@@@ Loss per-class:" << loss;
  // frame accuracy (assuming targets are binary),
  CuVector<double> frm_accu(correct_);
  frm_accu.MulElements(inv_frames);
  frm_accu.Scale(100.0);
  oss << "@@@ Frame-accuracy per-class:" << frm_accu;
  //
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
  KALDI_ASSERT(num_frames >= 0.0);

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // compute derivative w.r.t. neural nerwork outputs
  *diff = net_out;  // y
  diff->AddMat(-1.0, target);  // (y - t)
  diff->MulRowsVec(frame_weights_);  // weighting,

  // Compute MeanSquareError loss of mini-batch
  diff_pow_2_ = *diff;
  diff_pow_2_.MulElements(diff_pow_2_);  // (y - t)^2
  diff_pow_2_.MulRowsVec(frame_weights_);  // w*(y - t)^2
  double mean_square_error = 0.5 * diff_pow_2_.Sum();  // sum the matrix,

  KALDI_ASSERT(KALDI_ISFINITE(mean_square_error));

  // accumulate
  loss_ += mean_square_error;
  frames_ += num_frames;

  // progressive loss reporting
  if (opts_.loss_report_frames > 0) {
    frames_progress_ += num_frames;
    loss_progress_ += mean_square_error;
    if (frames_progress_ > opts_.loss_report_frames) {
      KALDI_LOG << "ProgressLoss[last "
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
  oss << "AvgLoss: " << loss_/frames_ << " (Mse), "
      << "[RMS " << root_mean_square << ", frames "
      << frames_ << "]" << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(), loss_vec_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;
  return oss.str();
}


/* MultiTaskLoss */

void MultiTaskLoss::InitFromString(const std::string& s) {
  std::vector<std::string> v;
  SplitStringToVector(s, ",:" /* delimiter */, false, &v);

  KALDI_ASSERT((v.size()-1) % 3 == 0);  // triplets,
  KALDI_ASSERT(v[0] == "multitask");  // header,

  // parse the definition of multitask loss,
  std::vector<std::string>::iterator it(v.begin()+1);  // skip header,
  for ( ; it != v.end(); ++it) {
    // type,
    if (*it == "xent") {
      loss_vec_.push_back(new Xent(opts_));
    } else if (*it == "mse") {
      loss_vec_.push_back(new Mse(opts_));
    } else {
      KALDI_ERR << "Unknown objective function code : " << *it;
    }
    ++it;
    // dim,
    int32 dim;
    if (!ConvertStringToInteger(*it, &dim)) {
      KALDI_ERR << "Cannot convert 'dim' " << *it << " to integer!";
    }
    loss_dim_.push_back(dim);
    ++it;
    // weight,
    BaseFloat weight;
    if (!ConvertStringToReal(*it, &weight)) {
      KALDI_ERR << "Cannot convert 'weight' " << *it << " to integer!";
    }
    KALDI_ASSERT(weight >= 0.0);
    loss_weights_.push_back(weight);
  }

  // build vector with starting-point offsets,
  loss_dim_offset_.resize(loss_dim_.size()+1, 0);  // 1st zero stays,
  for (int32 i = 1; i <= loss_dim_.size(); i++) {
    loss_dim_offset_[i] = loss_dim_offset_[i-1] + loss_dim_[i-1];
  }

  // sanity check,
  KALDI_ASSERT(loss_vec_.size() > 0);
  KALDI_ASSERT(loss_vec_.size() == loss_dim_.size());
  KALDI_ASSERT(loss_vec_.size() == loss_weights_.size());
}

void MultiTaskLoss::Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const Posterior& post,
            CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_output = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());
  KALDI_ASSERT(num_output == loss_dim_offset_.back());  // sum of loss-dims,

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_output, &tgt_mat_);

  // allocate diff matrix,
  diff->Resize(num_frames, num_output);

  /// One vector of frame_weights per loss-function,
  /// The original frame weights are multiplied with
  /// a mask of `defined targets' according to the 'Posterior'.
  std::vector<Vector<BaseFloat> > frmwei_have_tgt;
  for (int32 l = 0; l < loss_vec_.size(); l++) {
    // copy original weights,
    frmwei_have_tgt.push_back(Vector<BaseFloat>(frame_weights));
    // We need to mask-out the frames for which the 'posterior' is not defined (= is empty):
    int32 loss_beg = loss_dim_offset_[l];   // first column of loss target,
    int32 loss_end = loss_dim_offset_[l+1]; // (last+1) column of loss target,
    for (int32 f = 0; f < num_frames; f++) {
      bool tgt_defined = false;
      for (int32 p = 0; p < post[f].size(); p++) {
        if (post[f][p].first >= loss_beg && post[f][p].first < loss_end) {
          tgt_defined = true;
          break;
        }
      }
      if (!tgt_defined) {
        frmwei_have_tgt[l](f) = 0.0; // set zero_weight for the frame with no targets!
      }
    }
  }

  // call the vector of loss functions,
  CuMatrix<BaseFloat> diff_aux;
  for (int32 l = 0; l < loss_vec_.size(); l++) {
    loss_vec_[l]->Eval(frmwei_have_tgt[l],
      net_out.ColRange(loss_dim_offset_[l], loss_dim_[l]),
      tgt_mat_.ColRange(loss_dim_offset_[l], loss_dim_[l]),
      &diff_aux);
    // Scale the gradients,
    diff_aux.Scale(loss_weights_[l]);
    // Copy to diff,
    diff->ColRange(loss_dim_offset_[l], loss_dim_[l]).CopyFromMat(diff_aux);
  }
}

std::string MultiTaskLoss::Report() {
  // calculate overall loss (weighted),
  BaseFloat overall_loss = AvgLoss();
  // copy the loss-values into a vector,
  std::vector<BaseFloat> loss_values;
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    loss_values.push_back(loss_vec_[i]->AvgLoss());
  }

  // build the message,
  std::ostringstream oss;
  oss << "MultiTaskLoss, with " << loss_vec_.size()
      << " parallel loss functions." << std::endl;
  // individual loss reports first,
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    oss << "Loss " << i+1 << ", " << loss_vec_[i]->Report() << std::endl;
  }

  // overall loss is last,
  oss << "Loss (OVERALL), "
      << "AvgLoss: " << overall_loss << " (MultiTaskLoss), "
      << "weights " << loss_weights_ << ", "
      << "values " << loss_values << std::endl;

  return oss.str();
}

BaseFloat MultiTaskLoss::AvgLoss() {
  BaseFloat ans(0.0);
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    BaseFloat val = loss_weights_[i] * loss_vec_[i]->AvgLoss();
    if (!KALDI_ISFINITE(val)) {
      KALDI_WARN << "Loss " << i+1 << ", has bad objective function value '"
                 << val << "', using 0.0 instead.";
      val = 0.0;
    }
    ans += val;
  }
  return ans;
}

}  // namespace nnet1
}  // namespace kaldi
