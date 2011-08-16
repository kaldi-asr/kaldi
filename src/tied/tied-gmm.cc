// tied/tied-gmm.cc

// Copyright 2011  Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "tied/tied-gmm.h"

namespace kaldi {

void TiedGmm::Setup(int32 pdf_index, int32 nmix) {
  KALDI_ASSERT(nmix > 0);
  
  /// remember the pdf_index (within the AM)
  pdf_index_ = pdf_index;
  
  if (weights_.Dim() != nmix) 
    weights_.Resize(nmix);
  
  /// init weights with uniform distribution
  weights_.Set(1./nmix);
  
  valid_gconsts_ = false;
}

void TiedGmm::CopyFromTiedGmm(const TiedGmm &copy) {
  Setup(copy.pdf_index_, copy.weights_.Dim());
  gconsts_.CopyFromVec(copy.gconsts_);
  weights_.CopyFromVec(copy.weights_);
  valid_gconsts_ = copy.valid_gconsts_;
}

int32 TiedGmm::ComputeGconsts() {
  int32 num_mix = NumGauss();
  int32 num_bad = 0;
  gconsts_.Resize(num_mix);
  
  for (int32 mix = 0; mix < num_mix; mix++) {
    KALDI_ASSERT(weights_(mix) >= 0);  // Cannot have negative weights.

    // the codebook weights are assumed to be uniform, so subtract these before adding the actual weight
    BaseFloat gc = log(weights_(mix)) - std::log(num_mix);  // May be -inf if weights == 0

    if (KALDI_ISNAN(gc)) {  // negative infinity is OK but NaN is not acceptable
      KALDI_ERR << "At component "  << mix
                << ", not a number in gconst computation";
    }
    if (KALDI_ISINF(gc)) {
      num_bad++;
      // If positive infinity, make it negative infinity.
      // Want to make sure the answer becomes -inf in the end, not NaN.
      if (gc > 0) gc = -gc;
    }
    gconsts_(mix) = gc;
  }

  valid_gconsts_ = true;
  return num_bad;
}

BaseFloat TiedGmm::ComponentLogLikelihood(const VectorBase<BaseFloat> &scores,
                                          int32 comp_id) const {
  if (static_cast<int32>(scores.Dim()) != weights_.Dim()) {
    KALDI_ERR << "TiedGmm::ComponentLogLikelihood, dimension "
        << "mismatch" << (scores.Dim()) << "vs. "<< (weights_.Dim());
  }

  // ok, the tied GMM score is just the codebook component score minus
  // the uniform weight plus the actual component weight (log)
  return scores(comp_id) + gconsts_(comp_id); 
}

BaseFloat TiedGmm::LogLikelihood(const VectorBase<BaseFloat> &scores) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  Vector<BaseFloat> loglikes;
  LogLikelihoods(scores, &loglikes);
  BaseFloat log_sum = loglikes.LogSumExp();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  return log_sum;
}

void TiedGmm::LogLikelihoods(const VectorBase<BaseFloat> &scores,
                             Vector<BaseFloat> *loglikes) const {
  loglikes->Resize(gconsts_.Dim(), kUndefined);
  loglikes->CopyFromVec(gconsts_);
  loglikes->AddVec(1., scores);
}

void TiedGmm::LogLikelihoodsPreselect(const VectorBase<BaseFloat> &scores,
                                      const std::vector<int32> &indices,
                                      Vector<BaseFloat> *loglikes) const {
  KALDI_ASSERT(IsSortedAndUniq(indices) && !indices.empty()
               && indices.back() < NumGauss());

  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  
  if (static_cast<int32>(scores.Dim()) != weights_.Dim()) {
    KALDI_ERR << "TiedGmm::ComponentLogLikelihood, dimension "
        << "mismatch" << (scores.Dim()) << "vs. "<< (weights_.Dim());
  }
    
  int32 num_indices = static_cast<int32>(indices.size());
  loglikes->Resize(num_indices, kUndefined);
  if(indices.back() + 1 - indices.front() == num_indices) {
    // A special (but common) case when the indices form a contiguous range.
    int32 start_idx = indices.front();
    loglikes->CopyFromVec(SubVector<BaseFloat>(gconsts_, start_idx, num_indices));
    loglikes->AddVec(1., SubVector<BaseFloat>(scores, start_idx, num_indices));
  } else {
    for(int32 i = 0; i < num_indices; i++) {
      int32 idx = indices[i]; // The Gaussian index.
      (*loglikes)(i) = gconsts_(idx) + scores(idx);
    }
  }
}

// Gets likelihood of data given this. Also provides per-Gaussian posteriors.
BaseFloat TiedGmm::ComponentPosteriors(const VectorBase<BaseFloat> &scores,
                                       Vector<BaseFloat> *posterior) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  if (posterior == NULL) 
    KALDI_ERR << "NULL pointer passed as return argument.";

  Vector<BaseFloat> loglikes;
  LogLikelihoods(scores, &loglikes);
  BaseFloat log_sum = loglikes.ApplySoftMax();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  if (posterior->Dim() != loglikes.Dim())
    posterior->Resize(loglikes.Dim());
  posterior->CopyFromVec(loglikes);
  return log_sum;
}

void TiedGmm::Write(std::ostream &out_stream, bool binary) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before writing the model.";

  WriteMarker(out_stream, binary, "<TIEDGMM>");
  if (!binary) out_stream << "\n";
  WriteMarker(out_stream, binary, "<PDF_INDEX>");
  WriteBasicType(out_stream, binary, pdf_index_);
  if (!binary) out_stream << "\n";
  WriteMarker(out_stream, binary, "<GCONSTS>");
  gconsts_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<WEIGHTS>");
  weights_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</TIEDGMM>");
  if (!binary) out_stream << "\n";
}

std::ostream & operator <<(std::ostream & out_stream,
                           const kaldi::TiedGmm &gmm) {
  gmm.Write(out_stream, false);
  return out_stream;
}

void TiedGmm::Read(std::istream &in_stream, bool binary) {
  // ExpectMarker(in_stream, binary, "<TiedDiagGMM>");
  std::string marker;
  ReadMarker(in_stream, binary, &marker);
  if (marker != "<TIEDGMM>")
    KALDI_ERR << "Expected <TIEDGMM>, got " << marker;
  
  ReadMarker(in_stream, binary, &marker);
  if (marker != "<PDF_INDEX>")
    KALDI_ERR << "Expected <PDF_INDEX>, got " << marker;
  ReadBasicType(in_stream, binary, &pdf_index_);
  
  ReadMarker(in_stream, binary, &marker);
  if (marker == "<GCONSTS>") {  // The gconsts are optional.
    gconsts_.Read(in_stream, binary);
    ExpectMarker(in_stream, binary, "<WEIGHTS>");
  } else {
    if (marker != "<WEIGHTS>")
      KALDI_ERR << "TiedGmm::Read, expected <WEIGHTS> or <GCONSTS>, got "
                << marker;
  }
  weights_.Read(in_stream, binary);
  // ExpectMarker(in_stream, binary, "<TiedDiagGMM>");
  ReadMarker(in_stream, binary, &marker);
  // <DiagGMMEnd> is for compatibility. Will be deleted later
  if (marker != "</TIEDGMM>")
    KALDI_ERR << "Expected </TIEDGMM>, got " << marker;

  ComputeGconsts();  // safer option than trusting the read gconsts
}

std::istream & operator >>(std::istream & rIn, kaldi::TiedGmm &gmm) {
  gmm.Read(rIn, false);  // false == non-binary.
  return rIn;
}

}  // End namespace kaldi
