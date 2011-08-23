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
}

void TiedGmm::CopyFromTiedGmm(const TiedGmm &copy) {
  Setup(copy.pdf_index_, copy.weights_.Dim());
  weights_.CopyFromVec(copy.weights_);
}

// Compute the log-likelihood of the p(x|i) given the precomputed svq
BaseFloat TiedGmm::LogLikelihood(BaseFloat c, const VectorBase<BaseFloat> &svq) const {
  KALDI_ASSERT(svq.Dim() == weights_.Dim());

  // log p(x|i) = log(w_i^T v) + c, where c is the offset from the soft vector quantizer
  BaseFloat logl = log(VecVec(weights_, svq)) + c;

  if (KALDI_ISNAN(logl) || KALDI_ISINF(logl))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

  return logl;
}


// Compute log-likelihood of p(x|i) given the precomputed svq, also provide per-Gaussian posteriors
BaseFloat TiedGmm::ComponentPosteriors(BaseFloat c, const VectorBase<BaseFloat> &svq,
                                       Vector<BaseFloat> *posteriors) const {
  KALDI_ASSERT(posteriors != NULL);

  // compute pre-gaussian posterior
  posteriors->Resize(svq.Dim());
  posteriors->CopyFromVec(svq);
  posteriors->MulElements(weights_);
  
  // log-likelihood...
  BaseFloat log_sum = posteriors->Sum();

  // make posteriors
  posteriors->Scale(1. / log_sum);

  // add svq offset
  log_sum += c;

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  
  return log_sum;
}

// weights <- rho*weights + (1.-rho)source->weights, renorm weights afterwards
void TiedGmm::SmoothWithTiedGmm(BaseFloat rho, const TiedGmm *source) {
  KALDI_ASSERT(NumGauss() == source->NumGauss());
  KALDI_ASSERT(rho > 0. && rho < 1.);
  weights_.Scale(rho);
  weights_.AddVec(1. - rho, source->weights_);
  weights_.Scale(1. / weights_.Sum());
}

/*
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

void TiedGmm::LogLikelihoods(const VectorBase<BaseFloat> &svq,
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
*/
void TiedGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<TIEDGMM>");
  if (!binary) out_stream << "\n";
  WriteMarker(out_stream, binary, "<PDF_INDEX>");
  WriteBasicType(out_stream, binary, pdf_index_);
  if (!binary) out_stream << "\n";
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
  if (marker != "<WEIGHTS>")
    KALDI_ERR << "TiedGmm::Read, expected <WEIGHTS> got "
              << marker;
  weights_.Read(in_stream, binary);
  // ExpectMarker(in_stream, binary, "<TiedDiagGMM>");
  ReadMarker(in_stream, binary, &marker);
  // <DiagGMMEnd> is for compatibility. Will be deleted later
  if (marker != "</TIEDGMM>")
    KALDI_ERR << "Expected </TIEDGMM>, got " << marker;
}

std::istream & operator >>(std::istream & rIn, kaldi::TiedGmm &gmm) {
  gmm.Read(rIn, false);  // false == non-binary.
  return rIn;
}

}  // End namespace kaldi
