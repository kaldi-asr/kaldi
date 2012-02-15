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

void TiedGmm::Setup(int32 codebook_index, int32 nmix) {
  KALDI_ASSERT(nmix > 0);

  /// remember the pdf_index (within the AM)
  codebook_index_ = codebook_index;

  if (weights_.Dim() != nmix)
    weights_.Resize(nmix);

  /// init weights with uniform distribution
  weights_.Set(1.0 / nmix);
}

void TiedGmm::CopyFromTiedGmm(const TiedGmm &copy) {
  Setup(copy.codebook_index_, copy.weights_.Dim());
  weights_.CopyFromVec(copy.weights_);
}

// Compute the log-likelihood of the p(x|i) given the precomputed svq
BaseFloat TiedGmm::LogLikelihood(BaseFloat c,
                                 const VectorBase<BaseFloat> &svq) const {
  KALDI_ASSERT(svq.Dim() == weights_.Dim());

  // log p(x|i) = log(w_i^T v) + c, where c is the offset from the soft vector
  // quantizer
  BaseFloat logl = log(VecVec(weights_, svq)) + c;

  if (KALDI_ISNAN(logl) || KALDI_ISINF(logl))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

  return logl;
}


// Compute log-likelihood of p(x|i) given the precomputed svq, also provide
// per-Gaussian posteriors
BaseFloat TiedGmm::ComponentPosteriors(BaseFloat c,
                                       const VectorBase<BaseFloat> &svq,
                                       Vector<BaseFloat> *posteriors) const {
  KALDI_ASSERT(posteriors != NULL);

  // compute pre-gaussian posterior
  posteriors->Resize(svq.Dim());
  posteriors->CopyFromVec(svq);
  posteriors->MulElements(weights_);

  // log-likelihood...
  BaseFloat sum = posteriors->Sum();
  BaseFloat log_sum = log(sum);

  // make posteriors
  posteriors->Scale(1.0 / sum);

  // add svq offset
  log_sum += c;

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

  return log_sum;
}

/// this = rho x source + (1-rho) x this
void TiedGmm::Interpolate(BaseFloat rho, const TiedGmm &source) {
  KALDI_ASSERT(NumGauss() == source.NumGauss());
  KALDI_ASSERT(rho > 0.0 && rho < 1.0);

  // interpolate
  weights_.Scale(1.0 - rho);
  weights_.AddVec(rho, source.weights_);

  // renorm to sum to one
  weights_.Scale(1.0 / weights_.Sum());
}

/// Split the tied GMM weights based on the split sequence of the codebook
void TiedGmm::Split(std::vector<int32> *sequence) {
  KALDI_ASSERT(sequence != NULL);

  if (sequence->size() == 0)
    return;

  int32 oldsize = weights_.Dim();
  weights_.Resize(oldsize + sequence->size());

  // as in the gmm splitting, we'll distribute the weights evenly
  for (std::vector<int32>::iterator it = sequence->begin(),
       end = sequence->end(); it != end; ++it) {
    BaseFloat w = weights_(*it);
    weights_(*it) = w / 2.0;
    weights_(oldsize++) = w / 2.0;
  }

  // re-norm weights
  weights_.Scale(1.0 / weights_.Sum());
}

/// Merge the tied GMM weights based on the merge sequence of the codebook
void TiedGmm::Merge(std::vector<int32> *sequence) {
  KALDI_ASSERT(sequence != NULL);
  KALDI_ASSERT(sequence->size() % 2 == 0);

  if (sequence->size() == 0)
    return;

  // as in the gmm merging, we sum the weights of the candidates, and write them
  // to the first index
  std::vector<bool> discarded(weights_.Dim(), false);
  for (std::vector<int32>::iterator it = sequence->begin(),
       end = sequence->end(); it != end; it += 2) {
     weights_(*it) = weights_(*it) + weights_(*(it+1));
     discarded[*(it+1)] = true;
  }

  int32 m = 0;
  for (int32 i = 0; i < discarded.size(); ++i) {
    if (discarded[i])
      weights_.RemoveElement(m);
    else
      ++m;
  }

  weights_.Scale(1.0 / weights_.Sum());
}

void TiedGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteToken(out_stream, binary, "<TIEDGMM>");
  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<PDF_INDEX>");
  WriteBasicType(out_stream, binary, codebook_index_);
  //  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<WEIGHTS>");
  weights_.Write(out_stream, binary);
  WriteToken(out_stream, binary, "</TIEDGMM>");
  if (!binary) out_stream << "\n";
}

std::ostream & operator <<(std::ostream & out_stream,
                           const kaldi::TiedGmm &gmm) {
  gmm.Write(out_stream, false);
  return out_stream;
}

void TiedGmm::Read(std::istream &in_stream, bool binary) {
  // ExpectToken(in_stream, binary, "<TiedDiagGMM>");
  std::string token;
  ReadToken(in_stream, binary, &token);
  if (token != "<TIEDGMM>")
    KALDI_ERR << "Expected <TIEDGMM>, got " << token;

  ReadToken(in_stream, binary, &token);
  if (token != "<PDF_INDEX>")
    KALDI_ERR << "Expected <PDF_INDEX>, got " << token;
  ReadBasicType(in_stream, binary, &codebook_index_);

  ReadToken(in_stream, binary, &token);
  if (token != "<WEIGHTS>")
    KALDI_ERR << "TiedGmm::Read, expected <WEIGHTS> got "
              << token;
  weights_.Read(in_stream, binary);
  // ExpectToken(in_stream, binary, "<TiedDiagGMM>");
  ReadToken(in_stream, binary, &token);
  // <DiagGMMEnd> is for compatibility. Will be deleted later
  if (token != "</TIEDGMM>")
    KALDI_ERR << "Expected </TIEDGMM>, got " << token;
}

std::istream & operator >>(std::istream & rIn, kaldi::TiedGmm &gmm) {
  gmm.Read(rIn, false);  // false == non-binary.
  return rIn;
}

}  // End namespace kaldi
