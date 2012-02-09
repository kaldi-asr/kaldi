// tied/am-tied-diag-gmm.cc

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#include <queue>
#include <string>
#include <vector>
using std::vector;

#include "tied/am-tied-diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

AmTiedDiagGmm::~AmTiedDiagGmm() {
  DeletePointers(&densities_);
  DeletePointers(&tied_densities_);
}

void AmTiedDiagGmm::Init(const DiagGmm& proto) {
  if (densities_.size() != 0 || tied_densities_.size() != 0) {
    KALDI_WARN << "Init() called on a non-empty object. Contents will be "
        "overwritten";
    DeletePointers(&densities_);
    DeletePointers(&tied_densities_);
  }

  // the codebook...
  densities_.resize(1, NULL);
  densities_[0] = new DiagGmm();
  densities_[0]->CopyFromDiagGmm(proto);

  // make sure the weights are uniform
  Vector<BaseFloat> w(proto.NumGauss());
  w.Set(1.0 / w.Dim());
  densities_[0]->SetWeights(w);

  dim_ = proto.Dim();
}

void AmTiedDiagGmm::AddCodebook(const DiagGmm &gmm) {
  if (densities_.size() != 0)  // not the first gmm
    assert(static_cast<int32>(gmm.Dim()) == dim_);
  else
    dim_ = gmm.Dim();

  DiagGmm *gmm_ptr = new DiagGmm();
  gmm_ptr->CopyFromDiagGmm(gmm);

  // make sure the weights are uniform
  Vector<BaseFloat> w(gmm.NumGauss());
  w.Set(1.0 / w.Dim());
  gmm_ptr->SetWeights(w);

  densities_.push_back(gmm_ptr);
}

void AmTiedDiagGmm::AddTiedPdf(const TiedGmm &tied) {
  TiedGmm *tgmm_ptr = new TiedGmm();
  tgmm_ptr->CopyFromTiedGmm(tied);
  tied_densities_.push_back(tgmm_ptr);
}

/// Set the codebook of the designated tied pdf
void AmTiedDiagGmm::ReplaceCodebook(int32 tied_pdf_index,
                                    int32 new_codebook_index) {
  KALDI_ASSERT(static_cast<size_t>(new_codebook_index) < densities_.size());
  KALDI_ASSERT(static_cast<size_t>(tied_pdf_index) < tied_densities_.size());

  tied_densities_[tied_pdf_index]->SetCodebookIndex(new_codebook_index);
}

/// Replace the designated codebook
void AmTiedDiagGmm::ReplaceCodebook(int32 codebook_index, const DiagGmm &gmm) {
  if (densities_.size() != 0)  // not the first gmm
    assert(static_cast<int32>(gmm.Dim()) == dim_);

  KALDI_ASSERT(static_cast<size_t>(codebook_index) < densities_.size());

  densities_[codebook_index]->CopyFromDiagGmm(gmm);
}

void AmTiedDiagGmm::CopyFromAmTiedDiagGmm(const AmTiedDiagGmm &other) {
  if (densities_.size() != 0) {
    DeletePointers(&densities_);
  }

  if (tied_densities_.size() != 0) {
    DeletePointers(&tied_densities_);
  }

  densities_.resize(other.NumCodebooks(), NULL);
  tied_densities_.resize(other.NumTiedPdfs(), NULL);

  dim_ = other.dim_;

  for (int32 i = 0; i < densities_.size(); ++i) {
    densities_[i] = new DiagGmm();
    densities_[i]->CopyFromDiagGmm(*(other.densities_[i]));
  }

  for (int32 i = 0; i < tied_densities_.size(); ++i) {
    tied_densities_[i] = new TiedGmm();
    tied_densities_[i]->CopyFromTiedGmm(*(other.tied_densities_[i]));
  }
}

int32 AmTiedDiagGmm::ComputeGconsts() {
  int32 num_bad_diag = 0;
  for (std::vector<DiagGmm*>::iterator itr = densities_.begin(),
      end = densities_.end(); itr != end; ++itr) {
    num_bad_diag += (*itr)->ComputeGconsts();
  }

  if (num_bad_diag > 0)
    KALDI_WARN << "Found " << num_bad_diag
               << " bad Gaussian components in codebooks";

  return num_bad_diag;
}

void AmTiedDiagGmm::SetupPerFrameVars(
       TiedGmmPerFrameVars *per_frame_vars) const {
  // init containers
  per_frame_vars->Setup(dim_, NumCodebooks());

  // allocate the svqs
  for (int32 i = 0; i < NumCodebooks(); ++i) {
    per_frame_vars->ResizeSvq(i, GetCodebook(i).NumGauss());
    per_frame_vars->current[i] = false;
  }
}

void AmTiedDiagGmm::ComputePerFrameVars(
       const VectorBase<BaseFloat> &data,
       TiedGmmPerFrameVars *per_frame_vars) const {
  // copy the current data vector
  per_frame_vars->x.CopyFromVec(data);

  // set the currency indicators to false
  for (int32 i = 0; i < NumCodebooks(); ++i)
    per_frame_vars->current[i] = false;
}

BaseFloat AmTiedDiagGmm::ComputePerFrameVars(const VectorBase<BaseFloat> &data,
                                             int32 codebook_index,
                                             Vector<BaseFloat> *svq) const {
  // get loglikes
  densities_[codebook_index]->LogLikelihoods(data, svq);

  // subtract log(weight) = log(1/NumGauss) = -log(NumGauss)
  svq->Add(log(densities_[codebook_index]->NumGauss()));

  // normalize; speed-up by using svq->Max() instead?
  BaseFloat c = svq->LogSumExp();
  svq->Add(-c);
  svq->ApplyExp();

  // return offset
  return  c;
}

void AmTiedDiagGmm::Read(std::istream &in_stream, bool binary) {
  int32 num_pdfs;
  int32 num_tied_pdfs;

  if (densities_.size() > 0 || tied_densities_.size() > 0)
    KALDI_WARN << "Calling AmTiedDiagGmm.Read on alread initialized model!";

  ExpectMarker(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &dim_);

  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  densities_.reserve(num_pdfs);
  for (int32 i = 0; i < num_pdfs; i++) {
    densities_.push_back(new DiagGmm());
    densities_.back()->Read(in_stream, binary);
    KALDI_ASSERT(static_cast<int32>(densities_.back()->Dim())
                 == dim_);
  }

  ExpectMarker(in_stream, binary, "<NUMTIEDPDFS>");
  ReadBasicType(in_stream, binary, &num_tied_pdfs);
  KALDI_ASSERT(num_tied_pdfs > 0);
  tied_densities_.reserve(num_tied_pdfs);
  for (int32 i = 0; i < num_tied_pdfs; i++) {
    tied_densities_.push_back(new TiedGmm());
    tied_densities_.back()->Read(in_stream, binary);
  }
}

void AmTiedDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, dim_);
  if (!binary) out_stream << "\n";

  // write out codebooks
  WriteMarker(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, static_cast<int32>(densities_.size()));
  if (!binary) out_stream << "\n";

  for (std::vector<DiagGmm*>::const_iterator it = densities_.begin(),
      end = densities_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }

  // write out tied pdfs
  WriteMarker(out_stream, binary, "<NUMTIEDPDFS>");
  WriteBasicType(out_stream, binary,
                 static_cast<int32>(tied_densities_.size()));
  if (!binary) out_stream << "\n";

  for (std::vector<TiedGmm*>::const_iterator it = tied_densities_.begin(),
      end = tied_densities_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}

}  // namespace kaldi
