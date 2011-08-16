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
#include "tree/clusterable-classes.h"
#include "tree/cluster-utils.h"

namespace kaldi {

AmTiedDiagGmm::~AmTiedDiagGmm() {
  DeletePointers(&densities_);
  DeletePointers(&tied_densities_);
}

void AmTiedDiagGmm::Init(const DiagGmm& proto, int32 num_tied_pdfs) {
  if (densities_.size() != 0 || tied_densities_.size() != 0) {
    KALDI_WARN << "Init() called on a non-empty object. Contents will be "
        "overwritten";
    DeletePointers(&densities_);
    DeletePointers(&tied_densities_);
  }
  
  if (num_tied_pdfs == 0) {
    KALDI_WARN << "Init() called with number of pdfs = 0. Will do nothing.";
    return;
  }

  // the codebook...
  densities_.resize(1, NULL);
  densities_[0] = new DiagGmm();
  densities_[0]->CopyFromDiagGmm(proto);

  dim_ = proto.Dim();
  
  // the tied pdfs...
  tied_densities_.resize(num_tied_pdfs, NULL);
  for (vector<TiedGmm*>::iterator itr = tied_densities_.begin(),
      end = tied_densities_.end(); itr != end; ++itr) {
    *itr = new TiedGmm();
    (*itr)->Setup(0, proto.NumGauss());
  }
}

void AmTiedDiagGmm::AddPdf(const DiagGmm &gmm) {
  if (densities_.size() != 0)  // not the first gmm
    assert(static_cast<int32>(gmm.Dim()) == dim_);
  else
    dim_ = gmm.Dim();

  DiagGmm *gmm_ptr = new DiagGmm();
  gmm_ptr->CopyFromDiagGmm(gmm);
  densities_.push_back(gmm_ptr);
}

void AmTiedDiagGmm::AddTiedPdf(const TiedGmm &tied) {
  if (tied_densities_.size() != 0)  // not the first gmm
    assert(static_cast<int32>(tied.Dim()) == dim_);
  else
    dim_ = tied.Dim();

  TiedGmm *tgmm_ptr = new TiedGmm();
  tgmm_ptr->CopyFromTiedGmm(tied);
  tied_densities_.push_back(tgmm_ptr);
}

/// Remove designated codebook; caveat: does not take care of the tied dependents!
void AmTiedDiagGmm::RemovePdf(const int32 pdf_index) {
  KALDI_ASSERT(static_cast<size_t>(pdf_index) < densities_.size());
  delete densities_[pdf_index];
  densities_.erase(densities_.begin() + pdf_index);
}
  
/// Remove designated tied pdf
void AmTiedDiagGmm::RemoveTiedPdf(const int32 tied_pdf_index) {
  KALDI_ASSERT(static_cast<size_t>(tied_pdf_index) < tied_densities_.size());
  delete tied_densities_[tied_pdf_index];
  tied_densities_.erase(tied_densities_.begin() + tied_pdf_index);
}

/// Replace the codebook of the designated tied pdf by the new index
void AmTiedDiagGmm::ReplacePdf(const int32 tied_pdf_index, const int32 new_pdf_index) {
  KALDI_ASSERT(static_cast<size_t>(new_pdf_index) < densities_.size());
  KALDI_ASSERT(static_cast<size_t>(tied_pdf_index) < tied_densities_.size());
  
  tied_densities_[tied_pdf_index]->SetPdfIndex(new_pdf_index);
}
  
/// Replace the designated codebook
void AmTiedDiagGmm::ReplacePdf(const int32 pdf_index, DiagGmm &gmm) {
  if (densities_.size() != 0)  // not the first gmm
    assert(static_cast<int32>(gmm.Dim()) == dim_);
  KALDI_ASSERT(static_cast<size_t>(pdf_index) < densities_.size());
 
  densities_[pdf_index]->CopyFromDiagGmm(gmm);
}

void AmTiedDiagGmm::CopyFromAmTiedDiagGmm(const AmTiedDiagGmm &other) {
  if (densities_.size() != 0) {
    DeletePointers(&densities_);
  }
  if (tied_densities_.size() != 0) {
    DeletePointers(&tied_densities_);
  }

  densities_.resize(other.NumPdfs(), NULL);
  tied_densities_.resize(other.NumTiedPdfs(), NULL);

  dim_ = other.dim_;

  for (int32 i = 0, end = densities_.size(); i < end; ++i) {
    densities_[i] = new DiagGmm();
    densities_[i]->CopyFromDiagGmm(*other.densities_[i]);
  }

  for (int32 i = 0, end = tied_densities_.size(); i < end; ++i) {
    tied_densities_[i] = new TiedGmm();
    tied_densities_[i]->CopyFromTiedGmm(*other.tied_densities_[i]);
  }
}

int32 AmTiedDiagGmm::ComputeGconsts() {
  int32 num_bad = 0;
  for (std::vector<DiagGmm*>::iterator itr = densities_.begin(),
      end = densities_.end(); itr != end; ++itr) {
    num_bad += (*itr)->ComputeGconsts();
  }
  
  for (std::vector<TiedGmm*>::iterator itr = tied_densities_.begin(),
      end = tied_densities_.end(); itr != end; ++itr) {
    num_bad += (*itr)->ComputeGconsts();
  }
  
  if (num_bad > 0)
    KALDI_WARN << "Found " << num_bad << " Gaussian components.";

  return num_bad;
}

void AmTiedDiagGmm::SetupPerFrameVars(TiedGmmPerFrameVars *per_frame_vars) const {
  per_frame_vars->x.Resize(dim_);
  per_frame_vars->Resize(NumPdfs());
  for (int32 i = 0; i < NumPdfs(); ++i)
    per_frame_vars->Resize(i, GetPdf(i).NumGauss());
}

void AmTiedDiagGmm::ComputePerFrameVars(const VectorBase<BaseFloat> &data, 
                                        TiedGmmPerFrameVars *per_frame_vars) {
  per_frame_vars->x.CopyFromVec(data);
  for (int32 i = 0; i < NumPdfs(); ++i)
    densities_[i]->LogLikelihoods(data, per_frame_vars->ll[i]);
}

void AmTiedDiagGmm::Read(std::istream &in_stream, bool binary) {
  int32 num_pdfs;
  int32 num_tied_pdfs;

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
  tied_densities_.reserve(num_pdfs);
  for (int32 i = 0; i < num_pdfs; i++) {
    tied_densities_.push_back(new TiedGmm());
    tied_densities_.back()->Read(in_stream, binary);
    KALDI_ASSERT(static_cast<int32>(tied_densities_.back()->Dim())
                 == dim_);
  }
}

void AmTiedDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, dim_);
 
  // write out codebooks
  WriteMarker(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, static_cast<int32>(densities_.size()));
  for (std::vector<DiagGmm*>::const_iterator it = densities_.begin(),
      end = densities_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }

  // write out tied pdfs
  WriteMarker(out_stream, binary, "<NUMTIEDPDFS>");
  WriteBasicType(out_stream, binary, static_cast<int32>(tied_densities_.size()));
  for (std::vector<TiedGmm*>::const_iterator it = tied_densities_.begin(),
      end = tied_densities_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}

}  // namespace kaldi
