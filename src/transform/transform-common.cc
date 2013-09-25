// transform/transform-common.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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

#include <vector>

#include "base/kaldi-common.h"
#include "transform/transform-common.h"

namespace kaldi {


void AffineXformStats::Init(int32 dim, int32 num_gs) {
  if (dim == 0) {
    if (num_gs != 0) {
      KALDI_WARN << "Ignoring 'num_gs' (=" << num_gs << ") argument since "
                 << "dim = 0.";
    }
    beta_ = 0.0;
    K_.Resize(0, 0);
    G_.clear();
    dim_ = 0;
  } else {
    beta_ = 0.0;
    K_.Resize(dim, dim + 1, kSetZero);
    G_.resize(num_gs);
    for (int32 i = 0; i < num_gs; i++)
      G_[i].Resize(dim + 1, kSetZero);
    dim_ = dim;
  }
}

void AffineXformStats::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<DIMENSION>");
  WriteBasicType(out, binary, dim_);
  if (!binary) out << '\n';
  WriteToken(out, binary, "<BETA>");
  WriteBasicType(out, binary, beta_);
  if (!binary) out << '\n';
  WriteToken(out, binary, "<K>");
  Matrix<BaseFloat> tmp_k(K_);
  tmp_k.Write(out, binary);
  WriteToken(out, binary, "<G>");
  int32 g_size = static_cast<int32>(G_.size());
  WriteBasicType(out, binary, g_size);
  if (!binary) out << '\n';
  for (std::vector< SpMatrix<double> >::const_iterator itr = G_.begin(),
      end = G_.end(); itr != end; ++itr) {
    SpMatrix<BaseFloat> tmp_g(*itr);
    tmp_g.Write(out, binary);
  }
}

void AffineXformStats::Read(std::istream &in, bool binary, bool add) {
  ExpectToken(in, binary, "<DIMENSION>");
  ReadBasicType(in, binary, &dim_);
  ExpectToken(in, binary, "<BETA>");
  ReadBasicType(in, binary, &beta_);
  ExpectToken(in, binary, "<K>");
  Matrix<BaseFloat> tmp_k;
  tmp_k.Read(in, binary);
  K_.Resize(tmp_k.NumRows(), tmp_k.NumCols());
  if (add) {
    Matrix<double> tmp_k_d(tmp_k);
    K_.AddMat(1.0, tmp_k_d, kNoTrans);
  } else {
    K_.CopyFromMat(tmp_k, kNoTrans);
  }
  ExpectToken(in, binary, "<G>");
  int32 g_size;
  ReadBasicType(in, binary, &g_size);
  G_.resize(g_size);
  SpMatrix<BaseFloat> tmp_g;
  SpMatrix<double> tmp_g_d;
  if (add) { tmp_g_d.Resize(tmp_g.NumRows()); }
  for (size_t i = 0; i < G_.size(); i++) {
    tmp_g.Read(in, binary, false /*no add*/);
    G_[i].Resize(tmp_g.NumRows());
    if (add) {
      tmp_g_d.CopyFromSp(tmp_g);
      G_[i].AddSp(1.0, tmp_g_d);
    } else {
      G_[i].CopyFromSp(tmp_g);
    }
  }
}



void AffineXformStats::SetZero() {
  beta_ = 0.0;
  K_.SetZero();
  for (std::vector< SpMatrix<double> >::iterator it = G_.begin(),
      end = G_.end(); it != end; ++it) {
    it->SetZero();
  }
}

void AffineXformStats::CopyStats(const AffineXformStats &other) {
  KALDI_ASSERT(G_.size() == other.G_.size());
  KALDI_ASSERT(dim_ == other.dim_);
  beta_ = other.beta_;
  K_.CopyFromMat(other.K_, kNoTrans);
  for (size_t i = 0; i < G_.size(); i++)
    G_[i].CopyFromSp(other.G_[i]);
}

void AffineXformStats::Add(const AffineXformStats &other) {
  KALDI_ASSERT(G_.size() == other.G_.size());
  KALDI_ASSERT(dim_ == other.dim_);
  beta_ += other.beta_;
  K_.AddMat(1.0, other.K_, kNoTrans);
  for (size_t i = 0; i < G_.size(); i++)
    G_[i].AddSp(1.0, other.G_[i]);
}

bool ComposeTransforms(const Matrix<BaseFloat> &a, const Matrix<BaseFloat> &b,
                       bool b_is_affine,
                       Matrix<BaseFloat> *c) {
  if (b.NumRows() == 0 || a.NumCols() == 0) {
    KALDI_WARN  << "Empty matrix in ComposeTransforms\n";
    return false;
  }
  if (a.NumCols() == b.NumRows()) {
    c->Resize(a.NumRows(), b.NumCols());
    c->AddMatMat(1.0, a, kNoTrans, b, kNoTrans, 0.0);  // c = a * b.
    return true;
  } else if (a.NumCols() == b.NumRows()+1) {  // a is affine.
    if (b_is_affine) {  // append 0 0 0 0 ... 1 to b and multiply.
      Matrix<BaseFloat> b_ext(b.NumRows()+1, b.NumCols());
      SubMatrix<BaseFloat> b_part(b_ext, 0, b.NumRows(), 0, b.NumCols());
      b_part.CopyFromMat(b);
      b_ext(b.NumRows(), b.NumCols()-1) = 1.0;  // so the last row is 0 0 0 0 ... 0 1
      c->Resize(a.NumRows(), b.NumCols());
      c->AddMatMat(1.0, a, kNoTrans, b_ext, kNoTrans, 0.0);  // c = a * b_ext.
    } else {  // extend b by 1 row and column with all zeros except a 1 on diagonal.
      Matrix<BaseFloat> b_ext(b.NumRows()+1, b.NumCols()+1);
      SubMatrix<BaseFloat> b_part(b_ext, 0, b.NumRows(), 0, b.NumCols());
      b_part.CopyFromMat(b);
      b_ext(b.NumRows(), b.NumCols()) = 1.0;  // so the last row is 0 0 0 0 ... 0 1;
      // rest of last column is zero (this is the offset term)
      c->Resize(a.NumRows(), b.NumCols()+1);
      c->AddMatMat(1.0, a, kNoTrans, b_ext, kNoTrans, 0.0);  // c = a * b_ext.
    }
    return true;
  } else {
    KALDI_ERR << "ComposeTransforms: mismatched dimensions, a has " << a.NumCols()
              << " columns and b has " << b.NumRows() << " rows.";  // this is fatal.
    return false;
  }
}

void ApplyAffineTransform(const MatrixBase<BaseFloat> &xform,
                          VectorBase<BaseFloat> *vec) {
  int32 dim = xform.NumRows();
  KALDI_ASSERT(dim > 0 && xform.NumCols() == dim+1 && vec->Dim() == dim);
  Vector<BaseFloat> tmp(dim+1);
  SubVector<BaseFloat> tmp_part(tmp, 0, dim);
  tmp_part.CopyFromVec(*vec);
  tmp(dim) = 1.0;
  // next line is: vec = 1.0 * xform * tmp + 0.0 * vec
  vec->AddMatVec(1.0, xform, kNoTrans, tmp, 0.0);
}

}  // namespace kaldi

