// ivectorbin/ivector-compute-lda.cc

// Copyright 2013  Daniel Povey

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "thread/kaldi-task-sequence.h"

namespace kaldi {


class CovarianceStats {
 public:
  CovarianceStats(int32 dim): tot_covar_(dim),
                              between_covar_(dim),
                              num_spk_(0),
                              num_utt_(0) { }

  /// get total covariance, normalized per number of frames.
  void GetTotalCovar(SpMatrix<double> *tot_covar) const {
    KALDI_ASSERT(num_utt_ > 0);
    *tot_covar = tot_covar_;
    tot_covar->Scale(1.0 / num_utt_);
  }
  void GetWithinCovar(SpMatrix<double> *within_covar) {
    KALDI_ASSERT(num_utt_ - num_spk_ > 0);
    *within_covar = tot_covar_;
    within_covar->AddSp(-1.0, between_covar_);
    within_covar->Scale(1.0 / num_utt_);
  }
  void AccStats(const Matrix<double> &utts_of_this_spk) {
    int32 num_utts = utts_of_this_spk.NumRows();
    tot_covar_.AddMat2(1.0, utts_of_this_spk, kTrans, 1.0);
    Vector<double> spk_average(Dim());
    spk_average.AddRowSumMat(1.0 / num_utts, utts_of_this_spk);
    between_covar_.AddVec2(num_utts, spk_average);
    num_utt_ += num_utts;
    num_spk_ += 1;
  }
  /// Will return Empty() if the within-class covariance matrix would be zero.
  bool SingularTotCovar() { return (num_utt_ < Dim()); }
  bool Empty() { return (num_utt_ - num_spk_ == 0); }
  std::string Info() {
    std::ostringstream ostr;
    ostr << num_spk_ << " speakers, " << num_utt_ << " utterances. ";
    return ostr.str();
  }
  int32 Dim() { return tot_covar_.NumRows(); }
  // Use default constructor and assignment operator.
  void AddStats(const CovarianceStats &other) {
    tot_covar_.AddSp(1.0, other.tot_covar_);
    between_covar_.AddSp(1.0, other.between_covar_);
    num_spk_ += other.num_spk_;
    num_utt_ += other.num_utt_;
  }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CovarianceStats);
  SpMatrix<double> tot_covar_;
  SpMatrix<double> between_covar_;
  int32 num_spk_;
  int32 num_utt_;
};


template<class Real>
void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                 MatrixBase<Real> *proj) {
  int32 dim = covar.NumRows();
  TpMatrix<Real> C(dim);  // Cholesky of covar, covar = C C^T
  C.Cholesky(covar);
  C.Invert();  // The matrix that makes covar unit is C^{-1}, because
               // C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.
  proj->CopyFromTp(C, kNoTrans);  // set "proj" to C^{-1}.
}

void ComputeLdaTransform(
    const std::map<std::string, Vector<BaseFloat> *> &utt2ivector,
    const std::map<std::string, std::vector<std::string> > &spk2utt,
    BaseFloat total_covariance_factor,
    MatrixBase<BaseFloat> *lda_out) {
  KALDI_ASSERT(!utt2ivector.empty());
  int32 lda_dim = lda_out->NumRows(), dim = lda_out->NumCols();
  KALDI_ASSERT(dim == utt2ivector.begin()->second->Dim());
  KALDI_ASSERT(lda_dim > 0 && lda_dim <= dim);
  
  CovarianceStats stats(dim);
  
  std::map<std::string, std::vector<std::string> >::const_iterator iter;
  for (iter = spk2utt.begin(); iter != spk2utt.end(); ++iter) {
    const std::vector<std::string> &uttlist = iter->second;
    KALDI_ASSERT(!uttlist.empty());

    int32 N = uttlist.size(); // number of utterances.
    Matrix<double> utts_of_this_spk(N, dim);
    for (int32 n = 0; n < N; n++) {
      std::string utt = uttlist[n];
      KALDI_ASSERT(utt2ivector.count(utt) != 0);
      utts_of_this_spk.Row(n).CopyFromVec(
          *(utt2ivector.find(utt)->second));
    }
    stats.AccStats(utts_of_this_spk);
  }

  KALDI_LOG << "Stats have " << stats.Info();
  KALDI_ASSERT(!stats.Empty());
  KALDI_ASSERT(!stats.SingularTotCovar() &&
               "Too little data for iVector dimension.");


  SpMatrix<double> total_covar;
  stats.GetTotalCovar(&total_covar);
  SpMatrix<double> within_covar;
  stats.GetWithinCovar(&within_covar);


  SpMatrix<double> mat_to_normalize(dim);
  mat_to_normalize.AddSp(total_covariance_factor, total_covar);
  mat_to_normalize.AddSp(1.0 - total_covariance_factor, within_covar);
  
  Matrix<double> T(dim, dim); 
  ComputeNormalizingTransform(mat_to_normalize, &T);
  
  SpMatrix<double> between_covar(total_covar);
  between_covar.AddSp(-1.0, within_covar);

  SpMatrix<double> between_covar_proj(dim);
  between_covar_proj.AddMat2Sp(1.0, T, kNoTrans, between_covar, 0.0);

  Matrix<double> U(dim, dim);
  Vector<double> s(dim);
  between_covar_proj.Eig(&s, &U);
  bool sort_on_absolute_value = false; // any negative ones will go last (they
                                       // shouldn't exist anyway so doesn't
                                       // really matter)
  SortSvd(&s, &U, static_cast<Matrix<double>*>(NULL),
          sort_on_absolute_value);
  
  KALDI_LOG << "Singular values of between-class covariance after projecting "
            << "with interpolated [total/within] covariance with a weight of "
            << total_covariance_factor << " on the total covariance, are: " << s;

  // U^T is the transform that will diagonalize the between-class covariance.
  // U_part is just the part of U that corresponds to the kept dimensions.
  SubMatrix<double> U_part(U, 0, dim, 0, lda_dim);

  // We first transform by T and then by U_part^T.  This means T
  // goes on the right.
  Matrix<double> temp(lda_dim, dim);
  temp.AddMatMat(1.0, U_part, kTrans, T, kNoTrans, 0.0);
  lda_out->CopyFromMat(temp);
}

void ComputeAndSubtractMean(
    std::map<std::string, Vector<BaseFloat> *> utt2ivector,
    Vector<BaseFloat> *mean_out) {
  int32 dim = utt2ivector.begin()->second->Dim();
  size_t num_ivectors = utt2ivector.size();
  Vector<double> mean(dim);
  std::map<std::string, Vector<BaseFloat> *>::iterator iter;
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    mean.AddVec(1.0 / num_ivectors, *(iter->second));
  mean_out->Resize(dim);
  mean_out->CopyFromVec(mean);
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    iter->second->AddVec(-1.0, *mean_out);
}



}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute an LDA matrix for iVector system.  Reads in iVectors per utterance,\n"
        "and an utt2spk file which it uses to help work out the within-speaker and\n"
        "between-speaker covariance matrices.  Outputs an LDA projection to a\n"
        "specified dimension.  By default it will normalize so that the projected\n"
        "within-class covariance is unit, but if you set --normalize-total-covariance\n"
        "to true, it will normalize the total covariance.\n"
        "Note: the transform we produce is actually an affine transform which will\n"
        "also set the global mean to zero.\n"
        "\n"
        "Usage:  ivector-compute-lda [options] <ivector-rspecifier> <utt2spk-rspecifier> "
        "<lda-matrix-out>\n"
        "e.g.: \n"
        " ivector-compute-lda ark:ivectors.ark ark:utt2spk lda.mat\n";
    
    ParseOptions po(usage);

    int32 lda_dim = 100; // Dimension we reduce to
    BaseFloat total_covariance_factor = 0.0;
    bool binary = true;    

    po.Register("dim", &lda_dim, "Dimension we keep with the LDA transform");
    po.Register("total-covariance-factor", &total_covariance_factor,
                "If this is 0.0 we normalize to make the within-class covariance "
                "unit; if 1.0, the total covariance; if between, we normalize "
                "an interpolated matrix.");
    po.Register("binary", &binary, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_rspecifier = po.GetArg(1),
        utt2spk_rspecifier = po.GetArg(2),
        lda_wxfilename = po.GetArg(3);
    
    int32 num_done = 0, num_err = 0, dim = 0;
    
    SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);
    
    std::map<std::string, Vector<BaseFloat> *> utt2ivector;
    std::map<std::string, std::vector<std::string> > spk2utt;

    for (; !ivector_reader.Done(); ivector_reader.Next()) {
      std::string utt = ivector_reader.Key();
      const Vector<BaseFloat> &ivector = ivector_reader.Value();
      if (utt2ivector.count(utt) != 0) {
        KALDI_WARN << "Duplicate iVector found for utterance " << utt
                   << ", ignoring it.";
        num_err++;
        continue;
      }
      if (!utt2spk_reader.HasKey(utt)) {
        KALDI_WARN << "utt2spk has no entry for utterance " << utt
                   << ", skipping it.";
        num_err++;
        continue;
      }
      std::string spk = utt2spk_reader.Value(utt);
      utt2ivector[utt] = new Vector<BaseFloat>(ivector);
      if (dim == 0) {
        dim = ivector.Dim();
      } else {
        KALDI_ASSERT(dim == ivector.Dim() && "iVector dimension mismatch");
      }
      spk2utt[spk].push_back(utt);
      num_done++;
    }

    KALDI_LOG << "Read " << num_done << " utterances, "
              << num_err << " with errors.";

    if (num_done == 0) {
      KALDI_ERR << "Did not read any utterances.";
    } else {
      KALDI_LOG << "Computing within-class covariance.";
    }

    Vector<BaseFloat> mean;
    ComputeAndSubtractMean(utt2ivector, &mean);
    KALDI_LOG << "2-norm of iVector mean is " << mean.Norm(2.0);

    
    Matrix<BaseFloat> lda_mat(lda_dim, dim + 1); // LDA matrix without the offset term.
    SubMatrix<BaseFloat> linear_part(lda_mat, 0, lda_dim, 0, dim);
    ComputeLdaTransform(utt2ivector,
                        spk2utt,
                        total_covariance_factor,
                        &linear_part);
    Vector<BaseFloat> offset(lda_dim);
    offset.AddMatVec(-1.0, linear_part, kNoTrans, mean, 0.0);
    lda_mat.CopyColFromVec(offset, dim); // add mean-offset to transform
    
    KALDI_VLOG(2) << "2-norm of transformed iVector mean is "
                  << offset.Norm(2.0);
    
    WriteKaldiObject(lda_mat, lda_wxfilename, binary);

    KALDI_LOG << "Wrote LDA transform to "
              << PrintableWxfilename(lda_wxfilename);
    
    std::map<std::string, Vector<BaseFloat> *>::iterator iter;
    for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
      delete iter->second;
    utt2ivector.clear();

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
