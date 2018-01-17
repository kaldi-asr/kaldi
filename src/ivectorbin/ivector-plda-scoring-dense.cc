// ivectorbin/ivector-plda-scoring-dense.cc

// Copyright 2016-2018  David Snyder
//           2017-2018  Matthew Maciejewski

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
#include "util/stl-utils.h"
#include "ivector/plda.h"

namespace kaldi {

bool EstPca(const Matrix<BaseFloat> &ivector_mat, BaseFloat target_energy,
  Matrix<BaseFloat> *mat) {
  int32 num_rows = ivector_mat.NumRows(),
    num_cols = ivector_mat.NumCols();
  Vector<BaseFloat> sum;
  SpMatrix<BaseFloat> sumsq;
  sum.Resize(num_cols);
  sumsq.Resize(num_cols);
  sum.AddRowSumMat(1.0, ivector_mat);
  sumsq.AddMat2(1.0, ivector_mat, kTrans, 1.0);
  sum.Scale(1.0 / num_rows);
  sumsq.Scale(1.0 / num_rows);
  sumsq.AddVec2(-1.0, sum); // now sumsq is centered covariance.
  int32 full_dim = sum.Dim();

  Matrix<BaseFloat> P(full_dim, full_dim);
  Vector<BaseFloat> s(full_dim);

  try {
    if (num_rows > num_cols)
      sumsq.Eig(&s, &P);
    else
      Matrix<BaseFloat>(sumsq).Svd(&s, &P, NULL);
  } catch (...) {
    return false;
  }

  SortSvd(&s, &P);

  Matrix<BaseFloat> transform(P, kTrans); // Transpose of P.  This is what
                                       // appears in the transform.

  // We want the PCA transform to retain target_energy amount of the total
  // energy.
  BaseFloat total_energy = s.Sum();
  BaseFloat energy = 0.0;
  int32 dim = 1;
  while (energy / total_energy <= target_energy) {
    energy += s(dim-1);
    dim++;
  }
  Matrix<BaseFloat> transform_float(transform);
  mat->Resize(transform.NumCols(), transform.NumRows());
  mat->CopyFromMat(transform);
  mat->Resize(dim, transform_float.NumCols(), kCopyData);
  return true;
}

// Transforms i-vectors using the PLDA model.
void TransformIvectors(const Matrix<BaseFloat> &ivectors_in,
  const PldaConfig &plda_config, const Plda &plda,
  Matrix<BaseFloat> *ivectors_out) {
  int32 dim = plda.Dim();
  ivectors_out->Resize(ivectors_in.NumRows(), dim);
  for (int32 i = 0; i < ivectors_in.NumRows(); i++) {
    Vector<BaseFloat> transformed_ivector(dim);
    plda.TransformIvector(plda_config, ivectors_in.Row(i), 1.0,
      &transformed_ivector);
    ivectors_out->Row(i).CopyFromVec(transformed_ivector);
  }
}

// Transform the i-vectors using the recording-dependent PCA matrix.
void ApplyPca(const Matrix<BaseFloat> &ivectors_in,
  const Matrix<BaseFloat> &pca_mat, Matrix<BaseFloat> *ivectors_out) {
  int32 transform_cols = pca_mat.NumCols(),
        transform_rows = pca_mat.NumRows(),
        feat_dim = ivectors_in.NumCols();
  ivectors_out->Resize(ivectors_in.NumRows(), transform_rows);
  KALDI_ASSERT(transform_cols == feat_dim);
  ivectors_out->AddMatMat(1.0, ivectors_in, kNoTrans,
    pca_mat, kTrans, 0.0);
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Perform PLDA scoring for speaker diarization.  The input reco2utt\n"
      "should be of the form <recording-id> <seg1> <seg2> ... <segN> and\n"
      "there should be one iVector for each segment.  PLDA scoring is\n"
      "performed between all pairs of iVectors in a recording and outputs\n"
      "an archive of score matrices, one for each recording-id.  The rows\n"
      "and columns of the the matrix correspond the sorted order of the\n"
      "segments.\n"
      "Usage: ivector-plda-scoring-dense [options] <plda> <reco2utt>"
      " <ivectors-rspecifier> <scores-wspecifier>\n"
      "e.g.: \n"
      "  ivector-plda-scoring-dense plda reco2utt scp:ivectors.scp"
      " ark:scores.ark ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    BaseFloat target_energy = 0.5;
    PldaConfig plda_config;
    plda_config.Register(&po);

    po.Register("target-energy", &target_energy,
      "Reduce dimensionality of i-vectors using a recording-dependent"
      " PCA such that this fraction of the total energy remains.");
    KALDI_ASSERT(target_energy <= 1.0);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      ivector_rspecifier = po.GetArg(3),
      scores_wspecifier = po.GetArg(4);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatMatrixWriter scores_writer(scores_wspecifier);
    int32 num_reco_err = 0,
          num_reco_done = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      Plda this_plda(plda);
      std::string reco = reco2utt_reader.Key();

      std::vector<std::string> uttlist = reco2utt_reader.Value();
      std::vector<Vector<BaseFloat> > ivectors;

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];

        if (!ivector_reader.HasKey(utt)) {
          KALDI_ERR << "No iVector present in input for utterance " << utt;
        }

        Vector<BaseFloat> ivector = ivector_reader.Value(utt);
        ivectors.push_back(ivector);
      }
      if (ivectors.size() == 0) {
        KALDI_WARN << "Not producing output for recording " << reco
                   << " since no segments had iVectors";
        num_reco_err++;
      } else {
        Matrix<BaseFloat> ivector_mat(ivectors.size(), ivectors[0].Dim()),
                          ivector_mat_pca,
                          ivector_mat_plda,
                          pca_transform,
                          scores(ivectors.size(), ivectors.size());

        for (size_t i = 0; i < ivectors.size(); i++) {
          ivector_mat.Row(i).CopyFromVec(ivectors[i]);
        }
        if (EstPca(ivector_mat, target_energy, &pca_transform)) {
          // Apply the PCA transform to the raw i-vectors.
          ApplyPca(ivector_mat, pca_transform, &ivector_mat_pca);

          // Apply the PCA transform to the parameters of the PLDA model.
          this_plda.ApplyTransform(Matrix<double>(pca_transform));

          // Now transform the i-vectors using the reduced PLDA model.
          TransformIvectors(ivector_mat_pca, plda_config, this_plda,
            &ivector_mat_plda);
        } else {
          KALDI_WARN << "Unable to compute conversation dependent PCA for"
            << " recording " << reco << ".";
          ivector_mat_pca.Resize(ivector_mat.NumRows(), ivector_mat.NumCols());
          ivector_mat_pca.CopyFromMat(ivector_mat);
        }
        for (int32 i = 0; i < ivector_mat_plda.NumRows(); i++) {
          for (int32 j = 0; j < ivector_mat_plda.NumRows(); j++) {
            scores(i, j) = this_plda.LogLikelihoodRatio(Vector<double>(
              ivector_mat_plda.Row(i)), 1.0,
              Vector<double>(ivector_mat_plda.Row(j)));
          }
        }
        scores_writer.Write(reco, scores);
        num_reco_done++;
      }
    }
    KALDI_LOG << "Processed " << num_reco_done << " recordings, "
              << num_reco_err << " had errors.";
    return (num_reco_done != 0 ? 0 : 1 );
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
