// featbin/transform-feats.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Apply transform (e.g. LDA; HLDA; fMLLR/CMLLR; MLLT/STC)\n"
        "Linear transform if transform-num-cols == feature-dim, affine if\n"
        "transform-num-cols == feature-dim+1 (->append 1.0 to features)\n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Global if transform-rxfilename provided.\n"
        "Usage: transform-feats [options] (<transform-rspecifier>|<transform-rxfilename>) <feats-rspecifier> <feats-wspecifier>\n"
        "See also: transform-vec, copy-feats, compose-transforms\n";
        
    ParseOptions po(usage);
    std::string utt2spk_rspecifier;
    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to speaker map");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rspecifier_or_rxfilename = po.GetArg(1);
    std::string feat_rspecifier = po.GetArg(2);
    std::string feat_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    RandomAccessBaseFloatMatrixReaderMapped transform_reader;
    bool use_global_transform;
    Matrix<BaseFloat> global_transform;
    if (ClassifyRspecifier(transform_rspecifier_or_rxfilename, NULL, NULL)
       == kNoRspecifier) {
      // not an rspecifier -> interpret as rxfilename....
      use_global_transform = true;
      ReadKaldiObject(transform_rspecifier_or_rxfilename, &global_transform);
    } else {  // an rspecifier -> not a global transform.
      use_global_transform = false;
      if (!transform_reader.Open(transform_rspecifier_or_rxfilename,
                                 utt2spk_rspecifier)) {
        KALDI_ERR << "Problem opening transforms with rspecifier "
                  << '"' << transform_rspecifier_or_rxfilename << '"'
                  << " and utt2spk rspecifier "
                  << '"' << utt2spk_rspecifier << '"';
      }
    }

    enum { Unknown, Logdet, PseudoLogdet, DimIncrease };
    int32 logdet_type = Unknown;
    double tot_t = 0.0, tot_logdet = 0.0;  // to compute average logdet weighted by time...
    int32 num_done = 0, num_error = 0;
    BaseFloat cached_logdet = -1;
    
    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat(feat_reader.Value());

      if (!use_global_transform && !transform_reader.HasKey(utt)) {
        KALDI_WARN << "No fMLLR transform available for utterance "
                   << utt << ", producing no output for this utterance";
        num_error++;
        continue;
      }
      const Matrix<BaseFloat> &trans =
          (use_global_transform ? global_transform : transform_reader.Value(utt));
      int32 transform_rows = trans.NumRows(),
          transform_cols = trans.NumCols(),
          feat_dim = feat.NumCols();

      Matrix<BaseFloat> feat_out(feat.NumRows(), transform_rows);

      if (transform_cols == feat_dim) {
        feat_out.AddMatMat(1.0, feat, kNoTrans, trans, kTrans, 0.0);
      } else if (transform_cols == feat_dim + 1) {
        // append the implicit 1.0 to the input features.
        SubMatrix<BaseFloat> linear_part(trans, 0, transform_rows, 0, feat_dim);
        feat_out.AddMatMat(1.0, feat, kNoTrans, linear_part, kTrans, 0.0);
        Vector<BaseFloat> offset(transform_rows);
        offset.CopyColFromMat(trans, feat_dim);
        feat_out.AddVecToRows(1.0, offset);
      } else {
        KALDI_WARN << "Transform matrix for utterance " << utt << " has bad dimension "
                   << transform_rows << "x" << transform_cols << " versus feat dim "
                   << feat_dim;
        if (transform_cols == feat_dim+2)
          KALDI_WARN << "[perhaps the transform was created by compose-transforms, "
              "and you forgot the --b-is-affine option?]";
        num_error++;
        continue;
      }
      num_done++;

      if (logdet_type == Unknown) {
        if (transform_rows == feat_dim) logdet_type = Logdet;  // actual logdet.
        else if (transform_rows < feat_dim) logdet_type = PseudoLogdet;  // see below
        else logdet_type = DimIncrease;  // makes no sense to have any logdet.
        // PseudoLogdet is if we have a dimension-reducing transform T, we compute
        // 1/2 logdet(T T^T).  Why does this make sense?  Imagine we do MLLT after
        // LDA and compose the transforms; the MLLT matrix is A and the LDA matrix is L,
        // so T = A L.  T T^T = A L L^T A, so 1/2 logdet(T T^T) = logdet(A) + 1/2 logdet(L L^T).
        // since L L^T is a constant, this is valid for comparing likelihoods if we're
        // just trying to see if the MLLT is converging.
      }

      if (logdet_type != DimIncrease) { // Accumulate log-determinant stats.
        SubMatrix<BaseFloat> linear_transform(trans, 0, trans.NumRows(), 0, feat_dim);
        // "linear_transform" is just the linear part of any transform, ignoring
        // any affine (offset) component.
        SpMatrix<BaseFloat> TT(trans.NumRows());
        // TT = linear_transform * linear_transform^T
        TT.AddMat2(1.0, linear_transform, kNoTrans, 0.0);
        BaseFloat logdet;
        if (use_global_transform) {
          if (cached_logdet == -1)
            cached_logdet = 0.5 * TT.LogDet(NULL);
          logdet = cached_logdet;
        } else {
          logdet = 0.5 * TT.LogDet(NULL);
        }
        if (logdet != logdet || logdet-logdet != 0.0) // NaN or info.
          KALDI_WARN << "Matrix has bad logdet " << logdet;
        else {
          tot_t += feat.NumRows();
          tot_logdet += feat.NumRows() * logdet;
        }
      }
      feat_writer.Write(utt, feat_out);
    }
    if (logdet_type != Unknown && logdet_type != DimIncrease)
      KALDI_LOG << "Overall average " << (logdet_type == PseudoLogdet ? "[pseudo-]":"")
                << "logdet is " << (tot_logdet/tot_t) << " over " << tot_t
                << " frames.";
    KALDI_LOG << "Applied transform to " << num_done << " utterances; " << num_error
              << " had errors.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
