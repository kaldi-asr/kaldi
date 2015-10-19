// ivectorbin/logistic-regression-eval-on-feats.cc

// Copyright 2014  David Snyder
//           2015  Vimal Manohar

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
#include "ivector/logistic-regression.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Evaluates a model on input vectors and outputs either\n"
        "log posterior probabilities or scores.\n"
        "Usage1: logistic-regression-eval <model> <feats-rspecifier> "
        "<output-log-posteriors-wspecifier>\n";

  ParseOptions po(usage);

  bool apply_log = true;
  po.Register("apply-log", &apply_log,
              "If false, apply Exp to the log posteriors output. This is "
              "helpful when combining posteriors from multiple logistic "
              "regression models.");
  LogisticRegressionConfig config;
  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() != 3) {
    po.PrintUsage();
    exit(1);
  }

  std::string model = po.GetArg(1),
              feats_rspecifier = po.GetArg(2),
              log_posteriors_wspecifier = po.GetArg(3);

  LogisticRegression classifier;
  ReadKaldiObject(model, &classifier);

  Matrix<BaseFloat> feats; 

  SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
  BaseFloatMatrixWriter log_probs_writer(log_posteriors_wspecifier);

  int32 num_utt_done = 0;

  for (; !feats_reader.Done(); feats_reader.Next()) {
    const std::string &key = feats_reader.Key();
    const Matrix<BaseFloat> &feats = feats_reader.Value();

    Matrix<BaseFloat> log_posteriors;

    classifier.GetLogPosteriors(feats, &log_posteriors);
    if (!apply_log) 
      log_posteriors.ApplyExp();

    log_probs_writer.Write(key, log_posteriors);
    num_utt_done++;
  }

  KALDI_LOG << "Calculated log posteriors for " << num_utt_done << " vectors.";
  return (num_utt_done == 0 ? 1 : 0);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

