// ivectorbin/logistic-regression-eval.cc

// Copyright 2014  David Snyder

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

using namespace kaldi;

int ComputeLogPosteriors(ParseOptions &po, const LogisticRegressionConfig &config) {
  std::string model = po.GetArg(1),
      vector_rspecifier = po.GetArg(2),
      log_posteriors_wspecifier = po.GetArg(3);
  
  LogisticRegression classifier;
  ReadKaldiObject(model, &classifier);
  
  std::vector<Vector<BaseFloat> > vectors;
  SequentialBaseFloatVectorReader vector_reader(vector_rspecifier);
  BaseFloatVectorWriter posterior_writer(log_posteriors_wspecifier);
  std::vector<std::string> utt_list;
  int32 num_utt_done = 0;

  for (; !vector_reader.Done(); vector_reader.Next()) {
    std::string utt = vector_reader.Key();
    const Vector<BaseFloat> &vector = vector_reader.Value();
    Vector<BaseFloat> log_posteriors;
    classifier.GetLogPosteriors(vector, &log_posteriors);
    posterior_writer.Write(utt, log_posteriors);
    num_utt_done++;
  }
  KALDI_LOG << "Calculated log posteriors for " << num_utt_done << " vectors.";
  return (num_utt_done == 0 ? 1 : 0);
}

int32 ComputeScores(ParseOptions &po, const LogisticRegressionConfig &config) {
  std::string model_rspecifier = po.GetArg(1),
      trials_rspecifier = po.GetArg(2),
      vector_rspecifier = po.GetArg(3),
      scores_out = po.GetArg(4);

  SequentialInt32Reader class_reader(trials_rspecifier);
  LogisticRegression classifier = LogisticRegression();
  ReadKaldiObject(model_rspecifier, &classifier);

  std::vector<Vector<BaseFloat> > vectors;
  std::vector<int32> ys;
  std::vector<std::string> utt_list;
  int32 num_utt_done = 0, num_utt_err = 0;

  RandomAccessBaseFloatVectorReader vector_reader(vector_rspecifier);
  for (; !class_reader.Done(); class_reader.Next()) {
    std::string utt = class_reader.Key();
    int32 class_label = class_reader.Value();
    if (!vector_reader.HasKey(utt)) {
      KALDI_WARN << "No vector for utterance " << utt;
      num_utt_err++;
    } else {
      utt_list.push_back(utt);
      ys.push_back(class_label);
      const Vector<BaseFloat> &vector = vector_reader.Value(utt);
      vectors.push_back(vector);
      num_utt_done++;
    }
  }

  if (vectors.empty()) {
    KALDI_WARN << "Read no input";
    return 1;
  }
  
  Matrix<BaseFloat> xs(vectors.size(), vectors[0].Dim());
  for (int i = 0; i < vectors.size(); i++) {
    xs.Row(i).CopyFromVec(vectors[i]);
  }
 
  Matrix<BaseFloat> log_posteriors;
  classifier.GetLogPosteriors(xs, &log_posteriors);

  bool binary = false;
  Output ko(scores_out.c_str(), binary);
  
  for (int i = 0; i < ys.size(); i++) {
    ko.Stream() << utt_list[i] << " " << ys[i] << " " << log_posteriors(i, ys[i]) << std::endl;
  }
  KALDI_LOG << "Calculated scores for " << num_utt_done 
            << " vectors with "
            << num_utt_err << " missing. ";
  return (num_utt_done == 0 ? 1 : 0);
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Evaluates a model on input vectors and outputs either\n"
        "log posterior probabilities or scores.\n"
        "Usage1: logistic-regression-eval <model> <input-vectors-rspecifier>\n"
        "                                <output-log-posteriors-wspecifier>\n"
        "Usage2: logistic-regression-eval <model> <trials-file> <input-vectors-rspecifier>\n"
        "                                <output-scores-file>\n";
    
  ParseOptions po(usage);

  bool binary = false;
  LogisticRegressionConfig config;
  config.Register(&po);
  po.Register("binary", &binary, "Write output in binary mode");
  po.Read(argc, argv);

  if (po.NumArgs() != 3 && po.NumArgs() != 4) {
    po.PrintUsage();
    exit(1);
  }
  
  return (po.NumArgs() == 4) ?
      ComputeScores(po, config) :
      ComputeLogPosteriors(po, config);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
