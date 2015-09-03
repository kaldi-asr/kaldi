// ivectorbin/logistic-regression-train.cc

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


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Trains a model using Logistic Regression with L-BFGS from\n"
        "a set of vectors. The class labels in <classes-rspecifier>\n"
        "must be a set of integers such that there are no gaps in \n"
        "its range and the smallest label must be 0.\n" 
        "Usage: logistic-regression-train <vector-rspecifier>\n"
        "<classes-rspecifier> <model-out>\n";
    
    ParseOptions po(usage);

    bool binary = true;
    LogisticRegressionConfig config;
    config.Register(&po);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector_rspecifier = po.GetArg(1),
        class_rspecifier = po.GetArg(2),
        model_out = po.GetArg(3);

    RandomAccessBaseFloatVectorReader vector_reader(vector_rspecifier);
    SequentialInt32Reader class_reader(class_rspecifier);
    
    std::vector<int32> ys;
    std::vector<std::string> utt_ids;
    std::vector<Vector<BaseFloat> > vectors;

    int32 num_utt_done = 0, num_utt_err = 0;

    int32 num_classes = 0;
    for (; !class_reader.Done(); class_reader.Next()) {
      std::string utt = class_reader.Key();
      int32 class_label = class_reader.Value();
      if (!vector_reader.HasKey(utt)) {
        KALDI_WARN << "No vector for utterance " << utt;
        num_utt_err++;
      } else {
        ys.push_back(class_label);
        const Vector<BaseFloat> &vector = vector_reader.Value(utt);
        vectors.push_back(vector);
    
        // Since there are no gaps in the class labels and we
        // start at 0, the largest label is the number of the
        // of the classes - 1.
        if (class_label > num_classes) {
          num_classes = class_label;
        }
        num_utt_done++;
      }
    }

    // Since the largest label is 1 minus the number of
    // classes.
    num_classes += 1;

    KALDI_LOG << "Retrieved " << num_utt_done << " vectors with "
              << num_utt_err << " missing. "
              << "There were " << num_classes << " class labels.";

    if (num_utt_done == 0)
      KALDI_ERR << "No vectors processed. Unable to train.";

    Matrix<BaseFloat> xs(vectors.size(), vectors[0].Dim());
    for (int i = 0; i < vectors.size(); i++) {
      xs.Row(i).CopyFromVec(vectors[i]);
    }
    vectors.clear();
  
    LogisticRegression classifier = LogisticRegression();
    classifier.Train(xs, ys, config);
    WriteKaldiObject(classifier, model_out, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
