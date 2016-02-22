// segmenterbin/logistic-regression-train-on-feats.cc

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
        "Trains a model using Logistic Regression with L-BFGS from "
        "a set of features, where each row corresponds to a frame.\n"
        "The corresponding frame labels are in <labels-rspecifier> "
        "which is a vector of integers with one label for each frame.\n"
        "The number of targets is input by the user; the labels must be "
        "between 0 and <num-targets>-1.\n"
        "Usage: logistic-regression-train-on-feats <feats-rspecifier>\n"
        "<labels-rspecifier> <model-out>\n";

    ParseOptions po(usage);

    bool binary = true;
    int32 num_targets = 2;
    int32 num_frames = 200000;
    int32 srand_seed = 0;
    std::string model_rxfilename;

    LogisticRegressionConfig config;
    config.Register(&po);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-targets", &num_targets, "Number of target labels");
    po.Register("num-frames", &num_frames, 
                "Number of feature vectors to store in "
                "memory and train on (randomly chosen from the input features)");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("model-rxfilename", &model_rxfilename, "Initialize "
                "logistic-regression model");

    po.Read(argc, argv);
    
    srand(srand_seed);    

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feats_rspecifier = po.GetArg(1),
                labels_rspecifier = po.GetArg(2),
                model_out = po.GetArg(3);

    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
    RandomAccessInt32VectorReader labels_reader(labels_rspecifier);

    Matrix<BaseFloat> feats;
    std::vector<int32> labels(num_frames, -1);

    KALDI_ASSERT(num_frames > 0);
    KALDI_LOG << "Reading features (will keep " << num_frames << " frames.)";

    int64 num_read = 0, dim = 0;

    for (; !feats_reader.Done(); feats_reader.Next()) {
      const std::string &key = feats_reader.Key();
      const Matrix<BaseFloat>  &this_feats = feats_reader.Value();

      if (!labels_reader.HasKey(key)) {
        KALDI_WARN << "No labels found for utterance " << key;
        continue;
      }

      std::vector<int32> this_labels = labels_reader.Value(key);

      for (int32 t = 0; t < this_feats.NumRows(); t++) {
        num_read++;
        if (dim == 0) {
          dim = this_feats.NumCols();
          feats.Resize(num_frames, dim);
        } else if (this_feats.NumCols() != dim) {
          KALDI_ERR << "Features have inconsistent dims "
                    << this_feats.NumCols() << " vs. " << dim
                    << " (current utt is) " << feats_reader.Key();
        }
        if (num_read <= num_frames) {
          feats.Row(num_read - 1).CopyFromVec(this_feats.Row(t));
          labels[num_read - 1] = this_labels[t];
        } else {
          BaseFloat keep_prob = num_frames / static_cast<BaseFloat>(num_read);
          if (WithProb(keep_prob)) { // With probability "keep_prob"
            int32 t1 = RandInt(0, num_frames - 1);
            feats.Row(t1).CopyFromVec(this_feats.Row(t));
            if ( this_labels[t] < 0 || this_labels[t] >= num_targets ) {
              KALDI_ERR << "Label must be between 0 and <num-targets>-1; " 
                        << "; but found label " << this_labels[t];
            }
            labels[t1] = this_labels[t];
          }
        }
      }
    }
    
    if (num_read < num_frames) {
      KALDI_WARN << "Number of frames read " << num_read << " was less than "
                 << "target number " << num_frames << ", using all we read.";
      feats.Resize(num_read, dim, kCopyData);
    } else {
      BaseFloat percent = num_frames * 100.0 / num_read;
      KALDI_LOG << "Kept " << num_frames << " out of " << num_read
                << " input frames = " << percent << "%.";
    }
    
    LogisticRegression classifier = LogisticRegression();

    if (!model_rxfilename.empty()) {
      ReadKaldiObject(model_rxfilename, &classifier);
    }

    classifier.Train(feats, labels, config);
    WriteKaldiObject(classifier, model_out, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

