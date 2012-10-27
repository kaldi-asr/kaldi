// nnet-cpubin/nnet-train.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet-cpu/nnet-randomize.h"
#include "nnet-cpu/train-nnet.h"
#include "nnet-cpu/am-nnet.h"


/*
  Note: the features will be split into validation and train parts
  while we're still at the scp files, using the commands
  grep -w -F -f valid_uttlist
  grep -v -w -F -f valid_uttlist
  or filter_scp.pl (using the --exclude option for the second case).
*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train the neural network parameters with backprop and stochastic\n"
        "gradient descent using minibatches.  The training frames and labels\n"
        "are read via a pipe from nnet-randomize-frames.  This program uses a\n"
        "heuristic based on the validation set gradient to periodically update\n"
        "the learning rate of each layer.  It will train until the data from\n"
        "nnet-randomize-frames finishes. The validation-set training data is\n"
        "directly input to this program.\n"
        "\n"
        "Usage:  nnet-train [options] <model-in> <validation-features-rspecifier> "
        "<validation-pdfs-rspecifier> <training-examples-in> <model-out>\n"
        "\n"
        "e.g.:\n"
        "nnet-randomize-frames [args] | nnet-train 1.nnet \"$valid_feats\" \\\n"
        "  \"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:-|\" \\\n"
        "   ark:- 2.nnet\n";
    
    bool binary_write = true;
    std::string valid_spk_vecs_rspecifier;
    NnetAdaptiveTrainerConfig train_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("valid-spk-vecs", &valid_spk_vecs_rspecifier,
                "Rspecifier for speaker vectors for validation set");
    
    train_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        valid_feats_rspecifier = po.GetArg(2),
        valid_pdf_ali_rspecifier = po.GetArg(3),
        examples_rspecifier = po.GetArg(4),
        nnet_wxfilename = po.GetArg(5);


    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }
    
    NnetValidationSet validation_set; // stores validation utterances.

    { // This block adds files to "validation_set".
      SequentialBaseFloatMatrixReader feat_reader(valid_feats_rspecifier);
      RandomAccessInt32VectorReader pdf_ali_reader(valid_pdf_ali_rspecifier);
      RandomAccessBaseFloatVectorReader vecs_reader(valid_spk_vecs_rspecifier);
      // may be empty.
      
      int32 num_done = 0, num_err = 0;
    
      while (!feat_reader.Done()) {
        std::string key = feat_reader.Key();
        const Matrix<BaseFloat> &feats = feat_reader.Value();
        if (!pdf_ali_reader.HasKey(key)) {
          KALDI_WARN << "No pdf alignment for key " << key;
          num_err++;
        } else {
          const std::vector<int32> &pdf_ali = pdf_ali_reader.Value(key);
          Vector<BaseFloat> spk_info;

          if (valid_spk_vecs_rspecifier != "") {
            if (!vecs_reader.HasKey(key)) {
              KALDI_WARN << "No speaker vector for key " << key;
              num_err++;
              continue;
            } else {
              spk_info = vecs_reader.Value(key);
            }
          }
          BaseFloat utterance_weight = 1.0; // We don't support weighting
          // for now, at this level.
          validation_set.AddUtterance(feats, spk_info, pdf_ali, utterance_weight);
        }
      }
      KALDI_LOG << "Read " << num_done << " utterances from the validation set; "
                << num_err << " had errors.";
      if (num_done == 0)
        KALDI_ERR << "Read no validation set data.";
    }

    NnetAdaptiveTrainer trainer(train_config,
                                validation_set,
                                &(am_nnet.GetNnet()));
    
    SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);

    int64 num_examples = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_examples++)
      trainer.TrainOnExample(example_reader.Value());  // It all happens here!
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.  Wrote model to "
              << nnet_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


