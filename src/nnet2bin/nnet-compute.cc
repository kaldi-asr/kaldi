// nnet2bin/nnet-compute.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/train-nnet.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Does the neural net computation for each file of input features, and\n"
        "outputs as a matrix the result.  Used mostly for debugging.\n"
        "Note: if you want it to apply a log (e.g. for log-likelihoods), use\n"
        "--apply-log=true.  Unlike nnet-am-compute, this version reads a 'raw'\n"
        "neural net\n"
        "\n"
        "Usage:  nnet-compute [options] <raw-nnet-in> <feature-rspecifier> "
        "<feature-or-loglikes-wspecifier>\n";
    
    bool apply_log = false;
    bool pad_input = true;
    ParseOptions po(usage);
    po.Register("apply-log", &apply_log, "Apply a log to the result of the computation "
                "before outputting.");
    po.Register("pad-input", &pad_input, "If true, duplicate the first and last frames "
                "of input features as required for temporal context, to prevent #frames "
                "of output being less than those of input.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string raw_nnet_rxfilename = po.GetArg(1),
        features_rspecifier = po.GetArg(2),
        features_or_loglikes_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    
    int64 num_done = 0, num_frames = 0;
    SequentialBaseFloatCuMatrixReader feature_reader(features_rspecifier);
    BaseFloatCuMatrixWriter writer(features_or_loglikes_wspecifier);
    
    for (; !feature_reader.Done();  feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const CuMatrix<BaseFloat> &feats = feature_reader.Value();

      int32 output_frames = feats.NumRows(), output_dim = nnet.OutputDim();
      if (!pad_input)
        output_frames -= nnet.LeftContext() + nnet.RightContext();
      if (output_frames <= 0) {
        KALDI_WARN << "Skipping utterance " << utt << " because output "
                   << "would be empty.";
        continue;
      }
      CuMatrix<BaseFloat> output(output_frames, output_dim);
      NnetComputation(nnet, feats, pad_input, &output);

      if (apply_log) {
        output.ApplyFloor(1.0e-20);
        output.ApplyLog();
      }
      writer.Write(utt, output);
      num_frames += feats.NumRows();
      num_done++;
    }
    
    KALDI_LOG << "Processed " << num_done << " feature files, "
              << num_frames << " frames of input were processed.";
    
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


