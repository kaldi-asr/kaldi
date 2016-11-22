// nnet3bin/nnet3-copy-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

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
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples (single frames or fixed-size groups of frames) for neural\n"
        "network training, possibly changing the binary mode.  Supports multiple wspecifiers, in\n"
        "which case it will write the examples round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet3-copy-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "e.g.\n"
        "nnet3-copy-egs ark:train.egs ark,t:text.egs\n"
        "or:\n"
        "nnet3-copy-egs ark:train.egs ark:1.egs\n";

    ParseOptions po(usage);

    bool add_silence_output = true;
    bool add_speech_output = true;
    int32 srand_seed = 0;

    std::string keep_proportion_positive_rxfilename;
    std::string keep_proportion_negative_rxfilename;

    po.Register("add-silence-output", &add_silence_output,
                "Add silence output");
    po.Register("add-speech-output", &add_speech_output,
                "Add speech output");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --keep-proportion-vec is specified");
    po.Register("keep-proportion-positive-vec", &keep_proportion_positive_rxfilename, 
                "If a dimension of this is <1.0, this program will "
                "randomly set deriv weight 0 for this proportion of the input samples of the "
                "corresponding positive examples");
    po.Register("keep-proportion-negative-vec", &keep_proportion_negative_rxfilename, 
                "If a dimension of this is <1.0, this program will "
                "randomly set deriv weight 0 for this proportion of the input samples of the "
                "corresponding negative examples");

    Vector<BaseFloat> p_positive_vec(3);
    p_positive_vec.Set(1);
    if (!keep_proportion_positive_rxfilename.empty())
      ReadKaldiObject(keep_proportion_positive_rxfilename, &p_positive_vec);
    
    Vector<BaseFloat> p_negative_vec(3);
    p_negative_vec.Set(1);
    if (!keep_proportion_negative_rxfilename.empty())
      ReadKaldiObject(keep_proportion_negative_rxfilename, &p_negative_vec);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);
    std::string examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      std::string key = example_reader.Key();
      NnetExample eg = example_reader.Value();

      KALDI_ASSERT(eg.io.size() == 2);
      NnetIo &io = eg.io[1];

      KALDI_ASSERT(io.name == "output");

      NnetIo silence_output(io);
      silence_output.name = "output-silence";

      NnetIo speech_output(io);
      speech_output.name = "output-speech";

      NnetIo overlap_speech_output(io);
      overlap_speech_output.name = "output-overlap_speech";

      io.features.Uncompress();

      KALDI_ASSERT(io.features.Type() == kFullMatrix);
      const Matrix<BaseFloat> &feats = io.features.GetFullMatrix();

      typedef std::vector<std::pair<int32, BaseFloat> > SparseVec;
      std::vector<SparseVec> silence_post(feats.NumRows(), SparseVec());
      std::vector<SparseVec> speech_post(feats.NumRows(), SparseVec());
      std::vector<SparseVec> overlap_speech_post(feats.NumRows(), SparseVec());

      Vector<BaseFloat> silence_deriv_weights(feats.NumRows());
      Vector<BaseFloat> speech_deriv_weights(feats.NumRows());
      Vector<BaseFloat> overlap_speech_deriv_weights(feats.NumRows());

      for (int32 i = 0; i < feats.NumRows(); i++) {
        if (feats(i,0) < 0.5) {
          silence_deriv_weights(i) = WithProb(p_negative_vec(0)) ? 1.0 : 0.0;
          silence_post[i].push_back(std::make_pair(0, 1));
        } else {
          silence_deriv_weights(i) = WithProb(p_positive_vec(0)) ? 1.0 : 0.0;
          silence_post[i].push_back(std::make_pair(1, 1));
        }
        
        if (feats(i,1) < 0.5) {
          speech_deriv_weights(i) = WithProb(p_negative_vec(1)) ? 1.0 : 0.0;
          speech_post[i].push_back(std::make_pair(0, 1));
        } else {
          speech_deriv_weights(i) = WithProb(p_positive_vec(1)) ? 1.0 : 0.0;
          speech_post[i].push_back(std::make_pair(1, 1));
        }

        if (feats(i,2) < 0.5) {
          overlap_speech_deriv_weights(i) = WithProb(p_negative_vec(2)) ? 1.0 : 0.0;
          overlap_speech_post[i].push_back(std::make_pair(0, 1));
        } else {
          overlap_speech_deriv_weights(i) = WithProb(p_positive_vec(2)) ? 1.0 : 0.0;
          overlap_speech_post[i].push_back(std::make_pair(1, 1));
        }
      }

      SparseMatrix<BaseFloat> silence_feats(2, silence_post);  
      SparseMatrix<BaseFloat> speech_feats(2, speech_post);  
      SparseMatrix<BaseFloat> overlap_speech_feats(2, overlap_speech_post);  

      silence_output.features = silence_feats;
      speech_output.features = speech_feats;
      overlap_speech_output.features = overlap_speech_feats;

      io = overlap_speech_output;
      io.deriv_weights.MulElements(overlap_speech_deriv_weights);
      
      if (add_silence_output) {
        silence_output.deriv_weights.MulElements(silence_deriv_weights);
        eg.io.push_back(silence_output);
      }

      if (add_speech_output) {
        speech_output.deriv_weights.MulElements(speech_deriv_weights);
        eg.io.push_back(speech_output);
      }

      example_writer.Write(key, eg);
      num_written++;
    }

    KALDI_LOG << "Read " << num_read << " neural-network training examples, wrote "
              << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



