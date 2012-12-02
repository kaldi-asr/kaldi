// nnet-cpubin/nnet-randomize-frames.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Randomize frames of data for neural network training.\n"
        "Will typically output data to a pipe that is fed into the trainer.\n"
        "Note: this program keeps all the utterances it is given in memory,\n"
        "in compressed form (about one byte per element), so it can use a\n"
        "fair amount of memory (about 15M per hour of speech at 40 dims, 100\n"
        "frames/sec).\n"
        "\n"
        "Usage:  nnet-randomize-frames [options] <features-rspecifier> "
        "<pdfs-rspecifier> <training-examples-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet-randomize-frames --left-context=8 --right-context=8 \\\n"
        "  --num-samples=100000 exp/nnet/1.nnet \"$feats\" \\\n"
        "  \"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:-|\" \\\n"
        "   ark:- \n"
        "Note: you must set either --num-samples or --num-epochs, and the\n"
        "--frequency-power is also a potentially useful option (try 0.5).\n"
        "Note: the --left-context and --right-context would be derived from\n"
        "the output of nnet-info.";
        
    bool binary_write = true;
    int32 left_context = 0, right_context = 0;
    int32 srand_seed = 0;
    std::string spk_vecs_rspecifier;
    NnetDataRandomizerConfig randomize_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("spk-vecs", &spk_vecs_rspecifier, "Rspecifier for speaker vectors");
    po.Register("srand", &srand_seed, "Seed for random number generator");
    po.Register("left-context", &left_context, "Number of frames of left context "
                "the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right context "
                "the neural net requires.");
    
    randomize_config.Register(&po);
    
    po.Read(argc, argv);
    srand(srand_seed);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        pdf_alignments_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    NnetDataRandomizer randomizer(left_context,
                                  right_context,
                                  randomize_config);
    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessInt32VectorReader pdf_ali_reader(pdf_alignments_rspecifier);
    RandomAccessBaseFloatVectorReader vecs_reader(spk_vecs_rspecifier); // may be empty.
    NnetTrainingExampleWriter example_writer(examples_wspecifier);

    
    int32 num_done = 0, num_err = 0;
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!pdf_ali_reader.HasKey(key)) {
        KALDI_WARN << "No pdf alignment for key " << key;
        num_err++;
      } else {
        const std::vector<int32> &pdf_ali = pdf_ali_reader.Value(key);
        Vector<BaseFloat> spk_info;

        if (spk_vecs_rspecifier != "") {
          if (!vecs_reader.HasKey(key)) {
            KALDI_WARN << "No speaker vector for key " << key;
            num_err++;
            continue;
          } else {
            spk_info = vecs_reader.Value(key);
          }
        }
        randomizer.AddTrainingFile(feats, spk_info, pdf_ali);
        num_done++;
      }
    }

    kaldi::int64 num_written = 0;
    for (; !randomizer.Done(); randomizer.Next(), num_written++) {
      std::ostringstream os;
      os << num_written;
      std::string key = os.str(); // key in the archive is
      // the number of the example.
      example_writer.Write(key, randomizer.Value());
    }
    KALDI_LOG << "Finished randomizing features, wrote "
              << num_written << " examples, successfully processed " << num_done
              << " feature files, " << num_err << " with errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


