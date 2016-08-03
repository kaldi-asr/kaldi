// chainbin/nnet3-chain-normalize-egs.cc

// Copyright      2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet3/nnet-chain-example.h"
#include "chain/chain-supervision.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Add weights from 'normalization' FST to nnet3+chain examples.\n"
        "Should be done if and only if the <normalization-fst> argument of\n"
        "nnet3-chain-get-egs was not supplied when the original egs were\n"
        "created.\n"
        "\n"
        "Usage:  nnet3-chain-normalize-egs [options] <normalization-fst> <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "e.g.\n"
        "nnet3-chain-normalize-egs dir/normalization.fst ark:train_in.cegs ark:train_out.cegs\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string normalization_fst_rxfilename = po.GetArg(1),
                         examples_rspecifier = po.GetArg(2),
                         examples_wspecifier = po.GetArg(3);

    fst::StdVectorFst normalization_fst;
    ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);

    int64 num_written = 0, num_err = 0;;
    for (; !example_reader.Done(); example_reader.Next()) {
      std::string key = example_reader.Key();
      NnetChainExample eg = example_reader.Value();

      if (eg.outputs.size() != 1)
        KALDI_ERR << "Expected example to have exactly one output.";
      if (!AddWeightToSupervisionFst(normalization_fst,
                                     &(eg.outputs[0].supervision))) {
        KALDI_WARN << "For example " << key
                   << ", FST was empty after composing with normalization FST. "
                   << "This should be extremely rare (a few per corpus, at most)";
        num_err++;
      } else {
        example_writer.Write(key, eg);
        num_written++;
      }
    }

    KALDI_LOG << "Added normalization to " << num_written
              << " egs; had errors on " << num_err;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


