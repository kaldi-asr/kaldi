// nnet3bin/nnet3-acc-lda-stats.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-optimize.h"
#include "transform/lda-estimate.h"


namespace kaldi {
namespace nnet3 {

class NnetLdaStatsAccumulator {
 public:
  NnetLdaStatsAccumulator(BaseFloat rand_prune,
                          const Nnet &nnet):
      rand_prune_(rand_prune), nnet_(nnet), compiler_(nnet) { }

  void AccStats(const NnetExample &eg) {
    ComputationRequest request;
    bool need_backprop = false, store_stats = false;
    GetComputationRequest(nnet_, eg, need_backprop, store_stats, &request);
    const NnetComputation &computation = *(compiler_.Compile(request));
    NnetComputeOptions options;
    if (GetVerboseLevel() >= 3)
      options.debug = true;
    NnetComputer computer(options, computation, nnet_, NULL);

    computer.AcceptInputs(nnet_, eg.io);
    computer.Run();
    const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
    AccStatsFromOutput(eg, nnet_output);
  }

  void WriteStats(const std::string &stats_wxfilename, bool binary) {
    if (lda_stats_.TotCount() == 0) {
      KALDI_ERR << "Accumulated no stats.";
    } else {
      WriteKaldiObject(lda_stats_, stats_wxfilename, binary);
      KALDI_LOG << "Accumulated stats, soft frame count = "
                << lda_stats_.TotCount() << ".  Wrote to "
                << stats_wxfilename;
    }
  }
 private:
  void AccStatsFromOutput(const NnetExample &eg,
                          const CuMatrixBase<BaseFloat> &nnet_output) {
    BaseFloat rand_prune = rand_prune_;
    const NnetIo *output_supervision = NULL;
    for (size_t i = 0; i < eg.io.size(); i++)
      if (eg.io[i].name == "output")
        output_supervision = &(eg.io[i]);
    KALDI_ASSERT(output_supervision != NULL && "no output in eg named 'output'");
    int32 num_rows = output_supervision->features.NumRows(),
        num_pdfs = output_supervision->features.NumCols();
    KALDI_ASSERT(num_rows == nnet_output.NumRows());
    if (lda_stats_.Dim() == 0)
      lda_stats_.Init(num_pdfs, nnet_output.NumCols());
    if (output_supervision->features.Type() == kSparseMatrix) {
      const SparseMatrix<BaseFloat> &smat =
          output_supervision->features.GetSparseMatrix();
      for (int32 r = 0; r < num_rows; r++) {
        // the following, transferring row by row to CPU, would be wasteful
        // if we actually were using a GPU, but we don't anticipate doing this
        // in this program.
        CuSubVector<BaseFloat> cu_row(nnet_output, r);
        // "row" is actually just a redudant copy, since we're likely on CPU,
        // but we're about to do an outer product, so this doesn't dominate.
        Vector<BaseFloat> row(cu_row);

        const SparseVector<BaseFloat> &post(smat.Row(r));
        const std::pair<MatrixIndexT, BaseFloat> *post_data = post.Data(),
            *post_end = post_data + post.NumElements();
        for (; post_data != post_end; ++post_data) {
          MatrixIndexT pdf = post_data->first;
          BaseFloat weight = post_data->second;
          BaseFloat pruned_weight = RandPrune(weight, rand_prune);
          if (pruned_weight != 0.0)
            lda_stats_.Accumulate(row, pdf, pruned_weight);
        }
      }
    } else {
      Matrix<BaseFloat> output_mat;
      output_supervision->features.GetMatrix(&output_mat);
      for (int32 r = 0; r < num_rows; r++) {
        // the following, transferring row by row to CPU, would be wasteful
        // if we actually were using a GPU, but we don't anticipate doing this
        // in this program.
        CuSubVector<BaseFloat> cu_row(nnet_output, r);
        // "row" is actually just a redudant copy, since we're likely on CPU,
        // but we're about to do an outer product, so this doesn't dominate.
        Vector<BaseFloat> row(cu_row);

        SubVector<BaseFloat> post(output_mat, r);
        int32 num_pdfs = post.Dim();
        for (int32 pdf = 0; pdf < num_pdfs; pdf++) {
          BaseFloat weight = post(pdf);
          BaseFloat pruned_weight = RandPrune(weight, rand_prune);
          if (pruned_weight != 0.0)
            lda_stats_.Accumulate(row, pdf, pruned_weight);
        }
      }
    }
  }

  BaseFloat rand_prune_;
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;
  LdaEstimate lda_stats_;

};

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Accumulate statistics in the same format as acc-lda (i.e. stats for\n"
        "estimation of LDA and similar types of transform), starting from nnet\n"
        "training examples.  This program puts the features through the network,\n"
        "and the network output will be the features; the supervision in the\n"
        "training examples is used for the class labels.  Used in obtaining\n"
        "feature transforms that help nnet training work better.\n"
        "\n"
        "Usage:  nnet3-acc-lda-stats [options] <raw-nnet-in> <training-examples-in> <lda-stats-out>\n"
        "e.g.:\n"
        "nnet3-acc-lda-stats 0.raw ark:1.egs 1.acc\n"
        "See also: nnet-get-feature-transform\n";

    bool binary_write = true;
    BaseFloat rand_prune = 0.0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("rand-prune", &rand_prune,
                "Randomized pruning threshold for posteriors");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        lda_accs_wxfilename = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetLdaStatsAccumulator accumulator(rand_prune, nnet);

    int64 num_egs = 0;

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next(), num_egs++)
      accumulator.AccStats(example_reader.Value());

    KALDI_LOG << "Processed " << num_egs << " examples.";
    // the next command will die if we accumulated no stats.
    accumulator.WriteStats(lda_accs_wxfilename, binary_write);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
