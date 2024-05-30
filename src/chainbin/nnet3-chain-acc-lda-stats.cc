// nnet3bin/nnet3-chain-acc-lda-stats.cc

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
#include "lat/lattice-functions.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "transform/lda-estimate.h"


namespace kaldi {
namespace nnet3 {

class NnetChainLdaStatsAccumulator {
 public:
  NnetChainLdaStatsAccumulator(BaseFloat rand_prune,
                               const Nnet &nnet):
      rand_prune_(rand_prune), nnet_(nnet), compiler_(nnet) { }


  void AccStats(const NnetChainExample &eg) {
    ComputationRequest request;
    bool need_backprop = false, store_stats = false,
        need_xent = false, need_xent_deriv = false;

    GetChainComputationRequest(nnet_, eg, need_backprop, store_stats,
                               need_xent, need_xent_deriv, &request);

    const NnetComputation &computation = *(compiler_.Compile(request));

    NnetComputeOptions options;
    if (GetVerboseLevel() >= 3)
      options.debug = true;
    NnetComputer computer(options, computation, nnet_, NULL);

    computer.AcceptInputs(nnet_, eg.inputs);
    computer.Run();
    const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
    if (eg.outputs[0].supervision.fst.NumStates() > 0) {
      AccStatsFst(eg, nnet_output);
    } else {
      AccStatsAlignment(eg, nnet_output);
    }
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
  void AccStatsFst(const NnetChainExample &eg,
                   const CuMatrixBase<BaseFloat> &nnet_output) {
    BaseFloat rand_prune = rand_prune_;

    if (eg.outputs.size() != 1 || eg.outputs[0].name != "output")
      KALDI_ERR << "Expecting the example to have one output named 'output'.";


    const chain::Supervision &supervision = eg.outputs[0].supervision;
    // handling the one-sequence-per-eg case is easier so we just do that.
    KALDI_ASSERT(supervision.num_sequences == 1 &&
                 "This program expects one sequence per eg.");
    int32 num_frames = supervision.frames_per_sequence,
        num_pdfs = supervision.label_dim;
    KALDI_ASSERT(num_frames == nnet_output.NumRows());

    const fst::StdVectorFst &fst = supervision.fst;

    Lattice lat;
    // convert the FST to a lattice, putting all the weight on
    // the graph weight.  This is to save us having to implement the
    // forward-backward on FSTs.
    ConvertFstToLattice(fst, &lat);
    Posterior post;
    LatticeForwardBackward(lat, &post);
    KALDI_ASSERT(post.size() == static_cast<size_t>(num_frames));

    // Subtract one, to convert the (pdf-id + 1) which appears in the
    // supervision FST, to a pdf-id.
    for (size_t i = 0; i < post.size(); i++)
      for (size_t j = 0; j < post[i].size(); j++)
        post[i][j].first--;

    if (lda_stats_.Dim() == 0)
      lda_stats_.Init(num_pdfs,
                      nnet_output.NumCols());

    for (int32 t = 0; t < num_frames; t++) {
      // the following, transferring row by row to CPU, would be wasteful if we
      // actually were using a GPU, but we don't anticipate using a GPU in this
      // program.
      CuSubVector<BaseFloat> cu_row(nnet_output, t);
      // "row" is actually just a redudant copy, since we're likely on CPU,
      // but we're about to do an outer product, so this doesn't dominate.
      Vector<BaseFloat> row(cu_row);

      std::vector<std::pair<int32,BaseFloat> >::const_iterator
          iter = post[t].begin(), end = post[t].end();

      for (; iter != end; ++iter) {
        int32 pdf = iter->first;
        BaseFloat weight = iter->second;
        BaseFloat pruned_weight = RandPrune(weight, rand_prune);
        if (pruned_weight != 0.0)
          lda_stats_.Accumulate(row, pdf, pruned_weight);
      }
    }
  }


  void AccStatsAlignment(const NnetChainExample &eg,
                          const CuMatrixBase<BaseFloat> &nnet_output) {
    BaseFloat rand_prune = rand_prune_;

    if (eg.outputs.size() != 1 || eg.outputs[0].name != "output")
      KALDI_ERR << "Expecting the example to have one output named 'output'.";

    const chain::Supervision &supervision = eg.outputs[0].supervision;
    // handling the one-sequence-per-eg case is easier so we just do that.
    KALDI_ASSERT(supervision.num_sequences == 1 &&
                 "This program expects one sequence per eg.");

    int32 num_frames = supervision.frames_per_sequence,
        num_pdfs = supervision.label_dim;
    KALDI_ASSERT(num_frames == nnet_output.NumRows());

    if (supervision.alignment_pdfs.size() !=
        static_cast<size_t>(num_frames))
      KALDI_ERR << "Alignment pdfs not present or wrong length.  Using e2e egs?";

    if (lda_stats_.Dim() == 0)
      lda_stats_.Init(num_pdfs,
                      nnet_output.NumCols());

    for (int32 t = 0; t < num_frames; t++) {
      // the following, transferring row by row to CPU, would be wasteful if we
      // actually were using a GPU, but we don't anticipate using a GPU in this
      // program.
      CuSubVector<BaseFloat> cu_row(nnet_output, t);
      // "row" is actually just a redudant copy, since we're likely on CPU,
      // but we're about to do an outer product, so this doesn't dominate.
      Vector<BaseFloat> row(cu_row);

      int32 pdf = supervision.alignment_pdfs[t];
      BaseFloat weight = 1.0;
      BaseFloat pruned_weight = RandPrune(weight, rand_prune);
      if (pruned_weight != 0.0)
        lda_stats_.Accumulate(row, pdf, pruned_weight);
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
        "estimation of LDA and similar types of transform), starting from nnet+chain\n"
        "training examples.  This program puts the features through the network,\n"
        "and the network output will be the features; the supervision in the\n"
        "training examples is used for the class labels.  Used in obtaining\n"
        "feature transforms that help nnet training work better.\n"
        "Note: the time boundaries it gets from the chain supervision will be\n"
        "a little fuzzy (which is not ideal), but it should not matter much in\n"
        "this situation\n"
        "\n"
        "Usage:  nnet3-chain-acc-lda-stats [options] <raw-nnet-in> <training-examples-in> <lda-stats-out>\n"
        "e.g.:\n"
        "nnet3-chain-acc-lda-stats 0.raw ark:1.cegs 1.acc\n"
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

    // Note: this neural net is probably just splicing the features at this
    // point.
    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetChainLdaStatsAccumulator accumulator(rand_prune, nnet);

    int64 num_egs = 0;

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
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
