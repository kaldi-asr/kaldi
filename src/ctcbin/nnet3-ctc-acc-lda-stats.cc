// nnet3bin/nnet3-ctc-acc-lda-stats.cc

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
#include "nnet3/nnet-cctc-example.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "transform/lda-estimate.h"


namespace kaldi {
namespace nnet3 {

class NnetCctcLdaStatsAccumulator {
 public:
  NnetCctcLdaStatsAccumulator(BaseFloat rand_prune,
                              const ctc::CctcTransitionModel &trans_model,
                              const Nnet &nnet):
      rand_prune_(rand_prune), trans_model_(trans_model), nnet_(nnet),
      compiler_(nnet) { }


  void AccStats(const NnetCctcExample &eg) {
    ComputationRequest request;
    bool need_backprop = false, store_stats = false;

    GetCctcComputationRequest(nnet_, eg, need_backprop, store_stats, &request);

    const NnetComputation &computation = *(compiler_.Compile(request));

    NnetComputeOptions options;
    if (GetVerboseLevel() >= 3)
      options.debug = true;
    NnetComputer computer(options, computation, nnet_, NULL);

    computer.AcceptInputs(nnet_, eg.inputs);
    computer.Forward();
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
  void AccStatsFromOutput(const NnetCctcExample &eg,
                          const CuMatrixBase<BaseFloat> &nnet_output) {
    BaseFloat rand_prune = rand_prune_;

    if (eg.outputs.size() != 1 || eg.outputs[0].name != "output" ||
        eg.outputs[0].supervision.size() != 1)
      KALDI_ERR << "Expecting the example to have one output named 'output',"
                << "with one supervision object.";

    const ctc::CctcSupervision &supervision = eg.outputs[0].supervision[0];
    supervision.Check(trans_model_);
    int32 num_frames = supervision.num_frames;
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

    // pdf_post is the posteriors modified to exclude blanks and
    // to be mapped to the pdf-ids from the decision tree.
    Posterior pdf_post;
    MapPosterior(post, &pdf_post);

    if (lda_stats_.Dim() == 0)
      lda_stats_.Init(trans_model_.NumNonBlankIndexes(),
                      nnet_output.NumCols());

    for (int32 t = 0; t < num_frames; t++) {
      // the following, transferring row by row to CPU, would be wasteful
      // if we actually were using a GPU, but we don't anticipate doing this
      // in this program.
      CuSubVector<BaseFloat> cu_row(nnet_output, t);
      // "row" is actually just a redudant copy, since we're likely on CPU,
      // but we're about to do an outer product, so this doesn't dominate.
      Vector<BaseFloat> row(cu_row);

      std::vector<std::pair<int32,BaseFloat> >::const_iterator
          iter = pdf_post[t].begin(), end = pdf_post[t].end();

      for (; iter != end; ++iter) {
        int32 pdf = iter->first;
        BaseFloat weight = iter->second;
        BaseFloat pruned_weight = RandPrune(weight, rand_prune);
        if (pruned_weight != 0.0)
          lda_stats_.Accumulate(row, pdf, pruned_weight);
      }
    }
  }
  void MapPosterior(const Posterior &post,
                    Posterior *pdf_post) {
    size_t num_frames = post.size();
    pdf_post->clear();
    pdf_post->resize(num_frames);
    for (size_t i = 0; i < num_frames; i++) {
      const std::vector<std::pair<int32, BaseFloat> > &src = post[i];
      std::vector<std::pair<int32, BaseFloat> > &dest = (*pdf_post)[i];
      dest.reserve(src.size());
      std::vector<std::pair<int32, BaseFloat> >::const_iterator
          src_iter = src.begin(), src_end = src.end();
      BaseFloat weight_sum = 0.0;
      for (; src_iter != src_end; ++src_iter) {
        int32 graph_label = src_iter->first,
            output_index = trans_model_.GraphLabelToOutputIndex(graph_label);
        if (output_index < trans_model_.NumNonBlankIndexes()) {
          BaseFloat weight = src_iter->second;
          dest.push_back(std::pair<int32, BaseFloat>(output_index, weight));
          weight_sum += weight;
        }
      }
      if (weight_sum == 0.0)
        KALDI_ERR << "No non-blank labels encountered on this frame.";
      std::vector<std::pair<int32, BaseFloat> >::iterator
          dest_iter = dest.begin(), dest_end = dest.end();
      // Renormalize to sum to one over the non-blank labels.
      BaseFloat scale = 1.0 / weight_sum;
      for (; dest_iter != dest_end; ++dest_iter)
        dest_iter->second *= scale;
    }
  }


  BaseFloat rand_prune_;
  const ctc::CctcTransitionModel &trans_model_;
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
        "estimation of LDA and similar types of transform), starting from nnet+ctc\n"
        "training examples.  This program puts the features through the network,\n"
        "and the network output will be the features; the supervision in the\n"
        "training examples is used for the class labels.  Used in obtaining\n"
        "feature transforms that help nnet training work better.\n"
        "Note: the time boundaries it gets from the CTC supervision are a little\n"
        "fuzzy (hence not ideal), but it should not matter much in this situation.\n"
        "\n"
        "Usage:  nnet3-ctc-acc-lda-stats [options] <ctc-transition-model-in> <raw-nnet-in> <training-examples-in> <lda-stats-out>\n"
        "e.g.:\n"
        "nnet3-ctc-acc-lda-stats 0.trans 0.raw ark:1.cegs 1.acc\n"
        "See also: nnet-get-feature-transform\n";

    bool binary_write = true;
    BaseFloat rand_prune = 0.0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("rand-prune", &rand_prune,
                "Randomized pruning threshold for posteriors");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string cctc_trans_model_rxfilename = po.GetArg(1),
        nnet_rxfilename = po.GetArg(2),
        examples_rspecifier = po.GetArg(3),
        lda_accs_wxfilename = po.GetArg(4);


    ctc::CctcTransitionModel trans_model;
    ReadKaldiObject(cctc_trans_model_rxfilename, &trans_model);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetCctcLdaStatsAccumulator accumulator(rand_prune, trans_model, nnet);

    int64 num_egs = 0;

    SequentialNnetCctcExampleReader example_reader(examples_rspecifier);
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


