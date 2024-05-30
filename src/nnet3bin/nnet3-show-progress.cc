// nnet3bin/nnet3-show-progress.cc

// Copyright 2015 Johns Hopkins University (author:  Daniel Povey)
//           2015 Xingyu Na

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
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-diagnostics.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Given an old and a new 'raw' nnet3 network and some training examples\n"
        "(possibly held-out), show the average objective function given the\n"
        "mean of the two networks, and the breakdown by component of why this\n"
        "happened (computed from derivative information). Also shows parameter\n"
        "differences per layer. If training examples not provided, only shows\n"
        "parameter differences per layer.\n"
        "\n"
        "Usage:  nnet3-show-progress [options] <old-net-in> <new-net-in>"
        " [<training-examples-in>]\n"
        "e.g.: nnet3-show-progress 1.nnet 2.nnet ark:valid.egs\n";

    ParseOptions po(usage);

    int32 num_segments = 1;
    std::string use_gpu = "no";
    NnetComputeProbOptions compute_prob_opts;
    compute_prob_opts.compute_deriv = true;

    po.Register("num-segments", &num_segments,
                "Number of line segments used for computing derivatives");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    compute_prob_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet1_rxfilename = po.GetArg(1),
                nnet2_rxfilename = po.GetArg(2),
                examples_rspecifier = po.GetOptArg(3);

    Nnet nnet1, nnet2;
    ReadKaldiObject(nnet1_rxfilename, &nnet1);
    ReadKaldiObject(nnet2_rxfilename, &nnet2);

    if (NumParameters(nnet1) != NumParameters(nnet2)) {
      KALDI_WARN << "Parameter-dim mismatch, cannot show progress.";
      exit(0);
    }

    if (!examples_rspecifier.empty() && IsSimpleNnet(nnet1)) {
      std::vector<NnetExample> examples;
      SequentialNnetExampleReader example_reader(examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        examples.push_back(example_reader.Value());

      int32 num_examples = examples.size();

      if (num_examples == 0)
        KALDI_ERR << "No examples read.";

      int32 num_updatable = NumUpdatableComponents(nnet1);
      Vector<BaseFloat> diff(num_updatable);

      for (int32 s = 0; s < num_segments; s++) {
        // start and end segments of the line between 0 and 1
        BaseFloat start = (s + 0.0) / num_segments,
            end = (s + 1.0) / num_segments, middle = 0.5 * (start + end);
        Nnet interp_nnet(nnet2);
        ScaleNnet(middle, &interp_nnet);
        AddNnet(nnet1, 1.0 - middle, &interp_nnet);

        NnetComputeProb prob_computer(compute_prob_opts, interp_nnet);
        std::vector<NnetExample>::const_iterator eg_iter = examples.begin(),
                                                 eg_end = examples.end();
        for (; eg_iter != eg_end; ++eg_iter)
          prob_computer.Compute(*eg_iter);
        const SimpleObjectiveInfo *objf_info = prob_computer.GetObjective("output");
        double objf_per_frame = objf_info->tot_objective / objf_info->tot_weight;

        prob_computer.PrintTotalStats();
        const Nnet &nnet_gradient = prob_computer.GetDeriv();
        KALDI_LOG << "At position " << middle
                  << ", objf per frame is " << objf_per_frame;

        Vector<BaseFloat> old_dotprod(num_updatable), new_dotprod(num_updatable);
        ComponentDotProducts(nnet_gradient, nnet1, &old_dotprod);
        ComponentDotProducts(nnet_gradient, nnet2, &new_dotprod);
        old_dotprod.Scale(1.0 / objf_info->tot_weight);
        new_dotprod.Scale(1.0 / objf_info->tot_weight);
        diff.AddVec(1.0/ num_segments, new_dotprod);
        diff.AddVec(-1.0 / num_segments, old_dotprod);
        KALDI_VLOG(1) << "By segment " << s << ", objf change is "
                      << PrintVectorPerUpdatableComponent(nnet1, diff);
      }
      KALDI_LOG << "Total objf change per component is "
                << PrintVectorPerUpdatableComponent(nnet1, diff);
    }

    { // Get info about magnitude of parameter change.
      Nnet diff_nnet(nnet1);
      AddNnet(nnet2, -1.0, &diff_nnet);
      if (GetVerboseLevel() >= 1) {
        KALDI_VLOG(1) << "Printing info for the difference between the neural nets: "
                      << diff_nnet.Info();
      }
      int32 num_updatable = NumUpdatableComponents(diff_nnet);
      Vector<BaseFloat> dot_prod(num_updatable);
      ComponentDotProducts(diff_nnet, diff_nnet, &dot_prod);
      dot_prod.ApplyPow(0.5); // take sqrt to get l2 norm of diff
      KALDI_LOG << "Parameter differences per layer are "
                << PrintVectorPerUpdatableComponent(nnet1, dot_prod);

      Vector<BaseFloat> baseline_prod(num_updatable),
          new_prod(num_updatable);
      ComponentDotProducts(nnet1, nnet1, &baseline_prod);
      ComponentDotProducts(nnet2, nnet2, &new_prod);
      baseline_prod.ApplyPow(0.5);
      new_prod.ApplyPow(0.5);

      KALDI_LOG << "Norms of parameter matrices from <new-nnet-in> are "
                << PrintVectorPerUpdatableComponent(nnet2, new_prod);

      dot_prod.DivElements(baseline_prod);
      KALDI_LOG << "Relative parameter differences per layer are "
                << PrintVectorPerUpdatableComponent(nnet1, dot_prod);
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
