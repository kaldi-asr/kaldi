// nnet2bin/nnet-show-progress.cc

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
#include "nnet2/nnet-randomize.h"
#include "nnet2/train-nnet.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Given an old and a new model and some training examples (possibly held-out),\n"
        "show the average objective function given the mean of the two models,\n"
        "and the breakdown by component of why this happened (computed from\n"
        "derivative information)\n"
        "\n"
        "Usage:  nnet-show-progress [options] <old-model-in> <new-model-in> <training-examples-in>\n"
        "e.g.: nnet-show-progress 1.nnet 2.nnet ark:valid.egs\n";
    
    ParseOptions po(usage);

    int32 num_segments = 1;
    int32 batch_size = 1024;
    
    po.Register("num-segments", &num_segments,
                "Number of line segments used for computing derivatives");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet1_rxfilename = po.GetArg(1),
        nnet2_rxfilename = po.GetArg(2),
        examples_rspecifier = po.GetArg(3);

    TransitionModel trans_model;
    AmNnet am_nnet1, am_nnet2;
    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet1.Read(ki.Stream(), binary_read);
    }
    {
      bool binary_read;
      Input ki(nnet2_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet2.Read(ki.Stream(), binary_read);
    }    

    
    Nnet nnet_gradient(am_nnet2.GetNnet());
    const bool treat_as_gradient = true;
    nnet_gradient.SetZero(treat_as_gradient);
    
    std::vector<NnetTrainingExample> examples;
    SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next())
      examples.push_back(example_reader.Value());

    int32 num_examples = examples.size();
    
    int32 nu = am_nnet1.GetNnet().NumUpdatableComponents();
    Vector<BaseFloat> diff(nu);
    
    for (int32 s = 0; s < num_segments; s++) {
      // start and end segments of the line between 0 and 1
      BaseFloat start = (s + 0.0) / num_segments,
          end = (s + 1.0) / num_segments, middle = 0.5 * (start + end);
      Nnet interp_nnet(am_nnet2.GetNnet());
      interp_nnet.Scale(middle);
      interp_nnet.AddNnet(1.0 - middle, am_nnet1.GetNnet());
      
      Nnet nnet_gradient(am_nnet2.GetNnet());
      const bool treat_as_gradient = true;
      nnet_gradient.SetZero(treat_as_gradient);
      
      double objf_per_frame = ComputeNnetGradient(interp_nnet, examples,
                                                  batch_size, &nnet_gradient);
      KALDI_LOG << "At position " << middle << ", objf per frame is " << objf_per_frame;

      Vector<BaseFloat> old_dotprod(nu), new_dotprod(nu);
      nnet_gradient.ComponentDotProducts(am_nnet1.GetNnet(), &old_dotprod);
      nnet_gradient.ComponentDotProducts(am_nnet2.GetNnet(), &new_dotprod);
      old_dotprod.Scale(1.0 / num_examples);
      new_dotprod.Scale(1.0 / num_examples);
      diff.AddVec(1.0/ num_segments, new_dotprod);
      diff.AddVec(-1.0 / num_segments, old_dotprod);
      KALDI_LOG << "By segment " << s << ", diff is " << diff;
    }
    KALDI_LOG << "Total diff per component is " << diff;
    
    { // Get info about magnitude of parameter change.
      Nnet diff_nnet(am_nnet1.GetNnet());
      diff_nnet.AddNnet(-1.0, am_nnet2.GetNnet());
      int32 nu = diff_nnet.NumUpdatableComponents();
      Vector<BaseFloat> dot_prod(nu);
      diff_nnet.ComponentDotProducts(diff_nnet, &dot_prod);
      dot_prod.ApplyPow(0.5); // take sqrt to get l2 norm of diff
      KALDI_LOG << "Parameter differences per layer are "
                << dot_prod;
    }

    
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


