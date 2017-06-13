// nnet2bin/nnet-limit-degradation.cc

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
        "Given an old and a new model and some training examples (possibly held-out),\n"
        "produce an output model that will generally be the same as the new model,\n"
        "but in cases where changes a particular component led to an objective function\n"
        "change per example that's less than -t for a particular threshold t>=0\n"
        "(default --threshold=0.0001), will regress towards the old model.  Does this\n"
        "by repeatedly downscaling by a scale (default --scale=0.75) until we satisfy\n"
        "the criterion.\n"
        "Usage:  nnet-limit-degradation [options] <old-model-in> <new-model-in> <training-examples-in> <model-out>\n"
        "e.g.: nnet-limit-degradation 1.nnet 2.nnet ark:valid.egs 2new.nnet\n";
    
    ParseOptions po(usage);

    bool binary_write = false;
    BaseFloat scale = 0.75; // Each time we find we went too far, we multiply the parameter-change by this scale.
    BaseFloat threshold = 0.0001; // Maximum degradation.
    po.Register("binary", &binary_write, "If true, write model in binary format.");
    po.Register("scale", &scale, "Scale factor we multiply by each time we detect "
                "excess degradation");
    po.Register("threshold", &threshold, "Threshold of amount of loglike-per-frame "
                "degradation per layer that we will allow");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet1_rxfilename = po.GetArg(1),
        nnet2_rxfilename = po.GetArg(2),
        examples_rspecifier = po.GetArg(3),
        nnet_wxfilename = po.GetArg(4);
    
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
        
    std::vector<NnetExample> examples;
    SequentialNnetExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next())
      examples.push_back(example_reader.Value());

    int32 batch_size = 1024, max_iter = 10;
    int32 nu = am_nnet1.GetNnet().NumUpdatableComponents();
    Vector<BaseFloat> cur_scales(nu);
    cur_scales.Set(1.0);
    
    for (int32 iter = 0; iter < max_iter; iter++) {
      Nnet avg_nnet(am_nnet2.GetNnet());
      avg_nnet.Scale(0.5);
      avg_nnet.AddNnet(0.5, am_nnet1.GetNnet());
      
      Nnet nnet_gradient(am_nnet2.GetNnet());
      const bool treat_as_gradient = true;
      nnet_gradient.SetZero(treat_as_gradient);

      double objf_per_frame = ComputeNnetGradient(avg_nnet, examples,
                                                  batch_size, &nnet_gradient);

      int64 num_examples = examples.size();
      KALDI_LOG << "Saw " << num_examples << " examples, average "
                << "probability is " << objf_per_frame << " over "
                << examples.size() << " examples.";
      
      Vector<BaseFloat> old_dotprod(nu), new_dotprod(nu);
      nnet_gradient.ComponentDotProducts(am_nnet1.GetNnet(), &old_dotprod);
      nnet_gradient.ComponentDotProducts(am_nnet2.GetNnet(), &new_dotprod);
      old_dotprod.Scale(1.0 / num_examples);
      new_dotprod.Scale(1.0 / num_examples);
      Vector<BaseFloat> diff(new_dotprod);
      diff.AddVec(-1.0, old_dotprod);
      KALDI_LOG << "On iter " << iter << ", progress per component is " << diff;

      if (diff.Min() > -threshold) {
        KALDI_LOG << "Succeeded on iter " << iter;
        break;
      } else {
        Vector<BaseFloat> scales(nu);
        for (int32 i = 0; i < nu; i++) {
          if (diff(i) < -threshold) scales(i) = scale;
          else scales(i) = 1.0;
        }
        // Regress new model towards old model, using scales "scales".
        Vector<BaseFloat> old_scales(nu);
        for (int32 i = 0; i < nu; i++)
          old_scales(i) = 1.0 - scales(i);
        am_nnet2.GetNnet().ScaleComponents(scales);
        am_nnet2.GetNnet().AddNnet(old_scales, am_nnet1.GetNnet());
        cur_scales.MulElements(scales);
      }
    }
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet2.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Wrote model to " << PrintableWxfilename(nnet_wxfilename)
              << ", after scaling difference by " << cur_scales;
    return (examples.empty() ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


