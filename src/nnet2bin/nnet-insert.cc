// nnet2bin/nnet-insert.cc

// Copyright 2012-2014  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-functions.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Insert components into a neural network-based acoustic model.\n"
        "This is mostly intended for adding new hidden layers to neural networks.\n"
        "You can either specify the option --insert-at=n (specifying the index of\n"
        "the component after which you want your neural network inserted), or by\n"
        "default this program will insert it just before the component before the\n"
        "softmax component.  CAUTION: It will also randomize the parameters of the\n"
        "component before the softmax (typically AffineComponent), with stddev equal\n"
        "to the --stddev-factor option (default 0.1), times the inverse square root\n"
        "of the number of inputs to that component.\n"
        "Set --randomize-next-component=false to turn this off.\n"
        "\n"
        "Usage:  nnet-insert [options] <nnet-in> <raw-nnet-to-insert-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-insert 1.nnet \"nnet-init hidden_layer.config -|\" 2.nnet\n";

    bool binary_write = true;
    bool randomize_next_component = true;
    int32 insert_at = -1;
    BaseFloat stddev_factor = 0.1;
    int32 srand_seed = 0;
    
    ParseOptions po(usage);
    
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("randomize-next-component", &randomize_next_component,
                "If true, randomize the parameters of the next component after "
                "what we insert (which must be updatable).");
    po.Register("insert-at", &insert_at, "Inserts new components before the "
                "specified component (note: indexes are zero-based).  If <0, "
                "inserts before the component before the softmax.");
    po.Register("stddev-factor", &stddev_factor, "Factor on the standard "
                "deviation when randomizing next component (only relevant if "
                "--randomize-next-component=true");
    po.Register("srand", &srand_seed, "Seed for random number generator");
    
    po.Read(argc, argv);
    srand(srand_seed);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        raw_nnet_rxfilename = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    Nnet src_nnet; // the one we'll insert.
    ReadKaldiObject(raw_nnet_rxfilename, &src_nnet);

    if (insert_at == -1) {
      if ((insert_at = IndexOfSoftmaxLayer(am_nnet.GetNnet())) == -1)
        KALDI_ERR << "We don't know where to insert the new components: "
            "the neural net doesn't have exactly one softmax component, "
            "and you didn't use the --insert-at option.";
      insert_at--; // we want to insert before the linearity before
      // the softmax layer.
    }
    
    // This function is declared in nnet-functions.h
    InsertComponents(src_nnet,
                     insert_at,
                     &(am_nnet.GetNnet()));
    KALDI_LOG << "Inserted " << src_nnet.NumComponents() << " components at "
              << "position " << insert_at;

    if (randomize_next_component) {
      int32 c = insert_at + src_nnet.NumComponents();
      kaldi::nnet2::Component *component = &(am_nnet.GetNnet().GetComponent(c));
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(component);
      if (!uc)
        KALDI_ERR << "You have --randomize-next-component=true, but the "
                  << "component to randomize is not updatable: "
                  << component->Info();
      bool treat_as_gradient = false;
      uc->SetZero(treat_as_gradient);
      BaseFloat stddev = stddev_factor /
          std::sqrt(static_cast<BaseFloat>(uc->InputDim()));
      uc->PerturbParams(stddev);
      KALDI_LOG << "Randomized component index " << c << " with stddev "
                << stddev;
    }

   
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Write neural-net acoustic model to " <<  nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
