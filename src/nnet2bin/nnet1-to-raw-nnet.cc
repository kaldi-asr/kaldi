// nnet2bin/nnet1-to-raw-nnet.cc

// Copyright 2013  Johns Hopkins University (author:  Daniel Povey, Hainan Xu)

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
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-various.h"
#include "nnet2/nnet-nnet.h"
#include "nnet2/nnet-component.h"

namespace kaldi {

nnet2::Component *ConvertAffineTransformComponent(
    const nnet1::Component &nnet1_component,
    const bool use_preconditioned_affine_component) {
  const nnet1::AffineTransform *affine =
      dynamic_cast<const nnet1::AffineTransform*>(&nnet1_component);
  KALDI_ASSERT(affine != NULL);
  // default learning rate is 1.0e-05, you can use the --learning-rate or
  // --learning-rates option to nnet-am-copy to change it if you need.
  BaseFloat learning_rate = 1.0e-05;
  if (use_preconditioned_affine_component) {
    int32 rank_in = 20,
          rank_out = 80,
          update_period = 4;
    BaseFloat num_samples_history = 2000.,
              alpha = 4.;
    return new nnet2::AffineComponentPreconditionedOnline(
      nnet2::AffineComponent(affine->GetLinearity(),
        affine->GetBias(),
        learning_rate),
      rank_in,
      rank_out,
      update_period,
      num_samples_history,
      alpha);
  } else {
    return new nnet2::AffineComponent(affine->GetLinearity(),
      affine->GetBias(),
      learning_rate);
  }
}

nnet2::Component *ConvertSoftmaxComponent(
    const nnet1::Component &nnet1_component) {
  const nnet1::Softmax *softmax =
      dynamic_cast<const nnet1::Softmax*>(&nnet1_component);
  KALDI_ASSERT(softmax != NULL);
  return new nnet2::SoftmaxComponent(softmax->InputDim());
}

nnet2::Component *ConvertSigmoidComponent(
    const nnet1::Component &nnet1_component) {
  const nnet1::Sigmoid *sigmoid =
      dynamic_cast<const nnet1::Sigmoid*>(&nnet1_component);
  KALDI_ASSERT(sigmoid != NULL);
  return new nnet2::SigmoidComponent(sigmoid->InputDim());
}

nnet2::Component *ConvertSpliceComponent(
    const nnet1::Component &nnet1_component) {
  const nnet1::Splice *splice =
      dynamic_cast<const nnet1::Splice*>(&nnet1_component);
  KALDI_ASSERT(splice != NULL);
//  int32 low, high;
  std::vector<int32> frame_offsets;

  std::ostringstream ostr;
  splice->WriteData(ostr, false);

  std::istringstream istr(ostr.str());
  ReadIntegerVector(istr, false, &frame_offsets);

  nnet2::SpliceComponent *res = new nnet2::SpliceComponent();
  res->Init(splice->InputDim(), frame_offsets);
  return res;
}


nnet2::Component *ConvertAddShiftComponent(
    const nnet1::Component &nnet1_component) {
  const nnet1::AddShift *add_shift =
      dynamic_cast<const nnet1::AddShift*>(&nnet1_component);
  KALDI_ASSERT(add_shift != NULL);
  Vector<BaseFloat> bias(add_shift->NumParams());

  add_shift->GetParams(&bias);
  CuVector<BaseFloat> cu_bias(bias);

  nnet2::FixedBiasComponent *res = new nnet2::FixedBiasComponent();
  res->Init(cu_bias);
  return res;
}

nnet2::Component *ConvertRescaleComponent(
    const nnet1::Component &nnet1_component) {
  const nnet1::Rescale *rescale =
      dynamic_cast<const nnet1::Rescale*>(&nnet1_component);
  KALDI_ASSERT(rescale != NULL);

  Vector<BaseFloat> scale(rescale->NumParams());
  rescale->GetParams(&scale);

  CuVector<BaseFloat> cu_scale(scale);

  nnet2::FixedScaleComponent *res = new nnet2::FixedScaleComponent();
  res->Init(cu_scale);
  return res;
}

nnet2::Component *ConvertComponent(const nnet1::Component &nnet1_component,
    const bool use_preconditioned_affine_component) {
  nnet1::Component::ComponentType type_in = nnet1_component.GetType();
  switch (type_in) {
    case nnet1::Component::kAffineTransform:
      return ConvertAffineTransformComponent(nnet1_component,
          use_preconditioned_affine_component);
    case nnet1::Component::kSoftmax:
      return ConvertSoftmaxComponent(nnet1_component);
    case nnet1::Component::kSigmoid:
      return ConvertSigmoidComponent(nnet1_component);
    case nnet1::Component::kSplice:
      return ConvertSpliceComponent(nnet1_component); // note, this will for now only handle the
      // special case nnet1::Component::where all splice indexes in nnet1_component are contiguous, e.g.
      // -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 .
    case nnet1::Component::kAddShift:
      return ConvertAddShiftComponent(nnet1_component); // convert to FixedBiasComponent
    case nnet1::Component::kRescale:
      return ConvertRescaleComponent(nnet1_component); // convert to FixedScaleComponent
    default: KALDI_ERR << "Un-handled nnet1 component type "
                       << nnet1::Component::TypeToMarker(type_in);
    return NULL;
  }
}


nnet2::Nnet *ConvertNnet1ToNnet2(const nnet1::Nnet &nnet1,
    const bool use_preconditioned_affine_component) {
  // get a vector of nnet2::Component pointers and initialize the nnet2::Nnet with it.
  size_t size = nnet1.NumComponents();
  std::vector<nnet2::Component*> *components = new std::vector<nnet2::Component*>();
  components->resize(size);
  for (size_t i = 0; i < size; i++) {
      (*components)[i] = ConvertComponent(nnet1.GetComponent(i),
          use_preconditioned_affine_component);
  }

  nnet2::Nnet *res = new nnet2::Nnet();
  res->Init(components);
  delete components;
  return res;
}

}  // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert nnet1 neural net to nnet2 'raw' neural net\n"
        "\n"
        "Usage:  nnet1-to-raw-nnet [options] <nnet1-in> <nnet2-out>\n"
        "e.g.:\n"
        " nnet1-to-raw-nnet srcdir/final.nnet - | nnet-am-init dest/tree dest/topo - dest/0.mdl\n";

    bool binary_write = true, use_preconditioned_affine_component = false;
    int32 srand_seed = 0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Register("use_preconditioned_affine_component",
        &use_preconditioned_affine_component,
        "Using AffineComponentPreconditionOnline instead AffineComponent");

    po.Read(argc, argv);
    srand(srand_seed);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet1_rxfilename = po.GetArg(1),
        raw_nnet2_wxfilename = po.GetArg(2);

    nnet1::Nnet nnet1;
    ReadKaldiObject(nnet1_rxfilename, &nnet1);
    nnet2::Nnet *nnet2 = ConvertNnet1ToNnet2(nnet1,
        use_preconditioned_affine_component);
    WriteKaldiObject(*nnet2, raw_nnet2_wxfilename, binary_write);
    KALDI_LOG << "Converted nnet1 neural net to raw nnet2 and wrote it to "
              << PrintableWxfilename(raw_nnet2_wxfilename);
    delete nnet2;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
