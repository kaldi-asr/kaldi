// nnet2/online-nnet2-decodable-test.cc

// Copyright 2014  Johns Hopkins University (author:  Daniel Povey)

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

#include "hmm/transition-model.h"
#include "nnet2/nnet-component.h"
#include "nnet2/decodable-am-nnet.h"
#include "nnet2/online-nnet2-decodable.h"
#include "feat/online-feature.h"
#include "hmm/hmm-test-utils.h"

namespace kaldi {
namespace nnet2 {


void UnitTestNnetDecodable() {
  std::vector<int32> phones;
  phones.push_back(1);
  for (int32 i = 2; i < 20; i++)
    if (rand() % 2 == 0)
      phones.push_back(i);
  int32 N = 2 + rand() % 2, // context-size N is 2 or 3.
      P = rand() % N;  // Central-phone is random on [0, N)

  std::vector<int32> num_pdf_classes;

  ContextDependency *ctx_dep =
      GenRandContextDependencyLarge(phones, N, P,
                                    true, &num_pdf_classes);

  HmmTopology topo = GetDefaultTopology(phones);

  TransitionModel trans_model(*ctx_dep, topo);

  delete ctx_dep; // We won't need this further.
  ctx_dep = NULL;

  int32 input_dim = 40, output_dim = trans_model.NumPdfs();
  Nnet *nnet = GenRandomNnet(input_dim, output_dim);

  AmNnet am_nnet(*nnet);
  delete nnet;
  nnet = NULL;
  Vector<BaseFloat> priors(output_dim);
  priors.SetRandn();
  priors.ApplyExp();
  priors.Scale(1.0 / priors.Sum());

  am_nnet.SetPriors(priors);

  DecodableNnet2OnlineOptions opts;
  opts.max_nnet_batch_size = 20;
  opts.acoustic_scale = 0.1;

  opts.pad_input = (rand() % 2 == 0);

  int32 num_input_frames = 400;
  Matrix<BaseFloat> input_feats(num_input_frames, input_dim);
  input_feats.SetRandn();

  OnlineMatrixFeature matrix_feature(input_feats);

  DecodableNnet2Online online_decodable(am_nnet, trans_model,
                                        opts, &matrix_feature);

  DecodableAmNnet offline_decodable(trans_model, am_nnet,
                                    CuMatrix<BaseFloat>(input_feats),
                                    opts.pad_input,
                                    opts.acoustic_scale);

  KALDI_ASSERT(online_decodable.NumFramesReady() ==
               offline_decodable.NumFramesReady());
  int32 num_frames = online_decodable.NumFramesReady(),
      num_tids = trans_model.NumTransitionIds();

  for (int32 i = 0; i < 50; i++) {

    int32 t = rand() % num_frames, tid = 1 + rand() % num_tids;
    BaseFloat l1 = online_decodable.LogLikelihood(t, tid),
        l2 = offline_decodable.LogLikelihood(t, tid);
    KALDI_ASSERT(ApproxEqual(l1, l2));
  }
}

} // namespace nnet2
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;
  using kaldi::int32;

  for (int32 i = 0; i < 3; i++)
    UnitTestNnetDecodable();
  return 0;
}


