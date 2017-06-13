// nnet2/am-nnet-test.cc

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
#include "hmm/hmm-test-utils.h"
#include "nnet2/am-nnet.h"


namespace kaldi {
namespace nnet2 {


void UnitTestAmNnet() {
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

  bool binary = (rand() % 2 == 0);
  std::ostringstream os;
  am_nnet.Write(os, binary);
  AmNnet am_nnet2;
  std::istringstream is(os.str());
  am_nnet2.Read(is, binary);

  std::ostringstream os2;
  am_nnet2.Write(os2, binary);

  KALDI_ASSERT(os2.str() == os.str());
}

} // namespace nnet2
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;

  UnitTestAmNnet();
  return 0;
}

