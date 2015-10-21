// ctc/cctc-tombstone-test.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "ctc/cctc-transition-model.h"
#include "ctc/cctc-tombstone.h"
#include "ctc/cctc-graph.h"
#include "ctc/language-model.h"
#include "ctc/cctc-supervision.h"
#include "ctc/cctc-training.h"
#include "ctc/cctc-test-utils.h"
#include "fstext/fstext-lib.h"
#include "cudamatrix/cu-device.h"

// This test program tests things declared in ctc-supervision.h and
// cctc-training.h

namespace kaldi {
namespace ctc {

void Tensor3dCopyTest() {
  int32 dim_x = RandInt(1, 10), dim_y = RandInt(1, 5), dim_z = RandInt(1, 5);
  CuVector<BaseFloat> vec(dim_x * dim_y * dim_z), vec2(dim_x * dim_y * dim_z);
  vec.SetRandn();

  CuVector<BaseFloat> vec_rearranged(dim_x * dim_y * dim_z);
  Tensor3dCopy(dim_x, dim_y, dim_z,
               1, dim_x, dim_x * dim_y,
               dim_y * dim_z, dim_z, 1,
               vec.Data(), vec_rearranged.Data());
  Tensor3dCopy(dim_x, dim_y, dim_z,
               dim_y * dim_z, dim_z, 1,
               1, dim_x, dim_x * dim_y,
               vec_rearranged.Data(), vec2.Data());
  // KALDI_LOG << "vec is " << vec;
  // KALDI_LOG << "vec_rearranged is " << vec_rearranged;
  // KALDI_LOG << "vec2 is " << vec2;
  AssertEqual(vec, vec2);
}

void TestCctcTombstone(const CctcTransitionModel &trans_model) {
  CuMatrix<BaseFloat> weights;
  trans_model.ComputeWeights(&weights);
  CctcHmm hmm(trans_model);

  int32 num_sequences = RandInt(1, 5),
      num_time_steps = RandInt(10, 20);
  if (RandInt(0, 3) == 0)
    num_time_steps *= 30;  // test how it works on long sequences
  CuMatrix<BaseFloat> nnet_output(num_sequences * num_time_steps,
                                  trans_model.NumOutputIndexes());
  bool zero_output = (RandInt(0, 3) == 0);
  if (!zero_output)
    nnet_output.SetRandn();
  CuMatrix<BaseFloat> exp_nnet_output(nnet_output);
  exp_nnet_output.ApplyExp();

  CuMatrix<BaseFloat> denominators(nnet_output.NumRows(),
                                   trans_model.NumHistoryStates());

  denominators.AddMatMat(1.0, exp_nnet_output, kNoTrans, weights, kTrans, 0.0);

  CctcNegativeComputation negative_computation(trans_model, hmm,
                                               exp_nnet_output,
                                               denominators, num_sequences,
                                               NULL);

  BaseFloat forward_prob = negative_computation.Forward(),
      per_frame = forward_prob / (num_sequences * num_time_steps);
  KALDI_LOG << "Forward prob is " << forward_prob
            << " = " << per_frame << " per frame.";
  if (zero_output)
    KALDI_ASSERT(ApproxEqual(BaseFloat(log(2.0 / 3.0)),
                             per_frame));

  CuMatrix<BaseFloat> denominators_deriv(denominators.NumRows(),
                                         denominators.NumCols(),
                                         kUndefined),
      nnet_output_deriv(nnet_output.NumRows(),
                        nnet_output.NumCols(),
                        kUndefined);
  negative_computation.Backward(&nnet_output_deriv,
                                &denominators_deriv);

  { // a check
    BaseFloat output_deriv_sum = nnet_output_deriv.Sum();
    KALDI_LOG << "Sum of nnet-output-deriv is " << output_deriv_sum
              << " vs. expected " << (num_sequences * num_time_steps);
    AssertEqual(output_deriv_sum, BaseFloat(num_sequences * num_time_steps));
  }

  // compute the deriv w.r.t. the output by adding the term
  // that comes via the denominator.
  CuMatrix<BaseFloat> exp_nnet_output_deriv(nnet_output.NumRows(),
                                            nnet_output.NumCols());
  exp_nnet_output_deriv.AddMatMat(1.0, denominators_deriv, kNoTrans,
                                  weights, kNoTrans, 0.0);
  // make it the deriv (via denominator) w.r.t. the actual nnet output, using
  // df/d(exp x) = df/dx * exp(x).
  exp_nnet_output_deriv.MulElements(exp_nnet_output);
  BaseFloat den_sum = exp_nnet_output_deriv.Sum(),
      num_sum = nnet_output_deriv.Sum();
  KALDI_LOG << "den-sum = " << den_sum << ", num-sum = " << num_sum << " (should cancel)";
  KALDI_ASSERT(den_sum + num_sum < 0.05 * (fabs(den_sum) + fabs(num_sum)));

  nnet_output_deriv.AddMat(1.0, exp_nnet_output_deriv);  // combine with the
                                                         // term from the
                                                         // numerators.

  int32 num_tries = 3;
  BaseFloat epsilon = 1.0e-03;
  Vector<BaseFloat> predicted_objf_changes(num_tries),
      observed_objf_changes(num_tries);
  for (int32 p = 0; p < num_tries; p++) {
    CuMatrix<BaseFloat> nnet_delta_output(nnet_output.NumRows(),
                                          nnet_output.NumCols());
    nnet_delta_output.SetRandn();
    nnet_delta_output.Scale(epsilon);
    predicted_objf_changes(p) = TraceMatMat(nnet_output_deriv,
                                            nnet_delta_output, kTrans);
    CuMatrix<BaseFloat> exp_nnet_output_perturbed(nnet_delta_output);
    exp_nnet_output_perturbed.AddMat(1.0, nnet_output);
    exp_nnet_output_perturbed.ApplyExp();
    CuMatrix<BaseFloat> denominators_perturbed(nnet_output.NumRows(),
                                               trans_model.NumHistoryStates());

    denominators_perturbed.AddMatMat(1.0, exp_nnet_output_perturbed,
                                     kNoTrans, weights, kTrans, 0.0);

    CctcNegativeComputation negative_computation_perturbed(trans_model, hmm,
                                                           exp_nnet_output_perturbed,
                                                           denominators_perturbed,
                                                           num_sequences, NULL);

    BaseFloat forward_prob_perturbed = negative_computation_perturbed.Forward();
    observed_objf_changes(p) = forward_prob_perturbed - forward_prob;
  }
  KALDI_LOG << "Predicted objf changes are " << predicted_objf_changes;
  KALDI_LOG << "Observed objf changes are " << observed_objf_changes;
  KALDI_ASSERT(predicted_objf_changes.ApproxEqual(observed_objf_changes, 0.25));
}


void CctcTombstoneTest() {
  CctcTransitionModel trans_model;
  GenerateCctcTransitionModel(&trans_model);
  TestCctcTombstone(trans_model);
}


}  // namespace ctc
}  // namespace kaldi

int main() {
  using namespace kaldi;
  // will later change this to "< 2" and test with GPU.
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    for (int32 i = 0; i < 10; i++) {
      kaldi::ctc::Tensor3dCopyTest();
      kaldi::ctc::CctcTombstoneTest();
    }
  }
}
