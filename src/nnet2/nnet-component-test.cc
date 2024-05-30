// nnet2/nnet-component-test.cc

// Copyright 2012-2014  Johns Hopkins University (author:  Daniel Povey)
//                2015  Guoguo Chen

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

#include "nnet2/nnet-component.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet2 {


void UnitTestGenericComponentInternal(const Component &component,
                                      const ChunkInfo in_info,
                                      const ChunkInfo out_info)  {

  CuMatrix<BaseFloat> input(in_info.NumRows(), in_info.NumCols()),
      output(1, out_info.NumRows() * out_info.NumCols());
  input.SetRandn();
  CuVector<BaseFloat> objf_vec(out_info.NumCols()); // objective function is linear function of output.
  objf_vec.SetRandn(); // set to Gaussian noise.

  int32 rand_seed = Rand();

  RandomComponent *rand_component =
      const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
  if (rand_component != NULL) {
    srand(rand_seed);
    rand_component->ResetGenerator();
  }
  component.Propagate(in_info, out_info, input, &output);
  {
    bool binary = (Rand() % 2 == 0);
    Output ko("tmpf", binary);
    component.Write(ko.Stream(), binary);
  }
  Component *component_copy;
  {
    bool binary_in;
    Input ki("tmpf", &binary_in);
    component_copy = Component::ReadNew(ki.Stream(), binary_in);
  }
  unlink("tmpf");

  { // Test backward derivative is correct.
    CuVector<BaseFloat> output_objfs(out_info.NumRows());
    output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
    BaseFloat objf = output_objfs.Sum();


    CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
    for (int32 i = 0; i < output_deriv.NumRows(); i++)
      output_deriv.Row(i).CopyFromVec(objf_vec);

    CuMatrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());


    CuMatrix<BaseFloat> empty_mat;
    CuMatrix<BaseFloat> &input_ref =
        (component_copy->BackpropNeedsInput() ? input : empty_mat),
        &output_ref =
        (component_copy->BackpropNeedsOutput() ? output : empty_mat);

    component_copy->Backprop(in_info, out_info, input_ref, output_ref,
                             output_deriv, NULL, &input_deriv);

    int32 num_ok = 0, num_bad = 0, num_tries = 10;
    KALDI_LOG << "Comparing feature gradients " << num_tries << " times.";
    for (int32 i = 0; i < num_tries; i++) {
      CuMatrix<BaseFloat> perturbed_input(input.NumRows(), input.NumCols());
      {
        RandomComponent *rand_component =
            const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
        if (rand_component != NULL) {
          srand(rand_seed);
          rand_component->ResetGenerator();
        }
      }
      perturbed_input.SetRandn();
      perturbed_input.Scale(1.0e-04); // scale by a small amount so it's like a delta.
      BaseFloat predicted_difference = TraceMatMat(perturbed_input,
                                                   input_deriv, kTrans);
      perturbed_input.AddMat(1.0, input); // now it's the input + a delta.
      { // Compute objf with perturbed input and make sure it matches
        // prediction.
        CuMatrix<BaseFloat> perturbed_output(output.NumRows(), output.NumCols());
        {
          RandomComponent *rand_component =
              const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
          if (rand_component != NULL) {
            srand(rand_seed);
            rand_component->ResetGenerator();
          }
        }
        component.Propagate(in_info, out_info, perturbed_input, &perturbed_output);
        CuVector<BaseFloat> perturbed_output_objfs(out_info.NumRows());
        perturbed_output_objfs.AddMatVec(1.0, perturbed_output, kNoTrans,
                                         objf_vec, 0.0);
        BaseFloat perturbed_objf = perturbed_output_objfs.Sum(),
             observed_difference = perturbed_objf - objf;
        KALDI_LOG << "Input gradients: comparing " << predicted_difference
                  << " and " << observed_difference;
        if (fabs(predicted_difference - observed_difference) >
            0.15 * fabs((predicted_difference + observed_difference)/2) &&
            fabs(predicted_difference - observed_difference) > 1.0e-06) {
          KALDI_WARN << "Bad difference!";
          num_bad++;
        } else {
          num_ok++;
        }
      }
    }
    KALDI_LOG << "Succeeded for " << num_ok << " out of " << num_tries
              << " tries.";
    if (num_ok <= num_bad) {
      delete component_copy;
      KALDI_ERR << "Feature-derivative check failed";
    }
  }

  UpdatableComponent *ucomponent =
      dynamic_cast<UpdatableComponent*>(component_copy);

  if (ucomponent != NULL) { // Test parameter derivative is correct.

    int32 num_ok = 0, num_bad = 0, num_tries = 10;
    KALDI_LOG << "Comparing model gradients " << num_tries << " times.";
    for (int32 i = 0; i < num_tries; i++) {
      UpdatableComponent *perturbed_ucomponent =
          dynamic_cast<UpdatableComponent*>(ucomponent->Copy()),
          *gradient_ucomponent =
          dynamic_cast<UpdatableComponent*>(ucomponent->Copy());
      KALDI_ASSERT(perturbed_ucomponent != NULL);
      gradient_ucomponent->SetZero(true); // set params to zero and treat as gradient.
      BaseFloat perturb_stddev = 5.0e-04;
      perturbed_ucomponent->PerturbParams(perturb_stddev);

      CuVector<BaseFloat> output_objfs(out_info.NumRows());
      output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
      BaseFloat objf = output_objfs.Sum();

      CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
      for (int32 i = 0; i < output_deriv.NumRows(); i++)
        output_deriv.Row(i).CopyFromVec(objf_vec);
      CuMatrix<BaseFloat> input_deriv; // (input.NumRows(), input.NumCols());

      // This will compute the parameter gradient.
      ucomponent->Backprop(in_info, out_info, input, output, output_deriv,
                           gradient_ucomponent, &input_deriv);

      // Now compute the perturbed objf.
      BaseFloat objf_perturbed;
      {
        CuMatrix<BaseFloat> output_perturbed; // (num_egs, output_dim);
        {
          RandomComponent *rand_component =
              const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
          if (rand_component != NULL) {
            srand(rand_seed);
            rand_component->ResetGenerator();
          }
        }
        perturbed_ucomponent->Propagate(in_info, out_info, input, &output_perturbed);
        CuVector<BaseFloat> output_objfs_perturbed(out_info.NumRows());
        output_objfs_perturbed.AddMatVec(1.0, output_perturbed,
                                         kNoTrans, objf_vec, 0.0);
        objf_perturbed = output_objfs_perturbed.Sum();
      }

      BaseFloat delta_objf_observed = objf_perturbed - objf,
          delta_objf_predicted = (perturbed_ucomponent->DotProduct(*gradient_ucomponent) -
                                  ucomponent->DotProduct(*gradient_ucomponent));

      KALDI_LOG << "Model gradients: comparing " << delta_objf_observed
                << " and " << delta_objf_predicted;
      if (fabs(delta_objf_predicted - delta_objf_observed) >
          0.05 * (fabs(delta_objf_predicted + delta_objf_observed)/2) &&
          fabs(delta_objf_predicted - delta_objf_observed) > 1.0e-06) {
        KALDI_WARN << "Bad difference!";
        num_bad++;
      } else {
        num_ok++;
      }
      delete perturbed_ucomponent;
      delete gradient_ucomponent;
    }
    if (num_ok < num_bad) {
      delete component_copy;
      KALDI_ERR << "model-derivative check failed";
    }
  }
  delete component_copy; // No longer needed.
}

void UnitTestGenericComponentInternal(const Component &component) {
  int32 input_dim = component.InputDim(),
      output_dim = component.OutputDim();

  KALDI_LOG << component.Info();
  int32 num_egs = 10 + Rand() % 5;
  int32 num_chunks = 1,
        first_offset = 0,
        last_offset = num_egs-1;

  ChunkInfo in_info(input_dim, num_chunks, first_offset, last_offset);
  ChunkInfo out_info(output_dim, num_chunks, first_offset, last_offset);
  UnitTestGenericComponentInternal(component, in_info, out_info);
}



void UnitTestSigmoidComponent() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  int32 input_dim = 10 + Rand() % 50;
  {
    SigmoidComponent sigmoid_component(input_dim);
    UnitTestGenericComponentInternal(sigmoid_component);
  }
  {
    SigmoidComponent sigmoid_component;
    sigmoid_component.InitFromString("dim=15");
    UnitTestGenericComponentInternal(sigmoid_component);
  }
}

template<class T>
void UnitTestGenericComponent(std::string extra_str = "") {
  // works if it has an initializer from int,
  // e.g. tanh, sigmoid.

  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  int32 input_dim = 10 + Rand() % 50;
  {
    T component(input_dim);
    UnitTestGenericComponentInternal(component);
  }
  {
    T component;
    component.InitFromString(static_cast<std::string>("dim=15 ") + extra_str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestMaxoutComponent() {
  // works if it has an initializer from int,
  // e.g. tanh, sigmoid.

  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  for (int32 i = 0; i < 5; i++) {
    int32 output_dim = 10 + Rand() % 20,
        group_size = 1 + Rand() % 10,
        input_dim = output_dim * group_size;

    MaxoutComponent component(input_dim, output_dim);
    UnitTestGenericComponentInternal(component);
  }

  {
    MaxoutComponent component;
    component.InitFromString("input-dim=15 output-dim=5");
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestPnormComponent() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  int32 num_fail = 0, num_tries = 4;
  for (int32 i = 0; i < num_tries; i++) {
    try {
      int32 output_dim = 10 + Rand() % 20,
          group_size = 1 + Rand() % 10,
          input_dim = output_dim * group_size;
      BaseFloat p = 1.0 + 0.1 * (Rand() % 20);

      PnormComponent component(input_dim, output_dim, p);
      UnitTestGenericComponentInternal(component);
    } catch (...) {
      KALDI_WARN << "Ignoring test failure in UnitTestPnormComponent().";
      num_fail++;
    }
  }
  if (num_fail >= num_tries/2) {
    KALDI_ERR << "Too many test failures.";
  }
}

void UnitTestMaxpoolingComponent() {
  // works if it has an initializer from int,
  // e.g. tanh, sigmoid.
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  for (int32 i = 0; i < 5; i++) {
    int32 pool_stride = 5 + Rand() % 10,
          pool_size = 2 + Rand() % 3,
          num_pools = 1 + Rand() % 10;
    int32 output_dim = num_pools * pool_stride;
    int32 num_patches = num_pools * pool_size;
    int32 input_dim = pool_stride * num_patches;

    MaxpoolingComponent component(input_dim, output_dim,
                                  pool_size, pool_stride);
    UnitTestGenericComponentInternal(component);
  }

  {
    MaxpoolingComponent component;
    component.InitFromString("input-dim=192 output-dim=64 pool-size=3 pool-stride=16");
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestAffineComponent() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0;
  int32 input_dim = 5 + Rand() % 10, output_dim = 5 + Rand() % 10;
  {
    AffineComponent component;
    if (Rand() % 2 == 0) {
      component.Init(learning_rate, input_dim, output_dim,
                     param_stddev, bias_stddev);
    } else {
      Matrix<BaseFloat> mat(output_dim + 1, input_dim);
      mat.SetRandn();
      mat.Scale(param_stddev);
      WriteKaldiObject(mat, "tmpf", true);
      Sleep(0.5);
      component.Init(learning_rate, "tmpf");
      unlink("tmpf");
    }
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1";
    AffineComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestConvolutional1dComponent() {
  BaseFloat learning_rate = 0.01,
            param_stddev = 0.1, bias_stddev = 1.0;
  int32 patch_stride = 10, patch_step = 1, patch_dim = 4;
  int32 num_patches = 1 + (patch_stride - patch_dim) / patch_step;
  int32 num_splice = 5 + Rand() % 10, num_filters = 5 + Rand() % 10;
  int32 input_dim = patch_stride * num_splice;
  int32 filter_dim = patch_dim * num_splice;
  int32 output_dim = num_patches * num_filters;
  {
    Convolutional1dComponent component;
    if (Rand() % 2 == 0) {
      component.Init(learning_rate, input_dim, output_dim,
                     patch_dim, patch_step, patch_stride,
                     param_stddev, bias_stddev, true);
    } else {
      Matrix<BaseFloat> mat(num_filters, filter_dim + 1);
      mat.SetRandn();
      mat.Scale(param_stddev);
      WriteKaldiObject(mat, "tmpf", true);
      Sleep(0.5);
      component.Init(learning_rate, patch_dim,
                     patch_step, patch_stride, "tmpf", false);
      unlink("tmpf");
    }
    UnitTestGenericComponentInternal(component);
  }
  {
    // appended-conv is false by default
    const char *str = "learning-rate=0.01 input-dim=100 output-dim=70 param-stddev=0.1 patch-dim=4 patch-step=1 patch-stride=10";
    Convolutional1dComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=100 output-dim=70 param-stddev=0.1 patch-dim=4 patch-step=1 patch-stride=10 appended-conv=true";
    Convolutional1dComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestDropoutComponent() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  int32 num_fail = 0, num_tries = 4;
  for (int32 i = 0; i < num_tries; i++) {
    try {
      int32 input_dim = 10 + Rand() % 50;
      {
        DropoutComponent dropout_component(input_dim, 0.5, 0.3);
        UnitTestGenericComponentInternal(dropout_component);
      }
      {
        DropoutComponent dropout_component;
        dropout_component.InitFromString("dim=15 dropout-proportion=0.6 dropout-scale=0.1");
        UnitTestGenericComponentInternal(dropout_component);
      }
    } catch (...) {
      KALDI_WARN << "Ignoring test failure in UnitTestDropoutComponent().";
      num_fail++;
    }
  }
  if (num_fail >= num_tries/2) {
    KALDI_ERR << "Too many test failures.";
  }
}

void UnitTestAdditiveNoiseComponent() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  int32 num_fail = 0, num_tries = 4;
  for (int32 i = 0; i < num_tries; i++) {
    try {
      int32 input_dim = 10 + Rand() % 50;
      {
        AdditiveNoiseComponent additive_noise_component(input_dim, 0.1);
        UnitTestGenericComponentInternal(additive_noise_component);
      }
      {
        AdditiveNoiseComponent additive_noise_component;
        additive_noise_component.InitFromString("dim=15 stddev=0.2");
        UnitTestGenericComponentInternal(additive_noise_component);
      }
    } catch (...) {
      KALDI_WARN << "Ignoring failure in AdditiveNoiseComponent test";
      num_fail++;
    }
  }
  if (num_fail >= num_tries/2) {
    KALDI_ERR << "Too many test failures.";
  }
}

void UnitTestScaleComponent() {
  int32 dim = 1 + Rand() % 10;
  BaseFloat scale = 0.1 + Rand() % 3;
  {
    ScaleComponent component;
    if (Rand() % 2 == 0) {
      component.Init(dim, scale);
    } else {
      std::ostringstream str;
      str << "dim=" << dim << " scale=" << scale;
      component.InitFromString(str.str());
    }
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestAffineComponentPreconditioned() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0, alpha = 0.01,
      max_change = 100.0;
  int32 input_dim = 5 + Rand() % 10, output_dim = 5 + Rand() % 10;
  {
    AffineComponentPreconditioned component;
    if (Rand() % 2 == 0) {
      component.Init(learning_rate, input_dim, output_dim,
                     param_stddev, bias_stddev,
                     alpha, max_change);
    } else {
      Matrix<BaseFloat> mat(output_dim + 1, input_dim);
      mat.SetRandn();
      mat.Scale(param_stddev);
      WriteKaldiObject(mat, "tmpf", true);
      Sleep(0.5);
      component.Init(learning_rate, alpha, max_change, "tmpf");
      unlink("tmpf");
    }
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=16 output-dim=15 param-stddev=0.1 alpha=0.01";
    AffineComponentPreconditioned component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestAffineComponentPreconditionedOnline() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0, num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_sample = 0.1, update_period = 1;
  int32 input_dim = 5 + Rand() % 10, output_dim = 5 + Rand() % 10,
      rank_in = 1 + Rand() % 5, rank_out = 1 + Rand() % 5;
  {
    AffineComponentPreconditionedOnline component;
    if (Rand() % 2 == 0) {
      component.Init(learning_rate, input_dim, output_dim,
                     param_stddev, bias_stddev,
                     rank_in, rank_out, update_period,
                     num_samples_history, alpha,
                     max_change_per_sample);
    } else {
      Matrix<BaseFloat> mat(output_dim + 1, input_dim);
      mat.SetRandn();
      mat.Scale(param_stddev);
      WriteKaldiObject(mat, "tmpf", true);
      Sleep(0.5);
      component.Init(learning_rate, rank_in, rank_out,
                     update_period, num_samples_history, alpha,
                     max_change_per_sample, "tmpf");
      unlink("tmpf");
    }
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=16 output-dim=15 param-stddev=0.1 num-samples-history=3000 alpha=2.0 update-period=1 rank-in=5 rank-out=6";
    AffineComponentPreconditionedOnline component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestBlockAffineComponent() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 0.1;
  int32 num_blocks = 1 + Rand() % 3,
         input_dim = num_blocks * (2 + Rand() % 4),
        output_dim = num_blocks * (2 + Rand() % 4);

  {
    BlockAffineComponent component;
    component.Init(learning_rate, input_dim, output_dim,
                   param_stddev, bias_stddev, num_blocks);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1 num-blocks=5";
    BlockAffineComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestBlockAffineComponentPreconditioned() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0, alpha = 3.0;
  int32 num_blocks = 1 + Rand() % 3,
         input_dim = num_blocks * (2 + Rand() % 4),
        output_dim = num_blocks * (2 + Rand() % 4);

  {
    BlockAffineComponentPreconditioned component;
    component.Init(learning_rate, input_dim, output_dim,
                   param_stddev, bias_stddev, num_blocks, alpha);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1 num-blocks=5 alpha=3.0";
    BlockAffineComponentPreconditioned component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestSumGroupComponent() {
  std::vector<int32> sizes;
  int32 num_sizes = 1 + Rand() % 5;
  for (int32 i = 0; i < num_sizes; i++)
    sizes.push_back(1 + Rand() % 5);

  {
    SumGroupComponent component;
    component.Init(sizes);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "sizes=3:4:5";
    SumGroupComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestDctComponent() {
  int32 m = 3 + Rand() % 4, n = 3 + Rand() % 4,
  dct_dim = m, dim = m * n;
  bool reorder = (Rand() % 2 == 0);
  {
    DctComponent component;
    component.Init(dim, dct_dim, reorder);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=2";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=3";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=4";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestFixedLinearComponent() {
  int32 m = 1 + Rand() % 4, n = 1 + Rand() % 4;
  {
    CuMatrix<BaseFloat> mat(m, n);
    mat.SetRandn();
    FixedLinearComponent component;
    component.Init(mat);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestFixedAffineComponent() {
  int32 m = 15 + Rand() % 4, n = 15 + Rand() % 4;
  {
    CuMatrix<BaseFloat> mat(m, n);
    mat.SetRandn();
    FixedAffineComponent component;
    component.Init(mat);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestFixedScaleComponent() {
  int32 m = 1 + Rand() % 20;
  {
    CuVector<BaseFloat> vec(m);
    vec.SetRandn();
    FixedScaleComponent component;
    component.Init(vec);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestFixedBiasComponent() {
  int32 m = 1 + Rand() % 20;
  {
    CuVector<BaseFloat> vec(m);
    vec.SetRandn();
    FixedBiasComponent component;
    component.Init(vec);
    UnitTestGenericComponentInternal(component);
  }
}



void UnitTestParsing() {
  int32 i;
  BaseFloat f;
  bool b;
  std::vector<int32> v;
  std::string s = "x=y";
  KALDI_ASSERT(ParseFromString("foo", &s, &i) == false
               && s == "x=y");
  KALDI_ASSERT(ParseFromString("foo", &s, &f) == false
               && s == "x=y");
  KALDI_ASSERT(ParseFromString("foo", &s, &v) == false
               && s == "x=y");
  KALDI_ASSERT(ParseFromString("foo", &s, &b) == false
               && s == "x=y");
  {
    std::string s = "x=1";
    KALDI_ASSERT(ParseFromString("x", &s, &i) == true
                 && i == 1 && s == "");
    s = "a=b x=1";
    KALDI_ASSERT(ParseFromString("x", &s, &i) == true
                 && i == 1 && s == "a=b");
  }
  {
    std::string s = "foo=false";
    KALDI_ASSERT(ParseFromString("foo", &s, &b) == true
                 && b == false && s == "");
    s = "x=y foo=true a=b";
    KALDI_ASSERT(ParseFromString("foo", &s, &b) == true
                 && b == true && s == "x=y a=b");
  }

  {
    std::string s = "foobar x=1";
    KALDI_ASSERT(ParseFromString("x", &s, &f) == true
                 && f == 1.0 && s == "foobar");
    s = "a=b x=1 bxy";
    KALDI_ASSERT(ParseFromString("x", &s, &f) == true
                 && f == 1.0 && s == "a=b bxy");
  }
  {
    std::string s = "x=1:2:3";
    KALDI_ASSERT(ParseFromString("x", &s, &v) == true
                 && v.size() == 3 && v[0] == 1 && v[1] == 2 && v[2] == 3
                 && s == "");
    s = "a=b x=1:2:3 c=d";
    KALDI_ASSERT(ParseFromString("x", &s, &v) == true
                 && f == 1.0 && s == "a=b c=d");
  }

}

void UnitTestSpliceComponent() {
  int32 feat_dim = RandInt(1, 20),
      const_dim =  RandInt(0, 10),
      left_context = RandInt(-5, 0),
      right_context = RandInt(0, 5),
      num_chunks = RandInt(1, 20);
        // multiple chunks are required as splice component
        // has separate index computation logic for more than one chunks
  KALDI_LOG << " Feat_dim :" << feat_dim << " const_dim: " << const_dim  ;
  std::vector<bool> contiguous(2);
  contiguous[0] = true;
  contiguous[1] = false;
  for (int32 i = 0; i < contiguous.size(); i++) {
    std::vector<int32> splice_indexes;
    if (contiguous[i]) {
      // create contiguous set of splice indexes in the range
      // (-left_context, right_context)
      KALDI_LOG << "Testing contiguous splice component";
      splice_indexes.reserve(right_context - left_context + 1);
      for (int32 i = left_context; i <= right_context; i++)
        splice_indexes.push_back(i);
    } else  {
      // generate random splice indexes in range (-left_context, right_context)
      KALDI_LOG << "Testing non-contiguous splice component";
      int32 num_left_splice_indexes = RandInt(0, -left_context) + 1;
      int32 num_right_splice_indexes = RandInt(0, right_context);
      splice_indexes.reserve(num_left_splice_indexes + num_right_splice_indexes);
      while (splice_indexes.size() < num_left_splice_indexes)  {
        int32 new_index = RandInt(left_context, 0);
        // check if the index already exists in the vector
        if (std::find(splice_indexes.begin(), splice_indexes.end(), new_index)
            == splice_indexes.end())  {
          splice_indexes.push_back(new_index);
        }
      }
      while (splice_indexes.size() < num_left_splice_indexes + num_right_splice_indexes)  {
        int32 new_index = RandInt(0, right_context);
        // check if the index already exists in the vector
        if (std::find(splice_indexes.begin(), splice_indexes.end(), new_index)
            == splice_indexes.end())  {
          splice_indexes.push_back(new_index);
        }
      }
      sort(splice_indexes.begin(), splice_indexes.end());
      if (splice_indexes.back() < 0) // will fail assertion in init of component
        splice_indexes.push_back(0);
    }
    std::vector<int32> input_offsets;
    for (int32 i = 0; i < splice_indexes.size(); i++) {
      input_offsets.push_back(splice_indexes[i] - splice_indexes.front());
      KALDI_LOG << i << " : " << splice_indexes[i] << " : " << input_offsets[i] ;
    }
    int32 output_offset = -splice_indexes.front();
    SpliceComponent *component = new SpliceComponent();
    component->Init(feat_dim + const_dim, splice_indexes, const_dim);
    ChunkInfo in_info = ChunkInfo(feat_dim + const_dim, num_chunks,
                                  input_offsets),
              out_info = ChunkInfo(feat_dim * splice_indexes.size() + const_dim,
                                   num_chunks, output_offset, output_offset);
    UnitTestGenericComponentInternal(*component, in_info, out_info);
    delete component;
  }
}

void BasicDebugTestForSpliceMax(bool output=false) {
  int32 C=5,
        context_len=2,
        R= 3 + 2*context_len;

  SpliceMaxComponent *c = new SpliceMaxComponent();
  std::vector<int32> context(2 * context_len + 1);
  for (int32 i = -1 * context_len; i <= context_len; i++)
    context[i + context_len] = i;
  c->Init(C, context);
  CuMatrix<BaseFloat> in(R, C), in_deriv(R, C);
  CuMatrix<BaseFloat> out(R, c->OutputDim());
  ChunkInfo in_info = ChunkInfo(C, 1, 0, R - 1),
            out_info = ChunkInfo(C, 1, context_len, R - 1 - context_len);

  in.SetRandn();
  if (output)
    KALDI_LOG << in;

  c->Propagate(in_info, out_info, in, &out);

  if (output)
    KALDI_LOG << out;

  out.Set(5.0);

  if (output)
    KALDI_LOG << out;

  c->Backprop(in_info, out_info, in, in, out, c, &in_deriv);

  if (output)
    KALDI_LOG << in_deriv;

  delete c;
}


} // namespace nnet2
} // namespace kaldi

#include "matrix/matrix-functions.h"


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;

  int32 loop = 0;
#if HAVE_CUDA == 1
  for (loop = 0; loop < 2; loop++) {
    //// Uncomment the following line to expose the bug in UnitTestDropoutComponent
    //CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif

    BasicDebugTestForSpliceMax(true);
    // We used to test this 3 times, but now that nnet2 is rarely changed,
    // reducing it to once.
    for (int32 i = 0; i < 1; i++) {
      UnitTestGenericComponent<SigmoidComponent>();
      UnitTestGenericComponent<TanhComponent>();
      UnitTestGenericComponent<PowerComponent>("power=1.5");
      UnitTestGenericComponent<PowerComponent>("power=1.0");
      UnitTestGenericComponent<PermuteComponent>();
      UnitTestGenericComponent<SoftmaxComponent>();
      UnitTestGenericComponent<LogSoftmaxComponent>();
      UnitTestGenericComponent<RectifiedLinearComponent>();
      UnitTestGenericComponent<SoftHingeComponent>();
      UnitTestSpliceComponent();
      UnitTestMaxoutComponent();
      UnitTestPnormComponent();
      UnitTestMaxpoolingComponent();
      UnitTestGenericComponent<NormalizeComponent>();
      UnitTestSigmoidComponent();
      UnitTestAffineComponent();
      UnitTestScaleComponent();
      UnitTestBlockAffineComponent();
      UnitTestBlockAffineComponentPreconditioned();
      UnitTestSumGroupComponent();
      UnitTestDctComponent();
      UnitTestFixedLinearComponent();
      UnitTestFixedAffineComponent();
      UnitTestFixedScaleComponent();
      UnitTestFixedBiasComponent();
      UnitTestAffineComponentPreconditioned();
      UnitTestAffineComponentPreconditionedOnline();
      UnitTestConvolutional1dComponent();
      UnitTestDropoutComponent();
      UnitTestAdditiveNoiseComponent();
      UnitTestParsing();
      if (loop == 0)
        KALDI_LOG << "Tests without GPU use succeeded.";
      else
        KALDI_LOG << "Tests with GPU use (if available) succeeded.";
    }
#if HAVE_CUDA == 1
  } // No for loop if 'HAVE_CUDA != 1',
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}
