// nnet2/nnet-component-test.cc

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

#include "nnet2/nnet-component.h"
#include "util/common-utils.h"
#include <unistd.h> // for sleep().

namespace kaldi {
namespace nnet2 {


void UnitTestGenericComponentInternal(const Component &component) {
  int32 input_dim = component.InputDim(),
      output_dim = component.OutputDim();

  KALDI_LOG << component.Info();
  
  CuVector<BaseFloat> objf_vec(output_dim); // objective function is linear function of output.
  objf_vec.SetRandn(); // set to Gaussian noise.
  
  int32 num_egs = 10 + Rand() % 5;
  CuMatrix<BaseFloat> input(num_egs, input_dim),
      output(num_egs, output_dim);
  input.SetRandn();
  
  int32 rand_seed = Rand();

  RandomComponent *rand_component =
      const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
  if (rand_component != NULL) {
    srand(rand_seed);
    rand_component->ResetGenerator();
  }
  component.Propagate(input, 1, &output);
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
    CuVector<BaseFloat> output_objfs(num_egs);
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
    int32 num_chunks = 1;

    
    component_copy->Backprop(input_ref, output_ref,
                             output_deriv, num_chunks, NULL, &input_deriv);

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
        component.Propagate(perturbed_input, 1, &perturbed_output);
        CuVector<BaseFloat> perturbed_output_objfs(num_egs);
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
      
      CuVector<BaseFloat> output_objfs(num_egs);
      output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
      BaseFloat objf = output_objfs.Sum();

      CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
      for (int32 i = 0; i < output_deriv.NumRows(); i++)
        output_deriv.Row(i).CopyFromVec(objf_vec);
      CuMatrix<BaseFloat> input_deriv; // (input.NumRows(), input.NumCols());

      int32 num_chunks = 1;

      // This will compute the parameter gradient.
      ucomponent->Backprop(input, output, output_deriv, num_chunks,
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
        perturbed_ucomponent->Propagate(input, 1, &output_perturbed);
        CuVector<BaseFloat> output_objfs_perturbed(num_egs);
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
  // works if it has an initializer from int,
  // e.g. tanh, sigmoid.
  
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.

  for (int32 i = 0; i < 5; i++) {
    int32 output_dim = 10 + Rand() % 20,
        group_size = 1 + Rand() % 10,
        input_dim = output_dim * group_size;
    BaseFloat p = 0.8 + 0.1 * (Rand() % 20);
    
    PnormComponent component(input_dim, output_dim, p);
    UnitTestGenericComponentInternal(component);
  }

  {
    PnormComponent component;
    component.InitFromString("input-dim=15 output-dim=5 p=3.0");
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
      sleep(1);
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


void UnitTestPiecewiseLinearComponent() {
  BaseFloat learning_rate = 0.01, max_change = 0.1 * (Rand() % 2);
  int32 dim = 5 + Rand() % 10, N = 3 + 2 * (Rand() % 5);
  {
    PiecewiseLinearComponent component;
    component.Init(dim, N, learning_rate, max_change);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 dim=10 N=5 max-change=0.01";
    PiecewiseLinearComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
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
      sleep(1);
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
      sleep(1);
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


void UnitTestAffineComponentModified() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0, length_cutoff = 10.0,
      max_change = 0.1;
  int32 input_dim = 5 + Rand() % 10, output_dim = 5 + Rand() % 10;
  {
    AffineComponentModified component;
    if (Rand() % 2 == 0) {
      component.Init(learning_rate, input_dim, output_dim,
                     param_stddev, bias_stddev,
                     length_cutoff, max_change);
    } else {
      Matrix<BaseFloat> mat(output_dim + 1, input_dim);
      mat.SetRandn();
      mat.Scale(param_stddev);
      WriteKaldiObject(mat, "tmpf", true);
      sleep(1);
      component.Init(learning_rate, length_cutoff, max_change, "tmpf");
      unlink("tmpf");
    }
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=16 output-dim=15 param-stddev=0.1 cutoff-length=10.0 max-change=0.01";
    AffineComponentModified component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestAffinePreconInputComponent() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0,
      avg_samples = 100.0;
  int32 input_dim = 5 + Rand() % 10, output_dim = 5 + Rand() % 10;

  {
    AffinePreconInputComponent component;
    component.Init(learning_rate, input_dim, output_dim,
                   param_stddev, bias_stddev, avg_samples);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1 avg-samples=100";
    AffinePreconInputComponent component;
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

void UnitTestMixtureProbComponent() {
  BaseFloat learning_rate = 0.01,
      diag_element = 0.8;
  std::vector<int32> sizes;
  int32 num_sizes = 1 + Rand() % 5; // allow 
  for (int32 i = 0; i < num_sizes; i++)
    sizes.push_back(2 + Rand() % 5); // TODO: change to 1 + Rand() % 5
  // and fix test errors.  May be issue in the code itself.
  
  
  {
    MixtureProbComponent component;
    component.Init(learning_rate, diag_element, sizes);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 diag-element=0.9 dims=3:4:5";
    MixtureProbComponent component;
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
  int32 m = 1 + Rand() % 4, n = 1 + Rand() % 4,
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
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=1";
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

void BasicDebugTestForSplice(bool output=false) {
  int32 C=5;
  int32 K=4, contextLen=1;
  int32 R=3+2 * contextLen;
 
  SpliceComponent *c = new SpliceComponent();
  c->Init(C, contextLen, contextLen, K);
  CuMatrix<BaseFloat> in(R, C), in_deriv(R, C);
  CuMatrix<BaseFloat> out(R, c->OutputDim());

  in.SetRandn();
  if (output)
    KALDI_LOG << in;

  c->Propagate(in, 1, &out);
  
  if (output) 
    KALDI_LOG << out;

  out.Set(1);
  
  if (K > 0) {
    CuSubMatrix<BaseFloat> k(out, 0, out.NumRows(), c->OutputDim() - K, K);
    k.Set(-2);
  }

  if (output)
    KALDI_LOG << out;
  
  int32 num_chunks = 1;
  c->Backprop(in, in, out, num_chunks, c, &in_deriv);
  
  if (output)
    KALDI_LOG << in_deriv;
  delete c;
}

void BasicDebugTestForSpliceMax(bool output=false) {
  int32 C=5;
  int32 contextLen=2;
  int32 R= 3 + 2*contextLen;
 
  SpliceMaxComponent *c = new SpliceMaxComponent();
  c->Init(C, contextLen, contextLen);
  CuMatrix<BaseFloat> in(R, C), in_deriv(R, C);
  CuMatrix<BaseFloat> out(R, c->OutputDim());
  
  in.SetRandn();
  if (output)
    KALDI_LOG << in;

  c->Propagate(in, 1, &out);
  
  if (output) 
    KALDI_LOG << out;

  out.Set(5.0);
  
  if (output)
    KALDI_LOG << out;
  
  int32 num_chunks = 1;
  c->Backprop(in, in, out, num_chunks, c, &in_deriv);
  
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


  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
    
    BasicDebugTestForSplice(true);
    BasicDebugTestForSpliceMax(true);
    for (int32 i = 0; i < 3; i++) {
      UnitTestGenericComponent<SigmoidComponent>();
      UnitTestGenericComponent<TanhComponent>();
      UnitTestGenericComponent<PowerComponent>("power=1.5");
      UnitTestGenericComponent<PowerComponent>("power=1.0");
      UnitTestGenericComponent<PermuteComponent>();
      UnitTestGenericComponent<SoftmaxComponent>();
      UnitTestGenericComponent<RectifiedLinearComponent>();
      UnitTestGenericComponent<SoftHingeComponent>();
      UnitTestGenericComponent<PowerExpandComponent>("higher-power-scale=0.1");
      UnitTestMaxoutComponent(); 
      UnitTestPnormComponent(); 
      UnitTestGenericComponent<NormalizeComponent>();
      UnitTestSigmoidComponent();
      UnitTestAffineComponent();
      UnitTestPiecewiseLinearComponent();
      UnitTestScaleComponent();
      UnitTestAffinePreconInputComponent();
      UnitTestBlockAffineComponent();
      UnitTestBlockAffineComponentPreconditioned();
      UnitTestMixtureProbComponent();
      UnitTestSumGroupComponent();
      UnitTestDctComponent();
      UnitTestFixedLinearComponent();
      UnitTestFixedAffineComponent();
      UnitTestFixedScaleComponent();
      UnitTestFixedBiasComponent();
      UnitTestAffineComponentPreconditioned();
      UnitTestAffineComponentPreconditionedOnline();
      UnitTestAffineComponentModified();
      UnitTestDropoutComponent();
      UnitTestAdditiveNoiseComponent();
      UnitTestParsing();
      if (loop == 0)
        KALDI_LOG << "Tests without GPU use succeeded.";
      else
        KALDI_LOG << "Tests with GPU use (if available) succeeded.";
    }
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}
