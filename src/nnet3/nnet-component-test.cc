// nnet3/nnet-component-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {

void TestNnetComponentIo(Component *c) {
  bool binary = (Rand() % 2 == 0);
  std::ostringstream os1;
  c->Write(os1, binary);
  std::istringstream is(os1.str());
  Component *c2 = Component::ReadNew(is, binary);
  std::ostringstream os2;
  c2->Write(os2, binary);
  if (!binary) {
    KALDI_ASSERT(os2.str() == os1.str());
  }
  delete c2;
}

void TestNnetComponentCopy(Component *c) {
  Component *c2 = c->Copy();
  KALDI_ASSERT(c->Info() == c2->Info());
  delete c2;
}

void TestNnetComponentAddScale(Component *c) {
  Component *c2 = c->Copy();
  Component *c3 = c2->Copy();
  c3->Add(0.5, *c2);
  c2->Scale(1.5);
  KALDI_ASSERT(c2->Info() == c3->Info());
  delete c2;
  delete c3;
}


// tests the properties kPropagateAdds, kBackpropAdds,
// kBackpropNeedsInput, kBackpropNeedsOutput.
void TestSimpleComponentPropagateProperties(const Component &c) {
  Component *c_copy = NULL;
  if (RandInt(0, 1) == 0)
    c_copy = c.Copy();  // This will test backprop with an updatable component.
  int32 input_dim = c.InputDim(),
      output_dim = c.OutputDim(),
      num_rows = RandInt(1, 100);
  int32 properties = c.Properties();
  CuMatrix<BaseFloat> input_data(num_rows, input_dim),
      output_data1(num_rows, output_dim),
      output_data2(num_rows, output_dim),
      output_data3(input_data);
  output_data2.Add(1.0);

  if ((properties & kPropagateAdds) && (properties & kPropagateInPlace)) {
    KALDI_ERR << "kPropagateAdds and kPropagateInPlace flags are incompatible.";
  }

  c.Propagate(NULL, input_data, &output_data1);
  c.Propagate(NULL, input_data, &output_data2);
  if (properties & kPropagateInPlace) {
    c.Propagate(NULL, output_data3, &output_data3);
    if (!output_data1.ApproxEqual(output_data3)) {
      KALDI_ERR << "Test of kPropagateInPlace flag for component of type "
                << c.Type() << " failed.";
    }
  }
  if (properties & kPropagateAdds)
    output_data2.Add(-1.0); // remove the offset
  AssertEqual(output_data1, output_data2);

  CuMatrix<BaseFloat> output_deriv(num_rows, output_dim);
  output_deriv.SetRandn();
  CuMatrix<BaseFloat> input_deriv1(num_rows, input_dim),
      input_deriv2(num_rows, input_dim),
      input_deriv3(output_deriv);
  input_deriv2.Add(1.0);
  CuMatrix<BaseFloat> empty_mat;

  // test with input_deriv1 that's zero
  c.Backprop("foobar", NULL,
             ((properties & kBackpropNeedsInput) ? input_data : empty_mat),
             ((properties & kBackpropNeedsOutput) ? output_data1 : empty_mat),
             output_deriv,
             c_copy,
             &input_deriv1);
  // test with input_deriv2 that's all ones.
  c.Backprop("foobar", NULL,
             ((properties & kBackpropNeedsInput) ? input_data : empty_mat),
             ((properties & kBackpropNeedsOutput) ? output_data1 : empty_mat),
             output_deriv,
             c_copy,
             &input_deriv2);
  // test backprop in place, if supported.
  if (properties & kBackpropInPlace) {
    c.Backprop("foobar", NULL,
               ((properties & kBackpropNeedsInput) ? input_data : empty_mat),
               ((properties & kBackpropNeedsOutput) ? output_data1 : empty_mat),
               input_deriv3,
               c_copy,
               &input_deriv3);
  }

  if (properties & kBackpropAdds)
    input_deriv2.Add(-1.0);  // subtract the offset.
  AssertEqual(input_deriv1, input_deriv2);
  if (properties & kBackpropInPlace)
    AssertEqual(input_deriv1, input_deriv3);
  delete c_copy;
}

bool TestSimpleComponentDataDerivative(const Component &c,
                                       BaseFloat perturb_delta) {
  int32 input_dim = c.InputDim(),
      output_dim = c.OutputDim(),
      num_rows = RandInt(1, 100);
  int32 properties = c.Properties();
  CuMatrix<BaseFloat> input_data(num_rows, input_dim),
      output_data(num_rows, output_dim),
      output_deriv(num_rows, output_dim);
  input_data.SetRandn();
  output_deriv.SetRandn();

  c.Propagate(NULL, input_data, &output_data);

  CuMatrix<BaseFloat> input_deriv(num_rows, input_dim), empty_mat;
  c.Backprop("foobar", NULL,
             ((properties & kBackpropNeedsInput) ? input_data : empty_mat),
             ((properties & kBackpropNeedsOutput) ? output_data : empty_mat),
             output_deriv, NULL, &input_deriv);

  int32 test_dim = 3;
  BaseFloat original_objf = TraceMatMat(output_deriv, output_data, kTrans);
  Vector<BaseFloat> measured_objf_change(test_dim),
      predicted_objf_change(test_dim);
  for (int32 i = 0; i < test_dim; i++) {
    CuMatrix<BaseFloat> perturbed_input_data(num_rows, input_dim),
        perturbed_output_data(num_rows, output_dim);
    perturbed_input_data.SetRandn();
    perturbed_input_data.Scale(perturb_delta);
    // at this point, perturbed_input_data contains the offset at the input data.
    predicted_objf_change(i) = TraceMatMat(perturbed_input_data, input_deriv,
                                           kTrans);
    perturbed_input_data.AddMat(1.0, input_data);
    c.Propagate(NULL, perturbed_input_data, &perturbed_output_data);
    measured_objf_change(i) = TraceMatMat(output_deriv, perturbed_output_data,
                                          kTrans) - original_objf;
  }
  KALDI_LOG << "Predicted objf-change = " << predicted_objf_change;
  KALDI_LOG << "Measured objf-change = " << measured_objf_change;
  BaseFloat threshold = 0.1;
  bool ans = ApproxEqual(predicted_objf_change, measured_objf_change, threshold);
  if (!ans)
    KALDI_WARN << "Data-derivative test failed, component-type="
               << c.Type() << ", input-dim=" << input_dim
               << ", output-dim=" << output_dim;
  if (c.Type() == "NormalizeComponent" && input_dim == 1) {
    // derivatives are mathematically zero, but the measured and predicted
    // objf have different roundoff and the relative differences are large.
    // this is not unexpected.
    KALDI_LOG << "Accepting deriv differences since it is NormalizeComponent "
              << "with dim=1.";
    return true;
  }
  return ans;
}


// if test_derivative == false then the test only tests that the update
// direction is downhill.  if true, then we measure the actual model-derivative
// and check that it's accurate.
// returns true on success, false on test failure.
bool TestSimpleComponentModelDerivative(const Component &c,
                                        BaseFloat perturb_delta,
                                        bool test_derivative) {
  int32 input_dim = c.InputDim(),
      output_dim = c.OutputDim(),
      num_rows = RandInt(1, 100);
  int32 properties = c.Properties();
  if ((properties & kUpdatableComponent) == 0) {
    // nothing to test.
    return true;
  }

  CuMatrix<BaseFloat> input_data(num_rows, input_dim),
      output_data(num_rows, output_dim),
      output_deriv(num_rows, output_dim);
  input_data.SetRandn();
  output_deriv.SetRandn();

  c.Propagate(NULL, input_data, &output_data);

  BaseFloat original_objf = TraceMatMat(output_deriv, output_data, kTrans);

  Component *c_copy = c.Copy();

  const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(&c);
  UpdatableComponent *uc_copy = dynamic_cast<UpdatableComponent*>(c_copy);
  KALDI_ASSERT(uc != NULL && uc_copy != NULL);
  if (test_derivative) {
    bool is_gradient = true;
    uc_copy->SetZero(is_gradient);
  }

  CuMatrix<BaseFloat> input_deriv(num_rows, input_dim), empty_mat;
  c.Backprop("foobar", NULL,
             ((properties & kBackpropNeedsInput) ? input_data : empty_mat),
             ((properties & kBackpropNeedsOutput) ? output_data : empty_mat),
             output_deriv, c_copy,
             (RandInt(0, 1) == 0 ? &input_deriv : NULL));

  if (!test_derivative) { // Just testing that the model update is downhill.
    CuMatrix<BaseFloat> new_output_data(num_rows, output_dim);
    c_copy->Propagate(NULL, input_data, &new_output_data);

    BaseFloat new_objf = TraceMatMat(output_deriv, new_output_data, kTrans);
    bool ans = (new_objf > original_objf);
    if (!ans) {
      KALDI_WARN << "After update, new objf is not better than the original objf: "
                 << new_objf << " <= " << original_objf;
    }
    delete c_copy;
    return ans;
  } else {
    // check that the model derivative is accurate.
    int32 test_dim = 3;

    Vector<BaseFloat> measured_objf_change(test_dim),
        predicted_objf_change(test_dim);
    for (int32 i = 0; i < test_dim; i++) {
      CuMatrix<BaseFloat> perturbed_output_data(num_rows, output_dim);
      Component *c_perturbed = c.Copy();
      UpdatableComponent *uc_perturbed =
          dynamic_cast<UpdatableComponent*>(c_perturbed);
      KALDI_ASSERT(uc_perturbed != NULL);
      uc_perturbed->PerturbParams(perturb_delta);

      predicted_objf_change(i) = uc_copy->DotProduct(*uc_perturbed) -
          uc_copy->DotProduct(*uc);
      c_perturbed->Propagate(NULL, input_data, &perturbed_output_data);
      measured_objf_change(i) = TraceMatMat(output_deriv, perturbed_output_data,
                                            kTrans) - original_objf;
      delete c_perturbed;
    }
    KALDI_LOG << "Predicted objf-change = " << predicted_objf_change;
    KALDI_LOG << "Measured objf-change = " << measured_objf_change;
    BaseFloat threshold = 0.1;

    bool ans = ApproxEqual(predicted_objf_change, measured_objf_change,
                           threshold);
    if (!ans)
      KALDI_WARN << "Model-derivative test failed, component-type="
                 << c.Type() << ", input-dim=" << input_dim
                 << ", output-dim=" << output_dim;
    delete c_copy;
    return ans;
  }
}


void UnitTestNnetComponent() {
  for (int32 n = 0; n < 100; n++)  {
    Component *c = GenerateRandomSimpleComponent();
    KALDI_LOG << c->Info();
    TestNnetComponentIo(c);
    TestNnetComponentCopy(c);
    TestNnetComponentAddScale(c);
    TestSimpleComponentPropagateProperties(*c);
    if (!TestSimpleComponentDataDerivative(*c, 1.0e-04) &&
        !TestSimpleComponentDataDerivative(*c, 1.0e-03) &&
        !TestSimpleComponentDataDerivative(*c, 1.0e-05) &&
        !TestSimpleComponentDataDerivative(*c, 1.0e-06))
      KALDI_ERR << "Component data-derivative test failed";

    if (!TestSimpleComponentModelDerivative(*c, 1.0e-04, false) &&
        !TestSimpleComponentModelDerivative(*c, 1.0e-03, false) &&
        !TestSimpleComponentModelDerivative(*c, 1.0e-06, false))
      KALDI_ERR << "Component downhill-update test failed";

    if (!TestSimpleComponentModelDerivative(*c, 1.0e-04, true) &&
        !TestSimpleComponentModelDerivative(*c, 1.0e-03, true) &&
        !TestSimpleComponentModelDerivative(*c, 1.0e-05, true) &&
        !TestSimpleComponentModelDerivative(*c, 1.0e-06, true))
      KALDI_ERR << "Component model-derivative test failed";

    delete c;
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    UnitTestNnetComponent();
  }

  KALDI_LOG << "Nnet component ntests succeeded.";

  return 0;
}
