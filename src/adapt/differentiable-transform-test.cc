// adapt/differentiable-transform-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

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

#include "adapt/differentiable-transform.h"
#include "matrix/sp-matrix.h"

namespace kaldi {
namespace differentiable_transform {

// This function writes a random configuration file of dimension
// 'dim' (or a random dimension if dim == -1) to 'os'.
void WriteRandomConfigOfDim(std::ostream &os, int32 dim) {
  // nonrandom_dim is a randomly chosen dimension if dim == -1,
  // else it's dim.
  int32 actual_dim = (dim == -1 ? RandInt(10, 20) : dim);
  int32 i, num_transforms = RandInt(1, 3);

  while (true) {
    // we loop here in case we hit a case we don't want to handle.
    // We give more cases to the non-recursive transforms to ensure
    // the expected size of the config file is finite.
    switch(RandInt(0, 7)) {
      case 0:
        os << "NoOpTransform dim=" << actual_dim << "\n";
        return;
      case 1: case 2: case 3:
        os << "FmllrTransform dim=" << actual_dim << " smoothing-count="
           << 100.0 * RandInt(0, 2) << "\n";
        return;
      case 4: case 5:
        os << "MeanOnlyTransform dim=" << actual_dim << "\n";
        return;
      case 6:
        if (dim != -1)  // complicated to ensure a given dim for AppendTransform.
          continue;
        os << "AppendTransform num-transforms=" << num_transforms << "\n";
        for (i = 0; i < num_transforms; i++)
          WriteRandomConfigOfDim(os, -1);
        return;
      case 7:
        os << "SequenceTransform num-transforms=" << num_transforms << "\n";
        for (i = 0; i < num_transforms; i++)
          WriteRandomConfigOfDim(os, actual_dim);
        return;
    }
  }

}

// This function writes a random configuration file to 'os'.
void WriteRandomConfigFile(std::ostream &os) {
  WriteRandomConfigOfDim(os, -1);
}



void UnitTestReadFromConfig() {
  using namespace kaldi;
  using namespace kaldi::differentiable_transform;

  for (int32 i = 0; i < 100; i++) {
    std::ostringstream os;
    WriteRandomConfigFile(os);
    std::istringstream is(os.str());
    int32 num_classes = RandInt(20, 30);
    DifferentiableTransform *transform =
        DifferentiableTransform::ReadFromConfig(is, num_classes);
    KALDI_ASSERT(transform != NULL);
    delete transform;
  }
}

// Creates a random mean per class and adds it to the features, weighted
// according to the posteriors.   It makes the tests more realistic, if
// there are systematic differences between the classes.
void AddRandomMeanOffsets(BaseFloat scale,
                          int32 num_classes,
                          const Posterior &post,
                          CuMatrix<BaseFloat> *feats) {
  int32 T = feats->NumRows(), dim = feats->NumCols();
  CuMatrix<BaseFloat> class_means(num_classes, dim);
  class_means.SetRandn();
  class_means.Scale(scale);
  for (int32 t = 0; t < T; t++) {
    auto iter = post[t].begin(), end = post[t].end();
    BaseFloat tot_post = 0.0;
    for (; iter != end; ++iter)
      tot_post += iter->second;
    for (iter = post[t].begin(); iter != end; ++iter) {
      int32 i = iter->first;
      BaseFloat p = iter->second / tot_post;
      feats->Row(t).AddVec(p, class_means.Row(i));
    }
  }
}

void GetRandomPosterior(int32 num_frames, int32 num_classes,
                        Posterior *post) {
  post->resize(num_frames);
  for (int32 t = 0; t < num_frames; t++) {
    for (int32 i = 0; i < 3; i++) {
      if (RandInt(0, 1) == 0) {
        (*post)[t].push_back(std::pair<int32, BaseFloat>(
            RandInt(0, num_classes - 1), 0.1 + RandUniform()));
      }
    }
  }

}

void TestTraining(DifferentiableTransform *transform) {
  // test that the training process runs.
  int32 dim = transform->Dim(),
      num_classes = transform->NumClasses(),
      num_frames = RandInt(200, 300),
      num_spk = RandInt(2, 10),
      chunks_per_spk = RandInt(1, 4),
      num_rows = num_frames * num_spk * chunks_per_spk;
  CuMatrix<BaseFloat> input_feats(num_rows, dim),
      output_feats(num_rows, dim, kUndefined),
      output_deriv(num_rows, dim, kUndefined),
      input_deriv(num_rows, dim);
  input_feats.SetRandn();
  output_deriv.SetRandn();
  Posterior post;
  GetRandomPosterior(num_rows, num_classes, &post);
  AddRandomMeanOffsets(10.0, num_classes, post, &input_feats);

  int32 num_chunks = num_spk * chunks_per_spk;
  MinibatchInfoItf *info =
      transform->TrainingForward(input_feats, num_chunks, num_spk, post,
                                 &output_feats);
  CuMatrix<BaseFloat> diff(input_feats);
  diff.AddMat(-1.0, output_feats);
  KALDI_LOG << "Difference in features (relative) is "
            << (diff.FrobeniusNorm() / input_feats.FrobeniusNorm());


  transform->TrainingBackward(input_feats, output_deriv, num_chunks,
                              num_spk, post, info, &input_deriv);

  int32 n = 5;
  Vector<BaseFloat> expected_changes(n), observed_changes(n);
  BaseFloat epsilon = 1.0e-03;
  for (int32 i = 0; i < n; i++) {
    CuMatrix<BaseFloat> new_input_feats(num_rows, dim),
        new_output_feats(num_rows, dim, kUndefined);
    new_input_feats.SetRandn();
    new_input_feats.Scale(epsilon);
    expected_changes(i) = TraceMatMat(new_input_feats, input_deriv, kTrans);
    new_input_feats.AddMat(1.0, input_feats);
    MinibatchInfoItf *info2 =
        transform->TrainingForward(new_input_feats, num_chunks, num_spk,
                                   post, &new_output_feats);
    delete info2;
    new_output_feats.AddMat(-1.0, output_feats);
    observed_changes(i) = TraceMatMat(new_output_feats, output_deriv, kTrans);
  }
  KALDI_LOG << "Expected changes: " << expected_changes
            << ", observed changes: " << observed_changes;
  KALDI_ASSERT(expected_changes.ApproxEqual(observed_changes, 0.15));

  {
    // Test that if we do Accumulate() and Estimate() on the same data we
    // trained on, and then TestingForwardBatch(), we get the same answer
    // as during training.  Note: this may not be true for all examples
    // including SequenceTransform, due to how we treat the last of the
    // transforms specially.

    int32 num_final_iters = transform->NumFinalIterations();
    for (int32 i = 0; i < num_final_iters; i++) {
      transform->Accumulate(i, input_feats, num_chunks, num_spk, post);
      transform->Estimate(i);
    }
    CuMatrix<BaseFloat> output_feats2(output_feats.NumRows(),
                                      output_feats.NumCols(), kUndefined);
    transform->TestingForwardBatch(input_feats, num_chunks, num_spk, post,
                                   &output_feats2);
    output_feats2.AddMat(-1.0, output_feats);
    BaseFloat rel_diff = (output_feats2.FrobeniusNorm() /
                          output_feats.FrobeniusNorm());
    KALDI_LOG << "Difference in features train vs. test (relative) is "
              << rel_diff;
    if (rel_diff > 0.001) {
      KALDI_WARN << "Make sure this config would not be equivalent train "
          "vs. test (see config printed above).";
    }
  }
}


void UnitTestTraining() {
  for (int32 i = 0; i < 100; i++) {
    std::ostringstream os;
    WriteRandomConfigFile(os);
    std::istringstream is(os.str());
    int32 num_classes = RandInt(20, 30);
    DifferentiableTransform *transform =
        DifferentiableTransform::ReadFromConfig(is, num_classes);
    KALDI_LOG << "Config is: " << os.str();
    KALDI_ASSERT(transform != NULL);
    if (os.str().find("smoothing-count=0") == std::string::npos) {
      // Don't do this test if smoothing-count is zero: it can
      // fail but it doesn't indicate a real problem.
      TestTraining(transform);
    }
    delete transform;
  }
}


void UnitTestIo() {
  for (int32 i = 0; i < 100; i++) {
    std::ostringstream os;
    WriteRandomConfigFile(os);
    std::istringstream is(os.str());
    int32 num_classes = RandInt(20, 30);
    DifferentiableTransform *transform =
        DifferentiableTransform::ReadFromConfig(is, num_classes);
    KALDI_ASSERT(transform != NULL);

    std::ostringstream os2;
    bool binary = (RandInt(0,1) == 0);
    transform->Write(os2, binary);

    std::istringstream is2(os2.str());

    DifferentiableTransform *transform2 =
        DifferentiableTransform::ReadNew(is2, binary);
    std::ostringstream os3;
    transform2->Write(os3, binary);
    KALDI_ASSERT(os2.str() == os3.str());
    delete transform;
    delete transform2;
  }
}



}  // namespace kaldi
}  // namespace differentiable_transform



int main() {
  using namespace kaldi::differentiable_transform;

  for (int32 i = 0; i < 3; i++) {
    UnitTestReadFromConfig();
    UnitTestIo();
    UnitTestTraining();
  }
  std::cout << "Test OK.\n";
}
