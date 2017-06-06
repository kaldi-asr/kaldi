// nnet3/convolution-test.cc

// Copyright 2017    Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet3/convolution.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet3 {
namespace time_height_convolution {

// for testing purposes, create a random ConvolutionModel.
static void GetRandomConvolutionModel(ConvolutionModel *model) {
start:
  {
    model->num_filters_in = RandInt(1, 10);
    model->num_filters_out = RandInt(1, 10);
    model->height_in = RandInt(1, 10);
    int32 min_height_offset = RandInt(-2, 0),
        max_height_offset = RandInt(0, 2),
        min_time_offset = RandInt(-2, 0),
        max_time_offset = RandInt(0, 2);

    model->height_out = RandInt(1, model->height_in);
    model->height_subsample_out = 1;
    if (RandInt(0, 1) == 0) {
      if (model->height_out % 2 == 0) {
        model->height_out /= 2;
        model->height_subsample_out = 2;
      } else if (model->height_out % 3 == 0) {
        model->height_out /= 3;
        model->height_subsample_out = 3;
      }
    }
    std::vector<int32> all_time_offsets;
    int32 max_offsets = RandInt(1, 10);
    model->offsets.clear();
    model->required_time_offsets.clear();
    for (int32 i = 0; i < max_offsets; i++) {
      ConvolutionModel::Offset o;
      o.time_offset = RandInt(min_time_offset, max_time_offset);
      o.height_offset = RandInt(min_height_offset, max_height_offset);
      all_time_offsets.push_back(o.time_offset);
      model->offsets.push_back(o);
    }
    SortAndUniq(&(model->offsets));
    SortAndUniq(&all_time_offsets);
    std::random_shuffle(all_time_offsets.begin(), all_time_offsets.end());
    int32 num_required_offsets = RandInt(1, all_time_offsets.size());
    for (int32 i = 0; i < num_required_offsets; i++)
      model->required_time_offsets.insert(all_time_offsets[i]);
    model->ComputeDerived();
  }
  if (!model->Check()) {
    KALDI_WARN << "Regenerating model because it didn't pass the check: "
               << model->Info();
    goto start;
  }
}

// for testing purposes, create a set of input and output indexes for
// a convolution computation that are computable given this model.
static void GetRandomConvolutionIndexes(const ConvolutionModel &model,
                                        std::vector<Index> *input_indexes,
                                        std::vector<Index> *output_indexes) {
  KALDI_ASSERT(model.Check());

  std::vector<std::pair<int32, int32> > n_x_pairs;
  int32 num_n_x_pairs = RandInt(1, 3);
  for (int32 i = 0; i < num_n_x_pairs; i++) {
    int32 n = RandInt(0, 3), x = RandInt(0, 1);
    n_x_pairs.push_back(std::pair<int32, int32>(n, x));
  }
  SortAndUniq(&n_x_pairs);
  num_n_x_pairs = n_x_pairs.size();


  // 'output_t_values' is the set of *possible* output
  // t values; we'll later sub-sample from these.
  std::vector<int32> output_t_values;

  {
    int32 out_t_start = RandInt(-5, 5), out_t_step = RandInt(1, 3),
        num_t_out = RandInt(1, 4);
    for (int32 i = 0; i < num_t_out; i++)
      output_t_values.push_back(out_t_start + i * out_t_step);
  }

  input_indexes->clear();
  output_indexes->clear();
  for (size_t i = 0; i < n_x_pairs.size(); i++) {
    std::vector<int32> chosen_output_t_values;
    while (chosen_output_t_values.empty()) {
      for (size_t j = 0; j < output_t_values.size(); j++)
        if (RandInt(0, 1) != 0)
          chosen_output_t_values.push_back(output_t_values[j]);
    }
    KALDI_ASSERT(IsSortedAndUniq(chosen_output_t_values));

    std::set<int32> required_input_t_values,
        usable_input_t_values;
    for (size_t j = 0; j < chosen_output_t_values.size(); j++) {
      std::set<int32>::const_iterator iter;
      int32 t_out = chosen_output_t_values[j];
      for (iter = model.required_time_offsets.begin();
           iter != model.required_time_offsets.end(); iter++) {
        int32 offset = *iter;
        required_input_t_values.insert(t_out + offset);
      }
      for (iter = model.all_time_offsets.begin();
           iter != model.all_time_offsets.end(); iter++) {
        int32 offset = *iter;
        usable_input_t_values.insert(t_out + offset);
      }
    }

    // add to output_indexes
    for (size_t j = 0; j < chosen_output_t_values.size(); j++) {
      int32 t_out = chosen_output_t_values[j];
      Index index;
      index.n = n_x_pairs[i].first;
      index.x = n_x_pairs[i].second;
      index.t = t_out;
      output_indexes->push_back(index);
    }

    std::vector<int32> chosen_input_t_values(required_input_t_values.begin(),
                                             required_input_t_values.end());
    for (std::set<int32>::const_iterator iter = usable_input_t_values.begin();
         iter != usable_input_t_values.end(); ++iter) {
      int32 t = *iter;
      if (RandInt(0, 1) == 0)
        chosen_input_t_values.push_back(t);
    }
    SortAndUniq(&chosen_input_t_values);

    // add to input_indexes
    for (size_t j = 0; j < chosen_input_t_values.size(); j++) {
      int32 t_in = chosen_input_t_values[j];
      Index index;
      index.n = n_x_pairs[i].first;
      index.x = n_x_pairs[i].second;
      index.t = t_in;
      input_indexes->push_back(index);
    }
  }
}


void UnitTestTimeHeightConvolutionIo() {
  for (int32 i = 0; i < 10; i++) {
    KALDI_LOG << "iter = " << i;
    // Create a ConvolutionModel and test its I/O.
    ConvolutionModel conv_model;
    GetRandomConvolutionModel(&conv_model);
    std::ostringstream os1, os2;
    bool binary = (RandInt(0, 1) == 0);
    conv_model.Write(os1, binary);
    std::istringstream is(os1.str());
    ConvolutionModel conv_model2;
    conv_model2.Read(is, binary);
    conv_model2.Write(os2, binary);
    KALDI_ASSERT(os1.str() == os2.str() && conv_model2.Check());
  }
}

void TestComputationIo(const ConvolutionComputation &computation) {
  std::ostringstream os1, os2;
  bool binary = (RandInt(0, 1) == 0);
  computation.Write(os1, binary);
  std::istringstream is(os1.str());
  ConvolutionComputation computation2;
  computation2.Read(is, binary);
  computation2.Write(os2, binary);
  KALDI_ASSERT(os1.str() == os2.str());
  computation2.Check();
}


// This function exects indexes.size() == matrix->NumRows();
// it sets to zero any row i of the matrix for which
// indexes[i].t == kNoTime.
void ZeroBlankRows(const std::vector<Index> &indexes,
                   CuMatrix<BaseFloat> *matrix) {
  KALDI_ASSERT(static_cast<int32>(indexes.size()) == matrix->NumRows());
  int32 num_rows = matrix->NumRows();
  if (num_rows == 0) return;
  Vector<BaseFloat> mask(num_rows, kUndefined);
  mask.Set(1.0);
  const Index *indexes_ptr = &(indexes[0]);
  BaseFloat *mask_ptr = mask.Data();
  for (int32 r = 0; r < num_rows; r++) {
    if (indexes_ptr[r].t == kNoTime)
      mask_ptr[r] = 0.0;
  }
  CuVector<BaseFloat> cu_mask;
  cu_mask.Swap(&mask);
  matrix->MulRowsVec(cu_mask);
}

// This is a 'dumb' implementation of convolution, created to compare
// with ConvolveForward.
void ConvolveForwardSimple(
    const ConvolutionModel &model,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    const CuMatrixBase<BaseFloat> &input_cu,
    const CuMatrixBase<BaseFloat> &params_cu,
    CuMatrixBase<BaseFloat> *output_cu) {
  // these loops will be very slow on GPU, so do it all on CPU.
  Matrix<BaseFloat> input(input_cu), params(params_cu),
      output(*output_cu);
  std::unordered_map<Index, int32, IndexHasher> index_to_row;
  int32 input_rows = input.NumRows(),
      output_rows = output.NumRows();
  for (int32 r_in = 0; r_in < input_rows; r_in++) {
    if (input_indexes[r_in].t != kNoTime) {
      index_to_row[input_indexes[r_in]] = r_in;
    }
  }
  int32 num_offsets = model.offsets.size(),
      num_filters_in = model.num_filters_in,
      num_filters_out = model.num_filters_out,
      height_in = model.height_in,
      height_out = model.height_out,
      height_subsample_out = model.height_subsample_out;
  for (int32 r_out = 0; r_out < output_rows; r_out++) {
    Index index_out = output_indexes[r_out];
    if (index_out.t == kNoTime)
      continue;
    SubVector<BaseFloat> output_row(output, r_out);
    for (int32 o = 0; o < num_offsets; o++) {
      int32 time_offset = model.offsets[o].time_offset,
          height_offset = model.offsets[o].height_offset;
      Index index_in(index_out);
      index_in.t += time_offset;
      std::unordered_map<Index, int32, IndexHasher>::const_iterator iter =
          index_to_row.find(index_in);
      if (iter != index_to_row.end()) {
        SubMatrix<BaseFloat> params_part(params, 0, params.NumRows(),
                                         o * num_filters_in, num_filters_in);
        int32 r_in = iter->second;
        SubVector<BaseFloat> input_row(input, r_in);
        for (int32 h_out_subsampled = 0;
             h_out_subsampled < height_out;
             h_out_subsampled++) {
          int32 h_out = h_out_subsampled * height_subsample_out,
              h_in = h_out + height_offset;
          if (h_in < 0 || h_in >= height_in)
            continue;
          SubVector<BaseFloat> output_part(output_row,
                                           h_out_subsampled * num_filters_out,
                                           num_filters_out),
              input_part(input_row, h_in * num_filters_in, num_filters_in);
          output_part.AddMatVec(1.0, params_part, kNoTrans, input_part, 1.0);
        }
      }
    }
  }
  output_cu->CopyFromMat(output);
}



void TestRunningComputation(const ConvolutionModel &conv_model,
                            const std::vector<Index> &input_indexes,
                            const std::vector<Index> &output_indexes,
                            const ConvolutionComputation &computation) {
  CuMatrix<BaseFloat> input(input_indexes.size(), conv_model.InputDim(),
                            kSetZero, kStrideEqualNumCols),
      output(output_indexes.size(), conv_model.OutputDim(),
             kSetZero, kStrideEqualNumCols),
      output2(output),
      params(conv_model.ParamRows(), conv_model.ParamCols());
  input.SetRandn();
  params.SetRandn();
  ZeroBlankRows(input_indexes, &input);
  ConvolveForward(computation, input, params, &output);
  ZeroBlankRows(output_indexes, &output);

  ConvolveForwardSimple(conv_model, input_indexes, output_indexes,
                        input, params, &output2);
  KALDI_LOG << "Tested convolution for model: "
            << conv_model.Info();
  if (!output.ApproxEqual(output2, 0.001)) {
    KALDI_LOG << "Output is: " << output;
    KALDI_LOG << "Output2 is: " << output2;
    KALDI_ERR << "Convolution test failure.";
  }
}


void TestDataBackprop(const ConvolutionModel &conv_model,
                      const std::vector<Index> &input_indexes,
                      const std::vector<Index> &output_indexes,
                      const ConvolutionComputation &computation) {
  CuMatrix<BaseFloat>
      input_deriv(input_indexes.size(), conv_model.InputDim(),
                  kSetZero, kStrideEqualNumCols),
      input(input_indexes.size(), conv_model.InputDim(),
            kSetZero, kStrideEqualNumCols),
      output(output_indexes.size(), conv_model.OutputDim(),
             kSetZero, kStrideEqualNumCols),
      output_deriv(output_indexes.size(), conv_model.OutputDim(),
                   kSetZero, kStrideEqualNumCols),
      params(conv_model.ParamRows(), conv_model.ParamCols());

  input.SetRandn();
  params.SetRandn();
  output_deriv.SetRandn();

  ZeroBlankRows(output_indexes, &output_deriv);
  ConvolveBackwardData(computation, params, output_deriv, &input_deriv);
  ZeroBlankRows(input_indexes, &input_deriv);
  ZeroBlankRows(input_indexes, &input);

  // define the objf as TraceMatMat(output_deriv, output, kTrans).
  // we can work it out from the backpropagated data-derivative.
  BaseFloat expected_objf = TraceMatMat(input_deriv, input, kTrans);

  ConvolveForward(computation, input, params, &output);
  ZeroBlankRows(output_indexes, &output);

  BaseFloat observed_objf = TraceMatMat(output, output_deriv, kTrans);

  KALDI_LOG << "Expected objf = " << expected_objf
            << ", observed objf = " << observed_objf;
  if (!ApproxEqual(expected_objf, observed_objf, 0.1) &&
      fabs(expected_objf) < 1.0) {
    KALDI_ERR << "Difference in objf too large.";
  }
}


void TestParamsBackprop(const ConvolutionModel &conv_model,
                        const std::vector<Index> &input_indexes,
                        const std::vector<Index> &output_indexes,
                        const ConvolutionComputation &computation) {
  CuMatrix<BaseFloat>
      input(input_indexes.size(), conv_model.InputDim(),
            kSetZero, kStrideEqualNumCols),
      output(output_indexes.size(), conv_model.OutputDim(),
             kSetZero, kStrideEqualNumCols),
      output_deriv(output_indexes.size(), conv_model.OutputDim(),
                   kSetZero, kStrideEqualNumCols),
      params(conv_model.ParamRows(), conv_model.ParamCols()),
      params_deriv(conv_model.ParamRows(), conv_model.ParamCols());

  input.SetRandn();
  params.SetRandn();
  output_deriv.SetRandn();

  BaseFloat alpha = 0.5 * RandInt(1, 3);

  ZeroBlankRows(output_indexes, &output_deriv);
  ZeroBlankRows(input_indexes, &input);

  ConvolveBackwardParams(computation, input, output_deriv, alpha,
                         &params_deriv);

  BaseFloat expected_objf = TraceMatMat(params_deriv, params, kTrans) / alpha;

  ConvolveForward(computation, input, params, &output);

  ZeroBlankRows(output_indexes, &output);

  BaseFloat observed_objf = TraceMatMat(output, output_deriv, kTrans);

  KALDI_LOG << "Expected objf = " << expected_objf
            << ", observed objf = " << observed_objf;
  if (!ApproxEqual(expected_objf, observed_objf, 0.1) &&
      fabs(expected_objf) < 1.0) {
    KALDI_ERR << "Difference in objf too large.";
  }
}



void UnitTestTimeHeightConvolutionCompile() {
  for (int32 i = 0; i < 10; i++) {
    KALDI_LOG << "iter = " << i;
    // Create a ConvolutionModel
    ConvolutionModel conv_model;
    GetRandomConvolutionModel(&conv_model);
    std::vector<Index> input_indexes, output_indexes;
    GetRandomConvolutionIndexes(conv_model, &input_indexes, &output_indexes);

    ConvolutionComputationOptions opts;
    ConvolutionComputation computation;
    std::vector<Index> input_indexes_modified, output_indexes_modified;
    CompileConvolutionComputation(conv_model, input_indexes, output_indexes,
                                  opts, &computation,
                                  &input_indexes_modified,
                                  &output_indexes_modified);
    TestComputationIo(computation);
    TestRunningComputation(conv_model,
                           input_indexes_modified,
                           output_indexes_modified,
                           computation);
    TestDataBackprop(conv_model,
                     input_indexes_modified,
                     output_indexes_modified,
                     computation);
    TestParamsBackprop(conv_model,
                       input_indexes_modified,
                       output_indexes_modified,
                       computation);
    std::ostringstream os;
    os << "\nInput-indexes: ";
    WriteIndexVector(os, false, input_indexes);
    os << "\nInput-indexes-modified: ";
    WriteIndexVector(os, false, input_indexes_modified);
    os << "\nOutput-indexes: ";
    WriteIndexVector(os, false, output_indexes);
    os << "\nOutput-indexes-modified: ";
    WriteIndexVector(os, false, output_indexes_modified);
    KALDI_LOG << os.str();
  }
}


void UnitTestTimeHeightConvolution() {
  UnitTestTimeHeightConvolutionIo();
  UnitTestTimeHeightConvolutionCompile();
}



} // namespace time_height_convolution
} // namespace nnet3
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  using namespace kaldi::nnet3::time_height_convolution;
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
    for (int32 i = 0; i < 5; i++) {
      UnitTestTimeHeightConvolution();
    }
  }
}
