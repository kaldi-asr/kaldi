// nnet3/convolution.cc

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/convolution.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-compile-utils.h"

namespace kaldi {
namespace nnet3 {
namespace time_height_convolution {


/**
   This function, used in ConvolutionComputation::ComputeDerived(),
   reverses a mapping that may not be unique.  'columns' is a column
   mapping where each member is either -1 (meaning, copy a zero), or
   a number between 0 and input_dim - 1.

   Its output, 'backward_columns', is the reverse mapping, but it's a vector of
   vectors instead of just a vector because the mapping may have been
   many-to-one.  Each element of 'backward_columns' will be of dimension
   input_dim.  For each columns[i] = j such that j != -1,
   for some k we will have (*backward_columns)[k][j] = i.
*/
static void ReverseColumnMapping(
    const std::vector<int32> &columns,
    int32 input_dim,
    std::vector<std::vector<int32> > *backward_columns) {
  int32 columns_dim = columns.size();
  std::vector<std::vector<int32> > temp(input_dim);
  for (int32 i = 0; i < columns_dim; i++) {
    int32 j = columns[i];
    KALDI_ASSERT(j >= -1 && j < input_dim);
    if (j != -1)
      temp[j].push_back(i);
  }
  // 'max_overlap' is the largest number of times that some j >= 0 appears in
  // 'columns'.
  int32 max_overlap = 0;
  for (int32 j = 0; j < input_dim; j++)
    max_overlap = std::max(max_overlap,
                           static_cast<int32>(temp[j].size()));
  backward_columns->resize(max_overlap);
  for (int32 k = 0; k < max_overlap; k++) {
    (*backward_columns)[k].clear();
    (*backward_columns)[k].resize(input_dim, -1);
  }
  for (int32 j = 0; j < input_dim; j++) {
    for (int32 k = 0; k < static_cast<int32>(temp[j].size()); k++) {
      int32 i = temp[j][k];
      (*backward_columns)[k][j] = i;
    }
  }
}

// returns true if 'vec' is of the form
// [ n, n+1, n+2, .... ].
static bool VectorIsContiguous(const std::vector<int32> &vec) {
  KALDI_ASSERT(!vec.empty());
  int32 s = vec.size();
  for (int32 i = 0; i + 1 < s; i++)
    if (vec[i+1] != vec[i] + 1)
      return false;
  return true;
}


std::string ConvolutionModel::Info() const {
  std::ostringstream os;
  os << "num-filters-in=" << num_filters_in
     << ", num-filters-out=" << num_filters_out
     << ", height-in=" << height_in
     << ", height-out=" << height_out
     << ", height-subsample-out=" << height_subsample_out
     << ", {time,height}-offsets=[";
  for (size_t i = 0; i < offsets.size(); i++) {
    if (i > 0) os << ' ';
    os << offsets[i].time_offset << ',' << offsets[i].height_offset;
  }
  os << "], required-time-offsets=[";
  for (std::set<int32>::const_iterator iter = required_time_offsets.begin();
       iter != required_time_offsets.end(); ++iter) {
    if (iter != required_time_offsets.begin()) os << ',';
    os << *iter;
  }
  os << "], input-dim=" << InputDim() << ", output-dim=" << OutputDim();
  return os.str();
}

void ConvolutionModel::ComputeDerived() {
  { // compute all_time_offsets
    all_time_offsets.clear();
    for (std::vector<Offset>::const_iterator iter = offsets.begin();
         iter != offsets.end(); ++iter)
      all_time_offsets.insert(iter->time_offset);
  }
  { // compute time_offsets_modulus
    time_offsets_modulus = 0;
    std::set<int32>::iterator iter = all_time_offsets.begin();
    int32 cur_offset = *iter;
    for (++iter; iter != all_time_offsets.end(); ++iter) {
      int32 this_offset = *iter;
      time_offsets_modulus = Gcd(time_offsets_modulus,
                                 this_offset - cur_offset);
      cur_offset = this_offset;
    }
  }
}


bool ConvolutionModel::Check(bool check_heights_used,
                             bool allow_height_padding) const {
  if (num_filters_in <= 0 || num_filters_out <= 0 ||
      height_in <= 0 || height_out <= 0 ||
      height_subsample_out <=  0  || offsets.empty() ||
      required_time_offsets.empty()) {
    KALDI_WARN << "Convolution model fails basic check.";
    return false;
  }
  ConvolutionModel temp(*this);
  temp.ComputeDerived();
  if (!(temp == *this)) {
    KALDI_WARN << "Derived variables are incorrect.";
    return false;
  }
  // check that required_time_offsets is included in all_time_offsets.
  for (std::set<int32>::iterator iter = required_time_offsets.begin();
       iter != required_time_offsets.end(); ++iter) {
    if (all_time_offsets.count(*iter) == 0) {
      KALDI_WARN << "Required time offsets not a subset of all_time_offsets.";
      return false;
    }
  }
  KALDI_ASSERT(IsSortedAndUniq(offsets));
  std::vector<bool> h_in_used(height_in, false);
  std::vector<bool> offsets_used(offsets.size(), false);

  // check that in cases where we only have the minimum
  // required input (from required_time_offsets), each
  // height in the output is potentially nonzero.
  for (int32 h_out = 0; h_out < height_out * height_subsample_out;
       h_out += height_subsample_out) {
    bool some_input_available = false;
    for (size_t i = 0; i < offsets.size(); i++) {
      const Offset &offset = offsets[i];
      int32 h_in = h_out + offset.height_offset;
      if (h_in >= 0 && h_in < height_in) {
        offsets_used[i] = true;
        h_in_used[h_in] = true;
        if (required_time_offsets.count(offset.time_offset) != 0)
          some_input_available = true;
      } else {
        if (!allow_height_padding) {
          KALDI_WARN << "height padding not allowed but is required.";
          return false;
        }
      }
    }
    if (!some_input_available) {
      // none of the
      // input pixels for this output pixel were available (at least in the case
      // where we only have the 'required' inputs on the time dimension).
      std::ostringstream os;
      Write(os, false);
      KALDI_WARN << "for the " << (h_out / height_out) << "'th output height, "
                 "no input is available, if only required time-indexes "
                 "are available.";
      // We could later change this part of the validation code to accept
      // such models, if there is a legitimate use-case.
      return false;
    }
  }
  if (check_heights_used) {
    for (int32 h = 0; h < height_in; h++) {
      if (!h_in_used[h]) {
        KALDI_WARN << "The input at the " << h << "'th height is never used.";
        return false;
      }
    }
  }
  for (size_t i = 0; i < offsets_used.size(); i++) {
    if (!offsets_used[i]) {
      KALDI_WARN << "(time,height) offset (" << offsets[i].time_offset
                 << "," << offsets[i].height_offset
                 << ") of this computation is never used.";
      return false;
    }
  }
  return true;
}


bool ConvolutionModel::operator == (const ConvolutionModel &other) const {
  return num_filters_in == other.num_filters_in &&
      num_filters_out == other.num_filters_out &&
      height_in == other.height_in &&
      height_out == other.height_out &&
      height_subsample_out == other.height_subsample_out &&
      offsets == other.offsets &&
      required_time_offsets == other.required_time_offsets &&
      all_time_offsets == other.all_time_offsets &&
      time_offsets_modulus == other.time_offsets_modulus;
}


void ConvolutionModel::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ConvolutionModel>");
  WriteToken(os, binary, "<NumFiltersIn>");
  WriteBasicType(os, binary, num_filters_in);
  WriteToken(os, binary, "<NumFiltersOut>");
  WriteBasicType(os, binary, num_filters_out);
  WriteToken(os, binary, "<HeightIn>");
  WriteBasicType(os, binary, height_in);
  WriteToken(os, binary, "<HeightOut>");
  WriteBasicType(os, binary, height_out);
  WriteToken(os, binary, "<HeightSubsampleOut>");
  WriteBasicType(os, binary, height_subsample_out);
  WriteToken(os, binary, "<Offsets>");
  std::vector<std::pair<int32, int32> > pairs(offsets.size());
  for (size_t i = 0; i < offsets.size(); i++) {
    pairs[i].first = offsets[i].time_offset;
    pairs[i].second = offsets[i].height_offset;
  }
  WriteIntegerPairVector(os, binary, pairs);
  std::vector<int32> required_time_offsets_list(required_time_offsets.begin(),
                                                required_time_offsets.end());
  WriteToken(os, binary, "<RequiredTimeOffsets>");
  WriteIntegerVector(os, binary, required_time_offsets_list);
  WriteToken(os, binary, "</ConvolutionModel>");
}


void ConvolutionModel::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<ConvolutionModel>", "<NumFiltersIn>");
  ReadBasicType(is, binary, &num_filters_in);
  ExpectToken(is, binary, "<NumFiltersOut>");
  ReadBasicType(is, binary, &num_filters_out);
  ExpectToken(is, binary, "<HeightIn>");
  ReadBasicType(is, binary, &height_in);
  ExpectToken(is, binary, "<HeightOut>");
  ReadBasicType(is, binary, &height_out);
  ExpectToken(is, binary, "<HeightSubsampleOut>");
  ReadBasicType(is, binary, &height_subsample_out);
  ExpectToken(is, binary, "<Offsets>");
  std::vector<std::pair<int32, int32> > pairs;
  ReadIntegerPairVector(is, binary, &pairs);
  offsets.resize(pairs.size());
  for (size_t i = 0; i < offsets.size(); i++) {
    offsets[i].time_offset = pairs[i].first;
    offsets[i].height_offset = pairs[i].second;
  }
  std::vector<int32> required_time_offsets_list;
  ExpectToken(is, binary, "<RequiredTimeOffsets>");
  ReadIntegerVector(is, binary, &required_time_offsets_list);
  required_time_offsets.clear();
  required_time_offsets.insert(required_time_offsets_list.begin(),
                               required_time_offsets_list.end());
  ExpectToken(is, binary, "</ConvolutionModel>");
  ComputeDerived();
  KALDI_ASSERT(Check(false, true));
}


void ConvolutionComputation::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ConvComputation>");
  WriteToken(os, binary, "<NumFiltersInOut>");
  WriteBasicType(os, binary, num_filters_in);
  WriteBasicType(os, binary, num_filters_out);
  WriteToken(os, binary, "<HeightInOut>");
  WriteBasicType(os, binary, height_in);
  WriteBasicType(os, binary, height_out);
  WriteToken(os, binary, "<NumTInOut>");
  WriteBasicType(os, binary, num_t_in);
  WriteBasicType(os, binary, num_t_out);
  WriteToken(os, binary, "<NumImages>");
  WriteBasicType(os, binary, num_images);
  WriteToken(os, binary, "<TempRowsCols>");
  WriteBasicType(os, binary, temp_rows);
  WriteBasicType(os, binary, temp_cols);
  int32 num_steps = steps.size();
  WriteToken(os, binary, "<NumSteps>");
  WriteBasicType(os, binary, num_steps);
  for (int32 s = 0; s < num_steps; s++) {
    const ConvolutionStep &step = steps[s];
    WriteToken(os, binary, "<TimeShift>");
    WriteBasicType(os, binary, step.input_time_shift);
    WriteToken(os, binary, "<ParamsStartCol>");
    WriteBasicType(os, binary, step.params_start_col);
    WriteToken(os, binary, "<HeightMap>");
    WriteIntegerVector(os, binary, step.height_map);
  }
  WriteToken(os, binary, "</ConvComputation>");
}


void ConvolutionComputation::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<ConvComputation>", "<NumFiltersInOut>");
  ReadBasicType(is, binary, &num_filters_in);
  ReadBasicType(is, binary, &num_filters_out);
  ExpectToken(is, binary, "<HeightInOut>");
  ReadBasicType(is, binary, &height_in);
  ReadBasicType(is, binary, &height_out);
  ExpectToken(is, binary, "<NumTInOut>");
  ReadBasicType(is, binary, &num_t_in);
  ReadBasicType(is, binary, &num_t_out);
  ExpectToken(is, binary, "<NumImages>");
  ReadBasicType(is, binary, &num_images);
  ExpectToken(is, binary, "<TempRowsCols>");
  ReadBasicType(is, binary, &temp_rows);
  ReadBasicType(is, binary, &temp_cols);
  int32 num_steps;
  ExpectToken(is, binary, "<NumSteps>");
  ReadBasicType(is, binary, &num_steps);
  steps.resize(num_steps);
  for (int32 s = 0; s < num_steps; s++) {
    ConvolutionStep &step = steps[s];
    ExpectToken(is, binary, "<TimeShift>");
    ReadBasicType(is, binary, &step.input_time_shift);
    ExpectToken(is, binary, "<ParamsStartCol>");
    ReadBasicType(is, binary, &step.params_start_col);
    ExpectToken(is, binary, "<HeightMap>");
    ReadIntegerVector(is, binary, &step.height_map);
  }
  ExpectToken(is, binary, "</ConvComputation>");
  ComputeDerived();
  Check();
}


void ConvolutionComputation::Check() const {
  KALDI_ASSERT(num_filters_in > 0 && num_filters_out > 0 &&
               height_in > 0 && height_out > 0);
  KALDI_ASSERT(num_t_in >= num_t_out &&
               num_t_out > 0 && num_images > 0);
  KALDI_ASSERT((temp_rows == 0 && temp_cols == 0) ||
               (temp_rows <= num_t_out * num_images &&
                temp_cols > 0));
  KALDI_ASSERT(temp_rows % num_images == 0);
  bool temp_mat_required = false;
  int32 num_steps = steps.size();
  int32 num_extra_input_times = num_t_in - num_t_out,
      input_cols = num_filters_in * height_in,
      smallest_time_shift = 1000,
      largest_time_shift = 0;
  // check 'steps'
  for (int32 s = 0; s < num_steps; s++) {
    const ConvolutionStep &step = steps[s];
    KALDI_ASSERT(step.input_time_shift >= 0 &&
                 step.input_time_shift <= num_extra_input_times);
    if (step.input_time_shift < smallest_time_shift)
      smallest_time_shift = step.input_time_shift;
    if (step.input_time_shift > largest_time_shift)
      largest_time_shift = step.input_time_shift;
    KALDI_ASSERT(step.params_start_col >= 0 &&
                 step.params_start_col % num_filters_in == 0);
    if (s != 0) {
      KALDI_ASSERT(step.input_time_shift != steps[s-1].input_time_shift);
    }
    std::vector<int32> columns;
    step.columns.CopyToVec(&columns);
    KALDI_ASSERT(step.first_column == columns[0]);
    KALDI_ASSERT(step.columns.Dim() == step.height_map.size() * num_filters_in);
    bool all_negative = true;
    int32 temp_height = step.height_map.size();
    bool contiguous = true;
    for (int32 i = 0; i < temp_height; i++) {
      int32 h = step.height_map[i];
      KALDI_ASSERT(h >= -1 && h < height_in);
      if (i > 0 && step.height_map[i-1] != h-1)
        contiguous = false;
      if (h == -1) {
        contiguous = false;
        for (int32 f = 0; f < num_filters_in; f++) {
          KALDI_ASSERT(columns[i * num_filters_in + f] == -1);
        }
      } else {
        all_negative = false;
        for (int32 f = 0; f < num_filters_in; f++) {
          KALDI_ASSERT(columns[i * num_filters_in + f] ==
                       h * num_filters_in + f);
        }
      }
    }
    KALDI_ASSERT(contiguous == step.columns_are_contiguous);
    if (!contiguous || columns.size() != input_cols) {
      // we would need the temporary matrix.  Make sure the
      // temporary matrix is big enough.
      temp_mat_required = true;
      KALDI_ASSERT(columns.size() <= temp_cols);
    }
    KALDI_ASSERT(!all_negative);

    std::vector<int32> columns_reconstructed(columns.size(), -1);
    // reconstruct 'columns' from backward_columns as a way to
    // check that backward_columns is correct.
    // they are reverse-direction maps, but we may need
    // step.backward_columns.size() > 1 because of elements
    // in the input that are duplicated in the temp matrix.
    for (size_t k = 0; k < step.backward_columns.size(); k++) {
      std::vector<int32> backward_columns;
      step.backward_columns[k].CopyToVec(&backward_columns);
      KALDI_ASSERT(int32(backward_columns.size()) ==
                   num_filters_in * height_in);
      for (int32 l = 0; l < num_filters_in * height_in; l++) {
        int32 c = backward_columns[l];
        KALDI_ASSERT(c < int32(columns.size()));
        if (c != -1) {
          KALDI_ASSERT(columns_reconstructed[c] == -1);
          columns_reconstructed[c] = l;
        }
      }
    }
    KALDI_ASSERT(columns_reconstructed == columns);
  }
  // check that all rows of the input were used.
  KALDI_ASSERT(smallest_time_shift == 0 &&
               largest_time_shift == num_extra_input_times);

  // check that the temp matrix is only allocated if it is required.
  KALDI_ASSERT((temp_cols != 0) == temp_mat_required);
}


// Internal function called inside ConvolveForward.
// Note: the number of time steps covered may be different
// from that implied by cc.num_t_in and cc.num_t_out
// if the matrices are very large and we've broken the
// computation up into pieces to save memoiry.
static void ConvolveForwardInternal(
    const ConvolutionComputation &cc,
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &params,
    CuMatrixBase<BaseFloat> *temp_mat,
    CuMatrixBase<BaseFloat> *output) {
  KALDI_ASSERT(temp_mat->Stride() == temp_mat->NumCols());

  // num_t_out supersedes cc.num_t_out (they'll only be different in
  // cases where we are doing the computation in pieces to save memory).
  int32 input_rows = input.NumRows(),
      output_rows = output->NumRows();

  KALDI_ASSERT(output_rows <= input_rows &&
               input_rows % cc.num_images == 0 &&
               output_rows % cc.num_images == 0);

  int32 num_steps = cc.steps.size();
  for (int32 s = 0; s < num_steps; s++) {
    const ConvolutionComputation::ConvolutionStep &step = cc.steps[s];
    int32 input_row_start = step.input_time_shift * cc.num_images;
    // note: 'input_part' will normally be almost all of 'input', perhaps
    // minus one or two time steps at the start or end.
    CuSubMatrix<BaseFloat> input_part(input,
                                      input_row_start, output_rows,
                                      0, input.NumCols());
    int32 temp_num_cols = step.columns.Dim(),
        param_cols = temp_num_cols / cc.height_out;
    CuSubMatrix<BaseFloat> params_part(params,
                                       0, params.NumRows(),
                                       step.params_start_col,
                                       param_cols);
    CuSubMatrix<BaseFloat> output_reshaped(
        output->Data(), output_rows * cc.height_out,
        cc.num_filters_out, cc.num_filters_out);
    if (!step.columns_are_contiguous ||
        temp_num_cols != input.NumCols()) {
      // In most cases we will take this branch, where we have to copy the input
      // to a temporary matrix.  (however, different steps may require different
      // num-cols of the temporary matrix, so we create sub-parts of 'temp_mat'.

      // We create the sub-matrix 'temp_mat_part' in a lower-level way, using
      // pointers, because we need to ensure that its num-cols and the stride
      // are the same (this is necessary so that we can do reshaping in
      // ConvolutionReshapedMultiply()).
      CuSubMatrix<BaseFloat> temp_mat_part(temp_mat->Data(),
                                           temp_mat->NumRows(),
                                           temp_num_cols, temp_num_cols);
      if (!step.columns_are_contiguous) {
        // we're doing a column mapping.
        temp_mat_part.CopyCols(input_part, step.columns);
      } else {
        // we're just taking a sub-matrix of the input matrix, but we still need
        // to make a copy because we need the stride == num-cols (so that the
        // reshaping will work).
        temp_mat_part.CopyFromMat(input_part.ColRange(step.first_column,
                                                      step.columns.Dim()));
      }
      CuSubMatrix<BaseFloat> temp_mat_part_reshaped(
          temp_mat_part.Data(), temp_mat_part.NumRows() * cc.height_out,
          temp_num_cols / cc.height_out, temp_num_cols / cc.height_out);

      output_reshaped.AddMatMat(1.0, temp_mat_part_reshaped, kNoTrans,
                                params_part, kTrans, 1.0);
    } else {
      CuSubMatrix<BaseFloat> input_reshaped(
          input_part.Data(), input_part.NumRows() * cc.height_out,
          input_part.NumCols() / cc.height_out,
          input_part.NumCols() / cc.height_out);

      output_reshaped.AddMatMat(1.0, input_reshaped, kNoTrans,
                                params_part, kTrans, 1.0);
    }
  }
}

void ConvolveForward(
    const ConvolutionComputation &cc,
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &params,
    CuMatrixBase<BaseFloat> *output) {
  KALDI_ASSERT(input.NumCols() == input.Stride() &&
               output->NumCols() == output->Stride());
  KALDI_ASSERT(params.NumRows() == cc.num_filters_out);
  KALDI_ASSERT(output->NumRows() == cc.num_t_out * cc.num_images &&
               output->NumCols() == cc.height_out * cc.num_filters_out);
  // the input might need to be reshaped but we can check its total size.
  KALDI_ASSERT(input.NumRows() * input.NumCols() == cc.num_images *
               cc.num_t_in * cc.height_in * cc.num_filters_in);

  int32 input_rows = input.NumRows(),
      required_input_rows = cc.num_images * cc.num_t_in;

  // this if-statement handles reshaping the input and recursing if there
  // is subsampling.
  if (input_rows != required_input_rows) {
    if (input_rows % required_input_rows != 0)
      KALDI_ERR << "Input matrix has wrong size.";  // error in calling code.
    // nr is a multiple of required_nr.  Reshape the matrix.
    // we already checked that its Stride() == NumCols();
    int32 num_cols = input.NumCols(),
        multiple = input_rows / required_input_rows,
        new_num_cols = num_cols * multiple,
        new_stride = new_num_cols;
    CuSubMatrix<BaseFloat> input_reshaped(
        input.Data(), required_input_rows, new_num_cols, new_stride);
    ConvolveForward(cc, input_reshaped, params, output);
    return;
  }

  CuMatrix<BaseFloat> temp_mat(cc.temp_rows, cc.temp_cols,
                               kUndefined, kStrideEqualNumCols);

  // this if-statement handles breaking up the arguments
  // and the computation into row-ranges if the temporary
  // matrix would have been excessively large, and we've decided
  // to give it fewer rows than the output (this saves
  // memory).  normally we won't take this if-statement
  // so ignore it if you're trying to understand the framework.
  if (cc.temp_rows != 0 && cc.temp_rows != input_rows) {
    KALDI_ASSERT(cc.temp_rows % cc.num_images == 0);
    int32 num_time_steps_per_chunk = cc.temp_rows / cc.num_images;
    int32 num_extra_in = cc.num_t_in - cc.num_t_out;

    for (int32 t_start = 0; t_start < cc.num_t_out;
         t_start += num_time_steps_per_chunk) {
      int32 num_t_left = cc.num_t_out - t_start,
          this_num_t_out = std::min<int32>(num_t_left,
                                           num_time_steps_per_chunk),
          this_num_t_in = this_num_t_out + num_extra_in;
      CuSubMatrix<BaseFloat> input_part(input, t_start * cc.num_images,
                                        this_num_t_in * cc.num_images,
                                        0, input.NumCols());
      CuSubMatrix<BaseFloat> output_part(*output, t_start * cc.num_images,
                                         this_num_t_out * cc.num_images,
                                         0, output->NumCols());
      CuSubMatrix<BaseFloat> temp_part(temp_mat, 0,
                                       this_num_t_out * cc.num_images,
                                       0, temp_mat.NumCols());
      ConvolveForwardInternal(cc, input_part, params,
                              &temp_part, &output_part);
    }
    return;
  }
  ConvolveForwardInternal(cc, input, params, &temp_mat, output);
}


// Internal function called inside ConvolveBackwardData.
// Note: the number of time steps covered may be different
// from that implied by cc.num_t_in and cc.num_t_out
// if the matrices are very large and we've broken the
// computation up into pieces to save memory.
// We require that temp_mat should not contain inf's
// or nan's on entry.
static void ConvolveBackwardDataInternal(
    const ConvolutionComputation &cc,
    const CuMatrixBase<BaseFloat> &params,
    const CuMatrixBase<BaseFloat> &output_deriv,
    CuMatrixBase<BaseFloat> *temp_mat,
    CuMatrixBase<BaseFloat> *input_deriv) {
  KALDI_ASSERT(temp_mat->Stride() == temp_mat->NumCols());

  // num_t_out supersedes cc.num_t_out (they'll only be different in
  // cases where we are doing the computation in pieces to save memory).
  int32 input_rows = input_deriv->NumRows(),
      output_rows = output_deriv.NumRows();

  KALDI_ASSERT(output_rows <= input_rows &&
               input_rows % cc.num_images == 0 &&
               output_rows % cc.num_images == 0);

  int32 num_steps = cc.steps.size();
  for (int32 s = 0; s < num_steps; s++) {
    const ConvolutionComputation::ConvolutionStep &step = cc.steps[s];
    int32 input_row_start = step.input_time_shift * cc.num_images;
    CuSubMatrix<BaseFloat> input_deriv_part(*input_deriv,
                                            input_row_start, output_rows,
                                            0, input_deriv->NumCols());
    int32 temp_num_cols = step.columns.Dim(),
        param_cols = temp_num_cols / cc.height_out;
    CuSubMatrix<BaseFloat> params_part(params,
                                       0, params.NumRows(),
                                       step.params_start_col,
                                       param_cols);
    CuSubMatrix<BaseFloat> output_deriv_reshaped(
        output_deriv.Data(), output_rows * cc.height_out,
        cc.num_filters_out, cc.num_filters_out);

    if (!step.columns_are_contiguous ||
        temp_num_cols != input_deriv->NumCols()) {
      // In most cases we will take this branch, where we have to propagate the
      // input-derivative via a temporary matrix.  (however, different steps may
      // require different num-cols of the temporary matrix, so we create
      // sub-parts of 'temp_mat'.

      // We create the sub-matrix 'temp_mat_part' in a lower-level way, using
      // pointers, because we need to ensure that its num-cols and the stride
      // are the same (this is necessary so that we can do reshaping in
      // ConvolutionReshapedMultiply()).
      CuSubMatrix<BaseFloat> temp_mat_part(temp_mat->Data(),
                                           temp_mat->NumRows(),
                                           temp_num_cols, temp_num_cols),
          temp_mat_part_reshaped(
              temp_mat_part.Data(), temp_mat_part.NumRows() * cc.height_out,
              temp_num_cols / cc.height_out, temp_num_cols / cc.height_out);

      temp_mat_part_reshaped.AddMatMat(1.0, output_deriv_reshaped, kNoTrans,
                                       params_part, kNoTrans, 0.0);

      if (!step.columns_are_contiguous) {
        for (size_t i = 0; i < step.backward_columns.size(); i++) {
          input_deriv_part.AddCols(temp_mat_part, step.backward_columns[i]);
        }
      } else {
        // we're just taking a sub-matrix of the input matrix, but we still need
        // to make a copy because we need the stride == num-cols (so that the
        // reshaping will work).
        int32 num_cols = step.columns.Dim();
        input_deriv_part.ColRange(step.first_column,
                                  num_cols).AddMat(1.0, temp_mat_part);
      }
    } else {
      CuSubMatrix<BaseFloat> input_deriv_reshaped(
          input_deriv_part.Data(), input_deriv_part.NumRows() * cc.height_out,
          input_deriv_part.NumCols() / cc.height_out,
          input_deriv_part.NumCols() / cc.height_out);
      input_deriv_reshaped.AddMatMat(1.0, output_deriv_reshaped, kNoTrans,
                                     params_part, kNoTrans, 1.0);
    }
  }
}


void ConvolveBackwardData(
    const ConvolutionComputation &cc,
    const CuMatrixBase<BaseFloat> &params,
    const CuMatrixBase<BaseFloat> &output_deriv,
    CuMatrixBase<BaseFloat> *input_deriv) {
  KALDI_ASSERT(input_deriv->NumCols() == input_deriv->Stride() &&
               output_deriv.NumCols() == output_deriv.Stride());
  KALDI_ASSERT(params.NumRows() == cc.num_filters_out);
  KALDI_ASSERT(output_deriv.NumRows() == cc.num_t_out * cc.num_images &&
               output_deriv.NumCols() == cc.height_out * cc.num_filters_out);
  // the input might need to be reshaped but we can check its total size.
  KALDI_ASSERT(input_deriv->NumRows() * input_deriv->NumCols() ==
               cc.num_images * cc.num_t_in * cc.height_in * cc.num_filters_in);

  int32 input_rows = input_deriv->NumRows(),
      required_input_rows = cc.num_images * cc.num_t_in;

  // this if-statement handles reshaping the input and recursing if there
  // is subsampling.
  if (input_rows != required_input_rows) {
    if (input_rows % required_input_rows != 0)
      KALDI_ERR << "Input matrix has wrong size.";  // error in calling code.
    // nr is a multiple of required_nr.  Reshape the matrix.
    // we already checked that its Stride() == NumCols();
    int32 num_cols = input_deriv->NumCols(),
        multiple = input_rows / required_input_rows,
        new_num_cols = num_cols * multiple,
        new_stride = new_num_cols;
    CuSubMatrix<BaseFloat> input_deriv_reshaped(
        input_deriv->Data(), required_input_rows,
        new_num_cols, new_stride);
    ConvolveBackwardData(cc, params, output_deriv, &input_deriv_reshaped);
    return;
  }

  CuMatrix<BaseFloat> temp_mat(cc.temp_rows, cc.temp_cols,
                               kSetZero, kStrideEqualNumCols);

  // this if-statement handles breaking up the arguments
  // and the computation into row-ranges if the temporary
  // matrix would have been excessively large, and we've decided
  // to give it fewer rows than the output (this saves
  // memory).  normally we won't take this if-statement
  // so ignore it if you're trying to understand the framework.
  if (cc.temp_rows != 0 && cc.temp_rows != input_rows) {
    KALDI_ASSERT(cc.temp_rows % cc.num_images == 0);
    int32 num_time_steps_per_chunk = cc.temp_rows / cc.num_images;
    int32 num_extra_in = cc.num_t_in - cc.num_t_out;

    for (int32 t_start = 0; t_start < cc.num_t_out;
         t_start += num_time_steps_per_chunk) {
      int32 num_t_left = cc.num_t_out - t_start,
          this_num_t_out = std::min<int32>(num_t_left,
                                           num_time_steps_per_chunk),
          this_num_t_in = this_num_t_out + num_extra_in;
      CuSubMatrix<BaseFloat> input_deriv_part(
          *input_deriv, t_start * cc.num_images,
          this_num_t_in * cc.num_images,
          0, input_deriv->NumCols());
      CuSubMatrix<BaseFloat> output_deriv_part(
          output_deriv, t_start * cc.num_images,
          this_num_t_out * cc.num_images,
          0, output_deriv.NumCols());
      CuSubMatrix<BaseFloat> temp_part(
          temp_mat, 0, this_num_t_out * cc.num_images,
          0, temp_mat.NumCols());
      ConvolveBackwardDataInternal(cc, params, output_deriv_part,
                                   &temp_part, &input_deriv_part);
    }
    return;
  }
  ConvolveBackwardDataInternal(cc, params, output_deriv,
                               &temp_mat, input_deriv);
}


// Internal function called inside ConvolveBackwardParams.
// Note: the number of time steps covered may be different
// from that implied by cc.num_t_in and cc.num_t_out
// if the matrices are very large and we've broken the
// computation up into pieces to save memoiry.
static void ConvolveBackwardParamsInternal(
    const ConvolutionComputation &cc,
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &output_deriv,
    BaseFloat alpha,
    CuMatrixBase<BaseFloat> *temp_mat,
    CuMatrixBase<BaseFloat> *params_deriv) {
  KALDI_ASSERT(temp_mat->Stride() == temp_mat->NumCols());

  // num_t_out supersedes cc.num_t_out (they'll only be different in
  // cases where we are doing the computation in pieces to save memory).
  int32 input_rows = input.NumRows(),
      output_rows = output_deriv.NumRows();

  KALDI_ASSERT(output_rows <= input_rows &&
               input_rows % cc.num_images == 0 &&
               output_rows % cc.num_images == 0);

  int32 num_steps = cc.steps.size();
  for (int32 s = 0; s < num_steps; s++) {
    const ConvolutionComputation::ConvolutionStep &step = cc.steps[s];
    int32 input_row_start = step.input_time_shift * cc.num_images;
    // note: 'input_part' will normally be almost all of 'input', perhaps
    // minus one or two time steps at the start or end.
    CuSubMatrix<BaseFloat> input_part(input,
                                      input_row_start, output_rows,
                                      0, input.NumCols());
    int32 temp_num_cols = step.columns.Dim(),
        param_cols = temp_num_cols / cc.height_out;
    CuSubMatrix<BaseFloat> params_deriv_part(*params_deriv,
                                       0, params_deriv->NumRows(),
                                       step.params_start_col,
                                       param_cols);
    CuSubMatrix<BaseFloat> output_deriv_reshaped(
        output_deriv.Data(), output_rows * cc.height_out,
        cc.num_filters_out, cc.num_filters_out);
    if (!step.columns_are_contiguous ||
        temp_num_cols != input.NumCols()) {
      // In most cases we will take this branch, where we have to copy the input
      // to a temporary matrix.  (however, different steps may require different
      // num-cols of the temporary matrix, so we create sub-parts of 'temp_mat'.

      // We create the sub-matrix 'temp_mat_part' in a lower-level way, using
      // pointers, because we need to ensure that its num-cols and the stride
      // are the same (this is necessary so that we can do reshaping in
      // ConvolutionReshapedMultiply()).
      CuSubMatrix<BaseFloat> temp_mat_part(temp_mat->Data(),
                                           temp_mat->NumRows(),
                                           temp_num_cols, temp_num_cols);
      if (!step.columns_are_contiguous) {
        // we're doing a column mapping.
        temp_mat_part.CopyCols(input_part, step.columns);
      } else {
        // we're just taking a sub-matrix of the input matrix, but we still need
        // to make a copy because we need the stride == num-cols (so that the
        // reshaping will work).
        temp_mat_part.CopyFromMat(input_part.ColRange(step.first_column,
                                                      step.columns.Dim()));
      }
      CuSubMatrix<BaseFloat> temp_mat_part_reshaped(
          temp_mat_part.Data(), temp_mat_part.NumRows() * cc.height_out,
          temp_num_cols / cc.height_out, temp_num_cols / cc.height_out);

      params_deriv_part.AddMatMat(alpha, output_deriv_reshaped, kTrans,
                                  temp_mat_part_reshaped, kNoTrans, 1.0);
    } else {
      CuSubMatrix<BaseFloat> input_reshaped(
          input_part.Data(), input_part.NumRows() * cc.height_out,
          input_part.NumCols() / cc.height_out,
          input_part.NumCols() / cc.height_out);

      params_deriv_part.AddMatMat(alpha, output_deriv_reshaped, kTrans,
                                  input_reshaped, kNoTrans, 1.0);
    }
  }
}

void ConvolveBackwardParams(
    const ConvolutionComputation &cc,
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &output_deriv,
    BaseFloat alpha,
    CuMatrixBase<BaseFloat> *params_deriv) {
  KALDI_ASSERT(input.NumCols() == input.Stride() &&
              output_deriv.NumCols() == output_deriv.Stride());
  KALDI_ASSERT(params_deriv->NumRows() == cc.num_filters_out);
  KALDI_ASSERT(output_deriv.NumRows() == cc.num_t_out * cc.num_images &&
               output_deriv.NumCols() == cc.height_out * cc.num_filters_out);
  // the input might need to be reshaped but we can check its total size.
  KALDI_ASSERT(input.NumRows() * input.NumCols() == cc.num_images *
               cc.num_t_in * cc.height_in * cc.num_filters_in);

  int32 input_rows = input.NumRows(),
      required_input_rows = cc.num_images * cc.num_t_in;

  // this if-statement handles reshaping the input and recursing if there
  // is subsampling.
  if (input_rows != required_input_rows) {
    if (input_rows % required_input_rows != 0)
      KALDI_ERR << "Input matrix has wrong size.";  // error in calling code.
    // nr is a multiple of required_nr.  Reshape the matrix.
    // we already checked that its Stride() == NumCols();
    int32 num_cols = input.NumCols(),
        multiple = input_rows / required_input_rows,
        new_num_cols = num_cols * multiple,
        new_stride = new_num_cols;
    CuSubMatrix<BaseFloat> input_reshaped(
        input.Data(), required_input_rows, new_num_cols, new_stride);
    ConvolveBackwardParams(cc, input_reshaped, output_deriv, alpha,
                           params_deriv);
    return;
  }

  CuMatrix<BaseFloat> temp_mat(cc.temp_rows, cc.temp_cols,
                               kUndefined, kStrideEqualNumCols);

  // this if-statement handles breaking up the arguments
  // and the computation into row-ranges if the temporary
  // matrix would have been excessively large, and we've decided
  // to give it fewer rows than the output (this saves
  // memory).  normally we won't take this if-statement
  // so ignore it if you're trying to understand the framework.
  if (cc.temp_rows != 0 && cc.temp_rows != input_rows) {
    KALDI_ASSERT(cc.temp_rows % cc.num_images == 0);
    int32 num_time_steps_per_chunk = cc.temp_rows / cc.num_images;
    int32 num_extra_in = cc.num_t_in - cc.num_t_out;

    for (int32 t_start = 0; t_start < cc.num_t_out;
         t_start += num_time_steps_per_chunk) {
      int32 num_t_left = cc.num_t_out - t_start,
          this_num_t_out = std::min<int32>(num_t_left,
                                           num_time_steps_per_chunk),
          this_num_t_in = this_num_t_out + num_extra_in;
      CuSubMatrix<BaseFloat> input_part(
          input, t_start * cc.num_images,
          this_num_t_in * cc.num_images,
          0, input.NumCols());
      CuSubMatrix<BaseFloat> output_deriv_part(
          output_deriv, t_start * cc.num_images,
          this_num_t_out * cc.num_images,
          0, output_deriv.NumCols());
      CuSubMatrix<BaseFloat> temp_part(temp_mat,
                                       0, this_num_t_out * cc.num_images,
                                       0, temp_mat.NumCols());
      ConvolveBackwardParamsInternal(cc, input_part, output_deriv_part,
                                     alpha, &temp_part, params_deriv);
    }
    return;
  }
  ConvolveBackwardParamsInternal(cc, input, output_deriv,
                                 alpha, &temp_mat, params_deriv);
}



void PadModelHeight(const ConvolutionModel &model,
                    ConvolutionModel *model_padded) {
  *model_padded = model;
  KALDI_ASSERT(!model.offsets.empty());
  int32 min_height_offset = model.offsets[0].height_offset,
      max_height_offset = model.offsets[0].height_offset,
      num_offsets = model.offsets.size();
  for (int32 i = 1; i < num_offsets; i++) {
    min_height_offset = std::min<int32>(min_height_offset,
                                        model.offsets[i].height_offset);
    max_height_offset = std::max<int32>(max_height_offset,
                                        model.offsets[i].height_offset);
  }
  int32 max_output_height = model.height_subsample_out * (model.height_out - 1),
      max_required_input = max_height_offset + max_output_height,
      min_required_input = min_height_offset + 0;
  int32 bottom_padding = -min_required_input,
      top_padding = max_required_input - (model.height_in - 1);
  if (bottom_padding < 0)
    bottom_padding = 0;
  if (top_padding < 0)
    top_padding = 0;
  model_padded->height_in += bottom_padding + top_padding;
  for (int32 i = 0; i < num_offsets; i++)
    model_padded->offsets[i].height_offset += bottom_padding;

  // The reason why we say 'allow_height_padding = false' below is obvious--
  // we've 'manually' padded by changing the model, so this modified model
  // should not require height padding.  The reason we set 'check_heights_used'
  // is a little more non-obvious.  The very lowest and hightest heights
  // should always be used, but there may, in unusual models, be other heights
  // that are not used.  We found this in random testing.
  KALDI_ASSERT(model_padded->Check(false, false));
}


/** This function sets 'temp_rows' and 'temp_cols' in 'computation'.
 */
static void ComputeTempMatrixSize(const ConvolutionComputationOptions &opts,
                                  ConvolutionComputation *computation) {
  int32 temp_rows = 0, temp_cols = 0;
  for (size_t i = 0; i < computation->steps.size(); i++) {
    const ConvolutionComputation::ConvolutionStep &step = computation->steps[i];
    int32 height_map_size = step.height_map.size(),
        this_num_cols = height_map_size * computation->num_filters_in;
    bool columns_are_contiguous =
        (step.height_map[0] != -1 && VectorIsContiguous(step.height_map));
    bool need_temp_matrix = true;
    if (columns_are_contiguous && step.height_map[0] == 0 &&
        this_num_cols == computation->num_filters_in * computation->height_in) {
      // the only situation in which we wouldn't need the temporary matrix
      // for this step, is where the columns are all of the input matrix.
      need_temp_matrix = false;
    }
    if (need_temp_matrix && this_num_cols > temp_cols)
      temp_cols = this_num_cols;
  }
  if (temp_cols > 0) {
    // work out how many rows the temporary matrix should have, taking
    // into account the specified memory limit.
    temp_rows = computation->num_t_out * computation->num_images;
    BaseFloat num_megabytes = (4 * (temp_rows / 1000.0) * (temp_cols / 1000.0)),
        megabyte_limit = opts.max_memory_mb;
    // C++ rounds down; here, we want to round up so we add one.
    int32 ratio = 1.0 + num_megabytes / megabyte_limit;

    // divide the number of time steps into 'ratio' pieces that are as equal as
    // possible; round up when dividing, to make sure that new_temp_rows * ratio
    // >= temp_rows so that we don't have a small leftover piece.
    int32 new_num_t_out = (computation->num_t_out + ratio - 1) / ratio;
    temp_rows = new_num_t_out * computation->num_images;
    BaseFloat new_num_megabytes = (4 * (temp_rows / 1000.0) * (temp_cols / 1000.0));
    // make sure we're within the memory limit.
    if (new_num_megabytes > 1.01 * megabyte_limit) {
      KALDI_WARN << "Memory consumed in convolution is more than requested "
                 << "(maybe very long time sequence?)";
    }
  }
  computation->temp_rows = temp_rows;
  computation->temp_cols = temp_cols;

}

void UnPadModelHeight(const ConvolutionComputationOptions &opts,
                      const ConvolutionModel &model,
                      const ConvolutionModel &model_padded,
                      ConvolutionComputation *computation) {
  // First work out how much padding was done in PadModelHeight().
  int32 bottom_padding = (model_padded.offsets[0].height_offset -
                          model.offsets[0].height_offset),
      total_padding = model_padded.height_in - model.height_in,
      top_padding = total_padding - bottom_padding;

  int32 old_computation_height_in = computation->height_in;
  // The computation may have been built for the input appended over
  // several frames. Check that it is for an input height that's a multiple of
  // the model input height.
  KALDI_ASSERT(old_computation_height_in % model_padded.height_in == 0 &&
               computation->height_out == model.height_out);

  // 'ratio' is the same ratio from AppendInputFrames(), it's the number
  // of input frames in 'model' and 'model_padded' that get appended
  // to form a single frame in the computation.
  int32 num_steps = computation->steps.size(),
      unpadded_input_height = model.height_in,
      padded_input_height = model_padded.height_in,
      ratio = old_computation_height_in / padded_input_height;

  computation->height_in = ratio * unpadded_input_height;
  for (int32 s = 0; s < num_steps; s++) {
    ConvolutionComputation::ConvolutionStep &step = computation->steps[s];
    int32 height_map_size = step.height_map.size();
    for (int32 i = 0; i < height_map_size; i++) {
      int32 c = step.height_map[i];
      KALDI_ASSERT(c >= 0);  // there should be no -1's in the padded computation.
      // below, h is the actual height in terms of the padded computation, and m
      // is an index that goes from zero to (num-appended-frames - 1).
      int32 h = c % padded_input_height,
          m = c / padded_input_height;
      KALDI_ASSERT(m < ratio);
      if (h < bottom_padding || h >= padded_input_height - top_padding) {
        step.height_map[i] = -1;
      } else {
        step.height_map[i] = (h - bottom_padding) + m * unpadded_input_height;
      }
    }
  }
  ComputeTempMatrixSize(opts, computation);
  computation->ComputeDerived();
  computation->Check();
}


void PadComputationInputTime(const ConvolutionModel &model,
                             ConvolutionComputationIo *io) {
  if (model.time_offsets_modulus == 0) {
    // this can only happen if model->all_time_offsets.size() == 1,
    // and no padding could be required here. W return to avoid
    // special cases below in Gcd().
    return;
  }
  int32 min_time_offset = *model.all_time_offsets.begin(),
      max_time_offset = *model.all_time_offsets.rbegin();

  // it makes everything much simpler if we just enforce that the stride of the
  // input divides model.time_offsets_modulus and also the output stride.
  // (enforcing this may make the input stride smaller).  This may in certain
  // very odd cases cause us to require more inputs [actually 'blanks'] than
  // we really need, but it avoids a lot of careful thought.
  int32 old_t_step_in = io->t_step_in;
  io->t_step_in = Gcd(io->t_step_in, model.time_offsets_modulus);
  if (io->t_step_out != 0)
    io->t_step_in = Gcd(io->t_step_in, io->t_step_out);

  // to ensure that we cover all the original input points, now that
  // we changed the stride we may need to increase num_t_in.
  io->num_t_in = 1 + (old_t_step_in * (io->num_t_in - 1)) / io->t_step_in;

  // by 'desired' we mean usable as an input, not necessarily
  // required in the sense of 'required_time_offsets'.
  int32 first_desired_input_t = io->start_t_out + min_time_offset;
  if (first_desired_input_t < io->start_t_in) {
    KALDI_ASSERT((io->start_t_in - first_desired_input_t) %
                 io->t_step_in == 0);
    io->num_t_in += (io->start_t_in - first_desired_input_t) / io->t_step_in;
    io->start_t_in = first_desired_input_t;
  }

  int32 last_desired_input_t =
      io->start_t_out + (io->num_t_out - 1) * io->t_step_out + max_time_offset,
      last_input_t = io->start_t_in + (io->num_t_in - 1) * io->t_step_in;
  // if the following assert fails, it means we had provided more input than was
  // needed, which is not expected.  This could cause problems later, in
  // AppendInputFrames().
  KALDI_ASSERT(last_desired_input_t >= last_input_t);
  if (last_desired_input_t > last_input_t) {
    KALDI_ASSERT((last_desired_input_t - last_input_t) %
                 io->t_step_in == 0);
    io->num_t_in += (last_desired_input_t - last_input_t) / io->t_step_in;
  }
}

// returns i rounded down to a multiple of n,
// e.g. RoundDownToMultipleOf(3, 2) = 2,
//      RoundDownToMultipleOf(-1, 3) = -3
static int32 RoundDownToMultipleOf(int32 i, int32 n) {
  return n * DivideRoundingDown(i, n);
}


// shifts all time-offsets in the model (in 'offsets[*].time_offset',
// 'required_time_offsets', 'all_time_offsets') by adding 'shift' to them.
static void ShiftAllTimeOffsets(int32 shift,
                                ConvolutionModel *model) {
  { // shift 'offsets'.
    std::vector<ConvolutionModel::Offset>::iterator
        iter = model->offsets.begin(),
        end = model->offsets.end();
    for (; iter != end; ++iter)
      iter->time_offset += shift;
  }
  std::set<int32> temp;
  std::set<int32>::const_iterator iter;
  for (iter = model->required_time_offsets.begin();
       iter != model->required_time_offsets.end(); ++iter)
    temp.insert(*iter + shift);
  model->required_time_offsets.swap(temp);
  temp.clear();
  for (iter = model->all_time_offsets.begin();
       iter != model->all_time_offsets.end(); ++iter)
    temp.insert(*iter + shift);
  model->all_time_offsets.swap(temp);
}


/*
  \brief This function has been broken out of 'AppendInputFrames()' for clarity.  It
      deals with appending input frames together, in cases where the input stride
      is smaller than the output stride.

  \param [in,out] io  The input object representing the I/O of the convolution.
                     It may be modified slightly by this function, in two respects.
                     Firstly, if we are going to be reshaping the input into
                     an input with fewer frames of larger dimension, we need to
                     make sure the number of frames in the input of 'io' is a
                     multiple of the relevant ratio, so we pad with zeros.
                     Also, we may modify the stride of 'io' in cases where there
                     is exactly one frame.  This is for convenience of implementation
                     and does not affect the frames represented.
  \param [out] io_appended  The output object representing the I/O of the
                    possibly-frame-appended computation.  This may be the same
                    as I/O, but it won't be if the input stride is smaller than
                    the output stride-- in that case we need to append the frames.
                    Note: at exit, 'io' and 'io_appended' will really represent
                    two different 'views' of the same data, via a reshaping.

  \return  Returns the integer ratio >= 1 between the num-cols of the
             'appended' features and the original features; this also
             equals the number of frames we append together
*/
static int32 PrepareIoForAppending(ConvolutionComputationIo *io,
                                   ConvolutionComputationIo *io_appended) {
  // first make sure that the output has nonzero stride (it would only have zero
  // stride if there was only one output time index, which is unusual).  if
  // there's only one output time index we can set the stride to whatever we
  // want without affecting the list of output indexes.
  int32 ratio;
  if (io->t_step_out == 0) {
    KALDI_ASSERT(io->num_t_out == 1);
    io->t_step_out = io->t_step_in;
  }
  if (io->t_step_out == io->t_step_in) {
    // there is nothing to do; the output and input strides are the same.
    *io_appended = *io;
    ratio = 1;
    return ratio;
  }
  // Now, we ensured in PadComputationInputTime that if the output stride is
  // nonzero, then the input stride must divide the output stride; and if the
  // output stride was zero then we would have set it to the input stride just
  // above; and if both were zero we would have returned above.  So we can just
  // assert that the input stride divides the output stride.
  KALDI_ASSERT(io->t_step_out % io->t_step_in == 0);
  ratio = io->t_step_out / io->t_step_in;
  // ratio says how many input indexes we have for each output index,
  // ignoring end effects.  It is the number of input indexes we will
  // append together and 'pretend'

  // record this ratio in the 'input' I/O object, which we are also
  // modifying to record the extra required padding.
  io->reorder_t_in = ratio;
  if (io->num_t_in % ratio != 0) {
    // Round up the number of input frames to the nearest multiple (via
    // zero-padding) so we get an whole number of appended input frames.
    io->num_t_in += ratio - (io->num_t_in % ratio);
  }

  // OK, from this point we create the output io object.
  *io_appended = *io;
  io_appended->reorder_t_in = 1;
  io_appended->t_step_in = io->t_step_out;
  io_appended->num_t_in /= ratio;
  return ratio;
}

void AppendInputFrames(const ConvolutionModel &model,
                       ConvolutionComputationIo *io,
                       ConvolutionModel *model_appended,
                       ConvolutionComputationIo *io_appended) {
  int32 ratio = PrepareIoForAppending(io, io_appended);

  if (ratio == 1) {
    // we are not doing any appending of frames.
    *model_appended = model;
    return;
  }

  // we also need the time-step of the output (which is also now the
  // time-step of the appended input).
  // We know that the time step is not zero, because in that case we would
  // have ratio == 1 and would have returned above.
  int32 time_step_out = io_appended->t_step_out;
  KALDI_ASSERT(time_step_out == io_appended->t_step_in && time_step_out != 0);
  int32 orig_time_step_in = io->t_step_in;
  KALDI_ASSERT(orig_time_step_in * ratio == time_step_out);

  // make sure the difference between first input and output frames is what we
  // expect, else something could go wrong here.
  int32 first_time_offset = *(model.all_time_offsets.begin());
  KALDI_ASSERT(io->start_t_in - io->start_t_out == first_time_offset);

  ConvolutionModel model_temp(model);
  // shift so that the first time offset is zero.  this makes
  // the model conversion easier.
  ShiftAllTimeOffsets(-first_time_offset, &model_temp);

  model_appended->num_filters_in = model.num_filters_in;
  model_appended->num_filters_out = model.num_filters_out;
  model_appended->height_in = ratio * model.height_in;
  model_appended->height_out = model.height_out;
  model_appended->height_subsample_out = model.height_subsample_out;
  int32 num_offsets = model_temp.offsets.size(),
      old_height = model.height_in;
  model_appended->offsets.resize(num_offsets);
  model_appended->all_time_offsets.clear();
  for (int32 i = 0; i < num_offsets; i++) {
    const ConvolutionModel::Offset &old_offset = model_temp.offsets[i];
    ConvolutionModel::Offset &new_offset = model_appended->offsets[i];
    // The following two lines are important!!  They are the core of how
    // we handle subsampling in this framework.
    new_offset.time_offset = RoundDownToMultipleOf(old_offset.time_offset,
                                                   time_step_out);
    KALDI_ASSERT((old_offset.time_offset - new_offset.time_offset) %
                 orig_time_step_in == 0);
    int32 row_offset = (old_offset.time_offset - new_offset.time_offset) /
        orig_time_step_in;
    new_offset.height_offset = old_offset.height_offset +
        row_offset * old_height;
    model_appended->all_time_offsets.insert(new_offset.time_offset);
  }

  // Because the 'appended' model will always be used after zero-padding on the
  // time axis, we can just pretend that all desired time-offsets are required.
  // It's a kind of free error-checking.
  model_appended->required_time_offsets = model_appended->all_time_offsets;

  // Undo the time-shifting that we did before.
  ShiftAllTimeOffsets(first_time_offset, model_appended);

  model_appended->ComputeDerived();
  KALDI_ASSERT(model_appended->Check(false, false));
}

void ConvolutionComputation::ComputeDerived() {
  KALDI_ASSERT(!steps.empty());

  int32 input_dim = height_in * num_filters_in;

  int32 largest_required_temp_cols = 0;
  for (std::vector<ConvolutionStep>::iterator iter = steps.begin();
       iter != steps.end(); ++iter) {
    ConvolutionStep &step = *iter;
    std::vector<int32> columns;
    int32 temp_height = step.height_map.size();
    columns.resize(temp_height * num_filters_in);
    for (int32 h = 0; h < temp_height; h++) {
      KALDI_ASSERT(step.height_map[h] >= -1 && step.height_map[h] < height_in);
      if (step.height_map[h] != -1) {
        for (int32 f = 0; f < num_filters_in; f++)
          columns[h * num_filters_in + f] = step.height_map[h] * num_filters_in + f;
      } else {
        for (int32 f = 0; f < num_filters_in; f++)
          columns[h * num_filters_in + f] = -1;
      }
    }
    step.columns.CopyFromVec(columns);
    std::vector<std::vector<int32> > backward_columns;
    ReverseColumnMapping(columns, input_dim, &backward_columns);
    step.backward_columns.resize(backward_columns.size());
    for (size_t i = 0; i < backward_columns.size(); i++)
      step.backward_columns[i].CopyFromVec(backward_columns[i]);

    // we could replace height_map with columns in the line below and get the
    // same answer, but it would be a little slower.
    step.columns_are_contiguous =
        (step.height_map[0] != -1 && VectorIsContiguous(step.height_map));
    step.first_column = columns[0];


    bool need_temp_matrix =
        !(step.columns_are_contiguous && step.height_map[0] == 0 &&
          step.height_map.size() == height_in);
    if (need_temp_matrix) {
      largest_required_temp_cols = std::max<int32>(
          largest_required_temp_cols, static_cast<int32>(columns.size()));
    }
  }
  KALDI_ASSERT(temp_cols == largest_required_temp_cols);
}


// returns true if the time value 't' is one of the
// time values available on the input of 'io.
static bool TimeValueInInput(const ConvolutionComputationIo &io,
                             int32 t) {
  int32 t_step_in = std::max<int32>(1, io.t_step_in);
  return (t >= io.start_t_in &&
          t < io.start_t_in + (t_step_in * io.num_t_in) &&
          (t - io.start_t_in) % t_step_in == 0);
}

void CheckModelAndIo(const ConvolutionModel &model,
                     const ConvolutionComputationIo &io,
                     bool allow_extra_input) {
  KALDI_ASSERT(io.num_t_in > 0 && io.num_t_out > 0 &&
               !model.required_time_offsets.empty() &&
               !model.all_time_offsets.empty());
  if (!allow_extra_input) {
    KALDI_ASSERT(io.start_t_in >= io.start_t_out +
                 *model.all_time_offsets.begin());
    int32 last_t_in = io.start_t_in + io.t_step_in * (io.num_t_in - 1),
        last_t_out = io.start_t_out + io.t_step_out * (io.num_t_out - 1);
    KALDI_ASSERT(last_t_in <= last_t_out +
                 *model.all_time_offsets.rbegin());
  }

  std::set<int32> input_times_to_check;
  for (int32 n = 0; n < std::min(5, io.num_t_out); n++) {
    int32 t_out = io.start_t_out +
        RandInt(0, io.num_t_out - 1) * io.t_step_out;
    for (std::set<int32>::const_iterator iter =
             model.required_time_offsets.begin();
         iter != model.required_time_offsets.end();
         ++iter) {
      int32 offset = *iter;
      input_times_to_check.insert(t_out + offset);
    }
  }
  for (std::set<int32>::const_iterator iter = input_times_to_check.begin();
       iter != input_times_to_check.end(); ++iter) {
    int32 t = *iter;
    if (!TimeValueInInput(io, t)) {
      KALDI_ERR << "Error checking model and IO: time " << t
                << " is required but not in the input.";
    }
  }
}


void CompileConvolutionComputation(
    const ConvolutionModel &model,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    const ConvolutionComputationOptions &opts,
    ConvolutionComputation *computation,
    std::vector<Index> *input_indexes_modified,
    std::vector<Index> *output_indexes_modified) {

  // stage zero [preparing the input and output in a regular grid.]
  ConvolutionComputationIo io;
  GetComputationIo(input_indexes, output_indexes, &io);

  CheckModelAndIo(model, io, false);

  // stage 1.
  PadComputationInputTime(model, &io);

  CheckModelAndIo(model, io, false);

  // stage 2.
  ConvolutionModel model_padded;
  PadModelHeight(model, &model_padded);

  CheckModelAndIo(model_padded, io, false);

  // stage 3.
  ConvolutionModel model_appended;
  ConvolutionComputationIo io_appended;
  // make a 'fake' model and io for possibly-appended input frames.  'io' is
  // non-const because we may need to pad with a few extra frames.
  AppendInputFrames(model_padded, &io,
                    &model_appended, &io_appended);

  CheckModelAndIo(model_appended, io_appended, true);

  // stage 4.
  MakeComputation(model_appended, io_appended, opts, computation);

  // 'reverse' of stage 2.  [stage 3 kind of does its own
  // 'reverse' by modifying its input IO object.]
  // The computation is still specified for the appended input,
  // but the execution code can figure that out itself.
  UnPadModelHeight(opts, model, model_padded, computation);

  GetIndexesForComputation(io, input_indexes, output_indexes,
                           input_indexes_modified, output_indexes_modified);
}


// Returns the greatest common divisor of the differences between the values in
// 'vec', or zero if the vector has zero or one element.  It is an error if
// 'vec' has repeated elements (which could cause a crash in 'Gcd').
static int32 FindGcdOfDifferences(std::vector<int32> &vec) {
  size_t size = vec.size();
  int32 ans = 0;
  for (size_t i = 0; i + 1 < size; i++) {
    int32 diff = vec[i+1] - vec[i];
    // diff should not be zero.
    ans = Gcd(ans, diff);
  }
  return ans;
}

static void RegularizeTList(std::vector<int32> &t_values,
                            int32 *start,
                            int32 *step,
                            int32 *num_values) {
  KALDI_ASSERT(!t_values.empty() && IsSortedAndUniq(t_values));
  *start = t_values[0];
  *step = FindGcdOfDifferences(t_values);
  if (*step == 0) {
    KALDI_ASSERT(t_values.size() == 1);
    *num_values = 1;
  } else {
    int32 last_value = t_values.back();
    *num_values = 1 + (last_value - *start) / *step;
    KALDI_ASSERT((last_value - *start) % *step == 0);
  }
}



/**
   Creates a vector of indexes with a regular structure, according to these
   specifications.

   'n_x_pairs' is the list of (n,x) pairs to include; they will appear
   in this order.

   't_start', 't_step' and 'num_t_values' define the set of 't' values
   to include (note: t_step >= 0; they will appear in the natural order).

   If reorder_t == 1 (the normal case), then the order is simple: 't' has
   the higher stride, then (n, x).  So we'll output first all (n, x) pairs for
   t_start, then all pairs for t_start + t_step, and so on.

   If instead reorder_t > 1, then the order is a little different [note:
   we expect that num_t_values % reorder_t == 0).  Consider, for
   example, reorder_t == 2.  In that case the first block has the first
   two t values, the second block has the next two t values, and so on.
   And within each block, the 't' values have the smallest stride (of 1).
 */
static void CreateIndexes(const std::vector<std::pair<int32, int32> > &n_x_pairs,
                          int32 t_start, int32 t_step, int32 num_t_values,
                          int32 reorder_t, std::vector<Index> *indexes) {
  KALDI_ASSERT(reorder_t >= 1 && num_t_values % reorder_t == 0 && t_step >= 0);
  if (t_step == 0) {
    KALDI_ASSERT(num_t_values == 1);
    t_step = 1;
  }
  int32 num_n_x_pairs = n_x_pairs.size();
  indexes->clear();
  indexes->reserve(num_n_x_pairs * num_t_values);
  int32 outer_t_step = t_step * reorder_t,
      t_end = t_start + (num_t_values * t_step);
  Index index;
  for (int32 t_block = t_start; t_block < t_end; t_block += outer_t_step) {
    for (int32 nx = 0; nx < num_n_x_pairs; nx++) {
      index.n = n_x_pairs[nx].first;
      index.x = n_x_pairs[nx].second;
      for (int32 t = t_block; t < t_block + outer_t_step; t += t_step) {
        index.t = t;
        indexes->push_back(index);
      }
    }
  }
  // we can remove the next assert after a while.
  KALDI_ASSERT(indexes->size() == num_n_x_pairs * num_t_values);
}

/**
   This function modifies 'indexes' by, for any Indexes which was not present in
   'ref_indexes', setting the 't' value to kNoTime.  This will cause the nnet3
   framework to ignore such Indexes for certain purposes, it supresses certain
   error conditions that would otherwise happen from inserting unnecessary
   indexes into the input and output.
 */
static void SetSomeIndexesBlank(const std::vector<Index> &ref_indexes,
                                std::vector<Index> *indexes) {
  std::unordered_set<Index, IndexHasher> ref_set;
  for (std::vector<Index>::const_iterator iter = ref_indexes.begin();
       iter != ref_indexes.end(); ++iter)
    ref_set.insert(*iter);

  for (std::vector<Index>::iterator iter = indexes->begin();
       iter != indexes->end(); ++iter) {
    if (ref_set.count(*iter) == 0)
      iter->t = kNoTime;
  }
}

void GetComputationIo(
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    ConvolutionComputationIo *io) {
  std::vector<std::pair<int32, int32> > n_x_pairs;
  GetNxList(input_indexes, &n_x_pairs);
  KALDI_ASSERT(!n_x_pairs.empty());
  io->num_images = n_x_pairs.size();
  if (GetVerboseLevel() >= 3) {  // a debugging step.
    std::vector<std::pair<int32, int32> > n_x_pairs_2;
    GetNxList(output_indexes, &n_x_pairs_2);
    KALDI_ASSERT(n_x_pairs_2 == n_x_pairs);
  }
  std::vector<int32> t_values;
  GetTList(input_indexes, &t_values);
  RegularizeTList(t_values, &(io->start_t_in),
                  &(io->t_step_in), &(io->num_t_in));
  GetTList(output_indexes, &t_values);
  RegularizeTList(t_values, &(io->start_t_out),
                  &(io->t_step_out), &(io->num_t_out));
  io->reorder_t_in = 1;
}


void GetIndexesForComputation(
    const ConvolutionComputationIo &io,
    const std::vector<Index> &orig_input_indexes,
    const std::vector<Index> &orig_output_indexes,
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) {
  std::unordered_set<Index, IndexHasher> input_set, output_set;
  for (std::vector<Index>::const_iterator iter = orig_input_indexes.begin();
       iter != orig_input_indexes.end(); ++iter)
    input_set.insert(*iter);
  for (std::vector<Index>::const_iterator iter = orig_output_indexes.begin();
       iter != orig_output_indexes.end(); ++iter)
    output_set.insert(*iter);
  std::vector<std::pair<int32, int32> > n_x_pairs;
  GetNxList(orig_input_indexes, &n_x_pairs);
  KALDI_ASSERT(n_x_pairs.size() == io.num_images);
  CreateIndexes(n_x_pairs, io.start_t_in, io.t_step_in, io.num_t_in,
                io.reorder_t_in, input_indexes);
  SetSomeIndexesBlank(orig_input_indexes, input_indexes);
  CreateIndexes(n_x_pairs, io.start_t_out, io.t_step_out, io.num_t_out,
                1, output_indexes);
  SetSomeIndexesBlank(orig_output_indexes, output_indexes);
}


void MakeComputation(const ConvolutionModel &model,
                     ConvolutionComputationIo &io,
                     const ConvolutionComputationOptions &opts,
                     ConvolutionComputation *computation) {
  KALDI_ASSERT(io.t_step_in == io.t_step_out);
  computation->num_filters_in = model.num_filters_in;
  computation->num_filters_out = model.num_filters_out;
  computation->height_in = model.height_in;
  computation->height_out = model.height_out;
  computation->num_t_in = io.num_t_in;
  computation->num_t_out = io.num_t_out;
  computation->num_images = io.num_images;
  KALDI_ASSERT(io.reorder_t_in == 1);
  // first work out the steps of the computation, then
  // work out the dim of the temp matrix

  KALDI_ASSERT(IsSortedAndUniq(model.offsets));
  // Each distinct value of 'time_offset' in model.offsets
  // becomes one step of the computation.

  // if io.t_step_in was zero, use 1 (so divisions and the like will work as
  // expected).
  int32 t_step = std::max<int32>(1, io.t_step_in),
      num_t_extra = io.num_t_in - io.num_t_out;

  computation->steps.clear();

  int32 num_offsets = model.offsets.size(),
      cur_start_offset = 0, cur_end_offset = 0;
  for(; cur_start_offset < num_offsets; cur_start_offset = cur_end_offset) {
    cur_end_offset = cur_start_offset;
    while (cur_end_offset < num_offsets &&
           model.offsets[cur_end_offset].time_offset ==
           model.offsets[cur_start_offset].time_offset)
      cur_end_offset++;
    // we are processing the range of indexes into 'offsets'
    // from cur_start_offset to cur_end_offset - 1.
    int32 this_num_offsets = cur_end_offset - cur_start_offset;
    int32 time_offset = model.offsets[cur_start_offset].time_offset;

    ConvolutionComputation::ConvolutionStep step;
    // modified_time_offset will be used in working out the 'input_time_shift'
    // that determines which submatrix of the input matrix we'll use.
    // It equals the time-offset corrected for any time-difference between
    // the start of the output and of the input.
    int32 modified_time_offset = time_offset + io.start_t_out - io.start_t_in;
    KALDI_ASSERT(modified_time_offset >= 0 &&
                 modified_time_offset % t_step == 0);
    step.input_time_shift = modified_time_offset / t_step;
    KALDI_ASSERT(step.input_time_shift <= num_t_extra);
    step.params_start_col = model.num_filters_in * cur_start_offset;
    step.height_map.clear();
    step.height_map.reserve(model.height_out * this_num_offsets);
    for (int32 h_out = 0;
         h_out < model.height_out * model.height_subsample_out;
         h_out += model.height_subsample_out) {
      for (int32 o = cur_start_offset; o < cur_end_offset; o++) {
        int32 this_height_offset = model.offsets[o].height_offset,
            h_in = h_out + this_height_offset;
        // by the time we call MakeComputation, the user should already have
        // called PadModelHeight, so there should be no need for zero padding on
        // the height axis, hence the following check.  [we'll later modify the
        // resulting computation in UnPadModelHeight, and that's where
        // zero-padding gets taken account of.]
        KALDI_ASSERT(h_in >= 0 && h_in < model.height_in);
        step.height_map.push_back(h_in);
      }
    }
    computation->steps.push_back(step);
  }
  ComputeTempMatrixSize(opts, computation);
}


void ConvolutionComputationIo::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ConvCompIo>");
  WriteBasicType(os, binary, num_images);
  WriteBasicType(os, binary, start_t_in);
  WriteBasicType(os, binary, t_step_in);
  WriteBasicType(os, binary, num_t_in);
  WriteBasicType(os, binary, start_t_out);
  WriteBasicType(os, binary, t_step_out);
  WriteBasicType(os, binary, num_t_out);
  WriteBasicType(os, binary, reorder_t_in);
  WriteToken(os, binary, "</ConvCompIo>");
}


void ConvolutionComputationIo::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<ConvCompIo>");
  ReadBasicType(is, binary, &num_images);
  ReadBasicType(is, binary, &start_t_in);
  ReadBasicType(is, binary, &t_step_in);
  ReadBasicType(is, binary, &num_t_in);
  ReadBasicType(is, binary, &start_t_out);
  ReadBasicType(is, binary, &t_step_out);
  ReadBasicType(is, binary, &num_t_out);
  ReadBasicType(is, binary, &reorder_t_in);
  ExpectToken(is, binary, "</ConvCompIo>");
}

} // namespace time_height_convolution
} // namespace nnet3
} // namespace kaldi
