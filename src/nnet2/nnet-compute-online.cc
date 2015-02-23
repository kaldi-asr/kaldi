// nnet2/nnet-compute-online.cc

// Copyright 2014   Johns Hopkins University (author: Daniel Povey)
//                  Guoguo Chen

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

#include "nnet2/nnet-compute-online.h"

namespace kaldi {
namespace nnet2 {

NnetOnlineComputer::NnetOnlineComputer(const Nnet &nnet, bool pad_input)
  : nnet_(nnet), pad_input_(pad_input),
    is_first_chunk_(true), finished_(false) {
  data_.resize(nnet_.NumComponents() + 1);
  unused_input_.Resize(0, 0);
}

void NnetOnlineComputer::Compute(const CuMatrixBase<BaseFloat> &input,
                                 CuMatrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  KALDI_ASSERT(!finished_);
  int32 dim = input.NumCols();

  // If input is empty, we also set output to zero size.
  if (input.NumRows() == 0) {
    output->Resize(0, 0);
    return;
  }

  // Check if feature dimension matches that required by the neural network.
  if (dim != nnet_.InputDim()) {
    KALDI_ERR << "Feature dimension is " << dim << ", but network expects "
        << nnet_.InputDim();
  }

  // Pad at the start of the file if necessary.
  if (pad_input_ && is_first_chunk_) {
    KALDI_ASSERT(unused_input_.NumRows() == 0);
    if (nnet_.LeftContext() > 0)
      unused_input_.Resize(nnet_.LeftContext(), dim);
    for (int32 i = 0; i < nnet_.LeftContext(); i++)
      unused_input_.Row(i).CopyFromVec(input.Row(0));
    is_first_chunk_ = false;
  }

  int32 num_rows = unused_input_.NumRows() + input.NumRows();

  // Splice unused_input_ and input.
  CuMatrix<BaseFloat> &input_data(data_[0]);
  input_data.Resize(num_rows, dim);
  if (unused_input_.NumRows() > 0) {
    input_data.Range(0, unused_input_.NumRows(),
                     0, dim).CopyFromMat(unused_input_);
    input_data.Range(unused_input_.NumRows(), input.NumRows(),
                     0, dim).CopyFromMat(input);
  }

  if (num_rows > nnet_.LeftContext() + nnet_.RightContext()) {
    nnet_.ComputeChunkInfo(num_rows, 1, &chunk_info_);
    Propagate();
    *output = data_.back();
  } else {
    output->Resize(0, 0);
  }

  // Now store the part of input that will be needed in the next call of
  // Compute().
  int32 unused_num_rows = nnet_.LeftContext() + nnet_.RightContext();
  if (unused_num_rows > num_rows) { unused_num_rows = num_rows; }
  if (unused_num_rows > 0) {
    unused_input_.Resize(unused_num_rows, dim);
    unused_input_.CopyFromMat(input_data.Range(num_rows - unused_num_rows,
                                               unused_num_rows, 0, dim));
  } // else unused_input_ would already be empty, so no need to resize.
}

void NnetOnlineComputer::Flush(CuMatrix<BaseFloat> *output) {
  KALDI_ASSERT(!finished_);

  int32 num_frames_padding = (pad_input_ ? nnet_.RightContext() : 0);
  int32 num_input_rows = unused_input_.NumRows() + num_frames_padding;
  
  // If the amount of output would be empty, return at this point.
  if (num_input_rows <= nnet_.LeftContext() + nnet_.RightContext()) {
    output->Resize(0, 0);
    finished_ = true;
    return;
  }

  int32 dim = nnet_.InputDim();
  CuMatrix<BaseFloat> &input_data(data_[0]);
  input_data.Resize(num_input_rows, dim);
  input_data.Range(0, unused_input_.NumRows(),
                   0, dim).CopyFromMat(unused_input_);
  if (num_frames_padding > 0) {
    int32 last_row = unused_input_.NumRows() - 1;
    for (int32 i = 0; i < num_frames_padding; i++)
      input_data.Row(num_input_rows - i - 1).CopyFromVec(
          unused_input_.Row(last_row)); 
  }
  
  nnet_.ComputeChunkInfo(num_input_rows, 1,
                         &chunk_info_);
  Propagate();
  *output = data_.back();
  finished_ = true;
}

void NnetOnlineComputer::Propagate() {
  for (int32 c = 0; c < nnet_.NumComponents(); c++) {
    const Component &component = nnet_.GetComponent(c);
    CuMatrix<BaseFloat> &input_data = data_[c], &output_data = data_[c + 1];
    component.Propagate(chunk_info_[c], chunk_info_[c + 1],
                        input_data, &output_data);
  }
}


} // namespace nnet2
} // namespace kaldi
