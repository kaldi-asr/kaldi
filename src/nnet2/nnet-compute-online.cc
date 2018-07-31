// nnet2/nnet-compute-online.cc

// Copyright 2014   Johns Hopkins University (author: Daniel Povey)
//                  Guoguo Chen
//                  Vijayaditya Peddinti

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
#include <vector>

namespace kaldi {
namespace nnet2 {

NnetOnlineComputer::NnetOnlineComputer(const Nnet &nnet, bool pad_input)
    : nnet_(nnet), pad_input_(pad_input),
      is_first_chunk_(true), finished_(false) {
  data_.resize(nnet_.NumComponents() + 1);
  reusable_component_inputs_.resize(nnet_.NumComponents()+1);
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
  } else {
    // store the last frame as it might be needed for padding when Flush() is
    // called.
    if (last_seen_input_frame_.Dim() != input.NumCols())
      last_seen_input_frame_.Resize(input.NumCols());
    last_seen_input_frame_.CopyFromVec(input.Row(input.NumRows() - 1));
  }

  // Checking if feature dimension matches that required by the neural network.
  if (dim != nnet_.InputDim()) {
    KALDI_ERR << "Feature dimension is " << dim << ", but network expects "
        << nnet_.InputDim();
  }
  // num_effective_input_rows is the effective number of input rows we have, for
  // purposes of computing how much output we will get.  It is the number of
  // actual input rows plus the amount of context stored at intermediate layers
  // of the network (which if we have previously done the computation, will
  // equal nnet_.LeftContext() + nnet_.RightContext()).
  int32 num_effective_input_rows = 0;
  // Initialize the first element of data_, with input
  CuMatrix<BaseFloat> &input_data(data_[0]);
  if (is_first_chunk_)  {
    is_first_chunk_ = false;
    // assert that all the component-wise input buffers are empty
    for (int32 i = 0; i < reusable_component_inputs_.size(); i++)
      KALDI_ASSERT(reusable_component_inputs_[0].NumRows() == 0);
    // Pad at the start of the file if necessary.
    if ((pad_input_) && (nnet_.LeftContext() > 0))  {
        input_data.Resize(nnet_.LeftContext() + input.NumRows(), dim);
        input_data.Range(0, nnet_.LeftContext(), 0,
                    dim).CopyRowsFromVec(input.Row(0));
        input_data.Range(nnet_.LeftContext(), input.NumRows(),
                    0, dim).CopyFromMat(input);
    } else {
      input_data.Resize(input.NumRows(), input.NumCols());
      input_data.CopyFromMat(input);
    }
    num_effective_input_rows = input_data.NumRows();
  } else {
    int32 extra_input_rows = 0;
    // checking if we did forward pass for any chunks before.
    // if we did a forward pass, component input buffers would be non-empty
    // these buffers store information equivalent to having an nnet_input
    // buffer of (nnet_.LeftContext() + nnet_.RightContext())
    for (int32 i = 0; i < reusable_component_inputs_.size(); i++)  {
      if (reusable_component_inputs_[i].NumRows() > 0) {
        extra_input_rows = nnet_.LeftContext() + nnet_.RightContext();
        break;
      }
    }
    // add unprocessed input from the previous calls
    input_data.Resize(input.NumRows() + unprocessed_buffer_.NumRows(), dim);
    if (unprocessed_buffer_.NumRows() > 0)
      input_data.Range(0, unprocessed_buffer_.NumRows(),
                       0, dim).CopyFromMat(unprocessed_buffer_);
    input_data.Range(unprocessed_buffer_.NumRows(), input.NumRows(),
                     0, dim).CopyFromMat(input);
    unprocessed_buffer_.Resize(0, 0); // clearing the unprocessed buffer
    num_effective_input_rows = input_data.NumRows() + extra_input_rows;
  }
  if (num_effective_input_rows >=
      nnet_.LeftContext() + nnet_.RightContext() + 1) {
    // we have sufficient frames to compute at least one nnet output
    nnet_.ComputeChunkInfo(num_effective_input_rows, 1, &chunk_info_);
    Propagate();
    *output = data_.back();
  } else {
    // store the input in the unprocessed_buffer_
    unprocessed_buffer_ = input_data;
    // not enough input context so just return an empty array
    output->Resize(0, 0);
  }

}

void NnetOnlineComputer::Flush(CuMatrix<BaseFloat> *output) {
  KALDI_ASSERT(!finished_ && !is_first_chunk_);
  int32 num_frames_padding = (pad_input_ ? nnet_.RightContext() : 0);
  int32 num_stored_frames = nnet_.LeftContext() + nnet_.RightContext();
  int32 num_effective_input_rows =  num_stored_frames + num_frames_padding;
  // If the amount of output would be empty return at this point.
  if (num_effective_input_rows < nnet_.LeftContext() + nnet_.RightContext() + 1) {
    output->Resize(0, 0);
    finished_ = true;
    return;
  }

  int32 dim = nnet_.InputDim();
  CuMatrix<BaseFloat> &input_data(data_[0]);
  KALDI_ASSERT(num_frames_padding > 0);  // else we would have returned above.
  input_data.Resize(num_frames_padding, dim);
  input_data.CopyRowsFromVec(last_seen_input_frame_);

  // Note, we later modify this chunk-info, it isn't quite correct right now
  // because we add extra data at intermediate layers, and the actual number of
  // input rows doesn't equal num_effective_input_rows.
  nnet_.ComputeChunkInfo(num_effective_input_rows, 1,
                         &chunk_info_);
  Propagate();
  *output = data_.back();
  finished_ = true;
}

void NnetOnlineComputer::Propagate() {
  // This method is like the normal nnet propagate, but we reuse the frames
  // computed from the previous chunk, at each component.

  for (int32 c = 0; c < nnet_.NumComponents(); c++) {
    // we assume that the chunks are always contiguous
    chunk_info_[c].MakeOffsetsContiguous();
    chunk_info_[c + 1].MakeOffsetsContiguous();

    const Component &component = nnet_.GetComponent(c);
    CuMatrix<BaseFloat> &input_data = data_[c], &output_data = data_[c + 1];
    CuMatrix<BaseFloat> input_data_temp;

    if (component.Context().size() > 1)  {
      int32 dim = component.InputDim();
      if (reusable_component_inputs_[c].NumRows() > 0) {
        // concatenate any frames computed by previous component
        // in the last call, to the input of the current component
        input_data_temp.Resize(reusable_component_inputs_[c].NumRows()
                               + input_data.NumRows(), dim);
        input_data_temp.Range(0, reusable_component_inputs_[c].NumRows(),
                       0, dim).CopyFromMat(reusable_component_inputs_[c]);
        input_data_temp.Range(reusable_component_inputs_[c].NumRows(),
                              input_data.NumRows(), 0, dim).CopyFromMat(
                                  input_data);
        input_data = input_data_temp;
      }
      // store any frames which can be reused in the next call
      reusable_component_inputs_[c].Resize(component.Context().back() -
                                component.Context().front(), dim);
      reusable_component_inputs_[c].CopyFromMat(
          input_data.RowRange(input_data.NumRows() -
                              reusable_component_inputs_[c].NumRows(),
                              reusable_component_inputs_[c].NumRows()));
    }

    // chunk_info objects provided assume that we added all the reusable
    // context at the input of the nnet. However we are reusing hidden
    // activations computed in the previous call.
    // Hence we manipulate the chunk_info objects to reflect the state of the
    // actual chunk, each component is computing, in the current Propagate.
    // As before we always assume the chunks are contiguous.

    // modifying the input chunk_info
    int32 chunk_size_assumed = chunk_info_[c].ChunkSize();
    int32 last_offset = chunk_info_[c].GetOffset(chunk_size_assumed - 1);
    int32 first_offset = last_offset - input_data.NumRows() + 1;
    ChunkInfo input_chunk_info(chunk_info_[c].NumCols(),
                               chunk_info_[c].NumChunks(),
                               first_offset,
                               last_offset);
    // modifying the output chunk_info
    chunk_size_assumed = chunk_info_[c + 1].ChunkSize();
    last_offset = chunk_info_[c + 1].GetOffset(chunk_size_assumed - 1);
    first_offset = last_offset - (input_data.NumRows() -
                                  (component.Context().back() -
                                   component.Context().front())) + 1;
    ChunkInfo output_chunk_info(chunk_info_[c + 1].NumCols(),
                                chunk_info_[c + 1].NumChunks(),
                                first_offset,
                                last_offset);
    component.Propagate(input_chunk_info, output_chunk_info,
                        input_data, &output_data);
  }
}

}  // namespace nnet2
}  // namespace kaldi
