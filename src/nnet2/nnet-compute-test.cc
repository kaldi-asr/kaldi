// nnet2/nnet-compute-test.cc

// Copyright 2014  Johns Hopkins University (author:  Daniel Povey)
// Copyright 2015  David Snyder

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

#include "nnet2/nnet-nnet.h"
#include "nnet2/nnet-compute.h"
#include "nnet2/nnet-compute-online.h"

namespace kaldi {
namespace nnet2 {


void UnitTestNnetCompute() {
  int32 input_dim = 10 + rand() % 40, output_dim = 100 + rand() % 500;
  bool pad_input = (rand() % 2 == 0);
  
  Nnet *nnet = GenRandomNnet(input_dim, output_dim);
  KALDI_LOG << "Left context = " << nnet->LeftContext() << ", right context = "
            << nnet->RightContext() << ", pad-input = " << pad_input;
  KALDI_LOG << "NNet info is " << nnet->Info();
  int32 num_feats = 5 + rand() % 1000;
  CuMatrix<BaseFloat> input(num_feats, input_dim);
  input.SetRandn();

  int32 num_output_rows = num_feats -
      (pad_input ? 0 : nnet->LeftContext() + nnet->RightContext());
  if (num_output_rows <= 0)
    return;
  CuMatrix<BaseFloat> output1(num_output_rows, output_dim);
  NnetComputation(*nnet, input, pad_input, &output1);
  CuMatrix<BaseFloat> output2(output1.NumRows(), output1.NumCols());
  int32 cur_input_pos = 0, cur_output_pos = 0;

  NnetOnlineComputer computer(*nnet, pad_input);
  while (cur_input_pos <= num_feats) {
    int32 feats_left = num_feats - cur_input_pos;
    CuMatrix<BaseFloat> output_part;
    if (feats_left > 0) {
      int32 chunk_size = std::min<int32>(1 + rand() % 10, feats_left);
      CuSubMatrix<BaseFloat> input_part(input, cur_input_pos, chunk_size,
                                        0, input_dim);
      computer.Compute(input_part, &output_part);
      cur_input_pos += chunk_size;
    } else {
      computer.Flush(&output_part);
      cur_input_pos++; // will terminate the loop.
    }
    if (output_part.NumRows() != 0) {
      output2.Range(cur_output_pos, output_part.NumRows(),
                    0, output_dim).CopyFromMat(output_part);
      cur_output_pos += output_part.NumRows();
    }
  }  
  AssertEqual(output1, output2);
  for (int32 i = 0; i < output1.NumRows(); i++) {
    // just double-check that the frames near the end are right, in case
    // the test above somehow passed despite that.
    if (i < 10 || output1.NumRows() - i < 10) {
      CuSubVector<BaseFloat> vec1(output1, i), vec2(output2, i);
      AssertEqual(vec1, vec2);
    }
  }
  KALDI_LOG << "OK";
  delete nnet;
}

void UnitTestNnetComputeChunked() {
  int32 input_dim = 10 + rand() % 40, output_dim = 100 + rand() % 500;
  bool pad_input = true;
  
  Nnet *nnet = GenRandomNnet(input_dim, output_dim);
  int32 num_feats = 100 + rand() % 500;
  int32 chunk_size = num_feats / (2 + rand() % 10);
  CuMatrix<BaseFloat> input(num_feats, input_dim);
  input.SetRandn();

  KALDI_LOG << "Left context = " << nnet->LeftContext() 
            << ", right context = " << nnet->RightContext() 
            << ", chunk size = " << chunk_size;
  KALDI_LOG << "NNet info is " << nnet->Info();

  int32 num_output_rows = num_feats;
  CuMatrix<BaseFloat> cu_output1(num_output_rows, output_dim);
  CuMatrix<BaseFloat> cu_output2(num_output_rows, output_dim);
  NnetComputation(*nnet, input, pad_input, &cu_output1);
  NnetComputationChunked(*nnet, CuMatrix<BaseFloat>(input), chunk_size, 
                         &cu_output2);
  Matrix<BaseFloat> output1(cu_output1);
  Matrix<BaseFloat> output2(cu_output2);
  AssertEqual(output1, output2);
  for (int32 i = 0; i < output1.NumRows(); i++) {
    // just double-check that the frames near the end are right, in case
    // the test above somehow passed despite that.
    if (i < 10 || output1.NumRows() - i < 10) {
      SubVector<BaseFloat> vec1(output1, i), vec2(output2, i);
      AssertEqual(vec1, vec2);
    }
  }
  KALDI_LOG << "OK";
  delete nnet;
}

}  // namespace nnet2
}  // namespace kaldi

#include "matrix/matrix-functions.h"


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;

  for (int32 i = 0; i < 10; i++) 
    UnitTestNnetCompute();
    UnitTestNnetComputeChunked();
  return 0;
}
  
