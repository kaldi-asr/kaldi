// nnetbin/nnet-forward.cc

// Copyright 2015  Johns Hopkins Univesity (Author: Sri Harish Mallidi)

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

#include <limits>

#include "base/kaldi-common.h"

#include "util/common-utils.h"
#include "base/timer.h"

#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"

#include <bitset>

#include <iostream>

std::vector<int> toBinary(int n, int num_bits) {
  using namespace kaldi;

  if (num_bits > 60) {
    KALDI_ERR << "Doesn't support more than 10 streams";
  }

  std::bitset<60> bin_n(n); //10 upper limit, Hardcoded
  std::string mystring = bin_n.to_string<char,std::string::traits_type,std::string::allocator_type>();
  std::vector<int32> bin_vec(num_bits, 0);
  
  for ( int i=0; i<num_bits; i++) {
    std::string c = mystring.substr(mystring.size()-(1+i), 1);  
    int c_int = 0;
    bool ans = ConvertStringToInteger(c, &c_int);
    if (!ans) {
      KALDI_ERR << "Unable to contert to in " << c;
    }
    bin_vec[num_bits-(1+i)] = c_int;
  }

  return bin_vec;
}

int main(int argc, char *argv[]) {
  using namespace kaldi;

  try {
    const char *usage =
        "Apply stream mask"
        "\n"
        "Usage:  apply-feature-stream-mask [options] <stream-indices> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " apply-feature-stream-mask 0:84:168:252:336:420 ark:features.ark ark:masked.output.ark\n";

    ParseOptions po(usage);

    double block_dropout_retention = 0.5;
    po.Register("block-dropout-retention", &block_dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");

    int32 stream_combination = 0;
    po.Register("stream-combination", &stream_combination, "converts to binary to get stream mask");

    std::string stream_mask;
    po.Register("stream-mask", &stream_mask, "Stream mask (Kaldi rspecifier).");

    std::string crossvalidate = "false";
    po.Register("cross-validate", &crossvalidate, "If provided expects single --stream-combinations or --stream-mask, else does random masking");

    int32 seed = 777;
    po.Register("seed", &seed, "Seed for random number generator");

    using namespace kaldi;
    typedef kaldi::int32 int32;


    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string stream_indices_str = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);

    std::srand(seed);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    std::vector<int> stream_indices;
    bool ret1 = SplitStringToIntegers(stream_indices_str, ":", true, &stream_indices);
    if (!ret1) {
      KALDI_ERR << "Cannot parse the --stream-indices. It should be"
		<< "colon-separated list of integers";
    }
    int32 num_streams = stream_indices.size() - 1;

    if (crossvalidate == "true") {
      if ((stream_mask == "" ) && (stream_combination == 0)) {
	KALDI_ERR << "Given crossvalidate=true , but not provided valid stream-mask or stream-combination";
      }
    }

    RandomAccessBaseFloatMatrixReader stream_mask_reader;
    if (stream_mask != "") {
      stream_mask_reader.Open(stream_mask);
    }
    std::vector<int32> stream_combination_to_mask;
    stream_combination_to_mask = toBinary(stream_combination, num_streams);

    Timer time;
    double time_now = 0;
    int32 num_done = 1;

    CuMatrix<BaseFloat> feats, nnet_out;
    Matrix<BaseFloat> host_out;

    CuRand<BaseFloat> rand_;
    CuMatrix<BaseFloat> block_dropout_mask;
    Matrix<BaseFloat> block_dropout_mask_out;

    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << utt
                    << ", " << mat.NumRows() << "frm";

      if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }
      
      // Check if feature_dim == end of stream_indices
      KALDI_ASSERT(mat.NumCols() == stream_indices[stream_indices.size()-1]);

      // push it to gpu
      feats = mat;

      block_dropout_mask.Resize(feats.NumRows(), num_streams);
      CuMatrix<BaseFloat> this_stream_dropout_mask;
      this_stream_dropout_mask.Resize(block_dropout_mask.NumRows(), 1);

      CuVector<BaseFloat> this_stream_mask;
      this_stream_mask.Resize(this_stream_dropout_mask.NumRows());

      CuMatrix<BaseFloat> tmp_row_stream_mask;
      tmp_row_stream_mask.Resize(num_streams, 1);

      if (crossvalidate == "false") { // training stage, randomly dropout streams

	for (int32 j=0; j<num_streams; j++) {
	  
	  this_stream_dropout_mask.Set(block_dropout_retention);
	  rand_.BinarizeProbs(this_stream_dropout_mask, &this_stream_dropout_mask);
	  
	  this_stream_mask.CopyColFromMat(this_stream_dropout_mask, 0);
	  block_dropout_mask.CopyColFromVec(this_stream_mask, j);

	}

	// Fix rows having all zeros
	for (int32 i=0; i<block_dropout_mask.NumRows(); i++) {
	  CuSubVector<BaseFloat> this_row(block_dropout_mask, i);
	  if (this_row.Sum() == 0) {
	    while (1) {
	      tmp_row_stream_mask.Set(block_dropout_retention);
	      rand_.BinarizeProbs(tmp_row_stream_mask, &tmp_row_stream_mask);
	      // Copy to this_row
	      this_row.CopyColFromMat(tmp_row_stream_mask, 0);
	      if (this_row.Sum() !=0 ) {
		break;
	      }
	    }
	  }
	}
      } else { // testing stage
	if (stream_mask != "") {

	  Matrix<BaseFloat> this_utt_stream_mask;
	  this_utt_stream_mask = stream_mask_reader.Value(utt);
	  
	  block_dropout_mask = this_utt_stream_mask;
	} else {
	  if (stream_combination == 0) {
	    KALDI_ERR << "stream-combination=0, invalid options\n";
	  }

	  for (int32 j=0; j<num_streams; j++) {
	    CuSubMatrix<BaseFloat> this_stream_block_dropout_mask(block_dropout_mask.ColRange(j, 1));
	    this_stream_block_dropout_mask.Set(stream_combination_to_mask[j]);
	  }
	}
      }      

      for (int32 j=0; j<num_streams; j++) {
	this_stream_mask.CopyColFromMat(block_dropout_mask, j);
	CuSubMatrix<BaseFloat> this_stream_feats(feats.ColRange(stream_indices[j], stream_indices[j+1] -stream_indices[j]));
	this_stream_feats.MulRowsVec(this_stream_mask);
      }

      block_dropout_mask_out.Resize(block_dropout_mask.NumRows(), block_dropout_mask.NumCols());
      block_dropout_mask.CopyToMat(&block_dropout_mask_out);
      
      host_out.Resize(feats.NumRows(), feats.NumCols());
      feats.CopyToMat(&host_out);

      // Write
      feature_writer.Write(feature_reader.Key(), host_out);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();

    }


    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
