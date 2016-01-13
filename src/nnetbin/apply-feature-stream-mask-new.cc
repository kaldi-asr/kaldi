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

    std::string stream_combinations_str = "0";
    po.Register("stream-combinations", &stream_combinations_str, "If single   stream-combination is provided, assigns mask, so that for all frames stream-combination is present.\n"
		                                                 "If multiple stream-combination are provided, randomly selects one of the provided [used in training]. \n"
		                                                 "Multiple stream combinations have to seperated by :");

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

    RandomAccessBaseFloatMatrixReader stream_mask_reader;
    if (stream_mask != "") {
      stream_mask_reader.Open(stream_mask);
    }


    std::vector<int> stream_indices;
    bool ret1 = SplitStringToIntegers(stream_indices_str, ":", true, &stream_indices);
    if (!ret1) {
      KALDI_ERR << "Cannot parse the --stream-indices. It should be"
		<< "colon-separated list of integers";
    }

    std::vector<int64> stream_combinations;
    bool ret2 = SplitStringToIntegers(stream_combinations_str, ":", true, &stream_combinations);
    if (!ret2) {
      KALDI_ERR << "Cannot parse the --stream-combinations. It should be "
		<< "colon-separated list of integers";
    }
    
    // Sanity checks 
    if (crossvalidate == "true") {

      if (stream_combinations.size() > 1) {
	KALDI_ERR << "Given --cross-validate=true"
	          << " but provided multiple --stream-combinations=''";
      }

      if ((stream_mask == "") && (stream_combinations[0] == 0)) {
	KALDI_ERR << "Given --cross-validate=true"
	          << " but --stream-mask='' and --stream-combinations=0";
      }
    } else { // crossvalidate false
      if (stream_combinations[0] == 0) {
	KALDI_ERR << "Given --cross-validate=false"
	          << " but --stream-combinations=0 => you are asking to train on zeros";
      }
      if (stream_combinations.size() == 1) {
	KALDI_WARN << "Given --cross-validate=false"
	           << " but --stream-combinations.size()=1 => you are asking to do standard trainining";
      }
    }

    int32 num_streams = stream_indices.size() - 1;

    std::map<int64, std::vector<int32> > stream_combinations_to_mask;
    if (stream_mask == "") {
      for (int32 i=0; i<stream_combinations.size(); i++) {
	stream_combinations_to_mask[stream_combinations[i]] = toBinary(stream_combinations[i], num_streams);
      }
    }


    Timer time;
    double time_now = 0;
    int32 num_done = 1;

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

      if (crossvalidate == "false") {
	// Apply random stream mask to each row mat and put it in out_mat
	for (int32 i=0; i<mat.NumRows(); i++) {
	  int64 this_frame_stream_combination;// = stream_combinations[RandInt(0, stream_combinations.size()-1)];
	  if (stream_combinations.size() == 1) {
	    this_frame_stream_combination = stream_combinations[0];
	  } else {
	    this_frame_stream_combination = stream_combinations[RandInt(0, stream_combinations.size()-1)];
	  }
	  SubVector<BaseFloat> Row(mat, i);
	  std::vector<int> this_frame_stream_mask = stream_combinations_to_mask[this_frame_stream_combination];

	  for (int32 j=0; j<num_streams; j++) {
	    SubVector<BaseFloat> subRow(Row, stream_indices[j], stream_indices[j+1] - stream_indices[j]);
	    subRow.Scale(this_frame_stream_mask[j]);
	  }	  
	}

      } else {
	if (stream_mask != "") {
	  Matrix<BaseFloat> this_utt_stream_mask;
	  this_utt_stream_mask = stream_mask_reader.Value(utt);
	  for (int32 j=0; j<num_streams; j++) {

	    SubMatrix<BaseFloat> this_stream(mat.ColRange(stream_indices[j], stream_indices[j+1] - stream_indices[j]));
	    Vector<BaseFloat> this_stream_mask;

	    this_stream_mask.Resize(this_utt_stream_mask.NumRows());
	    this_stream_mask.CopyColFromMat(this_utt_stream_mask, j);
	    
	    this_stream.MulRowsVec(this_stream_mask);
	  }

	} else {
	  std::vector<int> frame_stream_mask = stream_combinations_to_mask[stream_combinations[0]];

	  for (int32 i=0; i<mat.NumRows(); i++) {
	    SubVector<BaseFloat> Row(mat, i);
	    for (int32 j=0; j<num_streams; j++) {
	      SubVector<BaseFloat> subRow(Row, stream_indices[j], stream_indices[j+1] - stream_indices[j]);
	      subRow.Scale(frame_stream_mask[j]);
	    }
	  }
	}
      }

      // Write
      feature_writer.Write(feature_reader.Key(), mat);

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
