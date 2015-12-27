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

  if (num_bits > 10) {
    KALDI_ERR << "Doesn't support more than 10 streams";
  }

  std::bitset<10> bin_n(n); //10 upper limit, Hardcoded
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

    int32 stream_combination = 0;
    po.Register("stream-combination", &stream_combination, "Assign mask, so that for all frames stream-combination is present");

    std::string stream_combination_pvals;
    po.Register("stream-combination-pvals", &stream_combination_pvals, "Probability values Feature transform in front of main network (in nnet format)");

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
    bool ret = SplitStringToIntegers(stream_indices_str, ":", true, &stream_indices);
    if (!ret) {
      KALDI_ERR << "Cannot parse the stream_indices. It should be"
		<< "colon-separated list of integers";
    }

    int32 num_streams = stream_indices.size() - 1;
    int32 num_stream_combns = pow(2, num_streams) - 1; // -1 to remove null combination

    // create a look up table for stream_combination_number to mask
    std::vector<std::vector<int32> > stream_combination_to_mask;
    stream_combination_to_mask.push_back(toBinary(0, num_streams));
    for (int32 i=1; i<=num_stream_combns; i++) {
      stream_combination_to_mask.push_back(toBinary(i, num_streams));
    }

    Timer time;
    double time_now = 0;
    int32 num_done = 0;
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

      // Apply stream mask to mat and put it in out_mat
      for (int32 i=0; i<mat.NumRows(); i++) {
	int32 this_frame_stream_combination = 0;
	if (stream_combination != 0 ) {
	  this_frame_stream_combination = stream_combination;
	} else {
	  // Randomly select stream combination
	  if (stream_combination_pvals == "" ) {
	    this_frame_stream_combination = RandInt(1, num_stream_combns);
	  } else {
	    // Biased random number generation based on stream_combination_pvals
	    KALDI_ERR << "Not yet implemented random number generation based on stream_combination_pvals";  
	  }
	}

        SubVector<BaseFloat> Row(mat, i);
	std::vector<int> this_frame_stream_mask = stream_combination_to_mask[this_frame_stream_combination];
	for (int32 j=0; j<num_streams; j++) {
	  SubVector<BaseFloat> subRow(Row, stream_indices[j], stream_indices[j+1] - stream_indices[j]);
	  subRow.Scale(this_frame_stream_mask[j]);
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
