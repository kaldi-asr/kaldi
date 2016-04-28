// ivectorbin/ivector-cluster.cc

// Copyright 2016  Matthew Maciejewski

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"

namespace kaldi {

void PrepareSpeakerNums (const std::string utt2num_rxfilename,
			 unordered_map<std::string, int32> *utt2num) {

  Input ki(utt2num_rxfilename);

  std::string line;
  while (std::getline(ki.Stream(), line)) {
    std::vector<std::string> split_line;
    SplitStringToVector(line, " ", false, &split_line);
    std::string utt = split_line[0],
	num_str = split_line[1];

    int32 num;
    if (!ConvertStringToInteger(num_str, &num)) {
      KALDI_ERR << "Invalid number of speakers in utt2num: " << line;
    }

    utt2num->insert(std::make_pair<std::string, int32>(utt, num));
  }
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
	"Cluster iVectors for speaker diarization\n"
	"Usage: ivector-cluster [options] <ivectors-rspecifier> "
	"<ivector-ranges-rspecifier> <cluster-ranges-wspecifier>\n"
	"e.g.: \n"
	" ivector-cluster ark,t:ivectors.1.ark ark,t:ivector_ranges.1.ark \\\n"
	"   ark,t:cluster_ranges.1.ark\n";

    ParseOptions po(usage);
    std::string ivector_weights_rspecifier,
	        utt2num_rxfilename;
    po.Register("ivector-weights-rspecifier", &ivector_weights_rspecifier,
		"If supplied, uses the frames per ivector as weights "
		"in the K-means clustering.");
    po.Register("utt2num-rxfilename", &utt2num_rxfilename,
		"If supplied, uses the number of speakers for each "
		"file as the number of clusters.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivectors_rspecifier = po.GetArg(1),
	ivector_ranges_rspecifier = po.GetArg(2),
	cluster_ranges_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader ivectors_reader(ivectors_rspecifier);
    RandomAccessTokenVectorReader ivector_ranges_reader(ivector_ranges_rspecifier);
    RandomAccessTokenVectorReader ivector_weights_reader(ivector_weights_rspecifier);
    TokenVectorWriter cluster_ranges_writer(cluster_ranges_wspecifier);

    unordered_map<std::string, int32> utt2num;
    if ( !utt2num_rxfilename.empty() ) {
      PrepareSpeakerNums(utt2num_rxfilename, &utt2num);
    }

    for (; !ivectors_reader.Done(); ivectors_reader.Next()) {
      std::string utt = ivectors_reader.Key();
      const Matrix<BaseFloat> &mat = ivectors_reader.Value();
      std::vector<int32> weights(mat.NumRows());

      if ( !ivector_weights_rspecifier.empty() ) {
	std::vector<std::string> weights_str = ivector_weights_reader.Value(utt);
	for (int32 i = 0; i < weights_str.size(); i++) {
	  if ( !ConvertStringToInteger(weights_str[i], &weights[i]) ) {
	    KALDI_ERR << "Invalid weights file: " << ivector_weights_rspecifier;
	  }
	}
      } else {
        for (int32 i = 0; i < weights.size(); i++) {
	  weights[i] = 1;
	}
      }

      std::vector<Clusterable*> ivectors_clusterables;
      std::vector<int32> spk_ids;
      ClusterKMeansOptions opts;
      for (int32 i = 0; i < mat.NumRows(); i++) {
        Vector<BaseFloat> ivec(mat.NumCols());
	ivec.CopyRowFromMat(mat, i);
        ivectors_clusterables.push_back(new VectorClusterable(ivec, static_cast<BaseFloat>(weights[i])));
      }
      ClusterKMeans(ivectors_clusterables, utt2num[utt], NULL, &spk_ids, opts);

      std::vector<std::string> ivector_ranges = ivector_ranges_reader.Value(utt);
      std::vector<std::string> cluster_ranges(ivector_ranges.size());
      for (int32 i = 0; i < ivector_ranges.size(); i++) {
	std::stringstream ss;
	ss << ivector_ranges[i] << ',' << spk_ids[i];
        cluster_ranges[i] = ss.str();
      }
      cluster_ranges_writer.Write(utt, cluster_ranges);

      DeletePointers(&ivectors_clusterables);
    }


  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
