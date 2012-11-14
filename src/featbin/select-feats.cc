// featbin/copy-feats.cc

// Copyright 2012 Korbinian Riedhammer

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

#include <sstream>
#include <algorithm>
#include <iterator>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;
    
    const char *usage =
      "Select certain dimensions of the feature file;  think of it as the unix\n"
      "command cut -f ...\n"
      "Usage: merge-feats selection in-rspecifier out-wspecifier\n"
      "  e.g. merge-feats \"0 1 3-12\" scp:feats.scp ark,scp:feat-red.ark,feat-red.scp\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }    

    string sspecifier = po.GetArg(1);
    string rspecifier = po.GetArg(2);
    string wspecifier = po.GetArg(3);
    
    // figure out the selected dimensions
    istringstream iss(sspecifier);
    vector<string> tokens;
    copy(istream_iterator<string>(iss),
         istream_iterator<string>(),
         back_inserter<vector<string> >(tokens));
    
    vector<int> indices;
    for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); it++) {
      size_t p = it->find('-');
      if (p != string::npos) {
        int s, e;
        istringstream(it->substr(0, it->length() - p - 1)) >> s;
        istringstream(it->substr(p+1)) >> e;
        for (int j = s; j <= e; ++j)
            indices.push_back(j);
      } else {
        int i;
        istringstream(*it) >> i;
        indices.push_back(i);
      }
    }
    
    if (indices.size() < 1)
      KALDI_ERR << "No indices in format string!";
    
    // set up i/o
    
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    BaseFloatMatrixWriter kaldi_writer(wspecifier);

    // process all keys
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {
      Matrix<BaseFloat> feats(kaldi_reader.Value().NumRows(), indices.size());
      
      // extract the desired columns
      for (int i = 0; i < kaldi_reader.Value().NumRows(); ++i) {
        for (int j = 0; j < indices.size(); ++j) {
          feats(i, j) = kaldi_reader.Value()(i, indices[j]);
        }
      }
      
      kaldi_writer.Write(kaldi_reader.Key(), feats);
    }
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
