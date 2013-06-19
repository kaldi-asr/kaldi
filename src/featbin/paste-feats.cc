// featbin/paste-feats.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;
    
    const char *usage =
      "Paste feature files (assuming they have the same lengths);  think of the\n"
      "unix command paste a b. You might be interested in select-feats, too.\n"
      "Usage: paste-feats in-rspecifier1 in-rspecifier2 [in-rspecifier3 ...] out-wspecifier\n"
      "  e.g. paste-feats ark:feats1.ark \"ark:select-feats 0-3 ark:feats2.ark ark:- |\" ark:feats-out.ark\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }
    
    // last argument is output
    string wspecifier = po.GetArg(po.NumArgs());
    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    
    // assemble vector of input readers, peek dimensions
    vector<SequentialBaseFloatMatrixReader *> input;
    vector<int32> offsets;
    int32 dim = 0;
    for (int i = 1; i < po.NumArgs(); ++i) {
      string rspecifier = po.GetArg(i);
      SequentialBaseFloatMatrixReader *rd = new SequentialBaseFloatMatrixReader(rspecifier);
      input.push_back(rd);
      offsets.push_back(dim);
      dim += rd->Value().NumCols();
    }
    
    bool done = false;
    while (!done) {
      string key = "";
      Matrix<BaseFloat> feats;
      int32 num;
      
      bool incomplete = false;
      for (int i = 0; i < input.size(); ++i) {
        SequentialBaseFloatMatrixReader *rd = input[i];
        
        // the first input specifies the output keys
        if (i == 0) {
          if (rd->Done()) {
            done = true;
            break;
          } else {
            key = rd->Key();
            num = rd->Value().NumRows();
            feats.Resize(num, dim);
            incomplete = false;
          }
        } else if (rd->Done()) {
          // we will ignore incomplete utts
          KALDI_WARN << "Unexpected end of archive in input " << (i+1)
                     << ";  expected key " << key;
          incomplete = true;
          break;
        } else if (key.compare(rd->Key()) != 0) {
          KALDI_WARN << "Error in input " << (i+1) << ";  expected key "
                     << key << " but got " << rd->Key() << ".  Terminating.";
          incomplete = true;
          done = true;
          break;
        } else if (num != rd->Value().NumRows()) {
          KALDI_WARN << "Error in input " << (i+1) << ";  got " << rd->Value().NumRows()
                     << " instead of " << num << " samples.  Ignoring utt " << key;
          incomplete = true;
          break;
        }

        // copy features
        if (!incomplete && !done) {
          feats.Range(0, num, offsets[i], rd->Value().NumCols()).CopyFromMat(rd->Value());
        }
        
        rd->Next();
      }

      if (done)
        break;
      
      if (!incomplete)
        kaldi_writer.Write(key, feats);
    }
    
    // delete the readers
    for (vector<SequentialBaseFloatMatrixReader *>::iterator it = input.begin();
         it != input.end(); it++) {
      delete (*it);
    }
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


