// featbin/paste-feats.cc

// Copyright 2012 Korbinian Riedhammer
// Copyright 2013 Brno University of Technology (Author: Karel Vesely)

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

    int32 length_tolerance = 0;
    po.Register("length-tolerance", &length_tolerance,
                "Tolerate small length differences of feats (warn and trim at end)");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }
    
    // Last argument is output
    string wspecifier = po.GetArg(po.NumArgs());
    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    
    // First input is sequential
    string rspecifier1 = po.GetArg(1);
    SequentialBaseFloatMatrixReader input1(rspecifier1);

    // Assemble vector of other input readers (with random-access)
    vector<RandomAccessBaseFloatMatrixReader *> input;
    for (int32 i = 2; i < po.NumArgs(); i++) {
      string rspecifier = po.GetArg(i);
      RandomAccessBaseFloatMatrixReader *rd = new RandomAccessBaseFloatMatrixReader(rspecifier);
      input.push_back(rd);
    }
  
    // Counters for final report 
    int32 num_done = 0, num_no_data = 0, num_bad_length = 0;

    // Get an utternace common to all the streams.
    while(1) {
      string utt = input1.Key();
      bool have_data = true;
      for (int32 i=0; i<input.size(); i++) {
        if(!input[i]->HasKey(utt)) {
          have_data = false;
          KALDI_WARN << "Missing utt " << utt << " in stream " << i+2;
        }
      }
      if(have_data) {
        break; //we can compute output dim...
      } else {
        input1.Next(); 
        num_no_data++;
        if(input1.Done()) {
          KALDI_ERR << "Could not compute output feature dim "
                    << "(no utterance common to all streams)";
        }
      }    
    }
    // Peek dimensions, compute dimension of output features.
    vector<int32> offsets; 
    int32 dim = input1.Value().NumCols(); //dim of input1
    {
      string utt = input1.Key();
      offsets.push_back(0); //offset for first input1 is 0
      for (int32 i=0; i<input.size(); i++) {
        offsets.push_back(dim);
        dim += input[i]->Value(utt).NumCols();
      }
    }
    KALDI_VLOG(1) << "Output dim is " << dim;
   
    // Main loop
    for (; !input1.Done(); input1.Next()) {
      string utt = input1.Key();
      KALDI_VLOG(2) << "Merging " << utt;
 
      // Collect features from streams to vector 'feats'
      vector<Matrix<BaseFloat> > feats;
      feats.push_back(input1.Value());
      bool have_data = true;
      for (int32 i=0; i<input.size(); i++) {
        RandomAccessBaseFloatMatrixReader &inputI = *input[i];
        if (inputI.HasKey(utt)) {
          feats.push_back(inputI.Value(utt));
        } else {
          KALDI_WARN << "Missing utt " << utt << " in stream " << i+2;
          have_data = false; 
        }
      }
      if(!have_data) {
        num_no_data++;
        continue;
      }

      // Check the lenghts
      int32 min_len = feats[0].NumRows(), 
        max_len = feats[0].NumRows();
      for (int32 i=1; i<feats.size(); i++) {
        int32 len = feats[i].NumRows();
        if(len < min_len) min_len = len;
        if(len > max_len) max_len = len;
      }
      if (max_len - min_len > length_tolerance) {
        KALDI_WARN << "Lenth mismatch " << max_len - min_len 
                   << " for utt " << utt 
                   << " is out of tolerance " << length_tolerance;
        num_bad_length++;
        continue;
      }
      if (max_len - min_len > 0) {
        KALDI_VLOG(1) << "Small length mismatch " << max_len - min_len
                      << " for utt " << utt 
                      << " is within tolerance " << length_tolerance
                      << " , trimming the ends";
      }

      // Paste the features
      Matrix<BaseFloat> output(min_len, dim);
      for (int32 i=0; i<feats.size(); i++) {
        SubMatrix<BaseFloat> output_submat = 
          output.ColRange(offsets[i],feats[i].NumCols());
        output_submat.CopyFromMat(feats[i].RowRange(0,min_len));
      }

      // Write...
      kaldi_writer.Write(utt, output);
      num_done++;
    }

    // delete the readers
    for (int32 i=0; i<input.size(); i++) {
      delete input[i];
    }
    input.clear();

    // log
    KALDI_LOG << "Done " << num_done << " utts, " 
              << " (" << num_no_data << " missing data "
              << num_bad_length << " length mismatch)";
   
    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


