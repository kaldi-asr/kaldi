// featbin/copy-feats-to-htk.cc

// Copyright 2013   Petr Motlicek

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
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-common.h"
#include "matrix/matrix-lib.h"

#include <sys/stat.h>

#if defined(_MSC_VER)
#include <direct.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;


    const char *usage =
        "Save features as HTK files:\n" 
        "Each utterance will be stored as a unique HTK file in a specified directory.\n"
        "The HTK filename will correspond to the utterance-id (key) in the input table, with the specified extension.\n"
        "Usage: copy-feats-to-htk [options] in-rspecifier\n"
        "Example: copy-feats-to-htk --output-dir=/tmp/HTK-features --output-ext=fea  scp:feats.scp\n";

    ParseOptions po(usage);
    std::string dir_out = "./";
    std::string ext_out = "fea";
    int32 sample_period = 100000; // 100ns unit : 10ms = 100000,
    int32 sample_kind = 9; // USER,
    /*
    0 WAVEFORM sampled waveform
    1 LPC linear prediction filter coefficients
    2 LPREFC linear prediction reflection coefficients
    3 LPCEPSTRA LPC cepstral coefficients
    4 LPDELCEP LPC cepstra plus delta coefficients
    5 IREFC LPC reflection coef in 16 bit integer format
    6 MFCC mel-frequency cepstral coefficients
    7 FBANK log mel-filter bank channel outputs
    8 MELSPEC linear mel-filter bank channel outputs
    9 USER user defined sample kind
    10 DISCRETE vector quantised data
    11 PLP PLP cepstral coefficients
    */

    po.Register("output-ext", &ext_out, "Output ext of HTK files");
    po.Register("output-dir", &dir_out, "Output directory");
    po.Register("sample-period", &sample_period, "HTK sampPeriod - sample period in 100ns units");
    po.Register("sample-kind", &sample_kind, "HTK parmKind - a code indicating the sample kind (e.g., 6=MFCC, 7=FBANK, 9=USER, 11=PLP)");



    po.Read(argc, argv);

    //std::cout << "Dir: " << dir_out << " ext: " << ext_out << "\n"; 

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);

    // check or create output dir:
    const char * c = dir_out.c_str();
   if ( access( c, 0 ) != 0 ){
#if defined(_MSC_VER)
       if (_mkdir(c) != 0)
#else
       if (mkdir(c, S_IRWXU|S_IRGRP|S_IXGRP) != 0)
#endif
       KALDI_ERR << "Could not create output directory: " << dir_out;
   /*
    else if (chdir(c) != 0)
       KALDI_ERR << "first chdir() error: " << dir_out;
    else if (chdir("..") != 0)
       KALDI_ERR << "second chdir() error: " << dir_out;
    else if (rmdir(c) != 0)
       KALDI_ERR << "rmdir() error: " << dir_out;
   */
    }


    // HTK parameters
    HtkHeader hdr;
    hdr.mSamplePeriod = sample_period;
    hdr.mSampleKind = sample_kind;

    
    // write to the HTK files
    int32 num_frames, dim, num_done=0;
    SequentialBaseFloatMatrixReader feats_reader(rspecifier);
    for (; !feats_reader.Done(); feats_reader.Next()) {
      std::string utt = feats_reader.Key();
      const Matrix<BaseFloat> &feats = feats_reader.Value();
      num_frames = feats.NumRows(), dim = feats.NumCols();
      //std::cout << "Utt: " << utt<< " Frames: " << num_frames << " Dim: " << dim << "\n";

      hdr.mNSamples = num_frames;
      hdr.mSampleSize = sizeof(float)*dim;

      Matrix<BaseFloat> output(num_frames, dim, kUndefined);
      std::stringstream ss;
      ss << dir_out << "/" << utt << "." << ext_out; 
      output.Range(0, num_frames, 0, dim).CopyFromMat(feats.Range(0, num_frames, 0, dim));    
      std::ofstream os(ss.str().c_str(), std::ios::out|std::ios::binary);
      WriteHtk(os, output, hdr);  
      num_done++;    
    }
    KALDI_LOG << num_done << " HTK feature files generated in the direcory: " << dir_out;
    return (num_done != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


