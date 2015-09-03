// featbin/process-pitch-feats.cc

// Copyright 2013   Bagher BabaAli
//                  Johns Hopkins University (author: Daniel Povey)
//
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

namespace kaldi {

// For the probability-of-voicing features (the first element of
// each row), change p -> log ((p + 0.0001) / (1.00001 - p))
// This makes it more Gaussian; otherwise it clumps up near
// the edges of the range [0, 1].
void ProcessPovFeatures(Matrix<BaseFloat> *mat) {
  int32 num_frames = mat->NumRows();
  for (int32 i = 0; i < num_frames; i++) {
    BaseFloat p = (*mat)(i, 0);
    KALDI_ASSERT(p >= 0.0 && p <= 1.0);
    (*mat)(i, 0) = Log((p + 0.0001) / (1.0001 - p));
  }
}

void TakeLogOfPitch(Matrix<BaseFloat> *mat) {
  int32 num_frames = mat->NumRows();
  for (int32 i = 0; i < num_frames; i++) {
    KALDI_ASSERT((*mat)(i, 1) > 0.0);
    (*mat)(i, 1) = Log((*mat)(i, 1));
  }
}

// Subtract the moving average over a largish window
// (e.g. 151 frames)
void SubtractMovingAverage(int32 normalization_window_size,
                           Matrix<BaseFloat> *mat) {
  int32 num_frames = mat->NumRows();
  Vector<BaseFloat> temp_pitch(num_frames);
  Matrix<BaseFloat> &features = *mat;
  int32 i;
  for (i = 0; i < num_frames; i++)
    temp_pitch(i) = features(i, 1);

  // Moving Window Normalization
  BaseFloat mean = 0.0;
  int32 mid_win = (normalization_window_size - 1) / 2;
  for (i = 0; (i < num_frames) && (i < normalization_window_size); i++) { 
    mean += features(i, 1);
  }
  mean /= i;

  if (num_frames <= normalization_window_size) {
    for (i = 0; i < num_frames; i++) { 
      features(i, 1) -= mean;
    }
  } else {
    for (i = 0; i <= mid_win; i++) { 
      features(i, 1) -= mean;
    } 
    for (i = (mid_win + 1); i < num_frames; i++) {
      if (i + (mid_win + 1) < num_frames)
        mean -= (temp_pitch(i - (mid_win + 1)) - 
                 temp_pitch(i + (mid_win + 1))) / normalization_window_size; 
      features(i,1) -= mean;
    }    
  }
}

// Set to the moving average over a small window, e.g. 5 frames.
void SetToMovingAverage(int32 average_window_size,
                        Matrix<BaseFloat> *mat) {
  int32 num_frames = mat->NumRows();
  Matrix<BaseFloat> &features = *mat;
  Vector<BaseFloat> temp_pitch(num_frames);
  int32 width = (average_window_size - 1) / 2, i;
  // e.g. if average_window_size is 5, width will equal 2.
        
  for (i = width; i < num_frames - width ; i++) {
    temp_pitch(i) = features(i, 1);
    for(int j = 1; j <= width; ++j) {
      temp_pitch(i) += (features(i - j, 1) + features(i + j, 1));
    }
    temp_pitch(i) /= (2 * width + 1);
  }
  for (i = width; i < num_frames - width; i++)
    features(i, 1) = temp_pitch(i);
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "This is a rather special-purpose program which processes 2-dimensional\n"
        "features consisting of (prob-of-voicing, pitch) into something suitable\n"
        "to put into a speech recognizer.  First use interpolate-feats\n"
        "Usage:  process-pitch-feats [options...] <feats-rspecifier> <feats-wspecifier>\n";

    
    // construct all the global objects
    ParseOptions po(usage);

    int32 normalization_window_size = 151; // should be odd number
    int32 average_window_size = 5;
    
    // Register the options
    po.Register("normalization-window-size",
                &normalization_window_size, "Size of window used for "
                "moving window nomalization (must be odd).");
    po.Register("average-window-size",
                &average_window_size,
                "Size of moving average window (must be odd).");
    
    // parse options (+filling the registered variables)
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    KALDI_ASSERT(average_window_size > 0 && average_window_size % 2 == 1 &&
                 "--average-window-size option must be an odd positive number.");
    KALDI_ASSERT(normalization_window_size > 0 && normalization_window_size % 2 == 1 &&
                 "--normalization-window-size option must be an odd positive number.");
    
    std::string input_rspecifier = po.GetArg(1);
    std::string output_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader reader(input_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.

    if (!kaldi_writer.Open(output_wspecifier))
       KALDI_ERR << "Could not initialize output with wspecifier "
                << output_wspecifier;

    int32 num_done = 0, num_err = 0;

    for (; !reader.Done(); reader.Next()) {
      std::string utt = reader.Key();   
      Matrix<BaseFloat> features = reader.Value();
      int num_frames = features.NumRows();

      if (num_frames == 0 && features.NumCols() != 2) {
        KALDI_WARN << "Feature file has bad size "
                   << features.NumRows() << " by " << features.NumCols();
        num_err++;
        continue;
      }
      
      ProcessPovFeatures(&features);
      TakeLogOfPitch(&features);
      SubtractMovingAverage(normalization_window_size, &features);
      SetToMovingAverage(average_window_size, &features);
      kaldi_writer.Write(utt, features);
      num_done++;
        
      if (num_done % 10 == 0)
        KALDI_LOG << "Processed " << num_done << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
    }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

