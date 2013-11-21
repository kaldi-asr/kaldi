// feat/pitch-functions-test.cc

// Copyright    2013  Pegah Ghahremani 

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


#include <iostream>
#include "feat/pitch-functions.cc"
#include "feat/feature-plp.h"
#include "base/kaldi-math.h"
#include "matrix/kaldi-matrix-inl.h"
#include "feat/wave-reader.h"
#include <boost/lexical_cast.hpp>
#include <sys/timeb.h>

using namespace kaldi;

static void UnitTestSimple() {
  std::cout << "=== UnitTestSimple() ===\n";

  Vector<BaseFloat> v(1000);
  Vector<BaseFloat> out;
  Matrix<BaseFloat> m;
  // init with noise
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = (abs( i * 433024253 ) % 65535) - (65535 / 2);
  }
  std::cout << "<<<=== Just make sure it runs... Nothing is compared\n";
  // the parametrization object
  PitchExtractionOptions op;
  // trying to have same opts as baseline.
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  // compute pitch.
  Compute(op, v, &m);
  std::cout << "Test passed :)\n\n";
}

static void UnitTestGetf0Compare1() {
  std::cout << "=== UnitTestGetf0Compare1() ===\n";
  std::ifstream is("test_data/test.wav");
  WaveData wave;  
  wave.Read(is);     
  KALDI_ASSERT(wave.Data().NumRows() == 1);        
  SubVector<BaseFloat> waveform(wave.Data(), 0); 
  // Run the Getf0 pitch features
  Matrix<BaseFloat> getf0_pitch;
  {
    std::ifstream is("test_data/getf0test.pitch");
    //getf0_pitch.Read(is);
  }
  // use pitch code with default configuration..
  PitchExtractionOptions op;  
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  // compute pitch.
  Matrix<BaseFloat> m;
  Compute(op, waveform, &m);    
}

// Compare pitch from Getf0 and Kaldi pitch tracker on KEELE corpora 
static void UnitTestGetf0CompareKeele() {
  std::cout << "=== UnitTestGetf0CompareKeele() ===\n";
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if( i < 6) {
      num = "f"+boost::lexical_cast<std::string>(i)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    } else {
      num = "m"+boost::lexical_cast<std::string>(i-5)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    }
    std::cout << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;  
    wave.Read(is);     
    KALDI_ASSERT(wave.Data().NumRows() == 1);       
    SubVector<BaseFloat> waveform(wave.Data(), 0); 
    // use pitch code with default configuration..
    PitchExtractionOptions op;
    op.frame_opts.window_type = "rectangular";
    op.frame_opts.remove_dc_offset = false;
    op.frame_opts.round_to_power_of_two = false;
    op.frame_opts.samp_freq = 8000;
    op.frame_opts.preemph_coeff = exp(-7000/op.frame_opts.samp_freq);
    // compute pitch.
    Matrix<BaseFloat> m;
    Compute(op, waveform, &m);    
    std::string outfile = "keele/kaldi/"+num+"-kaldi.txt";
    std::ofstream os(outfile.c_str()); 
    m.Write(os, false);
  }
}
/* change freq_weight to investigate the results */
static void UnitTestPenaltyFactor() {
  std::cout << "=== UnitTestPenaltyFactor() ===\n";
  for(int32 k = 1; k < 5; k++) {
    for (int32 i = 1; i < 4; i++) {
      std::string wavefile;
      std::string num;
      if( i < 6) {
        num = "f"+boost::lexical_cast<std::string>(i)+"nw0000";
        wavefile = "keele/resampled/"+num+".wav";
      } else {
        num = "m"+boost::lexical_cast<std::string>(i-5)+"nw0000";
        wavefile = "keele/resampled/"+num+".wav";
      }
      std::cout << "--- " << wavefile << " ---\n";
      std::ifstream is(wavefile.c_str());
      WaveData wave;  
      wave.Read(is);     
      KALDI_ASSERT(wave.Data().NumRows() == 1);        
      SubVector<BaseFloat> waveform(wave.Data(), 0); 
      // use pitch code with default configuration..
      PitchExtractionOptions op;  
      op.penalty_factor = k * 0.05;
      op.frame_opts.dither = 0.0;
      op.frame_opts.preemph_coeff = 0.0;
      op.frame_opts.window_type = "rectangular";
      op.frame_opts.remove_dc_offset = false;
      op.frame_opts.round_to_power_of_two = true;
      op.frame_opts.samp_freq = 8000;
      // compute pitch.
      Matrix<BaseFloat> m;
      Compute(op, waveform, &m);    
      std::string penaltyfactor = boost::lexical_cast<std::string>(k);
      std::string outfile = "keele/freqw/kaldi/"+num+"-kaldi-penalty-"+penaltyfactor+".txt";
      std::ofstream os(outfile.c_str()); 
      m.Write(os, false);
    }
  }
}

static void UnitTestKeeleNccfBallast() {
  std::cout << "=== UnitTestKeeleNccfBallast() ===\n";
  for(int32 k = 1; k < 10; k++) {
    for (int32 i = 1; i < 2; i++) {
      std::string wavefile;
      std::string pgtfile;
      std::string num;
      if( i < 6) {
        num = "f"+boost::lexical_cast<std::string>(i)+"nw0000";
        wavefile = "keele/resampled/"+num+".wav";
        pgtfile =  "keele/ptk/pgt/"+num+"-pgt.txt";
      } else {
        num = "m"+boost::lexical_cast<std::string>(i-5)+"nw0000";
        wavefile = "keele/resampled/"+num+".wav";
        pgtfile =  "keele/ptk/pgt/"+num+"-pgt.txt";
      }
      std::cout << "--- " << wavefile << " ---\n";
      std::ifstream is(wavefile.c_str());
      WaveData wave;  
      wave.Read(is);     
      KALDI_ASSERT(wave.Data().NumRows() == 1);        
      SubVector<BaseFloat> waveform(wave.Data(), 0); 
      // use pitch code with default configuration..
      PitchExtractionOptions op;  
      op.frame_opts.dither = 0.0;
      op.frame_opts.window_type = "rectangular";
      op.frame_opts.remove_dc_offset = false;
      op.frame_opts.round_to_power_of_two = true;
      op.frame_opts.samp_freq = 8000;
      op.frame_opts.preemph_coeff = exp(-7000/op.frame_opts.samp_freq);
      op.nccf_ballast = 0.1 * k;  
      std::cout << " nccf_ballast " << op.nccf_ballast << std::endl;
      // compute pitch.
      Matrix<BaseFloat> m;
      Compute(op, waveform, &m); 
      std::string nccfballast = boost::lexical_cast<std::string>(op.nccf_ballast); 
      std::string outfile = "keele/afact/"+num+"-kaldi-nccf-ballast-"+nccfballast+".txt";
      std::ofstream os(outfile.c_str());
      m.Write(os, false);
    }
  }
}
static void UnitTestVietnamese() {
  std::cout << "=== UnitTestVietnamese() ===\n";
  std::string wavefile, dir, fname;
  dir = "keele/babel/";
  fname = "58357_A_20120507_125021_025089";
  wavefile = dir + fname + ".wav";
  // read the wavefile
  std::cout << "--- " << fname << " ---\n";
  std::ifstream is(wavefile.c_str());
  WaveData wave;  
  wave.Read(is);     
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0); 
  // use pitch code with default configuration..
  PitchExtractionOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.window_type = "rectangular";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = false;
  op.frame_opts.samp_freq = 8000;
  op.frame_opts.preemph_coeff = exp(-7000/op.frame_opts.samp_freq);
  // compute pitch.
  Matrix<BaseFloat> m;
  Compute(op, waveform, &m);    
  std::string outfile = "keele/kaldi/"+fname+".txt";
  std::ofstream os(outfile.c_str()); 
  m.Write(os, false);
}
static void UnitTestResample() {
  std::cout << "== UnitTestResample ===\n";
  std::string wavefile, dir, fname; 
  dir = "keele/babel/";
  fname = "58357_A_20120507_125021_025089";
  wavefile = dir + fname + ".wav";
  // read the wavefile
  std::cout << "--- " << fname << " ---\n";
  std::ifstream is(wavefile.c_str());
  WaveData wave;  
  wave.Read(is);     
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  //lowpass filtering and resampling the wave file  
  double sample_freq = 8000;
  double resample_freq = 4000;
  int32 lowpass_filter_width = 2;
  double lowpass_filter_cutoff = 1500;
  double dt = sample_freq / resample_freq;
  int32 sample_num = wave.Data().NumCols(); 
  int32 resampled_len = static_cast<int>(sample_num/dt);
  std::vector<double> resampled_t(resampled_len); 
  for (int32 i = 0; i < resampled_len; i++) 
    resampled_t[i] = static_cast<double>(i) / resample_freq;
  ArbitraryResample resample(sample_num, sample_freq,
                             lowpass_filter_cutoff, resampled_t,
                             lowpass_filter_width);
  Matrix<double> input_wave(wave.Data());
  Matrix<double> resampled_wave(1, resampled_len);
  resample.Upsample(input_wave, &resampled_wave);
  std::cout << " original_wave \n";   
  for (int32 i = 0; i < 200; i++) 
    std::cout << input_wave(0,i) << " ";  
  std::cout << "\n resampled_wave \n";
  for (int32 i = 0; i < resampled_len; i++)
    std::cout << resampled_wave(0,i) << " ";
  std::cout << " \n resampled_t \n";
  for (int32 i = 0; i < 100; i++)  
    std::cout << resampled_t[i] << " "; 
}
static void UnitTestWeightedMwn1() {
  std::cout << "=== UnitTestWeightedMwn1() ===\n";
  // compare the results of WeightedMwn1 and Sliding CMN with uniform weights.
  for (int32 i = 0; i < 1000; i++) {
    int32 num_frames = 1 + (rand()%10 * 10);
    int32 normalization_win_size = 5 + rand() % 50;
    Matrix<BaseFloat> feat(num_frames, 2),
      output_feat(num_frames, 2);
    feat.SetRandn();
    for (int32 j = 0; j < num_frames; j++)
      feat(j, 0) = 1;
    WeightedMwn(normalization_win_size, feat, &output_feat);
    // SlidingWindow
    SlidingWindowCmnOptions opts;
    opts.cmn_window = normalization_win_size;
    opts.center = true;
    opts.min_window = 1 + rand() % 100;
    if (opts.min_window > opts.cmn_window)  
      opts.min_window = opts.cmn_window; 
    Matrix<BaseFloat> output_feat2(num_frames, 2);
    SlidingWindowCmn(opts, feat, &output_feat2);
    for (int32 j = 0; j < num_frames; j++)
      output_feat(j, 0) = 0.0;
    if ( ! output_feat.ApproxEqual(output_feat2, 0.001)) {
      KALDI_ERR << "Feafures differ " << output_feat << " vs." << output_feat2;
    }
  }
  // Weighted Moving Window Normalization with non-uniform weights
  int32 num_frames = 1 + (rand()%10 * 20);
  int32 normalization_win_size = 5 + rand() % 50;
  Matrix<BaseFloat> feat(num_frames, 2),
    output_feat(num_frames, 2);
  for (int32 j = 0; j < num_frames; j++) {
    int32 r = rand() % 2;
    feat(j,0) = RandUniform()/(1+1000.0*r);
    feat(j,1) = feat(j,1) * feat(j,0);
  }
  ProcessPovFeatures(&feat, 2, true);  
  WeightedMwn(normalization_win_size, feat, &output_feat);
  for (int32 j = 0; j < num_frames; j++)
    std::cout << feat(j, 0) << " " << feat(j, 1) <<  
      " "  << output_feat(j, 1) << std::endl;
}
static void UnitTestWeightedMwn2() {
  // Test WMWN with different window size on vietnamse wavefile
  // initialize using pitch + pov of Vietnamese wavefile
  Matrix<BaseFloat> feats(2, 2);
  std::string fname = "pitch_noMWN";
  std::string infile = "keele/kaldi/"+fname+".txt";
  std::ifstream is(infile.c_str()); 
  feats.Read(is, false, false);
  ProcessPovFeatures(&feats, 2, true);  
  int32 num_frames = feats.NumRows();
  Matrix<BaseFloat> output_feats(num_frames, 2),
    output_feats2(num_frames, 2);
  for (int32 i = 0; i < 9; i++) {   
    int32 normalization_win_size = (i+1)*25;  
    std::string wsize = boost::lexical_cast<std::string>(normalization_win_size); 
    WeightedMwn(normalization_win_size, feats, &output_feats);  
    for (int32 t = 0; t < num_frames; t++) {
      int32 window_begin, window_end; 
      window_begin = t - (normalization_win_size / 2),  
      window_end = window_begin + normalization_win_size;  
      int32 shift = 0;
      if (window_begin < 0) 
        shift = -window_begin;
      else if (window_end > num_frames)
        shift = num_frames - window_end;
      window_end += shift;
      window_begin += shift;
      double sum = 0.0, sum_pov = 0.0;
      for (int32 t2 = window_begin; t2 < window_end; t2++) {
        sum += feats(t2,1) * feats(t2,0);
        sum_pov += feats(t2,0);
      }
      double mean = sum / sum_pov,
      data = feats(t,1), norm_data = data - mean; 
      output_feats2(t,1) = norm_data;
    }
    std::string outfile = "keele/kaldi/"+fname+"-WMWN-ws"+wsize+".txt";  
    std::ofstream os(outfile.c_str());
    output_feats.Write(os,false);
    if ( ! output_feats.ApproxEqual(output_feats, 0.0001)) {
      KALDI_ERR << "Features differ " << output_feats << " vs. " << output_feats2;
    }
  }
}
static void UnitTestTakeLogOfPitch() {
 for (int32 i = 0; i < 100; i++) {
   int num_frame = 50 + (rand() % 200 * 200);
   Matrix<BaseFloat> input(num_frame, 2);
   input.SetRandn();
   input.Scale(100);
   Matrix<BaseFloat> output(input);
   for (int j = 0; j < num_frame; j++) {
     if (input(j,1) < 1 ) { 
       input(j, 1) = 10;
       output(j, 1) = 10;
     }
     output(j, 1) = log(input(j,2));   
   }
   TakeLogOfPitch(&input);
   if ( input.ApproxEqual(output, 0.0001)) {
     KALDI_ERR << " Log of Matrix differs " << input << " vs. " << output;
   }
 }
}
static void UnitTestPitchExtractionSpeed() {
  std::cout << "=== UnitTestPitchExtractionSpeed() ===\n";
  // use pitch code with default configuration..
  PitchExtractionOptions op_fast;
  op_fast.frame_opts.window_type = "rectangular";
  op_fast.frame_opts.remove_dc_offset = false;
  op_fast.frame_opts.round_to_power_of_two = false;
  op_fast.frame_opts.samp_freq = 8000;
  op_fast.frame_opts.preemph_coeff = exp(-7000/op_fast.frame_opts.samp_freq);
  op_fast.lowpass_cutoff = 1000;
  op_fast.max_f0 = 400; 
  for (int32 i = 1; i < 2; i++) {
    std::string wavefile;
    std::string num;
    if( i < 6) {
      num = "f"+boost::lexical_cast<std::string>(i)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    } else {
      num = "m"+boost::lexical_cast<std::string>(i-5)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    }
    std::cout << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;  
    wave.Read(is);     
    KALDI_ASSERT(wave.Data().NumRows() == 1);       
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    // compute pitch.
    int test_num = 10;
    Matrix<BaseFloat> m;
    struct timeb tstruct;
    int tstart = 0, tend = 0;
    double tot_ft = 0;
    // compute time for Pitch Extraction
    ftime( &tstruct );
    tstart = tstruct.time * 1000 + tstruct.millitm;
    for (int32 t = 0; t < test_num; t++)
      Compute(op_fast, waveform, &m);
    ftime( &tstruct );
    tend = tstruct.time * 1000 + tstruct.millitm;
    double tot_real_time = test_num * waveform.Dim() / op_fast.frame_opts.samp_freq;
    tot_ft = (tend - tstart)/tot_real_time;
    std::cout << " Pitch extraction time per second of speech " 
              << tot_ft << " msec " << std::endl;
  }
}
static void UnitTestPitchExtractorCompareKeele() {
  std::cout << "=== UnitTestPitchExtractorCompareKeele() ===\n";
  // use pitch code with default configuration..
  PitchExtractionOptions op;
  op.frame_opts.window_type = "rectangular";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = false;
  op.frame_opts.samp_freq = 8000;
  op.frame_opts.preemph_coeff = exp(-7000/op.frame_opts.samp_freq);
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if( i < 6) {
      num = "f"+boost::lexical_cast<std::string>(i)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    } else {
      num = "m"+boost::lexical_cast<std::string>(i-5)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    }
    std::cout << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;  
    wave.Read(is);     
    KALDI_ASSERT(wave.Data().NumRows() == 1);       
    SubVector<BaseFloat>  waveform(wave.Data(), 0);
    // compute pitch.
    Matrix<BaseFloat> m;
    Compute(op, waveform, &m);
    std::string outfile = "keele/kaldi/"+num+"-speedup-kaldi1.txt";
    std::ofstream os(outfile.c_str()); 
    m.Write(os, false);
  }
}
void UnitTestDiffSampleRate() {
  int sample_rate = 16000; 
  PitchExtractionOptions op_fast;
  op_fast.frame_opts.window_type = "rectangular";
  op_fast.frame_opts.remove_dc_offset = false;
  op_fast.frame_opts.round_to_power_of_two = false;
  op_fast.frame_opts.samp_freq = static_cast<double>(sample_rate);
  op_fast.frame_opts.preemph_coeff = exp(-7000/op_fast.frame_opts.samp_freq);
  op_fast.lowpass_cutoff = 1000;
  op_fast.max_f0 = 400;
  std::string samp_rate = boost::lexical_cast<std::string>(sample_rate/1000);
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if( i < 6) {
      num = "f"+boost::lexical_cast<std::string>(i)+"nw0000";
      //wavefile = "keele/resampled/10kHz/"+num+".wav";
      wavefile = "keele/"+samp_rate+"kHz/"+num+".wav";
    } else {
      num = "m"+boost::lexical_cast<std::string>(i-5)+"nw0000";
      //wavefile = "keele/resampled/10kHz/"+num+".wav";
      wavefile = "keele/"+samp_rate+"kHz/"+num+".wav";
    }
    std::cout << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;  
    wave.Read(is);     
    KALDI_ASSERT(wave.Data().NumRows() == 1);       
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    Matrix<BaseFloat> m;
    Compute(op_fast, waveform, &m);
    std::string outfile = "keele/kaldi/"+num+"-speedup-kaldi-"+samp_rate+"kHz.txt";
    std::ofstream os(outfile.c_str()); 
    m.Write(os, false); 
  } 
}
void UnitTestPostProcess() {
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if( i < 6) {
      num = "f"+boost::lexical_cast<std::string>(i)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    } else {
      num = "m"+boost::lexical_cast<std::string>(i-5)+"nw0000";
      wavefile = "keele/resampled/"+num+".wav";
    }
    std::cout << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;  
    wave.Read(is);     
    KALDI_ASSERT(wave.Data().NumRows() == 1);       
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    PitchExtractionOptions op;
    op.frame_opts.window_type = "rectangular";
    op.frame_opts.remove_dc_offset = false;
    op.frame_opts.round_to_power_of_two = false;
    op.frame_opts.samp_freq = 8000;
    op.frame_opts.preemph_coeff = exp(-7000/op.frame_opts.samp_freq);
    op.lowpass_cutoff = 1000;
    op.max_f0 = 400;
    Matrix<BaseFloat> m, m2;
    Compute(op, waveform, &m);
    PostProcessOptions postprop_op;
    postprop_op.pov_nonlinearity = 2;
    PostProcessPitch(postprop_op, m, &m2);
    std::string outfile = "keele/kaldi/"+num+"-speedup-kaldi-processed.txt";
    std::ofstream os(outfile.c_str()); 
    m2.Write(os, false); 
  } 
}
void UnitTestDeltaPitch() {
  std::cout << "=== UnitTestDeltaPitch() ===\n";
  for (int32 i = 0; i < 1; i++) {
    int32 num_frames = 1 + (rand()%10 * 10);
    Vector<BaseFloat> feat(num_frames),
      output_feat(num_frames), output_feat2(num_frames);
    for (int32 j = 0; j < num_frames; j++)
      feat(j) = 0.2 * j;
    for (int32 j = 2; j < num_frames-2; j++)  
      output_feat2(j) = 1.0 / 10.0  *
        (-2.0 * feat(j-2) - feat(j-1) + feat(j+1) + 2.0 * feat(j+2));
    PostProcessOptions op;  
    ExtractDeltaPitch(op, feat, &output_feat);
    for (int32 j = 0; j < num_frames; j++)
      std::cout << output_feat(j) << " , " << output_feat2(j) << " ";
  }
}
static void UnitTestFeat() {
  UnitTestSimple();
  UnitTestGetf0Compare1();
  //UnitTestGetf0CompareKeele();
  UnitTestPenaltyFactor();
  //UnitTestKeeleNccfBallast();
  //UnitTestVietnamese();
  UnitTestResample();
  UnitTestWeightedMwn1();
  UnitTestWeightedMwn2();
  UnitTestTakeLogOfPitch();
  //UnitTestPitchExtractionSpeed();
  //UnitTestPitchExtractorCompareKeele();
  UnitTestDiffSampleRate();
  UnitTestPostProcess();
  UnitTestDeltaPitch();
}




int main() {
  try {
    for (int i = 0; i < 1; i++)
      UnitTestFeat();
    std::cout << "Tests succeeded.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}


