// feat/inverse-stft.cc

// Copyright 2015  Hakan Erdogan

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


#include "feat/inverse-stft.h"


namespace kaldi {

Istft::Istft(const StftOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts), srfft_(NULL) {

    int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
    if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two
        srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

Istft::~Istft() {
    if (srfft_ != NULL)
        delete srfft_;
}

void Istft::Compute(const Matrix<BaseFloat> &input,
                    Vector<BaseFloat> *wave_out,
                    int32 wav_length) {
    KALDI_ASSERT(wave_out != NULL);

    int32 num_frames = input.NumRows();
    //int32 input_feat_size = input.NumCols();
    int32 window_size = opts_.frame_opts.PaddedWindowSize();
    int32 frame_length = opts_.frame_opts.WindowSize();
    BaseFloat samp_freq = opts_.frame_opts.samp_freq;
    int32 frame_shift_samp = static_cast<int32>(samp_freq * 0.001 * opts_.frame_opts.frame_shift_ms);
    // scale factor required for perfect reconstruction
    BaseFloat scale_factor = static_cast<BaseFloat>(frame_shift_samp) / static_cast<BaseFloat>(feature_window_function_.window.Sum());
    BaseFloat preemph_coeff = opts_.frame_opts.preemph_coeff;

    if (wav_length < 0) wav_length = frame_shift_samp * (num_frames-1) + window_size;

    // Get dimensions of output wav and allocate space
    wave_out->Resize(wav_length); // write single channel, now a vector (early version used a matrix)
    wave_out->SetZero(); // set to zero to initialize overlap-add correctly

    //KALDI_ASSERT(window_size+2 == input_feat_size);
    KALDI_ASSERT(opts_.output_type == "real_and_imaginary" || opts_.output_type == "amplitude_and_phase");

    //int32 Nfft = input.NumCols()-2; // also equal to window_size
    int32 Nfft = window_size;

    // middle frequency range
    int32 start_frq=1; // the one after DC
    int32 end_frq=Nfft/2; // the Nyquist frequency 
    int32 num_frq=Nfft/2+1; // including DC and Nyquist

    if (opts_.cut_dc) {
	start_frq--;
	end_frq--;
	num_frq--;
    }

    if (opts_.cut_nyquist) {
	end_frq--;
	num_frq--;
    }

    // Compute from all the frames, r is frame index..
    for (int32 r = 0; r < num_frames; r++) {

        Vector<BaseFloat> temp(Nfft);

        // convert from layouts to standard fft layout which is as follows
        // DC, Nyquist, RE, IM, RE, IM, ...
        int32 k=2;
        if (opts_.output_layout == "block") {
            if (opts_.output_type == "amplitude_and_phase") {
                if (opts_.cut_dc)
                  temp(0)=0;
                else
                  temp(0)=input(r,0) * std::cos(input(r,num_frq));
                if (opts_.cut_nyquist) 
                  temp(1)=0;
                else 
		  temp(1)=input(r,end_frq) * std::cos(input(r,end_frq+num_frq));
                for (int32 i=start_frq; i<end_frq; i++) { // start with first nonzero freq. at position 0 or 1
                    temp(k++)=input(r,i) * std::cos(input(r,i+num_frq));
                    temp(k++)=input(r,i) * std::sin(input(r,i+num_frq));
                }
            } else {
                if (opts_.cut_dc)
                  temp(0)=0;
                else
                  temp(0)=input(r,0);
                if (opts_.cut_nyquist) 
                  temp(1)=0;
                else 
                  temp(1)=input(r,end_frq);
                for (int32 i=start_frq; i<end_frq; i++) { // start with first nonzero freq. at position 0 or 1
                    temp(k++)=input(r,i);
                    temp(k++)=input(r,i+num_frq);
                }
            }
        } else if (opts_.output_layout == "interleaved") {
            if (opts_.output_type == "amplitude_and_phase") {
                if (opts_.cut_dc)
                  temp(0)=0;
                else
                  temp(0)=input(r,0) * std::cos(input(r,1));
                if (opts_.cut_nyquist) 
                  temp(1)=0;
                else 
                  temp(1)=input(r,2*end_frq) * std::cos(input(r,2*end_frq+1));
                for (int32 i=2*start_frq; i<2*end_frq; i+=2) { // start with first nonzero freq. now at position 2 (due to interlaved)
                    temp(k++)=input(r,i) * std::cos(input(r,i+1));
                    temp(k++)=input(r,i) * std::sin(input(r,i+1));
                }
            } else {
                if (opts_.cut_dc)
                  temp(0)=0;
                else
                  temp(0)=input(r,0);
                if (opts_.cut_nyquist) 
                  temp(1)=0;
                else 
                  temp(1)=input(r,2*end_frq);
                for (int32 i=2*start_frq; i<2*end_frq; i+=2) { // start with first nonzero freq. now at position 2 (due to interleaved)
                    temp(k++)=input(r,i);
                    temp(k++)=input(r,i+1);
                }
            }
        }

        if (srfft_ != NULL)  // Compute inverse FFT using split-radix algorithm.
            srfft_->Compute(temp.Data(), false);
        else  // An alternative algorithm that works for non-powers-of-two
            RealFft(&temp, false);
        temp.Scale(scale_factor/static_cast<BaseFloat>(Nfft)); // inverse fft does not do 1/Nfft
        if (preemph_coeff != 0.0) // please give this as zero if you want perfect reconstruction
           Deemphasize(&temp, preemph_coeff); 
        int32 start = r*frame_shift_samp;
        if (!opts_.frame_opts.snip_edges)
            start = -frame_length/2+frame_shift_samp/2+r*frame_shift_samp;
        OverlapAdd(temp, start, wav_length, wave_out);
    }
} //Istft::Compute
}  // namespace kaldi
