// feat/feature-stft.cc

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


#include "feat/feature-stft.h"


namespace kaldi {

Stft::Stft(const StftOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts), srfft_(NULL) {

    int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
    if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two
        srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

Stft::~Stft() {
    if (srfft_ != NULL)
        delete srfft_;
}

void Stft::Compute(const VectorBase<BaseFloat> &wave,
                   Matrix<BaseFloat> *output,
                   Vector<BaseFloat> *wave_remainder) {
    KALDI_ASSERT(output != NULL);

    // Get dimensions of output features
    int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
    int32 cols_out = opts_.frame_opts.PaddedWindowSize()+2;
    int32 start_frq = 0;
    int32 end_frq = opts_.frame_opts.PaddedWindowSize()/2+1; // after Nyquist, not inclusive

    if (opts_.cut_dc) {
        cols_out -= 2;
        start_frq = 1;
    }

    if (opts_.cut_nyquist) {
        cols_out -= 2;
        end_frq = end_frq-1;
    }
    
    if (opts_.output_type == "amplitude" || opts_.output_type == "phase")
        cols_out /= 2;

    if (rows_out == 0)
        KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
    // Prepare the output buffer
    output->Resize(rows_out, cols_out);

    // Optionally extract the remainder for further processing
    if (wave_remainder != NULL)
        ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);

    // Buffers
    Vector<BaseFloat> window;  // windowed waveform.

    // Compute all the frames, r is frame index..
    for (int32 r = 0; r < rows_out; r++) {
        // Cut the window, apply window function
        ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                      &window, NULL);
        // Compute energy after window function (not the raw one)

        if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
            srfft_->Compute(window.Data(), true);
        else  // An alternative algorithm that works for non-powers-of-two
            RealFft(&window, true);

        // Copy FFT vector of size N directly to output
        // order of FFT results is: X[0] X[N/2] Re(X[1]) Im(X[1]) Re(X[2]) Im(X[2]) ... Re(X[N/2-1]) Im(X[N/2-1])

        Vector<BaseFloat> spectrum1(window.Dim()/2+1); // real part
        Vector<BaseFloat> spectrum2(window.Dim()/2+1); // imaginary part

        spectrum1(0)=window(0); // DC frequency
        spectrum2(0)=0;
        int32 k=1;
        for (int32 i=2; i < window.Dim(); i+=2) {
            spectrum1(k)=window(i);
            spectrum2(k++)=window(i+1);
        }
        spectrum1(k)=window(1);  // Nyquist frequency
        spectrum2(k)=0;

        if (opts_.output_type == "amplitude_and_phase") {
            for (int32 i=0; i < spectrum1.Dim(); i++) {
                BaseFloat mag=std::sqrt(spectrum1(i)*spectrum1(i)+spectrum2(i)*spectrum2(i));
                BaseFloat phase=std::atan2(spectrum2(i),spectrum1(i));
                spectrum1(i)=mag;
                spectrum2(i)=phase;
            }
        } else if (opts_.output_type == "amplitude") {
            for (int32 i=0; i<spectrum1.Dim(); i++) {
                BaseFloat mag=std::sqrt(spectrum1(i)*spectrum1(i)+spectrum2(i)*spectrum2(i));
                spectrum1(i)=mag;
            }
        } else if (opts_.output_type == "phase") {
            for (int32 i=0; i<spectrum1.Dim(); i++) {
                BaseFloat phase=std::atan2(spectrum2(i),spectrum1(i));
                spectrum2(i)=phase;
            }
        }

        // start forming output
        Vector<BaseFloat> temp(cols_out);
        int32 kk=0;

        if (opts_.output_type == "amplitude" ) {
            for (int32 i=start_frq; i< end_frq; i++)
                temp(kk++)=spectrum1(i);
        } else if (opts_.output_type == "phase" ) {
            for (int32 i=start_frq; i< end_frq; i++)
                temp(kk++)=spectrum2(i);
        } else if (opts_.output_layout == "block" ) {
            for (int32 i=start_frq; i< end_frq; i++)
                temp(kk++)=spectrum1(i);
            for (int32 i=start_frq; i< end_frq; i++)
                temp(kk++)=spectrum2(i);
        } else if (opts_.output_layout == "interleaved" ) {
            for (int32 i=start_frq; i< end_frq; i++) {
                temp(kk++)=spectrum1(i);
                temp(kk++)=spectrum2(i);
            }
        } else {
            KALDI_ERR << "output_type or output_layout wrong!!!";
        }

        // Output buffers
        SubVector<BaseFloat> this_output(output->Row(r));
        this_output.CopyFromVec(temp);
    } //for loop
} //Stft::Compute

}  // namespace kaldi
