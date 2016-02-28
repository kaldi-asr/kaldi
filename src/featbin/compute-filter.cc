// featbin/compute-filters.cc

// Copyright 2016 Pegah Ghahremani


#include "feat/feature-functions.h"
#include "matrix/srfft.h"
#include "matrix/kaldi-matrix-inl.h"    
#include "feat/wave-reader.h"
namespace kaldi {

// This function computes correlation as 
// dot-product of zero-mean input with its shifted version
// normalized by their l2_norm.
// corr_coeffs(p) = sum_{i=0}^n wave_chunk(i) * wave_chunk(i+p) / (l2[p] * l2[p])
// l2[p] is l2_norm of wave shifted by p. 
// lpc_order is the size of autocorr_coeffs.
void ComputeCorrelation(const VectorBase<BaseFloat> &wave,
              int32 lpc_order, int32 corr_window_size,
              Vector<BaseFloat> *autocorr) {
  Vector<BaseFloat> zero_mean_wave(wave);
  // subtract mean to normalize wave.
  zero_mean_wave.Add(-wave.Sum() / wave.Dim());
  // append zero et end if the length of shifted version 
  // is less that corr_window_size.
  zero_mean_wave.Resize(lpc_order + corr_window_size, kCopyData);
  BaseFloat e1, e2, sum;
  SubVector<BaseFloat> sub_vec1(zero_mean_wave, 0, corr_window_size);
  e1 = VecVec(sub_vec1, sub_vec1);
  (*autocorr)(0) = 1.0;
  for (int32 lag = 1; lag <= lpc_order; lag++) {
    SubVector<BaseFloat> sub_vec2(zero_mean_wave, lag, corr_window_size);
    e2 = VecVec(sub_vec2, sub_vec2);
    sum = VecVec(sub_vec1, sub_vec2);
    (*autocorr)(lag) = sum / pow(e1 * e2, 0.5);
  }
}

// This function computes coefficients of forward linear predictor
// w.r.t autocorrelation coefficients by minimizing the prediction
// error using MSE.
// Durbin used to compute LP coefficients using autocorrelation coefficients.
// R(j) = sum_{i=1}^P R((i+j % p)] * a[i] j = 0, ..,P
// P is order of Linear prediction 
// lp_coeffs - linear prediction coefficients. (predicted_x[n] = sum_{i=1}^P{a[i] * x[n-i]})
// R(j) is the j_th autocorrelation coefficient.
void ComputeFilters(const VectorBase<BaseFloat> &autocorr, 
                    Vector<BaseFloat> *lp_coeffs) {
  int32 n = autocorr.Dim() - 1;
  lp_coeffs->Resize(n);
  // compute lpc coefficients using autocorrelation coefficients. 
  ComputeLpc(autocorr, lp_coeffs);  
}

} // namescape kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage = 
      "Computes LP coefficients per-speaker, by minimizing "
      "prediction error using MSE.\n"
      "This coefficient contain speaker-dependent information correspond to each speaker.\n"

      "Usage: compute-filters [options] <wave-rspecifier> <filter-rspecifier> \n"
      "e.g.: compute-filters " 
      " scp:data/train/wav.scp ark,scp:filter.ark,filter.scp\n";

    ParseOptions po(usage);
    std::string spk2utt_rspecifier;
    bool binary = true;
    int32 channel = -1,
      lpc_order = 100;
    po.Register("binary", &binary, "write in binary mode (applies only to global filters)");
    po.Register("lpc-order", &lpc_order, "number of LP coefficients used to extract filters.");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    int32 num_done = 0, num_err = 0;
    std::string wav_rspecifier = po.GetArg(1),
      wspecifier = po.GetArg(2);
    
    SequentialTableReader<WaveHolder> spk_reader(wav_rspecifier);

    BaseFloatVectorWriter writer(wspecifier);
      
    for (; !spk_reader.Done(); spk_reader.Next()) {
      std::string spk = spk_reader.Key();
      const WaveData &wave_data = spk_reader.Value();
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      KALDI_ASSERT(num_chan > 0);
      if (channel == -1) { 
        this_chan = 0;
        if (num_chan != 1)
          KALDI_WARN << "Channel not specified but you have data with "
                     << num_chan  << " channels; defaulting to zero";
      } else {
        if (this_chan >= num_chan) {
          KALDI_WARN << "File with id " << spk << " has "
                     << num_chan << " channels but you specified channel "
                     << channel << ", producing no output.";
          continue;
        }
      }
      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Vector<BaseFloat> filter;
      Vector<BaseFloat> autocorr(lpc_order);
      int32 corr_win_size = waveform.Dim();
      ComputeCorrelation(waveform, lpc_order, corr_win_size, &autocorr);
      ComputeFilters(autocorr, &filter);
      writer.Write(spk, filter);
    }
  } catch(const std::exception &e) { 
    std::cerr << e.what();
    return -1;
  }
}
