// featbin/compute-filters.cc

// Copyright 2016 Pegah Ghahremani


#include "feat/feature-functions.h"
#include "matrix/srfft.h"
#include "matrix/kaldi-matrix-inl.h"    
#include "feat/wave-reader.h"
namespace kaldi {

// This function computes and accumulates  
// dot-product of zero-mean input with its shifted version.
// inner_prod[j] - inner product of input vector with its shifted version by j
//                 sum_{i=0}^corr_window_size x[i] * x[i+j]
// norms[j]      - inner product of shifted input by itself as sum_{i=0}^corr_window_size x[i+j]^2
// lpc_order is the size of autocorr_coeffs.
// If mean == 0, the wave is normalized using its mean, otherwise
// it is mean-normalized using mean value.
void AccStatsForCorrelation(const VectorBase<BaseFloat> &wave,
                                int32 lpc_order, int32 corr_window_size,
                                Vector<BaseFloat> *inner_prod,
                                Vector<BaseFloat> *norms,
                                BaseFloat mean) {
  if (inner_prod->Dim() != lpc_order)
    inner_prod->Resize(lpc_order);

  if (norms->Dim() != lpc_order) 
    norms->Resize(lpc_order);

  Vector<BaseFloat> zero_mean_wave(wave);
  if (mean != 0.0) 
    zero_mean_wave.Add(mean);
   else 
    // subtract mean to normalize wave.
    zero_mean_wave.Add(-wave.Sum() / wave.Dim());

  // append zero et end if the length of shifted version 
  // is less that corr_window_size.
  zero_mean_wave.Resize(lpc_order + corr_window_size, kCopyData);
  BaseFloat e1, e2, sum;
  SubVector<BaseFloat> sub_vec1(zero_mean_wave, 0, corr_window_size);
  e1 = VecVec(sub_vec1, sub_vec1);
  (*norms)(0) += e1;
  (*inner_prod)(0) += e1;
  for (int32 lag = 1; lag < lpc_order; lag++) {
    SubVector<BaseFloat> sub_vec2(zero_mean_wave, lag, corr_window_size);
    e2 = VecVec(sub_vec2, sub_vec2);
    sum = VecVec(sub_vec1, sub_vec2);
    (*inner_prod)(lag) += sum;
    (*norms)(lag) += e2;
  }
}
// Compute autocorrelation coefficients using accumulated inner product
// and norms.
// autocorr[j] - inner_prod[j] / (norms[0] * norms[j])^0.5
// inner_prod[j] - inner product of input vector with its shifted version by j
//                 sum_{i=0}^n x[i] * x[i+j]
// norms[j]      - inner product of shifted input by itself as sum_{i=0}^n x[i+j]^2
void ComputeCorrelation(const VectorBase<BaseFloat> &inner_prod,
                        const VectorBase<BaseFloat> &norms,
                        Vector<BaseFloat> *autocorr) {
  KALDI_ASSERT(inner_prod.Dim() == norms.Dim());
  int32 lpc_order = inner_prod.Dim();
  autocorr->Resize(lpc_order);
  for (int32 lag = 0; lag < lpc_order; lag++) 
    (*autocorr)(lag) = inner_prod(lag) / 
      pow(norms(0) * norms(lag), 0.5);
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
    std::string spk2utt_rspecifier,
      spk_mean_rspecifier;
    bool binary = true, 
      use_mean = false;
    int32 channel = -1,
      lpc_order = 100;
    po.Register("binary", &binary, "write in binary mode (applies only to global filters)");
    po.Register("lpc-order", &lpc_order, "number of LP coefficients used to extract filters.");
    po.Register("use-mean-norm", &use_mean, "If true, autocorrelation computed on"
                " the utterance, which mean normalized using mean computed per speaker." 
                "If false, the utterance is mean-normalized w.r.t its local mean.");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    int32 num_done = 0, num_err = 0;
    std::string wav_rspecifier = po.GetArg(1),
      wspecifier = po.GetArg(2);
    
    BaseFloatVectorWriter writer(wspecifier);
    if (spk2utt_rspecifier != "") { 
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        std::string spk = spk2utt_reader.Key();
        const std::vector<std::string> &uttlist = spk2utt_reader.Value();
        Vector<BaseFloat> inner_prod(lpc_order),
          norms(lpc_order);
        int32 num_chan = wav_reader.Value(uttlist[0]).Data().NumRows();
        std::vector<BaseFloat> spk_sum(num_chan, 0),
          spk_count(num_chan, 0),
          spk_mean(num_chan, 0);
        // Accumulate waveform sum and count over all utterance in speaker
        // to compute mean of waveform for whole speaker over all channels.
        if (use_mean) {
          for (size_t i = 0; i < uttlist.size(); i++) {
            std::string utt = uttlist[i]; 
            const WaveData &wav_data = wav_reader.Value(utt);
            int32 num_chan = wav_data.Data().NumRows();
            for (int32 chan = 0; chan < num_chan; chan++) {
              spk_sum[chan] += wav_data.Data().Row(chan).Sum();
              spk_count[chan] += wav_data.Data().NumCols();
            }
          }
          for (int32 chan = 0; chan < num_chan; chan++) 
           spk_mean[chan] =  spk_sum[chan] / spk_count[chan];
        }

        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!wav_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find wave for utterance " << utt;
            num_err++;
            continue;
          }
          const WaveData &wav_data = wav_reader.Value(utt);
          int32 num_chan = wav_data.Data().NumRows(), this_chan = channel;
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
          SubVector<BaseFloat> waveform(wav_data.Data(), this_chan);
          int32 corr_win_size = waveform.Dim() - lpc_order;
          AccStatsForCorrelation(waveform, lpc_order, corr_win_size, 
                              &inner_prod, &norms, 
                             (use_mean ? 0.0 : spk_mean[this_chan]));
        }
        Vector<BaseFloat> filter, autocorr(lpc_order);
        ComputeCorrelation(inner_prod, norms, &autocorr);
        ComputeFilters(autocorr, &filter); 
        writer.Write(spk, filter); 
      }  
    } else { // assume the input is whole speaker.
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
      for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string spk = wav_reader.Key();
        const WaveData &wave_data = wav_reader.Value();
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
        Vector<BaseFloat> inner_prod(lpc_order), 
          norms(lpc_order), autocorr(lpc_order);
        int32 corr_win_size = waveform.Dim() - lpc_order;
        AccStatsForCorrelation(waveform, lpc_order, 
                               corr_win_size, &inner_prod, &norms, 0.0);
        ComputeCorrelation(inner_prod, norms, &autocorr);
        ComputeFilters(autocorr, &filter);
        writer.Write(spk, filter);
      }
    }
  } catch(const std::exception &e) { 
    std::cerr << e.what();
    return -1;
  }
}
