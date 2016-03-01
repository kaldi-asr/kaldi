// featbin/compute-filters.cc

// Copyright 2016 Pegah Ghahremani


#include "feat/feature-functions.h"
#include "matrix/srfft.h"
#include "matrix/kaldi-matrix-inl.h"    
#include "feat/wave-reader.h"
namespace kaldi {

// struct used to store statistics required for 
// computing correlation coefficients 
struct CorrelationStats {
  int32 corr_order; // number of correlation coefficient. R[0],..,R[corr_order - 1] 
  int32 num_samp;   // number of samples
  BaseFloat samp_sum; // sum of samples.
  Vector<BaseFloat> l1_norms; // l1_norms[j] - sum of values of input shifted by j as 
                             // sum_{i=0}^corr_window_size x[i+j]
  Vector<BaseFloat> l2_norms; // l2_norms[j] - inner product of shifted input by itself as
                             // sum_{i=0}^corr_window_size x[i+j]^2
  Vector<BaseFloat> inner_prod; // inner product of input vector with its shifted version by j
                                // sum_{i=0}^corr_window_size x[i] * x[i+j]
  CorrelationStats(): corr_order(100), num_samp(0), samp_sum(0) { 
    l1_norms.Resize(corr_order);
    l2_norms.Resize(corr_order);
    inner_prod.Resize(corr_order);}
    
  CorrelationStats(int32 corr_order): corr_order(corr_order), num_samp(0), samp_sum(0) {
    l1_norms.Resize(corr_order);
    l2_norms.Resize(corr_order);
    inner_prod.Resize(corr_order); }
};
/*
   This function computes and accumulates statistics 
   required for computing auto-correlation coefficient using waveform "wave",
   e.g dot-product of input with its shifted version.
   inner_prod[j] - inner product of input vector with its shifted version by j
                  sum_{i=0}^corr_window_size x[i] * x[i+j]
   l1_norms[j]     - sum of values of input shifted by j as sum_{i=0}^corr_window_size x[i+j]
   l2_norms[j]      - inner product of shifted input by itself as 
                      sum_{i=0}^corr_window_size x[i+j]^2
   lpc_order is the size of autocorr_coeffs.
*/
void AccStatsForCorrelation(const VectorBase<BaseFloat> &wave,
                            int32 lpc_order,
                            CorrelationStats *acc_corr_stats) { 
  KALDI_ASSERT(acc_corr_stats->inner_prod.Dim() == lpc_order);
  acc_corr_stats->samp_sum += wave.Sum();
  acc_corr_stats->num_samp += wave.Dim();

  int32 corr_window_size = wave.Dim() - lpc_order;
  SubVector<BaseFloat> sub_vec1(wave, 0, corr_window_size);
  BaseFloat local_l1_norm = sub_vec1.Sum(), 
    local_l2_norm = VecVec(sub_vec1, sub_vec1), sum;

  acc_corr_stats->l1_norms(0) += local_l1_norm;
  acc_corr_stats->l2_norms(0) += local_l2_norm;
  acc_corr_stats->inner_prod(0) += local_l2_norm;

  for (int32 lag = 1; lag < lpc_order; lag++) {
    SubVector<BaseFloat> sub_vec2(wave, lag, corr_window_size);
    int32 last_ind = corr_window_size + lag - 1;
    local_l1_norm += wave(last_ind) - wave(lag - 1);
    local_l2_norm += wave(last_ind) * wave(last_ind) - 
      wave(lag-1) * wave(lag - 1);
    sum = VecVec(sub_vec1, sub_vec2);
    acc_corr_stats->inner_prod(lag) += sum;
    acc_corr_stats->l1_norms(lag) += local_l1_norm;
    acc_corr_stats->l2_norms(lag) += local_l2_norm;
  }
}
/*
  Compute autocorrelation coefficients using accumulated unnormalized statistics
  such as inner product and l1 and l2 norms.
  inner product and l2_norms are normalized using mean as E[x].
  autocorr[j] - sum_{i=0}^n ([x[i] - E[x]) * (x[i+j] - E(x)) /
    [(sum_{i=0}^n ([x[i] - E[x])^2) * (sum_{i=0}^n ([x[i+j] - E[x])^2)]^0.5
  autocorr[j] - inner_prod[j] / (norms[0] * norms[j])^0.5
  inner_prod[j] - inner product of input vector with its shifted version by j
                   sum_{i=0}^n x[i] * x[i+j]
   l1_norms[j]     - sum of values of input shifted by j as sum_{i=0}^corr_window_size x[i+j]
   l2_norms[j]      - inner product of shifted input by itself as sum_{i=0}^n x[i+j]^2
*/
void ComputeCorrelation(const CorrelationStats &acc_corr_stats,
                        Vector<BaseFloat> *autocorr) {

  KALDI_ASSERT(acc_corr_stats.inner_prod.Dim() == acc_corr_stats.l2_norms.Dim() &&
               acc_corr_stats.inner_prod.Dim() == acc_corr_stats.l1_norms.Dim());

  int32 lpc_order = acc_corr_stats.inner_prod.Dim();
  autocorr->Resize(lpc_order);
  BaseFloat mean = acc_corr_stats.samp_sum / acc_corr_stats.num_samp;
  // If mean is nonzero, l2_norm and inner_prod modified
  // to be l2_norm and inner_prod computed on mean-normalized wave.
  // mean_norm_inner_prod[j] = sum_{i=0}^n (x[i] x[i+j] - 
  //                           E[x] * (x[i] + x[i+j]) + E[x]^2)
  // mean_norm_l2_norm[j] = sum_{i=0}^n (x[i+j] - E[x])^2
  Vector<BaseFloat> mean_norm_inner_prod(acc_corr_stats.inner_prod),
    mean_norm_l2_norms(acc_corr_stats.l2_norms);

  if (mean != 0) {
    for (int32 lag = 0; lag < lpc_order; lag++) {
    mean_norm_inner_prod(lag) +=  
       -mean * (acc_corr_stats.l1_norms(0) + acc_corr_stats.l1_norms(lag)) 
       + mean * mean;
    mean_norm_l2_norms(lag) +=  
      -2 * mean * acc_corr_stats.l1_norms(lag) + mean * mean;
    }
  }
  
  for (int32 lag = 0; lag < lpc_order; lag++) 
    (*autocorr)(lag) = mean_norm_inner_prod(lag) / 
      pow(mean_norm_l2_norms(0) * mean_norm_l2_norms(lag), 0.5);
}

/*
   This function computes coefficients of forward linear predictor
   w.r.t autocorrelation coefficients by minimizing the prediction
   error using MSE.
   Durbin used to compute LP coefficients using autocorrelation coefficients.
   R(j) = sum_{i=1}^P R((i+j % p)] * a[i] j = 0, ..,P
   P is order of Linear prediction 
   lp_coeffs - linear prediction coefficients. (predicted_x[n] = sum_{i=1}^P{a[i] * x[n-i]})
   R(j) is the j_th autocorrelation coefficient.
*/
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
        CorrelationStats acc_corr_stats(lpc_order);
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
              num_err++;
              continue;
            }
          }
          SubVector<BaseFloat> waveform(wav_data.Data(), this_chan);
          AccStatsForCorrelation(waveform, lpc_order,
                                 &acc_corr_stats);
        }
        Vector<BaseFloat> filter, autocorr(lpc_order);
        ComputeCorrelation(acc_corr_stats,
                           &autocorr);
        ComputeFilters(autocorr, &filter); 
        writer.Write(spk, filter);
        num_done++;
      }  
    } else { // assume the input waveform is per-speaker.
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
            num_err++;
            continue;
          }
        }
        SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
        Vector<BaseFloat> autocorr, filter;
        CorrelationStats acc_corr_stats(lpc_order);
        AccStatsForCorrelation(waveform, lpc_order,
                               &acc_corr_stats);

        ComputeCorrelation(acc_corr_stats,
                           &autocorr);
        ComputeFilters(autocorr, &filter);
        writer.Write(spk, filter);
        num_done++;
      }
    }
    KALDI_LOG << "Done " << num_done << " speakers, " << num_err
              << " with errors."; 
    return (num_done != 0 ? 0 : 1); 
  } catch(const std::exception &e) { 
    std::cerr << e.what();
    return -1;
  }
}
