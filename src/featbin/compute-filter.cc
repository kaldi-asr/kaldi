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
  Vector<BaseFloat> l2_norms; // l2_norms[j] - inner product of shifted input by itself as
                             // sum_{i=0}^corr_window_size x[i+j]^2
  Vector<BaseFloat> inner_prod; // inner product of input vector with its shifted version by j
                                // sum_{i=0}^corr_window_size x[i] * x[i+j]
  CorrelationStats(): corr_order(100), num_samp(0), samp_sum(0) { 
    l2_norms.Resize(corr_order);
    inner_prod.Resize(corr_order);}
    
  CorrelationStats(int32 corr_order): corr_order(corr_order), 
    num_samp(0), samp_sum(0) {
    l2_norms.Resize(corr_order);
    inner_prod.Resize(corr_order); }
};

/*
   This function computes and accumulates statistics 
   required for computing auto-correlation coefficient using waveform "wave",
   e.g dot-product of input with its shifted version.
   inner_prod[j] - inner product of input vector with its shifted version by j
                  sum_{i=0}^corr_window_size x[i] * x[i+j]
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
  Vector<BaseFloat> norm_wave(wave);
  SubVector<BaseFloat> sub_vec1(norm_wave, 0, corr_window_size);
  BaseFloat local_l2_norm = VecVec(sub_vec1, sub_vec1), sum;

  acc_corr_stats->inner_prod(0) += local_l2_norm;

  for (int32 lag = 1; lag < lpc_order; lag++) {
    SubVector<BaseFloat> sub_vec2(norm_wave, lag, corr_window_size);
    int32 last_ind = corr_window_size + lag - 1;
    local_l2_norm += (wave(last_ind) * wave(last_ind) - 
      wave(lag - 1) * wave(lag - 1));
    sum = VecVec(sub_vec1, sub_vec2);
    acc_corr_stats->inner_prod(lag) += sum;
    acc_corr_stats->l2_norms(lag) += local_l2_norm;
  }
}
/*
  Compute autocorrelation coefficients using accumulated unnormalized statistics
  such as inner product and l2 norms.
  inner product and l2_norms are normalized using mean as E[x].
  autocorr[j] - sum_{i=0}^n ([x[i] - E[x]) * (x[i+j] - E(x)) /
    [(sum_{i=0}^n ([x[i] - E[x])^2) * (sum_{i=0}^n ([x[i+j] - E[x])^2)]^0.5
  autocorr[j] - inner_prod[j] / (norms[0] * norms[j])^0.5
  inner_prod[j] - inner product of input vector with its shifted version by j
                   sum_{i=0}^n x[i] * x[i+j]
   l2_norms[j]      - inner product of shifted input by itself as sum_{i=0}^n x[i+j]^2
*/
void ComputeCorrelation(const CorrelationStats &acc_corr_stats,
                        Vector<BaseFloat> *autocorr) {

  KALDI_ASSERT(acc_corr_stats.inner_prod.Dim() == acc_corr_stats.l2_norms.Dim());

  int32 lpc_order = acc_corr_stats.inner_prod.Dim();
  autocorr->Resize(lpc_order);
  for (int32 lag = 0; lag < lpc_order; lag++)
    (*autocorr)(lag) = acc_corr_stats.inner_prod(lag);

  // scale outocorrelation between 0 and 1 using autocorr(0)
  autocorr->Scale(1.0 / (*autocorr)(0));
  
}
/*
   Durbin's recursion - converts autocorrelation coefficients to the LPC
   pTmp - temporal place [n]
   pAC - autocorrelation coefficients [n + 1]
   pLP - linear prediction coefficients [n] (predicted_sn = sum_1^P{a[i] * s[n-i]}})
*/
double DurbinInternal(int32 n, double *pAC, double *pLP, double *pTmp) {
  double ki;                // reflection coefficient

  // we add this bias term to pAC[0].
  // Adding bias term is equivalent to t = teoplitz(pAC) + diag(bias)
  // which is like shifting eigenvalues of teoplitz(pAC) using bias term
  // and we can make sure t is convertible.
  double durbin_bias = 1e-2;
  int32 max_repeats = 20;

  double E = pAC[0];
  pLP[0] = 1.0;
  for (int32 i = 1; i <= n; i++) {
    // next reflection coefficient
    ki = pAC[i];
    for (int32 j = 1; j < i; j++)
      ki += pLP[j] * pAC[i - j];
    ki = ki / E;

    if (abs(ki) > 1) {
      int32 num_repeats = int((pAC[0] - 1.0) / durbin_bias);
      KALDI_LOG << " warning: In Durbin algorithm, abs(ki) > 1 "
                << " for iteration = " << i 
                << " ki = " << ki
                << " autocorr[0] = " << pAC[0]
                << " num_repeats = " << num_repeats
                << "; the bias added";
      pAC[0] += durbin_bias;
      if (num_repeats < max_repeats)
        return -1;
    }
    // new error
    double c = 1 - ki * ki;
    if (c < 1.0e-5) // remove NaNs for constan signal 
      c = 1.0e-5;
   
    E *= c;
    // new LP coefficients
    pTmp[i] = -ki;
    for (int32 j = 1; j < i; j++)
      pTmp[j] = pLP[j] - ki * pLP[i - j];

    for (int32 j = 1; j <= i; j++)
      pLP[j] = pTmp[j];
  }
  return E;
}
/*
  This function computes coefficients of forward linear predictor
   w.r.t autocorrelation coefficients by minimizing the prediction
   error using MSE.
   Durbin used to compute LP coefficients using autocorrelation coefficients.
   R(j) = sum_{i=1}^P R((i+j % p)] * a[i] j = 0, ..,P
   P is order of Linear prediction
   lp_filter - [1, -a[1], -a[2] ... ,-a[P]]
   where a[i] are linear prediction coefficients. (predicted_x[n] = sum_{i=1}^P{a[i] * x[n-i]})
   x[n] - predicted_x[n] = sum_{i = 0}^P { lp_filter[i] .* x[n] }
                         = conv(x, lp_filter)
   R(j) is the j_th autocorrelation coefficient.
*/
void ComputeFilters(const VectorBase<BaseFloat> &autocorr, 
                    Vector<BaseFloat> *lp_filter) {
  int32 n = autocorr.Dim();
  lp_filter->Resize(n);
  // compute lpc coefficients using autocorrelation coefficients
  // with Durbin algorithm
  Vector<double> d_autocorr(autocorr),
   d_lpc_coeffs(n), d_tmp(n);

  KALDI_LOG << "compute lpc using correlations ";
  while (DurbinInternal(n, d_autocorr.Data(),
                 d_lpc_coeffs.Data(),
                 d_tmp.Data()) < 0);
  lp_filter->CopyFromVec(d_lpc_coeffs);
  if (KALDI_ISNAN(lp_filter->Sum())) {
    KALDI_WARN << "NaN encountered in lpc coefficients derived from Durbin algorithm.";
    lp_filter->Set(0.0);
    (*lp_filter)(0) = 1.0;
  }

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
          Vector<BaseFloat> waveform(wav_data.Data().Row(this_chan));
          waveform.Scale(1.0 / (1 << 15));
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
        Vector<BaseFloat> waveform(wave_data.Data().Row(this_chan));
        Vector<BaseFloat> autocorr, filter;
        waveform.Scale(1.0 / (1 << 15));
        KALDI_ASSERT(waveform.Max() <=1 && waveform.Min() >= -1);
        CorrelationStats acc_corr_stats(lpc_order);

        AccStatsForCorrelation(waveform, lpc_order,
                               &acc_corr_stats);
        ComputeCorrelation(acc_corr_stats,
                           &autocorr);
        //KALDI_LOG << "autocorr = " << autocorr;
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
