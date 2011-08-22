// gmm/mmie-diag-gmm-test.cc

// Copyright 2009-2011  Petr Motlicek

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



#include "gmm/diag-gmm.h"
#include "gmm/mmie-diag-gmm.h" 
#include "util/kaldi-io.h"


namespace kaldi {


void UnitTestEstimateMmieDiagGmm() {
  size_t dim = 15;  // dimension of the gmm
  size_t nMix = 2;  // number of mixtures in the data
  size_t maxiterations = 20;  // number of iterations for estimation

  // maximum number of densities in the GMM
  // larger than the number of mixtures in the data
  // so that we can test the removal of unseen components
  int32 maxcomponents = 10;

  // generate random feature vectors
  Matrix<BaseFloat> means_f(nMix, dim), vars_f(nMix, dim);
  // first, generate random mean and variance vectors
  for (size_t m = 0; m < nMix; m++) {
    for (size_t d= 0; d < dim; d++) {
      means_f(m, d) = kaldi::RandGauss()*100.0F;
      vars_f(m, d) = exp(kaldi::RandGauss())*1000.0F+ 1.0F;
    }
    //std::cout << "Gauss " << m << ": Mean = " << means_f.Row(m) << '\n'
    //          << "Vars = " << vars_f.Row(m) << '\n';
  }
   
   // Numerator stats
  // second, generate 1000 feature vectors for each of the mixture components
  size_t counter_num = 0, multiple = 200;
  Matrix<BaseFloat> feats_num(nMix*multiple, dim);
  for (size_t m = 0; m < nMix; m++) {
    for (size_t i = 0; i < multiple; i++) {
      for (size_t d = 0; d < dim; d++) {
        feats_num(counter_num, d) = means_f(m, d) + kaldi::RandGauss() *
            std::sqrt(vars_f(m, d));
      }
      counter_num++;
    }
  }

  // Denominator stats
  // second, generate 1000 feature vectors for each of the mixture components
  size_t counter_den = 0;
  Matrix<BaseFloat> feats_den(nMix*multiple, dim);
  for (size_t m = 0; m < nMix; m++) {
    for (size_t i = 0; i < multiple; i++) {
      for (size_t d = 0; d < dim; d++) {
        feats_den(counter_den, d) = means_f(m, d) + kaldi::RandGauss() *
            std::sqrt(vars_f(m, d));
      }
      counter_den++;
    }
  }

  // Compute the global mean and variance
  Vector<BaseFloat> mean_acc(dim);
  Vector<BaseFloat> var_acc(dim);
  Vector<BaseFloat> featvec(dim);
  for (size_t i = 0; i < counter_num; i++) {
    featvec.CopyRowFromMat(feats_num, i);
    mean_acc.AddVec(1.0, featvec);
    featvec.ApplyPow(2.0);
    var_acc.AddVec(1.0, featvec);
  }
  mean_acc.Scale(1.0F/counter_num);
  var_acc.Scale(1.0F/counter_num);
  var_acc.AddVec2(-1.0, mean_acc);
  //std::cout << "Mean acc = " << mean_acc << '\n' << "Var acc = "
  //         << var_acc << '\n';

   //write the feature vectors to a file
   std::ofstream of("tmpfeats");
   of.precision(10);
   of << feats_num; 
   of.close();

  // now generate randomly initial values for the GMM
  Vector<BaseFloat> weights(1);
  Matrix<BaseFloat> means(1, dim), vars(1, dim), invvars(1, dim);
  for (size_t d= 0; d < dim; d++) {
    means(0, d) = kaldi::RandGauss()*100.0F;
    vars(0, d) = exp(kaldi::RandGauss()) *10.0F + 1e-5F;
  }
  weights(0) = 1.0F;
  invvars.CopyFromMat(vars);
  invvars.InvertElements();

  // new GMM
  DiagGmm *gmm = new DiagGmm();
  gmm->Resize(1, dim);
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(invvars, means);
  gmm->ComputeGconsts();


  MmieDiagGmm mmie_gmm;

  MmieDiagGmmOptions  config;
  config.min_variance = 0.01;
  GmmFlagsType flags = kGmmAll;  // Should later try reducing this.
  
  AccumDiagGmm num;
  AccumDiagGmm den;

  num.Resize(gmm->NumGauss(), gmm->Dim(), flags);
  num.SetZero(flags);
  den.Resize(gmm->NumGauss(), gmm->Dim(), flags);
  den.SetZero(flags);
    
  mmie_gmm.Resize(gmm->NumGauss(), gmm->Dim(), flags);


// iterate
  size_t iteration = 0;
  float lastloglike = 0.0;
  int32 lastloglike_nM = 0;

  while (iteration < maxiterations) {
    Vector<BaseFloat> featvec_num(dim);
    Vector<BaseFloat> featvec_den(dim);
    num.Resize(gmm->NumGauss(), gmm->Dim(), flags);
    num.SetZero(flags);
    den.Resize(gmm->NumGauss(), gmm->Dim(), flags);
    den.SetZero(flags);
    mmie_gmm.Resize(gmm->NumGauss(), gmm->Dim(), flags);
  
  
    double loglike_num = 0.0;
    double loglike_den = 0.0;
    for (size_t i = 0; i < counter_num; i++) {
      featvec_num.CopyRowFromMat(feats_num, i);
      loglike_num += static_cast<double>(num.AccumulateFromDiag(*gmm,
        featvec_num, 1.0F));
      //std::cout << "Mean accum_num: " <<  num.mean_accumulator() << '\n';
    }
    for (size_t i = 0; i < counter_den; i++) {
      featvec_den.CopyRowFromMat(feats_den, i);
      loglike_den += static_cast<double>(den.AccumulateFromDiag(*gmm,
        featvec_den, 1.0F));
      //std::cout << "Mean accum_den: " <<  den.mean_accumulator() << '\n';
    }

    std::cout << "Loglikelihood Num before iteration " << iteration << " : "
        << std::scientific << loglike_num << " number of components: "
        << gmm->NumGauss() << '\n';

    std::cout << "Loglikelihood Den before iteration " << iteration << " : "
        << std::scientific << loglike_den << " number of components: "
        << gmm->NumGauss() << '\n';

  
   mmie_gmm.SubtractAccumulatorsISmoothing(num, den, config);
   BaseFloat obj, count;
   //Vector<double> mean_hlp(dim);
   //mean_hlp.CopyFromVec(gmm->means_invvars().Row(0));
   //std::cout << "MEANX: " << mean_hlp << '\n'; 
   std::cout << "MEANX: " << gmm->weights() << '\n'; 

   mmie_gmm.Update(config, flags, gmm, &obj, &count);
   //mean_hlp.CopyFromVec(gmm->means_invvars().Row(0));
   //std::cout << "MEANY: " << mean_hlp << '\n'; 
   std::cout << "MEANY: " << gmm->weights() << '\n'; 


   if ((iteration % 3 == 1) && (gmm->NumGauss() * 2 <= maxcomponents)) {
      gmm->Split(gmm->NumGauss() * 2, 0.001);
      std::cout << "Ngauss, Ndim: " << gmm->NumGauss() << " " << gmm->Dim() << '\n'; 
   
   }


   iteration++;
  }
}

}  // end namespace kaldi


int main() {
  // repeat the test 5 times
  for (int i = 0; i < 5; ++i) {
    kaldi::UnitTestEstimateMmieDiagGmm();
  }
  std::cout << "Test OK.\n";
}