// gmm/mmie-am-diag-gmm-test.cc

// Copyright 2009-2011  Saarland University
// Author:  Petr Motlicek

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




#include "gmm/mmie-am-diag-gmm.h" 
#include "util/kaldi-io.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/model-test-common.h"


using kaldi::AmDiagGmm;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;


namespace kaldi {



void UnitTestMmieAmDiagGmm() {
  int32 dim = 1 + kaldi::RandInt(0, 9),       // random dimension of the gmm
  num_pdfs = 2; // + kaldi::RandInt(0, 9);        // random number of GMMs(meaning states in HMM)
  int32 num_comp = 1; // + kaldi::RandInt(0, 9);  // random number of mixtures in each GMM
  int32  nMix = num_comp;
 
  /// generate random feature vectors
  Matrix<BaseFloat> means_f(nMix, dim), vars_f(nMix, dim);
  // first, generate random mean and variance vectors
  for (size_t m = 0; m < nMix; m++) {
    for (size_t d= 0; d < dim; d++) {
      means_f(m, d) = kaldi::RandGauss(); //*100.0F;
      vars_f(m, d) = exp(kaldi::RandGauss()); //*1000.0F+ 1.0F;
    }
    //std::cout << "Gauss " << m << ": Mean = " << means_f.Row(m) << '\n'
    //          << "Vars = " << vars_f.Row(m) << '\n';
  }
 

   // Numerator stats
  // generate 1000 feature vectors for each of the mixture components
  size_t counter_num = 0, multiple = 200;
  Matrix<BaseFloat> feats_num(nMix*multiple, dim);
  for (size_t m = 0; m < nMix; m++) {
    for (size_t i = 0; i < multiple; i++) {
      for (size_t d = 0; d < dim; d++) {
        feats_num(counter_num, d) =  kaldi::RandGauss() *
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
        feats_den(counter_den, d) =  kaldi::RandGauss() *
            std::sqrt(vars_f(m, d));
      }
      counter_den++;
    }
  }

  // now generate randomly initial values for the GMM
  Vector<BaseFloat> weights(1);
  Matrix<BaseFloat> means(1, dim), vars(1, dim), invvars(1, dim);
  for (size_t d= 0; d < dim; d++) {
    means(0, d) = kaldi::RandGauss(); //*100.0F;
    vars(0, d) = exp(kaldi::RandGauss()); // *10.0F + 1e-5F;
  }
  weights(0) = 1.0F;
  invvars.CopyFromMat(vars);
  invvars.InvertElements();



  /// ###########################################################################################
  /// Generate normal accumulator
  AccumDiagGmm num;
  AccumDiagGmm den;
  GmmFlagsType flags = kGmmAll;  // Should later try reducing this.

  // new GMM
  DiagGmm *gmm = new DiagGmm();
  gmm->Resize(1, dim);
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(invvars, means);
  gmm->ComputeGconsts();


    Vector<BaseFloat> featvec_num(dim);
    Vector<BaseFloat> featvec_den(dim);
    Vector<BaseFloat> posteriors(dim);
 
    size_t iteration = 0;
    size_t maxiterations = 2;
    MmieDiagGmmOptions config;
    BaseFloat obj, count;
    while (iteration < maxiterations) {
      std::cout << "Iteration :" << iteration << " Num Gauss: " <<  gmm->NumGauss() << '\n';

      num.Resize(gmm->NumGauss(), dim, flags);
      num.SetZero(flags);
      den.Resize(gmm->NumGauss(), dim, flags);
      den.SetZero(flags);

 
     /// get statistics
     double loglike_num = 0.0;
     double loglike_den = 0.0;
     for (size_t i = 0; i < counter_num; i++) {
      /// copy from matrix to vector and update the statistics of numerator
      featvec_num.CopyRowFromMat(feats_num, i);
      loglike_num += static_cast<double>(num.AccumulateFromDiag(*gmm,
        featvec_num, 1.0F));
  
     //std::cout << "Mean accum_num: " <<  num.mean_accumulator() << '\n';
     }
     for (size_t i = 0; i < counter_den; i++) {
      /// copy from matrix to vector and update the statistics of denominator
      featvec_den.CopyRowFromMat(feats_den, i);
      loglike_den += static_cast<double>(den.AccumulateFromDiag(*gmm,
        featvec_den, 1.0F));
      //std::cout << "Mean accum_den: " <<  den.mean_accumulator() << '\n';
     }

     /// get 2 mixtures from 1
     MleDiagGmmUpdate(config, num, flags, gmm, &obj, &count);
     /// Split gaussian
     if (iteration < maxiterations -1) gmm->Split(gmm->NumGauss() * 2, 0.001);

     iteration++;
   }
  

     /// generate set of PDFs AmDiagGmm class
    AmDiagGmm am_gmm;
    for (int32 i = 0; i < num_pdfs; ++i) {
    //    ut::InitRandDiagGmm(dim, num_comp, &gmm);
      Vector<BaseFloat> tmp_mean(dim);
      am_gmm.AddPdf(*gmm);
      am_gmm.GetGaussianMean(i,0,&tmp_mean);
      std::cout << "Mean" << i << ": " << tmp_mean << '\n';
    } 
    std::cout << "NumPdfs: " << am_gmm.NumPdfs() << '\n';
  
     /// Create from AccumDiag AccumAmDiag, write it to the file - Numerator
    // non-binary write    
    {
    Output o("tmp_am_num_stats",false);
    o.Stream() << "<NUMPDFS> 2 ";
    num.Write(o.Stream(),false);
    num.Write(o.Stream(),false);
    //num.Write(Output("tmp_num_stats", false).Stream(), false);
    }

     /// Create from AccumDiag AccumAmDiag, write it to the file - Denominator 
    {
    //den.Write(Output("tmp_den_stats", false).Stream(), false);
    Output o("tmp_am_den_stats",false);
    o.Stream() << "<NUMPDFS> 2 ";
    den.Write(o.Stream(),false);
    den.Write(o.Stream(),false);
    }

    ///MMieAccumAmDiag - read Num and Den
    MmieAccumAmDiagGmm mmi_am_accs;
     // non-binary read
    bool binary_in;
    {
    Input ki("tmp_am_num_stats", &binary_in); //1.acc
    mmi_am_accs.ReadNum(ki.Stream(), binary_in, false); // false = not adding.
    std::cout << "Num of mmi_am_accs read Numerator: " << mmi_am_accs.NumAccs() << '\n'; 
    }
    {
    Input ki("tmp_am_den_stats", &binary_in); //1.acc
    mmi_am_accs.ReadDen(ki.Stream(), binary_in, false); // false = not adding.
    std::cout << "Num of mmi_am_accs read Denominator: " << mmi_am_accs.NumAccs() << '\n'; 
    }
  
    /// check some of the accumulators of Numerator
    /// do all only for first accumulator
    /// Scaling:
    std::cout << "Scaling: " << '\n';
    std::cout << "Mean accumulator 1 before: " << mmi_am_accs.GetNumAcc(1).mean_accumulator() << '\n';
    mmi_am_accs.GetNumAcc(1).Scale(1.1, flags);
    std::cout << "Mean accumulator 1 after: " << mmi_am_accs.GetNumAcc(1).mean_accumulator() << '\n';


    /// update all PDFs in AmDiagGmm
    std::cout << "Gmm update: " << '\n';
    Vector<BaseFloat> tmp_mean(dim);
    am_gmm.GetGaussianMean(0,0,&tmp_mean);
    std::cout << "Mean of 1st Gmm before: " << tmp_mean << '\n';

    MmieAmDiagGmmUpdate(config, mmi_am_accs, flags, &am_gmm, &obj, &count);
    am_gmm.GetGaussianMean(0,0,&tmp_mean);
    std::cout << "Mean of 1st Gmm after: " << tmp_mean << '\n';


}


}  // end namespace kaldimu
	

int main() {
  for (int i = 0; i < 1; ++i)
    kaldi::UnitTestMmieAmDiagGmm();
  std::cout << "Test OK.\n";
  return 0;
}
