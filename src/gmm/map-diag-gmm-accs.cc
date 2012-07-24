// gmm/map-diag-gmm-accs.cpp

// Copyright 2012  Cisco Systems (author: Neha Agrawal)

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

#include "gmm/map-diag-gmm-accs.h"

namespace kaldi {

void MapDiagGmmAccs::Init(int32 num_pdf) {
    num_pdf_ = num_pdf;
    pdfs_mean_acc_.resize(num_pdf_);
    pdfs_weight_vec_.resize(num_pdf_);
}

void MapDiagGmmAccs::SetZero() {
    for (size_t pdf_id = 0; pdf_id < num_pdf_; ++pdf_id) {
        pdfs_mean_acc_[pdf_id].SetZero();
        pdfs_weight_vec_[pdf_id].SetZero();
    }
}

void MapDiagGmmAccs::AccumulateFromPosteriors(const DiagGmm &pdf,
                          int32 pdf_id,
                          const VectorBase<BaseFloat>& data,
                          Vector<BaseFloat> &posterior) {

    size_t num_comp = static_cast<int32>(pdf.NumGauss());
    Matrix<BaseFloat> &mean_accs = pdfs_mean_acc_[pdf_id];
    Vector<BaseFloat> &weights_vect = pdfs_weight_vec_[pdf_id];
    size_t dim = static_cast<int32>(pdf.Dim());

    //Initialize
    if( mean_accs.NumRows() == 0 ) {
        mean_accs.Resize(num_comp, dim);
        mean_accs.SetZero();
        weights_vect.Resize(num_comp);
        weights_vect.SetZero();
    }

    //Accumulate Stats
    weights_vect.AddVec(1, posterior);
    Matrix<BaseFloat> tmp(num_comp, dim);
    for(int m = 0; m < num_comp ; ++m) {
        tmp.CopyRowFromVec(data, m);
    }
    tmp.MulRowsVec(posterior);
    mean_accs.AddMat(1, tmp);
}

BaseFloat MapDiagGmmAccs::AccumulateForGmm(int32 pdf_id,
            const VectorBase<BaseFloat>& data,
            const AmDiagGmm &am_gmm,
            BaseFloat weight) {

    const DiagGmm &pdf = am_gmm.GetPdf(pdf_id);
    size_t num_comp = static_cast<int32>(pdf.NumGauss());
    Vector<BaseFloat> posterior(num_comp);
    BaseFloat loglike;

    loglike = pdf.ComponentPosteriors(data, &posterior);
    posterior.Scale(weight);
    AccumulateFromPosteriors(pdf, pdf_id, data, posterior);
    return loglike;
}

void MapDiagGmmAccs::Update(const AmDiagGmm &am_gmm,
        BaseFloat tau,
        AmDiagGmm &map_am_gmm) {

    map_am_gmm.CopyFromAmDiagGmm(am_gmm); //Copy given acoustic model
    for (size_t pdf_id = 0;pdf_id < num_pdf_; ++pdf_id) {
        int32 num_comp = pdfs_mean_acc_[pdf_id].NumRows();

        if(num_comp != 0) { // Compute MAP for only those 
                           // GMM which have accumulated training data
            DiagGmm &tmpdiag_gmm = map_am_gmm.GetPdf(pdf_id); // Get a reference
                                         // to the GMM which needs to be modified

            Matrix<BaseFloat> meanMatrix;
            tmpdiag_gmm.GetMeans(&meanMatrix); // Get diagonal gmm means
            KALDI_ASSERT(num_comp == meanMatrix.NumRows());

            //Initialize MAP adapted means
            Matrix<BaseFloat> map_pdf_means(meanMatrix.NumRows(), meanMatrix.NumCols());
            map_pdf_means.SetZero();

            map_pdf_means.AddMat(tau, meanMatrix); // tau*am_mean
            map_pdf_means.AddMat(1, pdfs_mean_acc_[pdf_id]);//tau*am_mean+AccumMean

            Vector<BaseFloat> sum_tau_weights(num_comp);
            sum_tau_weights.CopyFromVec(pdfs_weight_vec_[pdf_id]);
            sum_tau_weights.Add(tau); // tau+AccumWeights
            sum_tau_weights.InvertElements(); // 1/(tau+AccumWeights)

            // (tau*am_meam+AccumMean)/(tau+AccumWeight)
            map_pdf_means.MulRowsVec(sum_tau_weights);

            tmpdiag_gmm.SetMeans(map_pdf_means); // Set the modified GMM
            tmpdiag_gmm.ComputeGconsts();

        } else {
          //acoustic mean will not change
        }
    }
}

}

