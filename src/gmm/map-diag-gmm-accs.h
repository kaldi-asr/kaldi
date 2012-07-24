// gmm/map-diag-gmm-accs.h

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

#ifndef MAP_DIAG_GMM_ACCS_H_
#define MAP_DIAG_GMM_ACCS_H_
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/transform-common.h"

namespace kaldi {

class MapDiagGmmAccs {
public:
    void Init(int32 num_pdf);
    void SetZero();
    void AccumulateFromPosteriors(const DiagGmm &pdf,
            int32 pdf_id,
            const VectorBase<BaseFloat>& data,
            Vector<BaseFloat> &posterior);

    BaseFloat AccumulateForGmm(int32 pdf_id,
            const VectorBase<BaseFloat>& data,
            const AmDiagGmm &am_gmm,
            BaseFloat weight) ;

    void Update(const AmDiagGmm &am_gmm,
            BaseFloat tau,
            AmDiagGmm &map_am_gmm);
private:
    int32 num_pdf_;
    std::vector< Matrix<BaseFloat> > pdfs_mean_acc_;
    std::vector< Vector<BaseFloat> > pdfs_weight_vec_;
};

typedef TableWriter< KaldiObjectHolder<AmDiagGmm> >  MapAmDiagGmmWriter;
typedef RandomAccessTableReader< KaldiObjectHolder<AmDiagGmm> > RandomAccessMapAmDiagGmmReader;
typedef SequentialTableReader< KaldiObjectHolder<AmDiagGmm> > MapAmDiagGmmSeqReader;

}

#endif /* MAP_DIAG_GMM_ACCS_H_ */
