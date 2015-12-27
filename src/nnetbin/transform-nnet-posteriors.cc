// nnetbin/nnet-phone-prob.cc

// Copyright 2013-2014  The Johns Hopkins University (Author: Sri Harish Mallidi)

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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;

  try {
    const char *usage =
        "Transform nnet posteriors (e.g. log, logit, pdf-to-pseudo-phone). \n"
        "Perform forward pass through Neural Network and output phone (posterior) probabilities, as an archive matrix.\n"
        "\n"
        "Usage:  transform-nnet-posteriors [options] <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " transform-nnet-posteriors --pdf-to-pseudo-phone=pdf_to_pseudo_phone.txt nnet ark:pdf_posteriors.ark ark:phone_posteriors.ark\n";

    ParseOptions po(usage);
 
    std::string pdf_to_phone_rxfilename;
    po.Register("pdf-to-pseudo-phone", &pdf_to_phone_rxfilename, "A two column file "
                "that maps pdf-id to the corresponding pseudo phone-id."
		".");

    std::string apply_log="false";
    po.Register("apply-log", &apply_log, "Apply log to posteriors");

    std::string apply_logit="false";
    po.Register("apply-logit", &apply_logit, "Apply logit to posteriors");


    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        feature_wspecifier = po.GetArg(2);
        
    using namespace kaldi;
    typedef kaldi::int32 int32;

    // Reads pdf to phone map if supplied.
    std::vector<int32> pdf_to_phone_map;
    int32 max_phone_id = 0;
    if (pdf_to_phone_rxfilename != "") {
      std::vector<std::vector<int32> > vec;
      if (!ReadIntegerVectorVectorSimple(pdf_to_phone_rxfilename, &vec))
	KALDI_ERR << "Error reading pdf to phone map from "
            << PrintableRxfilename(pdf_to_phone_rxfilename);
      
      pdf_to_phone_map.resize(vec.size(), -1);
      for (size_t i = 0; i < vec.size(); i++) {
	if (vec[i].size() !=2 || vec[i][0] < 0 || vec[i][1] < 0)
	  KALDI_ERR << "Error reading pdf to phone map from "
		    << PrintableRxfilename(pdf_to_phone_rxfilename)
		    << " (bad line " << i << ")";
	if (vec[i][0] >= vec.size())
	  KALDI_ERR << "Pdf-id seems too large: given " << vec[i][0]
		    << " while expecting a number less than size " << vec.size();
	if (pdf_to_phone_map[vec[i][0]] != -1)
	  KALDI_ERR << "Pdf-id has been mapped to " << pdf_to_phone_map[vec[i][0]]
		    << ", please keep pdf to phone map unique.";

	pdf_to_phone_map[vec[i][0]] = vec[i][1];
        if (vec[i][1] > max_phone_id)
	  max_phone_id = vec[i][1];
      }
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << feature_reader.Key() 
                    << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      for(int32 r=0; r<mat.NumRows(); r++) {
        for(int32 c=0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          if(val != val) KALDI_ERR << "NaN in features of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << feature_reader.Key();
        }
      }

      // Map pdf-id's to phone-id's
      Matrix<BaseFloat> phone_post(mat);
      if (pdf_to_phone_rxfilename != "") {
	// Checks if every pdf has a corresponding phone
	KALDI_ASSERT(mat.NumCols() == pdf_to_phone_map.size());
	phone_post.Resize(mat.NumRows(), max_phone_id + 1);
	for (int32 i = 0; i < mat.NumRows(); i++) {
	  for (int32 j = 0; j <= max_phone_id; j++) {
	    phone_post(i, j) = 0;
	  }
	  for (int32 j = 0 ; j < mat.NumCols(); j++) {
	    phone_post(i, pdf_to_phone_map[j]) += mat(i, j);
	  }
	}
      }

      if ((apply_log == "true") || (apply_logit == "true")) {
        // check if posteriors
	if (!(phone_post.Min() >= 0.0 && phone_post.Max() <= 1.0)) {
	  KALDI_WARN << feature_reader.Key() << " "
		     << "Applying 'log' or 'logit' to data which don't seem to be probabilities ";
	}

	
	if (apply_log == "true") {
	  phone_post.Add(1e-20); // avoid log(0)
	  phone_post.ApplyLog();

	} else if (apply_logit == "true") {
	  // logit is more sensitive to log(0), 
	  // so higher smoothing value
	  phone_post.Add(1e-5); 
	  for (int32 i=0; i<phone_post.NumRows(); i++) {
	    // Normalize, after smoothing
	    SubVector<BaseFloat> Row(phone_post, i);
	    BaseFloat row_sum = Row.Sum();
	    Row.Scale(1/row_sum);
	  }
	
	  // Apply logit 
	  // logit = log(phone_post / (1 - phone_post))
	  
	  Matrix<BaseFloat> denom_logit_phone_post(phone_post);
	  denom_logit_phone_post.Scale(-1);
	  denom_logit_phone_post.Add(1); // 1 - phone_post
	  
	  phone_post.DivElements(denom_logit_phone_post);
	  phone_post.ApplyLog();
	} 
      }
    
      //check for NaN/inf
      for(int32 r=0; r<phone_post.NumRows(); r++) {
        for(int32 c=0; c<phone_post.NumCols(); c++) {
          BaseFloat val = phone_post(r, c);
          if(val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
        }
      }

      // write
      feature_writer.Write(feature_reader.Key(), phone_post);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
