// gst-plugin/gst-online-gmm-decode-faster.h

// Copyright 2013  Tanel Alumae, Tallinn University of Technology

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

#ifndef KALDI_GST_PLUGIN_GST_ONLINE_GMM_DECODE_FASTER_H_
#define KALDI_GST_PLUGIN_GST_ONLINE_GMM_DECODE_FASTER_H_

#include <vector>
#include <gst/gst.h>

#include "feat/feature-mfcc.h"
#include "online/online-audio-source.h"
#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"
#include "util/simple-options.h"
#include "gst-plugin/gst-audio-source.h"

namespace kaldi {


typedef OnlineFeInput<Mfcc> FeInput;

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_ONLINEGMMDECODEFASTER \
    (gst_online_gmm_decode_faster_get_type())
#define GST_ONLINEGMMDECODEFASTER(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_ONLINEGMMDECODEFASTER,GstOnlineGmmDecodeFaster))
#define GST_ONLINEGMMDECODEFASTER_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_ONLINEGMMDECODEFASTER,GstOnlineGmmDecodeFasterClass))
#define GST_IS_ONLINEGMMDECODEFASTER(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_ONLINEGMMDECODEFASTER))
#define GST_IS_ONLINEGMMDECODEFASTER_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_ONLINEGMMDECODEFASTER))

typedef struct _GstOnlineGmmDecodeFaster      GstOnlineGmmDecodeFaster;
typedef struct _GstOnlineGmmDecodeFasterClass GstOnlineGmmDecodeFasterClass;

uint32 kSampleFreq = 16000;

struct _GstOnlineGmmDecodeFaster {
  GstElement element;

  GstPad *sinkpad_, *srcpad_;

  bool silent_;

  OnlineFasterDecoder *decoder_;
  Matrix<BaseFloat> *lda_transform_;
  TransitionModel *trans_model_;
  AmDiagGmm *am_gmm_;
  fst::Fst<fst::StdArc> *decode_fst_;
  fst::SymbolTable *word_syms_;
  fst::VectorFst<LatticeArc> *out_fst_;
  GstBufferSource *au_src_;

  gchar* model_rspecifier_;
  gchar* fst_rspecifier_;
  gchar* word_syms_filename_;
  gchar* lda_mat_rspecifier_;
  std::vector<int32> *silence_phones_;


  BaseFloat acoustic_scale_;
  int32 cmn_window_;
  int32 min_cmn_window_;
  int32 right_context_, left_context_;

  OnlineFasterDecoderOpts *decoder_opts_;
  OnlineFeatureMatrixOptions *feature_reading_opts_;

  SimpleOptions *simple_options_;
};

struct _GstOnlineGmmDecodeFasterClass {
  GstElementClass parent_class;
  void (*hyp_word)(GstElement *element, const gchar *hyp_str);
};

GType gst_online_gmm_decode_faster_get_type(void);

G_END_DECLS
}
#endif  // KALDI_GST_PLUGIN_GST_ONLINE_GMM_DECODE_FASTER_H_
