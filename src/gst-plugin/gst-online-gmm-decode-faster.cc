// gst-plugin/gst-online-gmm-decode-faster.cc

// Copyright 2013  Tanel Alumae, Tallinn University of Technology
// Copyright 2012 Cisco Systems (author: Matthias Paulik)
// Modifications to the original contribution by Cisco Systems made by:
// Vassil Panayotov

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
/**
 * GStreamer plugin for automatic speecg recognition.
 * Based on Kaldi's OnlineGmmDecodeFaster decoder.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0  filesrc location=test.wav \
 *     ! decodebin ! audioconvert ! audioresample \
 *     ! onlinegmmdecodefaster rt-min=0.8 rt-max=0.85  max-active=4000 beam=12.0 acoustic-scale=0.0769 \
 *                              model=$ac_model/model fst=$ac_model/HCLG.fst \
 *                              word-syms=$ac_model/words.txt silence-phones="1:2:3:4:5" \
 *                              lda-mat=$trans_matrix \
 *     ! filesink location=$resultfile
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#else
#  define VERSION "1.0"
#endif

#include <utility>
#include <string>
#include <algorithm>

#include "gst-plugin/kaldimarshal.h"
#include "gst-plugin/gst-online-gmm-decode-faster.h"

#include "feat/feature-mfcc.h"
#include "online/online-audio-source.h"
#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"
#include "util/simple-options.h"
#include "util/parse-options.h"

namespace kaldi {

GST_DEBUG_CATEGORY_STATIC(gst_online_gmm_decode_faster_debug);
#define GST_CAT_DEFAULT gst_online_gmm_decode_faster_debug

enum {
  HYP_WORD_SIGNAL,
  LAST_SIGNAL
};

enum {
  PROP_0,
  PROP_SILENT,
  PROP_MODEL,
  PROP_FST,
  PROP_WORD_SYMS,
  PROP_SILENCE_PHONES,
  PROP_LDA_MAT,
  PROP_LAST
};

#define DEFAULT_MODEL           "final.mdl"
#define DEFAULT_FST             "HCLG.fst"
#define DEFAULT_WORD_SYMS       "words.txt"
#define DEFAULT_SILENCE_PHONES  "1:2:3:4:5"
#define DEFAULT_ACOUSTIC_SCALE  1.0/13
#define DEFAULT_LEFT_CONTEXT    4
#define DEFAULT_RIGHT_CONTEXT   4


/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(
                                "audio/x-raw, "
                                "format = (string) S16LE, "
                                "channels = (int) 1, "
                                "rate = (int) 16000 "));


static GstStaticPadTemplate src_factory =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("text/x-raw, format= { utf8 }"));

static guint gst_online_gmm_decode_faster_signals[LAST_SIGNAL];

#define gst_online_gmm_decode_faster_parent_class parent_class
G_DEFINE_TYPE(GstOnlineGmmDecodeFaster, gst_online_gmm_decode_faster, GST_TYPE_ELEMENT);


static void
gst_online_gmm_decode_faster_set_property(GObject * object, guint prop_id,
                                          const GValue * value,
                                          GParamSpec * pspec);
static void
gst_online_gmm_decode_faster_get_property(GObject * object, guint prop_id,
                                          GValue * value, GParamSpec * pspec);
static GstStateChangeReturn
gst_online_gmm_decode_faster_change_state(GstElement *element,
                                          GstStateChange transition);
static void
gst_online_gmm_decode_faster_finalize(GObject * object);

static gboolean
gst_online_gmm_decode_faster_sink_event(GstPad * pad, GstObject * parent,
                                        GstEvent * event);

static GstFlowReturn gst_online_gmm_decode_faster_chain(GstPad * pad,
                                                        GstObject * parent,
                                                        GstBuffer * buf);

static void
gst_online_gmm_decode_faster_loop(GstOnlineGmmDecodeFaster * filter);


/* GObject vmethod implementations */

/* initialize the onlinegmmdecodefaster's class */
static void gst_online_gmm_decode_faster_class_init(GstOnlineGmmDecodeFasterClass * klass) {
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_online_gmm_decode_faster_set_property;
  gobject_class->get_property = gst_online_gmm_decode_faster_get_property;
  gobject_class->finalize = gst_online_gmm_decode_faster_finalize;

  gstelement_class->change_state = gst_online_gmm_decode_faster_change_state;

  g_object_class_install_property(G_OBJECT_CLASS(klass),
                                  PROP_SILENT,
                                  g_param_spec_boolean("silent",
                                                       "Silence the decoder",
                                                       "Determines whether incoming audio is sent to the decoder or not",
                                                       false,
                                                       (GParamFlags) G_PARAM_READWRITE));
  g_object_class_install_property(G_OBJECT_CLASS(klass),
                                  PROP_MODEL,
                                  g_param_spec_string("model",
                                                      "Acoustic model",
                                                      "Filename of the acoustic model",
                                                      DEFAULT_MODEL,
                                                      (GParamFlags) G_PARAM_READWRITE));
  g_object_class_install_property(G_OBJECT_CLASS(klass),
                                  PROP_FST,
                                  g_param_spec_string("fst",
                                                      "Decoding FST",
                                                      "Filename of the HCLG FST",
                                                      DEFAULT_FST,
                                                      (GParamFlags) G_PARAM_READWRITE));
  g_object_class_install_property(G_OBJECT_CLASS(klass),
                                  PROP_WORD_SYMS,
                                  g_param_spec_string("word-syms",
                                                      "Word symbols",
                                                      "Name of word symbols file (typically words.txt)",
                                                      DEFAULT_WORD_SYMS,
                                                      (GParamFlags) G_PARAM_READWRITE));
  g_object_class_install_property(G_OBJECT_CLASS(klass),
                                  PROP_SILENCE_PHONES,
                                  g_param_spec_string("silence-phones",
                                                      "Silence phones",
                                                      "Colon-separated IDs of silence phones, e.g. '1:2:3:4:5'",
                                                      DEFAULT_SILENCE_PHONES,
                                                      (GParamFlags) G_PARAM_READWRITE));
  g_object_class_install_property(G_OBJECT_CLASS(klass),
                                  PROP_LDA_MAT,
                                  g_param_spec_string("lda-mat",
                                                      "LDA matrix",
                                                      "Filename of the LDA transform data",
                                                      "",
                                                      (GParamFlags) G_PARAM_READWRITE));

  gst_element_class_set_details_simple(gstelement_class,
                                       "OnlineGmmDecodeFaster",
                                       "Speech/Audio",
                                       "Convert speech to text",
                                       "Tanel Alumae <tanel.alumae@phon.ioc.ee>");

  gst_element_class_add_pad_template(gstelement_class,
                                     gst_static_pad_template_get(&src_factory));
  gst_element_class_add_pad_template(gstelement_class,
                                     gst_static_pad_template_get(&sink_factory));

  gst_online_gmm_decode_faster_signals[HYP_WORD_SIGNAL]
      = g_signal_new("hyp-word", G_TYPE_FROM_CLASS(klass), G_SIGNAL_RUN_LAST,
                     G_STRUCT_OFFSET(GstOnlineGmmDecodeFasterClass, hyp_word),
                     NULL, NULL, kaldi_marshal_VOID__STRING, G_TYPE_NONE, 1,
                     G_TYPE_STRING);
}



/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_online_gmm_decode_faster_init(GstOnlineGmmDecodeFaster * filter) {
  bool tmp_bool;
  int32 tmp_int;
  uint32 tmp_uint;
  float tmp_float;
  double tmp_double;
  std::string tmp_string;
  filter->silent_ = false;
  filter->model_rspecifier_ = g_strdup(DEFAULT_MODEL);
  filter->fst_rspecifier_ = g_strdup(DEFAULT_FST);
  filter->word_syms_filename_ = g_strdup(DEFAULT_WORD_SYMS);
  filter->lda_mat_rspecifier_ = g_strdup("");
  filter->silence_phones_ = new std::vector<int32>;
  SplitStringToIntegers(DEFAULT_SILENCE_PHONES, ":", false, filter->silence_phones_);

  filter->simple_options_ = new SimpleOptions();

  filter->decoder_opts_ = new OnlineFasterDecoderOpts();
  filter->decoder_opts_->Register(filter->simple_options_, true);
  filter->feature_reading_opts_ = new OnlineFeatureMatrixOptions();
  filter->feature_reading_opts_->Register(filter->simple_options_);

  filter->acoustic_scale_ = DEFAULT_ACOUSTIC_SCALE;
  filter->cmn_window_ = 600;
  filter->min_cmn_window_ = 100;  // adds 1 second latency, only at utterance start.
  filter->right_context_ = DEFAULT_RIGHT_CONTEXT;
  filter->left_context_ = DEFAULT_LEFT_CONTEXT;

  filter->simple_options_->Register("left-context", &filter->left_context_,
                                    "Number of frames of left context");
  filter->simple_options_->Register("right-context", &filter->right_context_,
                                    "Number of frames of right context");
  filter->simple_options_->Register("acoustic-scale", &filter->acoustic_scale_,
                                    "Scaling factor for acoustic likelihoods");

  filter->simple_options_->Register("cmn-window", &(filter->cmn_window_),
                                    "Number of feat. vectors used in the running average CMN calculation");
  filter->simple_options_->Register("min-cmn-window", &filter->min_cmn_window_,
                                    "Minumum CMN window used at start of decoding (adds "
                                      "latency only at start)");


  filter->sinkpad_ = gst_pad_new_from_static_template(&sink_factory, "sink");
  gst_pad_set_event_function(filter->sinkpad_,
                              GST_DEBUG_FUNCPTR(gst_online_gmm_decode_faster_sink_event));
  gst_pad_set_chain_function(filter->sinkpad_,
                              GST_DEBUG_FUNCPTR(gst_online_gmm_decode_faster_chain));

  gst_pad_use_fixed_caps(filter->sinkpad_);
  gst_element_add_pad(GST_ELEMENT(filter), filter->sinkpad_);

  filter->srcpad_ = gst_pad_new_from_static_template(&src_factory, "src");
  gst_pad_use_fixed_caps(filter->srcpad_);
  gst_element_add_pad(GST_ELEMENT(filter), filter->srcpad_);

  // init properties from various Kaldi Opts
  GstElementClass * klass = GST_ELEMENT_GET_CLASS(filter);

  std::vector<std::pair<std::string, SimpleOptions::OptionInfo> > option_info_list;
  option_info_list = filter->simple_options_->GetOptionInfoList();
  int32 i = 0;
  for (std::vector<std::pair<std::string,
      SimpleOptions::OptionInfo> >::iterator dx = option_info_list.begin();
      dx != option_info_list.end(); dx++) {
    std::pair<std::string, SimpleOptions::OptionInfo> result = (*dx);
    SimpleOptions::OptionInfo option_info = result.second;
    std::string name = result.first;
    switch (option_info.type) {
      case SimpleOptions::kBool:
        filter->simple_options_->GetOption(name, &tmp_bool);
        g_object_class_install_property(
                                        G_OBJECT_CLASS(klass),
                                        PROP_LAST + i,
                                        g_param_spec_boolean(
                                                             name.c_str(),
                                                             option_info.doc.c_str(),
                                                             option_info.doc.c_str(),
                                                             tmp_bool,
                                                             (GParamFlags) G_PARAM_READWRITE));
        break;
      case SimpleOptions::kInt32:
        filter->simple_options_->GetOption(name, &tmp_int);
        g_object_class_install_property(
                                        G_OBJECT_CLASS(klass),
                                        PROP_LAST + i,
                                        g_param_spec_int(
                                                         name.c_str(),
                                                         option_info.doc.c_str(),
                                                         option_info.doc.c_str(),
                                                         G_MININT,
                                                         G_MAXINT,
                                                         tmp_int,
                                                         (GParamFlags) G_PARAM_READWRITE));
        break;
      case SimpleOptions::kUint32:
        filter->simple_options_->GetOption(name, &tmp_uint);
        g_object_class_install_property(
                                        G_OBJECT_CLASS(klass),
                                        PROP_LAST + i,
                                        g_param_spec_uint(
                                                          name.c_str(),
                                                          option_info.doc.c_str(),
                                                          option_info.doc.c_str(),
                                                          0,
                                                          G_MAXUINT,
                                                          tmp_uint,
                                                          (GParamFlags) G_PARAM_READWRITE));
        break;
      case SimpleOptions::kFloat:
        filter->simple_options_->GetOption(name, &tmp_float);
        g_object_class_install_property(
                                        G_OBJECT_CLASS(klass),
                                        PROP_LAST + i,
                                        g_param_spec_float(
                                                           name.c_str(),
                                                           option_info.doc.c_str(),
                                                           option_info.doc.c_str(),
                                                           G_MINFLOAT,
                                                           G_MAXFLOAT,
                                                           tmp_float,
                                                           (GParamFlags) G_PARAM_READWRITE));
        break;
      case SimpleOptions::kDouble:
        filter->simple_options_->GetOption(name, &tmp_double);
        g_object_class_install_property(
                                        G_OBJECT_CLASS(klass),
                                        PROP_LAST + i,
                                        g_param_spec_double(
                                                            name.c_str(),
                                                            option_info.doc.c_str(),
                                                            option_info.doc.c_str(),
                                                            G_MINDOUBLE,
                                                            G_MAXDOUBLE,
                                                            tmp_double,
                                                            (GParamFlags) G_PARAM_READWRITE));
        break;
      case SimpleOptions::kString:
        filter->simple_options_->GetOption(name, &tmp_string);
        g_object_class_install_property(
                                        G_OBJECT_CLASS(klass),
                                        PROP_LAST + i,
                                        g_param_spec_string(
                                                            name.c_str(),
                                                            option_info.doc.c_str(),
                                                            option_info.doc.c_str(),
                                                            tmp_string.c_str(),
                                                            (GParamFlags) G_PARAM_READWRITE));
        break;
    }
    i += 1;
  }
}

static bool
gst_online_gmm_decode_faster_allocate(GstOnlineGmmDecodeFaster * filter) {
  if (!filter->decoder_) {
    GST_INFO_OBJECT(filter,  "Loading Kaldi decoder");
    filter->lda_transform_ = new Matrix<BaseFloat>;
    if (strlen(filter->lda_mat_rspecifier_) > 0) {
      bool binary_in;
      Input ki(filter->lda_mat_rspecifier_, &binary_in);
      filter->lda_transform_->Read(ki.Stream(), binary_in);
    }
    filter->trans_model_ = new TransitionModel();
    filter->am_gmm_ = new AmDiagGmm();
    {
      bool binary;
      Input ki(filter->model_rspecifier_, &binary);
      filter->trans_model_->Read(ki.Stream(), binary);
      filter->am_gmm_->Read(ki.Stream(), binary);
    }
    filter->word_syms_ = NULL;
    if (!(filter->word_syms_ = fst::SymbolTable::ReadText(filter->word_syms_filename_))) {
      GST_ERROR_OBJECT(filter, "Could not read symbol table from file %s", filter->word_syms_filename_);
      return false;
    }

    filter->decode_fst_ = ReadDecodeGraph(filter->fst_rspecifier_);

    int32 window_size = filter->right_context_ + filter->left_context_ + 1;
    filter->decoder_opts_->batch_size = std::max(filter->decoder_opts_->batch_size, window_size);

    filter->out_fst_ = new fst::VectorFst<LatticeArc> ();

    filter->au_src_ = new GstBufferSource();

    filter->decoder_ = new OnlineFasterDecoder(*(filter->decode_fst_),
                                               *(filter->decoder_opts_),
                                               *(filter->silence_phones_),
                                               *(filter->trans_model_));

    GST_INFO_OBJECT(filter,  "Finished loading Kaldi decoder");
  }
  return true;
}

static void
gst_online_gmm_decode_faster_finalize(GObject * object) {
  GstOnlineGmmDecodeFaster *filter = GST_ONLINEGMMDECODEFASTER(object);

  g_free(filter->model_rspecifier_);
  g_free(filter->fst_rspecifier_);
  g_free(filter->word_syms_filename_);
  g_free(filter->lda_mat_rspecifier_);
  delete filter->silence_phones_;
  delete filter->decoder_opts_;
  delete filter->feature_reading_opts_;
  if (filter->decoder_) {
    delete filter->decoder_;
    filter->decoder_ = NULL;
  }
  if (filter->lda_transform_) {
    delete filter->lda_transform_;
    filter->lda_transform_ = NULL;
  }
  if (filter->am_gmm_) {
    delete filter->am_gmm_;
    filter->am_gmm_ = NULL;
  }
  if (filter->word_syms_) {
    delete filter->word_syms_;
    filter->word_syms_ = NULL;
  }
  if (filter->decode_fst_) {
    delete filter->decode_fst_;
    filter->decode_fst_ = NULL;
  }
  if (filter->out_fst_) {
    delete filter->out_fst_;
    filter->out_fst_ = NULL;
  }
  if (filter->au_src_) {
    delete filter->au_src_;
    filter->au_src_ = NULL;
  }
  if (filter->simple_options_) {
    delete filter->simple_options_;
    filter->simple_options_ = NULL;
  }

  G_OBJECT_CLASS(parent_class)->finalize(object);
}


static bool
gst_online_gmm_decode_faster_deallocate(GstOnlineGmmDecodeFaster * filter) {
  /* We won't deallocate the decoder once it's already allocated, since model loading could take a lot of time */
  GST_INFO_OBJECT(filter, "Refusing to unload decoder");
  return true;
}

static void
gst_online_gmm_decode_faster_set_property(GObject * object, guint prop_id,
                                          const GValue * value, GParamSpec * pspec) {
  GstOnlineGmmDecodeFaster *filter = GST_ONLINEGMMDECODEFASTER(object);

  if (prop_id == PROP_SILENT) {
    filter->silent_ = g_value_get_boolean(value);
    return;
  }
  // All other props cannot be changed after initialization
  if (filter->decoder_) {
    GST_WARNING_OBJECT(filter,  "Decoder already initialized, cannot change it's properties");
    return;
  }
  switch (prop_id) {
    case PROP_MODEL:
      g_free(filter->model_rspecifier_);
      filter->model_rspecifier_ = g_value_dup_string(value);
      break;
    case PROP_FST:
      g_free(filter->fst_rspecifier_);
      filter->fst_rspecifier_ = g_value_dup_string(value);
      break;
    case PROP_WORD_SYMS:
      g_free(filter->word_syms_filename_);
      filter->word_syms_filename_ = g_value_dup_string(value);
      break;
    case PROP_LDA_MAT:
      g_free(filter->lda_mat_rspecifier_);
      filter->lda_mat_rspecifier_ = g_value_dup_string(value);
      break;
    case PROP_SILENCE_PHONES:
      SplitStringToIntegers(g_value_get_string(value), ":", false, filter->silence_phones_);
      break;
    default:
      if (prop_id >= PROP_LAST) {
        const gchar* name = g_param_spec_get_name(pspec);
        SimpleOptions::OptionType option_type;
        if (filter->simple_options_->GetOptionType(std::string(name), &option_type)) {
          switch (option_type) {
            case SimpleOptions::kBool:
              filter->simple_options_->SetOption(name, g_value_get_boolean(value));
              break;
            case SimpleOptions::kInt32:
              filter->simple_options_->SetOption(name, g_value_get_int(value));
              break;
            case SimpleOptions::kUint32:
              filter->simple_options_->SetOption(name, g_value_get_uint(value));
              break;
            case SimpleOptions::kFloat:
              filter->simple_options_->SetOption(name, g_value_get_float(value));
              break;
            case SimpleOptions::kDouble:
              filter->simple_options_->SetOption(name, g_value_get_double(value));
              break;
            case SimpleOptions::kString:
              filter->simple_options_->SetOption(name, g_value_dup_string(value));
              break;
          }
          break;
        }
      }
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_online_gmm_decode_faster_get_property(GObject * object, guint prop_id,
                                           GValue * value, GParamSpec * pspec) {
  bool tmp_bool;
  int32 tmp_int;
  uint32 tmp_uint;
  float tmp_float;
  double tmp_double;
  std::string tmp_string;

  GstOnlineGmmDecodeFaster *filter = GST_ONLINEGMMDECODEFASTER(object);
  std::ostringstream ss;

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean(value, filter->silent_);
      break;
    case PROP_MODEL:
      g_value_set_string(value, filter->model_rspecifier_);
      break;
    case PROP_FST:
      g_value_set_string(value, filter->fst_rspecifier_);
      break;
    case PROP_WORD_SYMS:
      g_value_set_string(value, filter->word_syms_filename_);
      break;
    case PROP_LDA_MAT:
      g_value_set_string(value, filter->lda_mat_rspecifier_);
      break;
    case PROP_SILENCE_PHONES:
      for (size_t j = 0; j < filter->silence_phones_->size(); j++) {
        if (j > 0) {
          ss << ":";
        }
        ss <<  (*filter->silence_phones_)[j];
      }
      g_value_set_string(value, ss.str().c_str());
      break;
    default:
      if (prop_id >= PROP_LAST) {
        const gchar* name = g_param_spec_get_name(pspec);
        SimpleOptions::OptionType option_type;
        if (filter->simple_options_->GetOptionType(std::string(name), &option_type)) {
          switch (option_type) {
            case SimpleOptions::kBool:
              filter->simple_options_->GetOption(name, &tmp_bool);
              g_value_set_boolean(value, tmp_bool);
              break;
            case SimpleOptions::kInt32:
              filter->simple_options_->GetOption(name, &tmp_int);
              g_value_set_int(value, tmp_int);
              break;
            case SimpleOptions::kUint32:
              filter->simple_options_->GetOption(name, &tmp_uint);
              g_value_set_uint(value, tmp_uint);
              break;
            case SimpleOptions::kFloat:
              filter->simple_options_->GetOption(name, &tmp_float);
              g_value_set_float(value, tmp_float);
              break;
            case SimpleOptions::kDouble:
              filter->simple_options_->GetOption(name, &tmp_double);
              g_value_set_double(value, tmp_double);
              break;
            case SimpleOptions::kString:
              filter->simple_options_->GetOption(name, &tmp_string);
              g_value_set_string(value, tmp_string.c_str());
              break;
          }
          break;
        }
      }
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}


static GstStateChangeReturn
gst_online_gmm_decode_faster_change_state(GstElement *element, GstStateChange transition) {
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstOnlineGmmDecodeFaster *filter = GST_ONLINEGMMDECODEFASTER(element);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      if (!gst_online_gmm_decode_faster_allocate(filter))
        return GST_STATE_CHANGE_FAILURE;
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE)
    return ret;

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_NULL:
      gst_online_gmm_decode_faster_deallocate(filter);
      break;
    default:
      break;
  }

  return ret;
}

/*
 * Emit a single recognized word:
 *   * emit through the sink pad of the element
 *   * emit by the hy-word signal
 */
static void
gst_online_gmm_decode_faster_push_word(GstOnlineGmmDecodeFaster * filter, GstPad *pad, std::string word) {
  const gchar *hyp = word.c_str();
  guint hyp_len = strlen(hyp);
  GST_DEBUG_OBJECT(filter,  "WORD: %s", hyp);
  /* +1 for terminating NUL character */
  GstBuffer *buffer = gst_buffer_new_and_alloc(hyp_len + 2);
  gst_buffer_fill(buffer, 0, hyp, hyp_len);
  gst_buffer_memset(buffer, hyp_len, ' ', 1);
  gst_buffer_memset(buffer, hyp_len + 1, '\0', 1);
  gst_buffer_set_size(buffer, hyp_len + 1);

  gst_pad_push(pad, buffer);
  /* Emit a signal for applications. */
  g_signal_emit(filter, gst_online_gmm_decode_faster_signals[HYP_WORD_SIGNAL], 0, hyp);
}


static void
gst_online_gmm_decode_faster_push_words(GstOnlineGmmDecodeFaster * filter, GstPad *pad,
                                        const std::vector<int32>& words,
                                        const fst::SymbolTable *word_syms,
                                        bool line_break) {
  KALDI_ASSERT(word_syms != NULL);
  std::stringstream ss;
  for (size_t i = 0; i < words.size(); i++) {
    std::string word = word_syms->Find(words[i]);
    if (word == "") {
      GST_ERROR_OBJECT(filter, "Word-id %d  not in symbol table!",  words[i]);
    }
    gst_online_gmm_decode_faster_push_word(filter, pad, word);
  }

  if (line_break) {
    gst_online_gmm_decode_faster_push_word(filter, pad, "<#s>");
  }
}

static void
gst_online_gmm_decode_faster_loop(GstOnlineGmmDecodeFaster * filter) {
  // We are not properly registering/exposing MFCC and frame extraction options,
  // because there are parts of the online decoding code, where some of these
  // options are hardwired(ToDo: we should fix this at some point)
  MfccOptions mfcc_opts;
  mfcc_opts.use_energy = false;
  int32 frame_length = mfcc_opts.frame_opts.frame_length_ms = 25;
  int32 frame_shift = mfcc_opts.frame_opts.frame_shift_ms = 10;

  // Up to delta-delta derivative features are calculated (unless LDA is used)
  const int32 kDeltaOrder = 2;
  Mfcc mfcc(mfcc_opts);
  FeInput fe_input(filter->au_src_, &mfcc,
                   frame_length * (kSampleFreq / 1000),
                   frame_shift * (kSampleFreq / 1000));
  OnlineCmnInput cmn_input(&fe_input, filter->cmn_window_, filter->min_cmn_window_);

  OnlineFeatInputItf *feat_transform = 0;
  if (strlen(filter->lda_mat_rspecifier_) > 0) {
    feat_transform = new OnlineLdaInput(&cmn_input, *(filter->lda_transform_),
                                        filter->left_context_,
                                        filter->right_context_);
  } else {
    DeltaFeaturesOptions opts;
    opts.order = kDeltaOrder;
    // Note from Dan: keeping the next statement for back-compatibility,
    // but I don't think this is really the right way to set the window-size
    // in the delta computation: it should be a separate config.
    opts.window = filter->left_context_ / 2;
    feat_transform = new OnlineDeltaInput(opts, &cmn_input);
  }


  // feature_reading_opts contains timeout, batch size.
  OnlineFeatureMatrix feature_matrix(*(filter->feature_reading_opts_),
                                     feat_transform);


  OnlineDecodableDiagGmmScaled decodable(*(filter->am_gmm_), *(filter->trans_model_),
                                         filter->acoustic_scale_, &feature_matrix);

  GST_DEBUG_OBJECT(filter,  "starting decoding loop");

  bool partial_res = false;
  filter->decoder_->InitDecoding();
  while (1) {
    OnlineFasterDecoder::DecodeState dstate = filter->decoder_->Decode(&decodable);

    if (dstate & (filter->decoder_->kEndFeats | filter->decoder_->kEndUtt)) {
      std::vector<int32> word_ids;
      filter->decoder_->FinishTraceBack(filter->out_fst_);
      fst::GetLinearSymbolSequence(*(filter->out_fst_),
                                   static_cast<std::vector<int32> *>(0),
                                   &word_ids,
                                   static_cast<LatticeArc::Weight*>(0));
      gst_online_gmm_decode_faster_push_words(filter, filter->srcpad_, word_ids, filter->word_syms_, partial_res || word_ids.size());
      partial_res = false;
      if (dstate == filter->decoder_->kEndFeats)
        break;
    } else {
      std::vector<int32> word_ids;
      if (filter->decoder_->PartialTraceback(filter->out_fst_)) {
        fst::GetLinearSymbolSequence(*(filter->out_fst_),
                                     static_cast<std::vector<int32> *>(0),
                                     &word_ids,
                                     static_cast<LatticeArc::Weight*>(0));
        gst_online_gmm_decode_faster_push_words(filter, filter->srcpad_, word_ids, filter->word_syms_, false);
        if (!partial_res)
          partial_res = (word_ids.size() > 0);
      }
    }
  }
  GST_DEBUG_OBJECT(filter, "Finished decoding loop");
  GST_DEBUG_OBJECT(filter, "Pushing EOS event");
  gst_pad_push_event(filter->srcpad_, gst_event_new_eos());

  GST_DEBUG_OBJECT(filter, "Pausing decoding task");
  gst_pad_pause_task(filter->srcpad_);
  delete feat_transform;
  delete filter->au_src_;
  filter->au_src_ = new GstBufferSource();
}

/* GstElement vmethod implementations */
/* this function handles sink events */
static gboolean
gst_online_gmm_decode_faster_sink_event(GstPad * pad, GstObject * parent, GstEvent * event) {
  gboolean ret;
  GstOnlineGmmDecodeFaster *filter;

  filter = GST_ONLINEGMMDECODEFASTER(parent);
  GST_DEBUG_OBJECT(filter, "Handling %s event", GST_EVENT_TYPE_NAME(event));

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_SEGMENT:
    {
      GST_DEBUG_OBJECT(filter,  "Starting decoding task");
      gst_pad_start_task(filter->srcpad_,
                         (GstTaskFunction) gst_online_gmm_decode_faster_loop, filter, NULL);

      GST_DEBUG_OBJECT(filter,  "Started decoding task");
      ret = TRUE;
      break;
    }
    case GST_EVENT_CAPS:
    {
      ret = TRUE;
      break;
    }
    case GST_EVENT_EOS:
    {
      /* end-of-stream, we should close down all stream leftovers here */
      GST_DEBUG_OBJECT(filter, "EOS received");
      filter->au_src_->SetEnded(true);
      ret = TRUE;
      break;
    }
    default:
      ret = gst_pad_event_default(pad, parent, event);
      break;
  }
  return ret;
}

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn gst_online_gmm_decode_faster_chain(GstPad * pad,
                                                        GstObject * parent,
                                                        GstBuffer * buf) {
  GstOnlineGmmDecodeFaster *filter;

  filter = GST_ONLINEGMMDECODEFASTER(parent);

  if (G_UNLIKELY(!filter->decoder_))
    goto not_negotiated;
  if (!filter->silent_) {
    filter->au_src_->PushBuffer(buf);
  }
  gst_buffer_unref(buf);
  return GST_FLOW_OK;

  /* special cases */
  not_negotiated: {
    GST_ELEMENT_ERROR(filter, CORE, NEGOTIATION, (NULL),
                      ("decoder wasn't allocated before chain function"));

    gst_buffer_unref(buf);
    return GST_FLOW_NOT_NEGOTIATED;
  }
}


/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
onlinegmmdecodefaster_init(GstPlugin * onlinegmmdecodefaster) {
  /* debug category for fltering log messages
   */
  GST_DEBUG_CATEGORY_INIT(gst_online_gmm_decode_faster_debug, "onlinegmmdecodefaster",
                           0, "Automatic Speech Recognition");

  return gst_element_register(onlinegmmdecodefaster, "onlinegmmdecodefaster", GST_RANK_NONE,
                               GST_TYPE_ONLINEGMMDECODEFASTER);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "myfirstonlinegmmdecodefaster"
#endif

/* gstreamer looks for this structure to register onlinegmmdecodefasters
 *
 * exchange the string 'Template onlinegmmdecodefaster' with your onlinegmmdecodefaster description
 */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    onlinegmmdecodefaster,
    "Online speech recognizer based on the Kaldi toolkit",
    onlinegmmdecodefaster_init,
    VERSION,
    "LGPL",  // Changing it into Apache prevents the plugin from loading, see gst/gstplugin.c in GStreamer source
    "Kaldi",
    "http://kaldi.sourceforge.net/"
)
}
