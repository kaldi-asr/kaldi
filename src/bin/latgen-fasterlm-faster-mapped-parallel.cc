// bin/latgen-fasterlm-faster-mapped .cc

// Copyright      2018  Zhehuai Chen

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "util/kaldi-thread.h"
#include "lm/faster-arpa-lm.h"
#include "decoder/lattice-biglm-faster-decoder.h"


namespace kaldi {

/// This class basically does the same job as the function
/// DecodeUtteranceLatticeFaster, but in a way that allows us
/// to build a multi-threaded command line program more easily.
/// The main computation takes place in operator (), and the output
/// happens in the destructor.
class DecodeUtteranceLatticeBiglmFasterClass {
 public:
  // Initializer sets various variables.
  // NOTE: we "take ownership" of "decoder" and "decodable".  These
  // are deleted by the destructor.  On error, "num_err" is incremented.
  DecodeUtteranceLatticeBiglmFasterClass(
      LatticeBiglmFasterDecoder *decoder,
      DecodableInterface *decodable,
      const TransitionModel &trans_model,
      const fst::SymbolTable *word_syms,
      std::string utt,
      BaseFloat acoustic_scale,
      bool determinize,
      bool allow_partial,
      Int32VectorWriter *alignments_writer,
      Int32VectorWriter *words_writer,
      CompactLatticeWriter *compact_lattice_writer,
      LatticeWriter *lattice_writer,
      double *like_sum, // on success, adds likelihood to this.
      int64 *frame_sum, // on success, adds #frames to this.
      int32 *num_done, // on success (including partial decode), increments this.
      int32 *num_err,  // on failure, increments this.
      int32 *num_partial) :  // If partial decode (final-state not reached), increments this.
    decoder_(decoder), decodable_(decodable), trans_model_(&trans_model),
    word_syms_(word_syms), utt_(utt), acoustic_scale_(acoustic_scale),
    determinize_(determinize), allow_partial_(allow_partial),
    alignments_writer_(alignments_writer),
    words_writer_(words_writer),
    compact_lattice_writer_(compact_lattice_writer),
    lattice_writer_(lattice_writer),
    like_sum_(like_sum), frame_sum_(frame_sum),
    num_done_(num_done), num_err_(num_err),
    num_partial_(num_partial),
    computed_(false), success_(false), partial_(false),
    clat_(NULL), lat_(NULL) { }
  
  void operator () () {// The decoding happens here. 
  computed_ = true; // Just means this function was called-- a check on the
  // calling code.
  success_ = true;
  using fst::VectorFst;
  if (!decoder_->Decode(decodable_)) {
    KALDI_WARN << "Failed to decode file " << utt_;
    success_ = false;
  }
  if (!decoder_->ReachedFinal()) {
    if (allow_partial_) {
      KALDI_WARN << "Outputting partial output for utterance " << utt_
                 << " since no final-state reached\n";
      partial_ = true;
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt_
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      success_ = false;
    }
  }
  if (!success_) return;

  // Get lattice, and do determinization if requested.
  lat_ = new Lattice;
  decoder_->GetRawLattice(lat_);
  if (lat_->NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt_;
  fst::Connect(lat_);
  if (determinize_) {
    clat_ = new CompactLattice;
    if (!DeterminizeLatticePhonePrunedWrapper(
            *trans_model_,
            lat_,
            decoder_->GetOptions().lattice_beam,
            clat_,
            decoder_->GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt_;
    delete lat_;
    lat_ = NULL;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale_ != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale_), clat_);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale_ != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale_), lat_);
  }
  }
  ~DecodeUtteranceLatticeBiglmFasterClass() { // Output happens here.
  if (!computed_)
    KALDI_ERR << "Destructor called without operator (), error in calling code.";

  if (!success_) {
    if (num_err_ != NULL) (*num_err_)++;
  } else { // successful decode.
    // Getting the one-best output is lightweight enough that we can do it in
    // the destructor (easier than adding more variables to the class, and
    // will rarely slow down the main thread.)
    double likelihood;
    LatticeWeight weight = LatticeWeight::Zero();
    int32 num_frames;
    { // First do some stuff with word-level traceback...
      // This is basically for diagnostics.
      fst::VectorFst<LatticeArc> decoded;
      decoder_->GetBestPath(&decoded);
      if (decoded.NumStates() == 0) {
        // Shouldn't really reach this point as already checked success.
        KALDI_ERR << "Failed to get traceback for utterance " << utt_;
      }
      std::vector<int32> alignment;
      std::vector<int32> words;
      GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
      num_frames = alignment.size();
      if (words_writer_->IsOpen())
        words_writer_->Write(utt_, words);
      if (alignments_writer_ && alignments_writer_->IsOpen())
        alignments_writer_->Write(utt_, alignment);
      if (word_syms_ != NULL) {
        std::cerr << utt_ << ' ';
        for (size_t i = 0; i < words.size(); i++) {
          std::string s = word_syms_->Find(words[i]);
          if (s == "")
            KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
          std::cerr << s << ' ';
        }
        std::cerr << '\n';
      }
      likelihood = -(weight.Value1() + weight.Value2());
    }

    // Ouptut the lattices.
    if (determinize_) { // CompactLattice output.
      KALDI_ASSERT(compact_lattice_writer_ != NULL && clat_ != NULL);
      if (clat_->NumStates() == 0) {
        KALDI_WARN << "Empty lattice for utterance " << utt_;
      } else {
        compact_lattice_writer_->Write(utt_, *clat_);
      }
      delete clat_;
      clat_ = NULL;
    } else {
      KALDI_ASSERT(lattice_writer_ != NULL && lat_ != NULL);
      if (lat_->NumStates() == 0) {
        KALDI_WARN << "Empty lattice for utterance " << utt_;
      } else {
        lattice_writer_->Write(utt_, *lat_);
      }
      delete lat_;
      lat_ = NULL;
    }

    // Print out logging information.
    KALDI_LOG << "Log-like per frame for utterance " << utt_ << " is "
              << (likelihood / num_frames) << " over "
              << num_frames << " frames.";
    KALDI_VLOG(2) << "Cost for utterance " << utt_ << " is "
                  << weight.Value1() << " + " << weight.Value2();

    // Now output the various diagnostic variables.
    if (like_sum_ != NULL) *like_sum_ += likelihood;
    if (frame_sum_ != NULL) *frame_sum_ += num_frames;
    if (num_done_ != NULL) (*num_done_)++;
    if (partial_ && num_partial_ != NULL) (*num_partial_)++;
  }
  // We were given ownership of these two objects that were passed in in
  // the initializer.
  delete decoder_;
  delete decodable_;

  }
 private:
  // The following variables correspond to inputs:
  LatticeBiglmFasterDecoder *decoder_;
  DecodableInterface *decodable_;
  const TransitionModel *trans_model_;
  const fst::SymbolTable *word_syms_;
  std::string utt_;
  BaseFloat acoustic_scale_;
  bool determinize_;
  bool allow_partial_;
  Int32VectorWriter *alignments_writer_;
  Int32VectorWriter *words_writer_;
  CompactLatticeWriter *compact_lattice_writer_;
  LatticeWriter *lattice_writer_;
  double *like_sum_;
  int64 *frame_sum_;
  int32 *num_done_;
  int32 *num_err_;
  int32 *num_partial_;

  // The following variables are stored by the computation.
  bool computed_; // operator ()  was called.
  bool success_; // decoding succeeded (possibly partial)
  bool partial_; // decoding was partial.
  CompactLattice *clat_; // Stored output, if determinize_ == true.
  Lattice *lat_; // Stored output, if determinize_ == false.
};



}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;
    using fst::ReadFstKaldi;

    const char *usage =
        "Generate lattices using on-the-fly composition.\n"
        "e.g. HCLG_1 - G_1 + (G_2a \\dynamic_int G_2b) \n"
        "User supplies LM used to generate decoding graph, and desired LM;\n"
        "this decoder applies the difference during decoding\n"
        "Usage: latgen-fasterlm-faster-mapped [options] model-in(for ctc, the model is ignored) HCLG-1-fstin "
        "G-1-oldlm G-1-weight G-2a-newlm G-2a-weight G-2b-newlm G-2b-weight G-2c... features-rspecifier"
        " lattice-wspecifier  words-wspecifier \n"
        "Notably, we always make G-1-weight = -1\n"
        "ctc example: /fgfs/users/zhc00/works/dyn_dec/kaldi_ctc/README\n"
        "hmm example: /fgfs/users/zhc00/works/dyn_dec/kaldi_minilibri/README\n"
        ;
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    int32 symbol_size = 0;
    bool ctc = false;
    LatticeBiglmFasterDecoderConfig config;
    TaskSequencerConfig sequencer_config; // has --num-threads option
    config.Register(&po);

    ArpaParseOptions arpa_options;
    arpa_options.Register(&po);
    sequencer_config.Register(&po);
    po.Register("ctc", &ctc, "is ctc decoding");
    po.Register("symbol-size", &symbol_size, "symbol table size");
    po.Register("unk-symbol", &arpa_options.unk_symbol,
                "Integer corresponds to unknown-word in language model. -1 if "
                "no such word is provided.");
    po.Register("bos-symbol", &arpa_options.bos_symbol,
                "Integer corresponds to <s>. You must set this to your actual "
                "BOS integer.");
    po.Register("eos-symbol", &arpa_options.eos_symbol,
                "Integer corresponds to </s>. You must set this to your actual "
                "EOS integer.");


    std::string word_syms_filename;
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 6 ) {
      po.PrintUsage();
      exit(1);
    }
   
    int start_lm = 3;
    int end_lm = po.NumArgs() - 3;
    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(po.NumArgs() - 2),
        lattice_wspecifier = po.GetArg(po.NumArgs() - 1),
        words_wspecifier = po.GetOptArg(po.NumArgs());
 
    assert((end_lm - start_lm+1) % 2 == 0); // one lm one weight
    //old_lm_fst_rxfilename = po.GetArg(3),
    //new_lm_fst_rxfilename = po.GetArg(4),   

    TransitionModel trans_model;
    if (!ctc)
        ReadKaldiObject(model_in_filename, &trans_model);

    /*
    FasterArpaLm old_lm;
    ReadKaldiObject(old_lm_fst_rxfilename, &old_lm);
    FasterArpaLmDeterministicFst old_lm_dfst(old_lm);
    ApplyProbabilityScale(-1.0, old_lm_dfst); // Negate old LM probs...
    */
    int lm_num=(end_lm-start_lm+1)/2;
    std::vector<FasterArpaLm> lm_vec;
    std::vector<FasterArpaLmDeterministicFst> dlm_vec;
    std::vector<fst::ComposeDeterministicOnDemandFst<StdArc>> clm_vec;
    lm_vec.reserve(lm_num);
    dlm_vec.reserve(lm_num);
    clm_vec.reserve(lm_num-1);
    for ( int i = start_lm; i < end_lm; i+=2 ) {
      std::string s_lm = po.GetArg(i);
      float w =  atof(po.GetArg(i+1).c_str());
      lm_vec.emplace_back(arpa_options, s_lm, symbol_size, w);
      dlm_vec.emplace_back(lm_vec.back());
      if (i == start_lm) continue;
      else if (i == start_lm+2) {
        clm_vec.emplace_back(&dlm_vec.front(),&dlm_vec.back());
      } else {
        clm_vec.emplace_back(&clm_vec.back(),&dlm_vec.back());
      }
    }
    // multiple compose
    fst::CacheDeterministicOnDemandFst<StdArc> cache_dfst(&clm_vec.back(), 1e7);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    //Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    double elapsed=0;
    TaskSequencer<DecodeUtteranceLatticeBiglmFasterClass> sequencer(sequencer_config);
    Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      timer.Reset();
      {
        for (; !loglike_reader.Done(); loglike_reader.Next()) {
          std::string utt = loglike_reader.Key();
          Matrix<BaseFloat> *loglikes =
            new Matrix<BaseFloat>(loglike_reader.Value());
          loglike_reader.FreeCurrent();
          if (loglikes->NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            delete loglikes;
            continue;
          }

          LatticeBiglmFasterDecoder* decoder = new LatticeBiglmFasterDecoder(
                  *decode_fst, config, &cache_dfst);
          DecodableInterface* decodable = NULL;
          if (!ctc) 
            decodable = new DecodableMatrixScaledMapped(trans_model, *loglikes, acoustic_scale);
          else {
            decodable = new DecodableMatrixScaledMappedCtc(*loglikes, acoustic_scale);
            decoder->GetOptions().det_opts.phone_determinize = false; // disable DeterminizeLatticePhonePrunedFirstPass
          }
          DecodeUtteranceLatticeBiglmFasterClass *task =
              new DecodeUtteranceLatticeBiglmFasterClass(
                  decoder, decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, NULL,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &tot_like, &frame_count, &num_success, &num_fail, NULL);

          sequencer.Run(task); // takes ownership of "task",
          // and will delete it when done.
        }
      }
    } else { // We have different FSTs for different utterances.
      assert(0);
    }
    sequencer.Wait();
    elapsed = timer.Elapsed();
    delete decode_fst;
      
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
