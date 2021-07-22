
// Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)

#include <chrono>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"


using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::Fst;
using fst::StdArc;


int main(int argc, char *argv[]) {
try {
    const char *usage =
        "Generate lattices, reading emulating am as matrices\n"
        " (model is needed only for the integer mappings in its transition-model)\n"
        "Usage: latgen-faster-mapped-fake-am [options] trans-model-in fst-in fam-rspecifier ali_rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
	int seed = 0;
	std::string temperature_str ("1.0");
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename;
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");
	po.Register("seed", &seed, "Random seed");
    po.Register("temperature", &temperature_str, "Temperature for average loglikes distribution. Can be like min_max for random T");

    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }
	
	// Random Temperature
	srand(seed);
	KALDI_LOG << "Random seed is "<< seed ;
	std::size_t T_spliter_id = temperature_str.find("_");
	bool rand_T = ( T_spliter_id != std::string::npos );
	BaseFloat temperature = 1.0;
	BaseFloat T_min = 1.0;
	BaseFloat T_max = 1.0;
	if (rand_T) {
		std::string::size_type sz; 
		T_min = std::stof(temperature_str, &sz);
		T_max = std::stof(temperature_str.substr(sz+1));
		KALDI_LOG << "Uing random temperature. Min = " << T_min << " Max = " << T_max ;

	} else {
		temperature = std::stof(temperature_str);
		KALDI_LOG << "Using constant temperature " << temperature ; 
	}
	// end


    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        fam_rspecifier = po.GetArg(3),
        ali_rspecifier = po.GetArg(4),
        lattice_wspecifier = po.GetArg(5),
        words_wspecifier = po.GetOptArg(6),
        alignment_wspecifier = po.GetOptArg(7);

    //### prue kaldi code
    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
     // #### end prue kaldi code ###

     // Reading Fake acoustic model form ark file
	 KALDI_LOG << "Loading Fake Acoustic Model";
     SequentialBaseFloatMatrixReader fam_model_read(fam_rspecifier);
     std::string fam_model_key = fam_model_read.Key();
     Matrix<BaseFloat> fam_model (fam_model_read.Value());
     KALDI_LOG << "Apply log." ;
	 fam_model.ApplyLog();
	 //if ( ! rand_T ) 
	 //	 fam_model.Scale(1/temperature);
     if (fam_model_key != "fam_model"){
        KALDI_ERR << fam_rspecifier << " - Wrong fam_model." ;
        po.PrintUsage();
        exit(1);
     }
	 Timer timer_loglikes;
	 KALDI_LOG << "Fake Acoustic loaded. Logarithm is applied. Shape is (" << fam_model.NumRows() << ", " << fam_model.NumCols() << ")";
     SequentialInt32VectorReader ali_reader(ali_rspecifier);
     if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      //SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
        LatticeFasterDecoder decoder(*decode_fst, config);
        for (; !ali_reader.Done(); ali_reader.Next()) {
		  timer_loglikes.Reset();
          std::string utt = ali_reader.Key();
          std::vector<int32> ali (ali_reader.Value());
		  KALDI_LOG << "Process " << utt << ". " << ali.size() << " frames"; 
          ali_reader.FreeCurrent();
           if (ali.size() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }
          // Inference fake AM
          kaldi::Matrix<BaseFloat> loglikes(ali.size(), fam_model.NumRows());
		  loglikes.SetZero();
          for (int i=0 ; i < ali.size() ; i++){
            int32 pdf_id = ali[i];
            loglikes.CopyRowFromVec(fam_model.Row(pdf_id), i);
			SubVector<BaseFloat> row(loglikes, i);
			if (rand_T) 
				temperature = T_min + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (T_max - T_min) ;
			row.Scale(1/temperature) ; 

          }
          //end
		  KALDI_LOG << "Loglikes generated. Elapsed time is " << timer_loglikes.Elapsed();
          DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);

          double like;
          if (DecodeUtteranceLatticeFaster(
                  decoder, decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &like)) {
            tot_like += like;
            frame_count += loglikes.NumRows();
            num_success++;
          } else num_fail++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
        KALDI_LOG << "FSTs not implemented yet.";
        exit(1);
//      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
//      RandomAccessBaseFloatMatrixReader loglike_reader(feature_rspecifier);
//      for (; !fst_reader.Done(); fst_reader.Next()) {
//        std::string utt = fst_reader.Key();
//        if (!loglike_reader.HasKey(utt)) {
//          KALDI_WARN << "Not decoding utterance " << utt
//                     << " because no loglikes available.";
//          num_fail++;
//          continue;
//        }
//        const Matrix<BaseFloat> &loglikes = loglike_reader.Value(utt);
//        if (loglikes.NumRows() == 0) {
//          KALDI_WARN << "Zero-length utterance: " << utt;
//          num_fail++;
//          continue;
//        }
//        LatticeFasterDecoder decoder(fst_reader.Value(), config);
//        DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);
//        double like;
//        if (DecodeUtteranceLatticeFaster(
//                decoder, decodable, trans_model, word_syms, utt, acoustic_scale,
//                determinize, allow_partial, &alignment_writer, &words_writer,
//                &compact_lattice_writer, &lattice_writer, &like)) {
//          tot_like += like;
//          frame_count += loglikes.NumRows();
//          num_success++;
//        } else num_fail++;
//      }
    }

    double elapsed = timer.Elapsed();
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

