// hmm/hmm-utils.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_HMM_HMM_UTILS_H_
#define KALDI_HMM_HMM_UTILS_H_

#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {


/// \defgroup hmm_group_graph Classes and functions for creating FSTs from HMMs
/// \ingroup hmm_group
/// @{

/// Configuration class for the GetHTransducer() function; see
/// \ref hmm_graph_config for context.
struct HTransducerConfig {
  /// Transition log-prob scale, see \ref hmm_scale.
  /// Note this doesn't apply to self-loops; GetHTransducer() does
  /// not include self-loops.
  BaseFloat transition_scale;
  int32 nonterm_phones_offset;

  HTransducerConfig():
      transition_scale(1.0),
      nonterm_phones_offset(-1) { }

  void Register (OptionsItf *opts) {
    opts->Register("transition-scale", &transition_scale,
                   "Scale of transition probs (relative to LM)");
    opts->Register("nonterm-phones-offset", &nonterm_phones_offset,
                   "The integer id of #nonterm_bos in phones.txt, if present. "
                   "Only needs to be set if you are doing grammar decoding, "
                   "see doc/grammar.dox.");
  }
};


struct HmmCacheHash {
  int operator () (const std::pair<int32, std::vector<int32> >&p) const {
    VectorHasher<int32> v;
    int32 prime = 103049;
    return prime*p.first + v(p.second);
  }
};

/// HmmCacheType is a map from (central-phone, sequence of pdf-ids) to FST, used
/// as cache in GetHmmAsFsa, as an optimization.
typedef unordered_map<std::pair<int32, std::vector<int32> >,
                      fst::VectorFst<fst::StdArc>*,
                      HmmCacheHash> HmmCacheType;


/// Called by GetHTransducer() and probably will not need to be called directly;
/// it creates and returns the FST corresponding to the phone.  It's actually an
/// acceptor (ilabels equal to olabels), which is why this is called "Fsa" not
/// "Fst".  This acceptor does not include self-loops; you have to call
/// AddSelfLoops() for that.  (We do that at a later graph compilation phase,
/// for efficiency).  The labels on the FSA correspond to transition-ids.
///
/// as the symbols.
/// For documentation in context, see \ref hmm_graph_get_hmm_as_fst
///   @param context_window  A vector representing the phonetic context; see
///            \ref tree_window "here" for explanation.
///   @param ctx_dep The object that contains the phonetic decision-tree
///   @param trans_model The transition-model object, which provides
///         the mappings to transition-ids and also the transition
///         probabilities.
///   @param config Configuration object, see \ref HTransducerConfig.
///   @param cache Object used as a lookaside buffer to save computation;
///       if it finds that the object it needs is already there, it will
///       just return a pointer value from "cache"-- not that this means
///       you have to be careful not to delete things twice.
fst::VectorFst<fst::StdArc> *GetHmmAsFsa(
    std::vector<int32> context_window,
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model,
    const HTransducerConfig &config,
    HmmCacheType *cache = NULL);


/// Included mainly as a form of documentation, not used in any other code
/// currently.  Creates the acceptor FST with self-loops, and with fewer
/// options.
fst::VectorFst<fst::StdArc>*
GetHmmAsFsaSimple(std::vector<int32> context_window,
                  const ContextDependencyInterface &ctx_dep,
                  const TransitionModel &trans_model,
                  BaseFloat prob_scale);


/**
  * Returns the H tranducer; result owned by caller.  Caution: our version of
  * the H transducer does not include self-loops; you have to add those later.
  * See \ref hmm_graph_get_h_transducer.  The H transducer has on the
  * input transition-ids, and also possibly some disambiguation symbols, which
  * will be put in disambig_syms.  The output side contains the identifiers that
  * are indexes into "ilabel_info" (these represent phones-in-context or
  * disambiguation symbols).  The ilabel_info vector allows GetHTransducer to map
  * from symbols to phones-in-context (i.e. phonetic context windows).  Any
  * singleton symbols in the ilabel_info vector which are not phones, will be
  * treated as disambiguation symbols.  [Not all recipes use these].  The output
  * "disambig_syms_left" will be set to a list of the disambiguation symbols on
  * the input of the transducer (i.e. same symbol type as whatever is on the
  * input of the transducer
  */
fst::VectorFst<fst::StdArc>*
GetHTransducer(const std::vector<std::vector<int32> > &ilabel_info,
               const ContextDependencyInterface &ctx_dep,
               const TransitionModel &trans_model,
               const HTransducerConfig &config,
               std::vector<int32> *disambig_syms_left);

/**
  * GetIlabelMapping produces a mapping that's similar to HTK's logical-to-physical
  * model mapping (i.e. the xwrd.clustered.mlist files).   It groups together
  * "logical HMMs" (i.e. in our world, phonetic context windows) that share the
  * same sequence of transition-ids.   This can be used in an
  * optional graph-creation step that produces a remapped form of CLG that can be
  * more productively determinized and minimized.  This is used in the command-line program
  * make-ilabel-transducer.cc.
  * @param ilabel_info_old [in] The original \ref tree_ilabel "ilabel_info" vector
  * @param ctx_dep [in] The tree
  * @param trans_model [in] The transition-model object
  * @param old2new_map [out] The output; this vector, which is of size equal to the
  *       number of new labels, is a mapping to the old labels such that we could
  *       create a vector ilabel_info_new such that
  *       ilabel_info_new[i] == ilabel_info_old[old2new_map[i]]
  */
void GetIlabelMapping(const std::vector<std::vector<int32> > &ilabel_info_old,
                      const ContextDependencyInterface &ctx_dep,
                      const TransitionModel &trans_model,
                      std::vector<int32> *old2new_map);



/**
  * For context, see \ref hmm_graph_add_self_loops.  Expands an FST that has been
  * built without self-loops, and adds the self-loops (it also needs to modify
  * the probability of the non-self-loop ones, as the graph without self-loops
  * was created in such a way that it was stochastic).  Note that the
  * disambig_syms will be empty in some recipes (e.g.  if you already removed
  * the disambiguation symbols).
  * This function will treat numbers over 10000000 (kNontermBigNumber) the
  * same as disambiguation symbols, assuming they are special symbols for
  * grammar decoding.
  *
  * @param trans_model [in] Transition model
  * @param disambig_syms [in] Sorted, uniq list of disambiguation symbols, required
  *       if the graph contains disambiguation symbols but only needed for sanity checks.
  * @param self_loop_scale [in] Transition-probability scale for self-loops; c.f.
  *                    \ref hmm_scale
  * @param reorder [in] If true, reorders the transitions (see \ref hmm_reorder).
  *                     You'll normally want this to be true.
  * @param check_no_self_loops [in]  If true, it will check that there are no
  *                      self-loops in the original graph; you'll normally want
  *                      this to be true.  If false, it will allow them, and
  *                      will add self-loops after the original self-loop
  *                      transitions, assuming reorder==true... this happens to
  *                      be what we want when converting normal to unconstrained
  *                      chain examples.  WARNING: this was added in 2018;
  *                      if you get a compilation error, add this as 'true',
  *                      which emulates the behavior of older code.
  * @param  fst [in, out] The FST to be modified.
  */
void AddSelfLoops(const TransitionModel &trans_model,
                  const std::vector<int32> &disambig_syms,  // used as a check only.
                  BaseFloat self_loop_scale,
                  bool reorder,
                  bool check_no_self_loops,
                  fst::VectorFst<fst::StdArc> *fst);

/**
  * Adds transition-probs, with the supplied
  * scales (see \ref hmm_scale), to the graph.
  * Useful if you want to create a graph without transition probs, then possibly
  * train the model (including the transition probs) but keep the graph fixed,
  * and add back in the transition probs.  It assumes the fst has transition-ids
  * on it.  It is not an error if the FST has no states (nothing will be done).
  * @param trans_model [in] The transition model
  * @param disambig_syms [in] A list of disambiguation symbols, required if the
  *                       graph has disambiguation symbols on its input but only
  *                       used for checks.
  * @param transition_scale [in] A scale on transition-probabilities apart from
  *                      those involving self-loops; see \ref hmm_scale.
  * @param self_loop_scale [in] A scale on self-loop transition probabilities;
  *                      see \ref hmm_scale.
  * @param  fst [in, out] The FST to be modified.
  */
void AddTransitionProbs(const TransitionModel &trans_model,
                        const std::vector<int32> &disambig_syms,
                        BaseFloat transition_scale,
                        BaseFloat self_loop_scale,
                        fst::VectorFst<fst::StdArc> *fst);

/**
   This is as AddSelfLoops(), but operates on a Lattice, where
   it affects the graph part of the weight (the first element
   of the pair). */
void AddTransitionProbs(const TransitionModel &trans_model,
                        BaseFloat transition_scale,
                        BaseFloat self_loop_scale,
                        Lattice *lat);


/// Returns a transducer from pdfs plus one (input) to  transition-ids (output).
/// Currenly of use only for testing.
fst::VectorFst<fst::StdArc>*
GetPdfToTransitionIdTransducer(const TransitionModel &trans_model);

/// Converts all transition-ids in the FST to pdfs plus one.
/// Placeholder: not implemented yet!
void ConvertTransitionIdsToPdfs(const TransitionModel &trans_model,
                                const std::vector<int32> &disambig_syms,
                                fst::VectorFst<fst::StdArc> *fst);

/// @} end "defgroup hmm_group_graph"

/// \addtogroup hmm_group
/// @{

/// SplitToPhones splits up the TransitionIds in "alignment" into their
/// individual phones (one vector per instance of a phone).  At output,
/// the sum of the sizes of the vectors in split_alignment will be the same
/// as the corresponding sum for "alignment".  The function returns
/// true on success.  If the alignment appears to be incomplete, e.g.
/// not ending at the end-state of a phone, it will still break it up into
/// phones but it will return false.  For more serious errors it will
/// die or throw an exception.
/// This function works out by itself whether the graph was created
/// with "reordering", and just does the right thing.
bool SplitToPhones(const TransitionModel &trans_model,
                   const std::vector<int32> &alignment,
                   std::vector<std::vector<int32> > *split_alignment);

/**
   ConvertAlignment converts an alignment that was created using one model, to
   another model.  Returns false if it could not be split to phones (e.g.
   because the alignment was partial), or because some other error happened,
   such as we couldn't convert the alignment because there were too few frames
   for the new topology.

   @param old_trans_model [in]  The transition model that the original alignment
                                used.
   @param new_trans_model [in]  The transition model that we want to use for the
                                new alignment.
   @param new_ctx_dep     [in]  The new tree
   @param old_alignment   [in]  The alignment we want to convert
   @param subsample_factor [in] The frame subsampling factor... normally 1, but
                                might be > 1 if we're converting to a reduced-frame-rate
                                system.
   @param repeat_frames [in]    Only relevant when subsample_factor != 1
                                If true, repeat frames of alignment by
                                'subsample_factor' after alignment
                                conversion, to keep the alignment the same
                                length as the input alignment.
                                [note: we actually do this by interpolating
                                'subsample_factor' separately generated
                                alignments, to keep the phone boundaries
                                the same as the input where possible.]
   @param reorder [in]          True if you want the pdf-ids on the new alignment to
                                be 'reordered'. (vs. the way they appear in
                                the HmmTopology object)
   @param phone_map [in]        If non-NULL, map from old to new phones.
   @param new_alignment [out]   The converted alignment.
*/

bool ConvertAlignment(const TransitionModel &old_trans_model,
                      const TransitionModel &new_trans_model,
                      const ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      int32 subsample_factor,  // 1 in the normal case -> no subsampling.
                      bool repeat_frames,
                      bool reorder,
                      const std::vector<int32> *phone_map,  // may be NULL
                      std::vector<int32> *new_alignment);

// ConvertPhnxToProns is only needed in bin/phones-to-prons.cc and
// isn't closely related with HMMs, but we put it here as there isn't
// any other obvious place for it and it needs to be tested.
// This function takes a phone-sequence with word-start and word-end
// markers in it, and a word-sequence, and outputs the pronunciations
// "prons"... the format of "prons" is, each element is a vector,
// where the first element is the word (or zero meaning no word, e.g.
// for optional silence introduced by the lexicon), and the remaining
// elements are the phones in the word's pronunciation.
// It returns false if it encounters a problem of some kind, e.g.
// if the phone-sequence doesn't seem to have the right number of
// words in it.
bool ConvertPhnxToProns(const std::vector<int32> &phnx,
                        const std::vector<int32> &words,
                        int32 word_start_sym,
                        int32 word_end_sym,
                        std::vector<std::vector<int32> > *prons);


/* Generates a random alignment for this phone, of length equal to
   alignment->size(), which is required to be at least the MinLength() of the
   topology for this phone, or this function will crash.
   The alignment will be without 'reordering'.
*/
void GetRandomAlignmentForPhone(const ContextDependencyInterface &ctx_dep,
                                const TransitionModel &trans_model,
                                const std::vector<int32> &phone_window,
                                std::vector<int32> *alignment);

/*
  If the alignment was non-reordered makes it reordered, and vice versa.
*/
void ChangeReorderingOfAlignment(const TransitionModel &trans_model,
                                 std::vector<int32> *alignment);


// GetPdfToPhonesMap creates a map which maps each pdf-id into its
// corresponding monophones.
void GetPdfToPhonesMap(const TransitionModel &trans_model,
                       std::vector<std::set<int32> > *pdf2phones);

/// @} end "addtogroup hmm_group"

} // end namespace kaldi


#endif
