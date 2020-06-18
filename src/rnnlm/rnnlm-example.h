// rnnlm/rnnlm-example.h

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_RNNLM_RNNLM_EXAMPLE_H_
#define KALDI_RNNLM_RNNLM_EXAMPLE_H_

#include "base/kaldi-common.h"
#include "util/kaldi-thread.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "rnnlm/sampling-lm.h"
#include "rnnlm/sampler.h"


namespace kaldi {
namespace rnnlm {


// A single minibatch for training an RNNLM.
struct RnnlmExample {
  int32 vocab_size;     // The vocabulary size (defined as largest integer word-id
                        // plus one) for which this example was obtained; mostly
                        //  used in bounds checking.
  int32 num_chunks;     // The number of parallel word sequences/chunks.  [note:
                        // some of the word sequences may actually be made up of
                        // smaller subsequences appended together.
  int32 chunk_length;   // The length of each sequence in a minibatch,
                        // including any terminating </s> symbols, which are
                        // included explicitly in the sequences.  Note:
                        // when </s> appears in the middle of sequences because
                        // we splice shorter word-sequences together, we
                        // will replace it with <s> on the input side of the network.
                        // Sentences, or pieces of sentences, that were shorter
                        // than 'chunk_length', will be padded as needed.

  int32 sample_group_size;  // derived from the sample_group_size option; this
                            // is the number of consecutive time-steps which
                            // form a single unit for sampling purposes.  This
                            // number Will always divide chunk_length.
                            // Example: if sample_group_size=2, we'll sample one
                            // set of words for t={0,1}, another for t={2,3}, and
                            // so on.  The sampling is for the denominator of
                            // the objective function.

  int32 num_samples;   // This is the number of words that we sample at the
                       // output of the nnet for each of the 'num_sample_groups'
                       // groups.  If we didn't do sampling because the user
                       // didn't provide the ARPA language model, this will be
                       // zero (in this case we'll do the summation over all
                       // words in the vocab).


  std::vector<int32> input_words;  // Contains the input word symbols 0 <= i <
                                   // vocab_size for each position in each
                                   // chunk; dimension == chunk_length *
                                   // num_chunks, where 0 <= t < chunk_length
                                   // has larger stride than 0 <= n <
                                   // num_chunks.  In the common case these will
                                   // be the same as the previous output symbol.
  std::vector<int32> output_words;  // The output (predicted) word symbols for
                                    // each position in each chunk; indexed in
                                    // the same way as 'input_words'.  What this
                                    // contains is different from 'input_words'
                                    // in the sampling case (i.e. if
                                    // !sampled_words.empty()).  In this case,
                                    // instead of the word-index it contains the
                                    // relative index 0 <= i < num_samples
                                    // within the block of sampled words.  In
                                    // the not-sampled case it contains actual
                                    // word indexes 0 <= i < vocab_size.



  // Weights for each of the output_words, indexed the same way as
  // 'output_words'.  These reflect any data-weighting we had in the original
  // data, plus some zeros that relate to padding sequences of uneven length.
  CuVector<BaseFloat> output_weights;

  // This vector contains the word-indexes that we sampled for each position in
  // the chunk and for each group of chunks.  (It will be empty if the
  // user didn't provide the ARPA language model).  Its dimension is
  // num_sample_groups * num_samples, where
  // num_sample_groups == (chunk_length / sample_group_size).
  // The sample-group index has the largest stride (you can think of the sample
  // group index as the number i = t / sample_group_size, in integer division,
  // where 0 <= t < chunk_length is the position in the chunk).  The sampled
  // words within each block of size 'num_samples' are sorted and unique.
  std::vector<int32> sampled_words;

  // This vector has the same dimension as 'sampled_words', and contains the
  // inverses of the probabilities probability 0 < p <= 1 with which that word
  // was included in the sampled set of words.  These inverse probabilities
  // appear in the objective function computation (it's related to importance
  // sampling).
  CuVector<BaseFloat> sample_inv_probs;

  RnnlmExample(): vocab_size(0), num_chunks(0), chunk_length(0),
                  sample_group_size(1), num_samples(0) { }


  // Shallow swap.
  void Swap(RnnlmExample *other);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);
};


// The word symbols we train on are zero-based integers
// (although zero is reserved for <eps> so this won't appear
// as a word
// we reserve 0 and 1 for the BOS symbol (usually <s>)
// and the EOS symbol (usually </s>) respectively.
// There are some subtleties regarding how we prepare the
// sentences and arrange them into minibatches.
//
// Firstly, we prepare original data that looks like the following:
//
// 1.0  Hello there
// 1.0  How are you
//
// [All of these words will be turned into integers by sym2int.pl before the C++
// tools are run].
//
// We want the models to be able to take advantage of context from previous
// sentences (say, in a dialogue or a series of sentences).  Therefore we allow
// the data preparation to create include multiple successive sentences on one
// line, as follows:
//
// 1.0  Hello there </s>  How are you
//
// The immedate left-context which the RNNLM sees when predicting "How" above
// will be </s>, unlike the normal case at start-of-sentence where it would be
// <s>.  So essentially </s>, when seen as left-context, means "we're beginning
// a new sentence here but the prior words were part of the same dialogue."
//
//
// We train on minibatches of a fixed size, so it may be necessary to split up
// or combine sentences to a fixed length.  Internally (inside
// rnnlm-create-egs), we read these variable-length sentences, randomize the
// order using a buffer, and then split and combine them as necessary to
// obtain a fixed length.
//
// The first stage in processing the sentences is to add an initial <s> and
// final <s> and to associate each word with a weight (which is just the
// sentence weight, at this point (1.0 in the example), except for the initial
// <s> which has zero weight).
//
//
// Next we split sentences which (when the </s> is included but not the <s>) are
// longer than chunk_length, into multiple pieces.  At the split points, to the
// RHS of each split we will add some left-context words, up to
// min_split_context (e.g. 3), and we'll give those words zero weight so the
// RNNLM doesn't try to predict them, they are just used as left-context to
// predict future words.  (The reason for the zero weight is to avoid counting
// these words twice).



struct RnnlmEgsConfig {
 int32 vocab_size;  // The vocabulary size: or more specifically, the largest
                     // integer word-id plus one.  Must be provided, as it
                     // gets included in each minibatch (mostly for checking
                     // purposes).
  int32 num_chunks_per_minibatch;
  int32 chunk_length;
  int32 min_split_context;
  int32 sample_group_size;  // affects the sampling: if sample_group_size is 2,
                            // then we'll sample words once for (t=0, t=1), then
                            // again for (t=2, t=3), and so on.  We support
                            // merging time-steps in this way (but not splitting
                            // them smaller), due to considerations of computing
                            // time if you assume we also have a network that
                            // learns word representation from their
                            // character-level features.
  int32 num_samples;    // the number of words we choose each time we do the
                        // sampling.
  int32 chunk_buffer_size;
  int32 bos_symbol;  // must be set.
  int32 eos_symbol;  // must be set.
  int32 brk_symbol;  // must be set.

  BaseFloat special_symbol_prob;  // sampling probability at the output for
                                  // words that aren't supposed to be predicted
                                  // (<s>, <brk>)-- this ensures that the model
                                  // makes their output probs small, which
                                  // avoids hassle when computing the normalizer
                                  // in test time (if we didn't sample them with
                                  // some probability to ensure their probs are
                                  // small, we'd have to exclude them from the
                                  // denominator sum.

  BaseFloat uniform_prob_mass;  // this value should be < 1.0; it takes this
                                // proportion of the unigram distribution used
                                // for sampling and assigns it to uniformly
                                // predict all words.  This may avoid certain
                                // pathologies during training, and ensuring
                                // that all words' probs are bounded away from
                                // zero might be necessary for the theory of
                                // importance sampling.
  RnnlmEgsConfig(): vocab_size(-1),
                    num_chunks_per_minibatch(128),
                    chunk_length(32),
                    min_split_context(3),
                    sample_group_size(2),
                    num_samples(512),
                    chunk_buffer_size(20000),
                    bos_symbol(1),  // we use standardized values in the scripts,
                    eos_symbol(2),  // so these are sensible defaults.
                    brk_symbol(3),
                    special_symbol_prob(1.0e-05),
                    uniform_prob_mass(0.05) { }

  void Register(OptionsItf *po) {
    po->Register("vocab-size", &vocab_size,
                 "Size of the vocabulary (more specifically: the largest integer "
                 "word-id plus one).");
    po->Register("chunk-length", &chunk_length,
                 "Length of sequences that we train on (actual sentences will be "
                 "split up and re-combined as necessary to achieve this legnth");
    po->Register("num-chunks-per-minibatch", &num_chunks_per_minibatch,
                 "Number of distinct sequences/chunks per minibatch.");
    po->Register("min-split-context", &min_split_context,
                 "Minimum left-context that we supply after breaking up "
                 "a training sequence into pieces.");
    po->Register("sample-group-size", &sample_group_size,
                 "Number of time-steps for which we draw a single sample of words. "
                 "Must divide chunk-length.");
    po->Register("num-samples", &num_samples,
                 "Number of words we sample, each time we sample (importance sampling). "
                 "Must be at least num-chunks-per-minibatch * sample-group-size.  "
                 "If you don't supply the ARPA LM to the program, or you set "
                 "num-samples to zero, or num-samples exceeds the number of words "
                 "with nonzero probability, then no sampling will be done.");
    po->Register("chunk-buffer-size", &chunk_buffer_size,
                 "Number of chunks of sentence that we buffer while "
                 "processing the input.  Larger means more complete "
                 "randomization but also more I/O before we produce any "
                 "output, and more memory used.");
    po->Register("bos-symbol", &bos_symbol,
                 "Integer id of the beginning-of-sentence symbol <s>. "
                 "Must be specified.");
    po->Register("eos-symbol", &eos_symbol,
                 "Integer id of the beginning-of-sentence symbol <s>. "
                 "Must be specified.");
    po->Register("brk-symbol", &brk_symbol,
                 "Integer id of the 'break' symbol <brk> (only used "
                 "during training, most likely); used to tell the network "
                 "that the context is partial.  Must be specified.");
    po->Register("special-symbol-prob", &special_symbol_prob,
                 "Probability with which we sample the special symbols "
                 "<s> and <brk> on each minibatch.  See code for reason.");
    po->Register("uniform-prob-mass", &uniform_prob_mass,
                 "We replace this proportion of the unigram distribution's "
                 "probability mass with a uniform distribution over words. "
                 "Probably not necessary or important.");
  }
  // Checks that the config makes sense, and dies if not.
  void Check() const {
    KALDI_ASSERT(chunk_length > min_split_context * 4 &&
                 num_chunks_per_minibatch > 0 &&
                 min_split_context >= 0 &&
                 sample_group_size >= 1 &&
                 chunk_length % sample_group_size == 0);
    if (vocab_size <= 0) {
      KALDI_ERR << "The --vocab-size option must be provided.";
    }
    if (!(bos_symbol > 0 && eos_symbol > 0 && brk_symbol > 0 &&
          bos_symbol != eos_symbol && brk_symbol != eos_symbol &&
          brk_symbol != bos_symbol)) {
      KALDI_ERR << "--bos-symbol, --eos-symbol and --brk-symbol "
          "must be specified, >0, and all different.";
    }
    KALDI_ASSERT(num_samples == 0 ||
                 num_samples >= num_chunks_per_minibatch * sample_group_size);
    KALDI_ASSERT(special_symbol_prob >= 0.0 && special_symbol_prob <= 1.0);
    KALDI_ASSERT(uniform_prob_mass >= 0.0 && uniform_prob_mass < 1.0);
  }
};



/**
   Class RnnlmExampleSampler encapsulates the logic for sampling words
   for a minibatch.  (the words at the output of the RNNLM are sampled and
   we train with an importance-sampling algorithm).
 */
class RnnlmExampleSampler {
 public:
  RnnlmExampleSampler(const RnnlmEgsConfig &config,
                      const SamplingLm &arpa_sampling);


  // Does the sampling for 'minibatch'.  'minibatch' is expected to already
  // have all fields populated except for 'sampled_words' and 'sample_probs'.
  // This function does the sampling and sets those fields.
  void SampleForMinibatch(RnnlmExample *minibatch) const;

  ~RnnlmExampleSampler() { delete sampler_; }

  int32 VocabSize() const {
    return arpa_sampling_.GetUnigramDistribution().size();
  }
 private:
  // does the part of the sampling for group 'g' (note: 'g' is the
  // same as the position 0 <= t < chunk_length in the sequence if
  // config_.sample_group_size == 1, and otherwise, each group
  // encompasses several successive 't' values.
  void SampleForGroup(int32 g, RnnlmExample *minibatch) const;


  // This function gets the combination of histories to be sampled from for the g'th
  // group of 't' values, for this minibatch.
  // It outputs to 'history_states' the the weighted
  // combination of history-states, as a list of pair (history, weight)
  // [with each history repeated only once], where for example
  // history == [] is the unigram backoff state, history=[10] means
  // we saw the word 10 as left-context, history[20, 10] means 10 is
  // the immediate left-contxt and 20 is before that.  The weight
  // 'weight' will be > 0 and will be a sum of the weights of the
  // output words in the minibatch that have that history.
  void GetHistoriesForGroup(
      int32 g, const RnnlmExample &minibatch,
      std::vector<std::pair<std::vector<int32>, BaseFloat> > *hist_weights) const;

  // This function renumbers 'output_words' so that instead of being
  // numbers 0 <= i < vocab_size, they are numbered as indexes into the
  // relevant block of the vector 'output_words'.
  void RenumberOutputWordsForGroup(
      int32 g, RnnlmExample *minibatch) const;


  // This function is used to obtain the history (of maximum length
  // 'max_history_length') used when predicting the t'th output word in the n'th
  // sequence of this minibatch.  The history is output to 'history'.  Note: the
  // only situation where the history-length would be less than
  // 'max_history_length' is due to edge effects.
  //
  // As an example of a normal case: if max_history_length is 2, and for the
  // provided n, the input words in 'minibatch.input_words[..]'  for t values up
  // to and including 't' are '.. the day of', then 'history' would be set to [
  // day of ] (obviously in integer form).
  //
  // As an example of edge effects: if this is the
  // first word of a chunk that's part of the sequence and max_history_length >
  // 1; in this case the history would either be [<s>] or [<brk>].
  void GetHistory(int32 t, int32 n,
                  const RnnlmExample &minibatch,
                  int32 max_history_length,
                  std::vector<int32> *history) const;



  RnnlmEgsConfig config_;
  // arpa_ stores the n-gram language model that we use for importance sampling.
  const SamplingLm &arpa_sampling_;
  // class Sampler does some of the lower-level aspects of sampling.
  Sampler *sampler_;
};


/// This class takes care of all of the logic of creating minibatches for RNNLM
/// training, including the sampling aspect.  It implements the bulk of the
/// functionality of the binary rnnlm-get-egs.
class RnnlmExampleCreator {
 public:
  // This constructor is for when you are using importance sampling from
  // an ARPA language model (the normal case).
  RnnlmExampleCreator(const RnnlmEgsConfig &config,
                      const TaskSequencerConfig &sequencer_config,
                      const RnnlmExampleSampler &minibatch_sampler,
                      TableWriter<KaldiObjectHolder<RnnlmExample> > *writer):
      config_(config), minibatch_sampler_(&minibatch_sampler),
      sampling_sequencer_(sequencer_config),
      writer_(writer), num_sequences_processed_(0), num_chunks_processed_(0),
      num_words_processed_(0), num_minibatches_written_(0) { Check(); }

  // This constructor is for when you are not using importance sampling,
  // so no samples will be stored in the minibatch and the training code
  // will presumably evaluate all the words each time.  This is intended
  // to be used for testing purposes.
  RnnlmExampleCreator(const RnnlmEgsConfig &config,
                      TableWriter<KaldiObjectHolder<RnnlmExample> > *writer):
      config_(config), minibatch_sampler_(NULL),
      sampling_sequencer_(TaskSequencerConfig()),
      writer_(writer), num_sequences_processed_(0),
      num_chunks_processed_(0), num_words_processed_(0),
      num_minibatches_written_(0) { Check(); }

  // The user calls this to provide a single sequence (a sentence; or multiple
  // sentences that are part of a continuous stream or dialogue, separated
  // by </s>), to this class.  This class will write out minibatches when
  // it's ready.
  // This will normally be the result of reading a line of text with the format:
  //   <weight> <word1> <word2> ....
  // e.g.:
  //   1.0  Hello there
  // [although the "hello there" would have been converted to integers
  // by the time it was read in, via sym2int.pl, so it would look like:
  //   1.0  7620  12309
  // We also allow:
  //   1.0  Hello there </s> Hi </s> My name is Bob
  // if you want to train the model to predict sentences given
  // the history of the conversation.
  void AcceptSequence(BaseFloat weight,
                      const std::vector<int32> &words);


  // Reads the lines from this input stream, calling AcceptSequence() on each
  // one.  Lines will be of the format:
  // <weight> <possibly-empty-sequence-of-integers>
  // e.g.:
  // 1.0  2560 8991
  void Process(std::istream &is);

  // Flush out any pending minibatches.
  void Flush() {
    while (ProcessOneMinibatch());
    sampling_sequencer_.Wait();
  }

  ~RnnlmExampleCreator();
 private:

  void Check() const;

  // Attempts to create a minibatch.  Returns true if it successfully did so,
  // and false if it could not do so because there was insufficient data.
  // If we are not doing sampling, this function will write
  // the minibatch to 'writer_' directly; if we are doing sampling, it will
  // give it to a background thread to be processed and written.
  bool ProcessOneMinibatch();

  struct SequenceChunk {
    // 'sequence' is a pointer to the word sequence (without initial <s>, but with
    // final </s> added by us, and possibly with </s> in the middle to demarcate
    // sentences that are part of a single conversation or piece of text.
    std::shared_ptr<std::vector<int32> > sequence;
    // 'weight' is the weight on this chunk of sequence, i.e. the
    // corpus weighting the user chose to apply on the
    // original sequences.
    BaseFloat weight;


    int32 begin;  // beginning position in the sequence, of the first predicted
                  // word (begin >= 0).
    int32 end;    // one past the end of the last predicted word; will be <= sequence->size().

    int32 context_begin;  // context_begin <= begin is the first word in
                          // the sequence that is seen as left-context.  This will
                          // be the same as 'begin' if begin == 0, but will be less
                          // by up to config_.min_split_context if begin > 0.
                          // note: we actually see one more word of left-context, namely
                          // <s> if context_begin==0 or <brk> otherwise, but this
                          // doesn't affect how many 't' values this sequence uses
                          // up because we get it 'for free' (since we see one word
                          // of left-context even without recurrence).

    SequenceChunk(const RnnlmEgsConfig &config,
                  const std::shared_ptr<std::vector<int32> > &seq,
                  BaseFloat w, int32 b, int32 e):
        sequence(seq), weight(w), begin(b), end(e),
        context_begin(std::max<int32>(0, b - config.min_split_context)) { }


    // The length (in 't' values) that this chunk of a sequence takes up.
    int32 Length() const { return end - context_begin; }
  };

  class SingleMinibatchCreator {
   public:
    SingleMinibatchCreator(const RnnlmEgsConfig &config);


    // The user calls this to ask it to accept a chunk into this
    // minibatch.  It returns true if it can do so, and false if it
    // can't do so because it's too big for any space that remains
    // in this minibatch.
    // If it returns true it will have taken ownership of 'chunk'
    // from a memory management point of view.
    bool AcceptChunk(SequenceChunk *chunk);


    // You call this when you've provided all the data you're going to provide
    // (usually because it already rejected a bunch of chunks due to no space
    // left), and you want to create a minibatch.  You will let this object go
    // out of scope or delete it right after this.
    // This function does everything but the sampling aspect of creating
    // the object 'minibatch'; the caller is responsible for that.
    void CreateMinibatch(RnnlmExample *minibatch);

    ~SingleMinibatchCreator();
   private:
    // called from CreateMinibatch, handles a single sequence
    void CreateMinibatchOneSequence(int32 n, RnnlmExample *minibatch);

    // This function writes to the minibatch for the n'th sequence,
    // (with 0 <= n < config_.minibatch_size), the t'th position
    // (with 0 <= t < config_.chunk_length).
    //   'input_word' is the word the RNNLM sees as its input;
    //   'output_word' is the word the RNNLM predicts as its output
    //     (and this will normally be the same as the 'input_word'
    //     for t+1, except at chunk boundaries);
    //   'weight' is the weight in the objective, for predicting the word
    //     'output_word'.  This is normally the same as the corpus weight for
    //     this data-source, but it could be zero for words that are only used
    //     for context after a split, or for where we are padding a sequence.
    void Set(int32 n, int32 t, int32 input_word, int32 output_word,
             BaseFloat weight, RnnlmExample *minibatch) const;

    const RnnlmEgsConfig &config_;

    // Indexed by 0 < n < config_.num_chunks_per_minibatch, and then a list of
    // SequenceChunk*.  It's a list instead of just one SequenceChunk* because
    // each chunk of the eg we write may actually contain more than one
    // sequence, or fragment of a sequence.  The pointers are owned here.
    std::vector<std::vector<SequenceChunk*> > eg_chunks_;

    // lists all eg_chunks 0 <= n < config_.num_chunks_per_minibatch that
    // are completely empty (i.e. eg_chunks[i].empty()).
    std::vector<int32> empty_eg_chunks_;

    // Lists all eg_chunks that are not empty but not completely full,
    // giving the amount of space left in the eg_chunk, as an unordered list
    // of pairs (n, space_left).
    // What this means specifically is as follows:
    // Let SpaceUsed(n) be equal to \sum_i eg_chunks[n][i]->MinLength(config_),
    // then partial_eg_chunks_ will contain a pair
    // (n, k) where k = config_.chunk_length - SpaceUsed(n),
    // wherever this would give us 0 < k < config_.chunk_length.
    // 'k' represents the largest MinLength() of a SequenceChunk that we
    // would be able to fit in this eg_chunk.
    std::vector<std::pair<int32, int32> > partial_eg_chunks_;
  };


  // This class is a wrapper class that, when provided to class TaskSequencer, allows us to
  // run the call 'sampler.SampleForMinibatch(minibatch)' in multiple threads, followed by
  // sequentially calling writer->Write(key, *minibatch) and deleting minibatch.
  class SamplerTask {
   public:
    SamplerTask(const RnnlmExampleSampler &sampler,
                const std::string &key,
                TableWriter<KaldiObjectHolder<RnnlmExample> > *writer,
                RnnlmExample *minibatch):
        sampler_(sampler), key_(key), writer_(writer), minibatch_(minibatch) { }

    void operator () () {
      sampler_.SampleForMinibatch(minibatch_);
    }
    ~SamplerTask() {
      writer_->Write(key_, *minibatch_);
      delete minibatch_;
    }
   private:
    const RnnlmExampleSampler &sampler_;
    std::string key_;
    TableWriter<KaldiObjectHolder<RnnlmExample> > *writer_;
    RnnlmExample *minibatch_; // owned here.
  };


  // Checks an input sequence, as read directly from the user.
  // Checks that weight > 0.0, that 'words' does not contain
  // <s> or <brk> (see bos_symbol and brk_symbol in the config).
  // Note: it may contain </s> internally to separate sentences
  // that are part of a sequence of utterances, such as a conversation.
  // It's not expected to contain </s> at the end, but this is
  // not checked for, because it's not absolutely disallowed.
  // (it would get processed into </s> </s> which would be an empty
  // turn in a multi-sentence conversation).
  // Also, while we don't expect to see empty 'words' often, it's
  // not disallowed because you might legitimately want the LM
  // to be able to generate the empty sequence in an ASR application.
  void CheckSequence(BaseFloat weight,
                     const std::vector<int32> &words);


  // This function splits a sequence into one or more
  // objects of type SequenceChunk, and appends them to 'chunks_'.j
  void SplitSequenceIntoChunks(BaseFloat weight,
                               const std::vector<int32> &words);

  // for a provided sequence_length > config_.chunk_length,
  // randomly chooses a list of chunk lengths with the following
  // properties:
  //  The chunk lengths sum to 'sequence_length'.
  //  (*chunk_lengths)[0] <= config_.chunk_length
  //  (*chunk_lengths)[i] <= config_.chunk_length - config_.min_split_context
  //    for i > 0
  // All but one of the chunk_lenghhs have the maximum possible
  // value (depending on their position).
  void ChooseChunkLengths(int32 sequence_length,
                          std::vector<int32> *chunk_lengths);

  // Removes, and returns, a randomly chosen SequenceChunk* from 'chunks_'.
  // Transfers ownership to caller.
  SequenceChunk *GetRandomChunk();

  // This stores pending chunks that we have not yet processed
  // into a minibatch.  The pointers are owned here.
  std::vector<SequenceChunk*> chunks_;

  const RnnlmEgsConfig &config_;
  const RnnlmExampleSampler *minibatch_sampler_;
  TaskSequencer<SamplerTask> sampling_sequencer_;
  TableWriter<KaldiObjectHolder<RnnlmExample> > *writer_;
  int32 num_sequences_processed_;
  int32 num_chunks_processed_;
  int32 num_words_processed_;
  int32 num_minibatches_written_;
};


typedef TableWriter<KaldiObjectHolder<RnnlmExample> > RnnlmExampleWriter;

typedef SequentialTableReader<KaldiObjectHolder<RnnlmExample> > SequentialRnnlmExampleReader;


} // namespace rnnlm
} // namespace kaldi

#endif // KALDI_RNNLM_RNNLM_EXAMPLE_H_
