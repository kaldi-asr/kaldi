# TODO: arg parsing refinement and check
text=$1
symbol_table=$2
arpa=$3

# TODO: KenLM installation check
if [ ! "kenlm is properly installed && can find the training binary: lmplz "] && echo "cannot find training tool *lmplz*, please check your kenlm installation and try again"

# the text should be properly pre-processed(cleand, normalized and possibly word-segmented in some language)

# get rid off irrelavent symbols, the rest of symbols are used as LM training vocabulary. 
cat $symbol_table | grep -v '<eps>' | grep -v '#0' | grep -v '<unk>' | grep -v '<UNK>' | grep -v '<s>' | grep -v '</s>' | awk '{print $1}' > $dir/ngram.vocab

# KenLM's vocabulary contoll is trick
# the behavior of option --limit_vocab_file is not as SRILM's -limit-vocab.
# --limit_vocab_file actually spcified a vocabulary containing all valid words,
# but a valid word, unseen in training text, won't appear in final arpa.
# that means, kenlm arpa's actual vocab is partly and implicitly defined by the training text
# this will bring inconsistency between kenlm and kaldi system
# so the trick is, exploiting the fact that kenlm will never prune unigram,
# we explicitly append kaldi's vocab to kenlm's training text, and feed kaldi vocab to --limit_vocab_file
# so we will always get an arpa that has strictly the same vocabulary as kaldi,
# and the effect of this trick is just as add-one smoothing, shouldn't have any significant impact in practice.
cat $dir/ngram.vocab $processed_text | lmplz $kenlm_opts --limit_vocab_file $dir/ngram.vocab > $arpa
