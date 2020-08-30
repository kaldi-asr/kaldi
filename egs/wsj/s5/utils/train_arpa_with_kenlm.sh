# TODO: arg parsing refinement and check
text=$1
symbol_table=$2
arpa=$3

# TODO: KenLM installation check
if [ ! "kenlm is properly installed && can find the training binary: lmplz "] && echo "cannot find training tool *lmplz*, please check your kenlm installation and try again"

# the text should be properly pre-processed(cleand, normalized and possibly word-segmented in some language)

# get rid off irrelavent symbols, the rest of symbols are used as LM training vocabulary. 
cat $symbol_table | grep -v '<eps>' | grep -v '#0' | grep -v '<unk>' | grep -v '<UNK>' | grep -v '<s>' | grep -v '</s>' | awk '{print $1}' > $dir/ngram.vocab

# KenLM training: cat vocab & text together to make sure kenlm has strictly the same vocabulary as kaldi symbol table
cat $dir/ngram.vocab $processed_text | lmplz $kenlm_opts --limit_vocab_file $dir/ngram.vocab > $arpa
