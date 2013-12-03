#keyword search default
glmFile=conf/glm
duptime=0.5
case_insensitive=false
# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="-oov <unk>"
boost_sil=1.5 #  note from Dan: I expect 1.0 might be better (equivalent to not
              # having the option)... should test.
cer=0

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source train and decode cmds.
