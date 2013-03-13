#keyword search default
glmFile=conf/glm
duptime=0.5
case_insensitive=false
# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="-oov <unk>"

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source train and decode cmds.
