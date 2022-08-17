
export PATH=$PWD/fairseq_ltlm/kaldi_utils/:$PATH
export PYTHONPATH=$PWD:$PWD/fairseq_ltlm:$PWD/fairseq_ltlm/recipes:$PYTHONPATH

c=$(which conda || echo '' )
if [ -z $c ] ; then
		current_args=( "$@" )
		set -- ""
		echo "Activating anacona (fairseq_ltlm/anaconda)"
		source fairseq_ltlm/anaconda/bin/activate
		set -- "${current_args[@]}"
fi
export PYTHONUNBUFFERED=1

