
./steps/online/nnet3/prepare_online_decoding.sh data/lang_chain exp/chain/tdnn_1b_all_sp ./nnet3_conf
copy_dir=../model/
if [ ! -d $copy_dir ];then
	mkdir -p $copy_dir
else
	rm -rf $copy_dir
	echo "dir already exist!"
	mkdir -p $copy_dir
fi

#mv nnet3_conf $copy_dir

cp exp/chain/tdnn_1b_all_sp/graph/ $copy_dir -R
cp conf/mfcc_hires.conf nnet3_conf/conf/

mv nnet3_conf $copy_dir

