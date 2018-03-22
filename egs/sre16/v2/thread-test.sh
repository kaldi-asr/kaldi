# conventional xvector computation with master kaldi

time /home/agorin/src/kaldi/kaldi-master/src/nnet3bin/nnet3-xvector-compute --min-chunk-size=25 --chunk-size=10000 '~ /home/agorin/src/kaldi/kaldi-master/src/nnet3bin/nnet3-copy --nnet-config=0003_sre16_v2_1a/exp/xvector_nnet_1a//extract.config 0003_sre16_v2_1a/exp/xvector_nnet_1a/final.raw - |' 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:dev/split2/1/feats_50.scp ark:- | select-voiced-frames ark:- scp,s,cs:dev/split2/1/vad.scp ark:- |' ark,t:xvec_conv.txt 

# real    1m35.620s
# user    1m35.406s
# sys      0m0.194s

# new single-thread
time /home/agorin/src/kaldi/kaldi-my/src/nnet3bin/nnet3-xvector-compute-parallel --min-chunk-size=25 --chunk-size=10000 'nnet3-copy --nnet-config=0003_sre16_v2_1a/exp/xvector_nnet_1a//extract.config 0003_sre16_v2_1a/exp/xvector_nnet_1a/final.raw - |' 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:dev/split2/1/feats_50.scp ark:- | select-voiced-frames ark:- scp,s,cs:dev/split2/1/vad.scp ark:- |' ark,t:xvec_para_1t.txt 

# real     1m34.160s
# #user    1m33.855s
# sys      0m0.291s

# new 4 threads 
time /home/agorin/src/kaldi/kaldi-my/src/nnet3bin/nnet3-xvector-compute-parallel --num-threads=4 --min-chunk-size=25 --chunk-size=10000 'nnet3-copy --nnet-config=0003_sre16_v2_1a/exp/xvector_nnet_1a//extract.config 0003_sre16_v2_1a/exp/xvector_nnet_1a/final.raw - |' 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:dev/split2/1/feats_50.scp ark:- | select-voiced-frames ark:- scp,s,cs:dev/split2/1/vad.scp ark:- |' ark,t:xvec_para_10t.txt 

# real    0m40.513s
# user    2m13.843s
# sys     0m0.620s


# new 10 threads
time /home/agorin/src/kaldi/kaldi-my/src/nnet3bin/nnet3-xvector-compute-parallel --num-threads=10 --min-chunk-size=25 --chunk-size=10000 'nnet3-copy --nnet-config=0003_sre16_v2_1a/exp/xvector_nnet_1a//extract.config 0003_sre16_v2_1a/exp/xvector_nnet_1a/final.raw - |' 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:dev/split2/1/feats_50.scp ark:- | select-voiced-frames ark:- scp,s,cs:dev/split2/1/vad.scp ark:- |' ark,t:xvec_para_10t.txt 

# real    0m42.263s
# user    2m17.649s
# sys     0m1.136s

# new 10 threads with limited cache capacity 3
time /home/agorin/src/kaldi/kaldi-my/src/nnet3bin/nnet3-xvector-compute-parallel --cache-capacity=3 --num-threads=10 --min-chunk-size=25 --chunk-size=10000 'nnet3-copy --nnet-config=0003_sre16_v2_1a/exp/xvector_nnet_1a/extract.config 0003_sre16_v2_1a/exp/xvector_nnet_1a//final.raw - |' 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:dev/split2/1/feats_50.scp ark:- | select-voiced-frames ark:- scp,s,cs:dev/split2/1/vad.scp ark:- |' ark,t:xvec_para_4t_c3.txt 

# real    0m43.296s
# user    2m16.898s
# sys     0m1.451s

