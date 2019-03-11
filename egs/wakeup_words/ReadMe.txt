注意事项：

1. github上传大小及版权限制，训练的语料未给出，仅仅在corpus目录下给出data dev test的格式

2. 字典是自己自定义的字典，语言模型的训练采用字典中的词训练出来的1-gram模型，并且提高了UNK的先验概率

3. 测试效果见视频 ： 模型测试效果视频.mp4

4. 训练生成的模型目录： model

5. 解码参数(仅提供wav解码器的参数)：
./online2-wav-nnet3-latgen-faster --do-endpointing=false --online=false --frame-subsampling-factor=3 --config=./model/nnet3_conf/conf/online.conf --add-pitch=false --max-active=7000 --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 --word-symbol-table=./model/graph/words.txt ./model/nnet3_conf/final.mdl ./model/graph/HCLG.fst 'ark:echo utterance-id1 utterance-id1|' "scp:echo utterance-id1 test.wav|" 'ark:/dev/null'

6. 具体详情参考CSDN: 基于kaldi训练唤醒词模型的一种方法 https://blog.csdn.net/cj1989111/article/details/88017908

7. 嘈杂环境下误唤醒测试
  在嘈杂环境下挂机5天，误唤醒一次，且误唤醒词的置信度为0.617118，详细log参考 test_result/wakeup_test.log
